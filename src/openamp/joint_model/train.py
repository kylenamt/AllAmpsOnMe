"""Joint training: encoder and FiLM generator, end to end, from scratch.

Forked from openamp.emulate.train and keeps its contracts (same
EmulationLoss, Adam + reduce-on-plateau, NaN guard, run-directory layout) so a
joint run's val_esr is directly comparable to the table-conditioned runs.

What's different:

- The embedding comes from the encoder (forward_emb); the generator's own
  nn.Embedding table is frozen and unused.
- Two parameter groups: the encoder gets its own learning rate (joint.enc_lr)
  since LR imbalance is a known failure mode and a dead encoder is invisible
  in ESR alone.
- emulate.embed_noise_frac jitters e on train steps (see `jitter`), so the
  generator has to decode a *neighbourhood* of each embedding rather than 405
  isolated points. Without it nothing pushes the space between amps to mean
  anything, which is exactly what an unseen amp needs.
- joint.enc_frame_window (dac_frames only) draws a random contiguous span of
  frames per step instead of the amp's whole fixed tensor, so one device id
  stops being one deterministic encoder input.

- A collapse tripwire runs every epoch: val_esr_shuffled (embeddings permuted
  within the batch) and emb_spread (how much e varies across amps). With no
  contrastive term, the cheap failure is an encoder emitting a near-constant
  vector while ESR still looks respectable. A good val_esr means nothing
  unless shuffled is much worse and emb_spread is not ~0.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from openamp.data import build_dataset, make_loader, manifest_signature
from openamp.encoders import encoder_class, ref_mode_for
from openamp.evaluation.metrics import pooled_esr_terms
from openamp.models import count_parameters
from openamp.emulate.train import _device_index
from openamp.losses import build_loss, device_loss_weights
from openamp.joint_model.model import build_joint, load_joint
from openamp.spec import RunSpec
from openamp.training import (GRAD_CLIP_NORM, RunDir, amp_setup, build_scheduler,
                              check_resume_compatible, param_group_lrs)

LOG_COLUMNS = ["epoch", "step", "train_loss", "train_esr", "train_stft", "val_esr",
               "val_esr_shuffled", "val_esr_holdout", "emb_spread", "lr_gen", "lr_enc",
               "elapsed_s"]

# Modes the conditioning can be cut in, for the collapse diagnostics (§8).
EVAL_MODES = ("true", "shuffled", "zero")


def _intervene(e: torch.Tensor, mode: str, generator: torch.Generator | None):
    """Apply a conditioning intervention to a batch of embeddings."""
    if mode == "true":
        return e
    if mode == "zero":
        return torch.zeros_like(e)
    if mode == "shuffled":
        if e.shape[0] < 2:
            return e
        perm = torch.randperm(e.shape[0], generator=generator,
                              device=generator.device if generator is not None else e.device)
        return e[perm.to(e.device)]
    raise ValueError(f"mode must be one of {EVAL_MODES}, got {mode!r}")


def jitter(e: torch.Tensor, frac: float, normalize: bool) -> torch.Tensor:
    """Train-time Gaussian jitter on the encoder's embedding (train steps only).

    Why it exists: with no jitter the generator only ever sees the finite set of
    points the encoder emits -- 405 of them under a device-id lookup -- and is
    free to carve 405 isolated wells with nothing decodable between them. An
    unseen amp then lands in dead space no matter how well the encoder places
    it. Jitter makes the neighbourhood of each embedding decode too, which is
    the precondition for the space between amps meaning anything.

    Same knob as :data:`openamp.spec.TrainingSpec.embed_noise_frac`, but
    it cannot share
    :func:`openamp.emulate.train._conditioning`'s scale: that one is a fraction
    of the embedding *table*'s std, and a joint run has no trainable table --
    the conditioning is whatever the encoder emits this step. So the scale comes
    from the embedding's own geometry instead:

    - ``normalize`` (joint.enc_normalize, on in every dac_frames run): ``e`` is a
      unit vector, so a coordinate's natural size is ``1/sqrt(De)`` and ``frac``
      is read against that. The jittered point is re-projected onto the sphere,
      because a plain additive perturbation would mostly move ``|e|`` -- an axis
      the generator never sees vary, so noise along it teaches nothing and
      changes the drive level instead.
    - otherwise: the batch's own std, the closest analogue of the table std.
      Detached, so jitter is never a gradient path back into the encoder.
    """
    if frac <= 0.0:
        return e
    if normalize:
        return F.normalize(e + torch.randn_like(e) * (frac / math.sqrt(e.shape[-1])),
                           dim=-1)
    return e + torch.randn_like(e) * (frac * e.detach().std())


@torch.no_grad()
def evaluate_esr(model, loader, device, coeff: float, *, n_batches: int | None = None,
                 mode: str = "true", seed: int = 0,
                 weights: torch.Tensor | None = None) -> dict:
    """Energy-pooled pre-emphasized ESR over the loader, plus embedding spread.

    Pooled as one ratio of sums exactly like
    :func:`openamp.emulate.train.evaluate_esr`, so the numbers are comparable
    across the two trainers.

    ``emb_spread`` is the mean over dimensions of the across-batch standard
    deviation of ``e`` — the direct readout of whether the encoder distinguishes
    amps at all. It collapses to ~0 long before ESR shows anything is wrong.

    The shuffle is drawn from a seeded generator so the number is comparable from
    epoch to epoch rather than re-rolled each time.

    ``weights`` [n_devices] applies the same per-device ESR weighting the train
    loss uses (:func:`openamp.emulate.train.device_loss_weights`), indexed by
    each item's true device (``batch["device_idx"]``) -- so checkpoint
    selection, ReduceLROnPlateau and early stopping score the objective the run
    actually optimizes. Only meaningful when ``loader``'s ``device_idx`` shares
    the training run's id_to_idx; a holdout loader uses its own, unrelated
    indexing and must not be passed weights fit on the training device set.
    """
    was_training = model.training
    model.eval()
    R = model.receptive_field
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    num = den = 0.0
    spreads = []
    for k, batch in enumerate(loader):
        if n_batches is not None and k >= n_batches:
            break
        inp = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        ref = batch["ref"].to(device, non_blocking=True)
        e = model.embed(ref)
        if e.shape[0] > 1:
            spreads.append(float(e.std(dim=0).mean()))
        out = model.render(inp, _intervene(e, mode, gen))[..., R:].squeeze(1)
        w = (None if weights is None
             else weights[batch["device_idx"].to(device, non_blocking=True)])
        n_i, d_i = pooled_esr_terms(out, target, coeff=coeff, weights=w)
        num += float(n_i.sum().double())
        den += float(d_i.sum().double())
    if was_training:
        model.train()
    return {"esr": num / (den + 1e-8) if den > 0.0 else float("nan"),
            "emb_spread": float(np.mean(spreads)) if spreads else float("nan")}


def _dataset_kwargs(spec: RunSpec, id_to_idx, R: int) -> dict:
    """The dataset arguments every split of a joint run shares."""
    j = spec.joint
    return dict(paths=spec.paths, receptive_field=R, id_to_idx=id_to_idx,
                clip_samples=spec.clip_samples, ref_samples=spec.ref_samples,
                ref_different_file=j.ref.different_file,
                ref_min_gap=int(round(j.ref.min_gap_seconds * spec.sample_rate)),
                sources=spec.data.sources, ref_mode=ref_mode_for(j.encoder.kind),
                **spec.data.kwargs)


def _build_datasets(spec: RunSpec, id_to_idx, R: int):
    """Train/val dataset pair for a run."""
    common = _dataset_kwargs(spec, id_to_idx, R)
    train_ds = build_dataset(spec.data.dataset, split="train",
                             pairs_per_epoch=spec.data.pairs_per_epoch,
                             seed=spec.seed, **common)
    val_ds = build_dataset(spec.data.dataset, split="val",
                           pairs_per_epoch=spec.training.val_pairs,
                           seed=spec.seed + 999, **common)
    return train_ds, val_ds, spec.clip_samples, spec.ref_samples


def _optimizer(model, spec: RunSpec):
    """Two groups: generator (minus its frozen table) and encoder.

    Under ``encoder.kind: fingerprint`` the encoder's fingerprint matrix is a
    *buffer*, so ``model.encoder.parameters()`` is exactly the adapter and the codec
    cannot be trained even by accident. Under ``dac_frames`` the frames themselves
    aren't even a buffer -- ``FrameStore`` is a plain attribute, not registered --
    so ``model.encoder.parameters()`` is exactly the attention pool plus the adapter.
    """
    tcfg = spec.training
    gen_params = [p for n, p in model.generator.named_parameters()
                  if p.requires_grad and not n.startswith("embedding.")]
    enc_wd = float(spec.joint.enc_weight_decay)
    enc_wd = tcfg.weight_decay if enc_wd < 0 else enc_wd
    return torch.optim.Adam(
        [{"params": gen_params, "lr": tcfg.lr, "name": "generator",
          "weight_decay": tcfg.weight_decay},
         {"params": list(model.encoder.parameters()), "lr": spec.joint.enc_lr,
          "name": "encoder", "weight_decay": enc_wd}],
        weight_decay=tcfg.weight_decay)


# --- Sanity ladder -------------------------------------------------------------
def overfit_one_batch(spec: RunSpec, *, device: str = "cuda", steps: int = 400,
                      limit_devices: int = 8,
                      batch_size: int | None = None) -> float:
    """Sanity #1: one fixed batch should drive ESR to ~0 through the encoder.

    Proves the gradient actually reaches the encoder. A *collapse* here is not
    conclusive — one fixed batch makes collapse an absorbing state — but a
    failure to fit is.
    """
    tcfg = spec.training
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(spec.seed)
    batch = int(batch_size or tcfg.batch_size)

    ids, id_to_idx, _ = _device_index(spec, limit_devices)
    model = build_joint(spec, len(ids)).to(dev)
    R = model.receptive_field
    train_ds, _, clip, ref_n = _build_datasets(spec, id_to_idx, R)
    train_ds.pairs_per_epoch = batch
    loader = make_loader(train_ds, batch, 0, shuffle=False, seed=spec.seed,
                         drop_last=False)
    b = next(iter(loader))
    inp = b["input"].to(dev)
    target = b["target"].to(dev).unsqueeze(1)
    ref = b["ref"].to(dev)

    lossfn = build_loss(spec.loss).to(dev)
    opt = _optimizer(model, spec)
    print(f"[overfit] devices={len(ids)} batch={batch} "
          f"encoder={count_parameters(model.encoder):,} "
          f"generator={count_parameters(model.generator):,} params "
          f"receptive_field={R} clip={clip} ref={ref_n}")
    final = float("nan")
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(inp, ref)[..., R:]
        loss, parts = lossfn(out.float(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        opt.step()
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            final = parts["esr"]
            print(f"  step {step:4d}  loss {loss.item():.5f}  esr {parts['esr']:.5f}  "
                  f"stft {parts['stft']:.4f}")
    flag = "OK" if final < 1e-2 else "HIGH — check model/data wiring"
    print(f"[overfit] final esr={final:.5f} -> {flag}")
    return final


# --- Main training loop --------------------------------------------------------
def train(spec: RunSpec, *, device: str = "cuda", epochs: int | None = None,
          resume: bool = False, limit_devices: int | None = None) -> dict:
    """Train encoder + generator jointly to a val-ESR plateau; returns a summary."""
    tcfg, jcfg, name = spec.training, spec.joint, spec.name
    max_epochs = int(epochs if epochs is not None else tcfg.epochs)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    torch.backends.cudnn.benchmark = True

    ids, id_to_idx, holdout_ids = _device_index(spec, limit_devices)
    if not ids:
        raise RuntimeError("No render-ok devices. Run `openamp render` first.")
    # Per-device ESR weighting, applied to the train loss and to the seen-device
    # val ESR alike -- exactly openamp.emulate.train's device_loss_weighting,
    # forked here because the two trainers' loss/eval plumbing differs. Not
    # applied to the holdout probe: holdout_loader indexes device_idx against
    # its own {holdout id -> row} map, unrelated to `ids`, so weights fit here
    # would land on the wrong devices there.
    strength = float(spec.loss.device_loss_weighting)
    if strength < 0.0:
        raise ValueError(f"loss.device_loss_weighting must be >= 0, got {strength}")
    dev_w = None
    if strength > 0.0:
        _w = device_loss_weights(spec.paths.renders_manifest, ids, strength)
        dev_w = torch.as_tensor(_w, dtype=torch.float32, device=dev)
        print(f"[loss]  device_loss_weighting={strength:g} "
              f"weights min={_w.min():.4g} p50={np.median(_w):.4g} "
              f"max={_w.max():.4g} (mean 1.0, ESR term only; seen-device val ESR "
              f"weighted too -- not comparable with a strength-0 run; holdout "
              f"stays unweighted, a different device set)")
    if tcfg.embed_noise_frac < 0.0:
        raise ValueError("training.embed_noise_frac must be >= 0 (a fraction of the "
                         f"embedding's own scale), got {tcfg.embed_noise_frac}")
    noise_frac = float(tcfg.embed_noise_frac)
    if noise_frac > 0.0:
        scale = ("1/sqrt(De) on the unit sphere, re-projected after"
                 if jcfg.encoder.kwargs.get("normalize") else "the batch's own std")
        print(f"[loss]  embed_noise_frac={noise_frac:g} against {scale} "
              f"(train steps only; val, shuffled and holdout all see clean e)")
    manifest_sig = manifest_signature(spec.paths)
    # Fingerprint provenance: the underlying data can be regenerated out from under
    # a finished run, and nothing about the checkpoint's shape would notice.
    fp_sha, fp_meta = encoder_class(jcfg.encoder.kind).provenance(jcfg.encoder)
    model = build_joint(spec, len(ids)).to(dev)
    R = model.receptive_field
    n_enc, n_gen = count_parameters(model.encoder), count_parameters(model.generator)
    rf_ms = 1000.0 * R / spec.sample_rate
    print(f"[model] name={name} arch={spec.model.arch} embed={model.embedding_dim} "
          f"encoder={n_enc:,} generator={n_gen:,} params "
          f"devices={len(ids)} holdout={len(holdout_ids)}")
    cap = model.encoder.capacity_report()
    if cap["head"] == "fingerprint":
        src = (f"fp={jcfg.encoder.kwargs.get('path')} preprocess={cap.get('preprocess')} "
              f"n_fingerprints={cap.get('n_fingerprints')}")
    elif cap["head"] == "dac_frames":
        win = cap.get("frame_window", 0)
        src = (f"fp_frames={jcfg.encoder.kwargs.get('frames_path')} "
              f"n_devices={cap.get('n_devices')} "
              f"pool={cap.get('n_queries')}x{cap.get('d_model')} "
              + (f"window={win}/{cap.get('n_frames')} frames (random in train, "
                 f"centre in eval)" if win else
                 f"window=all {cap.get('n_frames')} frames"))
    else:
        src = f"enc_channels={model.encoder.channels}"
    print(f"[model] head={cap['head']} {src} "
          f"pooled={cap['pooled_dim']}-d -> De={cap['embedding_dim']} "
          f"(ceiling {cap['ceiling']}) enc_rf={model.encoder.receptive_field} "
          f"normalize={model.encoder.normalize}")
    if cap["pooled_dim"] < cap["embedding_dim"]:
        print(f"[warn]  pooled width {cap['pooled_dim']} is below embedding_dim "
              f"{cap['embedding_dim']} — pooling, not the projection, is the bottleneck")
    print(f"[model] generator receptive_field={R} samples ({rf_ms:.1f} ms)")

    # --- Data -------------------------------------------------------------------
    train_ds, val_ds, clip, ref_n = _build_datasets(spec, id_to_idx, R)
    val_loader = make_loader(val_ds, tcfg.batch_size, tcfg.num_workers,
                              shuffle=False, seed=spec.seed + 999, drop_last=False)
    if train_ds.ref_mode == "device":
        ref_desc = "device-id lookup of a precomputed fingerprint"
    elif train_ds.same_file_mode:
        ref_desc = "same-file (gap %d)" % train_ds.ref_min_gap
    else:
        ref_desc = f"different-file (of {len(train_ds.ref_files)})"
    print(f"[data]  train_devices={len(train_ds.device_ids)} "
          f"val_devices={len(val_ds.device_ids)} "
          f"pairs/epoch={spec.data.pairs_per_epoch}")
    print(f"[data]  clip={clip} ref={ref_n} reference={ref_desc}")
    if train_ds.ref_mode == "audio" and train_ds.same_file_mode:
        print("[warn]  same-file references: the A/B barrier is a time gap, not a "
              "different recording — expect weaker protection against content leakage")

    # A held-out amp must never reach a training batch. Under fingerprint
    # conditioning its vector is present in the encoder's buffer (that is what makes
    # zero-shot evaluation possible at all), so assert the data side explicitly
    # rather than trusting that id_to_idx excluded it.
    leaked = sorted(set(holdout_ids) & set(int(d) for d in train_ds.device_ids))
    if leaked:
        raise RuntimeError(f"holdout devices {leaked} are in the training set")

    # Zero-shot probe: the number this experiment exists to produce, measured
    # every epoch instead of only at the end (a seen-vs-holdout gap can stay
    # invisible until the run finishes otherwise).
    holdout_loader = None
    if holdout_ids:
        h_idx = {int(d): i for i, d in enumerate(sorted(holdout_ids))}
        holdout_ds = build_dataset(
            spec.data.dataset, split="val", pairs_per_epoch=tcfg.val_pairs,
            seed=spec.seed + 4242, **_dataset_kwargs(spec, h_idx, R))
        holdout_loader = make_loader(holdout_ds, tcfg.batch_size, tcfg.num_workers,
                                      shuffle=False, seed=spec.seed + 4242,
                                      drop_last=False)
        print(f"[data]  holdout probe: {len(holdout_ds.device_ids)} unseen devices")

    # --- Optim / loss / bookkeeping --------------------------------------------
    lossfn = build_loss(spec.loss).to(dev)
    opt = _optimizer(model, spec)
    sched = build_scheduler(opt, tcfg)
    use_amp, amp_dtype, scaler = amp_setup(tcfg, dev)

    run = RunDir(spec.run_dir)
    ckpt_path, last_path = run.checkpoint, run.last

    start_epoch, best_val, since_improved = 0, float("inf"), 0
    resumed_from = None
    if resume and last_path.is_file():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        _check_resume_compatible(ck, spec, R, len(ids))
        model.encoder.load_state_dict(ck["encoder"])
        model.generator.load_state_dict(ck["generator"])
        opt.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        resumed_from = int(ck["epoch"])
        start_epoch = resumed_from + 1
        best_val = float(ck.get("best_val_esr", float("inf")))
        # Config is authoritative on resume (same rule as emulate/train.py): the
        # Adam moment buffers are what resuming preserves, not the lr value.
        enc_wd = float(jcfg.enc_weight_decay)
        enc_wd = tcfg.weight_decay if enc_wd < 0 else enc_wd
        for g in opt.param_groups:
            is_enc = g.get("name") == "encoder"
            g["lr"] = jcfg.enc_lr if is_enc else tcfg.lr
            # Per-group, not one value for both: joint.enc_weight_decay would
            # otherwise be silently reset to the generator's on every resume.
            g["weight_decay"] = enc_wd if is_enc else tcfg.weight_decay
        print(f"[resume] from epoch {start_epoch}, best_val_esr={best_val:.5f}; "
              f"lr_gen={tcfg.lr:.2e} lr_enc={jcfg.enc_lr:.2e} (from config)")
    else:
        run.start_log(LOG_COLUMNS)
    run.append_config_history(mode="resume" if resumed_from is not None else "start",
                              start_epoch=start_epoch, resumed_from=resumed_from,
                              best_val=best_val, max_epochs=max_epochs,
                              spec=spec.to_dict())

    def save_ckpt(path: Path, epoch: int, val_esr: float) -> None:
        run.save_checkpoint(path, {
            "encoder": model.encoder.state_dict(),
            "generator": model.generator.state_dict(),
            "optim": opt.state_dict(), "scaler": scaler.state_dict(),
            "spec": spec.to_dict(),
            "device_ids": ids, "id_to_idx": id_to_idx, "holdout_ids": holdout_ids,
            "manifest_sha256": manifest_sig, "receptive_field": R,
            "fingerprint_sha256": fp_sha, "fingerprint_meta": fp_meta,
            "ref_samples": ref_n, "clip_samples": clip,
            "encoder_params": n_enc, "generator_params": n_gen,
            "sample_rate": spec.sample_rate, "seed": spec.seed, "name": name,
            "epoch": epoch, "val_esr": val_esr, "best_val_esr": best_val,
        })

    # --- Loop -------------------------------------------------------------------
    t0 = time.time()
    step = start_epoch * (spec.data.pairs_per_epoch // tcfg.batch_size)
    epoch = start_epoch - 1
    # The per-epoch tripwire runs on a slice of val, not all of it: it is watched
    # for its trend, and the end-of-run number is measured over the whole set.
    probe_batches = max(1, min(25, tcfg.val_pairs // max(tcfg.batch_size, 1)))
    for epoch in range(start_epoch, max_epochs):
        train_ds.seed = spec.seed + epoch       # fresh window positions each epoch
        loader = make_loader(train_ds, tcfg.batch_size, tcfg.num_workers,
                              shuffle=True, seed=spec.seed + epoch, drop_last=True)
        model.train()
        ep_loss = ep_esr = ep_stft = 0.0
        n_steps = 0
        ep_t0 = time.time()
        for batch in loader:
            inp = batch["input"].to(dev, non_blocking=True)
            target = batch["target"].to(dev, non_blocking=True).unsqueeze(1)
            ref = batch["ref"].to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                e = jitter(model.embed(ref), noise_frac, model.encoder.normalize)
                out = model.render(inp, e)[..., R:]
            w = (None if dev_w is None else
                dev_w[batch["device_idx"].to(dev, non_blocking=True)])
            loss, parts = lossfn(out.float(), target, weights=w)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()

            ep_loss += loss.item(); ep_esr += parts["esr"]; ep_stft += parts["stft"]
            n_steps += 1
            step += 1
            if step % tcfg.log_every == 0:
                sps = n_steps * tcfg.batch_size / (time.time() - ep_t0 + 1e-9)
                print(f"  e{epoch:02d} step {step:>7d} loss {loss.item():.5f} "
                      f"esr {parts['esr']:.5f} stft {parts['stft']:.4f} {sps:5.1f} smp/s",
                      flush=True)      # stdout is block-buffered under nohup/redirects

        tr_loss = ep_loss / max(n_steps, 1)
        tr_esr = ep_esr / max(n_steps, 1)
        tr_stft = ep_stft / max(n_steps, 1)
        val = evaluate_esr(model, val_loader, dev, lossfn.coeff, seed=spec.seed,
                          weights=dev_w)
        val_esr, spread = val["esr"], val["emb_spread"]
        shuf = evaluate_esr(model, val_loader, dev, lossfn.coeff, mode="shuffled",
                            n_batches=probe_batches, seed=spec.seed,
                            weights=dev_w)["esr"]
        # holdout stays unweighted -- see the device_loss_weighting note above.
        hold = (evaluate_esr(model, holdout_loader, dev, lossfn.coeff,
                             n_batches=probe_batches, seed=spec.seed)["esr"]
                if holdout_loader is not None else float("nan"))

        # NaN guard, identical policy to emulate/train.py.
        if not np.isfinite(val_esr):
            if not ckpt_path.is_file():
                raise RuntimeError(
                    f"[nan-guard] epoch {epoch}: non-finite val ESR before any best "
                    "checkpoint exists — check model/data wiring, not precision")
            ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
            model.encoder.load_state_dict(ck["encoder"])
            model.generator.load_state_dict(ck["generator"])
            opt.load_state_dict(ck["optim"])
            if use_amp:
                use_amp, scaler = False, torch.amp.GradScaler("cuda", enabled=False)
                note = "AMP off, fp32 from here"
            else:
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                note = f"lr halved to {opt.param_groups[0]['lr']:.1e}"
            print(f"[nan-guard] epoch {epoch:02d}: non-finite val ESR — reloaded best "
                  f"(epoch {ck['epoch']}, val_esr={float(ck['val_esr']):.5f}); {note}")
            _log_row(run, epoch, step, tr_loss, tr_esr, tr_stft, val_esr, shuf,
                     hold, spread, opt, t0)
            continue
        if use_amp and not np.isfinite(tr_loss):
            use_amp, scaler = False, torch.amp.GradScaler("cuda", enabled=False)
            print(f"[nan-guard] epoch {epoch:02d}: non-finite train loss but finite val "
                  f"({val_esr:.5f}) — overflow absorbed; AMP off, fp32 from here")

        sched.step(val_esr)
        improved = val_esr < best_val
        if improved:
            best_val, since_improved = val_esr, 0
            save_ckpt(ckpt_path, epoch, val_esr)
        else:
            since_improved += 1
        save_ckpt(last_path, epoch, val_esr)
        _log_row(run, epoch, step, tr_loss, tr_esr, tr_stft, val_esr, shuf,
                 hold, spread, opt, t0)
        lrs = param_group_lrs(opt)
        # ratio > 1 means a wrong reference hurts, i.e. the embedding is being used
        ratio = shuf / val_esr if val_esr > 0 else float("nan")
        print(f"[epoch {epoch:02d}] train_esr={tr_esr:.5f} val_esr={val_esr:.5f} "
              f"best={best_val:.5f} shuffled={shuf:.5f} (x{ratio:.1f}) "
              f"holdout={hold:.5f} "
              f"emb_spread={spread:.4f} lr={lrs['generator']:.1e}/{lrs['encoder']:.1e}"
              f"{'  *saved*' if improved else ''}", flush=True)
        if since_improved >= tcfg.early_stop_patience:
            print(f"[stop] no val improvement for {since_improved} epochs")
            break

    # --- Reload best; full diagnostics; final artifacts -------------------------
    if ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.encoder.load_state_dict(ck["encoder"])
        model.generator.load_state_dict(ck["generator"])
    final = {m: evaluate_esr(model, val_loader, dev, lossfn.coeff, mode=m,
                             seed=spec.seed, weights=dev_w)
             for m in EVAL_MODES}
    val_true, val_shuf = final["true"]["esr"], final["shuffled"]["esr"]
    val_zero, spread = final["zero"]["esr"], final["true"]["emb_spread"]
    train_hours = (time.time() - t0) / 3600.0
    verdict = ("OK" if val_shuf > 2 * val_true and spread > 1e-3 else
               "WEAK — the encoder may have collapsed; see joint-eval")
    print(f"[cond]  val_esr={val_true:.5f} shuffled={val_shuf:.5f} zero={val_zero:.5f} "
          f"emb_spread={spread:.4f} -> {verdict}")

    run.save_spec(spec)
    metrics = {
        "name": name, "arch": spec.model.arch, "encoder_params": n_enc,
        "generator_params": n_gen, "params": n_enc + n_gen,
        "receptive_field_samples": R, "receptive_field_ms": round(rf_ms, 3),
        "embedding_dim": model.embedding_dim,
        "enc_channels": getattr(model.encoder, "channels", 0),
        "enc_head": cap["head"], "pooled_dim": cap["pooled_dim"],
        "channels": model.generator.channels,
        "n_devices": len(ids), "n_holdout": len(holdout_ids),
        "ref_samples": ref_n, "clip_samples": clip,
        "ref_different_file": not train_ds.same_file_mode,
        "best_val_esr": best_val, "val_esr": val_true,
        "val_esr_shuffled": val_shuf, "val_esr_zero": val_zero,
        "device_loss_weighting": strength,
        "embed_noise_frac": noise_frac,
        "enc_frame_window": cap.get("frame_window", 0),
        "emb_spread": spread, "epochs_run": epoch + 1,
        "train_hours": round(train_hours, 4), "manifest_sha256": manifest_sig,
    }
    run.save_metrics(metrics)
    print(f"[done]  best_val_esr={best_val:.5f} train_hours={train_hours:.2f} -> {run.path}")
    return metrics


# --- Bookkeeping ---------------------------------------------------------------
def _log_row(run, epoch, step, tr_loss, tr_esr, tr_stft, val_esr, shuf,
             hold, spread, opt, t0) -> None:
    lrs = param_group_lrs(opt)
    run.append_log([
        epoch, step, f"{tr_loss:.6f}", f"{tr_esr:.6f}", f"{tr_stft:.6f}",
        f"{val_esr:.6f}", f"{shuf:.6f}", f"{hold:.6f}", f"{spread:.6f}",
        f"{lrs['generator']:.2e}", f"{lrs['encoder']:.2e}", f"{time.time() - t0:.1f}"])


def _check_resume_compatible(ck: dict, spec: RunSpec, receptive_field: int,
                             n_devices: int) -> None:
    """Fail fast if a config edit made the checkpoint's weights incompatible.

    Two structural sections here rather than emulate's one: ``model:`` for the
    generator and ``joint.encoder:`` for the encoder. Both are compared whole --
    every key in either changes the network, or (``normalize``, the fingerprint
    paths) quietly feeds the generator a different vector, which is worse.
    """
    saved = ck.get("spec", {})
    check_resume_compatible(
        ck, [(saved.get("model", {}), spec.model.to_dict(), "model."),
             (saved.get("joint", {}).get("encoder", {}),
              spec.joint.encoder.to_dict(), "joint.encoder.")],
        receptive_field=receptive_field, n_devices=n_devices,
        dynamics="lr, enc_lr, plateau_*, epochs, loss, batch_size")


__all__ = ["jitter", "train", "overfit_one_batch", "evaluate_esr", "load_joint", "LOG_COLUMNS",
           "EVAL_MODES"]
