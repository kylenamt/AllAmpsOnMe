#!/usr/bin/env python3
"""Train an AttentionEncoder against a *frozen*, already-trained emulate decoder.

    python scripts/frozen_decoder_joint.py --emulate-run results/emulate/delta_a2_tanh_r32_512_v2

The question this asks: the joint runs so far trained encoder and generator
together from scratch, and the generator's ability to render an arbitrary point
was always a suspect. Here the generator is a finished artifact -- 150 epochs,
val ESR 0.0509 -- and the *only* thing that trains is the encoder. So a bad
number can only mean the encoder put the amp in the wrong place, and there is a
measured ceiling to compare it against: that run's own enrollment fitted a
vector directly per held-out amp and reached test ESR mean 0.0231 / median
0.0089. That is what one encoder forward pass has to match.

**A script, not a config-driven stage.** The library's joint trainer builds its
generator from scratch and has no freeze knob; adding both to ``JointSpec``,
``build_joint`` and the resume checks is real surface area for an experiment
that may not pan out. This owns its own loop instead and imports everything
else, so the dataset wiring, loss, ESR definition, jitter and collapse
diagnostics are *the same objects* the trainer uses -- the comparison against
``results/joint/*`` is not a re-implementation of them.

What it writes is a normal joint run directory, checkpoint schema included, so
``openamp joint-eval results/joint/<name>`` works on the result unchanged.

One thing it deliberately does not do: no generator fine-tuning. "Frozen" is
the experiment; a run that lets the decoder move is a different question and
would need its own control.

``--resume`` picks the encoder up from ``last.pt`` and continues into the same
run directory, appending to ``train_log.csv`` rather than truncating it. What it
carries over is the encoder's weights and AdamW's moment buffers; the *config*
is authoritative for everything else, so a resume is the seam where lr, weight
decay, the schedule and the epoch budget are meant to change. Structural knobs
are not: :func:`_check_resume_compatible` compares ``model:`` and
``joint.encoder:`` whole and refuses a resume that would reshape the network.

Notes on the two halves lining up, checked rather than assumed:

- The holdout is corpus-level (``data/manifests/emulate_holdout.txt``), not
  per-run, so the 84 amps the emulate run never saw are exactly the ones this
  run holds out. The zero-shot probe is therefore unseen by the *decoder* too,
  which is the only version of the claim worth making.
- ``embed_normalize`` generators read their table as ``embed_scale *
  normalize(row)``, and ``forward_emb`` does *not* apply ``embed_scale`` -- it
  takes an already-effective embedding. So the caller owns the radius, and
  :class:`ScaledJointModel` below is what sets it. This matters more here than
  in the library trainer: there the generator learns whatever radius the encoder
  emits, while a frozen decoder has only ever seen vectors of one length.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from openamp.data import (build_dataset, make_loader, manifest_signature,
                          training_device_index)
from openamp.encoders import AttentionEncoder, FrameStore
from openamp.losses import build_loss, device_loss_weights
from openamp.models import build_model, count_parameters
from openamp.models.joint import JointModel
from openamp.spec import EncoderSpec, JointSpec, RefSpec, RunSpec, spec_from_checkpoint
from openamp.training import (GRAD_CLIP_NORM, RunDir, amp_setup, build_scheduler,
                              param_group_lrs, step_scheduler)
# The trainer's own helpers, imported rather than re-implemented: a private name
# here is the price of the numbers being comparable by construction.
from openamp.training.joint import (EVAL_MODES, _build_datasets, _check_resume_compatible,
                                    _dataset_kwargs, evaluate_esr, jitter)

LOG_COLUMNS = ["epoch", "step", "train_loss", "train_esr", "train_stft", "val_esr",
               "val_esr_shuffled", "val_esr_holdout", "val_esr_holdout_w",
               "emb_spread", "lr_enc", "elapsed_s"]


class ScaledJointModel(JointModel):
    """JointModel that puts ``e`` on the sphere the frozen decoder was trained on.

    ``AttentionEncoder(normalize=True)`` ends in ``F.normalize``, so it emits
    radius exactly 1 and the adapter *cannot* change that -- the normalize is
    the last op. A generator built with ``embed_normalize`` projects every table
    row to ``embed_scale``, so radius 1 is a sphere it never saw: on the v2 run
    every row it trained against had length 1.0504, and a unit vector under-drives
    the conditioning by 4.8% with no way for training to correct it.

    ``project_embedding`` is the generator's own function and is idempotent on a
    unit vector (``embed_scale * normalize(u) == embed_scale * u``), and a plain
    identity when the generator has no ``embed_normalize``. Overriding ``embed``
    rather than scaling in the loop is what makes ``evaluate_esr`` -- which calls
    ``model.embed`` itself -- score the same vectors training used.
    """

    def embed(self, ref: torch.Tensor, mask: torch.Tensor | None = None):
        return self.generator.project_embedding(super().embed(ref, mask))


# --- The frozen half -------------------------------------------------------------
def load_frozen_generator(run_dir: Path, device) -> tuple[torch.nn.Module, RunSpec, dict]:
    """``(generator, the run's spec, its checkpoint)`` with every parameter frozen.

    Sized from the checkpoint's own embedding table rather than from this run's
    device count, so the ``load_state_dict`` is exact. The table is dead weight
    under encoder conditioning -- ``forward_emb`` never reads it -- but dropping
    it would make the load partial, and a partial load is how a silently
    half-initialized generator happens.
    """
    ck_path = Path(run_dir) / "checkpoint.pt"
    if not ck_path.is_file():
        raise SystemExit(f"no checkpoint.pt under {run_dir}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    spec = spec_from_checkpoint(ck)
    n_rows = int(ck["model"]["embedding.weight"].shape[0])

    gen = build_model(spec.model, n_rows)
    gen.load_state_dict(ck["model"])
    for p in gen.parameters():
        p.requires_grad_(False)
    return gen.to(device), spec, ck


def build_spec(args, emulate_spec: RunSpec) -> RunSpec:
    """The joint RunSpec for this experiment, forked from the decoder's own.

    ``model``, ``loss`` and the corpus come from the emulate run verbatim: the
    generator being loaded *is* that model, and reusing its loss is what makes
    this run's val ESR comparable to the 0.0509 it reached with a fitted table.
    """
    # dataclasses.replace, not a .replace method: only RunSpec has one.
    data = replace(emulate_spec.data, dataset="joint")
    overrides = {"scheduler": "cosine", "embed_noise_frac": args.embed_noise,
                 "epochs": args.epochs, "batch_size": args.batch_size,
                 "num_workers": args.num_workers,
                 # The generator group does not exist, so training.lr is unused;
                 # set it to the encoder's so nothing reads a misleading number.
                 "lr": args.enc_lr}
    # The cosine floor is a *fraction* of each group's own base rate, so an
    # absolute floor has to be converted against the lr it anneals from --
    # stating the endpoint is how anyone actually thinks about the schedule.
    if args.enc_lr_min is not None:
        if not 0.0 < args.enc_lr_min <= args.enc_lr:
            raise SystemExit(f"--enc-lr-min {args.enc_lr_min:g} must be >0 and no "
                             f"greater than --enc-lr {args.enc_lr:g}")
        overrides["cosine_min_lr_frac"] = args.enc_lr_min / args.enc_lr
    tcfg = replace(emulate_spec.training,
                   **{k: v for k, v in overrides.items() if v is not None})

    encoder = EncoderSpec("dac_frames_refine", {
        "frames_path": str(args.frames), "n_queries": args.n_queries,
        "d_model": args.d_model, "n_heads": args.n_heads,
        "mlp_ratio": args.mlp_ratio, "positional": not args.no_positional,
        "normalize": True})
    return emulate_spec.replace(
        name=args.name, data=data, training=tcfg,
        joint=JointSpec(encoder=encoder, ref=RefSpec(seconds=args.ref_seconds),
                        enc_lr=args.enc_lr, enc_weight_decay=args.enc_weight_decay))


def check_frames_cover(store: FrameStore, ids, holdout_ids) -> None:
    """Every amp this run touches must have frames, trained on or held out.

    A missing training id would raise inside ``forward``; a missing *holdout* id
    would only surface when the zero-shot probe first runs, which is the number
    the whole experiment exists to produce.
    """
    missing = sorted({int(d) for d in list(ids) + list(holdout_ids)} - set(store.device_ids))
    if missing:
        raise SystemExit(
            f"{len(missing)} device(s) have no frames in {store.root}: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}. The corpus and the frame store "
            f"disagree — re-run openamp.probes.extract for this corpus.")


def enroll_ceiling(run_dir: Path) -> dict:
    """The decoder run's own enrollment numbers -- the target this run aims at.

    Two of them, and they are **different statistics** (docs/decisions.md S6):

    - ``best_val_esr_pooled`` is pooled ratio-of-sums, pre-emphasized, device
      weighted -- the same thing ``evaluate_esr`` computes, so it is what
      ``val_esr_holdout_w`` below is comparable to.
    - ``test_esr_mean`` / ``test_esr_median`` are the mean over devices of
      per-window *raw* ratios with a silence gate. Nothing this script logs is
      comparable to those; reaching them means running the test-semantics
      evaluation on the finished encoder.
    """
    path = Path(run_dir) / "enroll" / "metrics.json"
    if not path.is_file():
        return {}
    m = json.loads(path.read_text())
    return {k: m[k] for k in ("best_val_esr_pooled", "test_esr_mean",
                              "test_esr_median", "n_enrolled") if k in m}


# --- Training --------------------------------------------------------------------
def train(args) -> dict:
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator, emulate_spec, gen_ck = load_frozen_generator(args.emulate_run, dev)
    spec = build_spec(args, emulate_spec)
    tcfg = spec.training
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    torch.backends.cudnn.benchmark = True

    ids, id_to_idx, holdout_ids = training_device_index(spec)
    if not ids:
        raise SystemExit("No render-ok devices. Run `openamp render` first.")
    gen_ids = [int(d) for d in gen_ck["device_ids"]]
    if sorted(gen_ids) != sorted(int(d) for d in ids):
        raise SystemExit(
            f"the frozen decoder was trained on {len(gen_ids)} devices and this "
            f"corpus resolves {len(ids)} — they must be the same set, or the "
            f"generator has seen amps this run calls held out. "
            f"decoder-only: {sorted(set(gen_ids) - set(map(int, ids)))[:8]}")

    store = FrameStore(args.frames)
    check_frames_cover(store, ids, holdout_ids)
    encoder = AttentionEncoder(
        embedding_dim=generator.embedding_dim, frames=store,
        n_queries=args.n_queries, d_model=args.d_model, n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio, positional=not args.no_positional,
        normalize=True).to(dev)
    if args.init_encoder is not None:
        ckd = torch.load(args.init_encoder, map_location="cpu", weights_only=False)
        ra = ckd.get("args", {})
        mismatches = [
            f"{k}: checkpoint={ra.get(k)} this run={v}"
            for k, v in (("n_queries", args.n_queries), ("d_model", args.d_model),
                        ("n_heads", args.n_heads), ("mlp_ratio", args.mlp_ratio))
            if int(ra.get(k, v)) != int(v)]
        ckpt_positional = not bool(ra.get("no_positional", False))
        if ckpt_positional != (not args.no_positional):
            mismatches.append(f"positional: checkpoint={ckpt_positional} "
                              f"this run={not args.no_positional}")
        if mismatches:
            raise SystemExit("--init-encoder: structural mismatch, cannot "
                             "warm-start:\n  " + "\n  ".join(mismatches))
        encoder.load_state_dict(ckd["state_dict"])
        print(f"[model] encoder warm-started from {args.init_encoder} "
              f"(fresh optimizer; this run's own lr/wd/epochs/schedule apply)")
    model = ScaledJointModel(encoder, generator)
    R = model.receptive_field
    emb_scale = float(getattr(generator, "embed_scale", torch.tensor(1.0)))

    n_enc, n_gen = count_parameters(encoder), count_parameters(generator)
    print(f"[model] name={spec.name} decoder={args.emulate_run} "
          f"arch={spec.model.arch} embed={model.embedding_dim}")
    print(f"[model] encoder={n_enc:,} trainable | generator={n_gen:,} FROZEN "
          f"(val_esr {float(gen_ck['val_esr']):.5f} at epoch {int(gen_ck['epoch'])})")
    print(f"[model] |e| = {emb_scale:.4f} (the decoder's own embed_scale; the encoder "
          f"emits radius 1 and is rescaled onto that sphere)")
    cap = encoder.capacity_report()
    print(f"[model] head={cap['head']} pool={cap['n_queries']}x{cap['d_model']}"
          f"x{cap['n_blocks']}blk/{cap['n_heads']}h over all {cap['n_frames']} frames "
          f"-> pooled={cap['pooled_dim']} -> De={cap['embedding_dim']} "
          f"(ceiling {cap['ceiling']}) normalize={encoder.normalize}")
    print(f"[model] generator receptive_field={R} samples "
          f"({1000.0 * R / spec.sample_rate:.1f} ms)")

    # --- Data --------------------------------------------------------------------
    train_ds, val_ds, clip, ref_n = _build_datasets(spec, id_to_idx, R)
    val_loader = make_loader(val_ds, tcfg.batch_size, tcfg.num_workers,
                             shuffle=False, seed=spec.seed + 999, drop_last=False)
    leaked = sorted(set(holdout_ids) & {int(d) for d in train_ds.device_ids})
    if leaked:
        raise SystemExit(f"holdout devices {leaked} are in the training set")

    holdout_loader = None
    if holdout_ids:
        h_idx = {int(d): i for i, d in enumerate(sorted(holdout_ids))}
        holdout_ds = build_dataset(spec.data.dataset, split="val",
                                   pairs_per_epoch=tcfg.val_pairs,
                                   seed=spec.seed + 4242,
                                   **_dataset_kwargs(spec, h_idx, R))
        holdout_loader = make_loader(holdout_ds, tcfg.batch_size, tcfg.num_workers,
                                     shuffle=False, seed=spec.seed + 4242,
                                     drop_last=False)
        print(f"[data]  holdout probe: {len(holdout_ds.device_ids)} devices unseen by "
              f"BOTH halves — the decoder never trained on them either")
    print(f"[data]  train_devices={len(train_ds.device_ids)} "
          f"pairs/epoch={spec.data.pairs_per_epoch} clip={clip} ref={ref_n}")

    # --- Loss, optimizer ---------------------------------------------------------
    lossfn = build_loss(spec.loss).to(dev)
    strength = float(spec.loss.device_loss_weighting)
    dev_w = None
    if strength > 0.0:
        _w = device_loss_weights(spec.paths.renders_manifest, ids, strength)
        dev_w = torch.as_tensor(_w, dtype=torch.float32, device=dev)
        print(f"[loss]  device_loss_weighting={strength:g} (as the decoder was trained)")
    # The holdout probe gets its OWN weights, over the holdout ids in h_idx's
    # order. The library trainer leaves this probe unweighted because it fits
    # weights on the *training* ids, which would land on the wrong devices here.
    # A weighted holdout is what makes the number comparable to the decoder run's
    # enrollment, which fitted its vectors under exactly this objective.
    hold_w = None
    if strength > 0.0 and holdout_ids:
        hold_w = torch.as_tensor(
            device_loss_weights(spec.paths.renders_manifest, sorted(holdout_ids),
                                strength), dtype=torch.float32, device=dev)
    noise_frac = float(tcfg.embed_noise_frac)
    # One group, not two: the generator has no trainable parameter to give one.
    enc_wd = (tcfg.weight_decay if spec.joint.enc_weight_decay < 0
              else spec.joint.enc_weight_decay)
    opt = torch.optim.AdamW([{"params": list(encoder.parameters()),
                              "lr": spec.joint.enc_lr, "name": "encoder",
                              "weight_decay": enc_wd}])
    max_epochs = int(tcfg.epochs)
    use_amp, amp_dtype, scaler = amp_setup(tcfg, dev)
    lr_floor = spec.joint.enc_lr * tcfg.cosine_min_lr_frac
    print(f"[optim] AdamW enc_lr={spec.joint.enc_lr:.2e} -> {lr_floor:.2e} "
          f"({tcfg.scheduler}, {max_epochs} epochs, floor "
          f"{tcfg.cosine_min_lr_frac:g} of base) wd={enc_wd:g} "
          f"| batch={tcfg.batch_size} | embed_noise={noise_frac:g} | amp={use_amp}")

    run = RunDir(spec.run_dir)

    # --- Resume ------------------------------------------------------------------
    start_epoch, best_val, since_improved = 0, float("inf"), 0
    resumed_from = None
    if args.resume:
        if not run.last.is_file():
            raise SystemExit(f"--resume: no last.pt under {run.path}")
        ck = torch.load(run.last, map_location=dev, weights_only=False)
        _check_resume_compatible(ck, spec, R, len(ids))
        # The decoder's identity is the experiment, and it is the one thing the
        # structural check cannot see: it compares `model:`, which two different
        # runs of the same architecture share.
        was = ck.get("frozen_decoder")
        if was is not None and was != str(args.emulate_run):
            raise SystemExit(
                f"--resume: this run froze {was!r}, not {str(args.emulate_run)!r}. "
                f"A different decoder behind the same encoder is a different "
                f"experiment — give it its own --name.")
        encoder.load_state_dict(ck["encoder"])
        # Not ck["generator"]: --emulate-run is authoritative for the frozen half,
        # and it never trains, so the checkpoint's copy is the same weights.
        opt.load_state_dict(ck["optim"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        resumed_from = int(ck["epoch"])
        start_epoch = resumed_from + 1
        best_val = float(ck.get("best_val_esr", float("inf")))
        if start_epoch >= max_epochs:
            raise SystemExit(
                f"--resume: the checkpoint is at epoch {resumed_from} and --epochs "
                f"is {max_epochs}, so there is nothing left to run. Raise --epochs.")
        # Config is authoritative on resume (the same rule as training/joint.py):
        # what resuming preserves is AdamW's moment buffers, not the lr. The catch
        # is `initial_lr` -- it rides along inside the loaded param_group, and
        # LambdaLR *setdefault*s it, so a stale one silently overrides --enc-lr and
        # the run keeps annealing from the old rate. Popping it is what makes a new
        # --enc-lr take effect at all.
        for g in opt.param_groups:
            g["lr"], g["weight_decay"] = spec.joint.enc_lr, enc_wd
            g.pop("initial_lr", None)
        print(f"[resume] from epoch {start_epoch} (best_val_esr={best_val:.5f}), "
              f"through {max_epochs}; enc_lr={spec.joint.enc_lr:.2e} wd={enc_wd:g} "
              f"scheduler={tcfg.scheduler} — all from the config, not the checkpoint")
    else:
        run.start_log(LOG_COLUMNS)
    # After the resume block, not before it: a cosine curve reads each group's lr
    # as its starting point, and on resume that lr is the config's (set just
    # above). start_epoch then replays the curve forward to where it left off.
    sched = build_scheduler(opt, tcfg, max_epochs=max_epochs, start_epoch=start_epoch)
    run.save_spec(spec)
    run.append_config_history(mode="resume" if resumed_from is not None else "start",
                              start_epoch=start_epoch, resumed_from=resumed_from,
                              best_val=best_val, max_epochs=max_epochs,
                              spec=spec.to_dict(),
                              frozen_decoder=str(args.emulate_run),
                              frozen_decoder_val_esr=float(gen_ck["val_esr"]))
    manifest_sig = manifest_signature(spec.paths)

    def save_ckpt(path: Path, epoch: int, val_esr: float, best: float) -> None:
        """The library's joint checkpoint schema, so `openamp joint-eval` reads it."""
        run.save_checkpoint(path, {
            "encoder": encoder.state_dict(), "generator": generator.state_dict(),
            "optim": opt.state_dict(), "scaler": scaler.state_dict(),
            "spec": spec.to_dict(), "device_ids": ids, "id_to_idx": id_to_idx,
            "holdout_ids": holdout_ids, "manifest_sha256": manifest_sig,
            "receptive_field": R, "ref_samples": ref_n, "clip_samples": clip,
            "encoder_params": n_enc, "generator_params": n_gen,
            "sample_rate": spec.sample_rate, "seed": spec.seed, "name": spec.name,
            "epoch": epoch, "val_esr": val_esr, "best_val_esr": best,
            "frozen_decoder": str(args.emulate_run),
        })

    # --- Loop --------------------------------------------------------------------
    t0 = time.time()
    # Reconstructed the way the library trainer does it, so the CSV's step column
    # stays continuous across a resume. elapsed_s does not: it is time in *this*
    # session, again as the trainer logs it.
    step = start_epoch * (spec.data.pairs_per_epoch // tcfg.batch_size)
    probe_batches = max(1, min(25, tcfg.val_pairs // max(tcfg.batch_size, 1)))
    for epoch in range(start_epoch, max_epochs):
        train_ds.seed = spec.seed + epoch
        loader = make_loader(train_ds, tcfg.batch_size, tcfg.num_workers,
                             shuffle=True, seed=spec.seed + epoch, drop_last=True)
        model.train()
        ep_loss = ep_esr = ep_stft = 0.0
        n_steps, ep_t0 = 0, time.time()
        for batch in loader:
            inp = batch["input"].to(dev, non_blocking=True)
            target = batch["target"].to(dev, non_blocking=True).unsqueeze(1)
            ref = batch["ref"].to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                # Jitter on the unit sphere (what its frac/sqrt(De) scale assumes),
                # then onto the decoder's sphere -- doing it the other way round
                # would let jitter's own re-normalize undo embed_scale.
                e = generator.project_embedding(
                    jitter(encoder(ref), noise_frac, encoder.normalize))
                out = model.render(inp, e)[..., R:]
            w = (None if dev_w is None else
                 dev_w[batch["device_idx"].to(dev, non_blocking=True)])
            loss, parts = lossfn(out.float(), target, weights=w)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # Encoder only: the generator's parameters have no .grad to clip.
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()

            ep_loss += loss.item(); ep_esr += parts["esr"]; ep_stft += parts["stft"]
            n_steps += 1
            step += 1
            if step % tcfg.log_every == 0:
                sps = n_steps * tcfg.batch_size / (time.time() - ep_t0 + 1e-9)
                print(f"  e{epoch:02d} step {step:>7d} loss {loss.item():.5f} "
                      f"esr {parts['esr']:.5f} stft {parts['stft']:.4f} "
                      f"{sps:5.1f} smp/s", flush=True)

        tr_loss, tr_esr, tr_stft = (ep_loss / max(n_steps, 1), ep_esr / max(n_steps, 1),
                                    ep_stft / max(n_steps, 1))
        val = evaluate_esr(model, val_loader, dev, lossfn.coeff, seed=spec.seed,
                           weights=dev_w)
        val_esr, spread = val["esr"], val["emb_spread"]
        shuf = evaluate_esr(model, val_loader, dev, lossfn.coeff, mode="shuffled",
                            n_batches=probe_batches, seed=spec.seed,
                            weights=dev_w)["esr"]
        def probe_holdout(w):
            return (evaluate_esr(model, holdout_loader, dev, lossfn.coeff,
                                 n_batches=probe_batches, seed=spec.seed,
                                 weights=w)["esr"]
                    if holdout_loader is not None else float("nan"))
        # Unweighted for comparison with results/joint/*, weighted for comparison
        # with the enrollment ceiling. A non-zero strength makes the two
        # incomparable with each other -- see losses.device_loss_weights.
        hold, hold_wt = probe_holdout(None), probe_holdout(hold_w)
        if not np.isfinite(val_esr):
            raise SystemExit(f"[nan] epoch {epoch}: non-finite val ESR. With a frozen "
                             f"decoder this is an encoder or data problem, not "
                             f"optimizer drift — inspect rather than resume.")

        step_scheduler(sched, val_esr)
        improved = val_esr < best_val
        if improved:
            best_val, since_improved = val_esr, 0
            save_ckpt(run.checkpoint, epoch, val_esr, best_val)
        else:
            since_improved += 1
        save_ckpt(run.last, epoch, val_esr, best_val)
        run.append_log([epoch, step, f"{tr_loss:.6f}", f"{tr_esr:.6f}",
                        f"{tr_stft:.6f}", f"{val_esr:.6f}", f"{shuf:.6f}",
                        f"{hold:.6f}", f"{hold_wt:.6f}", f"{spread:.6f}",
                        f"{param_group_lrs(opt)['encoder']:.2e}",
                        f"{time.time() - t0:.1f}"])
        ratio = shuf / val_esr if val_esr > 0 else float("nan")
        print(f"[epoch {epoch:02d}] train_esr={tr_esr:.5f} val_esr={val_esr:.5f} "
              f"best={best_val:.5f} shuffled={shuf:.5f} (x{ratio:.1f}) "
              f"holdout={hold:.5f}/{hold_wt:.5f} emb_spread={spread:.4f}"
              f"{'  *saved*' if improved else ''}", flush=True)
        if since_improved >= tcfg.early_stop_patience:
            print(f"[stop] no val improvement for {since_improved} epochs")
            break

    # --- Final diagnostics -------------------------------------------------------
    if run.checkpoint.is_file():
        encoder.load_state_dict(torch.load(run.checkpoint, map_location=dev,
                                           weights_only=False)["encoder"])
    final = {m: evaluate_esr(model, val_loader, dev, lossfn.coeff, mode=m,
                             seed=spec.seed, weights=dev_w) for m in EVAL_MODES}
    hold_full, hold_full_w = (
        (evaluate_esr(model, holdout_loader, dev, lossfn.coeff, seed=spec.seed,
                      weights=w)["esr"] if holdout_loader is not None else float("nan"))
        for w in (None, hold_w))
    ceiling = enroll_ceiling(args.emulate_run)
    val_true, val_shuf = final["true"]["esr"], final["shuffled"]["esr"]
    spread = final["true"]["emb_spread"]
    verdict = ("OK" if val_shuf > 2 * val_true and spread > 1e-3 else
               "WEAK — the encoder may have collapsed; see joint-eval")
    print(f"[cond]  val_esr={val_true:.5f} shuffled={val_shuf:.5f} "
          f"zero={final['zero']['esr']:.5f} emb_spread={spread:.4f} -> {verdict}")
    if ceiling:
        target = ceiling.get("best_val_esr_pooled")
        gap = f"{hold_full_w / target:.1f}x off" if target else "n/a"
        print(f"[target] holdout weighted={hold_full_w:.5f} vs enrollment's "
              f"{target:.5f} on the same {ceiling.get('n_enrolled')} amps -> {gap}")
        print(f"[target] enrollment's test_esr mean={ceiling.get('test_esr_mean')} "
              f"median={ceiling.get('test_esr_median')} is a DIFFERENT statistic "
              f"(raw per-window, silence-gated) — not comparable to anything here")
    metrics = run.save_metrics({
        "name": spec.name, "frozen_decoder": str(args.emulate_run),
        "decoder_val_esr": float(gen_ck["val_esr"]),
        "encoder_params": n_enc, "generator_params_frozen": n_gen,
        "best_val_esr": best_val, "val_esr": val_true,
        "val_esr_shuffled": val_shuf, "val_esr_zero": final["zero"]["esr"],
        "val_esr_holdout": hold_full, "val_esr_holdout_weighted": hold_full_w,
        "enroll_ceiling": ceiling, "embed_scale": emb_scale,
        "device_loss_weighting": strength, "embed_noise_frac": noise_frac,
        "emb_spread": spread, "verdict": verdict,
        "n_devices": len(ids), "n_holdout": len(holdout_ids),
        "train_hours": round((time.time() - t0) / 3600.0, 4),
    })
    print(f"[done]  best_val_esr={best_val:.5f} holdout={hold_full:.5f} "
          f"(weighted {hold_full_w:.5f}) -> {spec.run_dir}")
    print(f"[next]  openamp joint-eval {spec.run_dir}")
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--emulate-run", type=Path,
                   default=Path("results/emulate/delta_a2_tanh_r32_512_v2"),
                   help="run dir whose checkpoint.pt supplies the frozen decoder")
    p.add_argument("--frames", type=Path,
                   default=Path("results/dac_frames/sweep_train"),
                   help="openamp.probes.extract output the encoder reads")
    p.add_argument("--name", default=None,
                   help="joint run name (default: frozen_<decoder run name>)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-queries", type=int, default=8)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--no-positional", action="store_true",
                   help="drop the sinusoidal positions: the pool then reads the "
                        "frames as an unordered bag")
    p.add_argument("--enc-lr", type=float, default=5.0e-4,
                   help="cosine starts here")
    p.add_argument("--enc-lr-min", type=float, default=None,
                   help="absolute lr the cosine anneals down to over --epochs "
                        "(default: the decoder run's cosine_min_lr_frac of --enc-lr)")
    p.add_argument("--enc-weight-decay", type=float, default=0.01)
    p.add_argument("--embed-noise", type=float, default=0.1,
                   help="jitter e by this fraction on train steps")
    p.add_argument("--ref-seconds", type=float, default=2.0,
                   help="inert under a lookup encoder; kept so the spec is complete")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None,
                   help="default: the decoder run's own")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="continue this --name from its last.pt; the config wins on "
                        "lr, weight decay, scheduler and epochs")
    p.add_argument("--init-encoder", type=Path, default=None,
                   help="warm-start the encoder from a probes.probe_attn checkpoint "
                        "(probe_attn.pt); refused if n_queries/d_model/n_heads/"
                        "mlp_ratio/positional don't match this run's own. Ignored "
                        "if --resume finds an existing last.pt for this --name.")
    args = p.parse_args(argv)
    if args.name is None:
        args.name = f"frozen_{Path(args.emulate_run).name}"
    return args


if __name__ == "__main__":
    train(parse_args())
