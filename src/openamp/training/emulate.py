"""The table-conditioned training loop: one embedding row per device.

- Takes a :class:`~openamp.spec.RunSpec` and nothing else: the architecture
  (``model:``), the dataset (``data:``), the objective (``loss:``) and the
  schedule (``training:``) all arrive through registries, so the loop below
  never names one of them. AdamW, reduce-on-plateau, early stopped at the
  val-ESR plateau. Plain single-GPU PyTorch.
- Every run writes to results/emulate/<name>/: best/last checkpoints,
  resolved_config.yaml, metrics.json, train_log.csv (per-epoch curves).
- Sanity ladder before any long job: overfit_one_batch (one batch -> ESR ~0),
  then a mini-run (--limit-devices) checking val_esr_shuffled is worse than
  val_esr (proves the conditioning is doing real work).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from openamp.data import (build_dataset, make_loader, manifest_signature,
                          training_device_index)
from openamp.evaluation.metrics import pooled_esr_terms
from openamp.models import build_model, count_parameters
from openamp.losses import build_loss, device_loss_weights
from openamp.spec import RunSpec
from openamp.training import (GRAD_CLIP_NORM, RunDir, amp_setup, build_scheduler,
                              step_scheduler,
                              check_resume_compatible)

LOG_COLUMNS = ["epoch", "step", "train_loss", "train_esr", "train_stft", "val_esr",
               "lr", "elapsed_s"]


def _conditioning(model, device_idx: torch.Tensor, noise_frac: float,
                  prefix_sizes: torch.Tensor | None = None,
                  prefix_full_frac: float = 0.0) -> torch.Tensor:
    """The batch's embedding rows, optionally jittered and/or cut (train steps only).

    Noise std is ``noise_frac * table.std()`` recomputed each step, not a fixed
    number of embedding units: the table's scale drifts over a run, so a
    relative size keeps the perturbation the same fraction of the space
    throughout. Read off the whole table (not the batch) so a batch of similar
    devices doesn't shrink its own noise.

    The noisy vector is re-projected through ``model.project_embedding`` so an
    ``embed_normalize`` model gets its noise back on the sphere at the same
    radius (direction only jitters); on a model without ``embed_normalize`` this
    is a no-op and the raw noisy vector passes through unchanged.

    ``prefix_sizes`` (nested dropout): each row keeps its first ``k`` dims and
    zeroes the rest, ``k`` drawn per row from ``prefix_sizes``, or all dims with
    probability ``prefix_full_frac``. The cut comes *after* the projection, so a
    kept prefix holds its share of the full vector's size rather than being
    re-inflated to the sphere's radius -- a short prefix has to mean "less of
    the same amp", not a different, full-size one.
    """
    emb = model.embed(device_idx)
    if noise_frac > 0.0:
        std = noise_frac * model.embedding_table().detach().std()
        emb = model.project_embedding(emb + torch.randn_like(emb) * std)
    if prefix_sizes is None:
        return emb
    B, E = emb.shape
    k = prefix_sizes[torch.randint(len(prefix_sizes), (B,), device=emb.device)]
    full = torch.rand(B, device=emb.device) < prefix_full_frac
    k = torch.where(full, torch.full_like(k, E), k)
    keep = torch.arange(E, device=emb.device) < k.unsqueeze(1)
    return emb * keep


@torch.no_grad()
def evaluate_esr(model, loader, device, coeff: float, *, n_batches: int | None = None,
                 shuffle_devices: bool = False,
                 weights: torch.Tensor | None = None) -> float:
    """Energy-weighted pre-emphasized ESR over the loader (warmed region only).

    - The pooled definition from :mod:`openamp.evaluation.metrics`: error and
      signal energy accumulated across every window, divided once (a mean of
      per-window ratios would be dominated by quiet windows).
    - ``shuffle_devices`` permutes the conditioning within each batch — the
      control that should make ESR *worse* if the embeddings actually steer the
      model.
    - ``weights`` [n_devices] applies the same per-device weighting the train
      loss uses (:func:`device_loss_weights`), so val ESR scores the objective
      the run is actually optimizing. Indexed by each item's *true* device, not
      the shuffled conditioning — the target is the true device's render either
      way. At strength 1 the result equals the plain mean of per-device ESR.
    """
    was_training = model.training
    model.eval()
    R = model.receptive_field
    num = den = 0.0
    for k, batch in enumerate(loader):
        if n_batches is not None and k >= n_batches:
            break
        inp = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        di = batch["device_idx"].to(device, non_blocking=True)
        w = None if weights is None else weights[di]
        if shuffle_devices and di.shape[0] > 1:
            di = di[torch.randperm(di.shape[0], device=device)]
        out = model(inp, di)[..., R:].squeeze(1)      # [B, clip]
        n_i, d_i = pooled_esr_terms(out, target, coeff=coeff, weights=w)
        num += float(n_i.sum().double())
        den += float(d_i.sum().double())
    if was_training:
        model.train()
    return num / (den + 1e-8) if den > 0.0 else float("nan")


# --- Sanity ladder -------------------------------------------------------------
def overfit_one_batch(spec: RunSpec, *, device: str = "cuda",
                      steps: int = 400, limit_devices: int = 8) -> float:
    """Sanity #1: a single batch should reach ~zero ESR (spec §4.2)."""
    tcfg = spec.training
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(spec.seed)

    ids, id_to_idx, _ = training_device_index(spec, limit_devices)
    model = build_model(spec.model, len(ids)).to(dev)
    R = model.receptive_field
    ds = build_dataset(spec.data.dataset, paths=spec.paths, split="train",
                       receptive_field=R, id_to_idx=id_to_idx,
                       clip_samples=spec.clip_samples,
                       pairs_per_epoch=tcfg.batch_size, seed=spec.seed,
                       sources=spec.data.sources, **spec.data.kwargs)
    loader = make_loader(ds, tcfg.batch_size, 0, shuffle=False, seed=spec.seed,
                         drop_last=False)
    batch = next(iter(loader))
    inp = batch["input"].to(dev)
    target = batch["target"].to(dev).unsqueeze(1)     # [B, 1, clip]
    di = batch["device_idx"].to(dev)

    lossfn = build_loss(spec.loss).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
    print(f"[overfit] devices={len(ids)} params={count_parameters(model):,} "
          f"receptive_field={R} ({1000 * R / spec.sample_rate:.1f} ms)")
    final = float("nan")
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(inp, di)[..., R:]
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
    """Train the conditioned model to a val-ESR plateau; returns a summary dict."""
    tcfg, name = spec.training, spec.name
    max_epochs = int(epochs if epochs is not None else tcfg.epochs)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    torch.backends.cudnn.benchmark = True

    # --- Table + model ----------------------------------------------------------
    ids, id_to_idx, holdout_ids = training_device_index(spec, limit_devices)
    if not ids:
        raise RuntimeError("No render-ok devices. Run `openamp render` first.")
    paths = spec.paths
    manifest_sig = manifest_signature(paths)
    model = build_model(spec.model, len(ids)).to(dev)
    R = model.receptive_field
    n_params = count_parameters(model)
    clip = spec.clip_samples
    rf_ms = 1000.0 * R / spec.sample_rate
    arch = model.describe()
    print(f"[model] name={name} arch={arch['arch']} params={n_params:,} "
          f"devices={len(ids)} holdout={len(holdout_ids)} "
          f"channels={arch['channels']} layers={arch['blocks_x_layers']} "
          f"embed={model.embedding_dim}")
    print(f"[model] receptive_field={R} samples ({rf_ms:.1f} ms)  clip={clip}")

    # Embedding noise: train steps only (val/eval read the table verbatim).
    if tcfg.embed_noise_frac < 0.0:
        raise ValueError("training.embed_noise_frac must be >= 0 (fraction of the "
                         f"embedding table's std), got {tcfg.embed_noise_frac}")
    noise_frac = float(tcfg.embed_noise_frac)
    if noise_frac > 0.0:
        print(f"[model] embed_noise_frac={noise_frac:g}")

    # Nested dropout on the embedding: train steps only (val/eval read every dim).
    sizes = tuple(int(s) for s in tcfg.embed_prefix_sizes)
    full_frac = float(tcfg.embed_prefix_full_frac)
    prefix_sizes = None
    if sizes:
        E = model.embedding_dim
        if any(s < 1 or s > E for s in sizes):
            raise ValueError(f"training.embed_prefix_sizes must lie in [1, {E}] "
                             f"(the embedding dim), got {list(sizes)}")
        if not 0.0 <= full_frac <= 1.0:
            raise ValueError("training.embed_prefix_full_frac must be in [0, 1], "
                             f"got {full_frac}")
        prefix_sizes = torch.tensor(sizes, dtype=torch.long, device=dev)
        # How often the last dim is trained at all -- the tail's share of signal.
        tail = full_frac + (1.0 - full_frac) * sum(s == E for s in sizes) / len(sizes)
        print(f"[model] embed_prefix_sizes={list(sizes)} full_frac={full_frac:g}  "
              f"(last dim active on {100 * tail:.0f}% of samples)")

    # --- Data -------------------------------------------------------------------
    common = dict(paths=paths, receptive_field=R, id_to_idx=id_to_idx,
                  clip_samples=clip, sources=spec.data.sources, **spec.data.kwargs)
    train_ds = build_dataset(spec.data.dataset, split="train",
                             pairs_per_epoch=spec.data.pairs_per_epoch,
                             seed=spec.seed, **common)
    val_ds = build_dataset(spec.data.dataset, split="val",
                           pairs_per_epoch=tcfg.val_pairs,
                           seed=spec.seed + 999, **common)
    val_loader = make_loader(val_ds, tcfg.batch_size, tcfg.num_workers,
                              shuffle=False, seed=spec.seed + 999, drop_last=False)
    print(f"[data]  train_devices={len(train_ds.device_ids)} "
          f"val_devices={len(val_ds.device_ids)} "
          f"pairs/epoch={spec.data.pairs_per_epoch}")

    # --- Optim / loss / bookkeeping --------------------------------------------
    lossfn = build_loss(spec.loss).to(dev)
    # Per-device ESR weighting, applied to the train loss and val ESR alike.
    strength = float(spec.loss.device_loss_weighting)
    if strength < 0.0:
        raise ValueError(f"loss.device_loss_weighting must be >= 0, got {strength}")
    dev_w = None
    if strength > 0.0:
        _w = device_loss_weights(paths.renders_manifest, ids, strength)
        dev_w = torch.as_tensor(_w, dtype=torch.float32, device=dev)
        print(f"[loss]  device_loss_weighting={strength:g} "
              f"weights min={_w.min():.4g} p50={np.median(_w):.4g} "
              f"max={_w.max():.4g} (mean 1.0, ESR term only; val ESR weighted "
              f"too — not comparable with a strength-0 run)")
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    use_amp, amp_dtype, scaler = amp_setup(tcfg, dev)

    run = RunDir(spec.run_dir)
    ckpt_path, last_path = run.checkpoint, run.last

    start_epoch, best_val, since_improved = 0, float("inf"), 0
    resumed_from = None
    if resume and last_path.is_file():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        _check_resume_compatible(ck, spec, R, len(ids))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        resumed_from = int(ck["epoch"])
        start_epoch = resumed_from + 1
        best_val = float(ck.get("best_val_esr", float("inf")))
        # Config wins over the checkpoint's own lr/weight_decay, so a resumed
        # run can be hand-annealed; AdamW's moment buffers are still preserved.
        for g in opt.param_groups:
            g["lr"], g["weight_decay"] = tcfg.lr, tcfg.weight_decay
        print(f"[resume] from epoch {start_epoch}, best_val_esr={best_val:.5f}; "
              f"lr={tcfg.lr:.2e} weight_decay={tcfg.weight_decay:g} (from config)")
    else:
        run.start_log(LOG_COLUMNS)
    # After the resume block: see the note in training/joint.py -- a cosine
    # schedule starts from the lr the config just applied, not the checkpoint's.
    sched = build_scheduler(opt, tcfg, max_epochs=max_epochs, start_epoch=start_epoch)
    run.append_config_history(
        mode="resume" if resumed_from is not None else "start",
        start_epoch=start_epoch, resumed_from=resumed_from, best_val=best_val,
        max_epochs=max_epochs, applied_lr=opt.param_groups[0]["lr"],
        applied_weight_decay=opt.param_groups[0]["weight_decay"],
        spec=spec.to_dict())

    def save_ckpt(path: Path, epoch: int, val_esr: float) -> None:
        run.save_checkpoint(path, {
            "model": model.state_dict(), "optim": opt.state_dict(),
            "scaler": scaler.state_dict(), "spec": spec.to_dict(),
            "device_ids": ids, "id_to_idx": id_to_idx, "holdout_ids": holdout_ids,
            "manifest_sha256": manifest_sig, "receptive_field": R,
            "params": n_params, "sample_rate": spec.sample_rate, "seed": spec.seed,
            "name": name, "epoch": epoch, "val_esr": val_esr, "best_val_esr": best_val,
        })

    # --- Loop -------------------------------------------------------------------
    t0 = time.time()
    step = start_epoch * (spec.data.pairs_per_epoch // tcfg.batch_size)
    epoch = start_epoch - 1                         # defined even if the loop is empty
    for epoch in range(start_epoch, max_epochs):
        train_ds.seed = spec.seed + epoch          # fresh window positions each epoch
        loader = make_loader(train_ds, tcfg.batch_size, tcfg.num_workers,
                              shuffle=True, seed=spec.seed + epoch, drop_last=True)
        model.train()
        ep_loss = ep_esr = ep_stft = 0.0
        n_steps = 0
        ep_t0 = time.time()      # rate print resets per epoch; t0 stays cumulative
        for batch in loader:
            inp = batch["input"].to(dev, non_blocking=True)
            target = batch["target"].to(dev, non_blocking=True).unsqueeze(1)
            di = batch["device_idx"].to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model.forward_emb(inp, _conditioning(model, di, noise_frac,
                                                           prefix_sizes, full_frac))[..., R:]
            loss, parts = lossfn(out.float(), target,
                                 weights=None if dev_w is None else dev_w[di])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)      # GradScaler only skips non-finite grads; clip the rest
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()

            ep_loss += loss.item(); ep_esr += parts["esr"]; ep_stft += parts["stft"]
            n_steps += 1
            step += 1
            if step % tcfg.log_every == 0:
                sps = n_steps * tcfg.batch_size / (time.time() - ep_t0 + 1e-9)
                print(f"  e{epoch:02d} step {step:>7d} loss {loss.item():.5f} "
                      f"esr {parts['esr']:.5f} stft {parts['stft']:.4f} {sps:5.0f} smp/s",
                      flush=True)      # stdout is block-buffered under nohup/redirects

        tr_loss = ep_loss / max(n_steps, 1)
        tr_esr = ep_esr / max(n_steps, 1)
        tr_stft = ep_stft / max(n_steps, 1)
        val_esr = evaluate_esr(model, val_loader, dev, lossfn.coeff, weights=dev_w)

        # NaN guard: recover from an fp16 forward overflow instead of dying.
        # - Non-finite val ESR: weights are poisoned. Reload best, drop to fp32
        #   (or halve lr if already fp32).
        # - Non-finite train loss with finite val: GradScaler absorbed the
        #   overflow. Drop to fp32 and keep everything.
        if not np.isfinite(val_esr):
            if not ckpt_path.is_file():
                raise RuntimeError(
                    f"[nan-guard] epoch {epoch}: non-finite val ESR before any best "
                    "checkpoint exists — check model/data wiring, not precision")
            ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"])
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
            run.append_log([epoch, step, f"{tr_loss:.6f}", f"{tr_esr:.6f}",
                            f"{tr_stft:.6f}", f"{val_esr:.6f}",
                            f"{opt.param_groups[0]['lr']:.2e}",
                            f"{time.time() - t0:.1f}"])
            continue                                # dead epoch: no sched/best/last updates
        if use_amp and not np.isfinite(tr_loss):
            use_amp, scaler = False, torch.amp.GradScaler("cuda", enabled=False)
            print(f"[nan-guard] epoch {epoch:02d}: non-finite train loss but finite val "
                  f"({val_esr:.5f}) — overflow absorbed; AMP off, fp32 from here")

        step_scheduler(sched, val_esr)
        cur_lr = opt.param_groups[0]["lr"]
        improved = val_esr < best_val
        if improved:
            best_val, since_improved = val_esr, 0
            save_ckpt(ckpt_path, epoch, val_esr)
        else:
            since_improved += 1
        save_ckpt(last_path, epoch, val_esr)
        run.append_log([epoch, step, f"{tr_loss:.6f}", f"{tr_esr:.6f}",
                        f"{tr_stft:.6f}", f"{val_esr:.6f}", f"{cur_lr:.2e}",
                        f"{time.time() - t0:.1f}"])
        print(f"[epoch {epoch:02d}] train_esr={tr_esr:.5f} val_esr={val_esr:.5f} "
              f"best={best_val:.5f} lr={cur_lr:.1e}{'  *saved*' if improved else ''}",
              flush=True)
        if since_improved >= tcfg.early_stop_patience:
            print(f"[stop] no val improvement for {since_improved} epochs")
            break

    # --- Reload best; conditioning check; final artifacts -----------------------
    if ckpt_path.is_file():
        model.load_state_dict(torch.load(ckpt_path, map_location=dev,
                                         weights_only=False)["model"])
    val_true = evaluate_esr(model, val_loader, dev, lossfn.coeff, weights=dev_w)
    val_shuf = evaluate_esr(model, val_loader, dev, lossfn.coeff, shuffle_devices=True,
                            weights=dev_w)
    train_hours = (time.time() - t0) / 3600.0
    print(f"[cond]  val_esr={val_true:.5f}  val_esr_shuffled={val_shuf:.5f} "
          f"(shuffle should be worse){' OK' if val_shuf > val_true else ' — weak conditioning'}")

    run.save_spec(spec)
    emb_dim = model.embedding_dim
    save_embedding(run.path / "embedding.pt", model, ids, manifest_sig, emb_dim, name,
                   holdout_ids=holdout_ids)
    metrics = {
        "name": name, "arch": arch["arch"], "params": n_params,
        "receptive_field_samples": R, "receptive_field_ms": round(rf_ms, 3),
        "embedding_dim": emb_dim, "channels": arch["channels"],
        "blocks_x_layers": arch["blocks_x_layers"], "n_devices": len(ids),
        "device_loss_weighting": strength,
        "best_val_esr": best_val,
        "val_esr": val_true, "val_esr_shuffled": val_shuf,
        "epochs_run": epoch + 1, "train_hours": round(train_hours, 4),
        "manifest_sha256": manifest_sig,
    }
    run.save_metrics(metrics)
    print(f"[done]  best_val_esr={best_val:.5f} train_hours={train_hours:.2f} -> {run.path}")
    return metrics


def save_embedding(path: Path, model, device_ids, manifest_sig, embedding_dim, name, *,
                   holdout_ids=()) -> None:
    """Save the embedding table + provenance separately (Phase 5 extends it, spec §5)."""
    torch.save({
        "embedding": model.embedding_table().detach().cpu(),
        "device_ids": list(device_ids), "holdout_ids": list(holdout_ids),
        "manifest_sha256": manifest_sig,
        "embedding_dim": int(embedding_dim), "name": name,
    }, path)


def _check_resume_compatible(ck: dict, spec: RunSpec, receptive_field: int,
                             n_devices: int) -> None:
    """Fail fast if a config edit made the checkpoint's weights incompatible.

    The whole ``model:`` section is structural -- every key in it changes the
    network. Everything else in the spec is training dynamics and may change.
    """
    check_resume_compatible(
        ck, [(ck.get("spec", {}).get("model", {}), spec.model.to_dict(), "model.")],
        receptive_field=receptive_field, n_devices=n_devices,
        dynamics="lr, weight_decay, scheduler, plateau_*, cosine_*, epochs, loss, batch_size")
