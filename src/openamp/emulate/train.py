"""One training script for the amp foundation models (spec §4.2).

Trains the configured architecture (``emulate.arch``: FiLM-TCN or the A2
FiLM-WaveNet, built by :func:`openamp.emulate.models.build_model`) on
:class:`openamp.emulate.dataset.EmulationDataset`: pre-emphasized ESR +
multi-resolution STFT (auraloss), 1:1; Adam lr 5e-4, reduce-on-plateau, early
stopped at the val-ESR plateau. Plain single-GPU PyTorch — no Lightning. Every
run writes to ``results/emulate/<name>/``: best/last checkpoints, a copy of the
config, ``metrics.json`` (param count, receptive field, train hours), and
``train_log.csv`` (per-epoch train/val curves).

Sanity ladder (cheap, run before any long job):
  1. ``overfit_one_batch`` — a single batch should drive ESR to ~0.
  2. mini-run (``--limit-devices 10``) — devices sound distinct, and shuffling the
     embeddings across devices should *hurt* val ESR (proves conditioning works;
     reported as ``val_esr_shuffled`` at the end of every run).
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import auraloss
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from openamp.core.config import Config
from openamp.emulate.dataset import (EmulationDataset, build_device_index,
                                     load_or_create_holdout, manifest_signature,
                                     render_ok_devices)
from openamp.emulate.models import arch_summary, build_model
from openamp.emulate.tcn import count_parameters

# Safety net for loss spikes, not a training knob: healthy grad norms on this
# model sit at p50~4, p90~7, so 20 leaves normal steps untouched.
GRAD_CLIP_NORM = 20.0


# --- Losses --------------------------------------------------------------------
def pre_emphasis(x: torch.Tensor, coeff: float) -> torch.Tensor:
    """1st-order pre-emphasis ``y[n] = x[n] - coeff * x[n-1]`` (last-dim FIR)."""
    return x[..., 1:] - coeff * x[..., :-1]


def esr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Energy-weighted error-to-signal ratio ``sum||e||^2 / sum||t||^2`` over the batch.

    Pooled as one ratio of sums, not a mean of per-window ratios: window target
    energy spans ~8 orders of magnitude (near-silent passages in clean guitar),
    so per-window ratios let a handful of quiet windows dominate the batch.
    """
    num = torch.sum((pred - target) ** 2, dim=-1)
    den = torch.sum(target ** 2, dim=-1)
    return num.sum() / (den.sum() + eps)


def preemph_esr(pred: torch.Tensor, target: torch.Tensor, coeff: float,
                eps: float = 1e-8) -> torch.Tensor:
    """ESR on pre-emphasized signals — weights the high end amps distort most."""
    return esr(pre_emphasis(pred, coeff), pre_emphasis(target, coeff), eps)


class EmulationLoss(nn.Module):
    """Pre-emphasized ESR + multi-resolution STFT (auraloss), summed 1:1 (spec §4.2)."""

    def __init__(self, preemph_coeff: float = 0.85, stft_weight: float = 1.0):
        super().__init__()
        self.coeff = float(preemph_coeff)
        self.stft_weight = float(stft_weight)
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """``pred``/``target`` are ``[B, 1, T]``; returns ``(loss, parts)``."""
        esr_term = preemph_esr(pred, target, self.coeff)
        stft_term = self.mrstft(pred, target)
        loss = esr_term + self.stft_weight * stft_term
        return loss, {"esr": float(esr_term.item()), "stft": float(stft_term.item())}


# --- Data plumbing -------------------------------------------------------------
def _device_index(cfg: Config, ecfg, limit_devices: int | None):
    """Resolve ``(device_ids, id_to_idx, holdout_ids)`` for a run.

    One-to-one baseline picks a single device (no holdout — Phase 5 targets
    held-out devices this way); otherwise the ``holdout_frac`` device holdout is
    excluded first, then ``--limit-devices N`` keeps the first N (the mini-run).
    """
    if ecfg.single_device is not None and ecfg.single_device >= 0:
        ids, id_to_idx = build_device_index(cfg, ecfg.single_device)
        return ids, id_to_idx, []
    holdout = load_or_create_holdout(cfg, ecfg.holdout_frac, cfg.seed)
    ids, _ = build_device_index(cfg, exclude=holdout)
    if limit_devices:
        ids = ids[:int(limit_devices)]
    return ids, {d: i for i, d in enumerate(ids)}, holdout


def _make_loader(ds, batch_size, workers, *, shuffle, seed, drop_last):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      drop_last=drop_last, pin_memory=True, generator=gen,
                      persistent_workers=False)


@torch.no_grad()
def evaluate_esr(model, loader, device, coeff: float, *, n_batches: int | None = None,
                 shuffle_devices: bool = False) -> float:
    """Energy-weighted pre-emphasized ESR over the loader (warmed region only).

    Accumulates error and signal energy across every window and divides once, to
    match :func:`esr`; a mean of per-window ratios is dominated by quiet windows.

    ``shuffle_devices`` permutes the conditioning within each batch — the control
    that should make ESR *worse* if the embeddings actually steer the model.
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
        if shuffle_devices and di.shape[0] > 1:
            di = di[torch.randperm(di.shape[0], device=device)]
        out = model(inp, di)[..., R:].squeeze(1)      # [B, clip]
        pe_o, pe_t = pre_emphasis(out, coeff), pre_emphasis(target, coeff)
        num += float(torch.sum((pe_o - pe_t) ** 2).double())
        den += float(torch.sum(pe_t ** 2).double())
    if was_training:
        model.train()
    return num / (den + 1e-8) if den > 0.0 else float("nan")


# --- Sanity ladder -------------------------------------------------------------
def overfit_one_batch(cfg: Config, *, name: str = "overfit", device: str = "cuda",
                      steps: int = 400, limit_devices: int = 8) -> float:
    """Sanity #1: a single batch should reach ~zero ESR (spec §4.2)."""
    ecfg = cfg.emulate
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    ids, id_to_idx, _ = _device_index(cfg, ecfg, limit_devices)
    model = build_model(ecfg, len(ids)).to(dev)
    R = model.receptive_field
    clip = int(round(ecfg.clip_seconds * cfg.sample_rate))
    ds = EmulationDataset(cfg, "train", receptive_field=R, id_to_idx=id_to_idx,
                          clip_samples=clip, pairs_per_epoch=ecfg.batch_size, seed=cfg.seed)
    loader = _make_loader(ds, ecfg.batch_size, 0, shuffle=False, seed=cfg.seed, drop_last=False)
    batch = next(iter(loader))
    inp = batch["input"].to(dev)
    target = batch["target"].to(dev).unsqueeze(1)     # [B, 1, clip]
    di = batch["device_idx"].to(dev)

    lossfn = EmulationLoss(ecfg.preemph, ecfg.stft_weight).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=ecfg.lr)
    print(f"[overfit] devices={len(ids)} params={count_parameters(model):,} "
          f"receptive_field={R} ({1000 * R / cfg.sample_rate:.1f} ms)")
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
def train(cfg: Config, *, name: str = "default", device: str = "cuda",
          epochs: int | None = None, resume: bool = False,
          limit_devices: int | None = None) -> dict:
    """Train the FiLM-TCN to a val-ESR plateau; returns a small summary dict."""
    ecfg = cfg.emulate
    max_epochs = int(epochs if epochs is not None else ecfg.epochs)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    # --- Table + model ----------------------------------------------------------
    ids, id_to_idx, holdout_ids = _device_index(cfg, ecfg, limit_devices)
    if not ids:
        raise RuntimeError("No render-ok devices. Run `openamp render` first.")
    manifest_sig = manifest_signature(cfg)
    model = build_model(ecfg, len(ids)).to(dev)
    R = model.receptive_field
    n_params = count_parameters(model)
    clip = int(round(ecfg.clip_seconds * cfg.sample_rate))
    rf_ms = 1000.0 * R / cfg.sample_rate
    arch = arch_summary(ecfg)
    print(f"[model] name={name} arch={arch['arch']} params={n_params:,} "
          f"devices={len(ids)} holdout={len(holdout_ids)} "
          f"channels={arch['channels']} layers={arch['blocks_x_layers']} "
          f"embed={ecfg.embedding_dim}")
    print(f"[model] receptive_field={R} samples ({rf_ms:.1f} ms)  clip={clip}")

    # --- Data -------------------------------------------------------------------
    train_ds = EmulationDataset(cfg, "train", receptive_field=R, id_to_idx=id_to_idx,
                                clip_samples=clip, pairs_per_epoch=ecfg.pairs_per_epoch,
                                seed=cfg.seed)
    val_ds = EmulationDataset(cfg, "val", receptive_field=R, id_to_idx=id_to_idx,
                              clip_samples=clip, pairs_per_epoch=ecfg.val_pairs,
                              seed=cfg.seed + 999)
    val_loader = _make_loader(val_ds, ecfg.batch_size, ecfg.num_workers,
                              shuffle=False, seed=cfg.seed + 999, drop_last=False)
    print(f"[data]  train_devices={len(train_ds.device_ids)} "
          f"val_devices={len(val_ds.device_ids)} pairs/epoch={ecfg.pairs_per_epoch}")

    # --- Optim / loss / bookkeeping --------------------------------------------
    lossfn = EmulationLoss(ecfg.preemph, ecfg.stft_weight).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=ecfg.lr, weight_decay=ecfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=ecfg.plateau_factor, patience=ecfg.plateau_patience)
    use_amp = bool(ecfg.amp) and dev.type == "cuda"
    if ecfg.amp_dtype not in ("fp16", "bf16"):
        raise ValueError(f"emulate.amp_dtype must be 'fp16' or 'bf16', got {ecfg.amp_dtype!r}")
    amp_dtype = torch.bfloat16 if ecfg.amp_dtype == "bf16" else torch.float16
    # bf16 shares fp32's exponent range, so it needs no loss scaling
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    run_dir = cfg.emulate_run_dir(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"          # best (min val ESR)
    last_path = run_dir / "last.pt"
    log_path = run_dir / "train_log.csv"

    start_epoch, best_val, since_improved = 0, float("inf"), 0
    resumed_from = None
    if resume and last_path.is_file():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        _check_resume_compatible(ck, ecfg, R, len(ids))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        resumed_from = int(ck["epoch"])
        start_epoch = resumed_from + 1
        best_val = float(ck.get("best_val_esr", float("inf")))
        # Config is authoritative on resume: opt.load_state_dict just restored the
        # checkpoint's own lr/weight_decay, but edits to the config file must win so
        # a resumed run can be hand-annealed (read the last lr from train_log.csv,
        # set the config to that or lower, then resume). The Adam moment buffers are
        # preserved — that, not the lr value, is the point of resuming.
        for g in opt.param_groups:
            g["lr"], g["weight_decay"] = ecfg.lr, ecfg.weight_decay
        print(f"[resume] from epoch {start_epoch}, best_val_esr={best_val:.5f}; "
              f"lr={ecfg.lr:.2e} weight_decay={ecfg.weight_decay:g} (from config)")
    else:
        with log_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["epoch", "step", "train_loss", "train_esr",
                                     "train_stft", "val_esr", "lr", "elapsed_s"])
    _append_config_history(
        run_dir, ecfg, mode="resume" if resumed_from is not None else "start",
        start_epoch=start_epoch, resumed_from=resumed_from, best_val=best_val,
        max_epochs=max_epochs, applied_lr=opt.param_groups[0]["lr"],
        applied_wd=opt.param_groups[0]["weight_decay"])

    def save_ckpt(path: Path, epoch: int, val_esr: float) -> None:
        torch.save({
            "model": model.state_dict(), "optim": opt.state_dict(),
            "scaler": scaler.state_dict(), "emulate_cfg": asdict(ecfg),
            "device_ids": ids, "id_to_idx": id_to_idx, "holdout_ids": holdout_ids,
            "manifest_sha256": manifest_sig, "receptive_field": R,
            "params": n_params, "sample_rate": cfg.sample_rate, "seed": cfg.seed,
            "name": name, "epoch": epoch, "val_esr": val_esr, "best_val_esr": best_val,
        }, path)

    # --- Loop -------------------------------------------------------------------
    t0 = time.time()
    step = start_epoch * (ecfg.pairs_per_epoch // ecfg.batch_size)
    epoch = start_epoch - 1                         # defined even if the loop is empty
    for epoch in range(start_epoch, max_epochs):
        train_ds.seed = cfg.seed + epoch           # fresh window positions each epoch
        loader = _make_loader(train_ds, ecfg.batch_size, ecfg.num_workers,
                              shuffle=True, seed=cfg.seed + epoch, drop_last=True)
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
                out = model(inp, di)[..., R:]
            loss, parts = lossfn(out.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)      # GradScaler only skips non-finite grads; clip the rest
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()

            ep_loss += loss.item(); ep_esr += parts["esr"]; ep_stft += parts["stft"]
            n_steps += 1
            step += 1
            if step % ecfg.log_every == 0:
                sps = n_steps * ecfg.batch_size / (time.time() - ep_t0 + 1e-9)
                print(f"  e{epoch:02d} step {step:>7d} loss {loss.item():.5f} "
                      f"esr {parts['esr']:.5f} stft {parts['stft']:.4f} {sps:5.0f} smp/s",
                      flush=True)      # stdout is block-buffered under nohup/redirects

        tr_loss = ep_loss / max(n_steps, 1)
        tr_esr = ep_esr / max(n_steps, 1)
        tr_stft = ep_stft / max(n_steps, 1)
        val_esr = evaluate_esr(model, val_loader, dev, ecfg.preemph)

        # NaN guard: fp16 forward overflow (activations > 65504) once NaN-killed a
        # 52-epoch run — recover instead of dying. Non-finite val ESR means the
        # weights themselves are poisoned: reload best and drop to fp32 (or halve
        # the lr if already fp32). Non-finite train loss with finite val means the
        # GradScaler absorbed the overflow: just drop to fp32 and keep everything.
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
            with log_path.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([epoch, step, f"{tr_loss:.6f}", f"{tr_esr:.6f}",
                                         f"{tr_stft:.6f}", f"{val_esr:.6f}",
                                         f"{opt.param_groups[0]['lr']:.2e}",
                                         f"{time.time() - t0:.1f}"])
            continue                                # dead epoch: no sched/best/last updates
        if use_amp and not np.isfinite(tr_loss):
            use_amp, scaler = False, torch.amp.GradScaler("cuda", enabled=False)
            print(f"[nan-guard] epoch {epoch:02d}: non-finite train loss but finite val "
                  f"({val_esr:.5f}) — overflow absorbed; AMP off, fp32 from here")

        sched.step(val_esr)
        cur_lr = opt.param_groups[0]["lr"]
        improved = val_esr < best_val
        if improved:
            best_val, since_improved = val_esr, 0
            save_ckpt(ckpt_path, epoch, val_esr)
        else:
            since_improved += 1
        save_ckpt(last_path, epoch, val_esr)
        with log_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([epoch, step, f"{tr_loss:.6f}", f"{tr_esr:.6f}",
                                     f"{tr_stft:.6f}", f"{val_esr:.6f}", f"{cur_lr:.2e}",
                                     f"{time.time() - t0:.1f}"])
        print(f"[epoch {epoch:02d}] train_esr={tr_esr:.5f} val_esr={val_esr:.5f} "
              f"best={best_val:.5f} lr={cur_lr:.1e}{'  *saved*' if improved else ''}",
              flush=True)
        if since_improved >= ecfg.early_stop_patience:
            print(f"[stop] no val improvement for {since_improved} epochs")
            break

    # --- Reload best; conditioning check; final artifacts -----------------------
    if ckpt_path.is_file():
        model.load_state_dict(torch.load(ckpt_path, map_location=dev,
                                         weights_only=False)["model"])
    val_true = evaluate_esr(model, val_loader, dev, ecfg.preemph)
    val_shuf = evaluate_esr(model, val_loader, dev, ecfg.preemph, shuffle_devices=True)
    train_hours = (time.time() - t0) / 3600.0
    print(f"[cond]  val_esr={val_true:.5f}  val_esr_shuffled={val_shuf:.5f} "
          f"(shuffle should be worse){' OK' if val_shuf > val_true else ' — weak conditioning'}")

    _save_config_copy(run_dir, ecfg)
    save_embedding(run_dir / "embedding.pt", model, ids, manifest_sig, ecfg.embedding_dim, name,
                   holdout_ids=holdout_ids)
    metrics = {
        "name": name, "arch": arch["arch"], "params": n_params,
        "receptive_field_samples": R, "receptive_field_ms": round(rf_ms, 3),
        "embedding_dim": ecfg.embedding_dim, "channels": arch["channels"],
        "blocks_x_layers": arch["blocks_x_layers"], "n_devices": len(ids),
        "single_device": ecfg.single_device, "best_val_esr": best_val,
        "val_esr": val_true, "val_esr_shuffled": val_shuf,
        "epochs_run": epoch + 1, "train_hours": round(train_hours, 4),
        "manifest_sha256": manifest_sig,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[done]  best_val_esr={best_val:.5f} train_hours={train_hours:.2f} -> {run_dir}")
    return metrics


def save_embedding(path: Path, model, device_ids, manifest_sig, embedding_dim, name, *,
                   holdout_ids=()) -> None:
    """Save the embedding table + provenance separately (Phase 5 extends it, spec §5)."""
    torch.save({
        "embedding": model.embedding.weight.detach().cpu(),
        "device_ids": list(device_ids), "holdout_ids": list(holdout_ids),
        "manifest_sha256": manifest_sig,
        "embedding_dim": int(embedding_dim), "name": name,
    }, path)


def _save_config_copy(run_dir: Path, ecfg) -> None:
    import yaml
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"emulate": asdict(ecfg)}, sort_keys=False), encoding="utf-8")


# Shape-defining knobs: frozen for the life of a run because the checkpoint's
# weights depend on them. Everything else in EmulateConfig is training dynamics
# and may change on resume. Device count / receptive field are checked separately.
_STRUCTURAL_KEYS = ("arch", "blocks", "layers_per_block", "channels", "kernel_size",
                    "dilation_growth", "wn_channels", "embedding_dim", "single_device")


def _check_resume_compatible(ck: dict, ecfg, receptive_field: int, n_devices: int) -> None:
    """Fail fast (before the strict state_dict load) if the config changed a
    structural knob that would make the checkpoint weights incompatible, turning a
    cryptic ``load_state_dict`` shape error into a clear message. Training dynamics
    (lr, weight_decay, plateau_*, epochs, losses, batch_size) are free to change."""
    saved = ck.get("emulate_cfg", {})
    cur = asdict(ecfg)
    changed = [f"{k}: {saved[k]!r} -> {cur.get(k)!r}"
               for k in _STRUCTURAL_KEYS if k in saved and saved[k] != cur.get(k)]
    if ck.get("receptive_field") not in (None, receptive_field):
        changed.append(f"receptive_field: {ck['receptive_field']} -> {receptive_field}")
    if ck.get("device_ids") is not None and len(ck["device_ids"]) != n_devices:
        changed.append(f"device_count: {len(ck['device_ids'])} -> {n_devices}")
    if changed:
        raise RuntimeError(
            "cannot --resume: the config changed structural knobs that define the "
            "checkpoint's shape (" + "; ".join(changed) + "). Start a fresh run under "
            "a new config name, or revert these fields; only training dynamics (lr, "
            "weight_decay, plateau_*, epochs, losses, batch_size) may change on resume.")


def _append_config_history(run_dir: Path, ecfg, *, mode: str, start_epoch: int,
                           resumed_from, best_val: float, max_epochs: int,
                           applied_lr: float, applied_wd: float) -> None:
    """Append one JSON line per training session (start or resume) to
    ``config_history.jsonl`` — a never-overwritten audit trail of how each run and
    each resume was configured. Complements the end-of-run ``config.yaml`` copy
    (the as-finished snapshot)."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "start_epoch": start_epoch,
        "resumed_from_epoch": resumed_from,
        "best_val_esr_at_start": best_val if math.isfinite(best_val) else None,
        "max_epochs": max_epochs,
        "applied_lr": applied_lr,
        "applied_weight_decay": applied_wd,
        "emulate_cfg": asdict(ecfg),
    }
    with (run_dir / "config_history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
