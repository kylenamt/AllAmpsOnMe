"""Joint training: encoder and FiLM generator, end to end, from scratch.

Forked from :mod:`openamp.emulate.train` and keeps its contracts — same
:class:`~openamp.emulate.train.EmulationLoss` (pre-emphasized ESR + MRSTFT), same
Adam + reduce-on-plateau, same NaN guard, same run-directory layout — so a joint
run's ``val_esr`` is the same number the table-conditioned runs report and the two
are directly comparable.

What is different, and why:

- **The embedding comes from the encoder**, so the generator is driven through
  ``forward_emb``. Its own ``nn.Embedding`` table is frozen and unused; nothing
  should train a lookup table alongside the thing meant to replace it.
- **Two parameter groups.** The encoder gets its own learning rate
  (``joint.enc_lr``). Guidelines §9 lists LR imbalance as a failure mode, and a
  dead encoder is invisible in ESR alone.
- **A collapse tripwire runs every epoch.** ``val_esr_shuffled`` (embeddings
  permuted within the batch) and ``emb_spread`` (how much ``e`` actually varies
  across amps) are logged per epoch, not just at the end. With no contrastive
  term the only pressure on ``e`` is reconstruction, so the cheap failure is an
  encoder that emits a near-constant vector while ESR still looks respectable —
  the model having quietly learned one average amp. **A good val_esr means
  nothing unless the shuffled number is much worse and emb_spread is not ~0.**
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from openamp.core.config import Config
from openamp.emulate.dataset import manifest_signature
from openamp.emulate.tcn import count_parameters
from openamp.emulate.train import (GRAD_CLIP_NORM, EmulationLoss, _device_index,
                                   _make_loader, pre_emphasis)
from openamp.joint_model.dataset import JointDataset
from openamp.joint_model.model import build_joint, load_joint

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


@torch.no_grad()
def evaluate_esr(model, loader, device, coeff: float, *, n_batches: int | None = None,
                 mode: str = "true", seed: int = 0) -> dict:
    """Energy-pooled pre-emphasized ESR over the loader, plus embedding spread.

    Pooled as one ratio of sums exactly like
    :func:`openamp.emulate.train.evaluate_esr`, so the numbers are comparable
    across the two trainers.

    ``emb_spread`` is the mean over dimensions of the across-batch standard
    deviation of ``e`` — the direct readout of whether the encoder distinguishes
    amps at all. It collapses to ~0 long before ESR shows anything is wrong.

    The shuffle is drawn from a seeded generator so the number is comparable from
    epoch to epoch rather than re-rolled each time.
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
        pe_o, pe_t = pre_emphasis(out, coeff), pre_emphasis(target, coeff)
        num += float(torch.sum((pe_o - pe_t) ** 2).double())
        den += float(torch.sum(pe_t ** 2).double())
    if was_training:
        model.train()
    return {"esr": num / (den + 1e-8) if den > 0.0 else float("nan"),
            "emb_spread": float(np.mean(spreads)) if spreads else float("nan")}


def _build_datasets(cfg: Config, ecfg, jcfg, id_to_idx, R: int):
    """Train/val :class:`JointDataset` pair for a run."""
    clip = int(round(ecfg.clip_seconds * cfg.sample_rate))
    ref = int(round(jcfg.ref_seconds * cfg.sample_rate))
    gap = int(round(jcfg.ref_min_gap_seconds * cfg.sample_rate))
    common = dict(receptive_field=R, id_to_idx=id_to_idx, clip_samples=clip,
                  ref_samples=ref, ref_different_file=jcfg.ref_different_file,
                  ref_min_gap=gap, sources=tuple(jcfg.sources),
                  ref_mode=ref_mode_for(jcfg))
    train_ds = JointDataset(cfg, "train", pairs_per_epoch=ecfg.pairs_per_epoch,
                            seed=cfg.seed, **common)
    val_ds = JointDataset(cfg, "val", pairs_per_epoch=ecfg.val_pairs,
                          seed=cfg.seed + 999, **common)
    return train_ds, val_ds, clip, ref


def ref_mode_for(jcfg) -> str:
    """Which reference a conditioning source needs. One place, so train and eval agree."""
    return "device" if getattr(jcfg, "enc_kind", "wavenet") == "fingerprint" else "audio"


def _optimizer(model, ecfg, jcfg):
    """Two groups: generator (minus its frozen table) and encoder.

    Under ``enc_kind: fingerprint`` the encoder's fingerprint matrix is a *buffer*,
    so ``model.encoder.parameters()`` is exactly the adapter and the codec cannot be
    trained even by accident.
    """
    gen_params = [p for n, p in model.generator.named_parameters()
                  if p.requires_grad and not n.startswith("embedding.")]
    enc_wd = getattr(jcfg, "enc_weight_decay", -1.0)
    enc_wd = ecfg.weight_decay if enc_wd < 0 else float(enc_wd)
    return torch.optim.Adam(
        [{"params": gen_params, "lr": ecfg.lr, "name": "generator",
          "weight_decay": ecfg.weight_decay},
         {"params": list(model.encoder.parameters()), "lr": jcfg.enc_lr,
          "name": "encoder", "weight_decay": enc_wd}],
        weight_decay=ecfg.weight_decay)


# --- Sanity ladder -------------------------------------------------------------
def overfit_one_batch(cfg: Config, *, name: str = "overfit", device: str = "cuda",
                      steps: int = 400, limit_devices: int = 8,
                      batch_size: int | None = None) -> float:
    """Sanity #1: one fixed batch should drive ESR to ~0 through the encoder.

    Proves the gradient actually reaches the encoder. A *collapse* here is not
    conclusive — one fixed batch makes collapse an absorbing state — but a
    failure to fit is.
    """
    ecfg, jcfg = cfg.emulate, cfg.joint
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    batch = int(batch_size or ecfg.batch_size)

    ids, id_to_idx, _ = _device_index(cfg, ecfg, limit_devices)
    model = build_joint(ecfg, jcfg, len(ids)).to(dev)
    R = model.receptive_field
    train_ds, _, clip, ref_n = _build_datasets(cfg, ecfg, jcfg, id_to_idx, R)
    train_ds.pairs_per_epoch = batch
    loader = _make_loader(train_ds, batch, 0, shuffle=False, seed=cfg.seed, drop_last=False)
    b = next(iter(loader))
    inp = b["input"].to(dev)
    target = b["target"].to(dev).unsqueeze(1)
    ref = b["ref"].to(dev)

    lossfn = EmulationLoss(ecfg.preemph, ecfg.stft_weight).to(dev)
    opt = _optimizer(model, ecfg, jcfg)
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
def train(cfg: Config, *, name: str = "default", device: str = "cuda",
          epochs: int | None = None, resume: bool = False,
          limit_devices: int | None = None) -> dict:
    """Train encoder + generator jointly to a val-ESR plateau; returns a summary."""
    ecfg, jcfg = cfg.emulate, cfg.joint
    max_epochs = int(epochs if epochs is not None else ecfg.epochs)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    ids, id_to_idx, holdout_ids = _device_index(cfg, ecfg, limit_devices)
    if not ids:
        raise RuntimeError("No render-ok devices. Run `openamp render` first.")
    manifest_sig = manifest_signature(cfg)
    # Fingerprint provenance: the npz can be regenerated out from under a finished
    # run, and nothing about the checkpoint's shape would notice.
    fp_sha, fp_meta = "", {}
    if ref_mode_for(jcfg) == "device":
        from openamp.joint_model.fingerprint_encoder import FingerprintEncoder
        fp_sha = FingerprintEncoder.fingerprint_sha256(jcfg.fp_path)
        fp_meta = FingerprintEncoder.load_pooled(jcfg.fp_path)[3]
    model = build_joint(ecfg, jcfg, len(ids)).to(dev)
    R = model.receptive_field
    n_enc, n_gen = count_parameters(model.encoder), count_parameters(model.generator)
    rf_ms = 1000.0 * R / cfg.sample_rate
    print(f"[model] name={name} arch={ecfg.arch} embed={model.embedding_dim} "
          f"encoder={n_enc:,} generator={n_gen:,} params "
          f"devices={len(ids)} holdout={len(holdout_ids)}")
    cap = model.encoder.capacity_report()
    src = (f"fp={jcfg.fp_path} preprocess={cap.get('preprocess')} "
           f"n_fingerprints={cap.get('n_fingerprints')}"
           if cap["head"] == "fingerprint" else f"enc_channels={jcfg.enc_channels}")
    print(f"[model] head={cap['head']} {src} "
          f"pooled={cap['pooled_dim']}-d -> De={cap['embedding_dim']} "
          f"(ceiling {cap['ceiling']}) enc_rf={model.encoder.receptive_field} "
          f"normalize={jcfg.enc_normalize}")
    if cap["pooled_dim"] < cap["embedding_dim"]:
        print(f"[warn]  pooled width {cap['pooled_dim']} is below embedding_dim "
              f"{cap['embedding_dim']} — pooling, not the projection, is the bottleneck")
    print(f"[model] generator receptive_field={R} samples ({rf_ms:.1f} ms)")

    # --- Data -------------------------------------------------------------------
    train_ds, val_ds, clip, ref_n = _build_datasets(cfg, ecfg, jcfg, id_to_idx, R)
    val_loader = _make_loader(val_ds, ecfg.batch_size, ecfg.num_workers,
                              shuffle=False, seed=cfg.seed + 999, drop_last=False)
    if train_ds.ref_mode == "device":
        ref_desc = "device-id lookup of a precomputed fingerprint"
    elif train_ds.same_file_mode:
        ref_desc = "same-file (gap %d)" % train_ds.ref_min_gap
    else:
        ref_desc = f"different-file (of {len(train_ds.ref_files)})"
    print(f"[data]  train_devices={len(train_ds.device_ids)} "
          f"val_devices={len(val_ds.device_ids)} pairs/epoch={ecfg.pairs_per_epoch}")
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

    # Zero-shot probe. The number this experiment exists to produce, measured every
    # epoch instead of once at the end: proto's seen 0.439 vs holdout 0.805 gap was
    # invisible until the run finished.
    holdout_loader = None
    if holdout_ids:
        h_idx = {int(d): i for i, d in enumerate(sorted(holdout_ids))}
        holdout_ds = JointDataset(
            cfg, "val", receptive_field=R, id_to_idx=h_idx, clip_samples=clip,
            ref_samples=ref_n, ref_different_file=jcfg.ref_different_file,
            ref_min_gap=int(round(jcfg.ref_min_gap_seconds * cfg.sample_rate)),
            sources=tuple(jcfg.sources), ref_mode=ref_mode_for(jcfg),
            pairs_per_epoch=ecfg.val_pairs, seed=cfg.seed + 4242)
        holdout_loader = _make_loader(holdout_ds, ecfg.batch_size, ecfg.num_workers,
                                      shuffle=False, seed=cfg.seed + 4242,
                                      drop_last=False)
        print(f"[data]  holdout probe: {len(holdout_ds.device_ids)} unseen devices")

    # --- Optim / loss / bookkeeping --------------------------------------------
    lossfn = EmulationLoss(ecfg.preemph, ecfg.stft_weight).to(dev)
    opt = _optimizer(model, ecfg, jcfg)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=ecfg.plateau_factor, patience=ecfg.plateau_patience)
    use_amp = bool(ecfg.amp) and dev.type == "cuda"
    if ecfg.amp_dtype not in ("fp16", "bf16"):
        raise ValueError(f"emulate.amp_dtype must be 'fp16' or 'bf16', got {ecfg.amp_dtype!r}")
    amp_dtype = torch.bfloat16 if ecfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    run_dir = cfg.joint_run_dir(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, last_path = run_dir / "checkpoint.pt", run_dir / "last.pt"
    log_path = run_dir / "train_log.csv"

    start_epoch, best_val, since_improved = 0, float("inf"), 0
    resumed_from = None
    if resume and last_path.is_file():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        _check_resume_compatible(ck, ecfg, jcfg, R, len(ids))
        model.encoder.load_state_dict(ck["encoder"])
        model.generator.load_state_dict(ck["generator"])
        opt.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        resumed_from = int(ck["epoch"])
        start_epoch = resumed_from + 1
        best_val = float(ck.get("best_val_esr", float("inf")))
        # Config is authoritative on resume (same rule as emulate/train.py): the
        # Adam moment buffers are what resuming preserves, not the lr value.
        for g in opt.param_groups:
            g["lr"] = jcfg.enc_lr if g.get("name") == "encoder" else ecfg.lr
            g["weight_decay"] = ecfg.weight_decay
        print(f"[resume] from epoch {start_epoch}, best_val_esr={best_val:.5f}; "
              f"lr_gen={ecfg.lr:.2e} lr_enc={jcfg.enc_lr:.2e} (from config)")
    else:
        with log_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(LOG_COLUMNS)
    _append_config_history(run_dir, ecfg, jcfg,
                           mode="resume" if resumed_from is not None else "start",
                           start_epoch=start_epoch, resumed_from=resumed_from,
                           best_val=best_val, max_epochs=max_epochs)

    def save_ckpt(path: Path, epoch: int, val_esr: float) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({
            "encoder": model.encoder.state_dict(),
            "generator": model.generator.state_dict(),
            "optim": opt.state_dict(), "scaler": scaler.state_dict(),
            "emulate_cfg": asdict(ecfg), "joint_cfg": asdict(jcfg),
            "device_ids": ids, "id_to_idx": id_to_idx, "holdout_ids": holdout_ids,
            "manifest_sha256": manifest_sig, "receptive_field": R,
            "fingerprint_sha256": fp_sha, "fingerprint_meta": fp_meta,
            "ref_samples": ref_n, "clip_samples": clip,
            "encoder_params": n_enc, "generator_params": n_gen,
            "sample_rate": cfg.sample_rate, "seed": cfg.seed, "name": name,
            "epoch": epoch, "val_esr": val_esr, "best_val_esr": best_val,
        }, tmp)
        tmp.replace(path)                       # atomic: a crash never truncates a ckpt

    # --- Loop -------------------------------------------------------------------
    t0 = time.time()
    step = start_epoch * (ecfg.pairs_per_epoch // ecfg.batch_size)
    epoch = start_epoch - 1
    # The per-epoch tripwire runs on a slice of val, not all of it: it is watched
    # for its trend, and the end-of-run number is measured over the whole set.
    probe_batches = max(1, min(25, ecfg.val_pairs // max(ecfg.batch_size, 1)))
    for epoch in range(start_epoch, max_epochs):
        train_ds.seed = cfg.seed + epoch        # fresh window positions each epoch
        loader = _make_loader(train_ds, ecfg.batch_size, ecfg.num_workers,
                              shuffle=True, seed=cfg.seed + epoch, drop_last=True)
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
                out = model(inp, ref)[..., R:]
            loss, parts = lossfn(out.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()

            ep_loss += loss.item(); ep_esr += parts["esr"]; ep_stft += parts["stft"]
            n_steps += 1
            step += 1
            if step % ecfg.log_every == 0:
                sps = n_steps * ecfg.batch_size / (time.time() - ep_t0 + 1e-9)
                print(f"  e{epoch:02d} step {step:>7d} loss {loss.item():.5f} "
                      f"esr {parts['esr']:.5f} stft {parts['stft']:.4f} {sps:5.1f} smp/s",
                      flush=True)      # stdout is block-buffered under nohup/redirects

        tr_loss = ep_loss / max(n_steps, 1)
        tr_esr = ep_esr / max(n_steps, 1)
        tr_stft = ep_stft / max(n_steps, 1)
        val = evaluate_esr(model, val_loader, dev, ecfg.preemph, seed=cfg.seed)
        val_esr, spread = val["esr"], val["emb_spread"]
        shuf = evaluate_esr(model, val_loader, dev, ecfg.preemph, mode="shuffled",
                            n_batches=probe_batches, seed=cfg.seed)["esr"]
        hold = (evaluate_esr(model, holdout_loader, dev, ecfg.preemph,
                             n_batches=probe_batches, seed=cfg.seed)["esr"]
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
            _log_row(log_path, epoch, step, tr_loss, tr_esr, tr_stft, val_esr, shuf,
                     spread, opt, t0)
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
        _log_row(log_path, epoch, step, tr_loss, tr_esr, tr_stft, val_esr, shuf,
                 hold, spread, opt, t0)
        lrs = _lrs(opt)
        # ratio > 1 means a wrong reference hurts, i.e. the embedding is being used
        ratio = shuf / val_esr if val_esr > 0 else float("nan")
        print(f"[epoch {epoch:02d}] train_esr={tr_esr:.5f} val_esr={val_esr:.5f} "
              f"best={best_val:.5f} shuffled={shuf:.5f} (x{ratio:.1f}) "
              f"holdout={hold:.5f} "
              f"emb_spread={spread:.4f} lr={lrs['generator']:.1e}/{lrs['encoder']:.1e}"
              f"{'  *saved*' if improved else ''}", flush=True)
        if since_improved >= ecfg.early_stop_patience:
            print(f"[stop] no val improvement for {since_improved} epochs")
            break

    # --- Reload best; full diagnostics; final artifacts -------------------------
    if ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.encoder.load_state_dict(ck["encoder"])
        model.generator.load_state_dict(ck["generator"])
    final = {m: evaluate_esr(model, val_loader, dev, ecfg.preemph, mode=m, seed=cfg.seed)
             for m in EVAL_MODES}
    val_true, val_shuf = final["true"]["esr"], final["shuffled"]["esr"]
    val_zero, spread = final["zero"]["esr"], final["true"]["emb_spread"]
    train_hours = (time.time() - t0) / 3600.0
    verdict = ("OK" if val_shuf > 2 * val_true and spread > 1e-3 else
               "WEAK — the encoder may have collapsed; see joint-eval")
    print(f"[cond]  val_esr={val_true:.5f} shuffled={val_shuf:.5f} zero={val_zero:.5f} "
          f"emb_spread={spread:.4f} -> {verdict}")

    _save_config_copy(run_dir, ecfg, jcfg)
    metrics = {
        "name": name, "arch": ecfg.arch, "encoder_params": n_enc,
        "generator_params": n_gen, "params": n_enc + n_gen,
        "receptive_field_samples": R, "receptive_field_ms": round(rf_ms, 3),
        "embedding_dim": model.embedding_dim, "enc_channels": jcfg.enc_channels,
        "enc_head": cap["head"], "pooled_dim": cap["pooled_dim"],
        "channels": ecfg.wn_channels,
        "n_devices": len(ids), "n_holdout": len(holdout_ids),
        "ref_samples": ref_n, "clip_samples": clip,
        "ref_different_file": not train_ds.same_file_mode,
        "best_val_esr": best_val, "val_esr": val_true,
        "val_esr_shuffled": val_shuf, "val_esr_zero": val_zero,
        "emb_spread": spread, "epochs_run": epoch + 1,
        "train_hours": round(train_hours, 4), "manifest_sha256": manifest_sig,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[done]  best_val_esr={best_val:.5f} train_hours={train_hours:.2f} -> {run_dir}")
    return metrics


# --- Bookkeeping ---------------------------------------------------------------
def _lrs(opt) -> dict:
    return {g.get("name", str(i)): g["lr"] for i, g in enumerate(opt.param_groups)}


def _log_row(log_path: Path, epoch, step, tr_loss, tr_esr, tr_stft, val_esr, shuf,
             hold, spread, opt, t0) -> None:
    lrs = _lrs(opt)
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([
            epoch, step, f"{tr_loss:.6f}", f"{tr_esr:.6f}", f"{tr_stft:.6f}",
            f"{val_esr:.6f}", f"{shuf:.6f}", f"{hold:.6f}", f"{spread:.6f}",
            f"{lrs['generator']:.2e}", f"{lrs['encoder']:.2e}", f"{time.time() - t0:.1f}"])


def _save_config_copy(run_dir: Path, ecfg, jcfg) -> None:
    import yaml
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"emulate": asdict(ecfg), "joint": asdict(jcfg)}, sort_keys=False),
        encoding="utf-8")


# Shape-defining knobs, frozen for the life of a run. Beyond emulate's list,
# every encoder-shape knob, plus enc_normalize: it changes no shape, so a swap
# would load cleanly and quietly feed the generator a differently-scaled vector.
_STRUCTURAL_EMULATE = ("arch", "wn_channels", "wn_activation", "cond_hidden",
                       "cond_activation", "delta_rank", "embedding_dim", "single_device")
_STRUCTURAL_JOINT = ("enc_channels", "enc_activation", "enc_attn_hidden",
                     "enc_proj_hidden", "enc_normalize", "enc_head", "enc_taps",
                     "enc_tap_stride", "enc_expand_dim", "enc_n_heads",
                     "enc_expand_norm", "enc_norm_groups",
                     # fp_path and fp_preprocess change no shape, which is exactly the
                     # enc_normalize argument: a swap would load cleanly and quietly
                     # feed the generator a different vector for the same amp.
                     "enc_kind", "fp_path", "fp_preprocess", "fp_layernorm")

# A checkpoint written before a knob existed still ran *some* behaviour. Record it,
# or the `k in saved` test below silently waves the change through — and for
# enc_head that is precisely the accident this guard exists to stop, since the two
# heads have disjoint parameters and the load would fail far from the cause.
_LEGACY_JOINT_DEFAULTS = {"enc_head": "stats", "enc_taps": [6, 13, 22],
                          "enc_tap_stride": 8, "enc_expand_dim": 256,
                          "enc_n_heads": 2, "enc_expand_norm": "batch",
                          "enc_norm_groups": 32,
                          "enc_kind": "wavenet", "fp_path": "",
                          "fp_preprocess": "dry_l2", "fp_layernorm": False}


def _same(a, b) -> bool:
    """Compare config values that may be list-vs-tuple after a yaml round-trip."""
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a or []) == list(b or [])
    return a == b


def _check_resume_compatible(ck: dict, ecfg, jcfg, receptive_field: int,
                             n_devices: int) -> None:
    """Fail fast if a config edit made the checkpoint's weights incompatible."""
    changed = []
    for keys, saved, cur, prefix, legacy in (
            (_STRUCTURAL_EMULATE, ck.get("emulate_cfg", {}), asdict(ecfg), "emulate", {}),
            (_STRUCTURAL_JOINT, ck.get("joint_cfg", {}), asdict(jcfg), "joint",
             _LEGACY_JOINT_DEFAULTS)):
        for k in keys:
            if k in saved:
                was = saved[k]
            elif k in legacy:
                was = legacy[k]
            else:
                continue
            if not _same(was, cur.get(k)):
                changed.append(f"{prefix}.{k}: {was!r} -> {cur.get(k)!r}")
    if ck.get("receptive_field") not in (None, receptive_field):
        changed.append(f"receptive_field: {ck['receptive_field']} -> {receptive_field}")
    if ck.get("device_ids") is not None and len(ck["device_ids"]) != n_devices:
        changed.append(f"device_count: {len(ck['device_ids'])} -> {n_devices}")
    if changed:
        raise RuntimeError(
            "cannot --resume: the config changed structural knobs that define the "
            "checkpoint's shape (" + "; ".join(changed) + "). Start a fresh run under "
            "a new config name, or revert these fields; only training dynamics (lr, "
            "enc_lr, plateau_*, epochs, losses, batch_size) may change on resume.")


def _append_config_history(run_dir: Path, ecfg, jcfg, *, mode: str, start_epoch: int,
                           resumed_from, best_val: float, max_epochs: int) -> None:
    """One JSON line per training session — a never-overwritten audit trail."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode, "start_epoch": start_epoch, "resumed_from_epoch": resumed_from,
        "best_val_esr_at_start": best_val if math.isfinite(best_val) else None,
        "max_epochs": max_epochs,
        "emulate_cfg": asdict(ecfg), "joint_cfg": asdict(jcfg),
    }
    with (run_dir / "config_history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


__all__ = ["train", "overfit_one_batch", "evaluate_esr", "load_joint", "LOG_COLUMNS",
           "EVAL_MODES"]
