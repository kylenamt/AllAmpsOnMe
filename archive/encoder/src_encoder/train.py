"""Contrastive training for the tone encoder (see :mod:`openamp.encoder.model`).

The objective, and why each term is there:

- **SupCon on ``capture_id``, one per branch.** Labels are known — this is not
  self-supervised — so every other segment of the same capture in the batch is a
  positive, not just one augmented view. The batch is built P captures x K
  segments by :class:`~openamp.encoder.dataset.ToneSegmentDataset` so those
  positives exist by construction.
- **Cross-covariance between the branches** (Barlow-Twins style, but across the
  *pair* rather than across two views). Without it the two branches are free to
  learn the same thing twice, and the "static vs dynamic" split becomes
  decorative — which is exactly what the branch-redundancy diagnostic checks.
- **Two probes**, at low weight. SupCon alone would happily find *any* axis that
  separates captures; the probes assign meaning to each branch: amp identity
  (``tone_id``) to the static one, dynamics to the dynamic one.

**Augmentation is deliberately almost empty.** Additive noise on the wet channel
(a recording artifact, not a tone) and time masking are the only two. EQ, gain,
pitch shift, time stretch and reverb are *not* implemented and must not be
added: they **are** tone, so training the encoder to be invariant to them is
training it to ignore its own target. Time jitter is already free — the random
window draw shifts both channels identically by construction.

Everything is config-driven from :class:`~openamp.core.config.EncoderConfig`.
Runs write to ``results/encoder/<name>/``. The log columns are contrastive-loss
names on purpose: ``docs/decisions.md`` §6 exists because three unrelated
quantities called "ESR" got cross-compared once.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from openamp.core.config import Config, EncoderConfig
from openamp.emulate.dataset import (build_device_index, load_or_create_holdout,
                                     manifest_signature)
from openamp.encoder.dataset import (ToneSegmentDataset, device_labels,
                                     region_level_db, time_split_regions)
from openamp.encoder.model import ToneEncoder, count_parameters

GRAD_CLIP_NORM = 20.0            # same spike safety net as the emulator trainer

# Knobs that --resume must not cross. The first group would not load into the saved
# weights at all; the second would load fine and silently redefine `val_supcon`
# mid-run, so the best-val bookkeeping would be comparing two different numbers.
_STRUCTURAL_KEYS = ("stem_channels", "static_dilations", "dynamic_dilations",
                    "head_dim", "proj_hidden", "proj_dim", "norm_groups",
                    "segment_samples",
                    "time_val_frac", "time_split_mode", "time_split_blocks",
                    "usable_region")

LOG_COLUMNS = ["epoch", "step", "train_loss", "train_supcon_s", "train_supcon_d",
               "train_xcov", "train_amp_ce", "train_dyn", "val_supcon",
               "val_supcon_shuffled", "val_amp_acc", "val_crest_mse", "lr", "elapsed_s"]


# --- Losses --------------------------------------------------------------------
def supcon(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.1,
           eps: float = 1e-8) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., ``L_out``) over L2-normalized ``z``.

    ``z`` is ``[B, D]`` (already unit-norm — the projection heads normalize), and
    ``labels`` ``[B]`` is the capture id. Every pair sharing a label is a positive;
    the self-pair is masked out. Items with no positive in the batch contribute
    nothing (they would otherwise add a constant with no gradient).
    """
    if z.dim() != 2:
        raise ValueError(f"expected [B, D], got {tuple(z.shape)}")
    b = z.shape[0]
    sim = (z @ z.t()) / float(tau)
    eye = torch.eye(b, dtype=torch.bool, device=z.device)
    # Row-wise max subtracted for numerical stability; detached, so it is a shift,
    # not a gradient path.
    sim = sim - sim.masked_fill(eye, float("-inf")).max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim).masked_fill(eye, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

    pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye
    n_pos = pos.sum(dim=1)
    has_pos = n_pos > 0
    if not bool(has_pos.any()):
        return z.sum() * 0.0
    mean_log_prob = (pos.float() * log_prob).sum(dim=1) / n_pos.clamp(min=1)
    return -mean_log_prob[has_pos].mean()


def cross_covariance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Mean squared cross-correlation between two branch vectors over the batch.

    Each is standardized per feature, then the full ``[D, D]`` cross-correlation
    matrix is driven toward zero. Unlike Barlow Twins this penalizes the diagonal
    too: the branches are different *views of the same thing*, not two views of one
    input, so there is no feature correspondence worth preserving — any shared
    direction is the dynamic branch re-deriving static character.
    """
    a = (a - a.mean(dim=0)) / (a.std(dim=0) + eps)
    b = (b - b.mean(dim=0)) / (b.std(dim=0) + eps)
    c = (a.t() @ b) / float(a.shape[0])
    return c.pow(2).mean()


class Probes(nn.Module):
    """Linear read-outs that give each branch a meaning. Trained jointly, low weight."""

    def __init__(self, head_dim: int, n_tones: int, n_gain: int):
        super().__init__()
        self.amp = nn.Linear(head_dim, max(1, int(n_tones)))     # static  -> tone_id
        self.crest = nn.Linear(head_dim, 1)                      # dynamic -> crest delta
        self.gain = nn.Linear(head_dim, max(1, int(n_gain)))     # dynamic -> gain bucket


class EncoderLoss(nn.Module):
    """SupCon (x2) + cross-covariance + probes, weighted per :class:`EncoderConfig`."""

    def __init__(self, ncfg: EncoderConfig, n_tones: int, n_gain: int):
        super().__init__()
        self.cfg = ncfg
        self.probes = Probes(ncfg.head_dim, n_tones, n_gain)
        self.ce = nn.CrossEntropyLoss()
        self.ce_ignore = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse = nn.MSELoss()

    def forward(self, static, dynamic, proj_s, proj_d, batch) -> tuple:
        c = self.cfg
        dev = static.device
        capture = batch["device_idx"].to(dev)
        parts = {}

        ls = supcon(proj_s, capture, c.supcon_tau)
        ld = supcon(proj_d, capture, c.supcon_tau)
        lx = cross_covariance(static, dynamic)
        loss = c.supcon_static_w * ls + c.supcon_dynamic_w * ld + c.xcov_w * lx
        parts.update(supcon_s=float(ls.detach()), supcon_d=float(ld.detach()),
                     xcov=float(lx.detach()))

        amp_ce = torch.zeros((), device=dev)
        if c.amp_probe_w > 0:
            amp_ce = self.ce(self.probes.amp(static), batch["tone_idx"].to(dev))
            loss = loss + c.amp_probe_w * amp_ce
        parts["amp_ce"] = float(amp_ce.detach())

        dyn = torch.zeros((), device=dev)
        if c.dyn_probe == "crest" and c.dyn_probe_w > 0:
            dyn = self.mse(self.probes.crest(dynamic).squeeze(-1),
                           batch["crest"].to(dev).float())
        elif c.dyn_probe == "gain_bucket" and c.dyn_probe_w > 0:
            dyn = self.ce_ignore(self.probes.gain(dynamic), batch["gain_idx"].to(dev))
        if c.dyn_probe_w > 0 and c.dyn_probe != "none":
            loss = loss + c.dyn_probe_w * dyn
        parts["dyn"] = float(dyn.detach())

        # The second dynamics probe stays available as an independent term, so the
        # crest-vs-bucket ablation is a config edit rather than a code edit.
        if c.gain_probe_w > 0 and c.dyn_probe != "gain_bucket":
            gain_ce = self.ce_ignore(self.probes.gain(dynamic), batch["gain_idx"].to(dev))
            loss = loss + c.gain_probe_w * gain_ce
            parts["gain_ce"] = float(gain_ce.detach())

        parts["loss"] = float(loss.detach())
        return loss, parts


# --- Data plumbing -------------------------------------------------------------
def build_datasets(cfg: Config, ncfg: EncoderConfig, *, limit_devices: int | None = None):
    """``(train_ds, val_ds, meta)`` — one device set, split in time.

    The **device** axis is already used by ``emulate_holdout.txt`` (unseen amps,
    held back for the retrieval diagnostic in ``encode-eval``), so train/val here
    is a *time* split of the same captures: ``time_val_frac`` of every file is val.
    That is the split that answers "does this generalize to audio it hasn't seen",
    and the holdout answers "does it generalize to amps it hasn't seen". Both are
    needed; neither substitutes for the other.

    Which *part* of the file val gets is :func:`time_split_regions`' job, and it is
    not a detail: the capture signal's level falls across its length, so a tail
    split hands val a 14 dB quieter operating point and quietly changes what the
    number means. Default is the interleaved split.
    """
    holdout = load_or_create_holdout(cfg, ncfg.holdout_frac, cfg.seed)
    ids, _ = build_device_index(cfg, exclude=holdout)
    if limit_devices:
        ids = ids[:int(limit_devices)]
    if len(ids) < ncfg.devices_per_batch:
        raise RuntimeError(f"encoder.devices_per_batch is {ncfg.devices_per_batch} but only "
                           f"{len(ids)} devices are available — lower it or drop "
                           f"--limit-devices")
    id_to_idx = {d: i for i, d in enumerate(ids)}
    labels = device_labels(cfg, ids)
    split = "train"
    train_regions, val_regions = time_split_regions(
        ncfg.time_val_frac, mode=ncfg.time_split_mode, blocks=ncfg.time_split_blocks,
        within=tuple(ncfg.usable_region))
    common = dict(id_to_idx=id_to_idx, segment_samples=ncfg.segment_samples,
                  sources=ncfg.sources, devices_per_batch=ncfg.devices_per_batch,
                  segments_per_device=ncfg.segments_per_device, labels=labels,
                  undo_output_scale=ncfg.undo_output_scale)
    train_ds = ToneSegmentDataset(
        cfg, split, regions=train_regions, episodes=ncfg.episodes_per_epoch,
        seed=cfg.seed, augment=True, noise_p=ncfg.noise_p,
        noise_snr_db=tuple(ncfg.noise_snr_db), time_mask_p=ncfg.time_mask_p,
        time_mask_frac=ncfg.time_mask_frac, **common)
    # Val episodes cover every device at least once; fixed seed, never redrawn.
    val_eps = max(1, math.ceil(len(ids) / ncfg.devices_per_batch))
    val_ds = ToneSegmentDataset(
        cfg, split, regions=val_regions, episodes=val_eps,
        seed=cfg.seed + 99_991, augment=False, **common)
    meta = {"device_ids": ids, "id_to_idx": id_to_idx, "holdout_ids": holdout,
            "labels": labels, "n_tones": labels["n_tones"], "n_gain": labels["n_gain"],
            "train_regions": train_regions, "val_regions": val_regions}
    return train_ds, val_ds, meta


SPLIT_LEVEL_WARN_DB = 3.0        # train/val RMS gap above which the split is suspect


def _report_split(ncfg: EncoderConfig, meta: dict, train_ds, val_ds) -> None:
    """Print the realized time split and check that val is level-matched to train.

    Both halves are worth printing. ``time_val_frac`` is a *request* — the block
    grid rounds it — and the level gap is the thing that silently invalidates a
    split: an amp is a level-dependent nonlinearity, so validating on materially
    quieter audio measures a different operating point, not held-out audio. This
    is checked at run time rather than assumed because it depends on the corpus,
    and getting it wrong is invisible in the loss curve.
    """
    val_span = sum(b - a for a, b in meta["val_regions"])
    print(f"[encode] usable region={tuple(ncfg.usable_region)}  "
          f"time split={ncfg.time_split_mode} blocks={ncfg.time_split_blocks} "
          f"val={val_span:.1%} of each file over {len(meta['val_regions'])} region(s) "
          f"{[(round(a, 3), round(b, 3)) for a, b in meta['val_regions']]}", flush=True)
    try:
        tr_db, va_db = region_level_db(train_ds), region_level_db(val_ds)
    except Exception as e:                       # a level report must never kill a run
        print(f"[encode] split level check skipped: {e!r}", flush=True)
        return
    gap = abs(tr_db - va_db)
    print(f"[encode] split level: train {tr_db:.2f} dBFS  val {va_db:.2f} dBFS  "
          f"gap {gap:.2f} dB", flush=True)
    if gap > SPLIT_LEVEL_WARN_DB:
        print(f"[encode] WARNING val audio is {gap:.1f} dB from train — val SupCon and "
              f"the crest probe will read pessimistically, and not because the encoder "
              f"is weak. Adjust encoder.time_split_blocks.", flush=True)


def make_loader(ds, ncfg: EncoderConfig, *, workers: int | None = None) -> DataLoader:
    """Sequential loader at exactly ``P*K`` — shuffling would destroy the batch structure."""
    return DataLoader(ds, batch_size=ds.batch_size(), shuffle=False,
                      num_workers=ncfg.num_workers if workers is None else workers,
                      drop_last=True, pin_memory=True, persistent_workers=False)


def _resolve_device(name: str) -> str:
    return name if (name != "cuda" or torch.cuda.is_available()) else "cpu"


def _amp_dtype(ncfg: EncoderConfig):
    return torch.bfloat16 if str(ncfg.amp_dtype).lower() == "bf16" else torch.float16


# --- Evaluation ----------------------------------------------------------------
@torch.no_grad()
def evaluate(model, lossfn, loader, dev, ncfg: EncoderConfig) -> dict:
    """Val SupCon + probe read-outs, plus the label-shuffled control.

    ``val_supcon_shuffled`` permutes the capture labels within each batch. It is
    the encoder's analogue of the emulator's ``val_esr_shuffled``: if a real
    grouping scores no better than a random one, the embedding is not carrying
    capture identity and every other number here is noise.
    """
    was_training = (model.training, lossfn.training)
    model.eval()
    lossfn.eval()
    tot = {"supcon": 0.0, "shuffled": 0.0, "amp_correct": 0.0, "crest_se": 0.0, "n": 0.0}
    for batch in loader:
        x = batch["x"].to(dev, non_blocking=True)
        capture = batch["device_idx"].to(dev)
        _, s, d = model(x)
        ps, pd = model.project(s, d)
        b = x.shape[0]
        tot["supcon"] += float(supcon(ps, capture, ncfg.supcon_tau)) * b
        perm = torch.randperm(b, device=dev)
        tot["shuffled"] += float(supcon(ps, capture[perm], ncfg.supcon_tau)) * b
        pred = lossfn.probes.amp(s).argmax(dim=-1)
        tot["amp_correct"] += float((pred == batch["tone_idx"].to(dev)).sum())
        err = lossfn.probes.crest(d).squeeze(-1) - batch["crest"].to(dev).float()
        tot["crest_se"] += float(err.pow(2).sum())
        tot["n"] += b
    n = max(tot["n"], 1.0)
    model.train(was_training[0])            # restore, don't assume the caller was training
    lossfn.train(was_training[1])
    return {"val_supcon": tot["supcon"] / n, "val_supcon_shuffled": tot["shuffled"] / n,
            "val_amp_acc": tot["amp_correct"] / n, "val_crest_mse": tot["crest_se"] / n}


def supcon_floor(segments_per_device: int) -> float:
    """The best SupCon a *perfectly* separated batch can reach: ``log(K - 1)``.

    Not zero — and this trips people up. With positives collapsed onto one point
    and negatives at cosine 0, each row's softmax puts ``exp(1/tau)`` on each of
    its ``K-1`` positives and ``1`` on every negative, so ``-mean log prob`` bottoms
    out at ``log(K-1)``. Measured: K=4 gives 1.0986 = log 3. Read the ``--overfit``
    check against this number, never against 0.
    """
    return math.log(max(1, int(segments_per_device) - 1))


# --- Sanity #1: one batch to the SupCon floor ----------------------------------
def overfit_one_batch(cfg: Config, *, name: str = "default", device: str = "cuda",
                      steps: int = 200, limit_devices: int | None = 8) -> dict:
    """Drive one batch's SupCon down to :func:`supcon_floor`.

    If this does not descend, the loss, the P x K index mapping, or the model is
    wrong and nothing downstream is worth running.

    **A collapse here is suggestive, not conclusive.** One fixed batch at full LR
    is a harsher setting than training: contrastive collapse — every embedding on
    one point, SupCon pinned at ``log(P*K - 1)`` — is an absorbing state when the
    data never changes, while a real run gets pushed back out by the next batch.
    Measured 2026-08-12: a smoke batch collapsed on 4 of 5 torch seeds while the
    real 404-device run over the same regions descended 2.89 -> 2.64 -> 2.44 over
    three epochs. Hence the warmup below, matching ``train()``'s recipe rather
    than hitting a fresh model with ``lr`` on step 0. If this check collapses,
    re-run it on another seed before believing it.
    """
    ncfg = cfg.encoder
    dev = torch.device(_resolve_device(device))
    torch.manual_seed(cfg.seed)
    train_ds, _, meta = build_datasets(cfg, ncfg, limit_devices=limit_devices)
    # No augmentation for the memorization check: a fresh noise draw every step is a
    # moving target, so a non-zero floor would mean nothing.
    train_ds.augment = False
    loader = make_loader(train_ds, ncfg, workers=0)
    batch = next(iter(loader))
    batch = {k: v.to(dev) for k, v in batch.items()}

    model = ToneEncoder.from_config(ncfg).to(dev)
    lossfn = EncoderLoss(ncfg, meta["n_tones"], meta["n_gain"]).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(lossfn.parameters()),
                            lr=ncfg.lr, weight_decay=ncfg.weight_decay)
    floor = supcon_floor(ncfg.segments_per_device)
    print(f"[encode-overfit] params={count_parameters(model):,} batch={batch['x'].shape[0]} "
          f"devices={len(meta['device_ids'])} dev={dev} | SupCon floor = "
          f"log(K-1) = {floor:.4f}", flush=True)
    first = None
    warmup = max(1, int(steps) // 10)
    for step in range(int(steps)):
        for g in opt.param_groups:
            g["lr"] = ncfg.lr * min(1.0, (step + 1) / warmup)
        opt.zero_grad(set_to_none=True)
        _, s, d = model(batch["x"])
        ps, pd = model.project(s, d)
        loss, parts = lossfn(s, d, ps, pd, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(lossfn.parameters()), GRAD_CLIP_NORM)
        opt.step()
        if first is None:
            first = parts
        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {parts['loss']:.4f}  supcon_s {parts['supcon_s']:.4f}  "
                  f"supcon_d {parts['supcon_d']:.4f}  xcov {parts['xcov']:.4f}", flush=True)
    out = {"first": first, "last": parts, "steps": int(steps), "floor": floor}
    print(f"[encode-overfit] supcon_s {first['supcon_s']:.4f} -> {parts['supcon_s']:.4f}; "
          f"supcon_d {first['supcon_d']:.4f} -> {parts['supcon_d']:.4f}  "
          f"(floor {floor:.4f})", flush=True)
    return out


# --- The run -------------------------------------------------------------------
def _lr_scale(epoch: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine to zero (spec: AdamW, lr 1e-3 cosine)."""
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    p = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def _same(a, b) -> bool:
    """Compare config values that may be list-vs-tuple after a yaml round-trip."""
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a or []) == list(b or [])
    return a == b


# A checkpoint written before a knob existed still ran *some* behaviour. Record it,
# or `k in old` silently waves the change through — which for the split knobs is the
# exact accident the guard is there to stop.
_LEGACY_DEFAULTS = {"time_split_mode": "tail"}


def _check_resume_compatible(old: dict, new: EncoderConfig) -> None:
    bad = {}
    for k in _STRUCTURAL_KEYS:
        if k in old:
            was = old[k]
        elif k in _LEGACY_DEFAULTS:
            was = _LEGACY_DEFAULTS[k]
        else:
            continue
        if not _same(was, getattr(new, k)):
            bad[k] = (was, getattr(new, k))
    if bad:
        raise RuntimeError(
            "cannot --resume: structural knobs changed since the checkpoint — "
            + ", ".join(f"{k} {o!r} -> {n!r}" for k, (o, n) in bad.items()))


def train(cfg: Config, *, name: str = "default", device: str = "cuda",
          epochs: int | None = None, resume: bool = False,
          limit_devices: int | None = None) -> dict:
    """Train the tone encoder to a val-SupCon plateau; returns a summary dict."""
    ncfg = cfg.encoder
    dev = torch.device(_resolve_device(device))
    max_epochs = int(epochs if epochs is not None else ncfg.epochs)
    torch.manual_seed(cfg.seed)

    run_dir = Path(cfg.encoder_run_dir(name))
    run_dir.mkdir(parents=True, exist_ok=True)
    ck_path, last_path = run_dir / "checkpoint.pt", run_dir / "last.pt"
    log_path = run_dir / "encoder_log.csv"

    train_ds, val_ds, meta = build_datasets(cfg, ncfg, limit_devices=limit_devices)
    val_loader = make_loader(val_ds, ncfg)
    manifest_sig = manifest_signature(cfg)

    model = ToneEncoder.from_config(ncfg).to(dev)
    lossfn = EncoderLoss(ncfg, meta["n_tones"], meta["n_gain"]).to(dev)
    params = list(model.parameters()) + list(lossfn.parameters())
    opt = torch.optim.AdamW(params, lr=ncfg.lr, weight_decay=ncfg.weight_decay)
    use_amp = bool(ncfg.amp) and dev.type == "cuda"
    amp_dtype = _amp_dtype(ncfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    start_epoch, best_val, best_epoch = 0, float("inf"), -1
    if resume:
        if not last_path.is_file():
            raise RuntimeError(f"--resume: no {last_path}")
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        _check_resume_compatible(ck.get("encoder_cfg", {}), ncfg)
        model.load_state_dict(ck["model"])
        lossfn.load_state_dict(ck["loss"])
        opt.load_state_dict(ck["optim"])
        # The config is authoritative for training knobs — that is how a run is
        # hand-annealed — while Adam's moment buffers are preserved.
        for g in opt.param_groups:
            g["lr"], g["weight_decay"] = ncfg.lr, ncfg.weight_decay
        start_epoch = int(ck["epoch"]) + 1
        best_val = float(ck.get("best_val_supcon", float("inf")))
        best_epoch = int(ck.get("best_epoch", -1))
        print(f"[encode] resuming {name} at epoch {start_epoch} "
              f"(best val supcon {best_val:.4f} @ epoch {best_epoch})", flush=True)

    _append_config_history(run_dir, ncfg, cfg, name=name, resume=resume,
                           start_epoch=start_epoch, device=str(dev),
                           n_devices=len(meta["device_ids"]))
    if not log_path.is_file():
        with log_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(LOG_COLUMNS)

    n_params = count_parameters(model)
    print(f"[encode] run={name} params={n_params:,} devices={len(meta['device_ids'])} "
          f"tones={meta['n_tones']} batch={train_ds.batch_size()} "
          f"({ncfg.devices_per_batch}x{ncfg.segments_per_device}) "
          f"segment={ncfg.segment_samples} ({ncfg.segment_samples / cfg.sample_rate:.2f}s) "
          f"sources={list(ncfg.sources)} dev={dev}", flush=True)
    _report_split(ncfg, meta, train_ds, val_ds)

    t0 = time.time()
    stale, last_metrics, epoch = 0, {}, start_epoch - 1
    for epoch in range(start_epoch, max_epochs):
        scale = _lr_scale(epoch, ncfg.warmup_epochs, max_epochs)
        for g in opt.param_groups:
            g["lr"] = ncfg.lr * scale
        train_ds.seed = cfg.seed + epoch            # fresh windows every epoch
        loader = make_loader(train_ds, ncfg)
        model.train()
        lossfn.train()
        run = {k: 0.0 for k in ("loss", "supcon_s", "supcon_d", "xcov", "amp_ce", "dyn")}
        steps = 0
        for step, batch in enumerate(loader):
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                _, s, d = model(batch["x"])
                ps, pd = model.project(s, d)
            loss, parts = lossfn(s.float(), d.float(), ps.float(), pd.float(), batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()
            for k in run:
                run[k] += parts.get(k, 0.0)
            steps += 1
            if ncfg.log_every and step % ncfg.log_every == 0:
                print(f"e{epoch} step {step:4d}/{len(loader)}  loss {parts['loss']:.4f}  "
                      f"scs {parts['supcon_s']:.4f}  scd {parts['supcon_d']:.4f}  "
                      f"xcov {parts['xcov']:.4f}  amp {parts['amp_ce']:.3f}  "
                      f"dyn {parts['dyn']:.4f}", flush=True)
        steps = max(steps, 1)
        avg = {k: v / steps for k, v in run.items()}
        val = evaluate(model, lossfn, val_loader, dev, ncfg)
        lr_now = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"[epoch {epoch}] loss {avg['loss']:.4f} supcon_s {avg['supcon_s']:.4f} "
              f"| val {val['val_supcon']:.4f} (shuffled {val['val_supcon_shuffled']:.4f}) "
              f"amp_acc {val['val_amp_acc']:.3f} crest_mse {val['val_crest_mse']:.4f} "
              f"| lr {lr_now:.2e} {elapsed / 60:.1f} min", flush=True)
        with log_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                epoch, steps, f"{avg['loss']:.6f}", f"{avg['supcon_s']:.6f}",
                f"{avg['supcon_d']:.6f}", f"{avg['xcov']:.6f}", f"{avg['amp_ce']:.6f}",
                f"{avg['dyn']:.6f}", f"{val['val_supcon']:.6f}",
                f"{val['val_supcon_shuffled']:.6f}", f"{val['val_amp_acc']:.6f}",
                f"{val['val_crest_mse']:.6f}", f"{lr_now:.6g}", f"{elapsed:.1f}"])

        # Update the best-so-far BEFORE writing either file. Writing last.pt with the
        # pre-update value loses the improvement on a --resume: the run comes back
        # believing its best is worse than it was, and the next mediocre epoch
        # overwrites a better checkpoint.pt.
        improved = val["val_supcon"] < best_val
        if improved:
            best_val, best_epoch, stale = val["val_supcon"], epoch, 0
        else:
            stale += 1
        payload = dict(model=model.state_dict(), loss=lossfn.state_dict(),
                       optim=opt.state_dict(), encoder_cfg=asdict(ncfg),
                       device_ids=list(meta["device_ids"]), id_to_idx=meta["id_to_idx"],
                       holdout_ids=list(meta["holdout_ids"]),
                       tone_idx=meta["labels"]["tone_idx"],
                       gain_idx=meta["labels"]["gain_idx"],
                       n_tones=meta["n_tones"], n_gain=meta["n_gain"],
                       manifest_sha256=manifest_sig, params=n_params,
                       sample_rate=cfg.sample_rate, seed=cfg.seed, name=name,
                       epoch=epoch, val_supcon=val["val_supcon"],
                       best_val_supcon=best_val, best_epoch=best_epoch)
        _atomic_save(payload, last_path)
        if improved:
            _atomic_save(payload, ck_path)
        last_metrics = val
        if not improved and stale >= ncfg.early_stop_patience:
            print(f"[stop] no val improvement for {stale} epochs "
                  f"(best {best_val:.4f} @ epoch {best_epoch})", flush=True)
            break

    metrics = {
        "name": name, "params": n_params, "epochs_run": epoch - start_epoch + 1,
        "best_val_supcon": best_val, "best_epoch": best_epoch,
        "n_devices": len(meta["device_ids"]), "n_tones": meta["n_tones"],
        "n_holdout": len(meta["holdout_ids"]), "embedding_dim": ncfg.embedding_dim,
        "segment_samples": ncfg.segment_samples, "sources": list(ncfg.sources),
        "batch_size": train_ds.batch_size(), "train_hours": (time.time() - t0) / 3600.0,
        "manifest_sha256": manifest_sig, **last_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_config_snapshot(run_dir, ncfg)
    print(f"[encode] done: best val supcon {best_val:.4f} @ epoch {best_epoch} "
          f"-> {run_dir}", flush=True)
    return metrics


# --- Small file helpers (atomic, matching the rest of the pipeline) -------------
def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _write_config_snapshot(run_dir: Path, ncfg: EncoderConfig) -> None:
    import yaml

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"encoder": asdict(ncfg)}, sort_keys=False), encoding="utf-8")


def _append_config_history(run_dir: Path, ncfg: EncoderConfig, cfg: Config, **extra) -> None:
    """One never-overwritten line per start/resume — the run's audit trail."""
    row = {"at": datetime.now(timezone.utc).isoformat(), "seed": cfg.seed,
           "data_dir": str(cfg.data_dir), **extra, "encoder": asdict(ncfg)}
    with (run_dir / "config_history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def load_encoder(run_dir: Path, device: str = "cpu"):
    """``(model, checkpoint)`` from a finished encoder run."""
    run_dir = Path(run_dir)
    dev = torch.device(_resolve_device(device))
    ck = torch.load(run_dir / "checkpoint.pt", map_location=dev, weights_only=False)
    ncfg = EncoderConfig(**ck["encoder_cfg"])
    model = ToneEncoder.from_config(ncfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


__all__ = ["supcon", "cross_covariance", "EncoderLoss", "Probes", "build_datasets",
           "make_loader", "evaluate", "overfit_one_batch", "train", "load_encoder",
           "LOG_COLUMNS", "GRAD_CLIP_NORM"]
