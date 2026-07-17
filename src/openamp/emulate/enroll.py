"""Phase 5 enrollment: fit embeddings for unseen devices against a frozen run.

Loads a finished one-to-many run (``results/emulate/<name>/checkpoint.pt``),
freezes every network weight, and optimizes ONLY a fresh embedding table — one
row per enrolled device — on that device's clean/render pairs. This answers
the generalization question the ``emulate_holdout.txt`` devices exist for: can
the frozen FiLM stack model an amp it never saw, given just a new conditioning
vector?

All devices enroll jointly in one loop: row *i* only receives gradient from
batch items with ``device_idx == i``, so devices cannot interact through the
frozen network. (Not bit-identical to per-device loops — the pooled ESR
denominator and dense-Adam momentum couple step *sizes* across devices — but
that matches the dynamics the trained table itself experienced. Enroll one
device at a time via ``--devices`` when strict independence matters.)

Rows start at the trained table's mean — the best "generic amp" prior, since
the FiLM stack was trained on that distribution — and the per-device test ESR
at that init is the baseline enrollment must beat (the best embedding defaults
to the init, so an enrolled row is never worse than the prior on val). Training
is plain fp32: the trainable state is a few KB, so none of train.py's
AMP/nan-guard machinery applies. ``--pairs`` is an *optimization* budget
(fresh random windows every epoch), not a unique-audio budget; a fixed-window
audio-budget mode is deliberate follow-up work.

Two enrollment front doors share the same frozen-net fitting loop:

- :func:`enroll` — corpus holdout devices, from their on-disk renders (the CLI
  verb ``emulate-enroll``).
- :func:`enroll_pair` — ONE wet/dry recording pair (e.g. the NAM capture
  signal and an amp's recorded response — exactly what a TONE3000 model
  capture asks for), driven from ``notebooks/enroll_new_device.ipynb``. The
  pair is split by time into train/val regions; :class:`WetDryDataset` draws
  random warmed windows from it. Real captures carry a fixed reamp latency:
  :func:`blip_lag` reads it off the capture's leading NAM blip (as NAM's
  trainer does), with :func:`estimate_lag` as the blip-free fallback.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from openamp.core import constants as C
from openamp.core import manifest as manifests
from openamp.core.config import Config, EmulateConfig
from openamp.emulate.dataset import EmulationDataset, manifest_signature
from openamp.emulate.evaluate import _SILENCE_DBFS, _resolve_device, load_model
from openamp.emulate.train import (GRAD_CLIP_NORM, EmulationLoss, _make_loader,
                                   evaluate_esr, pre_emphasis)

ENROLL_LR = 1e-2                 # only embeddings train; the table itself was
                                 # trained at 5e-4-2e-3 *alongside* the network
ENROLL_EPOCHS = 30               # max; early-stopped on val ESR
ENROLL_PAIRS = 1000              # training pairs per device per epoch
EARLY_STOP_PATIENCE = 5
PLATEAU_PATIENCE = 2             # ReduceLROnPlateau (factor 0.5)
TEST_PAIRS_PER_DEVICE = 200

# One row of enrollment.csv, in order.
ENROLLMENT_COLUMNS = ["device_id", "pairs", "epochs_run", "val_esr",
                      "baseline_test_esr", "test_esr"]


def _resolve_enroll_ids(cfg: Config, ck: dict, requested) -> tuple[list[int], list[int]]:
    """Resolve ``(enroll_ids, skipped)`` for a run.

    Default is the checkpoint's holdout set. Explicit ids may be any render-ok
    device *not* in the trained table (a seen device measures something else —
    hard error). Devices without ``render_ok`` train+val renders are skipped
    with a warning; enrolling needs data to fit and data to early-stop on.
    """
    seen = {int(d) for d in ck["device_ids"]}
    if requested:
        wanted = sorted({int(d) for d in requested})
        bad = sorted(d for d in wanted if d in seen)
        if bad:
            raise RuntimeError(f"Device(s) {bad} are in the run's trained table — "
                               "enrollment targets unseen devices only")
    else:
        wanted = sorted(int(d) for d in ck.get("holdout_ids", []))
        if not wanted:
            raise RuntimeError("Run has no holdout devices — pass --devices explicitly")

    renders = manifests.read_manifest(cfg.renders_manifest_path, manifests.RENDERS_COLUMNS)
    ok = renders[renders["status"] == C.RENDER_OK] if not renders.empty else renders
    has = {s: ({int(d) for d in ok[ok["split"] == s]["device_id"].unique()}
               if not ok.empty else set())
           for s in ("train", "val")}
    enroll_ids = [d for d in wanted if d in has["train"] and d in has["val"]]
    skipped = [d for d in wanted if d not in enroll_ids]
    if skipped:
        print(f"[enroll] skipping {len(skipped)} device(s) without render_ok "
              f"train+val renders: {skipped}")
    if not enroll_ids:
        raise RuntimeError("No enrollable devices: every requested id lacks "
                           "render_ok train+val renders")
    return enroll_ids, skipped


@torch.no_grad()
def _per_device_esr(model, loader, dev, n_rows: int, *, preemph: float | None = None,
                    silence_dbfs: float | None = None, clip: int | None = None) -> np.ndarray:
    """Per-embedding-row ESR over a loader, keyed on ``batch["device_idx"]``.

    ``preemph`` set: pooled pre-emphasized ratio-of-sums per row — the val
    semantics of :func:`openamp.emulate.train.evaluate_esr`. Otherwise: mean of
    per-window *raw* ratios with the ``silence_dbfs`` gate — the test semantics
    of :func:`openamp.emulate.evaluate.evaluate_run`, directly comparable to
    ``comparison.csv``. Rows with no surviving windows come back NaN.
    """
    R = model.receptive_field
    num = torch.zeros(n_rows, dtype=torch.float64, device=dev)
    den = torch.zeros(n_rows, dtype=torch.float64, device=dev)
    floor = None
    if preemph is None:
        floor = clip * (10.0 ** (silence_dbfs / 20.0)) ** 2   # min target sum-of-squares
    for batch in loader:
        inp = batch["input"].to(dev, non_blocking=True)
        target = batch["target"].to(dev, non_blocking=True)
        di = batch["device_idx"].to(dev, non_blocking=True)
        out = model(inp, di)[..., R:].squeeze(1)              # [B, clip]
        if preemph is not None:
            pe_o, pe_t = pre_emphasis(out, preemph), pre_emphasis(target, preemph)
            num.index_add_(0, di, torch.sum((pe_o - pe_t) ** 2, dim=-1).double())
            den.index_add_(0, di, torch.sum(pe_t ** 2, dim=-1).double())
        else:
            t = torch.sum(target ** 2, dim=-1)
            keep = t > floor                                  # drop near-silent windows
            if keep.any():
                e = torch.sum((out[keep] - target[keep]) ** 2, dim=-1)
                num.index_add_(0, di[keep], (e / (t[keep] + 1e-8)).double())
                den.index_add_(0, di[keep], torch.ones_like(t[keep]).double())
    arr = (num / den).cpu().numpy()                           # 0/0 -> NaN (no data)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def enroll(cfg: Config, run_dir: Path, *, device_ids: list[int] | None = None,
           pairs: int = ENROLL_PAIRS, epochs: int = ENROLL_EPOCHS,
           lr: float = ENROLL_LR, device: str = "cuda", seed: int | None = None,
           test_pairs: int = TEST_PAIRS_PER_DEVICE) -> dict:
    """Enroll unseen devices against a frozen run; returns the summary metrics.

    Writes to ``<run_dir>/enroll/``: ``enrolled_embeddings.pt`` (mirrors the
    ``embedding.pt`` schema + provenance), ``enrollment.csv`` (one row per
    device, merged by device_id across re-runs), ``metrics.json``, and
    ``enroll_log.csv`` (per-epoch curves; epoch -1 is the init baseline).
    """
    run_dir = Path(run_dir)
    dev = torch.device(_resolve_device(device))
    seed = int(cfg.seed if seed is None else seed)
    torch.manual_seed(seed)

    model, ck = load_model(run_dir, str(dev))
    ecfg = EmulateConfig(**ck["emulate_cfg"])
    base_name = ck.get("name", run_dir.name)
    manifest_sig = manifest_signature(cfg)
    if ck.get("manifest_sha256") and manifest_sig != ck["manifest_sha256"]:
        print("[enroll] WARNING: manifest changed since this run was trained — "
              "renders may not match what the network saw")

    enroll_ids, skipped = _resolve_enroll_ids(cfg, ck, device_ids)
    enroll_idx = {d: i for i, d in enumerate(enroll_ids)}
    n = len(enroll_ids)

    _swap_embedding(model, ecfg, n, dev)

    R = model.receptive_field
    clip = int(round(ecfg.clip_seconds * cfg.sample_rate))
    print(f"[enroll] run={base_name} arch={ecfg.arch} devices={n} "
          f"(skipped {len(skipped)}) pairs/device={pairs} lr={lr:g} init=table_mean")

    train_ds = EmulationDataset(cfg, "train", receptive_field=R, id_to_idx=enroll_idx,
                                clip_samples=clip, pairs_per_epoch=pairs * n, seed=seed)
    val_ds = EmulationDataset(cfg, "val", receptive_field=R, id_to_idx=enroll_idx,
                              clip_samples=clip, pairs_per_epoch=ecfg.val_pairs,
                              seed=seed + 999)
    val_loader = _make_loader(val_ds, ecfg.batch_size, ecfg.num_workers,
                              shuffle=False, seed=seed + 999, drop_last=False)
    test_loader = None
    try:
        test_ds = EmulationDataset(cfg, "test", receptive_field=R, id_to_idx=enroll_idx,
                                   clip_samples=clip, pairs_per_epoch=test_pairs * n,
                                   seed=cfg.seed + 7)
        test_loader = _make_loader(test_ds, ecfg.batch_size, ecfg.num_workers,
                                   shuffle=False, seed=cfg.seed + 7, drop_last=False)
    except RuntimeError:
        print("[enroll] no test renders for any enrolled device — test ESR will be NaN")

    enroll_dir = run_dir / "enroll"
    enroll_dir.mkdir(parents=True, exist_ok=True)
    log_path = enroll_dir / "enroll_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["epoch", "step", "train_loss", "train_esr",
                                 "train_stft", "val_esr", "lr", "elapsed_s"])

    # --- Baseline: what the generic-amp prior scores before any optimization ----
    t0 = time.time()
    baseline_test = np.full(n, np.nan) if test_loader is None else _per_device_esr(
        model, test_loader, dev, n, silence_dbfs=_SILENCE_DBFS, clip=clip)
    init_val = evaluate_esr(model, val_loader, dev, ecfg.preemph)
    print(f"[enroll] baseline val_esr={init_val:.5f}")
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([-1, 0, "", "", "", f"{init_val:.6f}",
                                 f"{lr:.2e}", f"{time.time() - t0:.1f}"])

    best_emb, best_val, best_epoch, epochs_run = _fit_embedding(
        model, ecfg, train_ds, val_loader, dev, epochs=epochs, lr=lr, seed=seed,
        log_path=log_path, init_val=init_val, t0=t0)

    # --- Final per-device metrics at the best rows --------------------------------
    with torch.no_grad():
        model.embedding.weight.copy_(best_emb)
    val_per_dev = _per_device_esr(model, val_loader, dev, n, preemph=ecfg.preemph)
    test_per_dev = np.full(n, np.nan) if test_loader is None else _per_device_esr(
        model, test_loader, dev, n, silence_dbfs=_SILENCE_DBFS, clip=clip)

    rows = [{"device_id": d, "pairs": int(pairs), "epochs_run": epochs_run,
             "val_esr": _csv_num(val_per_dev[i]),
             "baseline_test_esr": _csv_num(baseline_test[i]),
             "test_esr": _csv_num(test_per_dev[i])}
            for i, d in enumerate(enroll_ids)]
    _merge_enrollment_rows(enroll_dir / "enrollment.csv", rows)

    torch.save({
        # embedding.pt schema (a future loader can concat these onto the table)
        "embedding": best_emb.cpu(), "device_ids": list(enroll_ids),
        "manifest_sha256": manifest_sig, "embedding_dim": int(ecfg.embedding_dim),
        "name": f"{base_name}-enroll",
        # enrollment provenance
        "base_run": base_name, "base_epoch": int(ck.get("epoch", -1)),
        "base_manifest_sha256": ck.get("manifest_sha256", ""),
        "init": "table_mean", "pairs": int(pairs), "epochs_run": epochs_run,
        "lr": float(lr), "seed": seed,
        "per_device": {int(d): {"val_esr": _json_num(val_per_dev[i]),
                                "test_esr": _json_num(test_per_dev[i]),
                                "baseline_test_esr": _json_num(baseline_test[i])}
                       for i, d in enumerate(enroll_ids)},
    }, enroll_dir / "enrolled_embeddings.pt")

    test_mean, test_median = _agg(test_per_dev)
    base_mean, base_median = _agg(baseline_test)
    metrics = {
        "run": base_name, "n_enrolled": n, "skipped": skipped,
        "pairs": int(pairs), "epochs": int(epochs), "epochs_run": epochs_run,
        "best_epoch": best_epoch, "lr": float(lr), "seed": seed, "init": "table_mean",
        "init_val_esr_pooled": _json_num(init_val),
        "best_val_esr_pooled": _json_num(best_val),
        "test_esr_mean": test_mean, "test_esr_median": test_median,
        "baseline_test_esr_mean": base_mean, "baseline_test_esr_median": base_median,
        "trained_test_esr_mean": _trained_test_esr(cfg, base_name),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (enroll_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                             encoding="utf-8")
    print(f"[enroll done] best_val_esr={best_val:.5f} (init {init_val:.5f})  "
          f"test_esr_mean={test_mean} baseline={base_mean} -> {enroll_dir}")
    return metrics


def _swap_embedding(model, ecfg: EmulateConfig, n: int, dev) -> torch.Tensor:
    """Freeze the network and swap in a fresh trainable ``n``-row table.

    Rows start at the trained table's mean (the generic-amp prior). Both archs
    consume the table only via ``self.embedding(device_idx)``, so nothing else
    changes. The model stays eval() throughout: no dropout/norm, grads still
    flow to the new rows. Returns the init (mean) vector.
    """
    table_mean = model.embedding.weight.detach().mean(dim=0)
    model.requires_grad_(False)
    emb = nn.Embedding(n, ecfg.embedding_dim).to(dev)
    with torch.no_grad():
        emb.weight.copy_(table_mean.expand_as(emb.weight))
    model.embedding = emb
    model.eval()
    assert {id(p) for p in model.parameters() if p.requires_grad} == \
        {id(model.embedding.weight)}, "only the enrollment embedding may train"
    return table_mean


def _fit_embedding(model, ecfg: EmulateConfig, train_ds, val_loader, dev, *,
                   epochs: int, lr: float, seed: int, log_path: Path,
                   init_val: float, t0: float):
    """Adam on the swapped-in embedding rows only, early-stopped on val ESR.

    ``train_ds`` just needs the EmulationDataset item contract plus a ``seed``
    attribute (bumped per epoch for fresh windows) — WetDryDataset qualifies.
    Best starts at the init rows, so the result is never worse than the prior
    on val. Appends per-epoch rows to ``log_path`` (train_log.csv columns).
    Returns ``(best_emb, best_val, best_epoch, epochs_run)``.
    """
    R = model.receptive_field
    lossfn = EmulationLoss(ecfg.preemph, ecfg.stft_weight).to(dev)
    opt = torch.optim.Adam(model.embedding.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=PLATEAU_PATIENCE)
    best_val = init_val                      # never keep rows worse than the prior
    best_emb = model.embedding.weight.detach().clone()
    best_epoch, since_improved, step, epoch = -1, 0, 0, -1
    for epoch in range(int(epochs)):
        train_ds.seed = seed + epoch         # fresh window positions each epoch
        loader = _make_loader(train_ds, ecfg.batch_size, ecfg.num_workers,
                              shuffle=True, seed=seed + epoch, drop_last=False)
        ep_loss = ep_esr = ep_stft = 0.0
        n_steps = 0
        ep_t0 = time.time()
        for batch in loader:
            inp = batch["input"].to(dev, non_blocking=True)
            target = batch["target"].to(dev, non_blocking=True).unsqueeze(1)
            di = batch["device_idx"].to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            out = model(inp, di)[..., R:]
            loss, parts = lossfn(out.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.embedding.parameters(), GRAD_CLIP_NORM)
            opt.step()

            ep_loss += loss.item(); ep_esr += parts["esr"]; ep_stft += parts["stft"]
            n_steps += 1
            step += 1
            if step % ecfg.log_every == 0:
                sps = n_steps * ecfg.batch_size / (time.time() - ep_t0 + 1e-9)
                print(f"  e{epoch:02d} step {step:>7d} loss {loss.item():.5f} "
                      f"esr {parts['esr']:.5f} stft {parts['stft']:.4f} {sps:5.0f} smp/s",
                      flush=True)

        val_esr = evaluate_esr(model, val_loader, dev, ecfg.preemph)
        sched.step(val_esr)
        cur_lr = opt.param_groups[0]["lr"]
        improved = val_esr < best_val
        if improved:
            best_val, best_epoch, since_improved = val_esr, epoch, 0
            best_emb = model.embedding.weight.detach().clone()
        else:
            since_improved += 1
        with log_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([epoch, step, f"{ep_loss / max(n_steps, 1):.6f}",
                                     f"{ep_esr / max(n_steps, 1):.6f}",
                                     f"{ep_stft / max(n_steps, 1):.6f}",
                                     f"{val_esr:.6f}", f"{cur_lr:.2e}",
                                     f"{time.time() - t0:.1f}"])
        print(f"[enroll epoch {epoch:02d}] val_esr={val_esr:.5f} best={best_val:.5f} "
              f"lr={cur_lr:.1e}{'  *' if improved else ''}", flush=True)
        if since_improved >= EARLY_STOP_PATIENCE:
            print(f"[enroll stop] no val improvement for {since_improved} epochs")
            break
    return best_emb, best_val, best_epoch, epoch + 1


# --- Wet/dry pair enrollment (TONE3000-style captures; see the notebook) --------
class WetDryDataset(Dataset):
    """Random aligned windows from ONE in-memory wet/dry pair.

    Same item contract as :class:`EmulationDataset`: ``input`` is the dry
    window with a receptive field of real left-context (zero-padded only
    before sample 0), ``target`` the aligned wet window, ``device_idx`` the
    embedding row. ``region`` bounds where *targets* are drawn — left-context
    may reach before it (real audio), which is why the val region sits at the
    end of the pair. Bump ``seed`` per epoch for fresh windows.
    """

    def __init__(self, dry: np.ndarray, wet: np.ndarray, *, receptive_field: int,
                 clip_samples: int, region: tuple[int, int] | None = None,
                 pairs_per_epoch: int = 500, seed: int = 1234, row: int = 0):
        if len(dry) != len(wet):
            raise ValueError("dry/wet must be sample-aligned (equal length); "
                             "use estimate_lag + trim first")
        self.dry = np.ascontiguousarray(dry, dtype=np.float32)
        self.wet = np.ascontiguousarray(wet, dtype=np.float32)
        self.receptive_field = int(receptive_field)
        self.clip_samples = int(clip_samples)
        self.region = ((0, len(dry)) if region is None
                       else (int(region[0]), int(region[1])))
        if self.region[1] - self.region[0] < self.clip_samples:
            raise ValueError(f"region {self.region} is shorter than one "
                             f"{self.clip_samples}-sample clip")
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.seed = int(seed)
        self.row = int(row)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def item_arrays(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, int(i)))
        R, clip = self.receptive_field, self.clip_samples
        lo, hi = self.region
        c = int(rng.integers(lo, hi - clip + 1))
        s, pad = c - R, 0
        if s < 0:
            pad, s = -s, 0
        inp = self.dry[s:c + clip]
        if pad:
            inp = np.concatenate([np.zeros(pad, dtype=np.float32), inp])
        return {"input": inp, "target": self.wet[c:c + clip], "device_idx": self.row}

    def __getitem__(self, i: int) -> dict:
        a = self.item_arrays(i)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(a["input"])),
            "target": torch.from_numpy(np.ascontiguousarray(a["target"])),
            "device_idx": torch.tensor(a["device_idx"], dtype=torch.long),
        }


def estimate_lag(dry: np.ndarray, wet: np.ndarray, *, max_lag: int = 4800,
                 probe_samples: int = 48_000 * 30, preemph: float = 0.95) -> int:
    """Coarse estimate of the samples ``wet`` trails ``dry`` (negative: early).

    Both signals are **pre-emphasized** (``y[n]=x[n]-preemph*x[n-1]``) before an
    FFT cross-correlation, and the peak is taken on ``|xcorr|`` so a polarity
    inversion in the reamp chain doesn't flip the sign. Whitening matters: a
    plain broadband xcorr is pulled tens of samples off by the amp's own
    frequency shaping — enough to hurt, since embedding-only enrollment is
    sensitive to sub-millisecond misalignment (a 37-sample error ~tripled the
    val ESR in testing).

    This still carries a residual bias equal to the device's group delay
    (order ~10-40 samples), so it is the **fallback** for material without a
    calibration blip. When the capture opens with the NAM blip, prefer
    :func:`blip_lag`, which reads latency off first-arrival and isolates the
    interface delay from the amp's group delay. Align with ``wet[lag:]`` /
    ``dry[-lag:]`` and trim to the common length before building a
    :class:`WetDryDataset`.
    """
    n = int(min(len(dry), len(wet), probe_samples))
    a = np.asarray(dry[:n], dtype=np.float64)
    b = np.asarray(wet[:n], dtype=np.float64)
    a = a[1:] - preemph * a[:-1]                     # whiten: sharpen the peak
    b = b[1:] - preemph * b[:-1]
    a -= a.mean()
    b -= b.mean()
    size = 1 << int(np.ceil(np.log2(2 * len(a) - 1)))
    # correlation theorem: irfft(FB * conj(FA))[k] = sum_t wet[t+k] * dry[t],
    # so the argmax over k is the delay of wet relative to dry.
    xc = np.abs(np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size))
    lags = np.concatenate([np.arange(max_lag + 1), np.arange(-max_lag, 0)])
    vals = np.concatenate([xc[:max_lag + 1], xc[-max_lag:]])
    return int(lags[int(np.argmax(vals))])


def _leading_edge(x: np.ndarray, thresh_frac: float, *, noise_mult: float = 8.0,
                  max_ramp: int = 256) -> int:
    """First arrival of the loudest transient in ``x``.

    Two passes, because an amp's blip *response* ramps up over ~10-20 samples
    (it starts immediately — ``h[0] != 0`` — but ~40 dB below its peak), so a
    high threshold reports the ramp, not the onset. We coarse-locate the
    transient at half-peak, estimate the noise floor from the lead-in ahead of
    it, then take the first crossing of ``max(noise_mult * noise, thresh_frac *
    peak)`` within ``max_ramp`` samples before the coarse hit. The noise term
    keeps a hissy high-gain capture from triggering on its own floor; on a clean
    digital lead-in it stays at ``thresh_frac`` and lands within a sample or two
    of true first-arrival.
    """
    seg = np.abs(np.asarray(x, dtype=np.float64) - float(np.mean(x)))
    peak = float(seg.max())
    if peak <= 0.0:
        raise ValueError("no signal in the blip search window")
    coarse = int(np.argmax(seg >= 0.5 * peak))
    lo = max(coarse - int(max_ramp), 0)
    noise = float(np.sqrt(np.mean(seg[:lo] ** 2))) if lo > 32 else 0.0
    thresh = max(noise_mult * noise, thresh_frac * peak)
    window = seg[lo:coarse + 1]
    hit = window >= thresh
    return lo + (int(np.argmax(hit)) if hit.any() else int(len(window) - 1))


def blip_lag(dry: np.ndarray, wet: np.ndarray, *, sample_rate: int = 48_000,
             search_seconds: float = 2.0, thresh_frac: float = 0.02) -> int:
    """Latency (samples ``wet`` trails ``dry``) from the leading capture blip.

    The NAM / TONE3000 capture signal opens with a loud broadband blip *for
    exactly this purpose*. We take the first arrival of that transient in each
    signal (:func:`_leading_edge`) and return ``wet_onset - dry_onset``.

    Why first-arrival, not cross-correlation: the interface round-trip is a pure
    delay ``L``, while the amp's impulse response ``h`` starts immediately
    (``h[0] != 0``) but has its energy centroid a few samples in (its group
    delay). So the blip *starts* arriving at ``d + L`` regardless of tone —
    leading-edge recovers ``L`` and leaves the amp's group delay in the target
    for the model to learn, whereas whole-signal xcorr recovers
    ``L + group_delay`` and can't separate them (that is the bias in
    :func:`estimate_lag`). This is essentially NAM's own delay calibration.

    Accuracy is a few samples, not exact: a real amp's blip response ramps up
    over ~10-20 samples from ~40 dB below its peak, so detection is threshold-
    and noise-limited. Measured against a known 512-sample latency through a
    real high-gain capture: ``+2`` at the ``thresh_frac`` default, vs ``+18`` at
    0.5 and ``+13`` for :func:`estimate_lag`. Lower ``thresh_frac`` tracks first
    arrival more tightly until the capture's noise floor stops you.

    Requires a real blip at the very start: play the *entire* NAM signal,
    including the leading blips. If the recording's loudest early transient is
    not the blip, shrink ``search_seconds`` to isolate it. ``ValueError`` if a
    window is silent.
    """
    ns = int(min(len(dry), len(wet), round(search_seconds * sample_rate)))
    if ns < 2:
        raise ValueError("signals too short for a blip search")
    return int(_leading_edge(wet[:ns], thresh_frac) - _leading_edge(dry[:ns], thresh_frac))


@torch.no_grad()
def render_dry(model, dry: np.ndarray, *, row: int = 0, device: str = "cpu",
               chunk_samples: int = 480_000) -> np.ndarray:
    """Stream ``dry`` through the model with real left-context; returns [len(dry)].

    Chunked so multi-minute captures fit in memory; each chunk re-reads its
    receptive field of context from the source, so the output is identical to
    a single full-length forward.
    """
    dev = torch.device(_resolve_device(device))
    R = model.receptive_field
    di = torch.tensor([int(row)], dtype=torch.long, device=dev)
    out = np.empty(len(dry), dtype=np.float32)
    for s in range(0, len(dry), int(chunk_samples)):
        e = min(s + int(chunk_samples), len(dry))
        a, pad = s - R, 0
        if a < 0:
            pad, a = -a, 0
        x = dry[a:e].astype(np.float32, copy=False)
        if pad:
            x = np.concatenate([np.zeros(pad, dtype=np.float32), x])
        xt = torch.from_numpy(np.ascontiguousarray(x))[None].to(dev)
        out[s:e] = model(xt, di)[..., R:].squeeze(0).squeeze(0).cpu().numpy()
    return out


def enroll_pair(cfg: Config, run_dir: Path, dry: np.ndarray, wet: np.ndarray, *,
                name: str, pairs: int = 500, epochs: int = ENROLL_EPOCHS,
                lr: float = ENROLL_LR, device: str = "cuda", seed: int | None = None,
                val_frac: float = 0.1, val_pairs: int = 200,
                sources: dict | None = None) -> dict:
    """Enroll ONE new device from a sample-aligned wet/dry pair (spec: Phase 5).

    ``dry`` is the capture input (e.g. the NAM sweep signal TONE3000 sends to
    every amp), ``wet`` the device's recorded response — align and level them
    first (see the notebook). The last ``val_frac`` of the pair is the val
    region; windows never cross the split. Writes to
    ``<run_dir>/enroll/pairs/<name>/``: ``enrolled_pair.pt`` (the fitted
    vector + provenance), ``metrics.json``, ``enroll_log.csv`` (epoch -1 is
    the table-mean baseline). Returns the metrics dict.
    """
    run_dir = Path(run_dir)
    dev = torch.device(_resolve_device(device))
    seed = int(cfg.seed if seed is None else seed)
    torch.manual_seed(seed)
    dry = np.ascontiguousarray(dry, dtype=np.float32)
    wet = np.ascontiguousarray(wet, dtype=np.float32)
    if len(dry) != len(wet):
        raise ValueError("dry/wet must be sample-aligned (equal length); "
                         "use estimate_lag + trim first")

    model, ck = load_model(run_dir, str(dev))
    ecfg = EmulateConfig(**ck["emulate_cfg"])
    base_name = ck.get("name", run_dir.name)
    _swap_embedding(model, ecfg, 1, dev)

    R = model.receptive_field
    clip = int(round(ecfg.clip_seconds * cfg.sample_rate))
    n_total = len(dry)
    val_len = max(clip, int(round(val_frac * n_total)))
    if n_total - val_len < clip:
        raise RuntimeError(f"pair too short: need >= {2 * clip} samples "
                           f"({2 * ecfg.clip_seconds:.0f} s) for a train+val split, "
                           f"got {n_total}")
    train_region, val_region = (0, n_total - val_len), (n_total - val_len, n_total)
    print(f"[enroll-pair] run={base_name} arch={ecfg.arch} name={name} "
          f"pair={n_total / cfg.sample_rate:.1f}s "
          f"(val last {val_len / cfg.sample_rate:.1f}s) pairs/epoch={pairs} lr={lr:g}")

    train_ds = WetDryDataset(dry, wet, receptive_field=R, clip_samples=clip,
                             region=train_region, pairs_per_epoch=pairs, seed=seed)
    val_ds = WetDryDataset(dry, wet, receptive_field=R, clip_samples=clip,
                           region=val_region, pairs_per_epoch=val_pairs,
                           seed=seed + 999)
    val_loader = _make_loader(val_ds, ecfg.batch_size, ecfg.num_workers,
                              shuffle=False, seed=seed + 999, drop_last=False)

    out_dir = run_dir / "enroll" / "pairs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "enroll_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["epoch", "step", "train_loss", "train_esr",
                                 "train_stft", "val_esr", "lr", "elapsed_s"])

    t0 = time.time()
    init_val = evaluate_esr(model, val_loader, dev, ecfg.preemph)
    print(f"[enroll-pair] baseline val_esr={init_val:.5f} (init=table_mean)")
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([-1, 0, "", "", "", f"{init_val:.6f}",
                                 f"{lr:.2e}", f"{time.time() - t0:.1f}"])

    best_emb, best_val, best_epoch, epochs_run = _fit_embedding(
        model, ecfg, train_ds, val_loader, dev, epochs=epochs, lr=lr, seed=seed,
        log_path=log_path, init_val=init_val, t0=t0)

    # Deterministic final check: render the whole val region once (real
    # left-context from the train side) and score it raw + pre-emphasized.
    with torch.no_grad():
        model.embedding.weight.copy_(best_emb)
    lo, hi = val_region
    ctx = min(R, lo)
    pred = render_dry(model, dry[lo - ctx:hi], row=0, device=str(dev))[ctx:]
    target = wet[lo:hi]
    val_raw = _np_esr(pred, target)
    val_pe = _np_esr(pred, target, coeff=ecfg.preemph)

    torch.save({
        "embedding": best_emb.cpu(),                  # [1, dim]: concat-compatible
        "name": name, "embedding_dim": int(ecfg.embedding_dim),
        "base_run": base_name, "base_epoch": int(ck.get("epoch", -1)),
        "base_manifest_sha256": ck.get("manifest_sha256", ""),
        "init": "table_mean", "pairs": int(pairs), "epochs_run": epochs_run,
        "lr": float(lr), "seed": seed, "sample_rate": cfg.sample_rate,
        "sources": dict(sources or {}),
        "val_esr_preemph_pooled": _json_num(best_val),
        "val_esr_render_raw": _json_num(val_raw),
        "val_esr_render_preemph": _json_num(val_pe),
    }, out_dir / "enrolled_pair.pt")

    metrics = {
        "run": base_name, "name": name, "pairs": int(pairs), "epochs": int(epochs),
        "epochs_run": epochs_run, "best_epoch": best_epoch, "lr": float(lr),
        "seed": seed, "init": "table_mean",
        "pair_seconds": round(n_total / cfg.sample_rate, 2),
        "val_seconds": round(val_len / cfg.sample_rate, 2),
        "init_val_esr_pooled": _json_num(init_val),
        "best_val_esr_pooled": _json_num(best_val),
        "val_esr_render_raw": _json_num(val_raw),
        "val_esr_render_preemph": _json_num(val_pe),
        "sources": dict(sources or {}),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                          encoding="utf-8")
    print(f"[enroll-pair done] best_val_esr={best_val:.5f} (init {init_val:.5f})  "
          f"val-render raw={val_raw:.5f} preemph={val_pe:.5f} -> {out_dir}")
    return metrics


def load_pair_model(run_dir: Path, name: str, device: str = "cpu"):
    """Rebuild a run's frozen model with an enrolled pair vector installed.

    Returns ``(model, blob)`` where ``blob`` is the saved ``enrolled_pair.pt``
    dict; the device renders as row 0 (``render_dry(model, dry)``).
    """
    run_dir = Path(run_dir)
    blob = torch.load(run_dir / "enroll" / "pairs" / name / "enrolled_pair.pt",
                      map_location="cpu", weights_only=False)
    dev = torch.device(_resolve_device(device))
    model, ck = load_model(run_dir, str(dev))
    ecfg = EmulateConfig(**ck["emulate_cfg"])
    _swap_embedding(model, ecfg, 1, dev)
    with torch.no_grad():
        model.embedding.weight.copy_(blob["embedding"].reshape(1, -1).to(dev))
    return model, blob


def _np_esr(pred: np.ndarray, target: np.ndarray, coeff: float | None = None) -> float:
    """Whole-signal ESR on numpy arrays (optionally pre-emphasized)."""
    if coeff is not None:
        pred = pred[1:] - coeff * pred[:-1]
        target = target[1:] - coeff * target[:-1]
    return float(np.sum((pred - target) ** 2) / (np.sum(target ** 2) + 1e-12))


def _merge_enrollment_rows(csv_path: Path, rows: list[dict]) -> None:
    """Write rows, replacing any existing row with the same ``device_id``."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[int, dict] = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                existing[int(r["device_id"])] = r
    for r in rows:
        existing[int(r["device_id"])] = {k: r[k] for k in ENROLLMENT_COLUMNS}
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ENROLLMENT_COLUMNS)
        w.writeheader()
        for d in sorted(existing):
            w.writerow(existing[d])
    print(f"enrollment -> {csv_path} ({len(existing)} devices)")


def _trained_test_esr(cfg: Config, run_name: str):
    """Best-effort trained-device reference from comparison.csv (None if absent)."""
    path = Path(cfg.emulate_comparison_path)
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("run_name") == run_name:
                try:
                    return float(r["test_ESR_mean"])
                except (KeyError, ValueError):
                    return None
    return None


def _agg(arr: np.ndarray):
    """(mean, median) over finite entries, or (None, None) if there are none."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    return round(float(finite.mean()), 6), round(float(np.median(finite)), 6)


def _json_num(x):
    """NaN-safe float for JSON/metadata (json has no NaN literal)."""
    x = float(x)
    return round(x, 6) if np.isfinite(x) else None


def _csv_num(x) -> str:
    """NaN-safe cell for enrollment.csv (empty when there was no data)."""
    x = float(x)
    return f"{x:.6f}" if np.isfinite(x) else ""
