"""Phase 5 enrollment: fit embeddings for unseen devices against a frozen run.

- Loads a finished run, freezes every network weight, and trains ONLY a fresh
  embedding table (one row per enrolled device) against that device's
  clean/render pairs -- can the frozen FiLM stack model an unseen amp given
  just a new conditioning vector?
- All devices enroll jointly in one loop: row i only gets gradient from batch
  items with device_idx == i. Enroll one device at a time via --devices for
  strict independence between rows.
- Rows start at the trained table's own init variance ("uniform", default) or
  its mean ("table_mean") -- see _swap_embedding. Training is plain fp32
  (trainable state is a few KB, so none of train.py's AMP/nan-guard machinery
  applies). --pairs is an optimization budget (fresh windows every epoch), not
  a unique-audio budget.
- Two front doors share one fitting loop:
  - enroll() -- corpus holdout devices, from their on-disk renders (CLI verb
    emulate-enroll).
  - enroll_pair() -- ONE wet/dry recording pair (e.g. a NAM capture signal and
    an amp's recorded response), driven from notebooks/enroll_new_device.ipynb.
    blip_lag() / estimate_lag() find the pair's reamp latency.
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
from openamp.emulate.wavenet import A2_EMBEDDING_STD

ENROLL_LR = 1e-2                 # only embeddings train (the table trained at 5e-4-2e-3 with the network)
ENROLL_EPOCHS = 30               # max; early-stopped on val ESR
ENROLL_PAIRS = 1000              # training pairs per device per epoch
EARLY_STOP_PATIENCE = 5
PLATEAU_PATIENCE = 2             # ReduceLROnPlateau (factor 0.5)
TEST_PAIRS_PER_DEVICE = 200

# "uniform": U(-a, a) at the trained table's own init variance (a = sqrt(3)*std)
# -- a new device starts where every trained device started. "table_mean":
# every row at the trained table's mean instead (see _swap_embedding).
ENROLL_INITS = ("uniform", "table_mean")
ENROLL_INIT = "uniform"
ENROLL_INIT_A = 3 ** 0.5 * A2_EMBEDDING_STD

ENROLLMENT_COLUMNS = ["device_id", "pairs", "epochs_run", "val_esr",
                      "baseline_test_esr", "test_esr"]   # one row of enrollment.csv


def _resolve_enroll_ids(cfg: Config, ck: dict, requested) -> tuple[list[int], list[int]]:
    """Resolve (enroll_ids, skipped): default is the checkpoint's holdout set.

    Explicit ids may be any render-ok device not already in the trained table
    (hard error otherwise). Devices without render_ok train+val renders are skipped.
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
    """Per-embedding-row ESR over a loader, keyed on batch["device_idx"].

    preemph set: pooled pre-emphasized ratio (val semantics, evaluate_esr).
    Otherwise: mean of per-window raw ratios with the silence_dbfs gate (test
    semantics, evaluate_run -- comparable to comparison.csv). No-data rows -> NaN.
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
           test_pairs: int = TEST_PAIRS_PER_DEVICE,
           early_stop_patience: int = EARLY_STOP_PATIENCE,
           stft_weight: float | None = None,
           batch_size: int | None = None,
           plateau_patience: int = PLATEAU_PATIENCE,
           plateau_factor: float = 0.5,
           init: str = ENROLL_INIT) -> dict:
    """Enroll unseen devices against a frozen run; returns the summary metrics.

    Writes to <run_dir>/enroll/: enrolled_embeddings.pt, enrollment.csv (merged
    by device_id across re-runs), metrics.json, enroll_log.csv (epoch -1 = init
    baseline). batch_size overrides the run's own (fp32 fit needs ~2x memory;
    try 8-16 against a batch-32 run on a 12 GB card).
    """
    run_dir = Path(run_dir)
    dev = torch.device(_resolve_device(device))
    seed = int(cfg.seed if seed is None else seed)
    torch.manual_seed(seed)

    model, ck = load_model(run_dir, str(dev))
    ecfg = EmulateConfig(**ck["emulate_cfg"])
    if stft_weight is not None:
        ecfg.stft_weight = float(stft_weight)
    if batch_size is not None:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        ecfg.batch_size = int(batch_size)
    base_name = ck.get("name", run_dir.name)
    manifest_sig = manifest_signature(cfg)
    if ck.get("manifest_sha256") and manifest_sig != ck["manifest_sha256"]:
        print("[enroll] WARNING: manifest changed since this run was trained — "
              "renders may not match what the network saw")

    enroll_ids, skipped = _resolve_enroll_ids(cfg, ck, device_ids)
    enroll_idx = {d: i for i, d in enumerate(enroll_ids)}
    n = len(enroll_ids)

    _swap_embedding(model, ecfg, n, dev, init=init)

    R = model.receptive_field
    clip = int(round(ecfg.clip_seconds * cfg.sample_rate))
    print(f"[enroll] run={base_name} arch={ecfg.arch} devices={n} "
          f"(skipped {len(skipped)}) pairs/device={pairs} batch={ecfg.batch_size} "
          f"lr={lr:g} init={init}")

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

    # --- Baseline: what the init rows score before any optimization -------------
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
        log_path=log_path, init_val=init_val, t0=t0,
        early_stop_patience=early_stop_patience,
        plateau_patience=plateau_patience, plateau_factor=plateau_factor)

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
        "init": init, "pairs": int(pairs), "epochs_run": epochs_run,
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
        "best_epoch": best_epoch, "lr": float(lr), "seed": seed, "init": init,
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


def require_enrollable(ecfg: EmulateConfig) -> None:
    """Reject archs with no shared embedding space to enroll into (tabledelta_wavenet).

    Explicit check because it would otherwise fail silently: _swap_embedding
    would install a table nothing reads, and the fit would report a plausible-
    looking but meaningless ESR.
    """
    from openamp.emulate.models import ENROLLABLE_ARCHS

    if ecfg.arch not in ENROLLABLE_ARCHS:
        raise RuntimeError(
            f"arch {ecfg.arch!r} cannot be enrolled: its conditioning is a "
            "per-device weight table, not a shared map from an embedding space, so "
            "there is nothing an unseen device can be positioned within. Enrollable "
            f"archs: {', '.join(ENROLLABLE_ARCHS)}.")


def _swap_embedding(model, ecfg: EmulateConfig, n: int, dev, *,
                    init: str = ENROLL_INIT) -> torch.Tensor:
    """Freeze the network and swap in a fresh trainable n-row table.

    init="uniform" (default): U(-a,a) at the trainer's own init variance -- a
    new device starts where every trained device started, no direction baked
    in. init="table_mean": every row at the trained table's mean (a much
    lower-ESR start, but not a point any trained device occupies).

    Model stays eval() throughout (no dropout/norm; grads still flow to the
    new rows). Returns the [n, dim] init rows.
    """
    require_enrollable(ecfg)
    if init not in ENROLL_INITS:
        raise ValueError(f"init must be one of {ENROLL_INITS}, got {init!r}")
    model.requires_grad_(False)
    emb = nn.Embedding(n, ecfg.embedding_dim).to(dev)
    with torch.no_grad():
        if init == "uniform":
            emb.weight.uniform_(-ENROLL_INIT_A, ENROLL_INIT_A)
        else:
            table_mean = model.embedding.weight.detach().mean(dim=0)
            emb.weight.copy_(table_mean.expand_as(emb.weight))
    model.embedding = emb
    model.eval()
    assert {id(p) for p in model.parameters() if p.requires_grad} == \
        {id(model.embedding.weight)}, "only the enrollment embedding may train"
    return emb.weight.detach().clone()


def _fit_embedding(model, ecfg: EmulateConfig, train_ds, val_loader, dev, *,
                   epochs: int, lr: float, seed: int, log_path: Path,
                   init_val: float, t0: float,
                   early_stop_patience: int = EARLY_STOP_PATIENCE,
                   plateau_patience: int = PLATEAU_PATIENCE,
                   plateau_factor: float = 0.5):
    """Adam on the swapped-in embedding rows only, early-stopped on val ESR.

    train_ds just needs the EmulationDataset item contract plus a seed
    attribute (bumped per epoch for fresh windows) -- WetDryDataset qualifies.
    Best starts at the init rows, so the result is never worse than init on
    val. early_stop_patience <= 0 disables early stopping (runs full epochs).
    Returns (best_emb, best_val, best_epoch, epochs_run).
    """
    R = model.receptive_field
    lossfn = EmulationLoss(ecfg.preemph, ecfg.stft_weight).to(dev)
    opt = torch.optim.Adam(model.embedding.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=plateau_factor, patience=plateau_patience)
    best_val = init_val                      # never keep rows worse than the init
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
        if early_stop_patience > 0 and since_improved >= early_stop_patience:
            print(f"[enroll stop] no val improvement for {since_improved} epochs")
            break
    return best_emb, best_val, best_epoch, epoch + 1


# --- Wet/dry pair enrollment (TONE3000-style captures; see the notebook) --------
class WetDryDataset(Dataset):
    """Random aligned windows from ONE in-memory wet/dry pair.

    Same item contract as EmulationDataset. region bounds where targets are
    drawn (left-context may reach before it). Bump seed per epoch for fresh windows.
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
    """Coarse estimate of the samples wet trails dry (negative: early).

    Pre-emphasized FFT cross-correlation, peak taken on |xcorr| (polarity-
    safe). Carries a residual bias equal to the device's group delay --
    fallback for material with no calibration blip; prefer blip_lag when one
    exists. Align with wet[lag:] / dry[-lag:] and trim to the common length.
    """
    n = int(min(len(dry), len(wet), probe_samples))
    a = np.asarray(dry[:n], dtype=np.float64)
    b = np.asarray(wet[:n], dtype=np.float64)
    a = a[1:] - preemph * a[:-1]                     # whiten: sharpen the peak
    b = b[1:] - preemph * b[:-1]
    a -= a.mean()
    b -= b.mean()
    size = 1 << int(np.ceil(np.log2(2 * len(a) - 1)))
    # correlation theorem: irfft(FB * conj(FA))[k] = sum_t wet[t+k] * dry[t]
    xc = np.abs(np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size))
    lags = np.concatenate([np.arange(max_lag + 1), np.arange(-max_lag, 0)])
    vals = np.concatenate([xc[:max_lag + 1], xc[-max_lag:]])
    return int(lags[int(np.argmax(vals))])


def _leading_edge(x: np.ndarray, thresh_frac: float, *, noise_mult: float = 8.0,
                  max_ramp: int = 256) -> int:
    """First arrival of the loudest transient in x.

    Coarse-locates the transient at half-peak, estimates the noise floor ahead
    of it, then takes the first crossing of max(noise_mult*noise, thresh_frac*peak)
    within max_ramp samples before that point -- avoids reporting an amp's
    ramped-up response instead of the true onset.
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
    """Latency (samples wet trails dry) from the leading NAM/TONE3000 capture blip.

    First-arrival, not cross-correlation: the interface round-trip is a pure
    delay L, while the amp's impulse response starts immediately but has its
    energy centroid (group delay) a few samples in. Leading-edge recovers L
    alone and leaves the group delay in the target; xcorr conflates the two
    (estimate_lag's bias). Requires the real blip at the very start of the
    signal; ValueError if a window is silent.
    """
    ns = int(min(len(dry), len(wet), round(search_seconds * sample_rate)))
    if ns < 2:
        raise ValueError("signals too short for a blip search")
    return int(_leading_edge(wet[:ns], thresh_frac) - _leading_edge(dry[:ns], thresh_frac))


# NAM standardized reamp-signal layout (sdatkinson/neural-amp-modeler,
# nam/train/core.py): train = signal[train_start:validation_start] (blips
# kept, lead-in dropped), validation = signal[validation_start:]. Only v3.0.0
# (current TONE3000 sweep) is listed; add older signals here as needed.
_NAM_SIGNALS = {
    "v3_0_0": {"length": 9_120_000, "train_start": 480_000,      # 3:10 @ 48 kHz
               "validation_start": -432_000, "blips": (504_000, 552_000)},
}


def nam_signal_regions(n_samples: int, *, sample_rate: int = 48_000,
                       version: str | None = None, tol: float = 0.005) -> dict | None:
    """NAM's own train/val regions for a standardized reamp signal, or None.

    Mirrors NAM's own split so a pair fit trains/validates on the same regions
    NAM reports, not a blind last-val_frac slice. Signal identified by length
    (within tol) unless version forces a layout. None when n_samples matches no
    known signal or the rate isn't 48kHz -- callers fall back to val_frac.
    """
    if sample_rate != 48_000:
        return None
    if version is not None and version not in _NAM_SIGNALS:
        raise ValueError(f"unknown NAM signal version {version!r}; "
                         f"known: {sorted(_NAM_SIGNALS)}")
    items = ([(version, _NAM_SIGNALS[version])] if version is not None
             else _NAM_SIGNALS.items())
    for name, spec in items:
        if abs(n_samples - spec["length"]) <= tol * spec["length"]:
            ts = int(spec["train_start"])
            vs = n_samples + int(spec["validation_start"])   # negative -> from end
            return {"version": name, "train": (ts, vs), "val": (vs, n_samples),
                    "lead_in": (0, ts), "blips": tuple(spec["blips"])}
    return None


def _norm_region(region: tuple[int, int], n: int) -> tuple[int, int]:
    """Validate a (start, stop) sample region, resolving negatives from n."""
    a, b = region
    a = int(a) if a >= 0 else n + int(a)
    b = int(b) if b >= 0 else n + int(b)
    if not 0 <= a < b <= n:
        raise ValueError(f"region {region} out of bounds for length {n}")
    return a, b


@torch.no_grad()
def render_dry(model, dry: np.ndarray, *, row: int = 0, device: str = "cpu",
               chunk_samples: int = 480_000) -> np.ndarray:
    """Stream dry through the model with real left-context; returns [len(dry)].

    Chunked so multi-minute captures fit in memory; each chunk re-reads its own
    receptive-field context, so output matches a single full-length forward.
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
                early_stop_patience: int = EARLY_STOP_PATIENCE,
                stft_weight: float | None = None,
                batch_size: int | None = None,
                plateau_patience: int = PLATEAU_PATIENCE,
                plateau_factor: float = 0.5,
                train_region: tuple[int, int] | None = None,
                val_region: tuple[int, int] | None = None,
                init: str = ENROLL_INIT,
                sources: dict | None = None) -> dict:
    """Enroll ONE new device from a sample-aligned wet/dry pair.

    dry is the capture input, wet the device's recorded response (align/level
    first -- see the notebook). Default split: last val_frac of the pair is
    val, everything before is train. Pass train_region/val_region (both,
    (start, stop) sample bounds, negatives from the end) to drive the split
    explicitly instead -- e.g. nam_signal_regions() for a standardized sweep.

    Writes to <run_dir>/enroll/pairs/<name>/: enrolled_pair.pt, metrics.json,
    enroll_log.csv. stft_weight=0.0 fits on pre-emphasized ESR alone -- useful
    when the STFT term dominates a single-pair fit's gradient (val ESR rises
    while train loss falls). batch_size overrides the run's own (fp32 fit
    needs ~2x memory; try 8-16 against a batch-32 run on a 12 GB card).
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
    if stft_weight is not None:
        ecfg.stft_weight = float(stft_weight)
    if batch_size is not None:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        ecfg.batch_size = int(batch_size)
    base_name = ck.get("name", run_dir.name)
    _swap_embedding(model, ecfg, 1, dev, init=init)

    R = model.receptive_field
    clip = int(round(ecfg.clip_seconds * cfg.sample_rate))
    sr = cfg.sample_rate
    n_total = len(dry)
    if train_region is not None or val_region is not None:
        if train_region is None or val_region is None:
            raise ValueError("pass both train_region and val_region, or neither")
        train_region = _norm_region(train_region, n_total)
        val_region = _norm_region(val_region, n_total)
        for tag, reg in (("train", train_region), ("val", val_region)):
            if reg[1] - reg[0] < clip:
                raise RuntimeError(f"{tag} region {reg} is shorter than one "
                                   f"{clip}-sample clip ({ecfg.clip_seconds:.0f} s)")
    else:
        val_len = max(clip, int(round(val_frac * n_total)))
        if n_total - val_len < clip:
            raise RuntimeError(f"pair too short: need >= {2 * clip} samples "
                               f"({2 * ecfg.clip_seconds:.0f} s) for a train+val split, "
                               f"got {n_total}")
        train_region, val_region = (0, n_total - val_len), (n_total - val_len, n_total)
    val_len = val_region[1] - val_region[0]
    print(f"[enroll-pair] run={base_name} arch={ecfg.arch} name={name} "
          f"pair={n_total / sr:.1f}s "
          f"train {train_region[0] / sr:.1f}-{train_region[1] / sr:.1f}s "
          f"val {val_region[0] / sr:.1f}-{val_region[1] / sr:.1f}s ({val_len / sr:.1f}s) "
          f"pairs/epoch={pairs} batch={ecfg.batch_size} lr={lr:g} "
          f"stft_weight={ecfg.stft_weight:g}")

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
    print(f"[enroll-pair] baseline val_esr={init_val:.5f} (init={init})")
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([-1, 0, "", "", "", f"{init_val:.6f}",
                                 f"{lr:.2e}", f"{time.time() - t0:.1f}"])

    best_emb, best_val, best_epoch, epochs_run = _fit_embedding(
        model, ecfg, train_ds, val_loader, dev, epochs=epochs, lr=lr, seed=seed,
        log_path=log_path, init_val=init_val, t0=t0,
        early_stop_patience=early_stop_patience,
        plateau_patience=plateau_patience, plateau_factor=plateau_factor)

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
        "init": init, "pairs": int(pairs), "epochs_run": epochs_run,
        "lr": float(lr), "seed": seed, "sample_rate": cfg.sample_rate,
        "stft_weight": float(ecfg.stft_weight),
        "sources": dict(sources or {}),
        "val_esr_preemph_pooled": _json_num(best_val),
        "val_esr_render_raw": _json_num(val_raw),
        "val_esr_render_preemph": _json_num(val_pe),
    }, out_dir / "enrolled_pair.pt")

    metrics = {
        "run": base_name, "name": name, "pairs": int(pairs), "epochs": int(epochs),
        "batch_size": int(ecfg.batch_size),
        "epochs_run": epochs_run, "best_epoch": best_epoch, "lr": float(lr),
        "seed": seed, "init": init, "stft_weight": float(ecfg.stft_weight),
        "pair_seconds": round(n_total / cfg.sample_rate, 2),
        "val_seconds": round(val_len / cfg.sample_rate, 2),
        "train_region": [int(train_region[0]), int(train_region[1])],
        "val_region": [int(val_region[0]), int(val_region[1])],
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

    Returns (model, blob) where blob is the saved enrolled_pair.pt dict; the
    device renders as row 0 (render_dry(model, dry)).
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
    """Write rows, replacing any existing row with the same device_id."""
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
