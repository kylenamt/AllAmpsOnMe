"""Size-comparison harness + listening demos for the emulation runs (spec §3, §4).

The comparison CSV **is** the architecture-exploration deliverable: point
:func:`compare` at any set of ``results/emulate/<name>/`` run dirs and it
evaluates each checkpoint on the held-out **test** split and appends one row per
run to ``results/emulate/comparison.csv``:

    run_name, arch, params, receptive_field_ms, embedding_dim, channels,
    blocks_x_layers, test_ESR_mean, test_ESR_median, test_MRSL_mean, train_hours

Arch/timing columns are read from the run's ``metrics.json`` (written by
training); only the test metrics are recomputed here.

:func:`validate_per_device` breaks one run's test ESR back out **per amp** into
``<run>/per_device_esr.csv`` — which devices the shared model learned well, and
whether the headline mean is skewed by a few outliers. :func:`export_demos`
writes clean/target/prediction WAVs for a handful of devices — five minutes of
listening catches failure modes ESR misses (spec §4, mandatory before "good").
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from openamp.core.config import Config, EmulateConfig
from openamp.emulate.dataset import EmulationDataset
from openamp.emulate.models import arch_summary, build_model
from openamp.emulate.train import esr, pre_emphasis

# One row of the comparison CSV, in order (spec §3).
COMPARISON_COLUMNS = [
    "run_name", "arch", "params", "receptive_field_ms", "embedding_dim",
    "channels", "blocks_x_layers", "test_ESR_mean", "test_ESR_median",
    "test_MRSL_mean", "train_hours",
]

# Per-example ESR divides by target energy, so a near-silent random test window
# (tiny denominator) can dominate the *mean*. Windows below this RMS are dropped
# from the ESR stats (the corpus is normalized to -18 dBFS, so -50 dBFS is silence).
_SILENCE_DBFS = -50.0


def _resolve_device(device: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    return device if (device == "cpu" or torch.cuda.is_available()) else "cpu"


def load_model(run_dir: Path, device: str = "cpu"):
    """Reconstruct ``(model, ckpt)`` from a run's ``checkpoint.pt`` (self-contained)."""
    device = _resolve_device(device)
    ck = torch.load(Path(run_dir) / "checkpoint.pt", map_location=device,
                    weights_only=False)
    # Old checkpoints predate the arch key; EmulateConfig defaults it to film_tcn.
    ecfg = EmulateConfig(**ck["emulate_cfg"])
    model = build_model(ecfg, len(ck["device_ids"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck


def _run_metrics(run_dir: Path) -> dict:
    path = Path(run_dir) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@torch.no_grad()
def evaluate_run(cfg: Config, run_dir: Path, *, device: str = "cpu",
                 n_pairs: int | None = None) -> dict:
    """Evaluate one run on the test split; return a comparison-CSV row dict."""
    import auraloss

    run_dir = Path(run_dir)
    dev = torch.device(_resolve_device(device))
    model, ck = load_model(run_dir, str(dev))
    ecfg = EmulateConfig(**ck["emulate_cfg"])
    R = model.receptive_field
    clip = int(round(ecfg.clip_seconds * cfg.sample_rate))
    n_pairs = int(n_pairs if n_pairs is not None else max(ecfg.val_pairs, 2000))

    ds = EmulationDataset(cfg, "test", receptive_field=R, id_to_idx=ck["id_to_idx"],
                          clip_samples=clip, pairs_per_epoch=n_pairs, seed=cfg.seed + 7)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=ecfg.batch_size, shuffle=False,
                        num_workers=ecfg.num_workers, pin_memory=True)
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    floor = clip * (10.0 ** (_SILENCE_DBFS / 20.0)) ** 2   # min target sum-of-squares
    esrs, mrsls = [], []
    for batch in loader:
        inp = batch["input"].to(dev, non_blocking=True)
        target = batch["target"].to(dev, non_blocking=True)
        di = batch["device_idx"].to(dev, non_blocking=True)
        out = model(inp, di)[..., R:].squeeze(1)       # [B, clip]
        t = torch.sum(target ** 2, dim=-1)
        keep = t > floor                               # drop near-silent windows
        if keep.any():
            e = torch.sum((out[keep] - target[keep]) ** 2, dim=-1)
            esrs.append((e / (t[keep] + 1e-8)).cpu())
        mrsls.append(float(mrstft(out.unsqueeze(1).float(), target.unsqueeze(1)).item()))
    esrs = torch.cat(esrs) if esrs else torch.tensor([float("nan")])
    meta = _run_metrics(run_dir)
    return {
        "run_name": ck.get("name", run_dir.name),
        **arch_summary(ecfg),
        "params": int(meta.get("params", ck.get("params", 0))),
        "receptive_field_ms": round(1000.0 * R / cfg.sample_rate, 3),
        "embedding_dim": ecfg.embedding_dim,
        "test_ESR_mean": round(float(esrs.mean()), 6),
        "test_ESR_median": round(float(esrs.median()), 6),
        "test_MRSL_mean": round(float(np.mean(mrsls)), 6),
        "train_hours": meta.get("train_hours", ""),
    }


def compare(cfg: Config, run_dirs, *, device: str = "cpu",
            n_pairs: int | None = None) -> list[dict]:
    """Evaluate each run and append its row to ``comparison.csv`` (dedup by name)."""
    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not (run_dir / "checkpoint.pt").is_file():
            print(f"  skip {run_dir.name}: no checkpoint.pt")
            continue
        print(f"  evaluating {run_dir.name} ...", flush=True)
        rows.append(evaluate_run(cfg, run_dir, device=device, n_pairs=n_pairs))
    _append_rows(cfg.emulate_comparison_path, rows)
    return rows


def _append_rows(csv_path: Path, rows: list[dict]) -> None:
    """Append rows, replacing any existing row with the same ``run_name``."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                existing[r["run_name"]] = r
    for r in rows:
        existing[r["run_name"]] = {k: r[k] for k in COMPARISON_COLUMNS}
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COMPARISON_COLUMNS)
        w.writeheader()
        for r in existing.values():
            w.writerow(r)
    print(f"comparison -> {csv_path} ({len(existing)} runs)")


# --- Per-device validation ------------------------------------------------------
# One row of <run>/per_device_esr.csv, in order.
PER_DEVICE_COLUMNS = [
    "device_id", "name", "make", "model", "gain_bucket", "architecture",
    "n_windows", "test_ESR", "test_MRSTFT",
]


def _text(value) -> str:
    """Manifest cell -> CSV string ('' for NaN/None), so rows stay well-formed."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _device_metadata(cfg: Config) -> dict[int, dict]:
    """``{device_id: {name, make, model, gain_bucket, architecture}}`` (best-effort).

    The working manifest is preferred — it carries the human title — with
    ``devices_final`` as the fallback for a workspace trimmed down to renders.
    Absent metadata just leaves the columns empty; it must not fail a validation.
    """
    from openamp.core import manifest as manifests
    from openamp.emulate.dataset import device_names

    fields = ("make", "model", "gain_bucket", "architecture")
    names = device_names(cfg)
    for path, cols in ((cfg.manifest_path, manifests.ALL_COLUMNS),
                       (cfg.devices_final_path, manifests.DEVICES_FINAL_COLUMNS)):
        df = manifests.read_manifest(path, cols)
        if df.empty or "device_id" not in df.columns:
            continue
        df = df.dropna(subset=["device_id"])
        out = {}
        for r in df.itertuples(index=False):
            did = int(r.device_id)
            row = {f: _text(getattr(r, f, "")) for f in fields}
            row["name"] = names.get(did, "") or f"device {did}"
            out[did] = row
        if out:
            return out
    return {}


@torch.no_grad()
def validate_per_device(cfg: Config, run_dir: Path, *, device: str = "cpu",
                        n_windows: int = 128, batch_size: int | None = None,
                        seed: int | None = None, out_path: Path | None = None
                        ) -> list[dict]:
    """Per-amp test ESR for one run -> ``<run>/per_device_esr.csv`` (one row per device).

    Where :func:`evaluate_run` collapses a run to a single test number, this keeps
    the per-device breakdown: which amps the shared model actually learned, and
    whether the headline mean is being dragged by a handful of outliers. Every
    device is scored on the *same* window grid (:class:`DeviceGridDataset`), so
    the rows are comparable to each other.

    ``test_ESR`` is pooled per device — one ratio of summed energies over that
    device's windows, the same definition training optimizes
    (:func:`openamp.emulate.train.esr`) — so a quiet window cannot dominate a
    device's score. Rows come back sorted best-ESR first.
    """
    import auraloss
    from collections import defaultdict
    from torch.utils.data import DataLoader

    from openamp.emulate.dataset import DeviceGridDataset

    run_dir = Path(run_dir)
    dev = torch.device(_resolve_device(device))
    model, ck = load_model(run_dir, str(dev))
    ecfg = EmulateConfig(**ck["emulate_cfg"])
    R = model.receptive_field
    clip = int(round(ecfg.clip_seconds * cfg.sample_rate))

    # Checkpoints predating the saved map imply it: the table is built in the
    # device_ids order (dataset.build_device_index).
    id_to_idx = ck.get("id_to_idx") or {int(d): i for i, d in enumerate(ck["device_ids"])}
    ds = DeviceGridDataset(cfg, "test", receptive_field=R, id_to_idx=id_to_idx,
                           device_ids=list(ck["device_ids"]), clip_samples=clip,
                           n_windows=n_windows,
                           seed=int(seed if seed is not None else cfg.seed + 7),
                           silence_dbfs=_SILENCE_DBFS)
    loader = DataLoader(ds, batch_size=int(batch_size or ecfg.batch_size), shuffle=False,
                        num_workers=ecfg.num_workers, pin_memory=(dev.type == "cuda"))
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    n_dev = len(ds.device_ids)
    missing = len(ck["device_ids"]) - n_dev
    print(f"  {run_dir.name}: {n_dev} devices x {ds.n_windows} test windows"
          + (f" ({missing} trained devices have no test renders)" if missing else ""),
          flush=True)

    num: dict[int, float] = defaultdict(float)     # sum of squared error, per device
    den: dict[int, float] = defaultdict(float)     # sum of target energy, per device
    mrsl: dict[int, float] = defaultdict(float)    # window-weighted MRSTFT sum
    seen: dict[int, int] = defaultdict(int)
    done = 0
    for batch in loader:
        inp = batch["input"].to(dev, non_blocking=True)
        target = batch["target"].to(dev, non_blocking=True)
        di = batch["device_idx"].to(dev, non_blocking=True)
        ids = batch["device_id"]
        out = model(inp, di)[..., R:].squeeze(1)                  # [B, clip]
        e = torch.sum((out - target) ** 2, dim=-1)
        t = torch.sum(target ** 2, dim=-1)
        # A batch straddles at most two devices (the grid is device-major), but
        # split by whatever it holds so batch_size need not divide n_windows.
        for did in ids.unique().tolist():
            m = (ids == did).to(dev)
            k = int(m.sum())
            num[did] += float(e[m].sum())
            den[did] += float(t[m].sum())
            mrsl[did] += k * float(mrstft(out[m].unsqueeze(1).float(),
                                          target[m].unsqueeze(1).float()).item())
            seen[did] += k
            if seen[did] >= ds.n_windows:                          # device finished
                done += 1
                if done % 25 == 0 or done == n_dev:
                    print(f"    {done}/{n_dev} devices", flush=True)

    meta = _device_metadata(cfg)
    rows = []
    for did in ds.device_ids:
        m = meta.get(did, {})
        rows.append({
            "device_id": did,
            "name": m.get("name", "") or f"device {did}",
            "make": m.get("make", ""),
            "model": m.get("model", ""),
            "gain_bucket": m.get("gain_bucket", ""),
            "architecture": m.get("architecture", ""),
            "n_windows": seen[did],
            "test_ESR": round(num[did] / (den[did] + 1e-8), 6),
            "test_MRSTFT": round(mrsl[did] / max(seen[did], 1), 6),
        })
    rows.sort(key=lambda r: r["test_ESR"])

    out_path = Path(out_path) if out_path else run_dir / "per_device_esr.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_DEVICE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"per-device ESR -> {out_path} ({len(rows)} devices)")
    return rows


def format_per_device_summary(rows: list[dict], *, n_extremes: int = 5) -> str:
    """Human summary of :func:`validate_per_device`: spread, tails, gain buckets."""
    if not rows:
        return "No devices evaluated."
    esrs = np.asarray([r["test_ESR"] for r in rows], dtype=float)
    lines = [
        f"devices {len(rows)}   ESR mean {esrs.mean():.4f}  median {np.median(esrs):.4f}"
        f"  p90 {np.percentile(esrs, 90):.4f}  min {esrs.min():.4f}  max {esrs.max():.4f}",
    ]

    def _block(title, subset):
        lines.append(title)
        for r in subset:
            label = (r["name"] or "")[:44]
            lines.append(f"    {r['test_ESR']:8.4f}  dev {r['device_id']:>4}  {label}")

    k = min(n_extremes, len(rows))
    _block(f"  best {k} (rows are sorted best-first):", rows[:k])
    _block(f"  worst {k}:", rows[-k:][::-1])

    buckets: dict[str, list[float]] = {}
    for r in rows:
        buckets.setdefault(r["gain_bucket"] or "(unknown)", []).append(r["test_ESR"])
    if len(buckets) > 1:
        lines.append("  by gain bucket:")
        for b, vals in sorted(buckets.items(), key=lambda kv: float(np.median(kv[1]))):
            v = np.asarray(vals)
            lines.append(f"    {b:<12} n={len(vals):<4} median {np.median(v):.4f}  "
                         f"mean {v.mean():.4f}")
    return "\n".join(lines)


@torch.no_grad()
def export_demos(cfg: Config, run_dir: Path, *, n_devices: int = 4, seconds: float = 10.0,
                 device: str = "cpu") -> list[Path]:
    """Export clean/target/prediction WAV triples for a few devices (spec §4)."""
    from openamp.dsp import audio as audio_io
    from openamp.core import manifest as manifests
    from openamp.core import constants as C

    run_dir = Path(run_dir)
    dev = torch.device(_resolve_device(device))
    model, ck = load_model(run_dir, str(dev))
    R = model.receptive_field
    name = ck.get("name", run_dir.name)
    n = int(round(seconds * cfg.sample_rate))

    corpus = manifests.read_manifest(cfg.corpus_manifest_path, manifests.CORPUS_COLUMNS)
    test = corpus[(corpus["split"] == "test") & (corpus["source"] == C.SOURCE_EGDB)]
    if test.empty:
        test = corpus[corpus["split"] == "test"]
    if test.empty:
        raise RuntimeError("No test files to build demos from.")
    file_id = str(test.iloc[0]["file_id"])
    clean_path = cfg.clean_split_dir("test") / f"{file_id}.{cfg.output_format}"

    rng = np.random.default_rng(cfg.seed)
    dev_ids = list(ck["device_ids"])
    pick = rng.choice(len(dev_ids), size=min(n_devices, len(dev_ids)), replace=False)
    out_dir = cfg.emulate_demos_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_full, _ = audio_io.read_audio(clean_path)
    clean_in = clean_full[:R + n]                        # warmup + demo window
    if len(clean_in) < R + n:
        clean_in = np.pad(clean_in, (0, R + n - len(clean_in)))
    x = torch.from_numpy(clean_in.astype(np.float32))[None].to(dev)   # [1, R+n]

    written: list[Path] = []
    for row in pick:
        device_id = dev_ids[int(row)]
        di = torch.tensor([int(row)], dtype=torch.long, device=dev)
        pred = model(x, di)[..., R:].squeeze().cpu().numpy()          # [n]
        render_path = cfg.device_render_dir(device_id) / f"{file_id}.{cfg.output_format}"
        target, _ = audio_io.read_audio(render_path)
        stem = out_dir / f"{name}_dev{int(device_id):04d}"
        for tag, sig in (("clean", clean_full[:n]), ("target", target[:n]), ("pred", pred[:n])):
            p = stem.with_name(stem.name + f"_{tag}.wav")
            audio_io.write_wav(p, sig, cfg.sample_rate)
            written.append(p)
    print(f"demos ({len(pick)} devices x clean/target/pred) -> {out_dir}")
    return written
