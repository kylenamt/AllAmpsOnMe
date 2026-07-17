"""Produce a wet/dry capture pair to test notebooks/enroll_new_device.ipynb.

Builds a NAM-style **dry** capture — a leading calibration blip, then a log
sweep and synthetic guitar-DI at the -18 dBFS level NAM captures expect — and a
**wet** made by pushing that dry through a real device's ``.nam`` model (default
device 28, an unseen holdout amp), with an injected interface latency and a
touch of converter noise so it mimics a hardware reamp. Point the notebook's
``DRY_PATH``/``WET_PATH`` at the two files it writes.

    conda run -n open-amp3000 python scripts/make_enroll_pair.py \
        --device 28 --latency 512 --seconds 35 --out-dir data/enroll_demo

The device is unseen by the trained runs, so enrolling recovers an embedding for
a genuinely new amp; the injected latency exercises the notebook's blip
alignment (``blip_lag`` should recover it, leaving ~0 residual).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from openamp.core import constants as C
from openamp.core.config import load_config
from openamp.corpus.render import chunked_forward
from openamp.dsp import audio as audio_io
from openamp.dsp import nam as nam_render
from openamp.dsp.audio import _log_sweep, _synth_di, peak_dbfs, rms_dbfs


def _blip(sr: int, peak: float = 0.9) -> np.ndarray:
    """A short broadband transient for first-arrival latency alignment."""
    b = np.zeros(int(0.05 * sr), dtype=np.float32)      # 50 ms slot, mostly silence
    b[:8] = np.array([1, -1, 1, -1, 1, -1, 1, -1], np.float32) * peak
    return b


def build_dry(sr: int, seconds: float, seed: int) -> np.ndarray:
    """NAM-style capture: [silence, blip, silence, sweep, DI...] at -18 dBFS RMS."""
    rng = np.random.default_rng(seed)
    body = [_log_sweep(sr, 2.0), np.zeros(int(0.5 * sr), np.float32)]
    have = sum(len(x) for x in body)
    need = int(seconds * sr)
    while have < need:                                  # fill with varied DI passages
        di = _synth_di(sr, 2.0, seed=int(rng.integers(1, 1 << 30)))
        body.extend([di, np.zeros(int(0.3 * sr), np.float32)])
        have += len(di) + int(0.3 * sr)
    body = np.concatenate(body).astype(np.float32)
    body *= 10 ** ((C.TARGET_RMS_DBFS - rms_dbfs(body)) / 20)   # -> -18 dBFS RMS
    lead = np.concatenate([np.zeros(int(0.2 * sr), np.float32), _blip(sr),
                           np.zeros(int(0.1 * sr), np.float32)])
    return np.concatenate([lead, body]).astype(np.float32)


def render_wet(model, dry: np.ndarray, sr: int, chunk: int) -> np.ndarray:
    """Push dry through the NAM model (resampling only if the model isn't 48 kHz)."""
    if model.sample_rate == sr:
        return chunked_forward(model.forward, dry, model.receptive_field, chunk)
    x = audio_io.resample(dry, sr, model.sample_rate)
    ch = int(round(chunk / sr * model.sample_rate))
    y = chunked_forward(model.forward, x, model.receptive_field, ch)
    return audio_io.resample(y, model.sample_rate, sr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=28, help="device id whose .nam is the amp")
    ap.add_argument("--latency", type=int, default=512, help="injected reamp latency (samples)")
    ap.add_argument("--seconds", type=float, default=35.0, help="capture length")
    ap.add_argument("--noise-dbfs", type=float, default=-66.0, help="converter noise floor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("data/enroll_demo"))
    args = ap.parse_args()

    cfg = load_config()
    sr = cfg.sample_rate
    manifest = pd.read_parquet(cfg.manifest_path)
    hit = manifest[manifest["device_id"] == args.device]
    if hit.empty:
        raise SystemExit(f"device {args.device} not in {cfg.manifest_path}")
    row = hit.iloc[0]
    nam_path = str(row["file_path"])
    print(f"amp: device {args.device} = {row['make']} {row['model']} "
          f"({row['gain_bucket']}, {row['architecture']}) -> {nam_path}")

    model = nam_render.load_model(nam_path, device="cpu")
    dry = build_dry(sr, args.seconds, args.seed)
    wet = render_wet(model, dry, sr, cfg.chunk_samples)
    if not np.all(np.isfinite(wet)):
        raise SystemExit("non-finite render — bad .nam model?")

    rng = np.random.default_rng(args.seed + 1)
    wet = wet + (10 ** (args.noise_dbfs / 20)) * rng.standard_normal(len(wet)).astype(np.float32)
    wet = np.concatenate([np.zeros(args.latency, np.float32), wet]).astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dry_path = args.out_dir / f"dev{args.device:04d}_dry.wav"
    wet_path = args.out_dir / f"dev{args.device:04d}_wet.wav"
    audio_io.write_wav(dry_path, dry, sr)
    audio_io.write_wav(wet_path, wet, sr)

    # sanity: blip alignment should recover exactly the latency we injected
    from openamp.emulate.enroll import blip_lag
    detected = blip_lag(dry, wet, sample_rate=sr)
    print(f"dry: {len(dry) / sr:5.1f}s  rms {rms_dbfs(dry):6.1f} / peak {peak_dbfs(dry):6.1f} dBFS -> {dry_path}")
    print(f"wet: {len(wet) / sr:5.1f}s  rms {rms_dbfs(wet):6.1f} / peak {peak_dbfs(wet):6.1f} dBFS -> {wet_path}")
    print(f"injected latency={args.latency}  blip_lag detected={detected}  "
          f"(residual {detected - args.latency:+d})")
    print("\nIn notebooks/enroll_new_device.ipynb set:")
    print(f"  DRY_PATH = Path('{dry_path}')")
    print(f"  WET_PATH = Path('{wet_path}')")
    print(f"  RUN_DIR  = Path('results/emulate/embed256')   # dev {args.device} is unseen by it")
    print(f"  NAME     = 'dev{args.device:04d}'")


if __name__ == "__main__":
    main()
