"""Build a sweep-only corpus: the TONE3000 capture signal, split NAM's way.

Answers "what if the one-to-many net were trained exactly the way TONE3000 trains a
single capture, but on every amp at once?" — the corpus becomes the capture signal
itself, sliced into NAM's own train/val regions, rather than 40 minutes of EGDB DI.

Written into a **separate data dir** (default ``data_sweep/``) so the existing EGDB
corpus and its 106 GB of renders are untouched and both experiments can coexist.
Only ``renders/`` costs real space (~8 GB for 450 devices); the acquisition manifest
and the emulate holdout list are copied so the trained device set and the 45 held-out
devices are **identical** to the EGDB runs and the two are directly comparable.

    python scripts/build_sweep_corpus.py                       # build the corpus
    OPENAMP_DATA_DIR=data_sweep openamp render                 # render it (~20 min)
    openamp emulate --config configs/emulate/nam_a2_256_sweep.yaml   # (with the same env var)

Three deliberate departures from `openamp corpus`, all documented at their site below:
region splitting instead of per-file splitting, native level instead of -18 dBFS RMS,
and a carved test region because NAM's protocol defines no test split.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from openamp.core import constants as C
from openamp.core import manifest as manifests
from openamp.core.config import load_config
from openamp.corpus.build import clip_grid
from openamp.dsp import audio as audio_io
from openamp.dsp.audio import peak_dbfs, rms_dbfs
from openamp.core.util import sha256_file

# Sample bounds at 48 kHz — NAM's own layout (nam/train/core.py), unmodified:
# lead-in [0:480_000] discarded, train [480_000:-432_000], validation [-432_000:].
# See `openamp.emulate.enroll.nam_signal_regions`, which this mirrors.
#
# NAM's protocol has **no test split** — it early-stops on validation and reports
# validation ESR, full stop. Config requires three splits and the evaluation harness
# (per_device_esr.csv, comparison.csv) reads `test`, so `test` is the *same 9 s block*
# as val rather than a carved-out third region: staying faithful to NAM matters more
# than having an independent in-signal test set, and taking a test region out of the
# train region would mean training on less than NAM does.
#
# So `test_ESR` here re-measures the val audio under the test metric (per-window raw
# ESR on a fixed grid, vs val's pooled pre-emphasized ESR) — it is NOT held-out. The
# genuine held-out check is the finished checkpoint evaluated against the EGDB data
# dir, where `test` is unseen songs; the device set is identical, so that just works.
_TRAIN = (480_000, 8_688_000)    #  10.0 - 181.0 s  (171 s) — NAM's train region
_VAL = (8_688_000, 9_120_000)    # 181.0 - 190.0 s  (  9 s) — NAM's validation block

REGIONS = {
    "sweep_train": ("train", *_TRAIN),
    "sweep_val":   ("val",   *_VAL),
    "sweep_test":  ("test",  *_VAL),   # deliberately identical to val (see above)
}


def find_sweep(cfg) -> Path:
    """The capture signal, tolerating the filename it actually ships under."""
    if cfg.nam_sweep_path.is_file():
        return cfg.nam_sweep_path
    hits = sorted(cfg.nam_sweep_path.parent.glob("*.wav"))
    if not hits:
        raise FileNotFoundError(f"no capture signal .wav in {cfg.nam_sweep_path.parent}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data_sweep"), help="new data dir")
    ap.add_argument("--sweep", type=Path, default=None, help="override the capture signal path")
    ap.add_argument("--force", action="store_true", help="overwrite an existing corpus")
    args = ap.parse_args()

    cfg = load_config()
    sr = cfg.sample_rate
    src = Path(args.sweep) if args.sweep else find_sweep(cfg)

    out = args.out
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    for s in C.SPLITS:
        (out / "clean" / s).mkdir(parents=True, exist_ok=True)

    if (out / "manifests" / C.CORPUS_MANIFEST).is_file() and not args.force:
        raise SystemExit(f"{out} already has a corpus; pass --force to rebuild")

    # The device set and the holdout must match the EGDB runs exactly, or the two
    # experiments aren't comparable. load_or_create_holdout treats the file as the
    # source of truth, so copying it pins the same 45 devices; render_subset.txt
    # pins the same 450 rendered devices (the acquisition manifest carries 844, and
    # `openamp render` with no --devices would render every one of them).
    for name in (C.ACQUISITION_MANIFEST, C.EMULATE_HOLDOUT, C.RENDER_SUBSET):
        s = cfg.manifests_dir / name
        if not s.is_file():
            raise FileNotFoundError(f"missing {s} — run `openamp finalize` first")
        shutil.copy2(s, out / "manifests" / name)
        print(f"copied {name}")

    x, file_sr = audio_io.read_audio(src)
    x = audio_io.resample(x, file_sr, sr).astype(np.float32)
    print(f"\n{src}  {len(x) / sr:.1f}s  rms {rms_dbfs(x):.2f}  peak {peak_dbfs(x):.2f} dBFS")
    if len(x) < max(b for _, _, b in REGIONS.values()):
        raise SystemExit(f"signal is {len(x)} samples, too short for the NAM v3 layout")

    corpus_rows, clip_rows = [], []
    for file_id, (split, a, b) in REGIONS.items():
        seg = np.ascontiguousarray(x[a:b])
        # NATIVE LEVEL, deliberately: `openamp corpus` would apply compute_gain_db
        # (RMS -> -18 dBFS, capped at a -1 dBFS peak), which on this file is a 0.9 dB
        # cut because its peaks already sit at -0.09 dBFS. TONE3000 trains its captures
        # on the signal as-is and enrollment feeds it as-is, so normalizing here would
        # reintroduce the exact train/enroll level mismatch this dataset exists to remove.
        path = out / "clean" / split / f"{file_id}.{cfg.output_format}"
        audio_io.write_flac(path, seg, sr)
        n = len(seg)
        corpus_rows.append({
            "file_id": file_id, "split": split, "source": C.SOURCE_SWEEP,
            "orig_path": f"{src}#{a}:{b}", "duration_s": n / sr,
            "applied_gain_db": 0.0, "sha256": sha256_file(path),
        })
        clip_rows += [{"clip_id": f"{file_id}_{k:05d}", "file_id": file_id, "split": split,
                       "start_sample": s0, "num_samples": ns}
                      for k, (s0, ns) in enumerate(clip_grid(n, cfg.clip_samples))]
        print(f"  {file_id:12s} {split:5s} {a / sr:6.1f}-{b / sr:6.1f}s  {n / sr:6.1f}s  "
              f"rms {rms_dbfs(seg):7.2f}  peak {peak_dbfs(seg):6.2f} dBFS -> {path}")

    manifests.write_manifest(pd.DataFrame(corpus_rows, columns=manifests.CORPUS_COLUMNS),
                             out / "manifests" / C.CORPUS_MANIFEST)
    manifests.write_manifest(pd.DataFrame(clip_rows, columns=manifests.CLIPS_COLUMNS),
                             out / "manifests" / C.CLIPS_MANIFEST)

    # Slicing must be lossless: the source is 24-bit PCM and we neither resample nor
    # apply gain, so each region must read back bit-identical to its source window.
    for file_id, (split, a, b) in REGIONS.items():
        back, _ = audio_io.read_audio(out / "clean" / split / f"{file_id}.{cfg.output_format}")
        if not np.array_equal(back, x[a:b]):
            raise SystemExit(f"{file_id} did not round-trip losslessly")
    print("\nround-trip check: all regions bit-identical to the source window")

    total = sum(r["duration_s"] for r in corpus_rows)
    subset = out / "manifests" / C.RENDER_SUBSET
    n_dev = sum(1 for l in subset.read_text().splitlines()
                if l.strip() and not l.lstrip().startswith("#"))
    print(f"corpus: {len(corpus_rows)} files, {total:.1f}s -> {out}/manifests/")
    print(f"est. render footprint: {total * n_dev * 90_000 / 1e9:.1f} GB for {n_dev} devices\n")
    print("next:")
    print(f"  OPENAMP_DATA_DIR={out} openamp render --devices @{subset}")
    print(f"  OPENAMP_DATA_DIR={out} openamp emulate --config configs/emulate/nam_a2_256_sweep.yaml")


if __name__ == "__main__":
    main()
