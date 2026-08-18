"""Stage ``data_sweep_a1/``: the sweep corpus re-pointed at the captures ``data/`` was rendered from.

``data_sweep/`` cannot be reused alongside ``data/``. ``data/renders/`` was produced on
2026-07-04 from **A1 (WaveNet)** captures; ``openamp migrate-a2`` then re-pointed the
acquisition manifest at A2 models (``acquire/migrate.py``: "Renders are NOT invalidated
... only future renders pick up A2"), so the 2026-08-08 ``data_sweep/`` render used **A2
(SlimmableContainer)** captures for 408 of 450 devices. Only the 42
``architecture_fallback`` devices match. An A1 and an A2 capture of the same amp are
different models; putting both behind one ``device_id`` would ask a single embedding row
to fit two target functions.

The A1 files are all still on disk, and ``renders.parquet`` records the ``nam_sha256``
each render used — so the capture behind every ``data/`` render is recoverable exactly.
This script writes a data dir whose acquisition manifest points at those A1 files, ready
for a sweep-only render:

    python scripts/stage_sweep_a1.py
    OPENAMP_DATA_DIR=data_sweep_a1 openamp render --devices @data_sweep_a1/manifests/render_subset.txt
    python scripts/build_mixed_corpus.py --sweep data_sweep_a1

Only the 3 sweep files x 450 devices are rendered (~20 min on the GPU, ~6.6 GB) — the 88
EGDB renders in ``data/`` are reused untouched by ``build_mixed_corpus.py``.

The clean audio is copied from ``data_sweep/`` verbatim, so it stays at native level and
the NAM train/val region bounds stay exactly the ones ``scripts/build_sweep_corpus.py``
carved.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from openamp.core import constants as C
from openamp.core import manifest as manifests
from openamp.core.util import sha256_file

A1_ARCH = "WaveNet"       # the `architecture` field inside a legacy A1 .nam


def resolve_by_sha(captures_dir: Path, wanted: set[str]) -> dict[str, Path]:
    """``{sha256: path}`` for every wanted hash found under ``captures_dir``."""
    found: dict[str, Path] = {}
    scanned = 0
    for p in sorted(captures_dir.rglob("*.nam")):
        scanned += 1
        h = sha256_file(p)
        if h in wanted and h not in found:
            found[h] = p
    print(f"scanned {scanned} .nam files; resolved {len(found)}/{len(wanted)} render hashes")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data_sweep_a1"))
    ap.add_argument("--egdb", type=Path, default=Path("data"),
                    help="the data dir whose captures we must match")
    ap.add_argument("--sweep", type=Path, default=Path("data_sweep"),
                    help="the data dir to take sweep clean audio + corpus rows from")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out, egdb, sweep = args.out, args.egdb, args.sweep
    if (out / "manifests" / C.ACQUISITION_MANIFEST).is_file() and not args.force:
        raise SystemExit(f"{out} already staged; pass --force to rebuild")

    rend = manifests.read_manifest(egdb / "manifests" / C.RENDERS_MANIFEST,
                                   manifests.RENDERS_COLUMNS)
    if rend.empty:
        raise SystemExit(f"{egdb}/manifests/{C.RENDERS_MANIFEST} is missing or empty")
    by_device = rend.groupby("device_id")["nam_sha256"].first()

    found = resolve_by_sha(egdb / "captures", set(by_device))
    missing = sorted(int(d) for d in by_device.index if by_device[d] not in found)
    if missing:
        raise SystemExit(f"no .nam on disk for the render hash of devices {missing}")

    # Every one of them must actually be A1 -- if a device's data/ render already used an
    # A2 capture, re-pointing it would be the bug this script exists to prevent.
    arch = {h: json.loads(p.read_text(encoding="utf-8")).get("architecture")
            for h, p in found.items()}
    odd = {h: a for h, a in arch.items() if a != A1_ARCH}
    if odd:
        print(f"note: {len(odd)} resolved captures are not {A1_ARCH}: "
              f"{sorted(set(odd.values()))} — re-pointing them anyway, since the goal is "
              f"to match what {egdb}/renders was rendered from, whatever that was")

    (out / "manifests").mkdir(parents=True, exist_ok=True)
    for s in C.SPLITS:
        (out / "clean" / s).mkdir(parents=True, exist_ok=True)

    # --- Acquisition manifest, re-pointed --------------------------------------
    acq = manifests.read_manifest(egdb / "manifests" / C.ACQUISITION_MANIFEST,
                                  manifests.ALL_COLUMNS)
    paths = {int(d): str(found[by_device[d]]) for d in by_device.index}
    archs = {int(d): arch[by_device[d]] for d in by_device.index}
    changed = 0
    for i, row in acq.iterrows():
        d = row.get("device_id")
        if pd.isna(d) or int(d) not in paths:
            continue
        if str(row.get("file_path")) != paths[int(d)]:
            changed += 1
        acq.at[i, "file_path"] = paths[int(d)]
        acq.at[i, "architecture"] = "A1" if archs[int(d)] == A1_ARCH else "A2"
        acq.at[i, "architecture_fallback"] = True
    manifests.write_manifest(acq, out / "manifests" / C.ACQUISITION_MANIFEST)
    print(f"acquisition manifest: {len(paths)} devices re-pointed "
          f"({changed} whose file_path actually moved) -> {out}/manifests/")

    # --- Sweep corpus + clean audio, verbatim from data_sweep -------------------
    for name in (C.CORPUS_MANIFEST, C.CLIPS_MANIFEST):
        shutil.copy2(sweep / "manifests" / name, out / "manifests" / name)
    for name in (C.RENDER_SUBSET, C.EMULATE_HOLDOUT):
        shutil.copy2(egdb / "manifests" / name, out / "manifests" / name)

    corpus = manifests.read_manifest(out / "manifests" / C.CORPUS_MANIFEST,
                                     manifests.CORPUS_COLUMNS)
    for r in corpus.itertuples(index=False):
        src = sweep / "clean" / r.split / f"{r.file_id}.flac"
        dst = out / "clean" / r.split / f"{r.file_id}.flac"
        shutil.copy2(src, dst)
        if sha256_file(dst) != r.sha256:
            raise SystemExit(f"{dst}: sha256 does not match the corpus manifest row")
    print(f"clean: {len(corpus)} sweep files copied, sha256 verified "
          f"({corpus['duration_s'].sum():.1f}s total)")

    subset = out / "manifests" / C.RENDER_SUBSET
    n_dev = sum(1 for l in subset.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.lstrip().startswith("#"))
    print(f"\nnext (~20 min on the GPU, ~{corpus['duration_s'].sum() * n_dev * 90_000 / 1e9:.1f} GB):")
    print(f"  OPENAMP_DATA_DIR={out} openamp render --devices @{subset}")
    print(f"  python scripts/build_mixed_corpus.py --sweep {out}")


if __name__ == "__main__":
    main()
