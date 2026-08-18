"""Assemble ``data_mixed/``: the EGDB corpus plus the NAM capture signal, without re-rendering.

Answers "what if the one-to-many net saw the capture signal *and* 40 minutes of EGDB DI?"
— the corpus every emulate run to date has trained on is sweep-free (``NAM_SWEEP_FILENAME``
is ``v3_0_0.wav``; the file on disk is ``T3K-sweep-v3.wav``, so ``corpus/build.py`` set
``have_sweep = False`` and said nothing), and the measured sweep domain gap is 16-70x on
sweep content.

    python scripts/build_mixed_corpus.py
    OPENAMP_DATA_DIR=data_mixed openamp emulate --config configs/emulate/nam_a2_tanh_ch16_512.yaml

**No GPU, no NAM, no re-render.** ``data_sweep/`` already holds sweep renders for all 450
devices, and a render is exactly ``raw_amp_output * output_scale`` with the scalar stored
in the manifest — so the raw output is recoverable and the renders can be re-scaled
arithmetically onto the combined corpus's per-device peak:

    raw_peak[d]    = max over that corpus's rows of 10**(out_peak_db/20) / output_scale
    mixed_peak[d]  = max(raw_peak_egdb[d], raw_peak_sweep[d])
    scale_mixed[d] = output_scale(mixed_peak[d])          # ceiling 0.999
    new_render     = stored_render * (scale_mixed[d] / stored_scale[d])

Measured on the current manifests, the sweep raises the peak on only 24 of 450 devices
(and the EGDB set raises it over the sweep on 4), so ~2,100 of 40,950 files are actually
rewritten. Everything else is symlinked back to ``data/`` / ``data_sweep/``, which stay
untouched — the eight existing runs keep the exact corpus they trained on. ``--copy``
makes real copies instead (~113 GB) if you need ``data_mixed/`` to stand alone.

Deliberate departures, both inherited from ``scripts/build_sweep_corpus.py``:

- **The sweep clean files keep their native level** (no -18 dBFS RMS / -1 dBFS peak
  normalization). That is the exact input its renders were computed from; re-normalizing
  would break the (clean, render) pair. The corpus is level-inhomogeneous by design.
- **``sweep_test`` is a byte-copy of ``sweep_val``** — NAM's protocol defines no test
  split. Both are the 181-190 s block and are disjoint from ``sweep_train`` (10-181 s),
  so neither leaks training audio; but test re-measures val on sweep content.

Also writes ``emulate_holdout.txt`` with device 486 appended — see ``DROP_DEVICE``.
"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from openamp.core import constants as C
from openamp.core import manifest as manifests
from openamp.core.util import sha256_file
from openamp.corpus.render import output_scale
from openamp.dsp import audio as audio_io
from openamp.dsp.audio import peak_dbfs, rms_dbfs

# Excluded from the one-to-many table in this corpus: a near-silent capture (median
# out_rms_db -40.2 vs -20.2 across the corpus) whose test ESR exceeds 1.0 in every run
# that has per-device output — i.e. the model does worse on it than predicting silence.
# Appended to data_mixed's holdout file only; data/manifests/emulate_holdout.txt is left
# alone so the existing 405-device runs stay --resume-able.
DROP_DEVICE = 486
DROP_DEVICE_NAME = "Lel 3000 1993 — output_amp2"

UNCHANGED = 1e-9        # |ratio - 1| below this means "the file is already correct"

# ``out_peak_db`` is stored to 3 dp, so a peak recovered from it carries at most
# 10**(0.0005/20) - 1 = 5.8e-5 relative error, and a rescaled file can therefore land up to
# 0.999 * (1 + 5.8e-5) = 0.99906 instead of exactly at the ceiling. That is immaterial —
# the ceiling is arbitrary, the scalar is still one constant per device, and 0.99906 is
# nowhere near clipping — but the guard has to admit it. Anything past this is a real bug.
PEAK_TOLERANCE = 1e-4


def _load(root: Path, name: str, columns) -> pd.DataFrame:
    df = manifests.read_manifest(root / "manifests" / name, columns)
    if df.empty:
        raise SystemExit(f"{root}/manifests/{name} is missing or empty")
    return df


def device_scales(renders: pd.DataFrame, label: str) -> tuple[pd.Series, pd.Series]:
    """``(stored_scale, raw_peak)`` per device, from the manifest alone.

    ``out_peak_db`` is the peak *after* scaling, so ``10**(db/20) / output_scale`` recovers
    the amp's raw peak. It is stored to 3 dp, i.e. ~1.2e-4 relative — four orders below
    what the 0.999 ceiling decision needs, and it cannot cause a clip either way because
    ``output_scale`` is defined to put the result *at* the ceiling.
    """
    if not (renders["status"] == C.RENDER_OK).all():
        raise SystemExit(f"{label}: not every render is {C.RENDER_OK}")
    per_dev = renders.groupby("device_id")["output_scale"].nunique()
    if (per_dev != 1).any():
        raise SystemExit(f"{label}: output_scale is not constant within a device")
    scale = renders.groupby("device_id")["output_scale"].first()
    raw = (10.0 ** (renders["out_peak_db"] / 20.0) / renders["output_scale"])
    return scale, raw.groupby(renders["device_id"]).max()


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())      # absolute, so it resolves from any cwd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data_mixed"), help="new data dir")
    ap.add_argument("--egdb", type=Path, default=Path("data"), help="EGDB data dir")
    ap.add_argument("--sweep", type=Path, default=Path("data_sweep"), help="sweep data dir")
    ap.add_argument("--copy", action="store_true",
                    help="copy unchanged renders instead of symlinking (~113 GB)")
    ap.add_argument("--workers", type=int, default=8, help="threads for the audio pass")
    ap.add_argument("--force", action="store_true", help="overwrite an existing data_mixed")
    args = ap.parse_args()

    out, egdb_root, sweep_root = args.out, args.egdb, args.sweep
    if (out / "manifests" / C.RENDERS_MANIFEST).is_file() and not args.force:
        raise SystemExit(f"{out} already assembled; pass --force to rebuild")

    e_corpus = _load(egdb_root, C.CORPUS_MANIFEST, manifests.CORPUS_COLUMNS)
    s_corpus = _load(sweep_root, C.CORPUS_MANIFEST, manifests.CORPUS_COLUMNS)
    e_clips = _load(egdb_root, C.CLIPS_MANIFEST, manifests.CLIPS_COLUMNS)
    s_clips = _load(sweep_root, C.CLIPS_MANIFEST, manifests.CLIPS_COLUMNS)
    e_rend = _load(egdb_root, C.RENDERS_MANIFEST, manifests.RENDERS_COLUMNS)
    s_rend = _load(sweep_root, C.RENDERS_MANIFEST, manifests.RENDERS_COLUMNS)

    overlap = set(e_corpus["file_id"]) & set(s_corpus["file_id"])
    if overlap:
        raise SystemExit(f"file_id collision between the two corpora: {sorted(overlap)}")

    e_scale, e_peak = device_scales(e_rend, str(egdb_root))
    s_scale, s_peak = device_scales(s_rend, str(sweep_root))
    if set(e_scale.index) != set(s_scale.index):
        raise SystemExit("the two corpora cover different device sets")
    # Same capture file behind both renders, or the reuse is not valid.
    nam = (e_rend.groupby("device_id")["nam_sha256"].first()
           != s_rend.groupby("device_id")["nam_sha256"].first())
    if nam.any():
        raise SystemExit(f"nam_sha256 differs on devices {sorted(nam[nam].index)}")

    devices = sorted(int(d) for d in e_scale.index)
    mixed = np.maximum(e_peak, s_peak).map(output_scale)
    ratio_e, ratio_s = mixed / e_scale, mixed / s_scale
    rescaled_egdb = sorted(int(d) for d in devices if abs(ratio_e[d] - 1.0) > UNCHANGED)
    rescaled_sweep = sorted(int(d) for d in devices if abs(ratio_s[d] - 1.0) > UNCHANGED)

    print(f"devices: {len(devices)}   files/device: {len(e_corpus)} EGDB + {len(s_corpus)} sweep "
          f"= {len(e_corpus) + len(s_corpus)}")
    print(f"EGDB renders to rescale : {len(rescaled_egdb):3d} devices x {len(e_corpus)} files "
          f"= {len(rescaled_egdb) * len(e_corpus)}")
    print(f"sweep renders to rescale: {len(rescaled_sweep):3d} devices x {len(s_corpus)} files "
          f"= {len(rescaled_sweep) * len(s_corpus)}")
    print(f"everything else: {'copied' if args.copy else 'symlinked'}\n")

    # --- Clean ------------------------------------------------------------------
    clean_len: dict[str, int] = {}
    for root, corpus, copy in ((egdb_root, e_corpus, args.copy), (sweep_root, s_corpus, True)):
        for r in corpus.itertuples(index=False):
            name = f"{r.file_id}.flac"
            src = root / "clean" / r.split / name
            link_or_copy(src, out / "clean" / r.split / name, copy=copy)
            clean_len[str(r.file_id)] = audio_io.audio_info(src)[0]
    print(f"clean: {len(clean_len)} files -> {out}/clean/ "
          f"(sweep files copied at native level, EGDB {'copied' if args.copy else 'linked'})")

    # --- Renders ----------------------------------------------------------------
    jobs: list[tuple[dict, float]] = []
    for src_rend, ratios in ((e_rend, ratio_e), (s_rend, ratio_s)):
        for r in src_rend.to_dict("records"):
            jobs.append((r, float(ratios[int(r["device_id"])])))

    written: list[dict] = []

    def do(job: tuple[dict, float]) -> dict:
        row, ratio = job
        did, fid = int(row["device_id"]), str(row["file_id"])
        src = Path(row["path"])
        dst = out / "renders" / f"{did:04d}" / f"{fid}.flac"
        scale = float(mixed[did])
        if abs(ratio - 1.0) <= UNCHANGED:
            link_or_copy(src, dst, copy=args.copy)
            new = dict(row, path=str(dst), output_scale=round(scale, 8))
        else:
            y, sr = audio_io.read_audio(src)
            y = (y * ratio).astype(np.float32)
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            if peak > C.OUTPUT_PEAK_CEILING + PEAK_TOLERANCE:
                raise SystemExit(f"device {did} {fid}: rescaled peak {peak:.6f} over ceiling")
            if len(y) != clean_len[fid]:
                raise SystemExit(f"device {did} {fid}: render {len(y)} != clean {clean_len[fid]}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            audio_io.write_flac(dst, y, sr)
            new = dict(row, path=str(dst), output_scale=round(scale, 8),
                       out_rms_db=round(rms_dbfs(y), 3), out_peak_db=round(peak_dbfs(y), 3),
                       render_sha256=sha256_file(dst))
        return {k: new[k] for k in manifests.RENDERS_COLUMNS}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, row in enumerate(ex.map(do, jobs), start=1):
            written.append(row)
            if i % 5000 == 0:
                print(f"  renders {i}/{len(jobs)}", flush=True)
    print(f"renders: {len(written)} rows -> {out}/renders/")

    # --- Manifests --------------------------------------------------------------
    md = out / "manifests"
    md.mkdir(parents=True, exist_ok=True)
    manifests.write_manifest(
        pd.concat([e_corpus, s_corpus], ignore_index=True)[manifests.CORPUS_COLUMNS],
        md / C.CORPUS_MANIFEST)
    manifests.write_manifest(
        pd.concat([e_clips, s_clips], ignore_index=True)[manifests.CLIPS_COLUMNS],
        md / C.CLIPS_MANIFEST)
    manifests.write_manifest(pd.DataFrame(written, columns=manifests.RENDERS_COLUMNS),
                             md / C.RENDERS_MANIFEST)
    for name in (C.ACQUISITION_MANIFEST, C.RENDER_SUBSET):
        shutil.copy2(egdb_root / "manifests" / name, md / name)

    (md / "rescaled_devices.txt").write_text(
        "# devices whose EGDB renders were rescaled for data_mixed (the sweep raises their\n"
        "# per-device peak, so output_scale drops). Their data/renders/ files differ in gain\n"
        "# from what a data_mixed run trains on — exclude them from cross-corpus comparisons.\n"
        + "".join(f"{d}\n" for d in rescaled_egdb), encoding="utf-8")

    holdout_src = (egdb_root / "manifests" / C.EMULATE_HOLDOUT).read_text(encoding="utf-8")
    ids = sorted({int(l) for l in holdout_src.splitlines()
                  if l.strip() and not l.lstrip().startswith("#")} | {DROP_DEVICE})
    head = [l for l in holdout_src.splitlines() if l.lstrip().startswith("#")]
    (md / C.EMULATE_HOLDOUT).write_text("\n".join(head + [
        f"# + device {DROP_DEVICE} ({DROP_DEVICE_NAME}), appended for data_mixed only: a",
        "#   near-silent capture (renders ~20 dB below the corpus) whose test ESR exceeds 1.0",
        "#   in every run. Excluded from training here, but NOT intended as an enrollment",
        "#   target -- pass `emulate-enroll --devices` explicitly to keep it out of Phase 5.",
    ] + [str(d) for d in ids]) + "\n", encoding="utf-8")
    print(f"manifests: {len(e_corpus) + len(s_corpus)} corpus rows, "
          f"{len(e_clips) + len(s_clips)} clip rows, {len(ids)} holdout ids "
          f"(table = {len(devices) - len(ids)} devices) -> {md}/")

    # --- Assertions -------------------------------------------------------------
    rd = manifests.read_manifest(md / C.RENDERS_MANIFEST, manifests.RENDERS_COLUMNS)
    n_files = len(clean_len)
    counts = rd.groupby("device_id")["file_id"].nunique()
    if not (counts == n_files).all():
        raise SystemExit(f"not every device has {n_files} render rows")
    rng = np.random.default_rng(0)
    sample = sorted(set(rescaled_egdb + rescaled_sweep) |
                    set(int(d) for d in rng.choice(devices, size=20, replace=False)))
    for d in sample:
        for fid, n in clean_len.items():
            p = out / "renders" / f"{d:04d}" / f"{fid}.flac"
            if not p.is_file():
                raise SystemExit(f"missing render {p}")
            if audio_io.audio_info(p)[0] != n:
                raise SystemExit(f"{p}: length != clean file's {n}")
    linked = rd[~rd["device_id"].isin(rescaled_egdb + rescaled_sweep)].sample(24, random_state=0)
    for r in linked.itertuples(index=False):
        if sha256_file(Path(r.path)) != r.render_sha256:
            raise SystemExit(f"{r.path}: sha256 does not match its inherited manifest row")
    # Decode the loudest file rather than reading its out_peak_db back: that column is
    # rounded to 3 dp, so quoting it would overstate the peak by more than the effect
    # being reported on.
    loudest = rd.loc[rd["out_peak_db"].idxmax()]
    worst = float(np.max(np.abs(audio_io.read_audio(Path(loudest["path"]))[0])))
    print(f"\nchecks passed: {n_files} files on every device; lengths verified on "
          f"{len(sample)} devices; sha256 re-verified on 24 linked renders")
    print(f"loudest sample in the corpus: {worst:.6f} (device {int(loudest['device_id'])} "
          f"{loudest['file_id']}) — ceiling {C.OUTPUT_PEAK_CEILING}, "
          f"+{worst - C.OUTPUT_PEAK_CEILING:.1e} from out_peak_db rounding; "
          f"clipping would be 1.0")

    print("\nnext:")
    print(f"  OPENAMP_DATA_DIR={out} openamp emulate "
          "--config configs/emulate/nam_a2_tanh_ch16_512.yaml")


if __name__ == "__main__":
    main()
