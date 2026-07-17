"""Stage: ``openamp corpus`` — build the clean input corpus (spec §3).

Scans EGDB direct-input WAVs (+ the NAM standardized sweep), converts to 48 kHz
mono float32, level-normalizes, selects to a minutes budget, splits **by file**
(never by clip), and writes 24-bit FLAC plus ``corpus.parquet`` / ``clips.parquet``.

The signal math (budget selection, split assignment, clip grid, gain) is factored
into pure functions that unit-test on numpy/pandas without any audio backend; the
``prepare`` wrapper does the file I/O via :mod:`openamp.audio`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from openamp.dsp.audio import peak_dbfs, rms_dbfs

from openamp.dsp import audio as audio_io
from openamp.core import constants as C
from openamp.core import manifest as manifests
from openamp.core.config import Config
from openamp.core.util import sha256_file

log = logging.getLogger("openamp.corpus")


# --- Pure helpers (unit-tested) ------------------------------------------------
def compute_gain_db(signal: np.ndarray, target_rms_db: float = C.TARGET_RMS_DBFS,
                    peak_limit_db: float = C.PEAK_LIMIT_DBFS) -> float:
    """Gain (dB) to bring ``signal`` to ``target_rms_db`` RMS, then backed off so
    the true peak does not exceed ``peak_limit_db`` (spec §3.2.3).

    Returns 0.0 for silent input (nothing to normalize).
    """
    r = rms_dbfs(signal)
    p = peak_dbfs(signal)
    if not np.isfinite(r) or not np.isfinite(p):
        return 0.0
    rms_gain = target_rms_db - r
    peak_headroom = peak_limit_db - p
    return float(min(rms_gain, peak_headroom))


def apply_gain(signal: np.ndarray, gain_db: float) -> np.ndarray:
    return (np.asarray(signal, dtype=np.float32) * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def clipping_fraction(signal: np.ndarray, thresh: float = 0.999) -> float:
    """Fraction of samples at/above full scale (spec §3.2.1)."""
    x = np.asarray(signal, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= thresh))


def select_to_budget(files: pd.DataFrame, minutes_total: float, seed: int) -> pd.DataFrame:
    """Seeded-shuffle ``files`` and keep rows until their duration reaches the
    budget (spec §3.2.4). Returns the selected rows (in shuffled order).

    ``files`` needs a ``duration_s`` column. Deterministic given ``seed``.
    """
    if files.empty:
        return files.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    shuffled = files.iloc[order].reset_index(drop=True)
    budget_s = minutes_total * 60.0
    cum = shuffled["duration_s"].cumsum()
    # Keep files until we first reach/exceed the budget (at least one file).
    n_keep = int((cum < budget_s).sum()) + 1
    n_keep = min(max(n_keep, 1), len(shuffled))
    return shuffled.iloc[:n_keep].reset_index(drop=True)


def assign_splits(files: pd.DataFrame, ratios: dict, seed: int) -> pd.DataFrame:
    """Assign a ``split`` to every file **by file, by duration** (spec §3.2.5).

    Rules: the sweep source goes to ``train`` only; ``test`` is EGDB (realistic
    playing) only; EGDB is split ~train/val/test by cumulative duration. Returns a
    copy of ``files`` with a filled ``split`` column. Deterministic given ``seed``.
    """
    out = files.copy().reset_index(drop=True)
    out["split"] = pd.NA

    is_sweep = out["source"] == C.SOURCE_SWEEP
    out.loc[is_sweep, "split"] = C.SPLIT_TRAIN

    egdb = out[~is_sweep]
    if not egdb.empty:
        rng = np.random.default_rng(seed + 1)  # distinct stream from budget shuffle
        order = rng.permutation(egdb.index.to_numpy())
        durs = out.loc[order, "duration_s"].to_numpy(dtype=float)
        total = float(durs.sum())
        cum = np.cumsum(durs)
        train_end = ratios[C.SPLIT_TRAIN] * total
        val_end = (ratios[C.SPLIT_TRAIN] + ratios[C.SPLIT_VAL]) * total
        for idx, c in zip(order, cum):
            if c <= train_end:
                out.at[idx, "split"] = C.SPLIT_TRAIN
            elif c <= val_end:
                out.at[idx, "split"] = C.SPLIT_VAL
            else:
                out.at[idx, "split"] = C.SPLIT_TEST
    return out


def clip_grid(num_samples: int, clip_samples: int) -> list[tuple[int, int]]:
    """Non-overlapping clip boundaries; the trailing remainder is dropped
    (spec §3.2.7). Returns ``[(start_sample, num_samples), ...]``."""
    n = int(num_samples) // int(clip_samples)
    return [(k * clip_samples, clip_samples) for k in range(n)]


# --- IO wrapper ----------------------------------------------------------------
class CorpusInputError(RuntimeError):
    """Raised when required raw inputs are missing."""


def _scan_egdb(config: Config) -> list[dict]:
    """Cheap metadata scan of EGDB WAVs (no full decode); reject too-short files."""
    egdb_dir = config.egdb_dir
    wavs = sorted(p for p in egdb_dir.rglob("*.wav")) if egdb_dir.is_dir() else []
    rows = []
    for i, path in enumerate(wavs):
        try:
            frames, sr = audio_io.audio_info(path)
        except Exception as exc:  # pragma: no cover - depends on file/backend
            log.warning("skip %s: %s", path.name, exc)
            continue
        duration_s = frames / float(sr) if sr else 0.0
        if duration_s < C.MIN_SOURCE_SECONDS:
            continue
        rows.append({"file_id": f"egdb_{i:04d}", "source": C.SOURCE_EGDB,
                     "orig_path": str(path), "duration_s": duration_s})
    return rows


def _missing_inputs_message(config: Config) -> str:
    return (
        "Corpus inputs are missing. Place them and re-run:\n"
        f"  1. EGDB direct-input WAVs under: {config.egdb_dir}\n"
        "     (download from the EGDB GitHub release; use only the clean DI audio)\n"
        f"  2. NAM sweep signal at:          {config.nam_sweep_path}\n"
        "     (v3_0_0.wav from the neural-amp-modeler training resources)\n"
        "See README.md for the exact links and steps."
    )


def prepare(config: Config, *, force: bool = False) -> pd.DataFrame:
    """Build the clean corpus + clip grid. Idempotent: a completed corpus with all
    clean files present is a no-op unless ``force`` (spec §8.5)."""
    config.ensure_dirs()

    if not force and _corpus_complete(config):
        log.info("corpus already prepared -> %s", config.corpus_manifest_path)
        return manifests.read_manifest(config.corpus_manifest_path, manifests.CORPUS_COLUMNS)

    egdb_rows = _scan_egdb(config)
    have_sweep = config.nam_sweep_path.is_file()
    if not egdb_rows and not have_sweep:
        raise CorpusInputError(_missing_inputs_message(config))
    if not egdb_rows:
        log.warning("no EGDB files found under %s; corpus will be sweep-only", config.egdb_dir)

    egdb_df = pd.DataFrame(egdb_rows, columns=["file_id", "source", "orig_path", "duration_s"])
    selected = select_to_budget(egdb_df, config.minutes_total, config.seed) if not egdb_df.empty else egdb_df

    files = selected.copy()
    if have_sweep:
        sweep_frames, sweep_sr = audio_io.audio_info(config.nam_sweep_path)
        files = pd.concat([files, pd.DataFrame([{
            "file_id": C.SOURCE_SWEEP, "source": C.SOURCE_SWEEP,
            "orig_path": str(config.nam_sweep_path),
            "duration_s": sweep_frames / float(sweep_sr) if sweep_sr else 0.0,
        }])], ignore_index=True)

    files = assign_splits(files, config.split_ratios, config.seed)

    corpus_rows: list[dict] = []
    clip_rows: list[dict] = []
    for row in files.itertuples(index=False):
        result = _process_one(config, row._asdict())
        if result is None:
            continue
        corpus_rows.append(result["corpus"])
        clip_rows.extend(result["clips"])

    corpus_df = pd.DataFrame(corpus_rows, columns=manifests.CORPUS_COLUMNS)
    clips_df = pd.DataFrame(clip_rows, columns=manifests.CLIPS_COLUMNS)
    manifests.write_manifest(corpus_df, config.corpus_manifest_path)
    manifests.write_manifest(clips_df, config.clips_manifest_path)
    log.info("corpus: %d files (%d clips) -> %s", len(corpus_df), len(clips_df),
             config.corpus_manifest_path)
    return corpus_df


def _process_one(config: Config, row: dict) -> dict | None:
    """Read → resample → normalize → write one clean file; build manifest rows."""
    orig_path = row["orig_path"]
    signal, sr = audio_io.read_audio(orig_path)
    if row["source"] == C.SOURCE_EGDB and clipping_fraction(signal) > C.CLIP_FRACTION_MAX:
        log.warning("reject %s: clipping", Path(orig_path).name)
        return None

    signal = audio_io.resample(signal, sr, config.sample_rate)
    gain_db = compute_gain_db(signal)
    signal = apply_gain(signal, gain_db)

    split = row["split"]
    out_path = config.clean_split_dir(split) / f"{row['file_id']}.{config.output_format}"
    audio_io.write_flac(out_path, signal, config.sample_rate)

    num_samples = len(signal)
    corpus = {
        "file_id": row["file_id"], "split": split, "source": row["source"],
        "orig_path": orig_path, "duration_s": num_samples / config.sample_rate,
        "applied_gain_db": round(gain_db, 4), "sha256": sha256_file(out_path),
    }
    clips = [{
        "clip_id": f"{row['file_id']}_{k:05d}", "file_id": row["file_id"],
        "split": split, "start_sample": start, "num_samples": n,
    } for k, (start, n) in enumerate(clip_grid(num_samples, config.clip_samples))]
    return {"corpus": corpus, "clips": clips}


def _corpus_complete(config: Config) -> bool:
    if not config.corpus_manifest_path.is_file() or not config.clips_manifest_path.is_file():
        return False
    df = manifests.read_manifest(config.corpus_manifest_path, manifests.CORPUS_COLUMNS)
    if df.empty:
        return False
    for r in df.itertuples(index=False):
        if not (config.clean_split_dir(r.split) / f"{r.file_id}.{config.output_format}").is_file():
            return False
    return True
