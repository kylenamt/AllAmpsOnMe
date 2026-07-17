"""Corpus tests: level normalization, budget selection, split integrity,
and the clip grid — all pure numpy/pandas, no audio backend."""

import numpy as np
import pandas as pd

from openamp.corpus import build as corpus
from openamp.core import constants as C
from openamp.dsp.audio import peak_dbfs, rms_dbfs


def test_compute_gain_hits_target_rms_when_unclipped():
    sr = 48_000
    t = np.arange(sr) / sr
    x = (0.01 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # quiet -> peak won't bind
    g = corpus.compute_gain_db(x)
    y = corpus.apply_gain(x, g)
    assert abs(rms_dbfs(y) - C.TARGET_RMS_DBFS) < 0.1
    assert peak_dbfs(y) < C.PEAK_LIMIT_DBFS


def test_compute_gain_backs_off_to_peak_limit():
    x = np.full(48_000, 1e-3, np.float32)
    x[0] = 0.5                                   # huge crest factor -> peak binds first
    g = corpus.compute_gain_db(x)
    y = corpus.apply_gain(x, g)
    assert abs(peak_dbfs(y) - C.PEAK_LIMIT_DBFS) < 0.05
    assert rms_dbfs(y) < C.TARGET_RMS_DBFS       # RMS stayed below target (backed off)


def test_compute_gain_silence_is_noop():
    assert corpus.compute_gain_db(np.zeros(1000, np.float32)) == 0.0


def test_select_to_budget_reaches_budget_and_is_deterministic():
    files = pd.DataFrame({"file_id": [f"egdb_{i:04d}" for i in range(100)],
                          "duration_s": [30.0] * 100})
    a = corpus.select_to_budget(files, minutes_total=5.0, seed=1234)
    b = corpus.select_to_budget(files, minutes_total=5.0, seed=1234)
    assert a["file_id"].tolist() == b["file_id"].tolist()          # deterministic
    assert a["duration_s"].sum() >= 5.0 * 60                        # covers the budget
    assert len(a) <= 12                                            # ~10 files, not all 100


def test_select_to_budget_caps_at_available():
    files = pd.DataFrame({"file_id": ["a", "b"], "duration_s": [10.0, 10.0]})
    got = corpus.select_to_budget(files, minutes_total=999.0, seed=1)
    assert len(got) == 2


def test_assign_splits_integrity():
    rows = [{"file_id": f"egdb_{i:04d}", "source": C.SOURCE_EGDB, "duration_s": 20.0}
            for i in range(50)]
    rows.append({"file_id": "sweep", "source": C.SOURCE_SWEEP, "duration_s": 180.0})
    files = pd.DataFrame(rows)
    out = corpus.assign_splits(files, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=1234)

    # Every file assigned exactly one split; no file in two splits.
    assert out["split"].notna().all()
    assert len(out) == out["file_id"].nunique()
    # Sweep -> train only; test -> EGDB only.
    assert out.loc[out["source"] == C.SOURCE_SWEEP, "split"].tolist() == [C.SPLIT_TRAIN]
    test_sources = set(out.loc[out["split"] == C.SPLIT_TEST, "source"])
    assert C.SOURCE_SWEEP not in test_sources
    # Rough duration split for EGDB (train is the largest bucket).
    egdb = out[out["source"] == C.SOURCE_EGDB]
    by = egdb.groupby("split")["duration_s"].sum()
    assert by.get(C.SPLIT_TRAIN, 0) > by.get(C.SPLIT_VAL, 0)
    assert by.get(C.SPLIT_TRAIN, 0) > by.get(C.SPLIT_TEST, 0)


def test_assign_splits_deterministic():
    files = pd.DataFrame([{"file_id": f"egdb_{i}", "source": C.SOURCE_EGDB,
                           "duration_s": 20.0} for i in range(30)])
    r = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = corpus.assign_splits(files, r, seed=7)["split"].tolist()
    b = corpus.assign_splits(files, r, seed=7)["split"].tolist()
    assert a == b


def test_clip_grid_drops_remainder():
    grid = corpus.clip_grid(3 * 96_000 + 500, 96_000)
    assert grid == [(0, 96_000), (96_000, 96_000), (192_000, 96_000)]
    assert corpus.clip_grid(50_000, 96_000) == []   # shorter than one clip


def test_clipping_fraction():
    x = np.zeros(1000, np.float32)
    x[:20] = 1.0
    assert abs(corpus.clipping_fraction(x) - 0.02) < 1e-9
