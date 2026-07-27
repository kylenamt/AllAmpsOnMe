"""NAM standardized-signal region math (openamp.emulate.enroll). Pure integer
geometry — no torch, checkpoint, or corpus — so it pins the split boundaries we
adopt from nam/train/core.py against silent drift."""

import pytest

pytest.importorskip("torch")   # enroll imports torch at module load

from openamp.emulate.enroll import _norm_region, nam_signal_regions

V3_LEN = 9_120_000          # 3:10 @ 48 kHz


def test_v3_split_matches_nam_source():
    r = nam_signal_regions(V3_LEN)
    assert r["version"] == "v3_0_0"
    assert r["lead_in"] == (0, 480_000)        # discarded 10 s lead-in
    assert r["train"] == (480_000, V3_LEN - 432_000)
    assert r["val"] == (V3_LEN - 432_000, V3_LEN)   # dedicated 9 s tail
    assert r["blips"] == (504_000, 552_000)    # inside the train slice
    # train and val meet with no gap; val is exactly 9 s.
    assert r["train"][1] == r["val"][0]
    assert r["val"][1] - r["val"][0] == 432_000


def test_length_tolerance_and_miss():
    # A render trimmed by a few samples still identifies as v3 (within tol)...
    assert nam_signal_regions(V3_LEN - 200) is not None
    # ...but an unrelated length (e.g. a stitched corpus DI) does not.
    assert nam_signal_regions(3_000_000) is None
    assert nam_signal_regions(V3_LEN // 2) is None


def test_non_standard_rate_is_unrecognized():
    assert nam_signal_regions(V3_LEN, sample_rate=44_100) is None


def test_forced_version_and_bad_version():
    assert nam_signal_regions(V3_LEN, version="v3_0_0")["version"] == "v3_0_0"
    with pytest.raises(ValueError):
        nam_signal_regions(V3_LEN, version="v9_9_9")


def test_norm_region_resolves_negatives_and_bounds():
    assert _norm_region((10, 20), 100) == (10, 20)
    assert _norm_region((10, -10), 100) == (10, 90)     # NAM-style tail offset
    for bad in ((-1, 10), (50, 50), (30, 20), (0, 101)):
        with pytest.raises(ValueError):
            _norm_region(bad, 100)
