"""Phase 2 render-subset selection: determinism, diversity caps, gain floor, and
the persisted id-file round-trip through ``render.parse_device_selection``."""

import math

import pandas as pd
import pytest

from data import render, subset


def make_manifest(n_makes=20, per_make=6, buckets=("clean", "crunch", "high_gain")):
    """Synthetic Phase-1-style manifest with controllable make/tone/creator spread."""
    rows = []
    did = 0
    for m in range(n_makes):
        for k in range(per_make):
            rows.append({
                "device_id": did,
                "make": f"make{m}",
                "model": f"model{m}_{k}",
                "gain_bucket": buckets[(m + k) % len(buckets)],
                "downloads": (n_makes * per_make) - did,   # strictly descending
                "tone_id": f"tone{m}_{k // 2}",            # 2 devices share a tone
                "creator_id": f"creator{m % 5}",           # 5 creators total
            })
            did += 1
    return pd.DataFrame(rows)


def test_deterministic():
    m = make_manifest()
    a = subset.select_render_subset(m, 40)
    b = subset.select_render_subset(m, 40)
    assert a == b
    assert a == sorted(a)                      # sorted, ascending
    assert len(set(a)) == len(a)               # unique
    assert set(a).issubset(set(m["device_id"]))


def test_respects_caps():
    m = make_manifest()
    ids = subset.select_render_subset(m, 60, cap_tone=1, cap_creator=8, cap_makemodel=2)
    sub = m[m["device_id"].isin(ids)]
    assert sub["tone_id"].value_counts().max() <= 1
    assert sub["creator_id"].value_counts().max() <= 8
    assert sub.groupby(["make", "model"]).size().max() <= 2


def test_gain_floor_met_when_feasible():
    # Plenty of every bucket; the 15% floor must hold for each known bucket.
    m = make_manifest(n_makes=30, per_make=6)
    size = 60
    ids = subset.select_render_subset(m, size, min_fraction=0.15,
                                      cap_tone=4, cap_creator=99, cap_makemodel=6)
    sub = m[m["device_id"].isin(ids)]
    floor = math.ceil(0.15 * size)
    counts = sub["gain_bucket"].value_counts()
    for bucket in ("clean", "crunch", "high_gain"):
        assert counts.get(bucket, 0) >= floor


def test_shortfall_when_caps_exhaust_pool():
    m = make_manifest(n_makes=10, per_make=6)   # 60 devices
    ids = subset.select_render_subset(m, 500, cap_tone=1, cap_creator=99, cap_makemodel=6)
    # cap_tone=1 over 30 tones caps the pool well under the 500 request.
    assert 0 < len(ids) <= 30


def test_larger_size_returns_more():
    m = make_manifest()
    small = subset.select_render_subset(m, 20)
    big = subset.select_render_subset(m, 60)
    assert len(big) > len(small)


def test_empty_and_zero():
    m = make_manifest()
    assert subset.select_render_subset(m, 0) == []
    assert subset.select_render_subset(m.iloc[0:0], 10) == []


def test_missing_device_id_raises():
    with pytest.raises(ValueError):
        subset.select_render_subset(pd.DataFrame({"make": ["a"]}), 5)


def test_subset_file_roundtrip(tmp_path):
    m = make_manifest()
    ids = subset.select_render_subset(m, 40)
    summary = subset.summarize(m, ids)
    path = tmp_path / "render_subset.txt"
    subset.write_subset_file(path, ids, summary, target=40,
                             caps=(4, 12, 6), min_fraction=0.15)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("#")                       # self-describing header
    # render consumes it via the "@file" syntax, ignoring comments.
    all_ids = list(m["device_id"])
    assert render.parse_device_selection(f"@{path}", all_ids) == ids


def test_summarize_counts_match():
    m = make_manifest()
    ids = subset.select_render_subset(m, 30)
    s = subset.summarize(m, ids)
    assert s["total"] == len(ids)
    assert sum(s["by_bucket"].values()) == len(ids)
