from collections import Counter

from conftest import make_candidates_frame

from t3k import select
from t3k.select import MAX_PER_CREATOR, MAX_PER_MAKEMODEL, MAX_PER_TONE


def test_selection_respects_caps(settings):
    cands = make_candidates_frame(n_makes=8, per_make=12)
    out = select.select(cands, settings, target=40)

    assert 0 < len(out) <= 40
    tone_c = Counter(out["tone_id"])
    creator_c = Counter(out["creator_id"])
    mm_c = Counter(zip(out["make"], out["model"]))
    assert max(tone_c.values()) <= MAX_PER_TONE
    assert max(creator_c.values()) <= MAX_PER_CREATOR
    assert max(mm_c.values()) <= MAX_PER_MAKEMODEL


def test_selection_is_deterministic(settings):
    cands = make_candidates_frame(n_makes=8, per_make=12)
    a = select.select(cands, settings, target=40, seed=7)
    b = select.select(cands, settings, target=40, seed=7)
    assert list(a["model_id"]) == list(b["model_id"])


def test_selection_hits_gain_minimums(settings):
    cands = make_candidates_frame(n_makes=8, per_make=12)
    out = select.select(cands, settings, target=40)
    total = len(out)
    counts = out["gain_bucket"].value_counts().to_dict()
    for bucket in ("clean", "crunch", "high_gain"):
        assert counts.get(bucket, 0) / total >= 0.15  # comfortably above acceptance floor


def test_hard_filters_drop_ineligible(settings):
    cands = make_candidates_frame(n_makes=4, per_make=6)
    # Break eligibility on a few rows.
    cands.loc[0, "gear_type"] = "full-rig"
    cands.loc[1, "model_url"] = ""
    cands.loc[2, "license"] = "All Rights Reserved"
    cands.loc[3, "architecture"] = "A0"
    out = select.select(cands, settings, target=100)
    chosen = set(out["model_id"])
    for bad in cands.loc[[0, 1, 2, 3], "model_id"]:
        assert bad not in chosen


def test_exclude_ids(settings):
    cands = make_candidates_frame(n_makes=4, per_make=6)
    exclude = set(cands["model_id"].iloc[:5])
    out = select.select(cands, settings, target=100, exclude_ids=exclude)
    assert not (set(out["model_id"]) & exclude)
