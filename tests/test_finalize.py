import pandas as pd
from conftest import make_candidate

from openamp.core import manifest
from openamp.acquire import finalize
from openamp.core.manifest import (
    STATUS_DUPLICATE,
    STATUS_REJECTED,
    STATUS_VALIDATED,
)


def _validated(model_id, make, model, **over):
    base = dict(
        tone_id=model_id, model_id=model_id, make=make, model=model,
        status=STATUS_VALIDATED, license="T3K", creator="alice",
        tone_url=f"https://t/{model_id}")
    base.update(over)
    return make_candidate(**base)


def test_finalize_assigns_stable_device_ids(settings):
    rows = [
        _validated(3, "vox", "ac30"),
        _validated(1, "fender", "deluxe"),
        _validated(2, "fender", "twin"),
        # excluded rows:
        _validated(4, "mesa", "recto", status=STATUS_DUPLICATE),
        _validated(5, "orange", "or120", creator=""),   # incomplete attribution -> rejected
        _validated(6, "peavey", "6505", status=STATUS_REJECTED),
    ]
    df = manifest.ensure_schema(pd.DataFrame(rows))

    final_df, rejected_df = finalize.finalize(df, settings)

    # Only the 3 complete validated rows are accepted.
    assert len(final_df) == 3
    assert set(final_df["device_id"]) == {0, 1, 2}
    # Ordered by (make, model, model_id): fender/deluxe, fender/twin, vox/ac30.
    assert list(final_df["model_id"]) == [1, 2, 3]

    rejected_ids = set(rejected_df["model_id"])
    assert {4, 5, 6} <= rejected_ids
    # Incomplete-license validated row lands in rejected with a reason.
    r5 = rejected_df[rejected_df["model_id"] == 5].iloc[0]
    assert r5["reject_reason"] == "incomplete_license_or_attribution"


def test_finalize_device_ids_are_reproducible(settings):
    rows = [_validated(i, f"make{i%3}", f"model{i%4}") for i in range(1, 20)]
    df = manifest.ensure_schema(pd.DataFrame(rows))
    a, _ = finalize.finalize(df.copy(), settings)
    b, _ = finalize.finalize(df.copy(), settings)
    assert list(zip(a["device_id"], a["model_id"])) == list(zip(b["device_id"], b["model_id"]))
