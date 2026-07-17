"""Shared pytest fixtures + helpers.

Adds ``src/`` to sys.path so tests run without an editable install, and provides
factories for a :class:`Config` and synthetic candidate frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures"

from openamp.core.config import Config  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Config:
    cfg = Config(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        publishable_key="t3k_pub_test",
        base_url="https://www.tone3000.com/api/v1",
        redirect_uri="http://localhost:3001",
        token_path=tmp_path / ".openamp_tokens.json",
        rate_limit_rpm=100000,   # effectively unthrottled for tests
        seed=1234,
    )
    cfg.ensure_dirs()
    return cfg


def make_candidate(**over) -> dict:
    """A single, fully-populated candidate row with sensible defaults."""
    row = {
        "tone_id": 1,
        "model_id": 1,
        "tone_title": "Fender Deluxe Clean",
        "model_name": "Deluxe Clean",
        "make": "fender",
        "model": "deluxe",
        "tags": ["clean"],
        "gear_type": "amp",
        "capture_type": "di",
        "architecture": "A2",
        "sample_rate": 48000,
        "license": "T3K",
        "creator": "alice",
        "creator_id": "u1",
        "downloads": 100,
        "favorites": 10,
        "created_at": "2025-01-01T00:00:00Z",
        "model_url": "https://www.tone3000.com/api/v1/files/1.nam",
        "tone_url": "https://www.tone3000.com/t/1",
        "gain_bucket": "clean",
        "raw_json": "{}",
    }
    row.update(over)
    return row


def make_candidates_frame(n_makes=6, per_make=10, buckets=("clean", "crunch", "high_gain")) -> pd.DataFrame:
    """A diverse candidate pool spanning makes / models / creators / buckets."""
    makes = ["fender", "marshall", "vox", "mesa boogie", "orange", "peavey",
             "soldano", "friedman"][:n_makes]
    rows = []
    mid = 1
    tone = 1
    for mi, make in enumerate(makes):
        for j in range(per_make):
            bucket = buckets[(mi + j) % len(buckets)]
            rows.append(make_candidate(
                tone_id=tone,
                model_id=mid,
                make=make,
                model=f"{make}-model-{j % 4}",     # a few distinct models per make
                tone_title=f"{make} tone {j}",
                model_name=f"{make} {bucket} {j}",
                creator=f"creator-{mi}",
                creator_id=f"u{mi}",
                gain_bucket=bucket,
                tags=[bucket],
                downloads=1000 - mid,
                model_url=f"https://www.tone3000.com/api/v1/files/{mid}.nam",
                tone_url=f"https://www.tone3000.com/t/{tone}",
            ))
            mid += 1
            if j % 2 == 1:   # ~2 models per tone
                tone += 1
        tone += 1
    return pd.DataFrame(rows)
