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
def write_corpus():
    """Factory: write a tiny clean corpus + per-device renders + their manifests.

    ``write(paths_or_spec, devices, n_files=…, samples=…, splits=…)`` — takes a
    :class:`~openamp.data.CorpusPaths`, or anything with a ``.paths`` (a
    ``RunSpec``). Each device's render
    is ``tanh(sig * (1 + device_id))``, so devices genuinely differ and a
    conditioning check has something to find. The signal stays inside +-0.95
    because the FLACs are PCM_24 and clip at +-1.0 — an unclipped fixture would
    put a different dry on disk than the renders were computed from.

    Used by tests/test_equivalence.py. test_emulate.py and test_joint.py still
    carry their own older builders; folding those in is a follow-up.
    """
    import numpy as np

    def write(target, devices, *, n_files=3, samples=24_000, splits=("train", "val"),
              sample_rate=48_000):
        from openamp.core import constants as C
        from openamp.core import manifest as manifests
        from openamp.dsp import audio as audio_io

        paths = getattr(target, "paths", target)
        sample_rate = getattr(target, "sample_rate", sample_rate)
        for d in (paths.clean_dir, paths.renders_dir,
                  paths.corpus_manifest.parent):
            d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        corpus_rows, render_rows, man_rows = [], [], []
        for split in splits:
            cdir = paths.clean_dir / split
            cdir.mkdir(parents=True, exist_ok=True)
            sigs = {}
            for f in range(n_files):
                fid = f"egdb_{split[:2]}{f:04d}"
                sig = np.clip(0.3 * rng.standard_normal(samples), -0.95, 0.95)
                sigs[fid] = sig.astype(np.float32)
                audio_io.write_flac(cdir / f"{fid}.{paths.output_format}", sigs[fid],
                                    sample_rate)
                corpus_rows.append({"file_id": fid, "split": split,
                                    "source": C.SOURCE_EGDB, "orig_path": "",
                                    "duration_s": samples / sample_rate,
                                    "applied_gain_db": 0.0, "sha256": ""})
            for d in devices:
                rdir = paths.renders_dir / f"{int(d):04d}"
                rdir.mkdir(parents=True, exist_ok=True)
                for fid, sig in sigs.items():
                    wet = np.tanh(sig * (1.0 + d)).astype(np.float32)
                    rp = rdir / f"{fid}.{paths.output_format}"
                    audio_io.write_flac(rp, wet, sample_rate)
                    render_rows.append({"device_id": d, "file_id": fid, "split": split,
                                        "path": str(rp), "status": C.RENDER_OK,
                                        "output_scale": 1.0})
        for d in devices:
            man_rows.append({"device_id": d, "tone_id": d // 2,
                             "gain_bucket": C.GAIN_BUCKETS[d % 3]})
        manifests.write_manifest(
            pd.DataFrame(corpus_rows, columns=manifests.CORPUS_COLUMNS),
            paths.corpus_manifest)
        manifests.write_manifest(
            manifests.ensure_schema(pd.DataFrame(render_rows), manifests.RENDERS_COLUMNS),
            paths.renders_manifest)
        manifests.write_manifest(
            manifests.ensure_schema(pd.DataFrame(man_rows), manifests.ALL_COLUMNS),
            paths.device_manifest)

    return write


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
