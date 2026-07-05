"""Phase 2 dataset tests: pure index/crop builders + array-level item wiring with
an injected numpy reader (no torch / soundfile needed)."""

import numpy as np
import pandas as pd
import pytest

from data import datasets
from data import constants as C
from data import manifests
from data.config import load_config


# --- pure builders -------------------------------------------------------------
def test_build_emulation_pairs_full_product():
    clip_idx, dev_idx = datasets.build_emulation_pairs(3, 4, None, seed=0)
    assert clip_idx.size == 12 and dev_idx.size == 12
    # every (clip, device) combination present exactly once
    pairs = set(zip(clip_idx.tolist(), dev_idx.tolist()))
    assert pairs == {(c, d) for c in range(3) for d in range(4)}


def test_build_emulation_pairs_subsample_deterministic():
    a = datasets.build_emulation_pairs(100, 50, 200, seed=42)
    b = datasets.build_emulation_pairs(100, 50, 200, seed=42)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[0].size == 200
    assert a[0].max() < 100 and a[1].max() < 50           # indices in range
    flat = a[0] * 50 + a[1]
    assert len(set(flat.tolist())) == 200                 # sampled without replacement


def test_build_emulation_pairs_empty():
    ci, di = datasets.build_emulation_pairs(0, 4, None, 0)
    assert ci.size == 0 and di.size == 0


def test_pick_two_crops_prefers_different_files():
    rng = np.random.default_rng(0)
    (fi, si), (fj, sj) = datasets.pick_two_crops([300_000, 300_000, 300_000],
                                                 96_000, rng, prefer_different_files=True)
    assert fi != fj
    assert 0 <= si <= 300_000 - 96_000 and 0 <= sj <= 300_000 - 96_000


def test_pick_two_crops_single_file_distinct_starts():
    rng = np.random.default_rng(0)
    (fi, si), (fj, sj) = datasets.pick_two_crops([300_000], 96_000, rng)
    assert fi == fj == 0
    assert si != sj


def test_pick_two_crops_raises_when_too_short():
    with pytest.raises(ValueError):
        datasets.pick_two_crops([1000, 2000], 96_000, np.random.default_rng(0))


# --- item wiring with injected reader -----------------------------------------
def _fake_reader_factory(record):
    def reader(path, start, num):
        record.append((path, start, num))
        return np.full(num, 0.5, dtype=np.float32)
    return reader


def test_emulation_item_arrays(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    clips = pd.DataFrame([
        {"clip_id": "egdb_0000_00000", "file_id": "egdb_0000", "split": "train",
         "start_sample": 0, "num_samples": cfg.clip_samples},
        {"clip_id": "egdb_0000_00001", "file_id": "egdb_0000", "split": "train",
         "start_sample": cfg.clip_samples, "num_samples": cfg.clip_samples},
    ], columns=manifests.CLIPS_COLUMNS)
    renders = pd.DataFrame([
        {"device_id": 0, "file_id": "egdb_0000", "split": "train",
         "status": C.RENDER_OK, "output_scale": 0.8},
        {"device_id": 5, "file_id": "egdb_0000", "split": "train",
         "status": C.RENDER_OK, "output_scale": 0.5},
    ], columns=manifests.RENDERS_COLUMNS)

    reads = []
    ds = datasets.EmulationDataset("train", cfg, clips_df=clips, renders_df=renders,
                                   reader=_fake_reader_factory(reads))
    assert len(ds) == 2 * 2                                 # 2 clips x 2 devices
    item = ds.item_arrays(0)
    assert item["input"].shape == (cfg.clip_samples,)
    assert item["target"].shape == (cfg.clip_samples,)
    assert item["device_id"] in (0, 5)
    assert item["output_scale"] in (0.8, 0.5)
    # the render path routes through the zero-padded device directory
    assert any(f"{item['device_id']:04d}" in p for p, _, _ in reads)


def test_emulation_device_subset(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    clips = pd.DataFrame([{"clip_id": "c0", "file_id": "egdb_0000", "split": "train",
                           "start_sample": 0, "num_samples": cfg.clip_samples}],
                         columns=manifests.CLIPS_COLUMNS)
    renders = pd.DataFrame([
        {"device_id": d, "file_id": "egdb_0000", "split": "train",
         "status": C.RENDER_OK, "output_scale": 1.0} for d in (0, 1, 2)
    ], columns=manifests.RENDERS_COLUMNS)
    ds = datasets.EmulationDataset("train", cfg, clips_df=clips, renders_df=renders,
                                   devices=[1], reader=_fake_reader_factory([]))
    assert len(ds) == 1
    assert ds.item_arrays(0)["device_id"] == 1


def test_contrastive_item_arrays(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    renders = pd.DataFrame([
        {"device_id": 3, "file_id": "egdb_0000", "split": "train", "status": C.RENDER_OK},
        {"device_id": 3, "file_id": "egdb_0001", "split": "train", "status": C.RENDER_OK},
    ], columns=manifests.RENDERS_COLUMNS)
    corpus = pd.DataFrame([
        {"file_id": "egdb_0000", "split": "train", "source": C.SOURCE_EGDB, "duration_s": 10.0},
        {"file_id": "egdb_0001", "split": "train", "source": C.SOURCE_EGDB, "duration_s": 10.0},
    ], columns=manifests.CORPUS_COLUMNS)

    reads = []
    ds = datasets.ContrastivePairDataset("train", cfg, renders_df=renders, corpus_df=corpus,
                                         samples_per_device=4, reader=_fake_reader_factory(reads))
    assert len(ds) == 1 * 4
    item = ds.item_arrays(0)
    assert item["view1"].shape == (cfg.clip_samples,)
    assert item["view2"].shape == (cfg.clip_samples,)
    assert item["device_id"] == 3
    assert all(f"{3:04d}" in p for p, _, _ in reads)       # both views from device 3


def test_getitem_returns_tensors_when_torch_present(tmp_path):
    """__getitem__ wraps the numpy item in torch tensors (skipped without torch)."""
    torch = pytest.importorskip("torch")
    cfg = load_config(data_dir=tmp_path)
    clips = pd.DataFrame([{"clip_id": "c0", "file_id": "egdb_0000", "split": "train",
                           "start_sample": 0, "num_samples": cfg.clip_samples}],
                         columns=manifests.CLIPS_COLUMNS)
    renders = pd.DataFrame([{"device_id": 0, "file_id": "egdb_0000", "split": "train",
                             "status": C.RENDER_OK, "output_scale": 0.9}],
                           columns=manifests.RENDERS_COLUMNS)
    ds = datasets.EmulationDataset("train", cfg, clips_df=clips, renders_df=renders,
                                   reader=_fake_reader_factory([]))
    item = ds[0]
    assert isinstance(item["input"], torch.Tensor)
    assert item["input"].dtype == torch.float32
    assert item["device_id"].dtype == torch.long
