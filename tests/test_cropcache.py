"""Crop pool: fast dataset over a hand-written memmap + per-device extraction."""

import json

import numpy as np
import pytest


def _write_pool(out_dir, split, pool):
    n, k, m = pool.shape
    pool.astype(np.int16).tofile(out_dir / f"{split}_pool.i16")
    np.save(out_dir / f"{split}_devices.npy", np.arange(n, dtype=np.int64) + 100)
    (out_dir / f"{split}_meta.json").write_text(json.dumps(
        {"split": split, "crops_per_device": k, "crop_samples": m,
         "seed": 0, "n_devices": n, "sample_rate": 48000}))


def test_crop_pool_dataset_yields_two_distinct_crops(tmp_path):
    pytest.importorskip("torch")
    from train.cropcache import CropPoolDataset
    # 2 devices, 4 distinct crops each (constant per crop so we can identify them).
    pool = np.zeros((2, 4, 10), dtype=np.int16)
    for d in range(2):
        for c in range(4):
            pool[d, c] = (d * 4 + c + 1) * 1000
    _write_pool(tmp_path, "train", pool)

    ds = CropPoolDataset(tmp_path, "train", samples_per_device=3, augment_gain=False, seed=0)
    assert len(ds) == 2 * 3
    a = ds.item_arrays(0)
    assert a["view1"].shape == (10,) and a["view2"].shape == (10,)
    assert a["device_id"] == 100                       # first device id
    # crops are distinct values (k>=2 -> replace=False)
    assert not np.allclose(a["view1"], a["view2"])
    # values are the dequantized int16 constants -> within [-1, 1]
    assert np.abs(a["view1"]).max() <= 1.0


def test_crop_pool_augment_changes_scale(tmp_path):
    pytest.importorskip("torch")
    from train.cropcache import CropPoolDataset
    pool = np.full((1, 4, 8), 5000, dtype=np.int16)
    _write_pool(tmp_path, "train", pool)
    plain = CropPoolDataset(tmp_path, "train", augment_gain=False, seed=1).item_arrays(0)
    aug = CropPoolDataset(tmp_path, "train", augment_gain=True, seed=1).item_arrays(0)
    # same underlying crop, different gain -> different magnitude
    assert not np.allclose(plain["view1"], aug["view1"])


def test_extract_device_crops_uses_reader(monkeypatch):
    from data import audio_io
    from train import cropcache

    # A ramp signal long enough for several crops.
    sig = np.linspace(-0.5, 0.5, 5000, dtype=np.float32)
    monkeypatch.setattr(audio_io, "read_audio", lambda p: (sig, 48000))

    crops = cropcache._extract_device_crops(
        ["f0.flac", "f1.flac", "f2.flac"], crops_per_device=6, crop_samples=100,
        rng=np.random.default_rng(0))
    assert crops.shape == (6, 100)
    assert crops.dtype == np.int16
    assert np.abs(crops).max() <= cropcache._SCALE
