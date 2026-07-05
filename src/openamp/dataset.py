"""PyTorch ``Dataset`` classes for downstream training (spec §6).

Two map-style datasets read the Phase 2 manifests and use ``soundfile`` seek-reads
(via :func:`data.audio_io.seek_read`) so DataLoader workers stream 2 s clips
without loading whole files:

- :class:`EmulationDataset` — (clip × device) pairs for one-to-many amp emulation.
- :class:`ContrastivePairDataset` — two crops of the same device for SimCLR.

The index math and crop selection are pure functions (``build_emulation_pairs``,
``pick_two_crops``) and the array-level item builders (``*_item_arrays``) take an
injectable ``reader``, so everything but the final tensor wrap is unit-tested on
numpy — no torch required to import this module.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import audio_io
from . import constants as C
from . import manifests
from .config import Phase2Config

# Subclass torch's Dataset when available, else fall back to object so the module
# imports on a bare stack (the classes stay valid map-style datasets either way).
try:  # pragma: no cover - depends on optional dep
    from torch.utils.data import Dataset as _BaseDataset
except Exception:  # noqa: BLE001
    _BaseDataset = object

Reader = Callable[[str, int, int], np.ndarray]


# --- Pure index / crop builders (unit-tested) ----------------------------------
def build_emulation_pairs(n_clips: int, n_devices: int, pairs_per_epoch: int | None,
                          seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(clip_idx, device_idx)`` over the (clip × device) product (spec §6.1).

    With ``pairs_per_epoch`` set (and smaller than the full product) the product is
    seeded-subsampled without replacement; otherwise the full grid is returned.
    """
    total = n_clips * n_devices
    if n_clips == 0 or n_devices == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if pairs_per_epoch is None or pairs_per_epoch >= total:
        clip_idx = np.repeat(np.arange(n_clips), n_devices)
        dev_idx = np.tile(np.arange(n_devices), n_clips)
        return clip_idx.astype(np.int64), dev_idx.astype(np.int64)
    rng = np.random.default_rng(seed)
    flat = rng.choice(total, size=int(pairs_per_epoch), replace=False)
    return (flat // n_devices).astype(np.int64), (flat % n_devices).astype(np.int64)


def pick_two_crops(file_lengths: list[int], crop_samples: int, rng: np.random.Generator,
                   prefer_different_files: bool = True) -> tuple[tuple[int, int], tuple[int, int]]:
    """Two distinct ``crop_samples`` crops, preferably from different files (spec §6.2).

    ``file_lengths`` are per-file sample counts for one device. Returns
    ``((file_i, start_i), (file_j, start_j))``. Raises if no file is long enough.
    """
    eligible = [i for i, n in enumerate(file_lengths) if n >= crop_samples]
    if not eligible:
        raise ValueError("no file is long enough for a crop")

    def _start(i: int) -> int:
        hi = file_lengths[i] - crop_samples
        return int(rng.integers(0, hi + 1))

    if prefer_different_files and len(eligible) >= 2:
        i, j = rng.choice(eligible, size=2, replace=False)
        return (int(i), _start(int(i))), (int(j), _start(int(j)))

    # Same file: draw two different start positions when possible.
    i = int(rng.choice(eligible))
    hi = file_lengths[i] - crop_samples
    s1 = int(rng.integers(0, hi + 1))
    s2 = int(rng.integers(0, hi + 1))
    if hi > 0:
        while s2 == s1:
            s2 = int(rng.integers(0, hi + 1))
    return (i, s1), (i, s2)


# --- Emulation dataset ---------------------------------------------------------
class EmulationDataset(_BaseDataset):
    """Indexable over (clip × device) pairs for amp-emulation training (spec §6.1)."""

    def __init__(self, split: str, config: Phase2Config, *,
                 clips_df: pd.DataFrame | None = None,
                 renders_df: pd.DataFrame | None = None,
                 devices: list[int] | None = None,
                 pairs_per_epoch: int | None = None,
                 seed: int | None = None,
                 reader: Reader | None = None):
        self.split = split
        self.config = config
        self.reader = reader or audio_io.seek_read
        seed = config.seed if seed is None else seed

        clips = clips_df if clips_df is not None else manifests.read(
            config.clips_manifest_path, manifests.CLIPS_COLUMNS)
        renders = renders_df if renders_df is not None else manifests.read(
            config.renders_manifest_path, manifests.RENDERS_COLUMNS)

        self._clips = clips[clips["split"] == split].reset_index(drop=True)
        ok = renders[(renders["split"] == split) & (renders["status"] == C.RENDER_OK)]
        avail = sorted(int(d) for d in ok["device_id"].unique())
        if devices is not None:
            avail = [d for d in avail if d in set(devices)]
        self._devices = avail
        self._scale = {int(d): float(s) for d, s in
                       zip(ok["device_id"], ok["output_scale"])}

        self._clip_idx, self._dev_idx = build_emulation_pairs(
            len(self._clips), len(self._devices), pairs_per_epoch, seed)

    def __len__(self) -> int:
        return int(self._clip_idx.size)

    def item_arrays(self, i: int) -> dict:
        """Numpy-only item (input/target/device_id/output_scale) — no torch."""
        clip = self._clips.iloc[int(self._clip_idx[i])]
        device_id = self._devices[int(self._dev_idx[i])]
        file_id, start, num = clip["file_id"], int(clip["start_sample"]), int(clip["num_samples"])
        clean = self.config.clean_split_dir(self.split) / f"{file_id}.{self.config.output_format}"
        render = self.config.device_render_dir(device_id) / f"{file_id}.{self.config.output_format}"
        return {
            "input": np.asarray(self.reader(str(clean), start, num), dtype=np.float32),
            "target": np.asarray(self.reader(str(render), start, num), dtype=np.float32),
            "device_id": int(device_id),
            "output_scale": float(self._scale.get(int(device_id), 1.0)),
        }

    def __getitem__(self, i: int) -> dict:
        import torch  # local heavy import
        a = self.item_arrays(i)
        return {
            "input": torch.from_numpy(a["input"]),
            "target": torch.from_numpy(a["target"]),
            "device_id": torch.tensor(a["device_id"], dtype=torch.long),
            "output_scale": torch.tensor(a["output_scale"], dtype=torch.float32),
        }


# --- Contrastive dataset -------------------------------------------------------
class ContrastivePairDataset(_BaseDataset):
    """Two different crops of the same device for SimCLR (spec §6.2)."""

    def __init__(self, split: str, config: Phase2Config, *,
                 renders_df: pd.DataFrame | None = None,
                 corpus_df: pd.DataFrame | None = None,
                 devices: list[int] | None = None,
                 samples_per_device: int = 64,
                 augment_gain: bool = False,
                 seed: int | None = None,
                 reader: Reader | None = None):
        self.split = split
        self.config = config
        self.reader = reader or audio_io.seek_read
        self.augment_gain = augment_gain
        self.samples_per_device = int(samples_per_device)
        self.seed = config.seed if seed is None else seed
        self.crop_samples = config.clip_samples

        renders = renders_df if renders_df is not None else manifests.read(
            config.renders_manifest_path, manifests.RENDERS_COLUMNS)
        corpus = corpus_df if corpus_df is not None else manifests.read(
            config.corpus_manifest_path, manifests.CORPUS_COLUMNS)

        ok = renders[(renders["split"] == split) & (renders["status"] == C.RENDER_OK)]
        avail = sorted(int(d) for d in ok["device_id"].unique())
        if devices is not None:
            avail = [d for d in avail if d in set(devices)]
        self._devices = avail

        # Per-device file list (file_id, num_samples), derived from the corpus
        # durations for this split (full files are stored, so crops are free).
        split_corpus = corpus[corpus["split"] == split]
        self._files = _device_files(config, split_corpus)

    def __len__(self) -> int:
        return len(self._devices) * self.samples_per_device

    def item_arrays(self, i: int) -> dict:
        device_id = self._devices[i % len(self._devices)]
        rng = np.random.default_rng((self.seed, device_id, i))
        lengths = [n for _, n in self._files]
        (fi, si), (fj, sj) = pick_two_crops(lengths, self.crop_samples, rng)
        v1 = self._crop(device_id, self._files[fi][0], si, rng)
        v2 = self._crop(device_id, self._files[fj][0], sj, rng)
        return {"view1": v1, "view2": v2, "device_id": int(device_id)}

    def __getitem__(self, i: int) -> dict:
        import torch  # local heavy import
        a = self.item_arrays(i)
        return {
            "view1": torch.from_numpy(a["view1"]),
            "view2": torch.from_numpy(a["view2"]),
            "device_id": torch.tensor(a["device_id"], dtype=torch.long),
        }

    def _crop(self, device_id: int, file_id: str, start: int, rng) -> np.ndarray:
        path = self.config.device_render_dir(device_id) / f"{file_id}.{self.config.output_format}"
        x = np.asarray(self.reader(str(path), int(start), self.crop_samples), dtype=np.float32)
        if self.augment_gain:
            x = x * float(10.0 ** (rng.uniform(-6.0, 6.0) / 20.0))
        return x.astype(np.float32)


def _device_files(config: Phase2Config, split_corpus: pd.DataFrame) -> list[tuple[str, int]]:
    """(file_id, num_samples) per file, derived from stored duration (spec §3.2.6)."""
    files = []
    for r in split_corpus.itertuples(index=False):
        n = int(round(float(r.duration_s) * config.sample_rate))
        files.append((str(r.file_id), n))
    return files


# --- Smoke test (spec §6.3) ----------------------------------------------------
def smoke(config: Phase2Config, split: str = C.SPLIT_TRAIN, *, n_batches: int = 50,
          batch_size: int = 16, num_workers: int = 4) -> dict:
    """Build a DataLoader for both datasets, pull ``n_batches``, assert shapes/dtype/
    no-NaN, and return throughput (clips/s)."""
    import time

    import torch
    from torch.utils.data import DataLoader

    report = {}
    emu = EmulationDataset(split, config, pairs_per_epoch=n_batches * batch_size * 4)
    loader = DataLoader(emu, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    t0 = time.time()
    seen = 0
    for k, batch in enumerate(loader):
        assert batch["input"].shape[1] == config.clip_samples, batch["input"].shape
        assert batch["input"].dtype == torch.float32
        assert torch.isfinite(batch["input"]).all() and torch.isfinite(batch["target"]).all()
        seen += batch["input"].shape[0]
        if k + 1 >= n_batches:
            break
    report["emulation_clips_per_s"] = seen / max(time.time() - t0, 1e-9)

    con = ContrastivePairDataset(split, config)
    cloader = DataLoader(con, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    for k, batch in enumerate(cloader):
        assert batch["view1"].shape[1] == config.clip_samples
        assert torch.isfinite(batch["view1"]).all() and torch.isfinite(batch["view2"]).all()
        if k + 1 >= n_batches:
            break
    return report
