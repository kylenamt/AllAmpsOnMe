
"""Precomputed in-RAM crop pool for fast contrastive training (Phase 3).

The Phase 2 renders live on NFS as 39 600 FLACs; random 2 s seek-reads cost
~10 ms each and don't parallelize, capping training at ~100 samples/s (data
bound) while the GPU can do ~3000/s. This module decodes a fixed pool of
``crops_per_device`` random 2 s crops per device **once** into a single int16
memmap on local disk, then serves training from it at RAM speed.

- :func:`build_crop_pool` — decode renders -> ``pool.i16`` memmap + ``meta.json``.
- :class:`CropPoolDataset` — drop-in for ``ContrastivePairDataset``: yields two
  distinct crops of the same device (``view1``/``view2``/``device_id``).

Crops keep the "different positions / files" property of the spec (they are drawn
across a random subset of each device's files); a training step then samples two
of the pooled crops, so the encoder still sees varied content per device. int16
(vs the 24-bit renders) costs ~96 dB SNR — inaudible for representation learning.
"""

from __future__ import annotations

import getpass
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from data import constants as C
from data import manifests

_I16 = np.int16
_SCALE = 32767.0


def default_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / f"oa3k_phase3_cache_{getpass.getuser()}"


def _device_paths(config, split: str) -> dict[int, list[str]]:
    """{device_id: [render_path, ...]} for successfully rendered files of a split."""
    renders = manifests.read(config.renders_manifest_path, manifests.RENDERS_COLUMNS)
    ok = renders[(renders["split"] == split) & (renders["status"] == C.RENDER_OK)]
    out: dict[int, list[str]] = {}
    for dev, path in zip(ok["device_id"], ok["path"]):
        out.setdefault(int(dev), []).append(str(path))
    return out


def _extract_device_crops(paths: list[str], crops_per_device: int, crop_samples: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Return ``[crops_per_device, crop_samples]`` int16 crops for one device.

    Spreads crops over a random subset of the device's files (reading each chosen
    file once) and takes a random window from each; falls back to same-file
    windows when a device has few files.
    """
    from data import audio_io

    paths = list(paths)
    rng.shuffle(paths)
    # crops-per-file ~4 so ~16 files cover 64 crops; never need more files than we have.
    per_file = 4
    n_files = min(len(paths), max(1, (crops_per_device + per_file - 1) // per_file))
    chosen = paths[:n_files]
    # Distribute crop count across the chosen files as evenly as possible.
    counts = [crops_per_device // n_files] * n_files
    for k in range(crops_per_device - sum(counts)):
        counts[k] += 1

    crops = np.empty((crops_per_device, crop_samples), dtype=_I16)
    w = 0
    for path, cnt in zip(chosen, counts):
        sig, _ = audio_io.read_audio(path)          # whole-file decode (fast, sequential)
        n = len(sig)
        hi = max(n - crop_samples, 0)
        for _ in range(cnt):
            s = int(rng.integers(0, hi + 1)) if hi > 0 else 0
            win = sig[s:s + crop_samples]
            if len(win) < crop_samples:
                win = np.pad(win, (0, crop_samples - len(win)))
            crops[w] = np.clip(win, -1.0, 1.0) * _SCALE
            w += 1
    return crops


def build_crop_pool(config, split: str, *, crops_per_device: int, crop_samples: int,
                    out_dir: Path, seed: int, workers: int = 8,
                    force: bool = False) -> dict:
    """Decode a per-device crop pool for ``split`` into ``out_dir`` (idempotent)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{split}_meta.json"
    pool_path = out_dir / f"{split}_pool.i16"
    dev_path = out_dir / f"{split}_devices.npy"

    want = {"split": split, "crops_per_device": int(crops_per_device),
            "crop_samples": int(crop_samples), "seed": int(seed)}
    if not force and meta_path.is_file():
        have = json.loads(meta_path.read_text())
        if {k: have.get(k) for k in want} == want and pool_path.is_file() and dev_path.is_file():
            return have

    dev_paths = _device_paths(config, split)
    devices = sorted(dev_paths)
    n = len(devices)
    pool = np.memmap(pool_path, dtype=_I16, mode="w+",
                     shape=(n, crops_per_device, crop_samples))

    def work(idx_dev):
        idx, dev = idx_dev
        rng = np.random.default_rng((seed, dev))
        pool[idx] = _extract_device_crops(dev_paths[dev], crops_per_device,
                                          crop_samples, rng)
        return idx

    with ThreadPoolExecutor(max_workers=workers) as ex:
        done = 0
        for _ in ex.map(work, enumerate(devices)):
            done += 1
            if done % 50 == 0 or done == n:
                print(f"[cropcache:{split}] {done}/{n} devices", flush=True)
    pool.flush()
    del pool
    np.save(dev_path, np.asarray(devices, dtype=np.int64))
    meta = {**want, "n_devices": n, "sample_rate": config.sample_rate}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[cropcache:{split}] wrote {pool_path} "
          f"({n}x{crops_per_device}x{crop_samples} int16, "
          f"{pool_path.stat().st_size / 1e9:.2f} GB)")
    return meta


# --- Fast dataset over the pool ------------------------------------------------
try:  # pragma: no cover - optional dep
    from torch.utils.data import Dataset as _BaseDataset
except Exception:  # noqa: BLE001
    _BaseDataset = object


class CropPoolDataset(_BaseDataset):
    """Two distinct crops of the same device, served from the int16 pool memmap."""

    def __init__(self, out_dir: Path, split: str, *, samples_per_device: int = 64,
                 augment_gain: bool = False, seed: int = 1234,
                 out_samples: int | None = None):
        out_dir = Path(out_dir)
        self.meta = json.loads((out_dir / f"{split}_meta.json").read_text())
        self.devices = np.load(out_dir / f"{split}_devices.npy")
        self.crop_samples = int(self.meta["crop_samples"])   # stored segment length
        self.k = int(self.meta["crops_per_device"])
        self._pool_path = out_dir / f"{split}_pool.i16"
        self._shape = (len(self.devices), self.k, self.crop_samples)
        self._pool = None                        # lazily memmapped per worker
        self.samples_per_device = int(samples_per_device)
        self.augment_gain = bool(augment_gain)
        self.seed = int(seed)
        # Output crop length; when stored segments are longer we random-sub-crop
        # each step, restoring continuous position diversity the fixed pool lost.
        self.out_samples = int(out_samples) if out_samples else self.crop_samples

    def _ensure_pool(self):
        if self._pool is None:
            self._pool = np.memmap(self._pool_path, dtype=_I16, mode="r",
                                   shape=self._shape)
        return self._pool

    def __len__(self) -> int:
        return len(self.devices) * self.samples_per_device

    def _view(self, seg, rng) -> np.ndarray:
        n = self.out_samples
        if len(seg) > n:                         # random 2 s window from a longer segment
            s = int(rng.integers(0, len(seg) - n + 1))
            seg = seg[s:s + n]
        x = seg.astype(np.float32) / (_SCALE + 1.0)
        if self.augment_gain:
            if rng.random() < 0.5:               # random polarity (effect-invariant)
                x = -x
            x = x * np.float32(10.0 ** (rng.uniform(-6.0, 6.0) / 20.0))   # +/-6 dB gain
        return np.ascontiguousarray(x)

    def item_arrays(self, i: int) -> dict:
        pool = self._ensure_pool()
        d = i % len(self.devices)
        rng = np.random.default_rng((self.seed, int(self.devices[d]), i))
        a, b = rng.choice(self.k, size=2, replace=(self.k < 2))
        v1 = self._view(pool[d, a], rng)
        v2 = self._view(pool[d, b], rng)
        return {"view1": v1, "view2": v2, "device_id": int(self.devices[d])}

    def __getitem__(self, i: int) -> dict:
        import torch

        a = self.item_arrays(i)
        return {
            "view1": torch.from_numpy(np.ascontiguousarray(a["view1"])),
            "view2": torch.from_numpy(np.ascontiguousarray(a["view2"])),
            "device_id": torch.tensor(a["device_id"], dtype=torch.long),
        }
