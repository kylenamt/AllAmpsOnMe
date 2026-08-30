"""Windows from one in-memory wet/dry recording pair.

Same item contract as :class:`~openamp.data.emulation.EmulationDataset`, but the
audio is handed in rather than read from the corpus — this is what a single
captured amp (a NAM capture signal plus its recorded response) is enrolled from.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["WetDryDataset"]


class WetDryDataset(Dataset):
    """Random aligned windows from ONE in-memory wet/dry pair.

    Same item contract as EmulationDataset. region bounds where targets are
    drawn (left-context may reach before it). Bump seed per epoch for fresh windows.
    """

    def __init__(self, dry: np.ndarray, wet: np.ndarray, *, receptive_field: int,
                 clip_samples: int, region: tuple[int, int] | None = None,
                 pairs_per_epoch: int = 500, seed: int = 1234, row: int = 0):
        if len(dry) != len(wet):
            raise ValueError("dry/wet must be sample-aligned (equal length); "
                             "use estimate_lag + trim first")
        self.dry = np.ascontiguousarray(dry, dtype=np.float32)
        self.wet = np.ascontiguousarray(wet, dtype=np.float32)
        self.receptive_field = int(receptive_field)
        self.clip_samples = int(clip_samples)
        self.region = ((0, len(dry)) if region is None
                       else (int(region[0]), int(region[1])))
        if self.region[1] - self.region[0] < self.clip_samples:
            raise ValueError(f"region {self.region} is shorter than one "
                             f"{self.clip_samples}-sample clip")
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.seed = int(seed)
        self.row = int(row)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def item_arrays(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, int(i)))
        R, clip = self.receptive_field, self.clip_samples
        lo, hi = self.region
        c = int(rng.integers(lo, hi - clip + 1))
        s, pad = c - R, 0
        if s < 0:
            pad, s = -s, 0
        inp = self.dry[s:c + clip]
        if pad:
            inp = np.concatenate([np.zeros(pad, dtype=np.float32), inp])
        return {"input": inp, "target": self.wet[c:c + clip], "device_idx": self.row}

    def __getitem__(self, i: int) -> dict:
        a = self.item_arrays(i)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(a["input"])),
            "target": torch.from_numpy(np.ascontiguousarray(a["target"])),
            "device_idx": torch.tensor(a["device_idx"], dtype=torch.long),
        }
