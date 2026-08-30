"""DataLoader construction, in one place.

The generator is seeded explicitly rather than left to torch's global RNG: it is
what makes an epoch's shuffle reproducible from ``(seed, epoch)`` alone, and the
trainers rebuild a loader every epoch with a bumped seed to get fresh windows.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

__all__ = ["make_loader"]


def make_loader(ds, batch_size, workers, *, shuffle, seed, drop_last) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      drop_last=drop_last, pin_memory=True, generator=gen,
                      persistent_workers=False)
