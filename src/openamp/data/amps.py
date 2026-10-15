"""Per-amp clip bundles: one item is one amp, not one window.

The unfolder steps each amp on a batch of its *own* clips and scores the result
on a second, disjoint batch of that same amp -- "the recording you enroll from",
then new audio. A window-per-item dataset cannot express that grouping, so this
one sits outside the ``input``/``target``/``device_idx`` item contract and is
not in :data:`openamp.data.DATASETS`.

Item ``j`` is ``2 * n`` independent window draws of amp ``ids[j]``:

=========  ====================  ===============================================
key        shape                 meaning
=========  ====================  ===============================================
``xf``     ``[n, R + clip]``     clean, with left-context -- the clips to fit on
``tf``     ``[n, clip]``         their render
``xs``     ``[n, R + clip]``     clean -- the clips to score on
``ts``     ``[n, clip]``         their render
``amp``    ``[]`` int64          ``j``, so the caller can look the id up
=========  ====================  ===============================================

Windows are drawn exactly as :class:`~openamp.data.EmulationDataset` draws them
(its ``_read_pair``, NFS retry included), and every item is a pure function of
``(seed, epoch, j)``.

Fit and score clips may come from different datasets (``score_base``): another
split, or another corpus entirely -- e.g. fit on a capture file's train region,
score on its val region. ``gains`` scales each amp's *targets*, for a corpus
whose renders were normalized to a different peak than the one a generator
trained on.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from openamp.data.emulation import EmulationDataset, read_with_retry

__all__ = ["AmpClipsDataset"]


class AmpClipsDataset(Dataset):
    """``n`` fit clips from ``base`` and ``n`` score clips from ``score_base`` per amp."""

    def __init__(self, base: EmulationDataset, ids, n: int, *, seed: int,
                 score_base: EmulationDataset | None = None,
                 gains: dict[int, float] | None = None):
        if int(n) < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.base, self.ids, self.n = base, [int(d) for d in ids], int(n)
        self.score_base = score_base if score_base is not None else base
        self.gains = {int(k): float(v) for k, v in (gains or {}).items()}
        self.seed, self.epoch = int(seed), 0     # bump epoch for fresh windows

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, j: int) -> dict:
        d = self.ids[j]
        rng = np.random.default_rng((self.seed, self.epoch, int(j)))

        def one(ds: EmulationDataset):
            def draw():
                f = ds.files[int(rng.integers(0, len(ds.files)))]
                return f, f"device={d} file={f['file_id']}"
            return read_with_retry(draw, lambda f: ds._read_pair(d, f, rng), split=ds.split)

        pairs = ([one(self.base) for _ in range(self.n)]
                 + [one(self.score_base) for _ in range(self.n)])
        x = torch.from_numpy(np.stack([p[0] for p in pairs]))
        t = torch.from_numpy(np.stack([p[1] for p in pairs])) * self.gains.get(d, 1.0)
        n = self.n
        return {"xf": x[:n], "tf": t[:n], "xs": x[n:], "ts": t[n:],
                "amp": torch.tensor(int(j))}
