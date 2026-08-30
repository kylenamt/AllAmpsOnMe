"""The ESR definitions, in one place.

There are two of them, and the difference is not cosmetic:

**Pooled** (:func:`pooled_esr_terms`) accumulates error and signal energy across
every window and divides *once*. This is what training optimizes and what
``val_esr`` / ``best_val_esr`` / ``per_device_esr.csv`` report. A quiet window
contributes little to either sum, so it cannot dominate.

**Per-window** (:func:`windowed_esr`) divides inside each window and averages the
ratios afterwards, dropping near-silent windows through a
:func:`silence_floor` gate first. This is what ``comparison.csv``'s
``test_ESR_mean`` / ``test_ESR_median`` report. Without the gate a window whose
target is nearly silent has a tiny denominator and swamps the mean; with it, the
statistic answers "how does a typical window score" rather than "how much of the
total energy is error".

The two disagree on the same audio, by design. Both are here so that the four
call sites -- the two training loops, the comparison harness, and enrollment --
compute the *terms* identically and differ only in how they reduce them, which
is the part that genuinely differs:

=========================================  ==========================================
caller                                     reduction
=========================================  ==========================================
``emulate.train.evaluate_esr``             one scalar over the loader
``joint_model.train.evaluate_esr``         one scalar, plus embedding spread
``evaluation.emulate.evaluate_run``        mean and median over kept windows
``evaluation.emulate.validate_per_device``  one scalar per device
``per_device_esr`` (enrollment, probes)    one scalar per embedding row
=========================================  ==========================================
"""

from __future__ import annotations

import numpy as np
import torch

from openamp.losses import pre_emphasis

__all__ = ["SILENCE_DBFS", "silence_floor", "pooled_esr_terms", "windowed_esr",
           "per_device_esr"]

# Per-example ESR divides by target energy, so a near-silent random test window
# (tiny denominator) can dominate the *mean*. Windows below this RMS are dropped
# from the ESR stats (the corpus is normalized to -18 dBFS, so -50 dBFS is silence).
SILENCE_DBFS = -50.0


def silence_floor(clip_samples: int, dbfs: float = SILENCE_DBFS) -> float:
    """Minimum target sum-of-squares for a window to count toward per-window ESR."""
    return clip_samples * (10.0 ** (dbfs / 20.0)) ** 2


def pooled_esr_terms(pred: torch.Tensor, target: torch.Tensor, *,
                     coeff: float | None = None,
                     weights: torch.Tensor | None = None):
    """``(error_energy, signal_energy)``, both ``[B]`` -- the pooled ESR's terms.

    ``coeff`` applies pre-emphasis to both signals first (the training/val
    definition); leave it ``None`` for the raw definition the test path uses.
    ``weights`` ``[B]`` scales both terms per item, which is how
    ``device_loss_weighting`` reaches val ESR -- scaling numerator and
    denominator alike keeps the result a weighted ratio rather than a rescaled one.

    Returned unreduced so the caller can sum them to a scalar or ``index_add_``
    them into per-device buckets without the definition forking.
    """
    if coeff is not None:
        pred, target = pre_emphasis(pred, coeff), pre_emphasis(target, coeff)
    num = torch.sum((pred - target) ** 2, dim=-1)
    den = torch.sum(target ** 2, dim=-1)
    if weights is not None:
        num, den = num * weights, den * weights
    return num, den


def windowed_esr(pred: torch.Tensor, target: torch.Tensor, floor: float):
    """``(ratios, keep)``: per-window ESR for the windows above ``floor``.

    ``keep`` is the ``[B]`` boolean mask, so a caller that buckets by device can
    index its own labels with it. ``ratios`` covers only the kept windows.
    """
    t = torch.sum(target ** 2, dim=-1)
    keep = t > floor
    if not bool(keep.any()):
        return target.new_zeros(0), keep
    e = torch.sum((pred[keep] - target[keep]) ** 2, dim=-1)
    return e / (t[keep] + 1e-8), keep


@torch.no_grad()
def per_device_esr(model, loader, dev, n_rows: int, *, preemph: float | None = None,
                   silence_dbfs: float | None = None, clip: int | None = None
                   ) -> np.ndarray:
    """Per-embedding-row ESR over a loader, keyed on ``batch["device_idx"]``.

    ``preemph`` set: pooled pre-emphasized ratio (val semantics, matching
    ``evaluate_esr``). Otherwise: mean of per-window raw ratios with the
    ``silence_dbfs`` gate (test semantics, comparable to ``comparison.csv``).
    Rows with no data come back NaN.
    """
    R = model.receptive_field
    num = torch.zeros(n_rows, dtype=torch.float64, device=dev)
    den = torch.zeros(n_rows, dtype=torch.float64, device=dev)
    floor = None if preemph is not None else silence_floor(clip, silence_dbfs)
    for batch in loader:
        inp = batch["input"].to(dev, non_blocking=True)
        target = batch["target"].to(dev, non_blocking=True)
        di = batch["device_idx"].to(dev, non_blocking=True)
        out = model(inp, di)[..., R:].squeeze(1)              # [B, clip]
        if preemph is not None:
            n_i, d_i = pooled_esr_terms(out, target, coeff=preemph)
            num.index_add_(0, di, n_i.double())
            den.index_add_(0, di, d_i.double())
        else:
            ratios, keep = windowed_esr(out, target, floor)
            if ratios.numel():
                num.index_add_(0, di[keep], ratios.double())
                den.index_add_(0, di[keep], torch.ones_like(ratios).double())
    arr = (num / den).cpu().numpy()                           # 0/0 -> NaN (no data)
    arr[~np.isfinite(arr)] = np.nan
    return arr
