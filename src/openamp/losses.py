"""Training objectives for the amp models.

An objective is a callable::

    (pred [B, 1, T], target [B, 1, T], weights [B] | None) -> (loss, parts)

where ``parts`` is a dict of float terms the trainers write to ``train_log.csv``.
Add one by writing a class here (or in its own module) and decorating it with
``@register_loss("name")``; the trainers reach it through :func:`build_loss`, so
a new objective never means editing a training loop.

Split out of ``emulate/train.py`` unchanged, so the numbers are identical to
every run made before the split. :func:`device_loss_weights` takes the renders
manifest path rather than a whole ``Config``: it is the one part of this module
that touches the corpus, and it only ever needed that one path.
"""

from __future__ import annotations

from pathlib import Path

import auraloss
import numpy as np
import torch
from torch import nn

from openamp.core import constants as C
from openamp.core import manifest as manifests

__all__ = ["pre_emphasis", "esr", "preemph_esr", "EmulationLoss",
           "device_loss_weights", "build_loss", "register_loss", "LOSSES",
           "DEFAULT_LOSS"]

DEFAULT_LOSS = "esr_mrstft"

LOSSES: dict[str, type] = {}


def register_loss(name: str):
    """Register an objective class under a config-facing name."""
    def wrap(cls):
        LOSSES[name] = cls
        return cls
    return wrap


def build_loss(name: str, **kwargs):
    """Construct a registered objective."""
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; known: {sorted(LOSSES)}")
    return LOSSES[name](**kwargs)


# --- Terms ----------------------------------------------------------------------
def pre_emphasis(x: torch.Tensor, coeff: float) -> torch.Tensor:
    """1st-order pre-emphasis ``y[n] = x[n] - coeff * x[n-1]`` (last-dim FIR)."""
    return x[..., 1:] - coeff * x[..., :-1]


def esr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8,
        weights: torch.Tensor | None = None) -> torch.Tensor:
    """Energy-weighted error-to-signal ratio ``sum||e||^2 / sum||t||^2`` over the batch.

    Pooled as one ratio of sums, not a mean of per-window ratios: window target
    energy spans ~8 orders of magnitude (near-silent passages in clean guitar),
    so per-window ratios would let a handful of quiet windows dominate the batch.

    ``weights`` [B] scales each item's numerator and denominator before the sums
    -- identical to scaling that item's audio by ``sqrt(w)``, but confined to
    this term. It is how :func:`device_loss_weights` offsets the corpus's output-
    level spread without reintroducing the per-window blow-up above. ``None``
    (the default) leaves the pooling untouched.
    """
    num = torch.sum((pred - target) ** 2, dim=-1)
    den = torch.sum(target ** 2, dim=-1)
    if weights is not None:
        w = weights.reshape(-1, *([1] * (num.dim() - 1))).to(num.dtype)
        num, den = num * w, den * w
    return num.sum() / (den.sum() + eps)


def preemph_esr(pred: torch.Tensor, target: torch.Tensor, coeff: float,
                eps: float = 1e-8,
                weights: torch.Tensor | None = None) -> torch.Tensor:
    """ESR on pre-emphasized signals — weights the high end amps distort most."""
    return esr(pre_emphasis(pred, coeff), pre_emphasis(target, coeff), eps,
               weights=weights)


# --- Objectives -------------------------------------------------------------------
@register_loss(DEFAULT_LOSS)
class EmulationLoss(nn.Module):
    """Pre-emphasized ESR + multi-resolution STFT (auraloss), summed 1:1 (spec §4.2)."""

    parts = ("esr", "stft")

    def __init__(self, preemph_coeff: float = 0.85, stft_weight: float = 1.0):
        super().__init__()
        self.coeff = float(preemph_coeff)
        self.stft_weight = float(stft_weight)
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weights: torch.Tensor | None = None):
        """``pred``/``target`` are ``[B, 1, T]``; returns ``(loss, parts)``.

        ``weights`` [B] reaches the ESR term only (see :func:`esr`). The MRSTFT
        term is deliberately left unweighted -- see :func:`device_loss_weights`.
        """
        esr_term = preemph_esr(pred, target, self.coeff, weights=weights)
        stft_term = self.mrstft(pred, target)
        loss = esr_term + self.stft_weight * stft_term
        return loss, {"esr": float(esr_term.item()), "stft": float(stft_term.item())}


# --- Per-device weighting -----------------------------------------------------------
def device_loss_weights(renders_manifest: Path, ids: list[int],
                        strength: float) -> np.ndarray:
    """Per-device ESR weights that offset the corpus's output-level spread.

    The ESR term pools error and signal energy over the whole batch
    (:func:`esr`), and devices are drawn uniformly, so a device's share of the
    gradient tracks its own render power. Measured on data_mixed: median render
    level spans -41.8..-4.9 dBFS (~37 dB, ~5000x in power), and the quietest
    half of the 404 devices carry ~14% of the pooled signal between them.

    ``w[d] = p[d] ** -strength``, where ``p[d]`` is the device's median render
    power over the *train* split (the rows the loss actually sees). strength=0
    gives every device weight 1 (the pooled loss exactly as before); strength=1
    equalizes every device's expected contribution.

    Weights are normalized to mean 1 so the loss keeps its magnitude -- and lr
    its meaning -- as ``strength`` changes. Devices with no train-split
    ``render_ok`` row fall back to 1.0.

    The same weights are applied to val ESR, so the checkpoint/plateau/early-stop
    criteria score what the run optimizes. The unweighted pooled metric is
    near-blind here: a 30% gain on the 100 quietest devices moves it -1.8%,
    against ~0.3% epoch-to-epoch noise, and a 5% cost on the loudest flips its
    sign. A non-zero strength therefore makes val ESR incomparable with a
    strength-0 run -- compare per-device ESR across runs.

    Only the ESR term is weighted. MRSTFT is left alone: its log-magnitude
    sub-term is already scale-invariant, and while auraloss's spectral-
    convergence sub-term pools batch-globally the same way ESR does, reweighting
    inside a third-party loss is a separate change from the one this knob makes.
    """
    n = len(ids)
    w = np.ones(n, dtype=np.float64)
    if strength == 0.0:
        return w

    renders = manifests.read_manifest(renders_manifest, manifests.RENDERS_COLUMNS)
    ok = renders[(renders["status"] == C.RENDER_OK) & (renders["split"] == C.SPLIT_TRAIN)]
    rms_db = ok.groupby("device_id")["out_rms_db"].median() if not ok.empty else {}

    missing = []
    for i, d in enumerate(ids):
        v = rms_db.get(int(d), np.nan) if len(rms_db) else np.nan
        if not np.isfinite(v):
            missing.append(int(d))
            continue
        w[i] = 10.0 ** (-strength * float(v) / 10.0)     # power**-strength
    w /= w.mean()
    if missing:
        print(f"[loss]  {len(missing)} device(s) have no train-split render level "
              f"— weight 1.0: {missing[:8]}{' ...' if len(missing) > 8 else ''}",
              flush=True)
    return w
