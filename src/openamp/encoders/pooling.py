"""Ways of collapsing a variable-length feature sequence to one fixed vector.

Every encoder in this package ends in one of these. They are separated from the
backbones that feed them because pooling is the axis the encoder experiments
actually move along -- what a reference is summarized *into* has mattered more
than what produced the features (see ``dac-frames-are-3d-not-the-pooling``).

- :class:`AttentiveStatsPool` / :class:`MultiHeadAttentiveStatsPool` /
  :class:`MultiTapStatsHead` -- attention-weighted (mean, std) over a
  ``[B, C, L]`` residual stream, at one, K, or several-taps-then-K widths.
- :class:`CrossAttentionPool` -- learned queries attending over ``[B, T, D]``
  codec frames.
- :class:`FrameStore` -- not a pool: the memmapped frame shards the last one
  reads. Here because it is the input side of the same seam.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["AttentiveStatsPool", "MultiHeadAttentiveStatsPool", "MultiTapStatsHead",
           "CrossAttentionPool", "FrameStore",
           "ENC_ATTN_HIDDEN", "ENC_TAP_STRIDE", "ENC_EXPAND_DIM", "ENC_N_HEADS",
           "ENC_EXPAND_NORM", "ENC_EXPAND_NORMS", "ENC_NORM_GROUPS"]

ENC_ATTN_HIDDEN = 64       # attention bottleneck, independent of C_enc

# Average-pools the concatenated taps before the attention head -- a WaveNet
# never downsamples, so the tap is still full sample-rate width, and the
# attention tensor (B, K*E, T) is a memory knob, not a modelling one.
# Average-pool, not striding, to avoid aliasing.
ENC_TAP_STRIDE = 8
ENC_EXPAND_DIM = 256       # E: pooled width is K * 2 * E, set independently of C_enc
ENC_N_HEADS = 2            # K: each head may weight the reference differently
ENC_EXPAND_NORMS = ("batch", "group", "none")
ENC_EXPAND_NORM = "batch"
ENC_NORM_GROUPS = 32       # "group" only; clamped to divide expand_dim


# --- Attentive statistics pooling ------------------------------------------------

def _masked_global_stats(h: torch.Tensor, mask: torch.Tensor | None, eps: float):
    """``(B, C, 1)`` mean and std over the valid frames only.

    Divides by the **valid-frame count**, not ``T``: dividing by ``T`` would pull
    every statistic toward zero in proportion to how much padding an item carries,
    which is a per-item bias disguised as a feature.
    """
    if mask is None:
        mean = h.mean(dim=-1, keepdim=True)
        var = h.var(dim=-1, unbiased=False, keepdim=True)
    else:
        m = mask.bool().unsqueeze(1).to(h.dtype)             # [B, 1, T]
        n = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (h * m).sum(dim=-1, keepdim=True) / n
        var = (((h - mean) ** 2) * m).sum(dim=-1, keepdim=True) / n
    return mean, var.clamp_min(eps).sqrt()


def _weighted_moments(alpha: torch.Tensor, h: torch.Tensor, eps: float):
    """Attention-weighted mean and std; ``alpha`` broadcasts against ``h``."""
    mean = (alpha * h).sum(dim=-1)
    # E[x^2] - E[x]^2, clamped: the two terms are close for a near-constant
    # channel and float error can put the difference below zero.
    var = (alpha * h * h).sum(dim=-1) - mean * mean
    return mean, var.clamp_min(eps).sqrt()


class AttentiveStatsPool(nn.Module):
    """(B, C, L) -> (B, 2C): attention-weighted mean and standard deviation.

    Per-channel attention (Okabe et al.), so different channels can read
    different parts of the reference. Masking is applied to the logits before
    softmax, so an invalid frame gets exactly zero weight in both statistics.
    """

    def __init__(self, channels: int, hidden: int = ENC_ATTN_HIDDEN,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.score = nn.Sequential(nn.Conv1d(channels, hidden, 1), nn.Tanh(),
                                   nn.Conv1d(hidden, channels, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(h)                               # [B, C, L]
        if mask is not None:
            if mask.shape[-1] != h.shape[-1]:
                raise ValueError(f"mask length {mask.shape[-1]} does not match feature "
                                 f"length {h.shape[-1]}")
            logits = logits.masked_fill(~mask.bool().unsqueeze(1), float("-inf"))
        alpha = torch.softmax(logits.float(), dim=-1).to(h.dtype)
        mean, std = _weighted_moments(alpha, h, self.eps)
        return torch.cat([mean, std], dim=-1)                # [B, 2C]


class MultiHeadAttentiveStatsPool(nn.Module):
    """(B, E, T) -> (B, K * 2E): K independent attention-weighted (mean, std).

    K heads let the same channel be read under several different weightings
    (e.g. loud vs quiet passages) instead of one. Each frame is scored from
    [h; global_mean; global_std] rather than h alone, so "is this frame loud"
    is judged against its own reference. Masking is again applied to the logits.
    """

    def __init__(self, channels: int, n_heads: int, hidden: int = ENC_ATTN_HIDDEN,
                 eps: float = 1e-5):
        super().__init__()
        self.channels = int(channels)
        self.n_heads = int(n_heads)
        self.eps = float(eps)
        self.score = nn.Sequential(nn.Conv1d(3 * self.channels, hidden, 1), nn.Tanh(),
                                   nn.Conv1d(hidden, self.n_heads * self.channels, 1))

    @property
    def pooled_dim(self) -> int:
        return self.n_heads * 2 * self.channels

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, E, T = h.shape
        if mask is not None and mask.shape[-1] != T:
            raise ValueError(f"mask length {mask.shape[-1]} does not match feature "
                             f"length {T}")
        gmean, gstd = _masked_global_stats(h, mask, self.eps)
        ctx = torch.cat([h, gmean.expand(-1, -1, T), gstd.expand(-1, -1, T)], dim=1)
        logits = self.score(ctx).view(B, self.n_heads, E, T)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool()[:, None, None, :], float("-inf"))
        alpha = torch.softmax(logits.float(), dim=-1).to(h.dtype)
        mean, std = _weighted_moments(alpha, h.unsqueeze(1), self.eps)   # [B, K, E]
        return torch.cat([mean, std], dim=-1).flatten(1)                 # [B, K*2E]


def _make_expand_norm(norm: str, channels: int, groups: int) -> nn.Module:
    key = str(norm).strip().lower()
    if key not in ENC_EXPAND_NORMS:
        raise ValueError(f"enc_expand_norm must be one of {ENC_EXPAND_NORMS}, got {norm!r}")
    if key == "none":
        return nn.Identity()
    if key == "batch":
        return nn.BatchNorm1d(channels)
    g = min(int(groups), channels)
    while g > 1 and channels % g:                            # must divide evenly
        g -= 1
    return nn.GroupNorm(g, channels)


class MultiTapStatsHead(nn.Module):
    """list[(B, C, L)] -> (B, De): taps -> decimate -> expand -> K-head pooling.

    Three independent capacity levers: taps widen *what* is summarized, the 1x1
    expansion widens *how many* statistics are taken (decoupled from C_enc),
    the K heads multiply that again. Pooled width is K*2*expand_dim; keep it
    well above De or the projection becomes a decorative bottleneck.

    Expansion norm defaults to BatchNorm: it subtracts a dataset-level mean, so
    per-amp deviation survives into the pool, unlike GroupNorm/LayerNorm which
    subtract a per-sample mean and delete exactly the statistic being pooled.
    "group" is offered for batch-independence; "none" matches the A2 schedule's
    no-normalization rule.

    tap_stride average-pools the concatenated taps before the expansion (a
    memory knob, not a modelling one -- decimating without a filter would alias).

    Masking is exact through the pool (padded frames get zero attention
    weight), but only exact through BatchNorm/GroupNorm in eval() mode, where
    each frame normalizes independently; in train() mode invalid frames still
    shift the batch statistics. Moot today since no caller passes a mask, but
    relevant to a future ragged-batch path.
    """

    def __init__(self, *, in_channels: int, embedding_dim: int,
                 expand_dim: int = ENC_EXPAND_DIM, n_heads: int = ENC_N_HEADS,
                 attn_hidden: int = ENC_ATTN_HIDDEN, proj_hidden: int = 0,
                 tap_stride: int = ENC_TAP_STRIDE, norm: str = ENC_EXPAND_NORM,
                 norm_groups: int = ENC_NORM_GROUPS, normalize: bool = True,
                 eps: float = 1e-5):
        super().__init__()
        self.tap_stride = int(tap_stride)
        self.normalize = bool(normalize)
        self.expand = nn.Conv1d(in_channels, expand_dim, 1)
        self.norm = _make_expand_norm(norm, expand_dim, norm_groups)
        self.act = nn.GELU()
        self.pool = MultiHeadAttentiveStatsPool(expand_dim, n_heads, attn_hidden, eps)
        # proj_hidden 0 = derive as pooled // 4, the spec's rule.
        hidden = int(proj_hidden) if int(proj_hidden) > 0 else max(1, self.pooled_dim // 4)
        self.proj = nn.Sequential(nn.Linear(self.pooled_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, embedding_dim))

    @property
    def pooled_dim(self) -> int:
        return self.pool.pooled_dim

    def forward(self, taps: "list[torch.Tensor]",
                mask: torch.Tensor | None = None) -> torch.Tensor:
        h = torch.cat(taps, dim=1)
        if mask is not None and mask.shape[-1] != h.shape[-1]:
            raise ValueError(f"mask length {mask.shape[-1]} does not match feature "
                             f"length {h.shape[-1]}")
        if self.tap_stride > 1:
            h = F.avg_pool1d(h, self.tap_stride, self.tap_stride, ceil_mode=True,
                             count_include_pad=False)
            if mask is not None:
                # A pooled frame is valid if *any* of its inputs were, matching how
                # avg_pool1d forms it. max_pool over the float mask says exactly that.
                mask = F.max_pool1d(mask.float().unsqueeze(1), self.tap_stride,
                                    self.tap_stride, ceil_mode=True).squeeze(1) > 0
        h = self.act(self.norm(self.expand(h)))
        e = self.proj(self.pool(h, mask))
        return F.normalize(e, dim=-1) if self.normalize else e


# --- Cross-attention pooling over codec frames -----------------------------------

class CrossAttentionPool(nn.Module):
    """``[B, T, D] -> [B, n_q * d]``: learned queries attend over the frames.

    The queries are ``nn.Parameter`` and amp-independent -- they encode *what to
    look for* in a capture, and the amp enters only through the keys and values.
    That is the whole difference from mean+std, which weights every one of the
    14,728 frames identically and collapses the cloud to ~2.8 effective
    dimensions.

    Softmax in float32 then cast back, matching ``AttentiveStatsPool``'s
    discipline: under autocast the logits are otherwise reduced in bf16 across
    14k frames, where the running sum loses the small terms that distinguish one
    amp's attention pattern from another's.
    """

    def __init__(self, in_dim: int, n_queries: int, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.k = nn.Linear(in_dim, d_model)
        self.v = nn.Linear(in_dim, d_model)
        self.q = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

    @property
    def pooled_dim(self) -> int:
        return self.q.shape[0] * self.d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        K, V = self.k(x), self.v(x)                       # [B, T, d]
        logits = (self.q @ K.transpose(1, 2)) / self.d_model ** 0.5      # [B, n_q, T]
        if mask is not None:
            if mask.shape[-1] != x.shape[1]:
                raise ValueError(f"mask length {mask.shape[-1]} does not match frame "
                                 f"count {x.shape[1]}")
            logits = logits.masked_fill(~mask.bool()[:, None, :], float("-inf"))
        alpha = torch.softmax(logits.float(), dim=-1).to(V.dtype)
        return (alpha @ V).flatten(1)                     # [B, n_q * d]


class FrameStore:
    """Lazily-memmapped ``{device_id: [T, D] float16}`` shards.

    Memmapped rather than resident: 450 shards is 12.64 GiB against 30 GB of
    system RAM, and the page cache holds it after the first epoch without
    competing with torch for the same allocation. Never a registered buffer --
    it would be checkpointed and pushed to the GPU.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"no meta.json under {self.root} — run "
                             f"`python -m openamp.encoder.extract` first")
        self.meta = json.loads(meta_path.read_text())
        self.device_ids = [int(d) for d in self.meta["device_ids"]]
        self.n_frames = int(self.meta["n_frames"])
        self.latent_dim = int(self.meta["latent_dim"])
        self._mm: dict[int, np.ndarray] = {}

    def __contains__(self, did: int) -> bool:
        return int(did) in set(self.device_ids)

    def get(self, did: int) -> np.ndarray:
        did = int(did)
        if did not in self._mm:
            path = self.root / "frames" / f"{did:04d}.npy"
            if not path.is_file():
                raise KeyError(f"no frames for device {did} under {self.root}")
            self._mm[did] = np.load(path, mmap_mode="r")
        return self._mm[did]

    def batch(self, ids, device, *, window: int | None = None,
              starts=None) -> torch.Tensor:
        """``[B, T, D]`` fp16, optionally one contiguous ``window`` per id.

        The slice is taken on the memmap, before the stack, so a windowed read
        touches only the pages it needs -- that is the point of the option, not
        just a smaller output tensor.

        ``starts`` is one offset per id (defaults to all-zero). Callers pick
        them: the frame store has no opinion about whether they should be
        random or fixed.
        """
        if window is None:
            arr = np.stack([np.asarray(self.get(d)) for d in ids])   # [B, T, D] fp16
        else:
            window = int(window)
            offs = [0] * len(ids) if starts is None else [int(s) for s in starts]
            if len(offs) != len(ids):
                raise ValueError(f"starts has {len(offs)} entries for {len(ids)} ids")
            # Checked before the slice: a short slice would otherwise surface
            # as np.stack's "all input arrays must have the same shape", which
            # says nothing about which start ran off the end.
            bad = [(int(d), s) for d, s in zip(ids, offs)
                   if s < 0 or s + window > self.n_frames]
            if bad:
                raise ValueError(f"window {window} overruns the {self.n_frames}-frame "
                                 f"shards at (device, start) {bad}")
            arr = np.stack([np.asarray(self.get(d)[s:s + window])
                            for d, s in zip(ids, offs)])
        return torch.from_numpy(arr).to(device, non_blocking=True)
