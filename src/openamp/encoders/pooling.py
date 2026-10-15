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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["AttentiveStatsPool", "MultiHeadAttentiveStatsPool", "MultiTapStatsHead",
           "CrossAttentionPool", "FrameStore",
           "ENC_ATTN_HIDDEN", "ENC_TAP_STRIDE", "ENC_EXPAND_DIM", "ENC_N_HEADS",
           "ENC_EXPAND_NORM", "ENC_EXPAND_NORMS", "ENC_NORM_GROUPS",
           "FRAME_GATHER_THREADS"]

# Threads FrameStore.batch copies its shards with. The cost there is page faults
# against the memmap, not Python, and numpy drops the GIL around the copy, so the
# stalls overlap. Measured on the DAC store (32 amps per batch, 10 cores, another
# run using the machine):
#
#     threads          1       2       4       6       8
#     7,364 frames  0.301   0.171   0.067   0.063   0.061  s
#     14,728        0.889   0.373   0.160   0.141   0.172  s
#
# 4 is where it flattens -- past that the limit is the NVMe and the page cache,
# not the cores -- and it leaves most of the box to whatever else is running.
#
# Threads, not DataLoader workers: openamp.encoders.frames records why processes
# lose this trade, a ~30 MiB item copied through shared memory and pin_memory
# costing more than the gather it was meant to hide. Threads share the memmap and
# copy nothing extra -- each one writes straight into its own row of the output.
# Order is preserved, so a batch is byte-identical to the single-threaded one and
# nothing about a run's numbers changes.
FRAME_GATHER_THREADS = 4

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
                 attn_hidden: int = ENC_ATTN_HIDDEN,
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
        # An adapter, not a stage: one Linear whose only job is to change width
        # from pooled_dim to De. No hidden layer, so nothing here narrows what
        # the pool found before the generator sees it.
        self.proj = nn.Linear(self.pooled_dim, embedding_dim)

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
                             f"`python -m openamp.probes.extract` first")
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

    def _gather(self, ids, take, rows: int) -> np.ndarray:
        """``[B, rows, D]`` fp16: ``take(memmap, k) -> [rows, D]`` per id, in order.

        Each thread writes its own row of the output, so there is no lock and no
        second copy of the batch -- the destination is allocated once here. The
        memmaps are opened in the calling thread, so the pool only ever touches
        ones that already exist.
        """
        mms = [self.get(d) for d in ids]
        out = np.empty((len(mms), int(rows), self.latent_dim), dtype=np.float16)

        def fill(k: int) -> None:
            out[k] = take(mms[k], k)

        n = min(int(FRAME_GATHER_THREADS), len(mms))
        if n <= 1:
            for k in range(len(mms)):
                fill(k)
        else:
            with ThreadPoolExecutor(n) as ex:
                list(ex.map(fill, range(len(mms))))
        return out

    def batch(self, ids, device, *, window: int | None = None,
              starts=None, indices=None) -> torch.Tensor:
        """``[B, T, D]`` fp16: every frame, one contiguous ``window`` per id, or
        an explicit set of frame ``indices`` per id.

        The slice or gather is taken on the memmap, before the stack, so a
        partial read touches only the pages it needs -- that is the point of
        these options, not just a smaller output tensor.

        ``starts`` is one offset per id (defaults to all-zero). ``indices`` is
        one ``[k]`` index array per id, ``[B, k]`` in all, for a scattered read
        rather than a contiguous one. Callers pick both: the frame store has no
        opinion about whether they should be random or fixed.
        """
        if indices is not None:
            if window is not None or starts is not None:
                raise ValueError("indices is exclusive with window/starts — a "
                                 "scattered draw and a contiguous span are "
                                 "different reads")
            idx = np.asarray(indices, dtype=np.int64)
            if idx.ndim != 2 or len(idx) != len(ids):
                raise ValueError(f"indices must be [{len(ids)}, k], got "
                                 f"{tuple(idx.shape)}")
            if idx.size and (idx.min() < 0 or idx.max() >= self.n_frames):
                raise ValueError(f"frame indices out of range for the "
                                 f"{self.n_frames}-frame shards: "
                                 f"[{idx.min()}, {idx.max()}]")
            arr = self._gather(ids, lambda m, k: np.asarray(m)[idx[k]], idx.shape[1])
        elif window is None:
            arr = self._gather(ids, lambda m, k: m, self.n_frames)   # [B, T, D] fp16
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
            arr = self._gather(ids, lambda m, k: m[offs[k]:offs[k] + window], window)
        return torch.from_numpy(arr).to(device, non_blocking=True)
