"""Iterative-refinement pooling over codec frames, and the encoder arm using it.

:class:`CrossAttentionPool` reads the frames once: the learned queries score
every frame, the softmax averages, done. The queries never see what they found,
so the second query cannot ask a question that depends on the first one's
answer, and no query can revise its own.

:class:`CrossAttentionPool2Block` runs that read twice. Each block owns its K/V
projection of the frames, so the second block looks at the same frames through a
*different* lens than the first, and between the two reads the queries attend to
each other and pass through an MLP. Two rounds, fixed -- not a depth knob.

Also new here, and independent of the refinement: a **fixed sinusoidal**
positional encoding added to the frames. A pool with no positions cannot tell a
decaying tail from the attack that preceded it -- it sees a bag of frames.

It is a switch, not a fixture: ``positional=False`` builds no table and adds
nothing to the frames, so the two blocks can be run with and without positions
and the difference read off directly.

This arm always reads the whole render, so a frame's position is simply its
index, ``0..n_frames-1``, and the pool fills that in itself. (The pool still
takes explicit ``frame_positions`` for a caller that crops: a crop's frames
carry their *absolute* index in the render, not their offset within the crop.
Nothing here does that.)

Sinusoids, not a learned table, and the difference is not small: a learned
``[n_frames, latent_dim]`` embedding is 15.1 M parameters on the DAC store
(14,728 x 1024) against the 263 k of the whole single-block arm, all of it in
the ``enc_lr`` group with two AdamW moments behind it. The sinusoid is a
non-persistent buffer instead -- **zero parameters**, nothing to checkpoint,
rebuilt deterministically at construction -- so the arm stays a test of
refinement rather than of a table 57x the size of the encoder it sits in.

Fixed amplitude is the tradeoff it buys: the sinusoids are unit-amplitude
against DAC frames whose std is ~2.9, so position enters at a scale nothing can
tune. A learned table started at the same order but could shrink or grow itself.

The pool is a drop-in for :class:`CrossAttentionPool`: same
``(in_dim, n_queries, d_model)`` constructor head, same ``pooled_dim``, same
``[B, T, D] -> [B, n_q * d]`` call. The encoder is therefore nothing but
:class:`FrameAttentionEncoder` with this pool substituted -- one constructor,
no forward of its own -- so a comparison between ``dac_frames`` and
``dac_frames_refine`` is a comparison of pooling alone.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from openamp.encoders.frames import FrameAttentionEncoder
from openamp.encoders.pooling import FrameStore

__all__ = ["CrossAttnBlock", "CrossAttentionPool2Block", "AttentionEncoder",
           "sinusoidal_positions", "REFINE_N_HEADS", "REFINE_MLP_RATIO",
           "REFINE_MAX_LEN", "REFINE_POS_BASE"]

REFINE_N_HEADS = 4         # heads of the query-to-query attention; must divide d_model
REFINE_MLP_RATIO = 4       # per-block MLP widens d_model by this, then comes back
REFINE_MAX_LEN = 20_000    # rows of the positional table; the encoder passes n_frames
REFINE_POS_BASE = 10_000.0  # the transformer geometric progression's base


def sinusoidal_positions(max_len: int, dim: int,
                         base: float = REFINE_POS_BASE) -> torch.Tensor:
    """``[max_len, dim]`` transformer sinusoids: sin on even dims, cos on odd.

    Wavelengths run as a geometric progression from 2*pi frames up to
    ``2*pi*base``. At ``dim=1024`` on the DAC store that spans ~6 frames (73 ms)
    to well past the 14,728-frame render, so the low dims separate neighbouring
    frames and the high ones read as a slow ramp across the whole capture.

    Not a parameter and not persistent: it is a pure function of
    ``(max_len, dim, base)``, so storing it in a checkpoint would only be a way
    for a checkpoint to disagree with the code that rebuilt it.
    """
    pos = torch.arange(int(max_len), dtype=torch.float32)[:, None]
    inv = torch.exp(-math.log(base) * torch.arange(0, dim, 2,
                                                   dtype=torch.float32) / dim)
    table = torch.zeros(int(max_len), int(dim))
    table[:, 0::2] = torch.sin(pos * inv)
    # dim may be odd, in which case the cos half is one column shorter.
    table[:, 1::2] = torch.cos(pos * inv)[:, :table[:, 1::2].shape[1]]
    return table


class CrossAttnBlock(nn.Module):
    """``[B, n_q, d] -> [B, n_q, d]``: one refinement step over the frames.

    Three residual sub-steps: cross-attend the queries into the frames through
    this block's **own** K/V projection, let the queries attend to each other,
    then an MLP. The query-to-query attention is what makes a stack different
    from one wider read -- it is ``n_q x n_q``, so it costs nothing next to the
    ``n_q x T`` cross-attention.

    Post-norm (norm after the residual add), as written. Frames come in as
    ``[B, T, in_dim]`` and are projected per block, never normalized here.
    """

    def __init__(self, in_dim: int, d_model: int, n_heads: int = REFINE_N_HEADS,
                 mlp_ratio: int = REFINE_MLP_RATIO):
        super().__init__()
        d_model, n_heads = int(d_model), int(n_heads)
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.k = nn.Linear(in_dim, d_model)
        self.v = nn.Linear(in_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, int(mlp_ratio) * d_model), nn.GELU(),
                                 nn.Linear(int(mlp_ratio) * d_model, d_model))
        self.norm3 = nn.LayerNorm(d_model)

    def cross_attend(self, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, n_q, d]``, the same arithmetic ``CrossAttentionPool`` does.

        Including the float32 softmax: under autocast the logits would otherwise
        be reduced in bf16 across 14k frames -- see that class's docstring.
        ``mask`` is ``True`` = **valid**, this package's convention.
        """
        logits = (q @ K.transpose(1, 2)) / self.d_model ** 0.5        # [B, n_q, T]
        if mask is not None:
            logits = logits.masked_fill(~mask.bool()[:, None, :], float("-inf"))
        alpha = torch.softmax(logits.float(), dim=-1).to(V.dtype)
        return alpha @ V                                              # [B, n_q, d]

    def forward(self, q: torch.Tensor, frames: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        K, V = self.k(frames), self.v(frames)                         # [B, T, d]
        
        q = self.norm1(q + self.cross_attend(q, K, V, mask))
        
        q = self.norm2(q + self.self_attn(q, q, q, need_weights=False)[0])
        return self.norm3(q + self.mlp(q))


class CrossAttentionPool2Block(nn.Module):
    """``[B, T, D] -> [B, n_q * d]``: two refinement blocks over positioned frames.

    Drop-in for :class:`~openamp.encoders.pooling.CrossAttentionPool`: same
    constructor head, same ``pooled_dim``, and ``forward(x, mask)`` still works
    -- ``frame_positions`` is optional and defaults to ``0..T-1``, which is only
    right when ``x`` is the *whole* render. A caller that crops must pass each
    frame's *absolute* index in the render; nothing in this package does.

    ``max_len`` bounds the positional table. The default is generous enough for
    the DAC store; the encoder below passes the store's own ``n_frames``, which
    is exactly the range of positions that can occur. The table is fixed
    sinusoids, so its only cost is the memory it occupies -- no parameters, no
    optimizer state, nothing in the state dict.

    ``positional=False`` drops that entirely: no table is built, nothing is
    added to the frames, and ``frame_positions`` is then a ``ValueError``
    rather than something quietly discarded.
    """

    def __init__(self, in_dim: int, n_queries: int, d_model: int, *,
                 n_heads: int = REFINE_N_HEADS, mlp_ratio: int = REFINE_MLP_RATIO,
                 max_len: int = REFINE_MAX_LEN, positional: bool = True):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.positional = bool(positional)
        if self.positional:
            # On the frames, not on the queries: it is the frames that have an
            # order. persistent=False keeps it out of the state dict -- see the
            # function.
            self.register_buffer("pos_table",
                                 sinusoidal_positions(self.max_len, int(in_dim)),
                                 persistent=False)
        self.q = nn.Parameter(torch.randn(int(n_queries), self.d_model) * 0.02)
        self.block1 = CrossAttnBlock(in_dim, self.d_model, n_heads, mlp_ratio)
        self.block2 = CrossAttnBlock(in_dim, self.d_model, n_heads, mlp_ratio)

    @property
    def pooled_dim(self) -> int:
        return self.q.shape[0] * self.d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                frame_positions: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        if mask is not None and mask.shape[-1] != T:
            raise ValueError(f"mask length {mask.shape[-1]} does not match frame "
                             f"count {T}")
        if not self.positional:
            if frame_positions is not None:
                raise ValueError("frame_positions given, but this pool was built "
                                 "with positional=False and adds no positions")
            pos = None
        elif frame_positions is None:
            if T > self.max_len:
                raise ValueError(f"{T} frames exceeds the positional table's "
                                 f"{self.max_len} rows")
            # Broadcast, not a gather: with no positions given every item reads
            # the same frames 0..T-1, so the rows are one [1, T, D] block rather
            # than a [B, T, D] copy of it. At B=32, T=14728, D=1024 that copy
            # would be 1.9 GiB of identical data.
            pos = self.pos_table[:T].unsqueeze(0)
        elif tuple(frame_positions.shape) != (B, T):
            raise ValueError(f"frame_positions must be [{B}, {T}], got "
                             f"{tuple(frame_positions.shape)}")
        else:
            pos = self.pos_table[frame_positions]

        if pos is not None:
            x = x + pos
        q = self.q.unsqueeze(0).expand(B, -1, -1)                     # [B, n_q, d]
        q = self.block2(self.block1(q, x, mask), x, mask)
        return q.flatten(1)                                           # [B, n_q * d]


class AttentionEncoder(FrameAttentionEncoder):
    """``[B]`` device ids ``-> [B, embedding_dim]``, refined pooling.

    :class:`~openamp.encoders.frames.FrameAttentionEncoder` with
    :class:`CrossAttentionPool2Block` where the single-block pool was, and
    nothing else: the memmapped lookup, the single-Linear adapter, the L2
    normalize and the forward itself are all inherited, so the two arms differ
    in exactly one place.

    **Always the whole render.** ``frame_window`` is not accepted here (passing
    it is a ``TypeError``, which is the intent -- a config that sets it is a
    config written for the other arm). Frames therefore always start at index
    0, which is what the pool assumes when no positions are given.

    Note what full frames costs, since the base class offers the window for a
    reason: one device id maps to exactly one fixed input tensor, so there are
    as many distinct encoder inputs as there are devices, and memorizing them
    is a strategy available to the encoder.
    """

    def __init__(self, *, embedding_dim: int, frames: FrameStore,
                 n_queries: int = 4, d_model: int = 64,
                 n_heads: int = REFINE_N_HEADS, mlp_ratio: int = REFINE_MLP_RATIO,
                 positional: bool = True, normalize: bool = True,
                 grad_checkpoint: bool = False):
        super().__init__(
            embedding_dim=embedding_dim, frames=frames, n_queries=n_queries,
            d_model=d_model, normalize=normalize, frame_window=0,
            grad_checkpoint=grad_checkpoint,
            # max_len = the store's frame count: reading the whole render, the
            # largest position that can occur is n_frames - 1. The table is
            # buffer-only, so this sizes memory, not parameters.
            pool=CrossAttentionPool2Block(frames.latent_dim, n_queries, d_model,
                                          n_heads=n_heads, mlp_ratio=mlp_ratio,
                                          max_len=frames.n_frames,
                                          positional=positional))
        self.n_heads = int(n_heads)          # reporting only; the pool has them
        self.mlp_ratio = int(mlp_ratio)

    def capacity_report(self) -> dict:
        return {**super().capacity_report(), "head": "dac_frames_refine",
                "n_blocks": 2, "n_heads": self.n_heads,
                "mlp_ratio": self.mlp_ratio, "max_len": self.pool.max_len,
                "positional": self.pool.positional}
