"""WaveNet reference encoder: ``(dry, wet) -> one embedding vector``.

The new half of the joint model. It reads a short reference recording of an amp
and emits the vector that :class:`~openamp.emulate.wavenet.FiLMWaveNet` consumes,
replacing the per-device row that model looks up today. Trained jointly with the
generator, from scratch; see ``guidelines.md`` §3 for the spec this implements.

Three things distinguish it from the generator it feeds:

- **Two input channels.** The reference is the ``(dry, wet)`` pair, so the encoder
  observes the transfer function directly rather than inferring it from the wet
  side alone. Input is deliberately **unnormalized** — amplitude is gain and
  compression information, so scaling it away would delete part of the answer.
- **Non-causal.** The reference is offline, so every conv is centered rather than
  left-padded. Same schedule, but a layer sees ~66 ms either side instead of
  132 ms behind.
- **No head, no conditioning.** The A2 head collapses ``C`` channels to one audio
  sample; here the point is the opposite, so the layer array runs and the
  residual stream ``(B, C_enc, L)`` is pooled instead. There is no FiLM hook and
  no embedding table — nothing conditions this network, it *is* the condition.

Deliberately **not** a :class:`~openamp.emulate.wavenet.FiLMWaveNet` subclass: it
would inherit the embedding table and the causal padding, which are the two things
it exists not to have. It does share that module's schedule constants and
activation factory, so "the same WaveNet" stays literally true.

**Two pooling heads.**

``head="stats"`` is the original: :class:`AttentiveStatsPool` over the final trunk,
then an MLP to ``embedding_dim``. Its pooled vector is ``2 * C_enc`` numbers, which
ties readout width to backbone width — at ``C_enc=16`` the embedding lives on a
32-dimensional manifold however wide the projection's output is. Measured on the
``proto`` run, the held-out capture embeddings spanned 8.55 effective dimensions
against the 51.8 of the lookup table that reaches val ESR 0.052 on the same corpus,
so 32 is a real ceiling *and* it was not being filled.

``head="multitap"`` (:class:`MultiTapStatsHead`) breaks that tie with three
independent levers: taps from several depths, a 1x1 channel expansion, and ``K``
attention heads. Pooled width becomes ``K * 2 * expand_dim``, set without touching
the backbone. Because every layer here same-pads and keeps ``channels`` constant,
the taps all come out at one length and one width — no trimming or alignment is
needed, and the concatenated width is just ``len(taps) * C_enc``.

**On normalization.** The layer array has none: the A2 schedule has none and this
follows it. The multitap head is the one exception, and the choice there is load
bearing — see :class:`MultiTapStatsHead`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from openamp.emulate.wavenet import (A2_ACTIVATION, A2_DILATIONS, A2_KERNEL_SIZES,
                                     make_activation, normalize_activation)

__all__ = ["WaveNetEncoder", "WaveNetEncoderLayer", "AttentiveStatsPool",
           "MultiHeadAttentiveStatsPool", "MultiTapStatsHead",
           "ENC_CHANNELS", "ENC_ATTN_HIDDEN", "ENC_PROJ_HIDDEN", "ENC_HEADS",
           "ENC_TAPS", "ENC_TAP_STRIDE", "ENC_EXPAND_DIM", "ENC_N_HEADS",
           "ENC_EXPAND_NORM", "ENC_EXPAND_NORMS", "ENC_NORM_GROUPS"]

ENC_CHANNELS = 32          # guidelines §7 puts C_enc at 16-32; see the width note above
ENC_ATTN_HIDDEN = 64       # attention bottleneck, independent of C_enc
ENC_PROJ_HIDDEN = 128      # hidden width of the pooled -> De projection

# --- multitap head ---------------------------------------------------------------
ENC_HEADS = ("stats", "multitap")
# End of each of the A2 schedule's three dilation runs (7 + 7 + 2 + 7 layers). Early
# taps see ~ms structure, the last sees the full 132 ms where sag and compression
# time constants live; pooling only the deepest view discards the short-scale one.
ENC_TAPS = (6, 13, 22)
# A WaveNet never downsamples, so a 2 s reference is still 96 000 frames at the tap.
# The attention tensor is (B, K*E, T), which is what makes that unaffordable: at
# batch 16 / E 512 / K 4 the head alone wants ~45 GB. Tone does not need
# sample-rate time resolution, so decimate first. Average-pool, not striding —
# plain striding aliases.
ENC_TAP_STRIDE = 8
ENC_EXPAND_DIM = 256       # E: pooled width is K * 2 * E, set independently of C_enc
ENC_N_HEADS = 2            # K: each head may weight the reference differently
ENC_EXPAND_NORMS = ("batch", "group", "none")
ENC_EXPAND_NORM = "batch"
ENC_NORM_GROUPS = 32       # "group" only; clamped to divide expand_dim


class WaveNetEncoderLayer(nn.Module):
    """One non-causal A2-schedule layer: ``(trunk, reference) -> (trunk, skip)``.

    Mirrors :class:`~openamp.emulate.wavenet.FiLMWaveNetLayer` minus the FiLM hook
    — ``z = conv(x) + mixer(reference)``, activation, residual back onto the trunk
    — with the causal left pad replaced by a centered one. Extra padding on an
    even kernel goes on the right, which only fixes an arbitrary half-sample
    convention; nothing downstream depends on the choice because the output is
    pooled over time.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 in_channels: int = 2, activation: str = A2_ACTIVATION):
        super().__init__()
        span = (kernel_size - 1) * dilation
        self.pad_left = span // 2
        self.pad_right = span - self.pad_left
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.input_mixer = nn.Conv1d(in_channels, channels, 1, bias=False)
        self.act = make_activation(activation)
        self.layer1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """``x`` trunk [B,C,L], ``cond`` the raw reference [B,in_channels,L]."""
        z = self.conv(F.pad(x, (self.pad_left, self.pad_right))) + self.input_mixer(cond)
        z = self.act(z)
        return x + self.layer1x1(z), z


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
    """``(B, C, L) -> (B, 2C)``: attention-weighted mean and standard deviation.

    Per-channel attention (Okabe et al.), so different feature channels may read
    different parts of the reference — the channel tracking sag wants the loud
    passages, the one tracking hiss wants the quiet ones.

    Masking is handled once, in the attention logits: an invalid frame is set to
    ``-inf`` before the softmax, so its weight is exactly zero and **both**
    statistics are masked by construction. Weighting the moments after the fact
    is the version that silently leaves the standard deviation unmasked.
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
    """``(B, E, T) -> (B, K * 2E)``: ``K`` independent attention-weighted (mean, std).

    Two changes from :class:`AttentiveStatsPool`:

    - **K heads.** One head gives each channel a single weighting over time. With
      ``K`` heads the *same* channel can be read under several — the loud passages
      and the quiet ones — instead of having to choose.
    - **Context-aware scoring.** Each frame is scored from ``[h ; global_mean ;
      global_std]`` (``3E`` channels) rather than from ``h`` alone, so "is this
      frame loud" is judged against the reference it sits in rather than an
      absolute threshold the network had to bake into its weights.

    Masking is again applied to the logits, so a padded frame gets weight exactly
    zero in both moments.
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


class MultiTapStatsHead(nn.Module):
    """``list[(B, C, L)] -> (B, De)``: taps -> decimate -> expand -> K-head pooling.

    The three capacity levers, in order. Taps widen *what* is summarized; the 1x1
    expansion widens *how many* statistics are taken, decoupled from ``C_enc``; the
    ``K`` heads multiply that again. Pooled width is ``K * 2 * expand_dim``, and the
    projection down to ``De`` is then the one deliberate bottleneck — keep
    ``pooled_dim`` comfortably above ``De`` or pooling is silently throttling it
    again, which is the failure this head exists to remove.

    **Why the expansion is BatchNorm by default.** A pooling head's whole output is
    per-sample statistics of ``h``, so the normalization in front of it must not be
    per-sample. BatchNorm subtracts a *dataset*-level mean, which carries no
    per-amp information, and every per-amp deviation survives into the pool — this
    is why the speaker-verification stacks this head follows (x-vector, ECAPA-TDNN)
    all put BatchNorm here. GroupNorm and LayerNorm subtract a *per-sample* mean
    over time, deleting exactly the statistic the pool is about to measure;
    ``"group"`` is offered because it is batch-independent, but it hands the pool a
    signal whose group-level mean is zero by construction. ``"none"`` matches the
    A2 schedule's no-normalization rule and leaves the expansion's scale unbounded.

    ``tap_stride`` average-pools the concatenated taps before the expansion. This
    is a memory knob, not a modelling one — the attention tensor is ``(B, K*E, T)``
    and ``T`` is the raw sample count — but average-pooling rather than striding
    matters, since decimating a 48 kHz trunk by 8 without a filter aliases.

    **Masking is exact through the pool, not through the normalization.** Padded
    frames get attention weight zero, so they cannot reach either moment. But
    BatchNorm's batch statistics and GroupNorm's per-sample statistics are both
    taken over *every* frame, invalid ones included, so in ``train()`` mode junk
    beyond the mask still shifts the normalization. In ``eval()`` mode BatchNorm
    uses fixed running stats and each frame is normalized independently, so masking
    is exact again; ``norm="none"`` is exact in both. This costs nothing today —
    ``JointDataset`` only ever emits fixed-length references and no caller passes a
    mask — but a ragged-batch path would have to face it.
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


class WaveNetEncoder(nn.Module):
    """``(B, 2, L_ref) -> (B, embedding_dim)``.

    ``normalize`` L2-projects the embedding onto the unit sphere (the interface
    contract's toggle, §2). Off by default so the module stays neutral, but the
    joint configs turn it **on**, and that is not a stylistic choice: nothing else
    bounds ``|e|``, and inflating it is a cheap early way for the reconstruction
    gradient to grow the generator's FiLM gain. Left free at lr 4e-3 this diverged
    on a fixed batch at ~step 500 — ``|e|`` reached 123, the generator's tanh
    saturated, and true vs. shuffled embeddings rendered *identically* (ESR ratio
    1.0, i.e. total collapse). Init is not the problem (``|e|`` starts near 3.7,
    comparable to the 1.6 of the table this replaces); the runaway is learned.
    Normalizing removes the direction rather than merely avoiding it.

    ``grad_checkpoint`` recomputes each layer's activations during backward
    instead of storing them. A WaveNet never downsamples, so a 2 s reference stays
    96 000 frames wide through all 23 layers; this is the escape hatch when that
    does not fit alongside the generator. It wraps the layer array only — the
    multitap head decimates by ``tap_stride`` before it does anything expensive.

    ``head`` picks the pooling head; see :class:`MultiTapStatsHead` for why
    ``"multitap"`` exists. The heads have disjoint parameters, so a checkpoint
    written under one cannot be loaded under the other — ``enc_head`` is
    structural, and ``train.py`` treats it that way.
    """

    def __init__(self, *, embedding_dim: int, in_channels: int = 2,
                 channels: int = ENC_CHANNELS, activation: str = A2_ACTIVATION,
                 kernel_sizes: tuple[int, ...] = A2_KERNEL_SIZES,
                 dilations: tuple[int, ...] = A2_DILATIONS,
                 attn_hidden: int = ENC_ATTN_HIDDEN,
                 proj_hidden: int = ENC_PROJ_HIDDEN,
                 normalize: bool = False, grad_checkpoint: bool = False,
                 head: str = "stats", taps: tuple[int, ...] = ENC_TAPS,
                 tap_stride: int = ENC_TAP_STRIDE,
                 expand_dim: int = ENC_EXPAND_DIM, n_heads: int = ENC_N_HEADS,
                 expand_norm: str = ENC_EXPAND_NORM,
                 norm_groups: int = ENC_NORM_GROUPS):
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError(f"kernel_sizes ({len(kernel_sizes)}) and dilations "
                             f"({len(dilations)}) must have the same length")
        if head not in ENC_HEADS:
            raise ValueError(f"enc_head must be one of {ENC_HEADS}, got {head!r}")
        self.embedding_dim = int(embedding_dim)
        self.in_channels = int(in_channels)
        self.channels = int(channels)
        self.activation = normalize_activation(activation, field="enc_activation")
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.dilations = tuple(int(d) for d in dilations)
        self.normalize = bool(normalize)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.head_kind = str(head)

        self.rechannel = nn.Conv1d(self.in_channels, self.channels, 1, bias=False)
        self.layers = nn.ModuleList(
            WaveNetEncoderLayer(self.channels, k, d, self.in_channels, self.activation)
            for k, d in zip(self.kernel_sizes, self.dilations))

        if self.head_kind == "multitap":
            self.taps = tuple(int(t) for t in taps)
            bad = [t for t in self.taps if not 0 <= t < len(self.layers)]
            if bad or not self.taps:
                raise ValueError(
                    f"enc_taps must be non-empty layer indices in [0, {len(self.layers)}), "
                    f"got {list(taps)}")
            self.head = MultiTapStatsHead(
                in_channels=len(self.taps) * self.channels,
                embedding_dim=self.embedding_dim, expand_dim=expand_dim,
                n_heads=n_heads, attn_hidden=attn_hidden, proj_hidden=proj_hidden,
                tap_stride=tap_stride, norm=expand_norm, norm_groups=norm_groups,
                normalize=self.normalize)
        else:
            self.taps = (len(self.layers) - 1,)
            self.pool = AttentiveStatsPool(self.channels, attn_hidden)
            hidden = int(proj_hidden) if int(proj_hidden) > 0 else max(1, self.channels // 2)
            self.proj = nn.Sequential(nn.Linear(2 * self.channels, hidden),
                                      make_activation(self.activation),
                                      nn.Linear(hidden, self.embedding_dim))

    @classmethod
    def from_config(cls, jcfg, embedding_dim: int) -> "WaveNetEncoder":
        """Construct from a :class:`~openamp.core.config.JointConfig`.

        ``embedding_dim`` comes from the *generator's* config, not this one — it is
        the interface contract, so it has exactly one source of truth.
        """
        return cls(embedding_dim=embedding_dim, channels=jcfg.enc_channels,
                   activation=jcfg.enc_activation, attn_hidden=jcfg.enc_attn_hidden,
                   proj_hidden=jcfg.enc_proj_hidden, normalize=jcfg.enc_normalize,
                   grad_checkpoint=jcfg.grad_checkpoint_encoder,
                   head=jcfg.enc_head, taps=tuple(jcfg.enc_taps),
                   tap_stride=jcfg.enc_tap_stride, expand_dim=jcfg.enc_expand_dim,
                   n_heads=jcfg.enc_n_heads, expand_norm=jcfg.enc_expand_norm,
                   norm_groups=jcfg.enc_norm_groups)

    @staticmethod
    def compute_receptive_field(kernel_sizes=A2_KERNEL_SIZES,
                                dilations=A2_DILATIONS) -> int:
        """Receptive field in samples. No head conv, so one term shorter than A2's.

        Unchanged by the head: every conv in either head has kernel size 1, and
        pooling reduces time rather than extending a frame's reach.
        """
        rf = 1
        for k, d in zip(kernel_sizes, dilations):
            rf += (k - 1) * d
        return rf

    @property
    def receptive_field(self) -> int:
        return self.compute_receptive_field(self.kernel_sizes, self.dilations)

    @property
    def pooled_dim(self) -> int:
        """Width of the vector the pooling head emits, before the projection."""
        return (self.head.pooled_dim if self.head_kind == "multitap"
                else 2 * self.channels)

    def capacity_report(self) -> dict:
        """What actually limits ``e``: the pool, or the projection you chose.

        ``ceiling`` is ``min(pooled_dim, embedding_dim)``. Keep ``pooled_dim`` well
        above ``embedding_dim`` — when they are comparable the pool is throttling
        and the projection's width is decorative.
        """
        return {"head": self.head_kind, "pooled_dim": self.pooled_dim,
                "embedding_dim": self.embedding_dim,
                "ceiling": min(self.pooled_dim, self.embedding_dim)}

    def _trunk(self, x: torch.Tensor, taps: "tuple[int, ...]") -> "list[torch.Tensor]":
        """Run the layer array, keeping the residual trunk after each index in ``taps``.

        Every layer same-pads and keeps ``channels``, so the kept tensors all share
        one shape ``(B, C_enc, L)`` — the concatenation downstream needs no trimming.
        """
        if x.dim() == 2:                                     # [B, L] -> single channel
            x = x.unsqueeze(1)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels (dry, wet), "
                             f"got {x.shape[1]}")
        want = set(taps)
        kept: list[torch.Tensor] = []
        h = self.rechannel(x)
        for i, layer in enumerate(self.layers):
            if self.grad_checkpoint and self.training:
                h, _ = checkpoint(layer, h, x, use_reentrant=False)
            else:
                h, _ = layer(h, x)
            if i in want:
                kept.append(h)
        return kept

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Layer array only: ``(B, 2, L) -> (B, C_enc, L)``, the residual stream.

        The trunk after the last layer, not the summed skip: guidelines §3 names
        the residual stream as the tap.
        """
        return self._trunk(x, (len(self.layers) - 1,))[0]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.head_kind == "multitap":
            return self.head(self._trunk(x, self.taps), mask)
        e = self.proj(self.pool(self.encode_frames(x), mask))
        return F.normalize(e, dim=-1) if self.normalize else e
