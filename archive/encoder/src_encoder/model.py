"""Two-branch waveform tone encoder: ``(dry, wet) -> 512-d tone vector``.

The amortized counterpart to :mod:`openamp.emulate.enroll`. Enrollment gets a
device's conditioning vector by *gradient descent* against a frozen network (30
epochs per amp); this encoder is a feed-forward map from the capture audio
itself, trained contrastively so that two segments of the same capture land in
the same place and two different captures don't.

Why the input is a **2-channel (dry, wet) pair** and not just the wet: an amp is
a transfer relation, not a sound. The dry channel is what lets the encoder see
"this input produced that output" rather than "this recording is bright". In
sweep-only training every device shares one dry file, so the dry channel carries
no device identity on its own — :mod:`openamp.encoder.evaluate` re-runs the
diagnostics with it zeroed to check the encoder actually uses it.

Two branches over one shared stem, because tone has two timescales:

- **static** (3 dilated blocks, RF 29 frames = 155 ms) — spectral character.
  Mean-pooled: tone is approximately stationary.
- **dynamic** (8 blocks, RF 1021 frames = 5.45 s, i.e. the whole segment) —
  attack/sustain/decay, compression, sag. Pooled as ``mean, std, |mean(dh)|,
  std(dh)``: the rate-of-change statistics a plain mean destroys.

Each branch head is L2-normalized **separately** before the concat, so one
branch cannot dominate the 512-d vector by norm alone. A Barlow-Twins style
cross-covariance penalty (in :mod:`openamp.encoder.train`) is what stops
the dynamic branch from simply re-learning the static one.

Everything is non-causal — this is offline encoding of a finished capture, not a
real-time path, so there is no reason to pay the causal-padding cost the
emulators (:mod:`openamp.emulate.wavenet`) do.

Architecture constants live here rather than in ``core/constants.py``, following
``wavenet.py``'s ``A2_*`` precedent: pipeline-wide facts are global, one model's
shape is the model's own business. Import is torch-only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ToneEncoder", "ProjectionHead", "BlurPool1d", "aggregate_capture",
           "STEM_KERNELS", "STEM_STRIDES", "STEM_CHANNELS", "STATIC_DILATIONS",
           "DYNAMIC_DILATIONS", "SEGMENT_SAMPLES", "STEM_DOWNSAMPLE"]

# --- Stem: strided conv front end, 256x downsample -> 187.5 Hz frame rate -------
# Layer 1 is the learned filterbank that replaces a mel front end; the rest trade
# time rate for channel depth until the frame rate matches the dynamic branch's
# receptive field. Kernels are odd so "same" padding is exact.
STEM_KERNELS = (15, 9, 9, 5)
STEM_STRIDES = (4, 4, 4, 4)
STEM_CHANNELS = (32, 64, 96, 128)
STEM_DOWNSAMPLE = 256                   # prod(STEM_STRIDES)

# Blur (anti-alias) kernel applied before every stride: naive striding aliases
# the top octave straight back down into the band the amp's harmonics live in.
BLUR_KERNEL = (1.0, 4.0, 6.0, 4.0, 1.0)

STATIC_DILATIONS = (1, 2, 4)                                  # RF 29 fr = 155 ms
DYNAMIC_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)             # RF 1021 fr = 5.45 s

NORM_GROUPS = 8                         # GroupNorm: batch-size-independent
DROPOUT = 0.1
HEAD_DIM = 256                          # per-branch output; the 512-d vector is 2x this
PROJ_HIDDEN = 256
PROJ_DIM = 128                          # SimCLR convention, contrastive only

# 5.46 s at 48 kHz. Chosen so the dynamic branch's receptive field covers a whole
# segment: a shorter segment wastes the branch, a longer one leaves frames it
# cannot see. Also ~31 segments of the 171 s NAM capture signal.
SEGMENT_SAMPLES = 262_144


def count_parameters(module: nn.Module) -> int:
    """Trainable parameter count of ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class BlurPool1d(nn.Module):
    """Fixed low-pass filter + stride (anti-aliased downsampling).

    The kernel is a normalized binomial window held in a **buffer**, not a
    parameter — same call as ``A2_HEAD_SCALE`` in ``wavenet.py``: it is a fixed
    signal-processing fact, and making it learnable just invites the network to
    unlearn the anti-aliasing it exists to provide. Depthwise (``groups=c``), so
    it costs one multiply-add per channel per tap.
    """

    def __init__(self, channels: int, stride: int, kernel=BLUR_KERNEL):
        super().__init__()
        self.channels = int(channels)
        self.stride = int(stride)
        k = torch.tensor(kernel, dtype=torch.float32)
        k = k / k.sum()                                  # unit DC gain
        self.pad = (len(kernel) - 1) // 2
        self.register_buffer("kernel", k.view(1, 1, -1).repeat(self.channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.kernel.to(x.dtype), stride=self.stride,
                        padding=self.pad, groups=self.channels)


class StemBlock(nn.Module):
    """Conv (stride 1, same) -> GroupNorm -> GELU -> blur-pooled stride."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int,
                 norm_groups: int = NORM_GROUPS):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(norm_groups, c_out)
        self.act = nn.GELU()
        self.pool = BlurPool1d(c_out, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.norm(self.conv(x))))


class DilatedBlock(nn.Module):
    """2x [dilated conv -> GroupNorm -> GELU -> dropout] with an identity residual.

    Non-causal: symmetric padding of ``dilation`` keeps the length exact for
    ``kernel_size=3``. Channels are constant across the stack so the residual
    needs no projection.
    """

    def __init__(self, channels: int, dilation: int, *, kernel_size: int = 3,
                 dropout: float = DROPOUT, norm_groups: int = NORM_GROUPS):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad),
            nn.GroupNorm(norm_groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad),
            nn.GroupNorm(norm_groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _masked_mean(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over time, honouring an optional ``[B, T]`` validity mask."""
    if mask is None:
        return h.mean(dim=-1)
    m = mask.unsqueeze(1).to(h.dtype)
    return (h * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)


class Branch(nn.Module):
    """Dilated stack + pooling + 2-layer head -> an L2-normalized ``head_dim`` vector.

    ``pooling="mean"`` (static) emits ``C`` features; ``pooling="stats"``
    (dynamic) emits ``4C`` — ``mean, std, |mean(dh)|, std(dh)`` where ``dh`` is
    the first difference along time. The delta terms are the point of the dynamic
    branch: a mean over 5.46 s throws away exactly the envelope behaviour that
    distinguishes a compressing amp from a stiff one.
    """

    def __init__(self, channels: int, dilations, *, pooling: str, head_dim: int = HEAD_DIM,
                 dropout: float = DROPOUT, norm_groups: int = NORM_GROUPS):
        super().__init__()
        if pooling not in ("mean", "stats"):
            raise ValueError(f"pooling must be 'mean' or 'stats', got {pooling!r}")
        self.pooling = pooling
        self.dilations = tuple(int(d) for d in dilations)
        self.blocks = nn.ModuleList([
            DilatedBlock(channels, d, dropout=dropout, norm_groups=norm_groups)
            for d in self.dilations
        ])
        pooled = channels if pooling == "mean" else 4 * channels
        self.head = nn.Sequential(
            nn.Linear(pooled, head_dim), nn.GELU(), nn.Linear(head_dim, head_dim),
        )

    @staticmethod
    def compute_receptive_field(dilations, kernel_size: int = 3) -> int:
        """Receptive field in **frames** (each block is 2 convs at one dilation)."""
        return 1 + 2 * sum((kernel_size - 1) * int(d) for d in dilations)

    @property
    def receptive_field(self) -> int:
        return self.compute_receptive_field(self.dilations)

    def pool(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.pooling == "mean":
            return _masked_mean(h, mask)
        d = h[..., 1:] - h[..., :-1]
        dmask = None if mask is None else (mask[:, 1:] * mask[:, :-1])
        return torch.cat([
            _masked_mean(h, mask),
            h.std(dim=-1),
            _masked_mean(d, dmask).abs(),
            d.std(dim=-1),
        ], dim=-1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return F.normalize(self.head(self.pool(h, mask)), dim=-1)


class ProjectionHead(nn.Module):
    """Training-only 2-layer MLP + L2-norm (SimCLR); the generator never sees it."""

    def __init__(self, in_dim: int = HEAD_DIM, hidden: int = PROJ_HIDDEN,
                 out_dim: int = PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ToneEncoder(nn.Module):
    """``(B, 2, T) -> (emb512, static256, dynamic256)``.

    Input is the sample-aligned ``(dry, wet)`` pair at 48 kHz, **unnormalized**:
    amplitude is gain and compression information, so scaling it away would
    delete part of the target. Output ``emb512`` is the concat of the two
    separately-L2-normalized branch vectors — the vector a generator would
    consume (the paper's phi).

    ``mask`` is an optional ``[B, T_frames]`` validity mask for ragged input; the
    corpus always yields full-length segments, so it is ``None`` in practice.
    """

    def __init__(self, *, in_channels: int = 2, stem_channels=STEM_CHANNELS,
                 stem_kernels=STEM_KERNELS, stem_strides=STEM_STRIDES,
                 static_dilations=STATIC_DILATIONS, dynamic_dilations=DYNAMIC_DILATIONS,
                 head_dim: int = HEAD_DIM, proj_hidden: int = PROJ_HIDDEN,
                 proj_dim: int = PROJ_DIM, dropout: float = DROPOUT,
                 norm_groups: int = NORM_GROUPS):
        super().__init__()
        stem_channels = tuple(int(c) for c in stem_channels)
        stem_kernels = tuple(int(k) for k in stem_kernels)
        stem_strides = tuple(int(s) for s in stem_strides)
        if not len(stem_channels) == len(stem_kernels) == len(stem_strides):
            raise ValueError("stem_channels / stem_kernels / stem_strides must be "
                             f"the same length, got {len(stem_channels)} / "
                             f"{len(stem_kernels)} / {len(stem_strides)}")
        self.in_channels = int(in_channels)
        self.stem_channels = stem_channels
        self.stem_strides = stem_strides
        self.head_dim = int(head_dim)

        stem, c_in = [], self.in_channels
        for c_out, k, s in zip(stem_channels, stem_kernels, stem_strides):
            stem.append(StemBlock(c_in, c_out, k, s, norm_groups))
            c_in = c_out
        self.stem = nn.Sequential(*stem)

        self.static = Branch(c_in, static_dilations, pooling="mean", head_dim=head_dim,
                             dropout=dropout, norm_groups=norm_groups)
        self.dynamic = Branch(c_in, dynamic_dilations, pooling="stats", head_dim=head_dim,
                              dropout=dropout, norm_groups=norm_groups)
        self.proj_static = ProjectionHead(head_dim, proj_hidden, proj_dim)
        self.proj_dynamic = ProjectionHead(head_dim, proj_hidden, proj_dim)

    @classmethod
    def from_config(cls, ncfg) -> "ToneEncoder":
        """Construct from an :class:`~openamp.core.config.EncoderConfig`."""
        return cls(stem_channels=ncfg.stem_channels, static_dilations=ncfg.static_dilations,
                   dynamic_dilations=ncfg.dynamic_dilations, head_dim=ncfg.head_dim,
                   proj_hidden=ncfg.proj_hidden, proj_dim=ncfg.proj_dim,
                   dropout=ncfg.dropout, norm_groups=ncfg.norm_groups)

    @property
    def embedding_dim(self) -> int:
        """Width of the concatenated tone vector (static + dynamic)."""
        return 2 * self.head_dim

    @property
    def downsample(self) -> int:
        """Samples per stem output frame (256 by default -> 187.5 Hz at 48 kHz)."""
        out = 1
        for s in self.stem_strides:
            out *= s
        return out

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Stem only: ``(B, 2, T) -> (B, C, T // downsample)``."""
        if x.dim() == 2:                                 # [B, T] -> single channel
            x = x.unsqueeze(1)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels (dry, wet), "
                             f"got {x.shape[1]}")
        return self.stem(x)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.encode_frames(x)
        s = self.static(h, mask)
        d = self.dynamic(h, mask)
        return torch.cat([s, d], dim=-1), s, d

    def project(self, static: torch.Tensor, dynamic: torch.Tensor):
        """Contrastive-only projections of the two branch vectors."""
        return self.proj_static(static), self.proj_dynamic(dynamic)


def aggregate_capture(static: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
    """Pool a capture's per-segment branch vectors into one ``[2 * head_dim]`` vector.

    ``static``/``dynamic`` are ``[n_segments, head_dim]``. The static side is a
    plain mean (character is stationary, so the segments are noisy copies of one
    thing); the dynamic side is ``mean + 0.5 * std``, because the *spread* across
    excitation regions — how differently the amp behaves on a quiet chirp vs a
    loud one — is itself dynamics information, not noise to average away. Both
    are renormalized so the concat keeps the per-branch unit norm that
    :class:`ToneEncoder` maintains per segment.
    """
    if static.dim() != 2 or dynamic.dim() != 2:
        raise ValueError("expected [n_segments, head_dim] tensors")
    s = F.normalize(static.mean(dim=0), dim=-1)
    n = dynamic.shape[0]
    spread = dynamic.std(dim=0) if n > 1 else torch.zeros_like(dynamic[0])
    d = F.normalize(dynamic.mean(dim=0) + 0.5 * spread, dim=-1)
    return torch.cat([s, d], dim=-1)
