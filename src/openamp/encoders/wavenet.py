"""WaveNet reference encoder: (dry, wet) -> one embedding vector.

Reads a short reference recording of an amp and emits the vector that
FiLMWaveNet consumes, replacing the per-device row it looks up today. Trained
jointly with the generator, from scratch (see guidelines.md §3).

Differs from the generator it feeds:

- Two input channels (dry, wet) -- observes the transfer function directly.
  Input is unnormalized: amplitude is gain/compression information.
- Non-causal: every conv is centered (offline reference), not left-padded.
- No head, no conditioning: the residual stream (B, C_enc, L) is pooled
  instead of collapsed to one audio sample. It IS the condition.

Not a FiLMWaveNet subclass (would inherit the embedding table and causal
padding, the two things it exists not to have); shares its schedule constants
and activation factory.

Two pooling heads, both from :mod:`openamp.encoders.pooling`:
- head="stats": AttentiveStatsPool over the final trunk -> MLP to
  embedding_dim. Pooled width is 2*C_enc, tying readout width to backbone width.
- head="multitap" (MultiTapStatsHead): taps from several depths + a 1x1
  channel expansion + K attention heads, decoupling pooled width
  (K*2*expand_dim) from the backbone entirely.

No normalization in the layer array (the A2 schedule has none); the multitap
head's expansion norm is the one exception -- see MultiTapStatsHead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from openamp.encoders.pooling import (ENC_ATTN_HIDDEN, ENC_EXPAND_DIM, ENC_EXPAND_NORM,
                                      ENC_N_HEADS, ENC_NORM_GROUPS, ENC_TAP_STRIDE,
                                      AttentiveStatsPool, MultiTapStatsHead)
from openamp.models import (A2_ACTIVATION, A2_DILATIONS, A2_KERNEL_SIZES,
                            make_activation, normalize_activation)

__all__ = ["WaveNetEncoder", "WaveNetEncoderLayer",
           "ENC_CHANNELS", "ENC_PROJ_HIDDEN", "ENC_HEADS", "ENC_TAPS"]

ENC_CHANNELS = 32          # guidelines §7 puts C_enc at 16-32
ENC_PROJ_HIDDEN = 128      # hidden width of the pooled -> De projection

ENC_HEADS = ("stats", "multitap")
# End of each of the A2 schedule's three dilation runs (7+7+2+7 layers) -- early
# taps see ~ms structure, the last sees the full 132 ms (sag/compression time
# constants), so pooling only the deepest view discards the short-scale one.
ENC_TAPS = (6, 13, 22)


class WaveNetEncoderLayer(nn.Module):
    """One non-causal A2-schedule layer: (trunk, reference) -> (trunk, skip).

    Mirrors FiLMWaveNetLayer minus the FiLM hook (conv + mixer(reference),
    activation, residual), with the causal left pad replaced by a centered
    one. Extra padding on an even kernel goes on the right -- an arbitrary
    half-sample convention; nothing downstream depends on it since the output
    is pooled over time.
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


class WaveNetEncoder(nn.Module):
    """(B, 2, L_ref) -> (B, embedding_dim).

    normalize L2-projects e onto the unit sphere. Off by default so the module
    stays neutral, but joint configs turn it on: nothing else bounds |e|, and
    an unbounded |e| is a cheap way for the reconstruction gradient to grow
    the generator's FiLM gain until it saturates and conditioning collapses.
    Normalizing removes the direction rather than merely discouraging it.

    grad_checkpoint recomputes each layer's activations in backward instead of
    storing them -- a WaveNet never downsamples, so a reference stays full
    width through all 23 layers; this is the escape hatch when that doesn't
    fit alongside the generator. Wraps the layer array only (multitap's
    tap_stride already handles the expensive part).

    head picks the pooling head (see MultiTapStatsHead). The two heads have
    disjoint parameters, so a checkpoint from one cannot load under the other
    -- enc_head is structural, and train.py treats it that way.
    """

    ref_mode = "audio"                                   # a real (dry, wet) window

    @classmethod
    def provenance(cls, jcfg) -> tuple[str, dict]:
        """``(sha256, meta)`` of the precomputed data this arm conditions on.

        Nothing here: the encoder is trained from scratch on the corpus the
        checkpoint already records. The lookup arms override it, since their
        data can be regenerated out from under a finished run without anything
        about the checkpoint's shape noticing.
        """
        return "", {}

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
