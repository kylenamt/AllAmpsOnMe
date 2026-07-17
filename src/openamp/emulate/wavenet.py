"""FiLM-conditioned NAM A2 WaveNet for one-to-many amp emulation.

The exact architecture of the corpus's own captures: every A2 (Slimmable NAM)
export on TONE3000 shares one full-width WaveNet — 23 dilated conv layers
(channels 8, the kernel/dilation schedule below), LeakyReLU(0.01), a per-layer
input mixin of the raw signal, a residual 1x1 per layer, summed skip outputs
through a kernel-16 head conv, scaled by ``head_scale``. Verified identical
across all 774 A2 captures in the corpus; the schedule constants here are read
straight from those exports.

Two deliberate departures from a capture make it one-to-many and trainable here:

- **FiLM conditioning**: NAM's A2 layer schema defines optional FiLM hooks (all
  inactive in single-device captures). This model activates the pre-activation
  position, driven by a learnable ``[N_devices x embedding_dim]`` embedding table
  — the same conditioning mechanism and placement (post-conv, pre-nonlinearity)
  as :class:`~openamp.emulate.tcn.FiLMTCN`. FiLM starts near identity (scale 1,
  shift 0), so an untrained net is a pure A2 WaveNet.
- **Causal same-length padding**: NAM uses valid convs and trims; here every conv
  is left-padded so output length equals input length, matching the training
  loop's contract (loss on ``out[..., R:]``). The two are sample-identical once
  the receptive field is warmed up.

Import is torch-only, like :mod:`~openamp.emulate.tcn`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["FiLMWaveNet", "A2_KERNEL_SIZES", "A2_DILATIONS", "A2_HEAD_KERNEL"]

# The one architecture shared by every A2 capture in the corpus (full-width
# submodel of the SlimmableContainer export): three 7-layer dilation runs at
# kernel 6 with two kernel-15 layers between runs 2 and 3.
A2_KERNEL_SIZES = (6,) * 14 + (15, 15) + (6,) * 7
A2_DILATIONS = (1, 3, 7, 17, 41, 101, 239) * 2 + (1, 13) + (1, 3, 7, 17, 41, 101, 239)
A2_HEAD_KERNEL = 16
A2_CHANNELS = 8
A2_HEAD_SCALE = 0.02


class FiLMWaveNetLayer(nn.Module):
    """One A2 WaveNet layer, causal, with device FiLM at the pre-activation hook.

    ``z = conv(x) + mixin(clean)`` is FiLM-modulated by the device embedding,
    then LeakyReLU. The activation feeds two paths: ``layer1x1`` back onto the
    residual trunk, and (head1x1 is inactive in A2) directly out as this layer's
    skip contribution to the head sum.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 embedding_dim: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation          # causal left pad
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.input_mixer = nn.Conv1d(1, channels, 1, bias=False)
        self.film = nn.Linear(embedding_dim, 2 * channels)
        # Near-identity start (scale ~1, shift ~0): the untrained net is a plain
        # A2 WaveNet, but the small weight keeps conditioning gradients alive.
        nn.init.normal_(self.film.weight, std=0.01)
        with torch.no_grad():
            self.film.bias[:channels] = 1.0
            self.film.bias[channels:] = 0.0
        self.act = nn.LeakyReLU(0.01)
        self.layer1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` trunk [B,C,T], ``cond`` clean signal [B,1,T], ``emb`` [B,E]."""
        z = self.conv(F.pad(x, (self.pad, 0))) + self.input_mixer(cond)
        gamma, beta = self.film(emb).chunk(2, dim=-1)    # [B, C], [B, C]
        z = gamma.unsqueeze(-1) * z + beta.unsqueeze(-1)
        z = self.act(z)
        return x + self.layer1x1(z), z                   # (residual, head term)


class FiLMWaveNet(nn.Module):
    """Conditioned A2-WaveNet emulator: ``(audio, device_idx) -> emulated audio``.

    Same contract as :class:`~openamp.emulate.tcn.FiLMTCN`: input ``[B, T]`` or
    ``[B, 1, T]`` mono at 48 kHz, ``device_idx`` a ``[B]`` long tensor of
    embedding rows, output ``[B, 1, T]`` causal. ``head_scale`` is a **fixed**
    buffer, as in NAM's own trainer: made learnable, Adam moves this single
    global-gain scalar ~lr per step — the fastest parameter in the model — and
    drives it to the silence solution (output 0, ESR 1.0) in the first epochs
    before anything else can fit (observed 0.02 -> -2e-4 by epoch 7).
    The kernel/dilation schedule defaults to the A2 constants; ``channels``
    (``wn_channels`` in config) is the width sweep knob.
    """

    def __init__(self, *, n_devices: int, channels: int = A2_CHANNELS,
                 embedding_dim: int = 64,
                 kernel_sizes: tuple[int, ...] = A2_KERNEL_SIZES,
                 dilations: tuple[int, ...] = A2_DILATIONS,
                 head_kernel: int = A2_HEAD_KERNEL,
                 head_scale: float = A2_HEAD_SCALE):
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError(f"kernel_sizes ({len(kernel_sizes)}) and dilations "
                             f"({len(dilations)}) must have the same length")
        self.n_devices = int(n_devices)
        self.channels = int(channels)
        self.embedding_dim = int(embedding_dim)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.dilations = tuple(int(d) for d in dilations)
        self.head_kernel = int(head_kernel)
        self.head_pad = self.head_kernel - 1             # causal left pad

        self.embedding = nn.Embedding(self.n_devices, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

        self.rechannel = nn.Conv1d(1, self.channels, 1, bias=False)
        self.layers = nn.ModuleList(
            FiLMWaveNetLayer(self.channels, k, d, self.embedding_dim)
            for k, d in zip(self.kernel_sizes, self.dilations))
        self.head_rechannel = nn.Conv1d(self.channels, 1, self.head_kernel)
        self.register_buffer("head_scale", torch.tensor(float(head_scale)))

    @classmethod
    def from_config(cls, ecfg, n_devices: int) -> "FiLMWaveNet":
        """Construct from an :class:`~openamp.core.config.EmulateConfig`."""
        return cls(n_devices=n_devices, channels=ecfg.wn_channels,
                   embedding_dim=ecfg.embedding_dim)

    @staticmethod
    def compute_receptive_field(kernel_sizes=A2_KERNEL_SIZES, dilations=A2_DILATIONS,
                                head_kernel: int = A2_HEAD_KERNEL) -> int:
        """Receptive field in samples for a schedule (before building the module)."""
        rf = 1 + (head_kernel - 1)
        for k, d in zip(kernel_sizes, dilations):
            rf += (k - 1) * d
        return rf

    @property
    def receptive_field(self) -> int:
        return self.compute_receptive_field(self.kernel_sizes, self.dilations,
                                            self.head_kernel)

    def forward(self, x: torch.Tensor, device_idx: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)                           # [B, T] -> [B, 1, T]
        emb = self.embedding(device_idx)                 # [B, embedding_dim]
        h = self.rechannel(x)
        head = torch.zeros_like(h)
        for layer in self.layers:
            h, head_term = layer(h, x, emb)              # cond = the raw clean signal
            head = head + head_term
        y = self.head_rechannel(F.pad(head, (self.head_pad, 0)))
        return self.head_scale * y                       # [B, 1, T]
