"""Fully-parametric FiLM-conditioned causal TCN for amp emulation (spec §4.1).

One model emulates *all* devices: a learnable ``[N_devices x embedding_dim]``
embedding table conditions every layer through FiLM (per-channel scale + shift),
so the shared convolution stack is steered per device. Every architecture size —
``blocks``, ``layers_per_block``, ``channels``, ``kernel_size``,
``dilation_growth``, ``embedding_dim`` — is a plain constructor argument fed from
:class:`~openamp.core.config.EmulateConfig`; no code changes for any size sweep.

The stack is **causal**: each dilated conv is left-padded by ``(k-1)*dilation`` so
output length equals input length and no sample sees its future. The receptive
field (logged at startup, used by the dataset for real left-context warmup) is
``1 + blocks * sum_l (k-1) * dilation_growth**l``. Import is torch-only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["FiLMTCN", "count_parameters"]


def count_parameters(module: nn.Module) -> int:
    """Trainable parameter count of ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class FiLMTCNLayer(nn.Module):
    """One causal dilated conv + FiLM(scale, shift) + PReLU, with a residual.

    ``film`` maps the per-device embedding to a ``(gamma, beta)`` pair per output
    channel; the conv output is affine-modulated before the nonlinearity. Channels
    are constant across the stack so the residual is a plain identity add.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 embedding_dim: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation          # causal left pad
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.film = nn.Linear(embedding_dim, 2 * channels)
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (self.pad, 0))                      # left-only -> causal, same T
        y = self.conv(y)
        gamma, beta = self.film(emb).chunk(2, dim=-1)    # [B, C], [B, C]
        y = gamma.unsqueeze(-1) * y + beta.unsqueeze(-1)
        y = self.act(y)
        return x + y                                     # identity residual


class FiLMTCN(nn.Module):
    """Conditioned amp emulator: ``(audio, device_idx) -> emulated audio``.

    Input is ``[B, T]`` or ``[B, 1, T]`` mono at 48 kHz; ``device_idx`` is a
    ``[B]`` long tensor of embedding-table rows. Output is ``[B, 1, T]`` (same
    length as the input, causal). Pass a permuted ``device_idx`` to probe that
    conditioning actually matters (the mini-run sanity check, spec §2).
    """

    def __init__(self, *, n_devices: int, blocks: int = 2, layers_per_block: int = 8,
                 channels: int = 16, kernel_size: int = 3, dilation_growth: int = 2,
                 embedding_dim: int = 64):
        super().__init__()
        self.n_devices = int(n_devices)
        self.blocks = int(blocks)
        self.layers_per_block = int(layers_per_block)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation_growth = int(dilation_growth)
        self.embedding_dim = int(embedding_dim)

        self.embedding = nn.Embedding(self.n_devices, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

        self.input = nn.Conv1d(1, self.channels, 1)      # 1x1: lift mono -> channels
        layers = []
        for _ in range(self.blocks):
            for lyr in range(self.layers_per_block):
                dilation = self.dilation_growth ** lyr   # resets each block
                layers.append(FiLMTCNLayer(self.channels, self.kernel_size,
                                           dilation, self.embedding_dim))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Conv1d(self.channels, 1, 1)     # 1x1: channels -> mono

    @classmethod
    def from_config(cls, ecfg, n_devices: int) -> "FiLMTCN":
        """Construct from an :class:`~openamp.core.config.EmulateConfig`."""
        return cls(n_devices=n_devices, blocks=ecfg.blocks,
                   layers_per_block=ecfg.layers_per_block, channels=ecfg.channels,
                   kernel_size=ecfg.kernel_size, dilation_growth=ecfg.dilation_growth,
                   embedding_dim=ecfg.embedding_dim)

    @staticmethod
    def compute_receptive_field(blocks: int, layers_per_block: int, kernel_size: int,
                                dilation_growth: int) -> int:
        """Receptive field in samples for a config (before building the module)."""
        rf = 1
        for _ in range(blocks):
            for lyr in range(layers_per_block):
                rf += (kernel_size - 1) * (dilation_growth ** lyr)
        return rf

    @property
    def receptive_field(self) -> int:
        return self.compute_receptive_field(self.blocks, self.layers_per_block,
                                            self.kernel_size, self.dilation_growth)

    def forward(self, x: torch.Tensor, device_idx: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)                           # [B, T] -> [B, 1, T]
        emb = self.embedding(device_idx)                 # [B, embedding_dim]
        h = self.input(x)
        for layer in self.layers:
            h = layer(h, emb)
        return self.output(h)                            # [B, 1, T]
