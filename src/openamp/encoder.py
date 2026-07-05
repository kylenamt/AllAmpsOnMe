"""1-D convolutional audio-effects encoder (Phase 3 spec §2).

SimCLR-style encoder over raw 48 kHz mono audio. A stem conv is followed by ~6
residual conv blocks (kernel 5) whose strides downsample the 2 s crop while the
channel count grows 16 -> 64; a global average pool over time yields the 64-d
**pre-projection** embedding used by every downstream probe/eval. A small MLP
projection head sits on top **for training only** (SimCLR discards it at eval).

Everything is config-driven from :class:`train.config.ModelConfig` so the layer
plan lives in ``configs/phase3.yaml``. Import is torch-only (no data deps), and
``python -m models.encoder`` prints the parameter count for the current config.
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock1d(nn.Module):
    """Pre-activation-free residual block: two k-conv/BN/GELU layers + skip.

    ``stride`` downsamples in the first conv; a 1x1 conv adapts the skip whenever
    the channel count or temporal length changes.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.GELU()
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + residual)


class ConvEncoder(nn.Module):
    """Raw-audio -> ``embed_dim`` embedding (spec §2).

    Input is ``[B, T]`` or ``[B, 1, T]`` mono audio at 48 kHz; output is a
    L2-normalizable ``[B, embed_dim]`` embedding (global-avg-pooled over time).
    """

    def __init__(self, *, stem_channels: int = 16, stem_stride: int = 4,
                 kernel_size: int = 5, blocks: list | None = None,
                 embed_dim: int = 64, in_channels: int = 1):
        super().__init__()
        blocks = blocks if blocks is not None else [[16, 4], [16, 4], [32, 4],
                                                    [48, 4], [56, 2], [64, 2]]
        pad = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size, stride=stem_stride,
                      padding=pad, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        layers = []
        c_in = stem_channels
        for c_out, stride in blocks:
            layers.append(ResBlock1d(c_in, int(c_out), kernel_size, int(stride)))
            c_in = int(c_out)
        self.blocks = nn.Sequential(*layers)
        self.out_channels = c_in
        # A final linear maps the pooled block output to embed_dim (identity when
        # they already match, which is the default 64 -> 64 case).
        self.proj = nn.Identity() if c_in == embed_dim else nn.Linear(c_in, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)              # [B, T] -> [B, 1, T]
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)                 # global average pool over time
        return self.proj(x)


class ProjectionHead(nn.Module):
    """Training-only 2-layer MLP projection head (SimCLR); discarded at eval."""

    def __init__(self, in_dim: int = 64, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_encoder(model_cfg) -> ConvEncoder:
    """Construct :class:`ConvEncoder` from a :class:`train.config.ModelConfig`."""
    return ConvEncoder(
        stem_channels=model_cfg.stem_channels,
        stem_stride=model_cfg.stem_stride,
        kernel_size=model_cfg.kernel_size,
        blocks=[list(b) for b in model_cfg.blocks],
        embed_dim=model_cfg.embed_dim,
    )


def build_projection_head(model_cfg) -> ProjectionHead:
    return ProjectionHead(model_cfg.embed_dim, model_cfg.proj_hidden, model_cfg.proj_dim)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":  # pragma: no cover - manual param/shape check
    from train.config import load_config

    cfg = load_config()
    enc = build_encoder(cfg.model)
    head = build_projection_head(cfg.model)
    n = int(round(cfg.train.crop_seconds * 48_000))
    x = torch.randn(4, n)
    z = enc(x)
    print(f"encoder params : {count_parameters(enc):,}")
    print(f"proj head params: {count_parameters(head):,} (training only)")
    print(f"input {tuple(x.shape)} -> embedding {tuple(z.shape)}")
