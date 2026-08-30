"""The NAM A2 WaveNet topology, in its three conditioned variants.

- Same architecture as the corpus's A2 (Slimmable NAM) captures: 23 dilated
  conv layers (channels 8), LeakyReLU, per-layer input mixin, residual 1x1,
  skip sum through a kernel-16 head conv. wn_activation also allows tanh.
- FiLM conditioning (pre-activation, near-identity init) and same-length causal
  padding are the only departures from a stock capture -- both fold away for
  export (see openamp.emulate.export).
- Three archs, differing only in how the embedding conditions a layer:
  FiLMWaveNet / MLPFiLMWaveNet (linear vs MLP gamma/beta generator) and
  DeltaWaveNet (low-rank residual on the conv weights instead of FiLM).
- All share one [N_devices x embedding_dim] table.

Each class carries the three arch-aware descriptors the rest of the codebase
used to hardcode elsewhere -- ``describe`` (log/CSV labels) and
``export_descriptor`` (the plugin bundle's ``arch`` block); see
:mod:`openamp.models` for the contract.

Constructor parameter names *are* the config's ``model:`` keys -- a ModelSpec's
kwargs reach ``__init__`` untouched, so nothing here reads a config object.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from openamp.models.blocks import (A2_ACTIVATION, A2_COND_ACTIVATION, A2_COND_HIDDEN,
                                   A2_DELTA_RANK, A2_DELTA_SCALE_INIT, A2_EMBEDDING_STD,
                                   NAM_ACTIVATION_NAMES, DeltaWeightGen, MLPFiLM,
                                   NormalizedEmbedding, init_delta_zero,
                                   init_film_identity, make_activation,
                                   normalize_activation, per_sample_conv1d)

__all__ = ["FiLMWaveNet", "MLPFiLMWaveNet", "DeltaWaveNet",
           "FiLMWaveNetLayer", "DeltaWaveNetLayer",
           "A2_KERNEL_SIZES", "A2_DILATIONS", "A2_HEAD_KERNEL", "A2_CHANNELS",
           "A2_HEAD_SCALE", "A2_FILM_INIT_STD"]

# The one architecture shared by every A2 capture: three 7-layer dilation runs
# at kernel 6, two kernel-15 layers between runs 2 and 3.
A2_KERNEL_SIZES = (6,) * 14 + (15, 15) + (6,) * 7
A2_DILATIONS = (1, 3, 7, 17, 41, 101, 239) * 2 + (1, 13) + (1, 3, 7, 17, 41, 101, 239)
A2_HEAD_KERNEL = 16
A2_CHANNELS = 8
A2_HEAD_SCALE = 0.02
A2_FILM_INIT_STD = 0.01              # near-identity FiLM start

# --- FiLM family: film_wavenet / mlpfilm_wavenet --------------------------------

class FiLMWaveNetLayer(nn.Module):
    """One A2 WaveNet layer, causal, with device FiLM at the pre-activation hook.

    z = conv(x) + mixin(clean), FiLM-modulated by the device embedding, then
    activation. Feeds layer1x1 back onto the trunk and out as the skip term.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 embedding_dim: int, activation: str = A2_ACTIVATION,
                 cond_hidden: int = 0,
                 cond_activation: str = A2_COND_ACTIVATION):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation          # causal left pad
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.input_mixer = nn.Conv1d(1, channels, 1, bias=False)
        # cond_hidden 0 -> linear generator (film_wavenet); >0 -> per-layer MLP
        # (mlpfilm_wavenet). Conditioning stays per-channel affine either way.
        self.film = (nn.Linear(embedding_dim, 2 * channels) if cond_hidden <= 0 else
                     MLPFiLM(embedding_dim, channels, cond_hidden, cond_activation))
        init_film_identity(self.film, channels, std=A2_FILM_INIT_STD)   # near-identity start
        self.act = make_activation(activation)
        self.layer1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x trunk [B,C,T], cond clean signal [B,1,T], emb [B,E]."""
        z = self.conv(F.pad(x, (self.pad, 0))) + self.input_mixer(cond)
        gamma, beta = self.film(emb).chunk(2, dim=-1)    # [B, C], [B, C]
        z = gamma.unsqueeze(-1) * z + beta.unsqueeze(-1)
        z = self.act(z)
        return x + self.layer1x1(z), z                   # (residual, head term)


class FiLMWaveNet(NormalizedEmbedding, nn.Module):
    """Conditioned A2-WaveNet emulator: (audio, device_idx) -> emulated audio.

    Input is [B,T] or [B,1,T] mono @ 48kHz, device_idx a [B] long tensor of
    embedding rows, output [B,1,T] causal.
    """

    arch = "film_wavenet"

    def __init__(self, *, n_devices: int, channels: int = A2_CHANNELS,
                 embedding_dim: int = 64,
                 activation: str = A2_ACTIVATION,
                 cond_hidden: int = 0,
                 cond_activation: str = A2_COND_ACTIVATION,
                 kernel_sizes: tuple[int, ...] = A2_KERNEL_SIZES,
                 dilations: tuple[int, ...] = A2_DILATIONS,
                 head_kernel: int = A2_HEAD_KERNEL,
                 head_scale: float = A2_HEAD_SCALE,
                 embed_normalize: bool = False):
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError(f"kernel_sizes ({len(kernel_sizes)}) and dilations "
                             f"({len(dilations)}) must have the same length")
        self.n_devices = int(n_devices)
        self.channels = int(channels)
        self.embedding_dim = int(embedding_dim)
        self.activation = normalize_activation(activation)
        self.cond_hidden = int(cond_hidden)
        self.cond_activation = normalize_activation(cond_activation,
                                                    field="cond_activation")
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.dilations = tuple(int(d) for d in dilations)
        self.head_kernel = int(head_kernel)
        self.head_pad = self.head_kernel - 1             # causal left pad

        self.embedding = nn.Embedding(self.n_devices, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, std=A2_EMBEDDING_STD)
        self._init_embed_normalize(embed_normalize, self.embedding_dim, A2_EMBEDDING_STD)

        self.rechannel = nn.Conv1d(1, self.channels, 1, bias=False)
        self.layers = nn.ModuleList(self._make_layer(k, d)
                                    for k, d in zip(self.kernel_sizes, self.dilations))
        self.head_rechannel = nn.Conv1d(self.channels, 1, self.head_kernel)
        # Fixed buffer, not learnable: as a free param this single global gain
        # races to the silence solution (output 0) before anything else can fit.
        self.register_buffer("head_scale", torch.tensor(float(head_scale)))

    def _make_layer(self, kernel_size: int, dilation: int) -> nn.Module:
        """The per-layer module -- the one thing a WaveNet arch here overrides."""
        return FiLMWaveNetLayer(self.channels, kernel_size, dilation,
                                self.embedding_dim, self.activation,
                                self.cond_hidden, self.cond_activation)

    @torch.no_grad()
    def neutralize_conditioning(self) -> None:
        """Make the conditioning a no-op for any device_idx (export's fold copy)."""
        for layer in self.layers:
            if hasattr(layer, "delta"):
                init_delta_zero(layer.delta)
            elif hasattr(layer, "film"):
                init_film_identity(layer.film, self.channels)

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

    # --- Self-description -------------------------------------------------------
    def _topology_tag(self) -> str:
        """The generator/activation suffix on the topology string.

        A non-capture activation tags onto the WaveNet's topology
        (``a2-23L-tanh``) rather than getting its own column, so activation runs
        stay distinguishable in the existing comparison table. Each conditioning
        generator prefixes its own knob the same way (``a2-23L-mlp32``,
        ``a2-23L-delta8``), leaving the two film_wavenet strings untouched so old
        rows stay comparable.
        """
        return "" if self.activation == A2_ACTIVATION else f"-{self.activation}"

    def describe(self) -> dict:
        """Arch descriptors for logs, ``metrics.json`` and the comparison CSV.

        ``channels``/``blocks_x_layers`` mean the arch's own size knobs, so CSV
        columns stay comparable across architectures.
        """
        return {"arch": self.arch, "channels": self.channels,
                "blocks_x_layers": f"a2-{len(self.dilations)}L{self._topology_tag()}"}

    def _export_generator(self) -> dict:
        """This arch's conditioning-generator entries in the bundle descriptor.

        Per-arch only, so ``film_wavenet_a2`` bundles stay byte-identical to
        every one already shipped.
        """
        return {}

    def export_descriptor(self) -> dict:
        """The plugin bundle's ``arch`` block. Key order is part of the format."""
        return {"type": f"{self.arch}_a2",
                "channels": self.channels,
                "embedding_dim": self.embedding_dim,
                # NAM's own spelling, so the plugin can hand it straight to
                # NeuralAmpModelerCore; absent in pre-knob bundles = LeakyReLU.
                "activation": NAM_ACTIVATION_NAMES[self.activation],
                **self._export_generator(),
                "kernel_sizes": list(self.kernel_sizes),
                "dilations": list(self.dilations),
                "head_kernel": self.head_kernel,
                "head_scale": float(self.head_scale),
                "receptive_field": self.receptive_field}

    # --- Forward ----------------------------------------------------------------
    def forward_emb(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Conditioned forward on an embedding vector [B, embedding_dim], not a table row.

        Split from forward() so emb can come from elsewhere -- a jointly-trained
        encoder (openamp.joint_model), or a morph point between two rows.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)                           # [B, T] -> [B, 1, T]
        h = self.rechannel(x)
        head = torch.zeros_like(h)
        for layer in self.layers:
            h, head_term = layer(h, x, emb)              # x = the raw clean signal
            head = head + head_term
        y = self.head_rechannel(F.pad(head, (self.head_pad, 0)))
        return self.head_scale * y                       # [B, 1, T]

    def forward(self, x: torch.Tensor, device_idx: torch.Tensor) -> torch.Tensor:
        return self.forward_emb(x, self.embed(device_idx))


class MLPFiLMWaveNet(FiLMWaveNet):
    """FiLMWaveNet whose per-layer FiLM generator is a small MLP instead of one Linear.

    Separate arch (not a knob on film_wavenet) since the two have incompatible
    state_dicts -- keeps every existing film_wavenet checkpoint/bundle exact.
    """

    arch = "mlpfilm_wavenet"

    def __init__(self, *, cond_hidden: int = A2_COND_HIDDEN,
                 cond_activation: str = A2_COND_ACTIVATION, **kwargs):
        if int(cond_hidden) < 1:
            raise ValueError("cond_hidden must be >= 1 for mlpfilm_wavenet "
                             f"(0 is film_wavenet's linear FiLM), got {cond_hidden}")
        super().__init__(cond_hidden=int(cond_hidden),
                         cond_activation=cond_activation, **kwargs)

    def _topology_tag(self) -> str:
        cact = ("" if self.cond_activation == A2_COND_ACTIVATION
                else f"-c{self.cond_activation}")
        return f"-mlp{self.cond_hidden}{cact}{super()._topology_tag()}"

    def _export_generator(self) -> dict:
        return {"cond_hidden": self.cond_hidden,
                "cond_activation": NAM_ACTIVATION_NAMES[self.cond_activation]}


# --- Delta family: delta_wavenet -------------------------------------------------

class DeltaWaveNetLayer(nn.Module):
    """One A2 WaveNet layer, causal, conditioned by a per-device delta on its conv.

    Same signature/audio path as FiLMWaveNetLayer, one step earlier:

        z = conv_{W + dW(e), b + db(e)}(x) + (mixin_w + dm(e)) * clean

    Strictly contains the FiLM hook (dW = (gamma-1)*W row-wise, etc.), but can
    also rotate a kernel, not just re-gain it. layer1x1 stays shared/unconditioned.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 embedding_dim: int, rank: int = A2_DELTA_RANK,
                 activation: str = A2_ACTIVATION):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.pad = (kernel_size - 1) * dilation          # causal left pad
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.input_mixer = nn.Conv1d(1, channels, 1, bias=False)
        # Sized off this layer's own kernel std, so the delta starts at the
        # same fraction of W in every layer despite the schedule's varying fan-in.
        self.delta = DeltaWeightGen(embedding_dim, channels, kernel_size, rank,
                                    scale=A2_DELTA_SCALE_INIT,
                                    weight_std=float(self.conv.weight.detach().std()))
        self.act = make_activation(activation)
        self.layer1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x trunk [B,C,T], cond clean signal [B,1,T], emb [B,E]."""
        C = self.channels
        dw, db, dm = self.delta(emb)                     # [B,C,C,K], [B,C], [B,C]
        z = per_sample_conv1d(x, self.conv.weight.unsqueeze(0) + dw,
                              self.conv.bias.unsqueeze(0) + db,
                              self.pad, self.dilation)
        # Per-channel gain on the raw signal as a broadcast multiply so the
        # delta can ride on it; cast to z's dtype first since autocast only
        # rewrites conv/matmul (a bare fp32 multiply would promote the layer back).
        mix = (self.input_mixer.weight.reshape(1, C, 1) + dm.unsqueeze(-1)).to(z.dtype)
        z = self.act(z + mix * cond.to(z.dtype))
        return x + self.layer1x1(z), z                   # (residual, head term)


class DeltaWaveNet(FiLMWaveNet):
    """A2 WaveNet conditioned by a per-device residual on each layer's conv weights.

    Same topology/receptive field/embedding table/forward as FiLMWaveNet;
    layers are DeltaWaveNetLayer instead of FiLMWaveNetLayer. delta_rank is the
    capacity knob; cond_* doesn't apply (no FiLM generator). Separate arch
    (not a knob) since its state_dict has no `film` and vice versa.
    """

    arch = "delta_wavenet"

    def __init__(self, *, delta_rank: int = A2_DELTA_RANK, **kwargs):
        if int(delta_rank) < 1:
            raise ValueError(f"delta_rank must be >= 1, got {delta_rank}")
        # Set before nn.Module.__init__: legal for a plain int, and necessary
        # since the base constructor calls _make_layer, which reads it.
        self.delta_rank = int(delta_rank)
        super().__init__(**kwargs)

    def _make_layer(self, kernel_size: int, dilation: int) -> nn.Module:
        return DeltaWaveNetLayer(self.channels, kernel_size, dilation,
                                 self.embedding_dim, self.delta_rank, self.activation)

    def _topology_tag(self) -> str:
        return f"-delta{self.delta_rank}{super()._topology_tag()}"

    def _export_generator(self) -> dict:
        return {"delta_rank": self.delta_rank}
