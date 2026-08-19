"""FiLM-conditioned NAM A2 WaveNet for one-to-many amp emulation.

- Same architecture as the corpus's A2 (Slimmable NAM) captures: 23 dilated
  conv layers (channels 8), LeakyReLU, per-layer input mixin, residual 1x1,
  skip sum through a kernel-16 head conv. wn_activation also allows tanh.
- FiLM conditioning (pre-activation, near-identity init) and same-length causal
  padding are the only departures from a stock capture -- both fold away for
  export (see openamp.emulate.export).
- Three archs, differing only in how the embedding conditions a layer:
  FiLMWaveNet / MLPFiLMWaveNet (linear vs MLP gamma/beta generator),
  DeltaWaveNet (low-rank residual on the conv weights instead of FiLM),
  TableDeltaWaveNet (same residual, free per-device instead of low-rank --
  cannot be enrolled, no shared embedding space).
- All share one [N_devices x embedding_dim] table.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["FiLMWaveNet", "MLPFiLMWaveNet", "DeltaWaveNet", "TableDeltaWaveNet",
           "MLPFiLM", "DeltaWeightGen", "film_output_linear", "init_film_identity",
           "init_delta_zero", "per_sample_conv1d", "make_activation",
           "normalize_activation",
           "ACTIVATIONS", "NAM_ACTIVATION_NAMES", "A2_KERNEL_SIZES", "A2_DILATIONS",
           "A2_HEAD_KERNEL", "A2_ACTIVATION", "A2_COND_HIDDEN", "A2_COND_ACTIVATION",
           "A2_DELTA_RANK", "A2_DELTA_SCALE_INIT", "DELTA_PARTS_ALL",
           "DELTA_PART_NAMES"]

# The one architecture shared by every A2 capture: three 7-layer dilation runs
# at kernel 6, two kernel-15 layers between runs 2 and 3.
A2_KERNEL_SIZES = (6,) * 14 + (15, 15) + (6,) * 7
A2_DILATIONS = (1, 3, 7, 17, 41, 101, 239) * 2 + (1, 13) + (1, 3, 7, 17, 41, 101, 239)
A2_HEAD_KERNEL = 16
A2_CHANNELS = 8
A2_HEAD_SCALE = 0.02
A2_EMBEDDING_STD = 0.1               # init std of the [N_devices x E] table
A2_FILM_INIT_STD = 0.01              # near-identity FiLM start

# --- Activation (wn_activation) --------------------------------------------------
# Parameter-free, so a checkpoint's state_dict is identical either way -- in
# train.py's _STRUCTURAL_KEYS to stop a silent architecture swap on resume.
ACTIVATIONS = ("leakyrelu", "tanh")
A2_ACTIVATION = "leakyrelu"          # what every A2 capture uses
A2_LEAKY_SLOPE = 0.01
NAM_ACTIVATION_NAMES = {"leakyrelu": "LeakyReLU", "tanh": "Tanh"}   # plugin bundle spelling

# --- Nonlinear FiLM generator (cond_hidden / cond_activation) -------------------
# mlpfilm_wavenet: Linear(E,H) -> act -> Linear(H,2C) per layer instead of one
# Linear(E,2C). Never runs in the plugin DSP loop (folded per morph point), so
# H only costs training params / bundle bytes, not real-time CPU.
A2_COND_HIDDEN = 32
A2_COND_ACTIVATION = "leakyrelu"

# --- Weight-delta generator (delta_rank) -----------------------------------------
# delta_wavenet: embedding writes a residual onto each layer's conv kernel/bias/
# mixin instead of FiLM-scaling the output. Low-rank (kernel-shaped basis
# directions shared across devices, per-device coeff) so it stays a shared
# embedding space rather than a per-device weight table.
A2_DELTA_RANK = 8
# Delta size at init, as a fraction of the base kernel's weight std -- must stay
# a residual (untrained net ~= stock A2). basis is unit-norm and scale is one
# learned magnitude scalar per layer, so growth is visible/decayable instead of
# a free trade between the two (see DeltaWeightGen).
A2_DELTA_SCALE_INIT = 0.03

# tabledelta_wavenet runtime mask: which slices of a weight residual apply.
DELTA_PARTS_ALL = (True, True, True)
DELTA_PART_NAMES = ("kernel", "bias", "mixin")


# --- General-purpose helpers, shared by every arch below -----------------------

def normalize_activation(name: str, field: str = "wn_activation") -> str:
    """Validate + canonicalize an activation name; `field` names the config knob for errors."""
    key = str(name).strip().lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"{field} must be one of {ACTIVATIONS}, got {name!r}")
    return key


def make_activation(name: str, field: str = "wn_activation") -> nn.Module:
    """The activation module for a name (case-insensitive)."""
    key = normalize_activation(name, field)
    return nn.LeakyReLU(A2_LEAKY_SLOPE) if key == "leakyrelu" else nn.Tanh()


def per_sample_conv1d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                      pad: int, dilation: int) -> torch.Tensor:
    """Causal conv1d where every batch item has its own weights.

    x [B,C,T], w [B,C,C,K], b [B,C] -> [B,C,T]. Disguises the batch as extra
    channels (groups=B) so B independent per-sample convolutions run as one
    grouped conv in a single kernel launch.
    """
    B, C, T = x.shape
    K = w.shape[-1]
    return F.conv1d(F.pad(x, (pad, 0)).reshape(1, B * C, -1),
                    w.reshape(B * C, C, K), b.reshape(B * C),
                    dilation=dilation, groups=B).reshape(B, C, T)


# --- FiLM family: film_wavenet / mlpfilm_wavenet --------------------------------
# Conditioning scales the layer's output per channel (gamma/beta); the generator
# maps embedding -> (gamma, beta) linearly (FiLMWaveNet) or via an MLP (MLPFiLMWaveNet).

def film_output_linear(film: nn.Module) -> nn.Linear:
    """The FiLM generator's final nn.Linear (the one producing gamma, beta).

    Zeroing its weight makes the generator emit its bias verbatim for any
    input -- what keeps the near-identity init and export's fold-neutralization exact.
    """
    linears = [m for m in film.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise TypeError(f"FiLM generator {type(film).__name__} has no nn.Linear")
    return linears[-1]


@torch.no_grad()
def init_film_identity(film: nn.Module, channels: int, std: float = 0.0) -> None:
    """Set a FiLM generator to (near-)identity: gamma ~ 1, beta ~ 0.

    std=0: exact identity (export's neutralized fold). std>0: training init --
    identity plus a small random tilt so conditioning gradients stay alive.
    """
    out = film_output_linear(film)
    if std > 0:
        out.weight.normal_(0.0, std)
    else:
        out.weight.zero_()
    out.bias[:channels] = 1.0
    out.bias[channels:] = 0.0


class MLPFiLM(nn.Module):
    """Nonlinear embedding -> (gamma, beta) generator: Linear -> act -> Linear.

    One independent generator per layer. Folded into the conv weights at
    export, like the single Linear it replaces -- never runs in the plugin DSP loop.
    """

    def __init__(self, embedding_dim: int, channels: int, hidden: int,
                 activation: str = A2_COND_ACTIVATION):
        super().__init__()
        if hidden < 1:
            raise ValueError(f"cond_hidden must be >= 1, got {hidden}")
        self.fc1 = nn.Linear(embedding_dim, hidden)
        self.act = make_activation(activation, field="cond_activation")
        self.fc2 = nn.Linear(hidden, 2 * channels)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(emb)))


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


class FiLMWaveNet(nn.Module):
    """Conditioned A2-WaveNet emulator: (audio, device_idx) -> emulated audio.

    Same contract as FiLMTCN: input [B,T] or [B,1,T] mono @ 48kHz, device_idx a
    [B] long tensor of embedding rows, output [B,1,T] causal.
    """

    def __init__(self, *, n_devices: int, channels: int = A2_CHANNELS,
                 embedding_dim: int = 64,
                 activation: str = A2_ACTIVATION,
                 cond_hidden: int = 0,
                 cond_activation: str = A2_COND_ACTIVATION,
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

    def layer_conditioning(self, emb: torch.Tensor):
        """Yield (layer, conditioning) for one embedding vector.

        Every arch hands the same emb to every layer except TableDeltaWaveNet,
        which slices it -- lets export fold/reference-forward arch-agnostically.
        """
        for layer in self.layers:
            yield layer, emb

    @torch.no_grad()
    def neutralize_conditioning(self) -> None:
        """Make the conditioning a no-op for any device_idx (export's fold copy)."""
        for layer in self.layers:
            if hasattr(layer, "delta"):
                init_delta_zero(layer.delta)
            elif hasattr(layer, "film"):
                init_film_identity(layer.film, self.channels)

    @classmethod
    def from_config(cls, ecfg, n_devices: int) -> "FiLMWaveNet":
        """Construct from an EmulateConfig."""
        return cls(n_devices=n_devices, channels=ecfg.wn_channels,
                   embedding_dim=ecfg.embedding_dim,
                   activation=ecfg.wn_activation)

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

    def forward_emb(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Conditioned forward on an embedding vector [B, embedding_dim], not a table row.

        Split from forward() so emb can come from elsewhere -- a jointly-trained
        encoder (openamp.joint_model), or a morph point between two rows.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)                           # [B, T] -> [B, 1, T]
        h = self.rechannel(x)
        head = torch.zeros_like(h)
        for layer, cond in self.layer_conditioning(emb):
            h, head_term = layer(h, x, cond)             # x = the raw clean signal
            head = head + head_term
        y = self.head_rechannel(F.pad(head, (self.head_pad, 0)))
        return self.head_scale * y                       # [B, 1, T]

    def forward(self, x: torch.Tensor, device_idx: torch.Tensor) -> torch.Tensor:
        return self.forward_emb(x, self.embedding(device_idx))


class MLPFiLMWaveNet(FiLMWaveNet):
    """FiLMWaveNet whose per-layer FiLM generator is a small MLP instead of one Linear.

    Separate arch (not a knob on film_wavenet) since the two have incompatible
    state_dicts -- keeps every existing film_wavenet checkpoint/bundle exact.
    """

    def __init__(self, *, cond_hidden: int = A2_COND_HIDDEN,
                 cond_activation: str = A2_COND_ACTIVATION, **kwargs):
        if int(cond_hidden) < 1:
            raise ValueError("cond_hidden must be >= 1 for mlpfilm_wavenet "
                             f"(0 is film_wavenet's linear FiLM), got {cond_hidden}")
        super().__init__(cond_hidden=int(cond_hidden),
                         cond_activation=cond_activation, **kwargs)

    @classmethod
    def from_config(cls, ecfg, n_devices: int) -> "MLPFiLMWaveNet":
        """Construct from an EmulateConfig."""
        return cls(n_devices=n_devices, channels=ecfg.wn_channels,
                   embedding_dim=ecfg.embedding_dim,
                   activation=ecfg.wn_activation,
                   cond_hidden=ecfg.cond_hidden,
                   cond_activation=ecfg.cond_activation)


# --- Delta family: delta_wavenet -------------------------------------------------
# Conditioning writes a low-rank residual onto each layer's conv weights instead
# of scaling the layer's output.

@torch.no_grad()
def init_delta_zero(gen: "DeltaWeightGen", scale: float = 0.0) -> None:
    """Set a weight-delta generator to (near-)zero: dW ~ 0, db ~ 0, dm ~ 0.

    scale=0: exact zero (export's neutralized fold). scale>0: training init --
    random unit-norm basis directions at that magnitude, so coeff's gradient
    stays live from step 1 (zeroing basis instead would also zero coeff's gradient).
    """
    if scale > 0:
        gen.basis.normal_(0.0, 1.0)      # directions only; forward() normalizes them
    gen.scale.fill_(float(scale))


class DeltaWeightGen(nn.Module):
    """Embedding -> low-rank residual on one layer's conv kernel, bias and mixin.

    - delta(e) = scale * (coeff(e) @ normalize(basis)): rank unit-norm kernel-
      shaped directions shared by every device, mixed by a per-device coeff
      vector, sized by one learned scalar. rank is the capacity knob.
    - basis (direction) is split from scale (magnitude) so growth costs one
      visible, decayable parameter instead of drifting for free through coeff/basis.
    - dm (mixin gain) exists so the family strictly contains FiLM's.
    - Training-time only: export ships scale*normalize(basis) pre-multiplied as
      one matrix, so the plugin's re-fold is a plain coeff(e) @ basis matmul.
    - Linear in e (not just the network) so enrollment/morph points stay
      well-behaved functions of embedding position.
    """

    def __init__(self, embedding_dim: int, channels: int, kernel_size: int, rank: int,
                 scale: float = A2_DELTA_SCALE_INIT, weight_std: float = 1.0):
        super().__init__()
        if rank < 1:
            raise ValueError(f"delta_rank must be >= 1, got {rank}")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.rank = int(rank)
        self.n_weight = self.channels * self.channels * self.kernel_size
        self.n_out = self.n_weight + 2 * self.channels   # kernel, bias, mixin gain
        self.coeff = nn.Linear(embedding_dim, self.rank)
        self.basis = nn.Parameter(torch.empty(self.rank, self.n_out))
        self.scale = nn.Parameter(torch.empty(()))
        init_delta_zero(self, 1.0)                       # directions, unit magnitude
        self._calibrate(scale, weight_std)

    @torch.no_grad()
    def _calibrate(self, scale: float, weight_std: float) -> None:
        """Solve for `scale` so the initial delta is `scale * weight_std` wide.

        Measured against a probe batch, not derived by hand (raw magnitude
        depends on rank/n_out/nn.Linear's own init). Probe uses a private RNG
        so calibration doesn't shift later layers' init.
        """
        if scale <= 0:
            self.scale.zero_()
            return
        g = torch.Generator().manual_seed(0)
        probe = torch.randn(256, self.coeff.in_features, generator=g) * A2_EMBEDDING_STD
        raw = (self.coeff(probe) @ self.effective_basis())[:, :self.n_weight]
        self.scale.mul_(scale * weight_std / max(float(raw.std()), 1e-12))

    def effective_basis(self) -> torch.Tensor:
        """scale * normalize(basis) -- shared by forward() and the bundle export."""
        return self.scale * F.normalize(self.basis, dim=1)

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """emb [B,E] (or a bare [E] morph point) -> dW [B,C,C,K], db, dm [B,C]."""
        flat = (self.coeff(emb) @ self.effective_basis()).reshape(-1, self.n_out)
        dw, db, dm = flat.split((self.n_weight, self.channels, self.channels), dim=-1)
        return dw.view(-1, self.channels, self.channels, self.kernel_size), db, dm


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

    @classmethod
    def from_config(cls, ecfg, n_devices: int) -> "DeltaWaveNet":
        """Construct from an EmulateConfig."""
        return cls(n_devices=n_devices, channels=ecfg.wn_channels,
                   embedding_dim=ecfg.embedding_dim,
                   activation=ecfg.wn_activation,
                   delta_rank=ecfg.delta_rank)


# --- TableDelta family: tabledelta_wavenet ---------------------------------------
# Full-rank limit of the delta family: each device's residual is a free table row
# instead of a low-rank read of a shared embedding.

class TableDeltaWaveNetLayer(nn.Module):
    """One A2 layer whose weight residual is handed in, not generated.

    Same audio path as DeltaWaveNetLayer; TableDeltaWaveNet slices the device's
    row out of one big table and passes dW/db/dm in directly.

    delta_parts gates the three slices at inference (plain attribute, not in
    the state_dict, so one checkpoint can be heard every way). kernel=False
    skips the per-sample conv for a shared one plus a broadcast bias.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 activation: str = A2_ACTIVATION):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.pad = (kernel_size - 1) * dilation          # causal left pad
        self.n_weight = self.channels * self.channels * self.kernel_size
        self.n_delta = self.n_weight + 2 * self.channels

        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.input_mixer = nn.Conv1d(1, channels, 1, bias=False)
        self.act = make_activation(activation)
        self.layer1x1 = nn.Conv1d(channels, channels, 1)
        self.delta_parts = DELTA_PARTS_ALL

    def split_delta(self, delta: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """delta [B, n_delta] -> dW [B,C,C,K], db [B,C], dm [B,C].

        Same layout as DeltaWeightGen.forward, so a generator's output and a
        table row are interchangeable.
        """
        dw, db, dm = delta.split((self.n_weight, self.channels, self.channels), dim=-1)
        return dw.view(-1, self.channels, self.channels, self.kernel_size), db, dm

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x trunk [B,C,T], cond clean [B,1,T], delta [B, n_delta]."""
        C = self.channels
        use_kernel, use_bias, use_mixin = self.delta_parts
        dw, db, dm = self.split_delta(delta)
        if use_kernel:
            # zeros_like(db), not a bare 0: per_sample_conv1d reshapes the bias
            # by B, so it still needs a [B,C] tensor even when masked off.
            z = per_sample_conv1d(x, self.conv.weight.unsqueeze(0) + dw,
                                  self.conv.bias.unsqueeze(0)
                                  + (db if use_bias else torch.zeros_like(db)),
                                  self.pad, self.dilation)
        else:
            z = self.conv(F.pad(x, (self.pad, 0)))       # shared conv, no per-device kernel
            if use_bias:
                z = z + db.unsqueeze(-1)
        mix = self.input_mixer.weight.reshape(1, C, 1)
        if use_mixin:
            mix = mix + dm.unsqueeze(-1)
        z = self.act(z + mix.to(z.dtype) * cond.to(z.dtype))   # see DeltaWaveNetLayer for the cast
        return x + self.layer1x1(z), z                   # (residual, head term)


class TableDeltaWaveNet(FiLMWaveNet):
    """A2 WaveNet where every device owns a free, unconstrained weight residual.

    Full-rank limit of DeltaWaveNet: same hook/audio path/dW-db-dm layout, but
    the residual is a whole-row lookup from a [N_devices, D] table rather than
    scale * coeff(e) @ normalize(basis). Nothing is shared between devices but
    the base kernels, isolating the low-rank basis as the only difference from
    delta_wavenet.

    - embedding_dim is derived from the schedule (D=10,352 at C=8), not
      configured; there is no capacity knob.
    - Parameters scale with n_devices (4.2M at 405).
    - Cannot be enrolled: held-out devices have no row and there's no shared
      structure to place one in. openamp.emulate.enroll refuses this arch.
    - Still folds/exports: a device's row is its delta, a slice-and-add.

    Table is zero-initialized (untrained net = stock A2, every device starts
    identical); its gradient is nonzero at zero, so no vanishing-gradient trap.
    """

    def __init__(self, **kwargs):
        kwargs.pop("embedding_dim", None)                 # derived from the schedule
        super().__init__(embedding_dim=1, **kwargs)       # placeholder; resized below
        widths = [layer.n_delta for layer in self.layers]
        self.delta_slices = []
        lo = 0
        for w in widths:
            self.delta_slices.append((lo, lo + w))
            lo += w
        self.embedding_dim = lo
        self.embedding = nn.Embedding(self.n_devices, self.embedding_dim)
        nn.init.zeros_(self.embedding.weight)             # every device starts at W

    def _make_layer(self, kernel_size: int, dilation: int) -> nn.Module:
        return TableDeltaWaveNetLayer(self.channels, kernel_size, dilation,
                                      self.activation)

    def set_delta_parts(self, *, kernel: bool = True, bias: bool = True,
                        mixin: bool = True) -> "TableDeltaWaveNet":
        """Choose which slices of the per-device delta are applied at inference. Returns self."""
        parts = (bool(kernel), bool(bias), bool(mixin))
        for layer in self.layers:
            layer.delta_parts = parts
        return self

    @property
    def delta_parts(self) -> tuple[bool, bool, bool]:
        """The current mask (layers are always set together)."""
        return self.layers[0].delta_parts

    def layer_conditioning(self, emb: torch.Tensor):
        """Each layer gets its own slice of the device row, not the whole thing."""
        for layer, (lo, hi) in zip(self.layers, self.delta_slices):
            yield layer, emb[..., lo:hi]

    @torch.no_grad()
    def neutralize_conditioning(self) -> None:
        self.embedding.weight.zero_()   # deltas live in the table; zeroing it is the no-op

    @classmethod
    def from_config(cls, ecfg, n_devices: int) -> "TableDeltaWaveNet":
        """Construct from an EmulateConfig."""
        return cls(n_devices=n_devices, channels=ecfg.wn_channels,
                   activation=ecfg.wn_activation)

    # No forward() override: layer_conditioning already does this arch's per-layer slicing.
