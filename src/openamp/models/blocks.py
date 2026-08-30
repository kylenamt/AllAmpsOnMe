"""Pieces the architectures are assembled from — no topology knowledge here.

Activations, the per-sample convolution the weight-delta path needs, the
normalized-embedding mixin, and the two conditioning generators (an MLP emitting
FiLM gamma/beta, a low-rank residual on a conv's weights). The A2 topology that
wires them together is :mod:`openamp.models.a2_wavenet`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MLPFiLM", "DeltaWeightGen", "NormalizedEmbedding", "film_output_linear",
           "init_film_identity", "init_delta_zero", "per_sample_conv1d",
           "make_activation", "normalize_activation", "count_parameters",
           "ACTIVATIONS", "NAM_ACTIVATION_NAMES", "A2_ACTIVATION", "A2_EMBEDDING_STD",
           "A2_COND_HIDDEN", "A2_COND_ACTIVATION", "A2_DELTA_RANK",
           "A2_DELTA_SCALE_INIT"]

A2_EMBEDDING_STD = 0.1               # init std of the [N_devices x E] table

# --- Activation (wn_activation) --------------------------------------------------
# Parameter-free, so a checkpoint's state_dict is identical either way -- in the
# archs' structural_keys() to stop a silent architecture swap on resume.
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


# --- General-purpose helpers ----------------------------------------------------

def count_parameters(module: nn.Module) -> int:
    """Trainable parameter count of ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


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


# --- Embedding ------------------------------------------------------------------

class NormalizedEmbedding:
    """Mixin: read the table as ``e = scale * normalize(row)`` instead of raw rows.

    - Direction lives on the unit sphere; magnitude is ONE global learned scalar
      shared by every device. Measured on nam_a2_256_mixed's trained table, the
      per-device radius carries nothing worth keeping -- corr(||e||, loudness)
      = -0.02, corr(||e||, test ESR) = +0.10, flat across gain buckets, and the
      whole table already sits in a shell (radius 4.04-4.93 at p25-p75). So this
      makes explicit what training converged to anyway, at one parameter rather
      than n_devices.
    - A global scale, not a fixed radius of 1: FiLM is affine with gamma's bias
      at 1, so e=0 is the *neutral* amp and ||e|| is distance from neutral.
      Pinning it at 1 would shrink every embedding to ~22% of the radius the
      table actually learns and systematically under-drive the conditioning
      (openamp.encoder.probe's objection to L2-normalizing against a frozen
      generator). The scale lets the sphere find its own radius instead.
    - NOT LayerNorm: nn.LayerNorm's default elementwise_affine breaks the
      constant norm outright, and even without it LN removes the per-vector mean
      -- forcing every embedding orthogonal to the all-ones direction, which is
      an arbitrary constraint on an embedding table (it rotates real rows by up
      to cos 0.969 here). F.normalize is the operation that means "unit sphere".
    - Gradients flow through the normalize, so updates are projected tangent to
      the sphere and the radius moves only via ``embed_scale`` -- the same
      direction/magnitude split DeltaWeightGen uses for its basis.

    ``forward_emb`` is deliberately NOT covered: it takes an already-effective
    embedding, so an encoder or morph point stays in charge of its own vector.
    """

    def _init_embed_normalize(self, enabled: bool, dim: int, std: float) -> None:
        """Register ``embed_scale`` when enabled. Init matches the raw table's own
        init magnitude (std*sqrt(dim)), so turning this on does not move the
        starting point."""
        self.embed_normalize = bool(enabled)
        if self.embed_normalize:
            self.embed_scale = nn.Parameter(torch.tensor(float(std) * dim ** 0.5))

    def project_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        """Raw row(s) -> effective embedding. Identity when the knob is off."""
        if not getattr(self, "embed_normalize", False):
            return emb
        return self.embed_scale * F.normalize(emb, dim=-1)

    def embed(self, device_idx: torch.Tensor) -> torch.Tensor:
        """Effective embedding rows for a ``[B]`` tensor of table indices."""
        return self.project_embedding(self.embedding(device_idx))

    def embedding_table(self) -> torch.Tensor:
        """The full effective ``[n_devices, dim]`` table (what export/enroll read)."""
        return self.project_embedding(self.embedding.weight)


# --- FiLM generator -------------------------------------------------------------
# Conditioning scales a layer's output per channel (gamma/beta); the generator
# maps embedding -> (gamma, beta) with one Linear or, here, an MLP.

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


# --- Weight-delta generator ------------------------------------------------------
# Conditioning writes a low-rank residual onto a layer's conv weights instead of
# scaling the layer's output.

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
