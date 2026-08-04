"""FiLM-conditioned NAM A2 WaveNet for one-to-many amp emulation.

The exact architecture of the corpus's own captures: every A2 (Slimmable NAM)
export on TONE3000 shares one full-width WaveNet — 23 dilated conv layers
(channels 8, the kernel/dilation schedule below), LeakyReLU(0.01), a per-layer
input mixin of the raw signal, a residual 1x1 per layer, summed skip outputs
through a kernel-16 head conv, scaled by ``head_scale``. Verified identical
across all 774 A2 captures in the corpus; the schedule constants here are read
straight from those exports.

The per-layer nonlinearity is a knob (``wn_activation`` in config): ``leakyrelu``
is what the captures use, ``tanh`` is the other activation NAM's A2 schema and
NeuralAmpModelerCore both implement, so a tanh run still folds into a
plugin-playable A2 WaveNet (see :mod:`openamp.emulate.export`).

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

Three architectures live here. Two are FiLM and differ *only* in the map from
embedding to ``(gamma, beta)``: :class:`FiLMWaveNet` (``film_wavenet``) uses one
``Linear(E, 2C)`` per layer, :class:`MLPFiLMWaveNet` (``mlpfilm_wavenet``) an
independent ``Linear(E, H) -> act -> Linear(H, 2C)``. The conditioning itself is
per-channel affine in both.

The third, :class:`DeltaWaveNet` (``delta_wavenet``), moves the conditioning one
step upstream: instead of rescaling the layer's *output* per channel, the device
embedding generates a low-rank **residual on the layer's own weights**,
``z = conv_{W + dW(e), b + db(e)}(x) + (mixin_w + dm(e)) * clean``. FiLM is the
special case ``dW = (gamma - 1) * W`` row-wise, ``db = (gamma - 1) * b + beta``,
``dm = (gamma - 1) * mixin_w``, so this is a strict superset of what the FiLM hook
can express at the same position, at comparable parameter cost
(:class:`DeltaWeightGen`).

All three fold into a plugin-playable A2 capture — the generator is never in the
real-time DSP loop, and for ``delta_wavenet`` the fold *is* the addition the layer
would have done anyway (see :mod:`openamp.emulate.export`). All three also keep
one shared ``[N_devices x embedding_dim]`` table, so enrollment, morphing and
profile export are architecture-independent.

Import is torch-only, like :mod:`~openamp.emulate.tcn`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["FiLMWaveNet", "MLPFiLMWaveNet", "DeltaWaveNet", "MLPFiLM",
           "DeltaWeightGen", "film_output_linear", "init_film_identity",
           "init_delta_zero", "make_activation", "normalize_activation",
           "ACTIVATIONS", "NAM_ACTIVATION_NAMES", "A2_KERNEL_SIZES", "A2_DILATIONS",
           "A2_HEAD_KERNEL", "A2_ACTIVATION", "A2_COND_HIDDEN", "A2_COND_ACTIVATION",
           "A2_DELTA_RANK", "A2_DELTA_SCALE_INIT"]

# The one architecture shared by every A2 capture in the corpus (full-width
# submodel of the SlimmableContainer export): three 7-layer dilation runs at
# kernel 6 with two kernel-15 layers between runs 2 and 3.
A2_KERNEL_SIZES = (6,) * 14 + (15, 15) + (6,) * 7
A2_DILATIONS = (1, 3, 7, 17, 41, 101, 239) * 2 + (1, 13) + (1, 3, 7, 17, 41, 101, 239)
A2_HEAD_KERNEL = 16
A2_CHANNELS = 8
A2_HEAD_SCALE = 0.02
A2_EMBEDDING_STD = 0.1               # init std of the [N_devices x E] table
A2_FILM_INIT_STD = 0.01              # near-identity FiLM start; see FiLMWaveNetLayer

# --- Activation (``wn_activation``) ---------------------------------------------
# Both options are parameter-free, so a checkpoint's state_dict is identical
# either way: switching this on an existing run loads silently and plays the
# wrong network. Hence "wn_activation" sits in train.py's _STRUCTURAL_KEYS.
ACTIVATIONS = ("leakyrelu", "tanh")
A2_ACTIVATION = "leakyrelu"          # what every A2 capture uses
A2_LEAKY_SLOPE = 0.01
# NAM's own schema spelling for each option, for the plugin bundle's arch block.
NAM_ACTIVATION_NAMES = {"leakyrelu": "LeakyReLU", "tanh": "Tanh"}

# --- Nonlinear FiLM generator (``cond_hidden`` / ``cond_activation``) -------------
# The mlpfilm_wavenet arch replaces each layer's single Linear(E, 2C) FiLM generator
# with an independent Linear(E, H) -> act -> Linear(H, 2C). At embedding_dim 256 /
# channels 8 the linear generator is 4,112 params per layer: H=16 is parameter-
# matched (4,384), H=32 is ~2x (8,752), H=64 quadruples the plugin bundle (17,488).
# The generator never runs in the plugin's DSP loop -- it is folded away per morph
# point -- so H costs training parameters and bundle bytes, not real-time CPU.
A2_COND_HIDDEN = 32
# LeakyReLU is positively homogeneous, so it is equally curved at any input scale.
# Tanh is ~linear until the embedding and first-layer weights grow (measured 0.35%
# nonlinear at init, E=256/H=64), i.e. a slower departure from film_wavenet.
A2_COND_ACTIVATION = "leakyrelu"

# --- Weight-delta generator (``delta_rank``) --------------------------------------
# The delta_wavenet arch drops FiLM entirely and lets the embedding write a residual
# straight onto each layer's conv kernel, bias and mixin gain. A full per-device
# kernel would be C*C*K numbers per layer per device (9,984 per device over the A2
# schedule at C=8) -- a private weight blob per amp, which is exactly what the shared
# embedding space exists to avoid. So the residual is low-rank: ``rank`` kernel-shaped
# directions (``basis``, shared by all devices) that the embedding mixes (``coeff``).
# Per layer that is E*R + R + R*(C*C*K + 2C) + 1, i.e. 16,263*R + 23 over the A2
# schedule at E=256 / C=8: rank 6 is parameter-matched to film_wavenet's 23 Linears
# (97,601 vs 94,576) and the rank-8 default is 1.36x them (130,127).
A2_DELTA_RANK = 8
# Initial delta size, as a fraction of the base kernel's own weight std. The residual
# must *stay* a residual: at 0.03 the untrained net is a plain A2 WaveNet to within a
# fraction of a dB, and the shared kernel keeps doing the work while the device only
# nudges it.
#
# This is a hard-won constant. The first delta_a2_256 run had no ``scale`` at all --
# magnitude lived diffusely in ``basis``, where ``coeff`` and ``basis`` can trade it
# back and forth for free. The optimizer took that trade: by epoch 12 the delta was
# 5.1x the base weight (max |delta| 27.9 vs |W| std 0.38), i.e. it had shrunk the
# shared kernel toward irrelevance and rebuilt each device's filter out of its own
# residual -- the per-device weight table this arch exists to avoid, reached the long
# way round. Pinning ``basis`` to unit-norm directions and putting magnitude in this
# one scalar per layer removes the free trade: growing the delta now costs one
# visible, decayable parameter instead of drifting through 3,200 diffuse ones.
A2_DELTA_SCALE_INIT = 0.03


def normalize_activation(name: str, field: str = "wn_activation") -> str:
    """Validate an activation value and return its canonical key.

    ``field`` names the config knob being validated, so ``cond_activation`` errors
    do not blame ``wn_activation``.
    """
    key = str(name).strip().lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"{field} must be one of {ACTIVATIONS}, got {name!r}")
    return key


def make_activation(name: str, field: str = "wn_activation") -> nn.Module:
    """The activation module for an activation value (case-insensitive)."""
    key = normalize_activation(name, field)
    return nn.LeakyReLU(A2_LEAKY_SLOPE) if key == "leakyrelu" else nn.Tanh()


def film_output_linear(film: nn.Module) -> nn.Linear:
    """The FiLM generator's final ``nn.Linear`` -- the one whose rows are (gamma, beta).

    Both generators end in a Linear (``nn.Linear`` itself for film_wavenet, ``fc2``
    for :class:`MLPFiLM`), and zeroing *that* weight makes the whole generator emit
    its bias verbatim for any finite input. That single fact is what keeps both the
    near-identity training init and export's fold neutralization exact for either
    generator (see :func:`openamp.emulate.export.folded_model`).
    """
    linears = [m for m in film.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise TypeError(f"FiLM generator {type(film).__name__} has no nn.Linear")
    return linears[-1]


@torch.no_grad()
def init_film_identity(film: nn.Module, channels: int, std: float = 0.0) -> None:
    """Set a FiLM generator to (near-)identity: gamma ~ 1, beta ~ 0.

    ``std=0`` is *exact* identity -- the generator returns (1, 0) bit for bit, since
    ``x @ 0^T`` is a sum of exact zeros. That is what export's neutralized fold copy
    needs. ``std>0`` is the training init: identity plus a small random tilt, so the
    untrained net is a plain A2 WaveNet while conditioning gradients stay alive.
    """
    out = film_output_linear(film)
    if std > 0:
        out.weight.normal_(0.0, std)
    else:
        out.weight.zero_()
    out.bias[:channels] = 1.0
    out.bias[channels:] = 0.0


@torch.no_grad()
def init_delta_zero(gen: "DeltaWeightGen", scale: float = 0.0) -> None:
    """Set a weight-delta generator to (near-)zero: ``dW ~ 0``, ``db ~ 0``, ``dm ~ 0``.

    ``scale=0`` is *exact* zero -- the generator multiplies by a zero scalar, so the
    layer is a plain A2 conv again for any finite embedding. That is what export's
    neutralized fold copy needs, the delta counterpart of
    :func:`init_film_identity`. ``scale>0`` is the training init: random basis
    directions at that magnitude.

    Note what is *not* done: zeroing ``basis``. It would be the tidier start, but it
    also zeroes ``coeff``'s gradient (``dL/dcoeff = dL/ddelta @ basis^T``), leaving
    the embedding read untrained until the basis drifts off zero on its own. Random
    directions with a small ``scale`` give the same near-identity net with both
    generator paths live from step 1.
    """
    if scale > 0:
        gen.basis.normal_(0.0, 1.0)      # directions only; forward() normalizes them
    gen.scale.fill_(float(scale))


class DeltaWeightGen(nn.Module):
    """Embedding -> low-rank residual on one layer's conv kernel, bias and mixin.

    ``delta(e) = scale * (coeff(e) @ normalize(basis))``, split into ``dW`` [C, C, K],
    ``db`` [C] and ``dm`` [C]: ``rank`` kernel-shaped **unit-norm directions** shared
    by every device, mixed by a per-device coefficient vector read linearly off the
    embedding, sized by one learned scalar for the whole layer. Rank is the capacity
    knob -- ``rank >= C*C*K + 2C`` would be an unconstrained per-device layer, which is
    both huge and pointless (no shared structure left to generalize with).

    Splitting direction (``basis``) from magnitude (``scale``) is the whole point of
    the parameterization, not a detail: without it ``coeff`` and ``basis`` can trade
    scale for free and the delta grows without ever paying for it (see
    :data:`A2_DELTA_SCALE_INIT` for what that cost in practice). Normalizing pins each
    direction's length, so the layer's total deviation from the shared kernel is one
    number that can be watched, weight-decayed, or clamped.

    ``dm`` (the raw-signal mixin gain) is here only so the family strictly contains
    FiLM's: FiLM's gamma scales ``conv(x) + mixin(clean)`` as a sum, so without a
    mixin term the delta would be missing something the arch it is measured against
    can do. It is C numbers per rank direction.

    The reparameterization is training-time only: export ships ``scale * normalize(
    basis)`` pre-multiplied as one matrix, so the plugin's re-fold stays the plain
    ``coeff(e) @ basis`` matmul it always was.

    Kept deliberately linear in ``e``: enrollment optimizes an embedding by
    gradient, and morph points are mixes of table rows, so a linear read keeps the
    delta a well-behaved (affine) function of position in embedding space. The
    *network* is nonlinear in the delta regardless.

    Never runs in the plugin's DSP loop -- evaluated once per morph point and added
    into the conv weights, exactly like the FiLM generators it replaces.
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
        """Solve for ``scale`` so the initial delta is ``scale * weight_std`` wide.

        ``scale`` is meant to read as "the delta starts at 3% of this layer's kernel",
        but the raw ``coeff(e) @ normalize(basis)`` magnitude depends on rank, on
        ``n_out`` and on how ``nn.Linear`` happened to initialize ``coeff`` — so the
        constant is measured against a probe batch rather than derived. Deriving it
        by hand is what put the first attempt three orders of magnitude off.

        The probe draws from a private generator: calibration must not consume the
        global RNG stream, or adding a layer would shift every later layer's init.
        """
        if scale <= 0:
            self.scale.zero_()
            return
        g = torch.Generator().manual_seed(0)
        probe = torch.randn(256, self.coeff.in_features, generator=g) * A2_EMBEDDING_STD
        raw = (self.coeff(probe) @ self.effective_basis())[:, :self.n_weight]
        self.scale.mul_(scale * weight_std / max(float(raw.std()), 1e-12))

    def effective_basis(self) -> torch.Tensor:
        """``scale * normalize(basis)`` -- the matrix the delta is actually read from.

        The single definition of the reparameterization, shared by :meth:`forward` and
        the bundle export, so the two can never drift apart.
        """
        return self.scale * F.normalize(self.basis, dim=1)

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``emb`` [B,E] (or a bare [E] morph point) -> ``dW`` [B,C,C,K], ``db``,
        ``dm`` [B,C]."""
        flat = (self.coeff(emb) @ self.effective_basis()).reshape(-1, self.n_out)
        dw, db, dm = flat.split((self.n_weight, self.channels, self.channels), dim=-1)
        return dw.view(-1, self.channels, self.channels, self.kernel_size), db, dm


class MLPFiLM(nn.Module):
    """Nonlinear embedding -> (gamma, beta) generator: ``Linear -> act -> Linear``.

    One independent generator per WaveNet layer (no shared trunk), so every layer
    bends the embedding space its own way. Never runs in the plugin's DSP loop: it
    is evaluated once per morph point and folded into the conv weights, exactly like
    the single Linear it replaces.

    Named submodules rather than :class:`~torch.nn.Sequential` so the state_dict
    reads ``fc1.weight`` and stays stable if a layer is ever inserted.
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

    ``z = conv(x) + mixin(clean)`` is FiLM-modulated by the device embedding,
    then passed through ``activation`` (LeakyReLU(0.01) as in the captures, or
    Tanh). The activation feeds two paths: ``layer1x1`` back onto the residual
    trunk, and (head1x1 is inactive in A2) directly out as this layer's skip
    contribution to the head sum.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 embedding_dim: int, activation: str = A2_ACTIVATION,
                 cond_hidden: int = 0,
                 cond_activation: str = A2_COND_ACTIVATION):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation          # causal left pad
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.input_mixer = nn.Conv1d(1, channels, 1, bias=False)
        # cond_hidden 0 = the linear generator (film_wavenet); > 0 swaps in the
        # per-layer MLP (mlpfilm_wavenet). Only the emb -> (gamma, beta) map
        # changes: the conditioning itself stays per-channel affine either way, so
        # both fold into a stock A2 capture identically.
        self.film = (nn.Linear(embedding_dim, 2 * channels) if cond_hidden <= 0 else
                     MLPFiLM(embedding_dim, channels, cond_hidden, cond_activation))
        # Near-identity start (scale ~1, shift ~0): the untrained net is a plain
        # A2 WaveNet, but the small weight keeps conditioning gradients alive.
        init_film_identity(self.film, channels, std=A2_FILM_INIT_STD)
        self.act = make_activation(activation)
        self.layer1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` trunk [B,C,T], ``cond`` clean signal [B,1,T], ``emb`` [B,E]."""
        z = self.conv(F.pad(x, (self.pad, 0))) + self.input_mixer(cond)
        gamma, beta = self.film(emb).chunk(2, dim=-1)    # [B, C], [B, C]
        z = gamma.unsqueeze(-1) * z + beta.unsqueeze(-1)
        z = self.act(z)
        return x + self.layer1x1(z), z                   # (residual, head term)


class DeltaWaveNetLayer(nn.Module):
    """One A2 WaveNet layer, causal, conditioned by a per-device delta on its conv.

    Same signature and same audio path as :class:`FiLMWaveNetLayer` -- the device
    enters one step earlier. Instead of scaling a shared conv's output per channel,
    the embedding writes a residual onto the layer's own weights::

        z = conv_{W + dW(e), b + db(e)}(x) + (mixin_w + dm(e)) * clean

    then ``activation`` and the same two outputs (residual trunk, head term). This
    strictly contains the FiLM hook: ``gamma * (conv_{W,b}(x) + mixin(clean)) +
    beta`` is the delta ``dW = (gamma - 1) * W`` row-wise, ``db = (gamma - 1) * b +
    beta``, ``dm = (gamma - 1) * mixin_w`` -- one point in this family, which is why
    a null result against film_wavenet is informative rather than a capacity
    artifact. A delta can also *rotate* a kernel, retiming the filter rather than
    only re-gaining it, which is the reason to try it. ``layer1x1`` stays shared and
    unconditioned (FiLM does not reach it either).

    The delta is a shared low-rank basis mixed per device (:class:`DeltaWeightGen`),
    not a per-device kernel table, so the arch keeps the one ``[N, E]`` embedding
    space everything else in the pipeline is built on.
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
        # The generator sizes its init off this layer's own kernel, so the delta
        # starts at the same *fraction* of W in every layer of the schedule (the
        # kernel-15 layers have a smaller fan-in std than the kernel-6 ones).
        self.delta = DeltaWeightGen(embedding_dim, channels, kernel_size, rank,
                                    scale=A2_DELTA_SCALE_INIT,
                                    weight_std=float(self.conv.weight.detach().std()))
        self.act = make_activation(activation)
        self.layer1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` trunk [B,C,T], ``cond`` clean signal [B,1,T], ``emb`` [B,E]."""
        B, C, T = x.shape
        dw, db, dm = self.delta(emb)                     # [B,C,C,K], [B,C], [B,C]
        w = self.conv.weight.unsqueeze(0) + dw
        b = self.conv.bias.unsqueeze(0) + db
        # A batch mixes devices, so there is no single weight tensor to convolve
        # with: one grouped conv over B groups *is* B independent convs, each with
        # its own sample's weights, in one kernel launch.
        z = F.conv1d(F.pad(x, (self.pad, 0)).reshape(1, B * C, -1),
                     w.reshape(B * C, C, self.kernel_size), b.reshape(B * C),
                     dilation=self.dilation, groups=B).reshape(B, C, T)
        # input_mixer is a 1x1 conv with no bias, i.e. a per-channel gain on the raw
        # signal — written out as a broadcast multiply so the delta can ride on it.
        # The module still holds the shared weight (state_dict and bundle key). Cast
        # to the conv's dtype first: autocast only rewrites conv/matmul, so a bare
        # fp32 multiply here would promote this layer's activation back to fp32 and
        # cost ~25% peak memory under amp for nothing.
        mix = (self.input_mixer.weight.reshape(1, C, 1) + dm.unsqueeze(-1)).to(z.dtype)
        z = self.act(z + mix * cond.to(z.dtype))
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
    (``wn_channels`` in config) is the width sweep knob and ``activation``
    (``wn_activation``) picks the per-layer nonlinearity.
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
        self.register_buffer("head_scale", torch.tensor(float(head_scale)))

    def _make_layer(self, kernel_size: int, dilation: int) -> nn.Module:
        """The per-layer module — the one thing a WaveNet arch here overrides.

        Everything else (schedule, embedding table, rechannel/head, forward) is
        shared, so an arch is defined by how its layer takes the embedding.
        """
        return FiLMWaveNetLayer(self.channels, kernel_size, dilation,
                                self.embedding_dim, self.activation,
                                self.cond_hidden, self.cond_activation)

    @classmethod
    def from_config(cls, ecfg, n_devices: int) -> "FiLMWaveNet":
        """Construct from an :class:`~openamp.core.config.EmulateConfig`."""
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


class MLPFiLMWaveNet(FiLMWaveNet):
    """A2 FiLM-WaveNet whose per-layer FiLM generator is a small MLP.

    Identical to :class:`FiLMWaveNet` everywhere audio flows -- same A2 schedule,
    same per-channel affine ``z = gamma * z + beta`` at the same pre-activation
    hook, same fold-into-a-stock-A2-capture export path. The only change is the map
    from the device embedding to ``(gamma, beta)``: an independent
    ``Linear(E, H) -> act -> Linear(H, 2C)`` per layer instead of one
    ``Linear(E, 2C)``, so a device's position in embedding space can steer the
    network nonlinearly rather than only through a single linear read.

    A separate arch rather than a knob on ``film_wavenet``: the two have
    incompatible state_dicts, so keeping the names apart means every existing
    film_wavenet run, checkpoint and plugin bundle stays exactly what it is.
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
        """Construct from an :class:`~openamp.core.config.EmulateConfig`."""
        return cls(n_devices=n_devices, channels=ecfg.wn_channels,
                   embedding_dim=ecfg.embedding_dim,
                   activation=ecfg.wn_activation,
                   cond_hidden=ecfg.cond_hidden,
                   cond_activation=ecfg.cond_activation)


class DeltaWaveNet(FiLMWaveNet):
    """A2 WaveNet conditioned by a per-device residual on each layer's conv weights.

    Identical to :class:`FiLMWaveNet` in topology, receptive field, embedding table
    and forward — the layers are :class:`DeltaWaveNetLayer` instead of
    :class:`FiLMWaveNetLayer`, so the device steers ``W`` itself rather than the
    conv's per-channel output gain. ``delta_rank`` is the capacity knob; the
    ``cond_*`` knobs do not apply (there is no FiLM generator to shape).

    A separate arch rather than a knob: its state_dict has no ``film`` and the FiLM
    archs have no ``delta``, so every existing film_wavenet / mlpfilm_wavenet run,
    checkpoint and plugin bundle stays exactly what it is, and a cross-arch load
    fails loudly instead of reinterpreting weights.

    Still exports: for a fixed embedding the delta is a constant, so folding is the
    addition the layer would have performed, leaving a bit-for-bit stock A2 capture
    (see :func:`openamp.emulate.export.folded_model`).
    """

    def __init__(self, *, delta_rank: int = A2_DELTA_RANK, **kwargs):
        if int(delta_rank) < 1:
            raise ValueError(f"delta_rank must be >= 1, got {delta_rank}")
        # Set before nn.Module.__init__ runs — legal for a plain int (only Parameter
        # / Module assignment needs the module dicts) and necessary, because the
        # base constructor calls _make_layer, which reads it.
        self.delta_rank = int(delta_rank)
        super().__init__(**kwargs)

    def _make_layer(self, kernel_size: int, dilation: int) -> nn.Module:
        return DeltaWaveNetLayer(self.channels, kernel_size, dilation,
                                 self.embedding_dim, self.delta_rank, self.activation)

    @classmethod
    def from_config(cls, ecfg, n_devices: int) -> "DeltaWaveNet":
        """Construct from an :class:`~openamp.core.config.EmulateConfig`."""
        return cls(n_devices=n_devices, channels=ecfg.wn_channels,
                   embedding_dim=ecfg.embedding_dim,
                   activation=ecfg.wn_activation,
                   delta_rank=ecfg.delta_rank)
