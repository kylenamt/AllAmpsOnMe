"""Deep-unfolded enrollment: K learned gradient steps from an encoder's guess.

Enrollment fits an unseen amp's vector by thousands of gradient steps through
the frozen generator; an encoder guesses it in one forward pass and lands far
off. The unfolder sits between the two: start from the encoder's guess and take
K steps, each *shaped* by weights trained across many amps.

**The closed form it learns.** A ``delta_wavenet`` reads the amp vector ``e``
only through one linear map per layer, ``c_l = W_l e + b_l``
(``layer.delta.coeff``, ``W_l`` is ``[rank, D]``). So the gradient of any loss
with respect to ``e`` is exactly ``sum_l W_l^T g_l``, with ``g_l = dL/dc_l``.
Plain enrollment steps along that. The unfolded step replaces each ``W_l^T``
with a trainable copy per step, plus a trainable per-dimension step size::

    g_hat = g / ||g||                                    per amp, over all layers
    e    <- P(e - exp(log_s_k) * sum_l V_{k,l}^T g_hat_l)        k = 1..K

- ``P`` is the generator's own ``project_embedding``: under ``embed_normalize``
  it puts the vector back on the sphere every trained row lives on; otherwise
  it is the identity.
- ``V_{k,l}`` starts as an exact copy of ``W_l``, and ``log_s`` is calibrated so
  every step starts at one length -- so an untrained unfolder *is* normalized
  gradient descent, and whatever training adds is the learned part's doing.
- ``g`` is unit-normalized per amp because raw gradient size spans orders of
  magnitude between amps (an amp the encoder places badly has ESR 5-40); an
  unnormalized step that suits one amp throws another off the map.

**First-order.** Each ``g`` is computed on a detached copy of ``e`` and treated
as a constant: the outer loss never differentiates through a gradient, so memory
stays at one generator pass however large K is. The step sizes and ``V`` still
get an exact gradient through the explicit update path.

Trained by :mod:`openamp.training.unfold`; :func:`load_unfold` reads a run back.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from openamp.losses import pre_emphasis

__all__ = ["Unfolder", "coeff_layers", "coeff_grads", "per_amp_esr",
           "zero_shot_vectors", "load_unfold"]


def coeff_layers(gen) -> list[nn.Linear]:
    """The per-layer ``e -> coeff`` maps the closed form runs through.

    Only ``delta_wavenet`` has them. Raising here rather than deeper in a forward
    hook is the difference between naming the arch and a cryptic AttributeError.
    """
    layers = getattr(gen, "layers", None)
    if layers is None or not all(hasattr(l, "delta") for l in layers):
        raise ValueError(
            f"unfolding needs a delta_wavenet generator (each layer's delta.coeff is "
            f"the linear map the closed form inverts), got "
            f"{getattr(gen, 'arch', type(gen).__name__)!r}")
    return [l.delta.coeff for l in layers]


def per_amp_esr(y: torch.Tensor, t: torch.Tensor, coeff: float) -> torch.Tensor:
    """``[A, n, T]`` pair -> ``[A]`` pre-emphasized ESR, pooled over each amp's clips.

    One ratio per amp, not one over the batch: a pooled ratio would scale every
    amp's gradient by the batch's total energy, coupling amps that the unfolder
    steps independently.
    """
    yp, tp = pre_emphasis(y, coeff), pre_emphasis(t, coeff)
    return ((yp - tp) ** 2).sum((1, 2)) / ((tp ** 2).sum((1, 2)) + 1e-8)


def _autocast(dev: torch.device, amp_dtype: torch.dtype | None):
    return (torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype is not None
            else nullcontext())


def coeff_grads(gen, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor, coeff: float,
                *, amp_dtype: torch.dtype | None = None):
    """``(g [A, L, r], esr [A])``: each amp's gradient at its coeff outputs.

    ``x`` ``[A, n, R + T]`` clean with left-context, ``t`` ``[A, n, T]`` its
    render, ``e`` ``[A, D]`` the vectors the generator sees. ``e`` is detached --
    this is the first-order step. Each layer's coeff output is captured by a
    forward hook and differentiated directly, so ``g`` is exactly the ``g_l`` of
    the closed form, summed over the amp's clips.
    """
    maps = coeff_layers(gen)
    A, n = x.shape[:2]
    R = gen.receptive_field
    outs: list[torch.Tensor] = []
    hooks = [m.register_forward_hook(lambda _m, _i, o: outs.append(o)) for m in maps]
    try:
        e_in = e.detach().repeat_interleave(n, 0).requires_grad_(True)
        with torch.enable_grad(), _autocast(x.device, amp_dtype):
            y = gen.forward_emb(x.reshape(A * n, -1), e_in)[..., R:]
            esr = per_amp_esr(y.float().view(A, n, -1), t, coeff)
            gs = torch.autograd.grad(esr.sum(), outs)
    finally:
        for h in hooks:
            h.remove()
    g = torch.stack([gi.float().view(A, n, -1).sum(1) for gi in gs], 1)
    return g, esr.detach()


class Unfolder(nn.Module):
    """``V [K, L, r, D]`` and ``log_s [K, D]``: the K learned steps.

    ``W`` (the generator's own coeff weights) is kept as a buffer: it is ``V``'s
    starting point, and the distance from it is the readout of how far training
    moved the steps away from plain gradient descent.
    """

    def __init__(self, gen, steps: int):
        super().__init__()
        if int(steps) < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        W = torch.stack([m.weight.detach().float() for m in coeff_layers(gen)])
        self.register_buffer("W", W.clone())
        self.V = nn.Parameter(W[None].repeat(int(steps), 1, 1, 1).clone())
        self.log_s = nn.Parameter(torch.zeros(int(steps), W.shape[-1]))

    @property
    def steps(self) -> int:
        return int(self.V.shape[0])

    def direction(self, k: int, g_hat: torch.Tensor) -> torch.Tensor:
        """``[A, L, r]`` normalized coeff grads -> ``[A, D]`` step ``k``'s direction."""
        return torch.einsum("alr,lrd->ad", g_hat, self.V[k])

    @staticmethod
    def normalize(g: torch.Tensor) -> torch.Tensor:
        return g / g.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None]

    @torch.no_grad()
    def calibrate(self, gen, e0: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                  coeff: float, init_step_frac: float, *,
                  amp_dtype: torch.dtype | None = None) -> float:
        """Set every step's size so its first move is ``init_step_frac * |e0|`` long.

        Measured on a real batch rather than derived: the direction's length
        depends on the generator's trained ``W`` and the gradient's spread over
        layers. Returns the step length it calibrated to.
        """
        g, _ = coeff_grads(gen, x, t, e0, coeff, amp_dtype=amp_dtype)
        raw = float(self.direction(0, self.normalize(g)).norm(dim=1).mean())
        target = float(init_step_frac) * float(e0.norm(dim=1).mean())
        self.log_s.fill_(math.log(target / raw))
        return target

    def forward(self, gen, e0: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                coeff: float, *, amp_dtype: torch.dtype | None = None):
        """K steps from ``e0`` ``[A, D]`` (already projected) on clips ``x``/``t``.

        Returns ``(e_K, esr at e0 [A], mean step length)``. The graph from ``V``
        and ``log_s`` to ``e_K`` is kept only when grad is enabled.
        """
        e, esr0, lens = e0, None, []
        for k in range(self.steps):
            g, esr = coeff_grads(gen, x, t, e, coeff, amp_dtype=amp_dtype)
            esr0 = esr if esr0 is None else esr0
            step = self.log_s[k].exp() * self.direction(k, self.normalize(g))
            e = gen.project_embedding(e - step)
            lens.append(float(step.detach().norm(dim=1).mean()))
        return e, esr0, float(np.mean(lens))

    def drift(self) -> float:
        """``|V - W| / |W|`` per step: 0 for plain gradient descent."""
        with torch.no_grad():
            return float((self.V - self.W[None]).norm() / self.W.norm()
                         / self.steps ** 0.5)


# --- The frozen halves ---------------------------------------------------------------
def zero_shot_vectors(encoder_run: Path, ids: list[int], dev, *,
                      frames_path: str = "") -> dict[int, np.ndarray]:
    """``{device_id: raw vector}`` from a joint run's encoder, in eval mode.

    Raw, as the encoder emits it; the caller projects. ``frames_path`` overrides
    the store the checkpoint recorded -- the dac_frames runs name a host-local
    copy (``/var/tmp/...``) that does not exist on another machine.
    """
    from openamp.encoders import build_encoder
    from openamp.spec import EncoderSpec, spec_from_checkpoint

    ck = torch.load(Path(encoder_run) / "checkpoint.pt", map_location="cpu",
                    weights_only=False)
    es = spec_from_checkpoint(ck).joint.encoder
    if frames_path:
        es = EncoderSpec(es.kind, {**es.kwargs, "frames_path": str(frames_path)})
    enc = build_encoder(es, int(ck["generator"]["embedding.weight"].shape[1]))
    enc.load_state_dict(ck["encoder"])
    enc.to(dev).eval()
    out: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(ids), 8):
            chunk = [int(d) for d in ids[i:i + 8]]
            v = enc(torch.tensor(chunk, device=dev)).float().cpu().numpy()
            out.update({d: v[j] for j, d in enumerate(chunk)})
    return out


def load_unfold(run_dir: Path, device: str = "cpu", *, which: str = "checkpoint"):
    """``(unfolder, generator, ckpt)`` from an unfold run, both halves frozen."""
    from openamp.evaluation import load_model
    from openamp.spec import spec_from_checkpoint

    ck = torch.load(Path(run_dir) / f"{which}.pt", map_location=device,
                    weights_only=False)
    spec = spec_from_checkpoint(ck)
    gen, _ = load_model(Path(spec.unfold.generator_run), device)
    gen.requires_grad_(False)
    unf = Unfolder(gen, spec.unfold.steps).to(device)
    unf.load_state_dict(ck["unfolder"])
    return unf, gen, ck
