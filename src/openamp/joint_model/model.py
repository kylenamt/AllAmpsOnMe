"""The two halves wired together: encoder -> embedding -> generator.

Deliberately thin. The generator is
:class:`~openamp.models.FiLMWaveNet` exactly as it trains anywhere else
— this only drives it through ``forward_emb`` (a vector) instead of ``forward``
(a table row), which is the one seam the joint model needs.

Encoder and generator stay separate attributes rather than being fused, because
every diagnostic in :mod:`~openamp.evaluation.joint` works by cutting the
wire between them: render window A with a *zeroed* embedding, a *shuffled* one, or
one computed from another amp's reference. :meth:`JointModel.embed` and
:meth:`JointModel.render` are those two halves exposed separately;
:meth:`forward` is just their composition.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from openamp.encoders import build_encoder
from openamp.models import build_model
from openamp.spec import RunSpec, spec_from_checkpoint

__all__ = ["JointModel", "build_joint", "load_joint"]


class JointModel(nn.Module):
    """``(dry window A, (dry, wet) reference B) -> emulated wet A``."""

    def __init__(self, encoder: nn.Module, generator: nn.Module):
        super().__init__()
        if encoder.embedding_dim != generator.embedding_dim:
            raise ValueError(
                f"embedding width mismatch: encoder emits {encoder.embedding_dim}, "
                f"generator expects {generator.embedding_dim}. Both read "
                f"model.embedding_dim, so this means one was built from a stale config.")
        self.encoder = encoder
        self.generator = generator

    @property
    def receptive_field(self) -> int:
        """The *generator's* receptive field — what the training window must warm up.

        The encoder has its own, but it is non-causal and pools over the whole
        reference, so it imposes no left-context requirement on window A.
        """
        return self.generator.receptive_field

    @property
    def embedding_dim(self) -> int:
        return self.generator.embedding_dim

    def embed(self, ref: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, 2, L_ref] -> [B, embedding_dim]``."""
        return self.encoder(ref, mask)

    def render(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """``([B, T] or [B, 1, T], [B, embedding_dim]) -> [B, 1, T]``."""
        return self.generator.forward_emb(x, emb)

    def forward(self, x: torch.Tensor, ref: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.render(x, self.embed(ref, mask))


def build_joint(spec: RunSpec, n_devices: int) -> JointModel:
    """Build both halves from a :class:`~openamp.spec.RunSpec`.

    ``n_devices`` only sizes the generator's (unused, frozen) embedding table; the
    conditioning comes from the encoder. The table is kept so the class, its
    checkpoints and every downstream tool stay exactly what they are — but it is
    frozen here, so a joint checkpoint can never quietly contain a half-trained
    lookup table competing with the encoder.
    """
    generator = build_model(spec.model, n_devices)
    generator.embedding.weight.requires_grad_(False)
    if getattr(generator, "embed_normalize", False):
        # The encoder produces the effective embedding itself (encoder.normalize),
        # so the table's radius must not train here either.
        generator.embed_scale.requires_grad_(False)
    encoder = build_encoder(spec.joint.encoder, generator.embedding_dim,
                            grad_checkpoint=spec.joint.grad_checkpoint_encoder)
    return JointModel(encoder, generator)


def load_joint(run_dir: Path, device: str = "cpu", *, which: str = "checkpoint"):
    """Reconstruct ``(model, ckpt)`` from a run's checkpoint (self-contained)."""
    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    ck = torch.load(Path(run_dir) / f"{which}.pt", map_location=device,
                    weights_only=False)
    model = build_joint(spec_from_checkpoint(ck), len(ck["device_ids"]))
    model.encoder.load_state_dict(ck["encoder"])
    model.generator.load_state_dict(ck["generator"])
    model.to(device).eval()
    return model, ck
