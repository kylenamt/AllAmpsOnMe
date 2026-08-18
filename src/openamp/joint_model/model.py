"""The two halves wired together: encoder -> embedding -> generator.

Deliberately thin. The generator is
:class:`~openamp.emulate.wavenet.FiLMWaveNet` exactly as it trains anywhere else
— this only drives it through ``forward_emb`` (a vector) instead of ``forward``
(a table row), which is the one seam the joint model needs.

Encoder and generator stay separate attributes rather than being fused, because
every diagnostic in :mod:`~openamp.joint_model.evaluate` works by cutting the
wire between them: render window A with a *zeroed* embedding, a *shuffled* one, or
one computed from another amp's reference. :meth:`JointModel.embed` and
:meth:`JointModel.render` are those two halves exposed separately;
:meth:`forward` is just their composition.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from openamp.core.config import EmulateConfig, JointConfig
from openamp.emulate.models import ENROLLABLE_ARCHS, build_model
from openamp.joint_model.fingerprint_encoder import FingerprintEncoder
from openamp.joint_model.wavenet_encoder import WaveNetEncoder

__all__ = ["JointModel", "build_joint", "load_joint"]


class JointModel(nn.Module):
    """``(dry window A, (dry, wet) reference B) -> emulated wet A``."""

    def __init__(self, encoder: WaveNetEncoder, generator: nn.Module):
        super().__init__()
        if encoder.embedding_dim != generator.embedding_dim:
            raise ValueError(
                f"embedding width mismatch: encoder emits {encoder.embedding_dim}, "
                f"generator expects {generator.embedding_dim}. Both read "
                f"emulate.embedding_dim, so this means one was built from a stale config.")
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


def build_joint(ecfg: EmulateConfig, jcfg: JointConfig, n_devices: int) -> JointModel:
    """Build both halves from config.

    ``n_devices`` only sizes the generator's (unused, frozen) embedding table; the
    conditioning comes from the encoder. The table is kept so the class, its
    checkpoints and every downstream tool stay exactly what they are — but it is
    frozen here, so a joint checkpoint can never quietly contain a half-trained
    lookup table competing with the encoder.
    """
    if ecfg.arch not in ENROLLABLE_ARCHS:
        raise ValueError(
            f"emulate.arch {ecfg.arch!r} cannot be driven by an encoder — it has no "
            f"shared embedding space an unseen amp could be positioned within. "
            f"Choose one of {ENROLLABLE_ARCHS}.")
    generator = build_model(ecfg, n_devices)
    if not hasattr(generator, "forward_emb"):
        raise ValueError(f"emulate.arch {ecfg.arch!r} has no forward_emb path")
    generator.embedding.weight.requires_grad_(False)
    if jcfg.enc_kind == "fingerprint":
        encoder = FingerprintEncoder.from_config(jcfg, generator.embedding_dim)
    else:
        encoder = WaveNetEncoder.from_config(jcfg, generator.embedding_dim)
    return JointModel(encoder, generator)


def load_joint(run_dir: Path, device: str = "cpu", *, which: str = "checkpoint"):
    """Reconstruct ``(model, ckpt)`` from a run's checkpoint (self-contained)."""
    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    ck = torch.load(Path(run_dir) / f"{which}.pt", map_location=device,
                    weights_only=False)
    model = build_joint(EmulateConfig(**ck["emulate_cfg"]),
                        JointConfig(**ck["joint_cfg"]), len(ck["device_ids"]))
    model.encoder.load_state_dict(ck["encoder"])
    model.generator.load_state_dict(ck["generator"])
    model.to(device).eval()
    return model, ck
