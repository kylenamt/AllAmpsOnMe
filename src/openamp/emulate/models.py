"""Architecture selection for the emulation runs.

One dispatch point: everything downstream (dataset, losses, training loop,
checkpointing, comparison CSV, demos) is architecture-agnostic as long as the
model satisfies the shared contract — ``forward(x, device_idx) -> [B, 1, T]``
same-length causal, an integer ``receptive_field`` property, an ``embedding``
table, and construction from ``(EmulateConfig, n_devices)``. Adding an
architecture = one class honoring that contract + one branch here.
"""

from __future__ import annotations

from openamp.emulate.tcn import FiLMTCN
from openamp.emulate.wavenet import A2_DILATIONS, FiLMWaveNet

__all__ = ["build_model", "arch_summary"]

ARCHS = ("film_tcn", "film_wavenet")


def build_model(ecfg, n_devices: int):
    """Construct the configured architecture from an ``EmulateConfig``."""
    if ecfg.arch == "film_tcn":
        return FiLMTCN.from_config(ecfg, n_devices)
    if ecfg.arch == "film_wavenet":
        return FiLMWaveNet.from_config(ecfg, n_devices)
    raise ValueError(f"emulate.arch must be one of {ARCHS}, got {ecfg.arch!r}")


def arch_summary(ecfg) -> dict:
    """Arch descriptors for logs, ``metrics.json``, and the comparison CSV.

    ``channels``/``blocks_x_layers`` mean the arch's own size knobs, so the CSV
    columns stay comparable across architectures.
    """
    if ecfg.arch == "film_wavenet":
        return {"arch": ecfg.arch, "channels": ecfg.wn_channels,
                "blocks_x_layers": f"a2-{len(A2_DILATIONS)}L"}
    return {"arch": ecfg.arch, "channels": ecfg.channels,
            "blocks_x_layers": f"{ecfg.blocks}x{ecfg.layers_per_block}"}
