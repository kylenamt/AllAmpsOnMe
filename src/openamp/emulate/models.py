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
from openamp.emulate.wavenet import (A2_ACTIVATION, A2_COND_ACTIVATION, A2_DILATIONS,
                                     DeltaWaveNet, FiLMWaveNet, MLPFiLMWaveNet,
                                     normalize_activation)

__all__ = ["build_model", "arch_summary"]

ARCHS = ("film_tcn", "film_wavenet", "mlpfilm_wavenet", "delta_wavenet")

# The A2-topology archs: same shape and same fold contract, differing only in how
# the device embedding reaches each layer.
WAVENET_ARCHS = ("film_wavenet", "mlpfilm_wavenet", "delta_wavenet")


def build_model(ecfg, n_devices: int):
    """Construct the configured architecture from an ``EmulateConfig``."""
    if ecfg.arch == "film_tcn":
        return FiLMTCN.from_config(ecfg, n_devices)
    if ecfg.arch == "film_wavenet":
        return FiLMWaveNet.from_config(ecfg, n_devices)
    if ecfg.arch == "mlpfilm_wavenet":
        return MLPFiLMWaveNet.from_config(ecfg, n_devices)
    if ecfg.arch == "delta_wavenet":
        return DeltaWaveNet.from_config(ecfg, n_devices)
    raise ValueError(f"emulate.arch must be one of {ARCHS}, got {ecfg.arch!r}")


def arch_summary(ecfg) -> dict:
    """Arch descriptors for logs, ``metrics.json``, and the comparison CSV.

    ``channels``/``blocks_x_layers`` mean the arch's own size knobs, so the CSV
    columns stay comparable across architectures. A non-capture activation is
    tagged onto the WaveNet's topology string (``a2-23L-tanh``) rather than given
    a column of its own, so activation runs are distinguishable in the existing
    comparison table. The MLP FiLM generator is tagged the same way
    (``a2-23L-mlp32``, ``a2-23L-mlp16-ctanh``), as is the weight-delta generator
    (``a2-23L-delta8``), leaving the two film_wavenet strings untouched so old rows
    stay comparable.
    """
    if ecfg.arch in WAVENET_ARCHS:
        act = normalize_activation(ecfg.wn_activation)
        suffix = "" if act == A2_ACTIVATION else f"-{act}"
        if ecfg.arch == "mlpfilm_wavenet":
            cact = normalize_activation(ecfg.cond_activation, field="cond_activation")
            csuffix = "" if cact == A2_COND_ACTIVATION else f"-c{cact}"
            suffix = f"-mlp{int(ecfg.cond_hidden)}{csuffix}{suffix}"
        elif ecfg.arch == "delta_wavenet":
            suffix = f"-delta{int(ecfg.delta_rank)}{suffix}"
        return {"arch": ecfg.arch, "channels": ecfg.wn_channels,
                "blocks_x_layers": f"a2-{len(A2_DILATIONS)}L{suffix}"}
    return {"arch": ecfg.arch, "channels": ecfg.channels,
            "blocks_x_layers": f"{ecfg.blocks}x{ecfg.layers_per_block}"}
