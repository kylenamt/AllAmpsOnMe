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
                                     DELTA_PART_NAMES, DELTA_PARTS_ALL, DeltaWaveNet,
                                     FiLMWaveNet, MLPFiLMWaveNet, TableDeltaWaveNet,
                                     normalize_activation)

__all__ = ["build_model", "arch_summary"]

ARCHS = ("film_tcn", "film_wavenet", "mlpfilm_wavenet", "delta_wavenet",
         "tabledelta_wavenet")

# The A2-topology archs: same shape and same fold contract, differing only in how
# the device embedding reaches each layer.
WAVENET_ARCHS = ("film_wavenet", "mlpfilm_wavenet", "delta_wavenet",
                 "tabledelta_wavenet")

# Archs whose conditioning is a *shared* map from one embedding space, so an unseen
# device can be enrolled by finding its point in that space. tabledelta_wavenet is
# excluded on purpose: its "embedding" is a per-device weight blob with no shared
# structure to position a new amp within (see enroll.require_enrollable).
ENROLLABLE_ARCHS = ("film_tcn", "film_wavenet", "mlpfilm_wavenet", "delta_wavenet")


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
    if ecfg.arch == "tabledelta_wavenet":
        return TableDeltaWaveNet.from_config(ecfg, n_devices)
    raise ValueError(f"emulate.arch must be one of {ARCHS}, got {ecfg.arch!r}")


def _parts_tag(model) -> str:
    """``""`` for the full delta, else the enabled slices (``-kernel``, ``-bias+mixin``).

    Empty for the default so an ordinary tabledelta row reads ``a2-23L-table`` and
    stays comparable with every other arch's string.
    """
    parts = getattr(model, "delta_parts", DELTA_PARTS_ALL)
    if tuple(parts) == DELTA_PARTS_ALL:
        return ""
    on = [n for n, keep in zip(DELTA_PART_NAMES, parts) if keep]
    return "-" + ("+".join(on) if on else "none")


def arch_summary(ecfg, model=None) -> dict:
    """Arch descriptors for logs, ``metrics.json``, and the comparison CSV.

    - ``channels``/``blocks_x_layers`` mean the arch's own size knobs, so CSV
      columns stay comparable across architectures.
    - A non-capture activation tags onto the WaveNet's topology string
      (``a2-23L-tanh``) rather than getting its own column, so activation runs
      stay distinguishable in the existing comparison table. The MLP FiLM
      generator (``a2-23L-mlp32``, ``a2-23L-mlp16-ctanh``), weight-delta
      generator (``a2-23L-delta8``) and free per-device table (``a2-23L-table``)
      tag the same way, leaving the two film_wavenet strings untouched so old
      rows stay comparable.
    - ``model`` is optional and only read for state that isn't in the config:
      tabledelta_wavenet's inference-time part mask, which changes what a row
      measures (``a2-23L-table-kernel``) and would otherwise be invisible in the
      CSV — two rows of the same run, evaluated differently, must not look alike.
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
        elif ecfg.arch == "tabledelta_wavenet":
            suffix = f"-table{_parts_tag(model)}{suffix}"
        return {"arch": ecfg.arch, "channels": ecfg.wn_channels,
                "blocks_x_layers": f"a2-{len(A2_DILATIONS)}L{suffix}"}
    return {"arch": ecfg.arch, "channels": ecfg.channels,
            "blocks_x_layers": f"{ecfg.blocks}x{ecfg.layers_per_block}"}
