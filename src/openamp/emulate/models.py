"""Architecture selection for the emulation runs.

One dispatch point: everything downstream (dataset, losses, training loop,
checkpointing, comparison CSV, demos) is architecture-agnostic as long as the
model satisfies the shared contract — ``forward(x, device_idx) -> [B, 1, T]``
same-length causal, the same forward on a given vector as
``forward_emb(x, emb)``, an integer ``receptive_field`` property, an
``embedding`` table, and construction from ``(EmulateConfig, n_devices)``.
Adding an architecture = one class honoring that contract + one branch here.

Every arch shares the NAM A2 topology and one shared embedding space (so any of
them can be enrolled); they differ only in how the embedding reaches a layer.
"""

from __future__ import annotations

from openamp.emulate.wavenet import (A2_ACTIVATION, A2_COND_ACTIVATION, A2_DILATIONS,
                                     DeltaWaveNet, FiLMWaveNet, MLPFiLMWaveNet,
                                     normalize_activation)

__all__ = ["build_model", "arch_summary"]

ARCHS = ("film_wavenet", "mlpfilm_wavenet", "delta_wavenet")


def build_model(ecfg, n_devices: int):
    """Construct the configured architecture from an ``EmulateConfig``."""
    if ecfg.arch == "film_wavenet":
        return FiLMWaveNet.from_config(ecfg, n_devices)
    if ecfg.arch == "mlpfilm_wavenet":
        return MLPFiLMWaveNet.from_config(ecfg, n_devices)
    if ecfg.arch == "delta_wavenet":
        return DeltaWaveNet.from_config(ecfg, n_devices)
    raise ValueError(f"emulate.arch must be one of {ARCHS}, got {ecfg.arch!r}")


def arch_summary(ecfg, model=None) -> dict:
    """Arch descriptors for logs, ``metrics.json``, and the comparison CSV.

    - ``channels``/``blocks_x_layers`` mean the arch's own size knobs, so CSV
      columns stay comparable across architectures.
    - A non-capture activation tags onto the WaveNet's topology string
      (``a2-23L-tanh``) rather than getting its own column, so activation runs
      stay distinguishable in the existing comparison table. The MLP FiLM
      generator (``a2-23L-mlp32``, ``a2-23L-mlp16-ctanh``) and weight-delta
      generator (``a2-23L-delta8``) tag the same way, leaving the two
      film_wavenet strings untouched so old rows stay comparable.
    - ``model`` is accepted and unused; it stays in the signature because the
      callers all have one and a future arch may carry state the config doesn't.
    """
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
