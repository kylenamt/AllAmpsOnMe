"""Conditioned amp architectures, and the one place an arch name is resolved.

**The model contract.** Everything downstream (dataset, losses, training loop,
checkpointing, comparison CSV, export) is architecture-agnostic as long as the
model provides:

=========================  =====================================================
``receptive_field: int``   left context the input must carry before the loss region
``embedding_dim: int``     width of the conditioning vector
``arch: str``              its own registry name
``forward(x, device_idx)`` ``([B,T] or [B,1,T], [B] long) -> [B,1,T]``, causal
``forward_emb(x, emb)``    the same forward on a ``[B,E]`` vector, not a table row
``embed(device_idx)``      ``[B] long -> [B,E]``, the effective embedding
``embedding_table()``      the full effective ``[n_devices, E]`` table
``describe()``             ``{arch, channels, blocks_x_layers}`` for logs and the CSV
``structural_keys()``      config knobs a ``--resume`` may not change
``export_descriptor()``    the plugin bundle's ``arch`` block
``from_config(ecfg, n)``   construction from an ``EmulateConfig``
=========================  =====================================================

The last four are why this contract is worth having: each was previously a
branch in a *different* file (``arch_summary``'s string ladder, two
``_STRUCTURAL_*`` lists, export's isinstance chain), so adding an architecture
meant editing six files. Now it is one file plus a line in :data:`MODELS`.

Documented, not enforced -- this is a research codebase, and a model that
breaks the contract fails loudly at the call site anyway.

**Adding an architecture** = one new module with a class honouring that
contract, plus either an entry in :data:`MODELS` or a ``@register_model("name")``
decorator on the class. Then ``arch: <name>`` in the config; no training-loop
change. The built-ins are listed explicitly rather than self-registering, so a
name is never missing because its module happened not to be imported.
"""

from __future__ import annotations

from openamp.models.a2_wavenet import (A2_CHANNELS, A2_DILATIONS, A2_HEAD_KERNEL,
                                       A2_HEAD_SCALE, A2_KERNEL_SIZES, DeltaWaveNet,
                                       DeltaWaveNetLayer, FiLMWaveNet,
                                       FiLMWaveNetLayer, MLPFiLMWaveNet)
from openamp.models.blocks import (A2_ACTIVATION, A2_COND_ACTIVATION, A2_COND_HIDDEN,
                                   A2_DELTA_RANK, A2_EMBEDDING_STD, ACTIVATIONS,
                                   NAM_ACTIVATION_NAMES, DeltaWeightGen, MLPFiLM,
                                   NormalizedEmbedding, count_parameters,
                                   film_output_linear, init_delta_zero,
                                   init_film_identity, make_activation,
                                   normalize_activation, per_sample_conv1d)

__all__ = ["MODELS", "ARCHS", "register_model", "build_model", "model_class",
           "FiLMWaveNet", "MLPFiLMWaveNet", "DeltaWaveNet",
           "FiLMWaveNetLayer", "DeltaWaveNetLayer",
           "MLPFiLM", "DeltaWeightGen", "NormalizedEmbedding", "film_output_linear",
           "init_film_identity", "init_delta_zero", "per_sample_conv1d",
           "make_activation", "normalize_activation", "count_parameters",
           "ACTIVATIONS", "NAM_ACTIVATION_NAMES", "A2_ACTIVATION", "A2_CHANNELS",
           "A2_COND_ACTIVATION", "A2_COND_HIDDEN", "A2_DELTA_RANK", "A2_DILATIONS",
           "A2_EMBEDDING_STD", "A2_HEAD_KERNEL", "A2_HEAD_SCALE", "A2_KERNEL_SIZES"]

# Every arch shares the NAM A2 topology and one shared embedding space (so any
# of them can be enrolled); they differ only in how the embedding reaches a layer.
MODELS: dict[str, type] = {
    "film_wavenet": FiLMWaveNet,
    "mlpfilm_wavenet": MLPFiLMWaveNet,
    "delta_wavenet": DeltaWaveNet,
}
ARCHS = tuple(MODELS)


def register_model(name: str):
    """Register an architecture defined outside this package."""
    def wrap(cls):
        MODELS[name] = cls
        return cls
    return wrap


def model_class(arch: str) -> type:
    """The class for an arch name -- descriptors readable without building it."""
    if arch not in MODELS:
        raise ValueError(f"emulate.arch must be one of {tuple(MODELS)}, got {arch!r}")
    return MODELS[arch]


def build_model(ecfg, n_devices: int):
    """Construct the configured architecture from an ``EmulateConfig``."""
    return model_class(ecfg.arch).from_config(ecfg, n_devices)
