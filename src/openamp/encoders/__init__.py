"""Conditioning encoders: something that identifies an amp -> its embedding.

Each one replaces the generator's per-device table lookup with a vector computed
from evidence about the amp. They disagree about what that evidence is, which is
the whole experiment.

**The encoder contract.**

=========================  =====================================================
``embedding_dim: int``     width of the vector emitted; always the *generator's*
``ref_mode: str``          what the dataset must yield as ``ref`` (see below)
``receptive_field: int``   left context the reference needs; 0 for a lookup arm
``__call__(ref, mask)``    ``-> [B, embedding_dim]``; ``mask`` may be ignored
``pooled_dim: int``        width the pooling head emits, before the projection
``grad_checkpoint``        constructor kwarg; a lookup arm accepts and ignores it
``capacity_report()``      ``{head, pooled_dim, embedding_dim, ceiling, ...}``
``from_spec(spec, De)``    construction from an ``EncoderSpec``
``provenance(spec)``       ``(sha256, meta)`` of any precomputed data it reads
=========================  =====================================================

``ref_mode`` is a **class** attribute, not a property, because the training and
evaluation loops both have to build the dataset before the encoder exists. It
picks which of :data:`openamp.data.REF_MODES` the dataset yields:

- ``"audio"``  -- ``[2, ref_samples]``, a real disjoint (dry, wet) window.
- ``"device"`` -- ``[]`` int64, the device id, for an arm that looks a
  precomputed fingerprint up rather than encoding audio on the fly.

That one attribute is why it is here: the same
``enc_kind in ("fingerprint", "dac_frames")`` test used to be written out in
three separate files (``build_joint``, ``train.ref_mode_for``,
``evaluate.ref_mode_for_ck``), so a fourth encoder meant remembering all three.

``spec.kwargs`` reaches the constructor, so the config's ``joint.encoder:`` keys
are the encoder's own parameter names -- except for the one path key each lookup
arm opens before constructing anything.

Documented, not enforced -- a mismatch fails loudly at the call site.

**Adding an encoder** = one new module with a class honouring that contract,
plus either an entry in :data:`ENCODERS` or a ``@register_encoder("name")``
decorator. Then ``enc_kind: <name>`` in the config. The built-ins are listed
explicitly rather than self-registering, so a name is never missing because its
module happened not to be imported.
"""

from __future__ import annotations

from openamp.encoders.fingerprint import (DRY_ID, FINGERPRINT_PREPROCESS,
                                          FingerprintEncoder)
from openamp.encoders.frames import FrameAttentionEncoder
from openamp.encoders.pooling import (AttentiveStatsPool, CrossAttentionPool,
                                      FrameStore, MultiHeadAttentiveStatsPool,
                                      MultiTapStatsHead)
from openamp.encoders.wavenet import WaveNetEncoder, WaveNetEncoderLayer

__all__ = ["ENCODERS", "register_encoder", "build_encoder", "encoder_class",
           "ref_mode_for", "WaveNetEncoder", "WaveNetEncoderLayer",
           "FingerprintEncoder", "FrameAttentionEncoder", "AttentiveStatsPool",
           "MultiHeadAttentiveStatsPool", "MultiTapStatsHead", "CrossAttentionPool",
           "FrameStore", "FINGERPRINT_PREPROCESS", "DRY_ID"]

ENCODERS: dict[str, type] = {
    "wavenet": WaveNetEncoder,
    "fingerprint": FingerprintEncoder,
    "dac_frames": FrameAttentionEncoder,
}


def register_encoder(name: str):
    """Register an encoder defined outside this package."""
    def wrap(cls):
        ENCODERS[name] = cls
        return cls
    return wrap


def encoder_class(kind: str) -> type:
    """The class for an encoder kind -- ``ref_mode`` readable without building it."""
    if kind not in ENCODERS:
        raise ValueError(f"joint.encoder.kind must be one of {tuple(ENCODERS)}, "
                         f"got {kind!r}")
    return ENCODERS[kind]


def ref_mode_for(kind: str) -> str:
    """Which reference an ``enc_kind`` needs, so train and eval cannot disagree.

    Defaults to the WaveNet arm, which keeps checkpoints written before the
    lookup arms existed loading exactly as they did.
    """
    return encoder_class(kind or "wavenet").ref_mode


def build_encoder(spec, embedding_dim: int, **extra):
    """Construct the encoder an :class:`~openamp.spec.EncoderSpec` names.

    ``embedding_dim`` comes from the *generator's* spec, not this one -- it is
    the interface contract, so it has exactly one source of truth.
    """
    return encoder_class(spec.kind).from_spec(spec, embedding_dim, **extra)
