"""Joint model: a reference encoder trained end-to-end with a FiLM NAM generator.

Today a new amp needs :mod:`openamp.emulate.enroll` — gradient descent on one
fresh embedding against a frozen generator. This package replaces that fitting
step with a feed-forward encoder: give it a short ``(dry, wet)`` reference clip
and it emits the embedding directly, so an unseen amp costs one forward pass
instead of thirty epochs.

What makes the embedding carry *tone* rather than *audio*: it is computed from a
different segment of the recording — in practice a different corpus file — than
the one the generator has to reconstruct. Only what is shared across segments
survives that gap, and what is shared is the amp.

- :mod:`~openamp.joint_model.wavenet_encoder` — the new half; the generator is
  :class:`~openamp.emulate.wavenet.FiLMWaveNet` unchanged, driven through its
  ``forward_emb`` vector path instead of its embedding table.
- :mod:`~openamp.joint_model.dataset` — pairs a generator window with a
  disjoint reference segment of the same device.
- :mod:`~openamp.joint_model.train` — the joint loop; reconstruction loss only.
- :mod:`~openamp.joint_model.evaluate` — the collapse and leakage diagnostics
  that say whether the embedding means anything, which reconstruction ESR alone
  does not.

``guidelines.md`` in this directory is the spec.
"""

from __future__ import annotations

__all__ = ["wavenet_encoder", "dataset", "train", "evaluate"]
