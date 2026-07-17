"""One-to-many amp emulation (spec §4): FiLM-conditioned models that emulate
every rendered device from a single network + a per-device embedding.

The model learns the forward amp *transfer function*, conditioned on a learnable
per-device embedding; architecture and sizes are config-driven (``emulate.arch``
+ knobs), so trying variants is a one-line job.

- :mod:`openamp.emulate.tcn`      — the fully-parametric FiLM-TCN model.
- :mod:`openamp.emulate.wavenet`  — the NAM A2 WaveNet topology, FiLM-conditioned.
- :mod:`openamp.emulate.models`   — arch selection (``build_model``).
- :mod:`openamp.emulate.dataset`  — clean-in / render-out training pairs.
- :mod:`openamp.emulate.train`    — the one training script (+ sanity ladder).
- :mod:`openamp.emulate.evaluate` — size-comparison harness + demo export.
- :mod:`openamp.emulate.enroll`   — unseen-device embedding enrollment (Phase 5).
"""
