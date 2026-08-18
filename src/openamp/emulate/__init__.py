"""One-to-many amp emulation (spec §4): device-conditioned models that emulate
every rendered device from a single network + a per-device embedding.

The model learns the forward amp *transfer function*, conditioned on a learnable
per-device embedding; architecture and sizes are config-driven (``emulate.arch``
+ knobs), so trying variants is a one-line job.

- :mod:`openamp.emulate.tcn`      — the fully-parametric FiLM-TCN model.
- :mod:`openamp.emulate.wavenet`  — the NAM A2 WaveNet topology, FiLM-conditioned
  (linear FiLM generator), its MLP-generator variant, and the weight-delta variant
  that conditions the conv kernels themselves instead of FiLM.
- :mod:`openamp.emulate.models`   — arch selection (``build_model``).
- :mod:`openamp.emulate.dataset`  — clean-in / render-out training pairs.
- :mod:`openamp.emulate.train`    — the one training script (+ sanity ladder).
- :mod:`openamp.emulate.evaluate` — size-comparison harness + demo export.
- :mod:`openamp.emulate.enroll`   — unseen-device embedding enrollment (Phase 5).

The amortized counterpart to enrollment — mapping an amp straight to a
conditioning vector instead of fitting one — lives in :mod:`openamp.joint_model`.
Nothing here consumes its output; the dependency runs the other way, since that
package reuses this one's :class:`~openamp.emulate.dataset.EmulationDataset`
corpus plumbing and :class:`~openamp.emulate.train.EmulationLoss`.
"""
