"""One-to-many amp emulation (spec §4): device-conditioned models that emulate
every rendered device from a single network + a per-device embedding.

The model learns the forward amp *transfer function*, conditioned on a learnable
per-device embedding; architecture and sizes are config-driven (``emulate.arch``
+ knobs), so trying variants is a one-line job.

- :mod:`openamp.emulate.train`    — the one training script (+ sanity ladder).
- :mod:`openamp.emulate.enroll`   — unseen-device embedding enrollment (Phase 5).
- :mod:`openamp.emulate.export`   — morph-plugin bundle + pasteable profiles.

The networks themselves are not here: they live in :mod:`openamp.models`, with
the corpus plumbing in :mod:`openamp.data`, the objective in
:mod:`openamp.losses` and the scoring in :mod:`openamp.evaluation`, so all four
are shared with :mod:`openamp.joint_model`.

The amortized counterpart to enrollment — mapping an amp straight to a
conditioning vector instead of fitting one — lives in :mod:`openamp.joint_model`.
Nothing here consumes its output; the dependency runs the other way.
"""
