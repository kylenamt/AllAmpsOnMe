"""Tone encoder: ``(dry, wet) capture audio -> 512-d tone vector``, in one forward pass.

The amortized counterpart to :mod:`openamp.emulate.enroll`. Enrollment obtains a
device's conditioning vector by *gradient descent* against a frozen network (up to
30 epochs per amp); this package learns a feed-forward map from the capture audio
itself, trained contrastively so two segments of one capture land together and two
different captures do not.

- :mod:`openamp.encoder.model`    — the two-branch (static / dynamic) waveform encoder.
- :mod:`openamp.encoder.dataset`  — P x K capture episodes, and the time-split policy.
- :mod:`openamp.encoder.train`    — SupCon + cross-covariance + probes, and the loop.
- :mod:`openamp.encoder.evaluate` — the diagnostics gate (``openamp encode-eval``).

**Standalone by design.** Nothing in :mod:`openamp.emulate` consumes the tone
vector: whether to wire it into a generator's embedding table is a separate
decision that ``encode-eval`` exists to inform. The dependency runs one way —
``openamp.encoder.dataset`` reuses ``emulate.dataset``'s corpus plumbing (manifest
resolution, source filtering, the NFS read-retry policy), and no emulator module
imports from here.

Runs live in ``results/encoder/<name>/``, deliberately not ``results/emulate/``:
they log a contrastive loss, not ESR, and ``docs/decisions.md`` §6 records what
happened the one time metrics sharing a name got cross-compared.
"""
