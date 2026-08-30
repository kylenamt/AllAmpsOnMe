"""openamp — a one-to-many neural amp emulator, from data acquisition to training.

One package, one pipeline, grouped by purpose into subpackages:

    acquire -> corpus.build -> corpus.subset -> corpus.render -> corpus.verify
        -> training.{emulate,joint} -> evaluation.{emulate,joint} -> export

**Getting the corpus.** Stable; rarely touched.

- ``openamp.core``      — shared foundation: pipeline config, constants,
                          manifest I/O, small utils, diversity selection.
- ``openamp.dsp``       — audio I/O + level/ESR metrics + validation probe
                          (``audio``) and the neural-amp-modeler backend (``nam``).
- ``openamp.acquire``   — TONE3000-specific acquisition (OAuth, API client,
                          catalog heuristics, discover/select/download/validate/
                          dedup/finalize). The one API-facing group.
- ``openamp.corpus``    — build the clean corpus, pick a render subset, GPU-render
                          it through every device, and verify the result.

**Training on it.** Each of these is a seam: one registry, one contract, and a
config section that names which implementation to use. See ``openamp.spec``.

- ``openamp.spec``      — the resolved experiment (``RunSpec``). The boundary:
                          below it nothing reads a file or an env var.
- ``openamp.data``      — datasets and dataloaders over a ``CorpusPaths``.
- ``openamp.models``    — the conditioned architectures.
- ``openamp.encoders``  — conditioning encoders, for the joint arm.
- ``openamp.losses``    — the objectives.
- ``openamp.training``  — the two loops, plus the bookkeeping they share.

**Reading the result.**

- ``openamp.evaluation``— the ESR definitions and the scoring harnesses.
- ``openamp.enroll``    — fitting an embedding for an unseen amp.
- ``openamp.export``    — the morph plugin's weight bundle and profiles.
- ``openamp.probes``    — standalone feasibility probes, wired into nothing.

The CLI (``openamp.cli``) imports each stage module lazily inside its command
body, so metadata-only commands and ``--help`` stay fast without every heavy
dependency (torch, NAM, soundfile) loading up front. ``openamp.dsp.audio`` and
``openamp.dsp.nam`` still import their backends lazily to turn a missing optional
dependency into a clear install hint.

Run ``openamp --help`` for the CLI, or see docs/architecture.md and README.md.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.2.0"
