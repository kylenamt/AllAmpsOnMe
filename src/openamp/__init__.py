"""openamp — a one-to-many neural amp emulator, from data acquisition to training.

One package, one pipeline, grouped by purpose into subpackages:

    acquire -> corpus.build -> corpus.subset -> corpus.render -> corpus.verify
        -> emulate.train -> emulate.evaluate

Subpackages:

- ``openamp.core``      — shared foundation: config, constants, manifest I/O,
                          small utils, and the diversity-selection engine.
- ``openamp.dsp``       — audio I/O + level/ESR metrics + validation probe
                          (``audio``) and the neural-amp-modeler backend (``nam``).
- ``openamp.acquire``   — TONE3000-specific acquisition (OAuth, API client,
                          catalog heuristics, discover/select/download/validate/
                          dedup/finalize). The one API-facing group.
- ``openamp.corpus``    — build the clean corpus, pick a render subset, GPU-render
                          it through every device, and verify the result.
- ``openamp.emulate``   — the FiLM-conditioned TCN emulator: dataset, training
                          loop, size-comparison harness, and demo export.

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
