"""Scoring a trained run: the metric definitions, and the harnesses that use them.

- :mod:`~openamp.evaluation.metrics` -- the two ESR definitions (pooled and
  per-window) that every caller shares, plus the per-device breakdown.
- :mod:`~openamp.evaluation.emulate` -- ``load_model``, the comparison CSV, the
  per-device validation, and the listening demos.
- :mod:`~openamp.evaluation.joint` -- the collapse and leakage diagnostics.
- :mod:`~openamp.evaluation.geometry` -- k-NN retrieval and the t-SNE writer.

The last two are deliberately *not* re-exported here. Importing them pulls in
the whole joint stack, and ``evaluation.joint`` imports ``joint_model.train``,
which imports ``emulate.train``, which imports the metrics below -- so
re-exporting it would close an import cycle through this file. Their callers
name the submodule directly, which is what they already did.
"""

from __future__ import annotations

from openamp.evaluation.emulate import load_model, resolve_device
from openamp.evaluation.metrics import (SILENCE_DBFS, per_device_esr,
                                        pooled_esr_terms, silence_floor,
                                        windowed_esr)

__all__ = ["load_model", "resolve_device", "SILENCE_DBFS", "silence_floor",
           "pooled_esr_terms", "windowed_esr", "per_device_esr"]
