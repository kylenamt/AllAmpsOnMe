"""Everything a training run writes to disk, and the check that guards a resume.

Both trainers produce the same set of files under ``results/<stage>/<name>/``:

===========================  ===================================================
``checkpoint.pt``            best so far, by val ESR
``last.pt``                  most recent epoch, what ``--resume`` reads
``train_log.csv``            one row per epoch; the columns are the loop's own
``config_history.jsonl``     append-only, one line per start or resume
``resolved_config.yaml``     the whole :class:`~openamp.spec.RunSpec`, corpus and
                             seed included -- enough to rebuild the experiment
``metrics.json``             the end-of-run summary
===========================  ===================================================

:class:`RunDir` owns the writing; the loops own what goes in. That split is the
point -- the payloads differ between the two trainers (one model vs an
encoder/generator pair, different CSV columns), the file handling does not, and
it was duplicated line for line.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

__all__ = ["RunDir", "check_resume_compatible", "same_config_value"]


class RunDir:
    """The output directory of one run, created on construction."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.path / "checkpoint.pt"     # best (min val ESR)
        self.last = self.path / "last.pt"
        self.log = self.path / "train_log.csv"

    # --- Per-epoch log ------------------------------------------------------------
    def start_log(self, columns) -> None:
        """Truncate the CSV and write its header. A resume skips this and appends."""
        with self.log.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(list(columns))

    def append_log(self, row) -> None:
        with self.log.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(list(row))

    # --- Checkpoints ---------------------------------------------------------------
    def save_checkpoint(self, path: Path, payload: dict) -> None:
        """Write through a temporary file, so a crash never truncates a checkpoint."""
        tmp = Path(path).with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    # --- Config provenance ----------------------------------------------------------
    def append_config_history(self, *, mode: str, start_epoch: int, resumed_from,
                              best_val: float, max_epochs: int, **extra) -> None:
        """Append one JSON line per training session (start or resume).

        A never-overwritten audit trail of how each run and resume was
        configured, complementing the end-of-run ``config.yaml`` (the
        as-finished snapshot). ``extra`` is the stage's own fields, written after
        the common ones in the order given.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "start_epoch": start_epoch,
            "resumed_from_epoch": resumed_from,
            "best_val_esr_at_start": best_val if math.isfinite(best_val) else None,
            "max_epochs": max_epochs,
            **extra,
        }
        with (self.path / "config_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def save_spec(self, spec) -> None:
        """Write the resolved experiment beside its outputs.

        Not the config *file* the user passed: the resolved spec, so a run
        records the corpus, seed and name it actually used rather than leaving
        them to an environment variable nobody wrote down.
        """
        import yaml
        (self.path / "resolved_config.yaml").write_text(
            yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")

    def save_metrics(self, metrics: dict) -> dict:
        (self.path / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                                encoding="utf-8")
        return metrics


# --- Resume compatibility ---------------------------------------------------------
def same_config_value(a, b) -> bool:
    """Compare config values that may be list-vs-tuple after a yaml round-trip."""
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a or []) == list(b or [])
    return a == b


def check_resume_compatible(ck: dict, sections, *, receptive_field: int,
                            n_devices: int, dynamics: str) -> None:
    """Fail fast, before the strict ``load_state_dict``, if the config changed a
    knob that defines the checkpoint's shape -- turning a cryptic shape error
    into a message naming the field.

    ``sections`` is one ``(saved, current, prefix)`` tuple per *structural*
    section of the run spec. There is no key list: the v2 schema draws ``model:``
    and ``joint.encoder:`` so that every key in them changes the network, so the
    whole section is compared and a new architecture's knobs are covered without
    anyone remembering to add them. ``dynamics`` names the fields that *are* free
    to change, for the error message.
    """
    changed = []
    for saved, current, prefix in sections:
        for k in sorted(set(saved) | set(current)):
            if not same_config_value(saved.get(k), current.get(k)):
                changed.append(f"{prefix}{k}: {saved.get(k)!r} -> {current.get(k)!r}")
    if ck.get("receptive_field") not in (None, receptive_field):
        changed.append(f"receptive_field: {ck['receptive_field']} -> {receptive_field}")
    if ck.get("device_ids") is not None and len(ck["device_ids"]) != n_devices:
        changed.append(f"device_count: {len(ck['device_ids'])} -> {n_devices}")
    if changed:
        raise RuntimeError(
            "cannot --resume: the config changed structural knobs that define the "
            "checkpoint's shape (" + "; ".join(changed) + "). Start a fresh run under "
            f"a new config name, or revert these fields; only training dynamics "
            f"({dynamics}) may change on resume.")
