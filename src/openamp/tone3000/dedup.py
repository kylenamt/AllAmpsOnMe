"""Stage: `t3k dedup` — remove near-duplicate captures (spec §5.6).

Two passes over the validated set:
  1. Exact file-hash duplicates dropped globally.
  2. Probe-output ESR within each normalized (make, model) group; pairs with
     ESR < 0.01 collapse to the higher-download-count capture.
The loser of each duplicate pair gets ``status=duplicate``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import manifest
from .config import Settings
from .manifest import STATUS_DUPLICATE, STATUS_VALIDATED
from .probe import esr

log = logging.getLogger("t3k.dedup")

ESR_DUPLICATE_THRESHOLD = 0.01


def _downloads(row: pd.Series) -> int:
    try:
        return int(row.get("downloads") or 0)
    except (TypeError, ValueError):
        return 0


def _load_probe(path) -> np.ndarray | None:
    try:
        if path is None or not Path(str(path)).is_file():
            return None
        return np.load(str(path)).astype(np.float32).reshape(-1)
    except Exception:
        return None


def dedup(df: pd.DataFrame, settings: Settings, *,
          esr_threshold: float = ESR_DUPLICATE_THRESHOLD) -> pd.DataFrame:
    df = manifest.ensure_schema(df).reset_index(drop=True)
    live = df.index[df["status"] == STATUS_VALIDATED].tolist()

    # --- Pass 1: exact file-hash duplicates (global) --------------------------
    by_hash: dict[str, list[int]] = {}
    for i in live:
        h = df.at[i, "sha256"]
        if isinstance(h, str) and h:
            by_hash.setdefault(h, []).append(i)
    dropped_hash = 0
    for idxs in by_hash.values():
        if len(idxs) < 2:
            continue
        keep = max(idxs, key=lambda i: _downloads(df.loc[i]))
        for i in idxs:
            if i != keep:
                _mark_dup(df, i, keep, "exact_file_hash")
                dropped_hash += 1

    # --- Pass 2: ESR within normalized (make, model) groups -------------------
    live = df.index[df["status"] == STATUS_VALIDATED].tolist()
    groups: dict[tuple, list[int]] = {}
    for i in live:
        key = (df.at[i, "make"], df.at[i, "model"])
        groups.setdefault(key, []).append(i)

    probes: dict[int, np.ndarray] = {}
    for i in live:
        p = _load_probe(df.at[i, "probe_path"])
        if p is not None:
            probes[i] = p

    dropped_esr = 0
    for key, idxs in groups.items():
        # Higher downloads first so a survivor is always the more popular one.
        idxs = sorted(idxs, key=lambda i: -_downloads(df.loc[i]))
        for a_pos in range(len(idxs)):
            a = idxs[a_pos]
            if df.at[a, "status"] != STATUS_VALIDATED or a not in probes:
                continue
            for b in idxs[a_pos + 1:]:
                if df.at[b, "status"] != STATUS_VALIDATED or b not in probes:
                    continue
                if esr(probes[b], probes[a]) < esr_threshold:
                    _mark_dup(df, b, a, "esr_near_duplicate")
                    dropped_esr += 1

    manifest.write_manifest(df, settings.manifest_path)
    log.info("dedup complete: %d hash dups, %d ESR dups removed", dropped_hash, dropped_esr)
    return df


def _mark_dup(df: pd.DataFrame, loser: int, keeper: int, reason: str) -> None:
    df.at[loser, "status"] = STATUS_DUPLICATE
    df.at[loser, "reject_reason"] = f"{reason} of model {df.at[keeper, 'model_id']}"
