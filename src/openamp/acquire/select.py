"""Stage: `openamp select` — deterministic diversity selection of ~430 (spec §5.3).

Hard filters -> caps (per tone/creator/make-model) -> gain-style minimums ->
brand round-robin fill. Fully deterministic given the same candidates + seed, so
re-running yields the identical selection.
"""

from __future__ import annotations

import logging
import random

import pandas as pd

from openamp.core import manifest
from openamp.core.selection import two_phase_select
from . import normalize
from openamp.core.config import Config
from .catalog import GAIN_CLEAN, GAIN_CRUNCH, GAIN_HIGH
from openamp.core.manifest import ALL_COLUMNS, STATUS_SELECTED

log = logging.getLogger("openamp.select")

# Caps (spec §5.3.2).
MAX_PER_TONE = 2
MAX_PER_CREATOR = 8
MAX_PER_MAKEMODEL = 6

# Gain-style target: >=20% of the selection in each known bucket (spec §5.3.3).
KNOWN_BUCKETS = (GAIN_CLEAN, GAIN_CRUNCH, GAIN_HIGH)
BUCKET_MIN_FRACTION = 0.20

# License strings that forbid use — candidates matching are excluded (spec §5.3.1).
NO_USE_LICENSE_KEYWORDS = (
    "no use", "do not use", "no download", "no redistribution allowed",
    "all rights reserved", "private", "not for use",
)


def _license_ok(license_text: str) -> bool:
    t = (license_text or "").strip().lower()
    return not any(kw in t for kw in NO_USE_LICENSE_KEYWORDS)


def _eligible(df: pd.DataFrame, exclude_ids: set | None = None) -> pd.DataFrame:
    """Apply hard filters (spec §5.3.1) and return a normalized eligible frame."""
    df = manifest.ensure_schema(df)
    exclude_ids = exclude_ids or set()
    rows = []
    for row in df.to_dict("records"):
        if row.get("model_id") in exclude_ids:
            continue
        gain = row.get("gain_bucket")
        if not isinstance(gain, str) or not gain:
            row["gain_bucket"] = normalize.classify_gain(
                normalize.text_list(row.get("tags")),
                str(row.get("tone_title") or ""), str(row.get("model_name") or ""))
        # DI amp only.
        if not normalize.is_amp_gear(row.get("gear_type")):
            continue
        if row.get("architecture") not in ("A1", "A2"):
            continue
        if not str(row.get("model_url") or "").strip():
            continue
        if not _license_ok(str(row.get("license") or "")):
            continue
        if not str(row.get("make") or "").strip():
            row["make"] = "unknown"
        rows.append(row)
    return pd.DataFrame(rows) if rows else df.iloc[0:0]


def select(candidates: pd.DataFrame, settings: Config, *,
           target: int | None = None, seed: int | None = None,
           exclude_ids: set | None = None) -> pd.DataFrame:
    target = target or settings.select_target
    seed = settings.seed if seed is None else seed
    rng = random.Random(seed)

    eligible = _eligible(candidates, exclude_ids)
    if eligible.empty:
        log.warning("no eligible candidates after hard filters")
        return manifest.empty_manifest()

    rows = eligible.to_dict("records")
    # Deterministic base order; seed only nudges equal-download ties.
    order = lambda r: (-int(r.get("downloads") or 0), str(r.get("model_id")))
    rng.shuffle(rows)
    rows.sort(key=order)

    selected = two_phase_select(
        rows, target, caps=(MAX_PER_TONE, MAX_PER_CREATOR, MAX_PER_MAKEMODEL),
        min_fraction=BUCKET_MIN_FRACTION, known_buckets=KNOWN_BUCKETS,
        id_of=lambda r: r["model_id"], sort_key=order)

    out = manifest.ensure_schema(pd.DataFrame(selected))
    out["status"] = STATUS_SELECTED
    out = out[ALL_COLUMNS]
    log.info("selected %d captures (target %d)", len(out), target)
    return out


def summarize(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {"total": 0, "by_make": {}, "by_bucket": {}}
    return {
        "total": len(selected),
        "n_makes": int(selected["make"].nunique()),
        "by_make": selected["make"].value_counts().to_dict(),
        "by_bucket": selected["gain_bucket"].value_counts().to_dict(),
    }


def format_summary(summary: dict) -> str:
    lines = [f"Selected {summary['total']} captures across {summary.get('n_makes', 0)} makes."]
    lines.append("\nBy gain bucket:")
    for bucket, n in sorted(summary.get("by_bucket", {}).items(), key=lambda kv: -kv[1]):
        pct = 100 * n / summary["total"] if summary["total"] else 0
        lines.append(f"  {bucket:12s} {n:4d}  ({pct:4.1f}%)")
    lines.append("\nBy make:")
    for make, n in sorted(summary.get("by_make", {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"  {make:20s} {n:4d}")
    return "\n".join(lines)
