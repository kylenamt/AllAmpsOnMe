"""Stage: `t3k select` — deterministic diversity selection of ~430 (spec §5.3).

Hard filters -> caps (per tone/creator/make-model) -> gain-style minimums ->
brand round-robin fill. Fully deterministic given the same candidates + seed, so
re-running yields the identical selection.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict

import pandas as pd

from . import manifest, normalize
from .config import Settings
from .constants import GAIN_CLEAN, GAIN_CRUNCH, GAIN_HIGH
from .manifest import ALL_COLUMNS, STATUS_SELECTED

log = logging.getLogger("t3k.select")

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


def _caps_ok(row: dict, tone_c: Counter, creator_c: Counter, mm_c: Counter) -> bool:
    return (
        tone_c[row["tone_id"]] < MAX_PER_TONE
        and creator_c[row["creator_id"]] < MAX_PER_CREATOR
        and mm_c[(row["make"], row["model"])] < MAX_PER_MAKEMODEL
    )


def _apply_caps(row: dict, tone_c: Counter, creator_c: Counter, mm_c: Counter) -> None:
    tone_c[row["tone_id"]] += 1
    creator_c[row["creator_id"]] += 1
    mm_c[(row["make"], row["model"])] += 1


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


def _pool_by_make(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by make, each queue ordered best-first (downloads desc)."""
    pools: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pools[row["make"]].append(row)
    for make in pools:
        pools[make].sort(key=lambda r: (-int(r.get("downloads") or 0), str(r.get("model_id"))))
    return pools


def _round_robin(pools: dict[str, list[dict]], need: int, chosen_ids: set,
                 tone_c: Counter, creator_c: Counter, mm_c: Counter,
                 make_order: list[str]) -> list[dict]:
    """Greedy round-robin across makes, respecting caps. Consumes ``pools``."""
    picked: list[dict] = []
    while need > 0:
        progressed = False
        for make in make_order:
            queue = pools.get(make)
            if not queue:
                continue
            # Pop until we find a usable candidate for this make this round.
            while queue:
                cand = queue.pop(0)
                if cand["model_id"] in chosen_ids:
                    continue
                if not _caps_ok(cand, tone_c, creator_c, mm_c):
                    continue  # caps only tighten -> safe to drop permanently
                _apply_caps(cand, tone_c, creator_c, mm_c)
                chosen_ids.add(cand["model_id"])
                picked.append(cand)
                need -= 1
                progressed = True
                break
            if need <= 0:
                break
        if not progressed:
            break
    return picked


def select(candidates: pd.DataFrame, settings: Settings, *,
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
    rng.shuffle(rows)
    rows.sort(key=lambda r: (-int(r.get("downloads") or 0), str(r.get("model_id"))))

    tone_c: Counter = Counter()
    creator_c: Counter = Counter()
    mm_c: Counter = Counter()
    chosen_ids: set = set()
    selected: list[dict] = []

    make_order = sorted({r["make"] for r in rows})

    # Phase A: guarantee each known gain bucket reaches its minimum share.
    per_bucket_min = math.ceil(BUCKET_MIN_FRACTION * target)
    for bucket in KNOWN_BUCKETS:
        bucket_rows = [r for r in rows if r.get("gain_bucket") == bucket and r["model_id"] not in chosen_ids]
        pools = _pool_by_make(bucket_rows)
        need = min(per_bucket_min, max(0, target - len(selected)))
        selected += _round_robin(pools, need, chosen_ids, tone_c, creator_c, mm_c, make_order)

    # Phase B: fill the remainder via brand round-robin over everything left.
    remaining = [r for r in rows if r["model_id"] not in chosen_ids]
    pools = _pool_by_make(remaining)
    selected += _round_robin(pools, target - len(selected), chosen_ids,
                             tone_c, creator_c, mm_c, make_order)

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
