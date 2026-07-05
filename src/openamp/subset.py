"""Diversity-preserving device subset for rendering (spec §4, extension).

Phase 1 ``finalize`` keeps *every* validated capture, so the manifest can run
well past the ~400 target (currently 844). Rendering all of them is storage-heavy
and gain-imbalanced: clean DI-amp captures are rare on TONE3000 (~8.5% of the full
set) and concentrated in a handful of tones/creators, so more devices actually
means a *worse* gain balance. This module picks a diverse, gain-balanced subset of
``device_id``s to render, reusing the Phase 1 selection policy — per-tone/creator/
make-model caps, a per-gain-bucket floor, and brand round-robin fill — but on the
already-validated manifest (no hard filters; those passed in Phase 1).

The caps default *looser* than Phase 1's (``2/8/6``): we are drawing ~450 from an
already-curated 844, and the scarce, concentrated clean captures cannot satisfy a
15% floor under the strict caps (they top out near 267 devices). ``4/12/6`` reaches
~450 with clean near 12%.

Deterministic given the manifest and parameters: rows are ranked by
``(downloads desc, device_id)``, and the chosen ids are persisted to a file, so
``render run --devices @<path>`` and its resumes always see the same set. The
manifest itself is never modified — enlarge ``size`` later to top up.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from t3k.constants import GAIN_CLEAN, GAIN_CRUNCH, GAIN_HIGH

log = logging.getLogger("oa3k.subset")

# Diversity caps for the render subset (looser than Phase 1's 2/8/6 — see module
# docstring). Round-robin admits at most this many devices per tone / creator /
# (make, model) so no single upload, author, or amp model dominates the set.
DEFAULT_CAP_TONE = 4
DEFAULT_CAP_CREATOR = 12
DEFAULT_CAP_MAKEMODEL = 6

# Guarantee each known gain bucket at least this share of the target size.
DEFAULT_MIN_FRACTION = 0.15
KNOWN_BUCKETS = (GAIN_CLEAN, GAIN_CRUNCH, GAIN_HIGH)

# Columns the selector reads from the Phase 1 manifest.
SUBSET_COLUMNS = [
    "device_id", "make", "model", "gain_bucket", "downloads", "tone_id", "creator_id",
]


def _pool_by_make(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by make, each queue ordered best-first (downloads desc)."""
    pools: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pools[str(row["make"])].append(row)
    for make in pools:
        pools[make].sort(key=lambda r: (-int(r.get("downloads") or 0), int(r["device_id"])))
    return pools


def _round_robin(pools: dict[str, list[dict]], need: int, chosen: set[int],
                 caps: tuple[int, int, int],
                 counters: tuple[Counter, Counter, Counter],
                 make_order: Sequence[str]) -> list[dict]:
    """Greedy round-robin across makes, respecting caps. Consumes ``pools``."""
    cap_tone, cap_creator, cap_mm = caps
    tone_c, creator_c, mm_c = counters
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
                did = int(cand["device_id"])
                if did in chosen:
                    continue
                if not (tone_c[cand["tone_id"]] < cap_tone
                        and creator_c[cand["creator_id"]] < cap_creator
                        and mm_c[(cand["make"], cand["model"])] < cap_mm):
                    continue  # caps only tighten -> safe to drop permanently
                tone_c[cand["tone_id"]] += 1
                creator_c[cand["creator_id"]] += 1
                mm_c[(cand["make"], cand["model"])] += 1
                chosen.add(did)
                picked.append(cand)
                need -= 1
                progressed = True
                break
            if need <= 0:
                break
        if not progressed:
            break
    return picked


def select_render_subset(manifest: pd.DataFrame, size: int, *,
                         min_fraction: float = DEFAULT_MIN_FRACTION,
                         cap_tone: int = DEFAULT_CAP_TONE,
                         cap_creator: int = DEFAULT_CAP_CREATOR,
                         cap_makemodel: int = DEFAULT_CAP_MAKEMODEL) -> list[int]:
    """Return a sorted list of ``device_id``s: a diverse, gain-balanced subset.

    Mirrors ``t3k.select``: guarantee each known gain bucket its ``min_fraction``
    share (Phase A), then fill the remainder by brand round-robin (Phase B). Fewer
    than ``size`` ids come back only when the caps exhaust the pool — the caller
    reports the shortfall.
    """
    if "device_id" not in manifest.columns:
        raise ValueError("manifest has no device_id column")
    if size <= 0:
        return []

    rows = manifest.to_dict("records")
    # Total order (device_id is unique) -> fully deterministic, no RNG needed.
    rows.sort(key=lambda r: (-int(r.get("downloads") or 0), int(r["device_id"])))

    caps = (cap_tone, cap_creator, cap_makemodel)
    counters: tuple[Counter, Counter, Counter] = (Counter(), Counter(), Counter())
    chosen: set[int] = set()
    selected: list[dict] = []
    make_order = sorted({str(r["make"]) for r in rows})

    # Phase A: guarantee each known gain bucket reaches its minimum share.
    per_bucket_min = math.ceil(min_fraction * size)
    for bucket in KNOWN_BUCKETS:
        bucket_rows = [r for r in rows
                       if r.get("gain_bucket") == bucket and int(r["device_id"]) not in chosen]
        need = min(per_bucket_min, max(0, size - len(selected)))
        selected += _round_robin(_pool_by_make(bucket_rows), need, chosen,
                                 caps, counters, make_order)

    # Phase B: fill the remainder via brand round-robin over everything left.
    remaining = [r for r in rows if int(r["device_id"]) not in chosen]
    selected += _round_robin(_pool_by_make(remaining), size - len(selected),
                             chosen, caps, counters, make_order)

    log.info("selected %d/%d devices (caps %d/%d/%d, clean floor %.0f%%)",
             len(selected), size, cap_tone, cap_creator, cap_makemodel, 100 * min_fraction)
    return sorted(int(r["device_id"]) for r in selected)


def summarize(manifest: pd.DataFrame, device_ids: Sequence[int]) -> dict:
    """Diversity/balance stats for a chosen subset (for the report + file header)."""
    ids = {int(i) for i in device_ids}
    sub = manifest[manifest["device_id"].astype(int).isin(ids)]
    total = len(sub)
    by_bucket = {str(k): int(v) for k, v in sub["gain_bucket"].value_counts().items()}
    return {
        "total": total,
        "n_makes": int(sub["make"].nunique()),
        "n_creators": int(sub["creator_id"].nunique()),
        "n_tones": int(sub["tone_id"].nunique()),
        "by_bucket": by_bucket,
    }


def _bucket_line(summary: dict) -> str:
    total = summary["total"] or 1
    parts = [f"{b} {n} ({100 * n / total:.1f}%)"
             for b, n in sorted(summary["by_bucket"].items(), key=lambda kv: -kv[1])]
    return "gain: " + "  ".join(parts)


def format_summary(summary: dict, target: int | None = None) -> str:
    head = f"Render subset: {summary['total']} devices"
    if target is not None and summary["total"] < target:
        head += f" (target {target}; caps exhausted the pool)"
    return "\n".join([
        head,
        f"  makes={summary['n_makes']}  creators={summary['n_creators']}  tones={summary['n_tones']}",
        "  " + _bucket_line(summary),
    ])


def write_subset_file(path: Path, device_ids: Sequence[int], summary: dict, *,
                      target: int, caps: tuple[int, int, int],
                      min_fraction: float) -> None:
    """Persist the id list (one per line) with a self-describing comment header.

    Read back by ``render run --devices @<path>``; ``#`` comments are ignored.
    """
    path = Path(path)
    header = [
        f"# Open-Amp3000 render subset — {len(device_ids)} devices (target {target})",
        f"# caps: tone<={caps[0]} creator<={caps[1]} makemodel<={caps[2]} | clean floor {min_fraction:.0%}",
        f"# {_bucket_line(summary)}",
        f"# consume with: oa3k render run --devices @{path}",
    ]
    body = [str(int(d)) for d in device_ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
