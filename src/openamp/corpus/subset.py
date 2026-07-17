"""Diversity-preserving device subset for rendering (spec §4, extension).

The acquisition ``finalize`` keeps *every* validated capture, so the manifest can run
well past the ~400 target (currently 844). Rendering all of them is storage-heavy
and gain-imbalanced: clean DI-amp captures are rare on TONE3000 (~8.5% of the full
set) and concentrated in a handful of tones/creators, so more devices actually
means a *worse* gain balance. This module picks a diverse, gain-balanced subset of
``device_id``s to render, reusing the acquisition selection policy — per-tone/creator/
make-model caps, a per-gain-bucket floor, and brand round-robin fill — but on the
already-validated manifest (no hard filters; those passed at acquisition).

The caps default *looser* than acquisition's (``2/8/6``): we are drawing ~450 from an
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
from pathlib import Path
from typing import Sequence

import pandas as pd

from openamp.core.selection import two_phase_select
from openamp.core.constants import GAIN_CLEAN, GAIN_CRUNCH, GAIN_HIGH

log = logging.getLogger("openamp.subset")

# Diversity caps for the render subset (looser than acquisition's 2/8/6 — see module
# docstring). Round-robin admits at most this many devices per tone / creator /
# (make, model) so no single upload, author, or amp model dominates the set.
DEFAULT_CAP_TONE = 4
DEFAULT_CAP_CREATOR = 12
DEFAULT_CAP_MAKEMODEL = 6

# Guarantee each known gain bucket at least this share of the target size.
DEFAULT_MIN_FRACTION = 0.15
KNOWN_BUCKETS = (GAIN_CLEAN, GAIN_CRUNCH, GAIN_HIGH)

# Columns the selector reads from the acquisition manifest.
SUBSET_COLUMNS = [
    "device_id", "make", "model", "gain_bucket", "downloads", "tone_id", "creator_id",
]


def select_render_subset(manifest: pd.DataFrame, size: int, *,
                         min_fraction: float = DEFAULT_MIN_FRACTION,
                         cap_tone: int = DEFAULT_CAP_TONE,
                         cap_creator: int = DEFAULT_CAP_CREATOR,
                         cap_makemodel: int = DEFAULT_CAP_MAKEMODEL) -> list[int]:
    """Return a sorted list of ``device_id``s: a diverse, gain-balanced subset.

    A thin adapter over :func:`openamp.core.selection.two_phase_select` (shared with
    ``tone3000.select``): rows are identified by ``device_id`` and ordered by
    ``(downloads desc, device_id)`` — a total order, so no RNG is needed. Fewer
    than ``size`` ids come back only when the caps exhaust the pool.
    """
    if "device_id" not in manifest.columns:
        raise ValueError("manifest has no device_id column")
    if size <= 0:
        return []

    order = lambda r: (-int(r.get("downloads") or 0), int(r["device_id"]))
    rows = sorted(manifest.to_dict("records"), key=order)
    selected = two_phase_select(
        rows, size, caps=(cap_tone, cap_creator, cap_makemodel),
        min_fraction=min_fraction, known_buckets=KNOWN_BUCKETS,
        id_of=lambda r: int(r["device_id"]), sort_key=order)

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
        f"# consume with: openamp render --devices @{path}",
    ]
    body = [str(int(d)) for d in device_ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
