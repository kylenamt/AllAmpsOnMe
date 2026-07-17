"""Shared diversity-preserving selection engine (caps + gain floor + round-robin).

Both acquisition selection (:mod:`openamp.acquire.select`, over candidate models)
and render-subset selection (:mod:`openamp.corpus.subset`, over the finalized manifest)
pick a diverse, gain-balanced subset under the same policy:

  Phase A — guarantee each known gain bucket at least ``min_fraction`` of ``target``.
  Phase B — fill the remainder by brand round-robin over everything left.

Both phases respect per-tone / per-creator / per-(make, model) caps so no single
upload, author, or amp model dominates. The two callers differ only in row schema
(which column is the identity, and the within-make tie-break), so those are passed
in as ``id_of`` / ``sort_key`` and the engine here stays schema-agnostic.

Rows are plain dicts carrying at least ``make``, ``model``, ``tone_id``,
``creator_id``, and ``gain_bucket``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Hashable, Sequence

Row = dict
Caps = tuple[int, int, int]                 # (per_tone, per_creator, per_makemodel)
Counters = tuple[Counter, Counter, Counter]  # tone / creator / (make, model)


def _caps_ok(row: Row, counters: Counters, caps: Caps) -> bool:
    tone_c, creator_c, mm_c = counters
    cap_tone, cap_creator, cap_mm = caps
    return (tone_c[row["tone_id"]] < cap_tone
            and creator_c[row["creator_id"]] < cap_creator
            and mm_c[(row["make"], row["model"])] < cap_mm)


def _apply_caps(row: Row, counters: Counters) -> None:
    tone_c, creator_c, mm_c = counters
    tone_c[row["tone_id"]] += 1
    creator_c[row["creator_id"]] += 1
    mm_c[(row["make"], row["model"])] += 1


def _pool_by_make(rows: Sequence[Row], sort_key: Callable[[Row], object]
                  ) -> dict[str, list[Row]]:
    """Group rows by make, each queue ordered best-first by ``sort_key``."""
    pools: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        pools[str(row["make"])].append(row)
    for make in pools:
        pools[make].sort(key=sort_key)
    return pools


def _round_robin(pools: dict[str, list[Row]], need: int, chosen: set,
                 caps: Caps, counters: Counters, make_order: Sequence[str],
                 id_of: Callable[[Row], Hashable]) -> list[Row]:
    """Greedy round-robin across makes, respecting caps. Consumes ``pools``."""
    picked: list[Row] = []
    while need > 0:
        progressed = False
        for make in make_order:
            queue = pools.get(make)
            if not queue:
                continue
            # Pop until we find a usable candidate for this make this round.
            while queue:
                cand = queue.pop(0)
                if id_of(cand) in chosen:
                    continue
                if not _caps_ok(cand, counters, caps):
                    continue  # caps only tighten -> safe to drop permanently
                _apply_caps(cand, counters)
                chosen.add(id_of(cand))
                picked.append(cand)
                need -= 1
                progressed = True
                break
            if need <= 0:
                break
        if not progressed:
            break
    return picked


def two_phase_select(rows: Sequence[Row], target: int, *, caps: Caps,
                     min_fraction: float, known_buckets: Sequence[str],
                     id_of: Callable[[Row], Hashable],
                     sort_key: Callable[[Row], object]) -> list[Row]:
    """Return the chosen rows: gain-floor Phase A then brand round-robin Phase B.

    Fewer than ``target`` rows come back only when the caps exhaust the pool; the
    caller reports any shortfall. Deterministic given a deterministically ordered
    ``rows`` and the same parameters.
    """
    counters: Counters = (Counter(), Counter(), Counter())
    chosen: set = set()
    selected: list[Row] = []
    make_order = sorted({str(r["make"]) for r in rows})

    # Phase A: guarantee each known gain bucket reaches its minimum share.
    per_bucket_min = math.ceil(min_fraction * target)
    for bucket in known_buckets:
        bucket_rows = [r for r in rows
                       if r.get("gain_bucket") == bucket and id_of(r) not in chosen]
        need = min(per_bucket_min, max(0, target - len(selected)))
        selected += _round_robin(_pool_by_make(bucket_rows, sort_key), need, chosen,
                                 caps, counters, make_order, id_of)

    # Phase B: fill the remainder via brand round-robin over everything left.
    remaining = [r for r in rows if id_of(r) not in chosen]
    selected += _round_robin(_pool_by_make(remaining, sort_key), target - len(selected),
                             chosen, caps, counters, make_order, id_of)
    return selected
