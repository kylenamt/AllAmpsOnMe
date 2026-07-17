"""Stage: `openamp discover` — build the 2k–4k candidate model pool (spec §5.2).

Runs several search passes (multiple sorts + a keyword sweep) to escape ranking
bias, fetches each tone's models once, normalizes + filters to amp-only NAM
A1/A2 captures, dedupes by model id, and writes ``candidates.parquet``.

Idempotent: tones already recorded in the candidates file are skipped on re-run,
so the expensive ``/models`` calls are never repeated.
"""

from __future__ import annotations

import logging

import pandas as pd

from openamp.core import manifest
from . import normalize
from .client import APIError, T3KClient
from openamp.core.config import Config
from .catalog import AMP_SEARCH_TERMS
from openamp.core.manifest import DISCOVERY_COLUMNS

log = logging.getLogger("openamp.discover")

DEFAULT_SORTS = ("trending", "downloads", "newest")
FLUSH_EVERY = 25  # tones


def _passes(sorts, terms, base):
    """Enumerate search-pass filter dicts (deduped downstream by tone id)."""
    passes = [dict(sort=s, **base) for s in sorts]
    passes += [dict(query=t, **base) for t in terms]
    return passes


def iter_candidate_tones(client: T3KClient, sorts, terms, base, *,
                         max_pages: int, page_size: int):
    """Yield unique tone records across all passes."""
    seen: set = set()
    for pf in _passes(sorts, terms, base):
        try:
            for tone in client.iter_search(max_pages=max_pages, page_size=page_size, **pf):
                tid = normalize.first(tone, ("id", "tone_id"))
                if tid is None or tid in seen:
                    continue
                seen.add(tid)
                yield tone
        except APIError as exc:
            log.warning("search pass %s failed: %s", pf, exc)
            continue


def discover(client: T3KClient, settings: Config, *,
             sorts=DEFAULT_SORTS, terms=AMP_SEARCH_TERMS,
             gears: str = "amp", architecture: int | str | None = None,
             max_candidates: int = 4000, max_pages: int = 20, page_size: int = 50,
             resume: bool = True, progress: bool = True) -> pd.DataFrame:
    settings.ensure_dirs()
    architecture = settings.architecture if architecture is None else architecture

    existing = (manifest.read_manifest(settings.candidates_path, DISCOVERY_COLUMNS)
                if resume else manifest.empty_manifest(DISCOVERY_COLUMNS))
    done_tone_ids: set = set(existing["tone_id"].dropna().tolist())
    seen_model_ids: set = set(existing["model_id"].dropna().tolist())
    rows: list[dict] = existing.to_dict("records") if not existing.empty else []

    log.info("Resuming with %d existing candidates across %d tones.",
             len(rows), len(done_tone_ids))

    base = {"gears": gears, "architecture": architecture}
    processed = 0
    for tone in iter_candidate_tones(client, sorts, terms, base,
                                     max_pages=max_pages, page_size=page_size):
        tid = normalize.first(tone, ("id", "tone_id"))
        if tid in done_tone_ids:
            continue
        tone_norm = normalize.normalize_tone(tone)
        # Cheap gear-level exclusion before spending a /models call.
        if not normalize.is_amp_gear(tone_norm["gear_type"]):
            done_tone_ids.add(tid)
            continue
        try:
            # The architecture filter is NOT inherited from the tone search: GET
            # /models defaults to A1, so omitting it here harvests A1 captures out
            # of tones the search selected for having A2. Pass it explicitly.
            models = client.list_models(tid, architecture=architecture)
        except APIError as exc:
            log.warning("list_models(%s) failed: %s", tid, exc)
            continue

        for model in models:
            try:
                row = normalize.build_candidate_row(tone_norm, model)
            except Exception as exc:  # defensive: never crash on one bad record
                log.debug("skipping malformed model under tone %s: %s", tid, exc)
                continue
            mid = row["model_id"]
            if mid is None or mid in seen_model_ids:
                continue
            if not normalize.candidate_is_amp_nam(row):
                continue
            rows.append(row)
            seen_model_ids.add(mid)

        done_tone_ids.add(tid)
        processed += 1
        if progress and processed % FLUSH_EVERY == 0:
            log.info("processed %d tones, %d candidate models so far", processed, len(rows))
        if processed % FLUSH_EVERY == 0:
            _flush(rows, settings)
        if len(rows) >= max_candidates:
            log.info("reached max_candidates=%d, stopping discovery", max_candidates)
            break

    df = _flush(rows, settings)
    log.info("discovery complete: %d candidate models across %d tones",
             len(df), df["tone_id"].nunique() if not df.empty else 0)
    return df


def _flush(rows: list[dict], settings: Config) -> pd.DataFrame:
    df = manifest.ensure_schema(pd.DataFrame(rows), DISCOVERY_COLUMNS)[DISCOVERY_COLUMNS] \
        if rows else manifest.empty_manifest(DISCOVERY_COLUMNS)
    manifest.write_manifest(df, settings.candidates_path)
    return df
