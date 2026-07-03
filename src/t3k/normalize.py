"""Defensive normalization of live TONE3000 API objects into candidate rows.

The live schema (github.com/tone-3000/api ``types.ts``) differs from the field
names in the spec — e.g. ``gear`` (not ``gears``), ``architecture_version``,
``downloads_count``, tone-level ``makes[]``, and ``license``/``tags`` may be
objects rather than plain strings. Every accessor here tolerates missing,
renamed, or wrapped fields and falls back to the spec names, so the pipeline
survives schema drift (spec §3, §5.2). Deviations are documented in the README.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .constants import (
    AMP_GEAR_VALUES,
    EXCLUDE_KEYWORDS,
    GAIN_CLEAN,
    GAIN_CRUNCH,
    GAIN_HIGH,
    GAIN_KEYWORDS,
    GAIN_UNKNOWN,
    MAKE_ALIASES,
    MODEL_TO_MAKE,
    NAM_FORMAT_VALUES,
)

_WS = re.compile(r"\s+")


def first(d: dict, keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


def text(value: Any) -> str:
    """Coerce a scalar/object (str | {name|slug|title|id|value: ...}) to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in ("name", "slug", "title", "label", "identifier", "id", "value"):
            if value.get(key):
                return str(value[key]).strip()
        return ""
    return str(value).strip()


def text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = [text(v) for v in value]
        return [x for x in out if x]
    single = text(value)
    return [single] if single else []


def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())


# --- classification helpers -----------------------------------------------------
def is_amp_gear(gear: Any) -> bool:
    return _norm(text(gear)) in AMP_GEAR_VALUES


def is_nam_format(fmt: Any) -> bool:
    t = _norm(text(fmt))
    return t in NAM_FORMAT_VALUES if t else True  # absent format -> don't exclude here


def map_architecture(value: Any) -> str | None:
    """Map an ``architecture_version`` to canonical ``"A2"`` / ``"A1"`` (or None)."""
    t = _norm(text(value))
    if not t:
        return None
    if t in {"a2", "2", "v2"} or t.startswith("2"):
        return "A2"
    if t in {"a1", "1", "v1"} or t.startswith("1"):
        return "A1"
    return None


def should_exclude(*texts: str) -> bool:
    """True if any exclusion keyword appears in the combined metadata text."""
    blob = " " + _norm(" ".join(t for t in texts if t)) + " "
    return any(kw in blob for kw in EXCLUDE_KEYWORDS)


def normalize_make(raw: Any) -> str:
    t = _norm(text(raw))
    return MAKE_ALIASES.get(t, t)


def derive_make(makes: Any, title: str, model_name: str) -> str:
    """Pick a canonical make from tone ``makes[]`` else infer from title tokens."""
    for m in text_list(makes):
        canon = normalize_make(m)
        if canon:
            return canon
    blob = _norm(f"{title} {model_name}")
    for token, make in MODEL_TO_MAKE.items():
        if token in blob:
            return make
    # last resort: first word of the title
    head = _norm(title).split(" ")
    return normalize_make(head[0]) if head and head[0] else ""


def normalize_model(model_name: str, title: str, make: str) -> str:
    """A normalized (lowercase, trimmed) model token for capping/dedup grouping."""
    base = _norm(model_name) or _norm(title)
    if make and base.startswith(make + " "):
        base = base[len(make) + 1:]
    return base.strip()


def classify_gain(tags: list[str], title: str, model_name: str) -> str:
    blob = _norm(" ".join([*tags, title, model_name]))
    for bucket in (GAIN_HIGH, GAIN_CRUNCH, GAIN_CLEAN):  # most-specific first
        if any(kw in blob for kw in GAIN_KEYWORDS[bucket]):
            return bucket
    return GAIN_UNKNOWN


# --- row builders ---------------------------------------------------------------
def normalize_tone(tone: dict) -> dict:
    """Extract the tone-level fields shared by all its models."""
    title = text(first(tone, ("title", "name")))
    tags = text_list(tone.get("tags"))
    makes = tone.get("makes") if "makes" in tone else tone.get("make")
    user = tone.get("user") or {}
    return {
        "tone_id": first(tone, ("id", "tone_id")),
        "tone_title": title,
        "tone_url": text(first(tone, ("url", "tone_url"))),
        "gear_type": text(first(tone, ("gear", "gear_type", "gears"))),
        "format": text(first(tone, ("format", "platform"))),
        "license": text(first(tone, ("license",), "")) or "",
        "tags": tags,
        "makes": makes,
        "creator": text(first(user, ("username", "name")) or first(tone, ("creator",))),
        "creator_id": first(user, ("id", "user_id")) or first(tone, ("creator_id", "user_id")),
        "downloads": _to_int(first(tone, ("downloads_count", "downloads"), 0)),
        "favorites": _to_int(first(tone, ("favorites_count", "favorites"), 0)),
        "created_at": text(tone.get("created_at")),
    }


def build_candidate_row(tone_norm: dict, model: dict) -> dict:
    """Combine tone-level fields with a single model into a candidate row."""
    model_name = text(first(model, ("name", "model_name")))
    make = derive_make(tone_norm.get("makes"), tone_norm["tone_title"], model_name)
    arch = map_architecture(first(model, ("architecture_version", "architecture")))
    tags = tone_norm["tags"]
    return {
        "tone_id": tone_norm["tone_id"],
        "model_id": first(model, ("id", "model_id")),
        "tone_title": tone_norm["tone_title"],
        "model_name": model_name,
        "make": make,
        "model": normalize_model(model_name, tone_norm["tone_title"], make),
        "tags": tags,
        "gear_type": tone_norm["gear_type"],
        "capture_type": "di" if is_amp_gear(tone_norm["gear_type"]) else _norm(tone_norm["gear_type"]),
        "architecture": arch,
        "sample_rate": _to_int(first(model, ("sample_rate",), None)),
        "license": tone_norm["license"],
        "creator": tone_norm["creator"],
        "creator_id": tone_norm["creator_id"],
        "downloads": tone_norm["downloads"],
        "favorites": tone_norm["favorites"],
        "created_at": tone_norm["created_at"],
        "model_url": text(first(model, ("model_url", "url"))),
        "tone_url": tone_norm["tone_url"],
        "gain_bucket": classify_gain(tags, tone_norm["tone_title"], model_name),
        "raw_json": json.dumps({"tone": _strip_models(tone_norm), "model": model},
                               default=str, ensure_ascii=False),
    }


def candidate_is_amp_nam(row: dict) -> bool:
    """Discovery-stage inclusion test (spec §5.2 exclusions)."""
    if not is_amp_gear(row["gear_type"]):
        return False
    if row["architecture"] not in ("A1", "A2"):
        return False
    if should_exclude(row["tone_title"], row["model_name"], " ".join(row["tags"]),
                      row["gear_type"]):
        return False
    return True


def _strip_models(tone_norm: dict) -> dict:
    # raw_json keeps the tone-level snapshot without the models array (which is
    # stored per-row via the model object) to avoid bloating every row.
    return {k: v for k, v in tone_norm.items() if k != "makes_models"}


def _to_int(value: Any, default: Any = None) -> Any:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
