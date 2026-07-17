"""Stage: `openamp migrate-a2` — re-point the finalized devices at their A2 captures.

Rewrites the final manifest in place: each device is paired with the A2 model of
the same name under the same tone, which is downloaded and re-validated. Runs on
the final manifest rather than through select/finalize so that ``device_id`` is
never reassigned -- the rendered audio on disk is addressed by ``device_id``, and
re-deriving those ids would orphan it.

Pairing is by *exact* model name. TONE3000 did not retrain every model on
multi-model tones, so a device whose name has no A2 counterpart keeps its A1
capture and is flagged ``architecture_fallback``. Nothing is paired by
similarity: the models on one tone are the same amp at different gain settings,
so a near-miss would silently swap a device's identity, which is the one property
the whole corpus is built on.

The renders themselves are NOT invalidated. They were produced from the A1
captures and ``renders.parquet`` records the ``nam_sha256`` they came from, so the
provenance stays truthful; only future renders pick up A2.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from openamp.core import manifest
from openamp.core.config import Config
from openamp.core.manifest import STATUS_VALIDATED
from openamp.core.util import sha256_file, utc_now
from openamp.dsp import nam as nam_backend
from . import normalize, validate as validate_mod
from .client import T3KClient

log = logging.getLogger("openamp.migrate")

FLUSH_EVERY = 20


def _a2_by_name(client: T3KClient, tone_id) -> dict[str, dict]:
    """The tone's A2 models, keyed by exact model name."""
    try:
        models = client.list_models(tone_id, architecture=2)
    except Exception as exc:
        log.warning("list_models(%s, architecture=2) failed: %s", tone_id, exc)
        return {}
    out: dict[str, dict] = {}
    for model in models:
        name = normalize.text(normalize.first(model, ("name", "model_name")))
        if name:
            out[name] = model
    return out


def _swap_raw_json(raw_json, a2_model: dict) -> str:
    """Keep the tone half of raw_json; replace the model half with the A2 record."""
    try:
        blob = json.loads(raw_json) if isinstance(raw_json, str) and raw_json else {}
    except ValueError:
        blob = {}
    blob["model"] = a2_model
    return json.dumps(blob, default=str, ensure_ascii=False)


def migrate_one(client: T3KClient, row: dict, settings: Config,
                render_fn=None, cache: dict | None = None,
                claimed: set | None = None) -> dict | None:
    """Download + validate the A2 twin of one device. None if it has to stay A1."""
    if str(row.get("architecture") or "") == "A2":
        return None

    tone_id = row["tone_id"]
    if cache is None:
        cache = {}
    if tone_id not in cache:
        cache[tone_id] = _a2_by_name(client, tone_id)

    name = normalize.text(row.get("model_name"))
    a2 = cache[tone_id].get(name)
    if a2 is None:
        log.info("device %s: no A2 model named %r on tone %s; keeping A1",
                 row.get("device_id"), name, row["tone_id"])
        return {"architecture_fallback": True}

    model_id = normalize.first(a2, ("id", "model_id"))
    model_url = normalize.text(normalize.first(a2, ("model_url", "url")))
    if model_id is None or not model_url:
        return {"architecture_fallback": True}

    # Two devices under one tone can carry the same model name. Handing both the
    # same A2 capture would collapse two device_ids onto one device, so the second
    # claimant keeps its (distinct) A1 capture.
    if claimed is not None and model_id in claimed:
        log.warning("device %s: A2 model %s already claimed by another device; keeping A1",
                    row.get("device_id"), model_id)
        return {"architecture_fallback": True}

    dest = settings.captures_dir / str(row["tone_id"]) / f"{model_id}.nam"
    try:
        sha, nbytes = client.download_model(model_url, dest)
    except Exception as exc:
        log.warning("device %s: A2 download failed (%s); keeping A1",
                    row.get("device_id"), exc)
        return {"architecture_fallback": True}

    # Re-validate: a swapped file must clear the same load/render probe as any
    # freshly downloaded capture, and it needs its own probe .npy for dedup.
    candidate = dict(row)
    candidate.update({"file_path": str(dest), "sample_rate": a2.get("sample_rate")})
    fields = validate_mod.validate_one(
        candidate, settings, render_fn or nam_backend.default_load_and_render)
    if fields.get("status") != STATUS_VALIDATED:
        log.warning("device %s: A2 capture failed validation (%s); keeping A1",
                    row.get("device_id"), fields.get("reject_reason"))
        dest.unlink(missing_ok=True)
        return {"architecture_fallback": True}

    return {
        "model_id": model_id,
        "model_url": model_url,
        "architecture": "A2",
        "architecture_fallback": False,
        "file_path": str(dest),
        "sha256": sha or sha256_file(dest),
        "file_bytes": nbytes,
        "downloaded_at": utc_now(),
        "raw_json": _swap_raw_json(row.get("raw_json"), a2),
        **fields,
    }


def migrate(client: T3KClient, df: pd.DataFrame, settings: Config, *,
            render_fn=None, limit: int | None = None,
            progress: bool = True) -> pd.DataFrame:
    """Swap every device in the final manifest to its A2 capture, in place."""
    settings.ensure_dirs()
    df = df.reset_index(drop=True)

    todo = [i for i in df.index if str(df.at[i, "architecture"]) != "A2"]
    if limit is not None:
        todo = todo[:limit]
    log.info("%d devices to migrate to A2 (%d already A2)",
             len(todo), len(df) - len(todo))

    # One-time snapshot of the A1 manifest: the renders on disk were produced from
    # these rows, so the mapping stays recoverable after the swap.
    backup = settings.manifests_dir / "manifest.a1.parquet"
    if not backup.exists():
        manifest.write_manifest(df, backup)
        log.info("A1 manifest backed up -> %s", backup)

    # The 844 devices sit on ~305 tones; list the A2 models once per tone, not
    # once per device (the API is rate-limited to 80 req/min).
    cache: dict = {}
    claimed: set = set(df.loc[df["architecture"] == "A2", "model_id"].dropna())
    swapped = kept = 0
    for count, i in enumerate(todo, start=1):
        fields = migrate_one(client, df.loc[i].to_dict(), settings, render_fn, cache, claimed)
        if fields is None:
            continue
        if fields.get("model_id") is not None:
            claimed.add(fields["model_id"])
        for key, value in fields.items():
            if key not in df.columns:
                df[key] = pd.NA
            df.at[i, key] = value
        if fields.get("architecture") == "A2":
            swapped += 1
        else:
            kept += 1
        if count % FLUSH_EVERY == 0:
            manifest.write_manifest(df, settings.manifest_path)
            if progress:
                log.info("migrated %d/%d (%d swapped, %d kept on A1)",
                         count, len(todo), swapped, kept)

    manifest.write_manifest(df, settings.manifest_path)
    log.info("migrate complete: %d swapped to A2, %d kept on A1 (no A2 twin)",
             swapped, kept)
    return df


def summarize(df: pd.DataFrame) -> str:
    arch = df["architecture"].value_counts().to_dict()
    fallback = int(df["architecture_fallback"].eq(True).sum())
    return (f"devices: {len(df)} | by architecture: {arch} | "
            f"kept on A1 (no A2 twin): {fallback}")
