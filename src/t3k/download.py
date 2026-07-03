"""Stage: `t3k download` — resumable, checksummed model downloads (spec §5.4).

Downloads each selected capture's ``.nam`` to
``data/captures/{tone_id}/{model_id}.nam`` with Bearer auth. Skips files already
present with a matching recorded hash. On a permanent A2 failure it falls back to
the same tone's A1 model, flagging ``architecture_fallback``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import manifest, normalize
from .client import T3KClient
from .config import Settings
from .manifest import (
    STATUS_DOWNLOAD_FAILED,
    STATUS_DOWNLOADED,
    STATUS_SELECTED,
)

log = logging.getLogger("t3k.download")

MAX_RETRIES = 5
FLUSH_EVERY = 20


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dest_path(settings: Settings, tone_id, model_id) -> Path:
    return settings.captures_dir / str(tone_id) / f"{model_id}.nam"


def _sha256_file(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    n = 0
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            sha.update(chunk)
            n += len(chunk)
    return sha.hexdigest(), n


def _download_with_retries(client: T3KClient, model_url: str, dest: Path,
                           max_retries: int = MAX_RETRIES,
                           sleep_fn=time.sleep) -> tuple[str, int]:
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.download_model(model_url, dest)
        except Exception as exc:  # includes APIError; client already backs off on 5xx/429
            last = exc
            log.warning("download attempt %d/%d failed for %s: %s",
                        attempt, max_retries, dest.name, exc)
            if attempt < max_retries:
                sleep_fn(min(2 ** attempt, 30))
    assert last is not None
    raise last


def _try_a1_fallback(client: T3KClient, row: dict, settings: Settings,
                     sleep_fn=time.sleep) -> dict | None:
    """Attempt the tone's A1 sibling as a substitute for a failed A2 model."""
    if row.get("architecture") != "A2":
        return None
    try:
        models = client.list_models(row["tone_id"], architecture=1)
    except Exception as exc:
        log.warning("A1 fallback list_models(%s) failed: %s", row["tone_id"], exc)
        return None
    for model in models:
        if normalize.map_architecture(
                normalize.first(model, ("architecture_version", "architecture"))) != "A1":
            continue
        model_url = normalize.text(normalize.first(model, ("model_url", "url")))
        model_id = normalize.first(model, ("id", "model_id"))
        if not model_url or model_id is None:
            continue
        dest = _dest_path(settings, row["tone_id"], model_id)
        try:
            sha, nbytes = _download_with_retries(client, model_url, dest, sleep_fn=sleep_fn)
        except Exception:
            continue
        return {
            "model_id": model_id,
            "model_name": normalize.text(normalize.first(model, ("name", "model_name"))),
            "model_url": model_url,
            "architecture": "A1",
            "architecture_fallback": True,
            "file_path": str(dest),
            "sha256": sha,
            "file_bytes": nbytes,
            "downloaded_at": _now(),
            "status": STATUS_DOWNLOADED,
        }
    return None


def download(client: T3KClient, df: pd.DataFrame, settings: Settings, *,
             progress: bool = True, sleep_fn=time.sleep) -> pd.DataFrame:
    settings.ensure_dirs()
    df = manifest.ensure_schema(df).reset_index(drop=True)

    todo = df.index[df["status"].isin([STATUS_SELECTED, STATUS_DOWNLOAD_FAILED])].tolist()
    log.info("%d captures to download", len(todo))
    processed = 0

    for i in todo:
        row = df.loc[i].to_dict()
        dest = _dest_path(settings, row["tone_id"], row["model_id"])

        # Resumable: a present file with a matching recorded hash is already done.
        if dest.is_file() and dest.stat().st_size > 0:
            sha, nbytes = _sha256_file(dest)
            recorded = row.get("sha256")
            if not isinstance(recorded, str) or not recorded or recorded == sha:
                _set(df, i, status=STATUS_DOWNLOADED, file_path=str(dest),
                     sha256=sha, file_bytes=nbytes,
                     downloaded_at=row.get("downloaded_at") or _now())
                continue

        model_url = str(row.get("model_url") or "").strip()
        if not model_url:
            _set(df, i, status=STATUS_DOWNLOAD_FAILED, reject_reason="missing model_url")
            continue

        try:
            sha, nbytes = _download_with_retries(client, model_url, dest, sleep_fn=sleep_fn)
            _set(df, i, status=STATUS_DOWNLOADED, file_path=str(dest),
                 sha256=sha, file_bytes=nbytes, downloaded_at=_now(),
                 architecture_fallback=bool(row.get("architecture_fallback") or False))
        except Exception as exc:
            fallback = _try_a1_fallback(client, row, settings, sleep_fn=sleep_fn)
            if fallback is not None:
                log.info("A2->A1 fallback succeeded for tone %s", row["tone_id"])
                _set(df, i, **fallback)
            else:
                _set(df, i, status=STATUS_DOWNLOAD_FAILED, reject_reason=str(exc)[:300])

        processed += 1
        if processed % FLUSH_EVERY == 0:
            manifest.write_manifest(df, settings.manifest_path)
            if progress:
                log.info("downloaded %d/%d", processed, len(todo))

    manifest.write_manifest(df, settings.manifest_path)
    n_ok = int((df["status"] == STATUS_DOWNLOADED).sum())
    log.info("download complete: %d downloaded, %d failed",
             n_ok, int((df["status"] == STATUS_DOWNLOAD_FAILED).sum()))
    return df


def _set(df: pd.DataFrame, idx, **fields) -> None:
    for key, value in fields.items():
        if key not in df.columns:
            df[key] = pd.NA
        df.at[idx, key] = value
