"""Stage: `t3k validate` — load each capture and run the render probe (spec §5.5).

For every downloaded file: parse the JSON, load it with neural-amp-modeler, push
the fixed 5 s probe through it on CPU, and assert the output is finite, audible,
non-exploding, and quiet where the input is silent. On an A2 load failure with a
client available, swaps in the tone's A1 sibling and retries. Stores the probe
output as ``{model_id}.probe.npy`` for the dedup stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import manifest
from . import nam_backend
from . import probe as probe_mod
from .client import T3KClient
from .config import NAM_DEFAULT_SAMPLE_RATE, Settings
from .manifest import STATUS_DOWNLOADED, STATUS_REJECTED, STATUS_VALIDATED
from .probe import (
    EXPLODE_MAX_DBFS,
    NOT_SILENT_MIN_DBFS,
    SILENCE_OUT_MAX_DBFS,
    peak_dbfs,
    rms_dbfs,
)

log = logging.getLogger("t3k.validate")

RenderFn = Callable[[str, np.ndarray, int], np.ndarray]

FLUSH_EVERY = 20


def _probe_path(file_path: str) -> Path:
    return Path(file_path).with_suffix(".probe.npy")


def _coerce_sample_rate(value) -> int | None:
    """Return an int sample rate, or None if missing/NaN/unparseable.

    Guards the `or` fallthrough below: a pandas missing value is a float ``NaN``
    which is *truthy*, so `None or NaN` would yield NaN and crash `int(NaN)`.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _check_output(out: np.ndarray, probe: probe_mod.Probe) -> str | None:
    """Return a reject reason, or None if all checks pass (spec §5.5.4)."""
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    if out.size == 0:
        return "empty_output"
    if not np.all(np.isfinite(out)):
        return "non_finite_output"

    n = min(out.size, probe.signal.size)
    out = out[:n]

    if peak_dbfs(out) >= EXPLODE_MAX_DBFS:
        return "exploding_output"

    ns = probe.non_silent
    ns = ns[ns < n]
    if ns.size and rms_dbfs(out[ns]) <= NOT_SILENT_MIN_DBFS:
        return "silent_output"

    sil_start = min(probe.silence.start, n)
    sil_stop = min(probe.silence.stop, n)
    sil = out[sil_start:sil_stop]
    if sil.size and rms_dbfs(sil) >= SILENCE_OUT_MAX_DBFS:
        return "noise_in_silence"

    return None


def validate_one(row: dict, settings: Settings, render_fn: RenderFn,
                 client: T3KClient | None = None) -> dict:
    """Validate a single downloaded capture; returns fields to write back."""
    file_path = str(row.get("file_path") or "")
    if not file_path or not Path(file_path).is_file():
        return {"status": STATUS_REJECTED, "reject_reason": "file_missing"}

    try:
        summary = nam_backend.parse_nam(file_path)
    except Exception as exc:
        return {"status": STATUS_REJECTED, "reject_reason": f"parse_error: {exc}"[:300]}

    sample_rate = (_coerce_sample_rate(summary.get("sample_rate"))
                   or _coerce_sample_rate(row.get("sample_rate"))
                   or NAM_DEFAULT_SAMPLE_RATE)
    probe = probe_mod.generate_probe(sample_rate)

    out, used = _render_with_fallback(row, file_path, probe, sample_rate, render_fn, client, settings)
    if out is None:
        return {"status": STATUS_REJECTED, "reject_reason": "load_or_render_failed"}

    reason = _check_output(out, probe)
    if reason is not None:
        return {"status": STATUS_REJECTED, "reject_reason": reason, **used}

    probe_path = _probe_path(used.get("file_path", file_path))
    np.save(probe_path, np.asarray(out, dtype=np.float32))
    return {
        "status": STATUS_VALIDATED,
        "sample_rate": sample_rate,
        "probe_path": str(probe_path),
        "validated_at": _now(),
        **used,
    }


def _render_with_fallback(row, file_path, probe, sample_rate, render_fn, client, settings):
    """Render; on A2 load failure with a client, swap in the A1 sibling."""
    try:
        return render_fn(file_path, probe.signal, sample_rate), {}
    except Exception as exc:
        log.warning("render failed for %s: %s", Path(file_path).name, exc)

    if row.get("architecture") == "A2" and client is not None:
        from .download import _try_a1_fallback  # local import avoids cycle at module load
        fb = _try_a1_fallback(client, row, settings)
        if fb is not None:
            try:
                out = render_fn(fb["file_path"], probe.signal, sample_rate)
                return out, {k: fb[k] for k in ("model_id", "model_name", "model_url",
                                                "architecture", "architecture_fallback",
                                                "file_path", "sha256", "file_bytes")}
            except Exception as exc:
                log.warning("A1 fallback render also failed: %s", exc)
    return None, {}


def validate(df: pd.DataFrame, settings: Settings, *,
             render_fn: RenderFn | None = None, client: T3KClient | None = None,
             progress: bool = True) -> pd.DataFrame:
    render_fn = render_fn or nam_backend.default_load_and_render
    df = manifest.ensure_schema(df).reset_index(drop=True)
    todo = df.index[df["status"] == STATUS_DOWNLOADED].tolist()
    log.info("%d captures to validate", len(todo))

    for count, i in enumerate(todo, start=1):
        fields = validate_one(df.loc[i].to_dict(), settings, render_fn, client)
        for key, value in fields.items():
            if key not in df.columns:
                df[key] = pd.NA
            df.at[i, key] = value
        if count % FLUSH_EVERY == 0:
            manifest.write_manifest(df, settings.manifest_path)
            if progress:
                log.info("validated %d/%d", count, len(todo))

    manifest.write_manifest(df, settings.manifest_path)
    log.info("validate complete: %d validated, %d rejected",
             int((df["status"] == STATUS_VALIDATED).sum()),
             int((df["status"] == STATUS_REJECTED).sum()))
    return df


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
