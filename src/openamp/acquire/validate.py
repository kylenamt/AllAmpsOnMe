"""Stage: `openamp validate` — load each capture and run the render probe (spec §5.5).

For every downloaded file: parse the JSON, load it with neural-amp-modeler, push
the fixed 5 s probe through it on CPU, and assert the output is finite, audible,
non-exploding, and quiet where the input is silent. Stores the probe output as
``{model_id}.probe.npy`` for the dedup stage. (A2 captures that fail to load
already fall back to their A1 sibling at download time, in ``download.py``.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from openamp.core import constants as C
from openamp.core import manifest
from openamp.dsp import nam as nam_backend
from openamp.dsp import audio as probe_mod
from openamp.core.config import Config
from openamp.core.manifest import STATUS_DOWNLOADED, STATUS_REJECTED, STATUS_VALIDATED
from openamp.core.util import utc_now
from openamp.dsp.audio import (
    EXPLODE_MAX_DBFS,
    PROBE_NOT_SILENT_MIN_DBFS,
    SILENCE_OUT_MAX_DBFS,
    peak_dbfs,
    rms_dbfs,
)

log = logging.getLogger("openamp.validate")

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
    if ns.size and rms_dbfs(out[ns]) <= PROBE_NOT_SILENT_MIN_DBFS:
        return "silent_output"

    sil_start = min(probe.silence.start, n)
    sil_stop = min(probe.silence.stop, n)
    sil = out[sil_start:sil_stop]
    if sil.size and rms_dbfs(sil) >= SILENCE_OUT_MAX_DBFS:
        return "noise_in_silence"

    return None


def validate_one(row: dict, settings: Config, render_fn: RenderFn) -> dict:
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
                   or C.SAMPLE_RATE)
    probe = probe_mod.generate_probe(sample_rate)

    try:
        out = render_fn(file_path, probe.signal, sample_rate)
    except Exception as exc:
        log.warning("render failed for %s: %s", Path(file_path).name, exc)
        return {"status": STATUS_REJECTED, "reject_reason": "load_or_render_failed"}

    reason = _check_output(out, probe)
    if reason is not None:
        return {"status": STATUS_REJECTED, "reject_reason": reason}

    probe_path = _probe_path(file_path)
    np.save(probe_path, np.asarray(out, dtype=np.float32))
    return {
        "status": STATUS_VALIDATED,
        "sample_rate": sample_rate,
        "probe_path": str(probe_path),
        "validated_at": utc_now(),
    }


def validate(df: pd.DataFrame, settings: Config, *,
             render_fn: RenderFn | None = None,
             progress: bool = True) -> pd.DataFrame:
    render_fn = render_fn or nam_backend.default_load_and_render
    df = manifest.ensure_schema(df).reset_index(drop=True)
    todo = df.index[df["status"] == STATUS_DOWNLOADED].tolist()
    log.info("%d captures to validate", len(todo))

    for count, i in enumerate(todo, start=1):
        fields = validate_one(df.loc[i].to_dict(), settings, render_fn)
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
