"""Manifest schema, parquet I/O, and status transitions (spec §5, §6).

A single DataFrame carries a capture from discovery through finalize. Each stage
reads it, mutates ``status`` (+ stage-specific columns), and writes it back
atomically. Re-running a stage is a no-op for rows already past that stage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Status vocabulary (spec §5) ------------------------------------------------
STATUS_CANDIDATE = "candidate"
STATUS_SELECTED = "selected"
STATUS_DOWNLOADED = "downloaded"
STATUS_DOWNLOAD_FAILED = "download_failed"
STATUS_VALIDATED = "validated"
STATUS_REJECTED = "rejected"
STATUS_DUPLICATE = "duplicate"

# Columns produced at discovery (spec §5.2).
DISCOVERY_COLUMNS = [
    "tone_id", "model_id", "tone_title", "model_name", "make", "model",
    "tags", "gear_type", "capture_type", "architecture", "sample_rate",
    "license", "creator", "creator_id", "downloads", "favorites",
    "created_at", "model_url", "tone_url", "raw_json",
]

# Columns added by later stages.
WORKING_COLUMNS = [
    "status", "gain_bucket", "architecture_fallback",
    "file_path", "sha256", "file_bytes", "downloaded_at",
    "probe_path", "validated_at", "reject_reason", "device_id",
]

ALL_COLUMNS = DISCOVERY_COLUMNS + WORKING_COLUMNS

# Final manifest column order (spec §6).
FINAL_COLUMNS = [
    "device_id", "tone_id", "model_id", "tone_title", "model_name",
    "make", "model", "tags", "gain_bucket", "architecture",
    "architecture_fallback", "sample_rate", "file_path", "sha256", "file_bytes",
    "license", "creator", "creator_id", "tone_url",
    "downloads", "favorites", "created_at", "probe_path",
    "downloaded_at", "validated_at", "raw_json",
]


def empty_manifest(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=columns or ALL_COLUMNS)


def ensure_schema(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a copy with all expected columns present (missing filled with NA)."""
    columns = columns or ALL_COLUMNS
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def read_manifest(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet manifest, or return an empty schema-complete frame."""
    path = Path(path)
    if not path.is_file():
        return ensure_schema(empty_manifest(), columns)
    return ensure_schema(pd.read_parquet(path), columns)


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    """Atomically write ``df`` to ``path`` (parquet)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def set_status(df: pd.DataFrame, mask, status: str, **fields) -> pd.DataFrame:
    """Set ``status`` (+ optional extra columns) for rows selected by ``mask``.

    Mutates and returns ``df`` for convenience.
    """
    df.loc[mask, "status"] = status
    for key, value in fields.items():
        if key not in df.columns:
            df[key] = pd.NA
        df.loc[mask, key] = value
    return df


def counts_by_status(df: pd.DataFrame) -> dict[str, int]:
    if "status" not in df.columns or df.empty:
        return {}
    return df["status"].value_counts(dropna=False).to_dict()
