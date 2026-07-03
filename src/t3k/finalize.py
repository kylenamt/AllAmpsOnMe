"""Stage: `t3k finalize` — top up to quota, assign device_ids, emit outputs (spec §5.7).

If the validated, non-duplicate count is below the target, optionally loops back
to select/download/validate/dedup replacement candidates. Then assigns the stable
``device_id`` (zero-padded rank by make, model, model_id — the embedding-table
index that must never change) and writes ``manifest.parquet`` (final) plus
``rejected.parquet``.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from . import manifest, select
from .config import Settings
from .manifest import (
    ALL_COLUMNS,
    FINAL_COLUMNS,
    STATUS_SELECTED,
    STATUS_VALIDATED,
)

log = logging.getLogger("t3k.finalize")


def _s(value) -> str:
    """NA/None-safe string coercion (pd.NA is ambiguous in boolean contexts)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _is_complete(row: pd.Series) -> bool:
    """Acceptance §7.4: non-empty license, creator, and tone_url."""
    def ok(v) -> bool:
        s = _s(v).lower()
        return bool(s) and s != "unknown"
    return ok(row.get("license")) and ok(row.get("creator")) and ok(row.get("tone_url"))


def _final_pool(df: pd.DataFrame) -> pd.Index:
    validated = df[df["status"] == STATUS_VALIDATED]
    complete = validated[validated.apply(_is_complete, axis=1)] if not validated.empty else validated
    return complete.index


def assign_device_ids(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Assign stable, zero-padded device_ids over ``index`` (spec §5.7)."""
    sub = df.loc[index].copy()
    sub["_mk"] = sub["make"].astype(str)
    sub["_md"] = sub["model"].astype(str)
    sub["_id"] = sub["model_id"].astype(str)
    order = sub.sort_values(["_mk", "_md", "_id"]).index
    width = max(4, len(str(max(len(order) - 1, 0))))
    for rank, i in enumerate(order):
        df.at[i, "device_id"] = int(rank)
        df.at[i, "device_id_str"] = str(rank).zfill(width)
    return df


def finalize(df: pd.DataFrame, settings: Settings, *,
             candidates: pd.DataFrame | None = None,
             top_up_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
             max_rounds: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = manifest.ensure_schema(df).reset_index(drop=True)
    target = settings.final_target

    # --- Optional top-up loop (spec §5.7) -------------------------------------
    rounds = 0
    while len(_final_pool(df)) < target and top_up_fn is not None and candidates is not None \
            and rounds < max_rounds:
        need = target - len(_final_pool(df))
        used_ids = set(df["model_id"].dropna().tolist())
        extra = select.select(candidates, settings,
                              target=int(need * 1.3) + 1, exclude_ids=used_ids)
        if extra.empty:
            log.info("no more replacement candidates available")
            break
        extra["status"] = STATUS_SELECTED
        df = manifest.ensure_schema(pd.concat([df, extra], ignore_index=True))
        df = top_up_fn(df)  # runs download -> validate -> dedup on the new rows
        rounds += 1

    # --- Assign device_ids over the accepted set ------------------------------
    final_index = _final_pool(df)
    df = assign_device_ids(df, final_index)

    final_df = df.loc[final_index].copy()
    final_df = _order_final(final_df)

    rejected_df = df.loc[~df.index.isin(final_index)].copy()
    rejected_df = _fill_reject_reasons(rejected_df)

    manifest.write_manifest(final_df, settings.manifest_path)
    manifest.write_manifest(rejected_df, settings.rejected_path)

    log.info("finalize complete: %d accepted, %d rejected", len(final_df), len(rejected_df))
    return final_df, rejected_df


def _order_final(final_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FINAL_COLUMNS if c in final_df.columns]
    final_df = final_df.sort_values("device_id")[cols + [c for c in final_df.columns if c not in cols]]
    return final_df.reset_index(drop=True)


def _fill_reject_reasons(rejected: pd.DataFrame) -> pd.DataFrame:
    if rejected.empty:
        return rejected
    def reason(row):
        existing = _s(row.get("reject_reason"))
        if existing:
            return existing
        if row.get("status") == STATUS_VALIDATED:
            return "incomplete_license_or_attribution"
        return _s(row.get("status")) or "not_selected"
    rejected = rejected.copy()
    rejected["reject_reason"] = rejected.apply(reason, axis=1)
    return rejected


def summarize(final_df: pd.DataFrame) -> str:
    if final_df.empty:
        return "No captures in final manifest."
    lines = [f"Final manifest: {len(final_df)} captures, "
             f"{final_df['make'].nunique()} makes."]
    lines.append("\nGain buckets:")
    total = len(final_df)
    for bucket, n in final_df["gain_bucket"].value_counts().items():
        lines.append(f"  {bucket:12s} {n:4d}  ({100*n/total:4.1f}%)")
    lines.append("\nLicense breakdown:")
    for lic, n in final_df["license"].value_counts().items():
        lines.append(f"  {str(lic)[:40]:40s} {n:4d}")
    return "\n".join(lines)
