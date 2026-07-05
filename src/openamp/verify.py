"""Stage: ``oa3k render verify`` — completeness + sanity + QA + report (spec §5).

Checks the render manifest and files, then writes the **final device list**
``devices_final.parquet`` (devices that fail verification are excluded). Signal
checks are pure functions (unit-tested with numpy); the wrapper reads audio and
emits QA WAV/spectrogram exports + ``results/phase2_report.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from t3k.probe import esr, rms_dbfs

from . import audio_io
from . import constants as C
from . import manifests
from .config import Phase2Config

log = logging.getLogger("oa3k.verify")

QA_SECONDS = 10.0
N_QA_DEVICES = 8
N_ALIGN_CHECKS = 5


# --- Pure check helpers (unit-tested) ------------------------------------------
def is_pass_through(inp: np.ndarray, out: np.ndarray,
                    min_esr: float = C.PASS_THROUGH_MIN_ESR) -> bool:
    """True if ``out`` is ~identical to ``inp`` (a broken/pass-through model,
    spec §5.2): ``ESR(out, inp) <= min_esr``."""
    return esr(out, inp) <= min_esr


def is_silent(out: np.ndarray, min_rms_db: float = C.NOT_SILENT_MIN_DBFS) -> bool:
    return rms_dbfs(out) <= min_rms_db


def has_non_finite(out: np.ndarray) -> bool:
    return not bool(np.all(np.isfinite(np.asarray(out))))


def signal_flags(inp: np.ndarray, out: np.ndarray) -> dict:
    """All per-signal sanity flags in one place (spec §5.2)."""
    return {
        "non_finite": has_non_finite(out),
        "silent": is_silent(out),
        "pass_through": is_pass_through(inp, out),
    }


def aligned(clean_num_samples: int, render_num_samples: int) -> bool:
    """NAM forward is same-length; any mismatch is a bug (spec §5.3)."""
    return int(clean_num_samples) == int(render_num_samples)


def missing_pairs(renders_df: pd.DataFrame, device_ids, file_ids) -> list[tuple]:
    """(device_id, file_id) pairs expected but absent/not ok (spec §5.1)."""
    ok = set()
    if not renders_df.empty:
        good = renders_df[renders_df["status"] == C.RENDER_OK]
        ok = {(int(d), str(f)) for d, f in zip(good["device_id"], good["file_id"])}
    missing = []
    for d in device_ids:
        for f in file_ids:
            if (int(d), str(f)) not in ok:
                missing.append((int(d), str(f)))
    return missing


# --- Verification wrapper ------------------------------------------------------
def verify(config: Phase2Config, *, seed: int | None = None) -> pd.DataFrame:
    """Run all checks, emit QA + report, and write ``devices_final.parquet``."""
    config.ensure_dirs()
    rng = np.random.default_rng(config.seed if seed is None else seed)

    renders = manifests.read(config.renders_manifest_path, manifests.RENDERS_COLUMNS)
    corpus = manifests.read(config.corpus_manifest_path, manifests.CORPUS_COLUMNS)
    manifest = manifests.read(config.phase1_manifest_path,
                              ["device_id", "tone_id", "model_id", "make", "model",
                               "gain_bucket", "architecture", "sample_rate",
                               "file_path", "license", "creator"])
    if renders.empty:
        raise RuntimeError("No renders found. Run `oa3k render run` first.")

    egdb_file_ids = corpus.loc[corpus["source"] == C.SOURCE_EGDB, "file_id"].astype(str).tolist()
    all_file_ids = corpus["file_id"].astype(str).tolist()
    device_ids = sorted(int(d) for d in renders["device_id"].unique())

    per_device: list[dict] = []
    for device_id in device_ids:
        per_device.append(_verify_device(config, device_id, renders, corpus,
                                          egdb_file_ids, all_file_ids, rng))

    stats = pd.DataFrame(per_device)
    finals = _build_devices_final(stats, manifest)
    manifests.write(finals, config.devices_final_path)

    _export_qa(config, stats, corpus, rng)
    _write_report(config, stats, corpus, finals)

    n_final = int((stats["status"] == C.RENDER_OK).sum())
    log.info("verify: %d/%d devices passed -> %s", n_final, len(stats),
             config.devices_final_path)
    if n_final < 400:
        log.warning("final device count %d is below the 400 target (spec §5).", n_final)
    return finals


def _verify_device(config, device_id, renders, corpus, egdb_file_ids, all_file_ids, rng) -> dict:
    sub = renders[renders["device_id"] == device_id]
    result = {"device_id": device_id, "status": C.RENDER_OK, "reason": "",
              "n_files": 0, "out_peak_db": np.nan, "output_scale": np.nan}

    # Completeness for this device.
    miss = missing_pairs(sub, [device_id], all_file_ids)
    if miss or (sub["status"] == C.RENDER_FAILED).any():
        result.update(status=C.RENDER_FAILED, reason="incomplete_or_failed")
        return result

    ok = sub[sub["status"] == C.RENDER_OK]
    result["n_files"] = int(len(ok))
    result["out_peak_db"] = float(ok["out_peak_db"].max()) if len(ok) else np.nan
    result["output_scale"] = float(ok["output_scale"].iloc[0]) if len(ok) else np.nan

    # Signal sanity on the EGDB files (skip the synthetic sweep for silence/ESR).
    check_ids = egdb_file_ids or all_file_ids
    for fid in check_ids:
        clean = _read_clean(config, corpus, fid)
        render = _read_render(config, device_id, fid)
        if clean is None or render is None:
            result.update(status=C.RENDER_FAILED, reason=f"missing_file:{fid}")
            return result
        if not aligned(len(clean), len(render)):
            result.update(status=C.RENDER_FAILED, reason=f"misaligned:{fid}")
            return result
        flags = signal_flags(clean, render)
        for name, bad in flags.items():
            if bad:
                result.update(status=C.RENDER_FAILED, reason=f"{name}:{fid}")
                return result
    return result


def _build_devices_final(stats: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    good = stats[stats["status"] == C.RENDER_OK].copy()
    merged = manifest.merge(good, on="device_id", how="inner")
    cols = [c for c in manifests.DEVICES_FINAL_COLUMNS if c in merged.columns]
    return merged[cols].sort_values("device_id").reset_index(drop=True)


def _export_qa(config, stats, corpus, rng) -> None:
    """Export clean/render WAV pairs + spectrogram PNGs for a few devices (spec §5.4)."""
    good = stats[stats["status"] == C.RENDER_OK]["device_id"].tolist()
    egdb = corpus.loc[corpus["source"] == C.SOURCE_EGDB, "file_id"].astype(str).tolist()
    if not good or not egdb:
        return
    pick = rng.choice(good, size=min(N_QA_DEVICES, len(good)), replace=False)
    n = int(round(QA_SECONDS * config.sample_rate))
    for device_id in pick:
        fid = str(rng.choice(egdb))
        clean = _read_clean(config, corpus, fid)
        render = _read_render(config, int(device_id), fid)
        if clean is None or render is None:
            continue
        clean, render = clean[:n], render[:n]
        stem = config.qa_dir / f"dev{int(device_id):04d}_{fid}"
        audio_io.write_wav(stem.with_name(stem.name + "_clean.wav"), clean, config.sample_rate)
        audio_io.write_wav(stem.with_name(stem.name + "_render.wav"), render, config.sample_rate)
        _spectrogram_png(stem.with_suffix(".png"), clean, render, config.sample_rate)


def _spectrogram_png(path: Path, clean: np.ndarray, render: np.ndarray, sr: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dep
        log.warning("skip spectrogram (%s): %s", path.name, exc)
        return
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, sig, title in ((axes[0], clean, "clean"), (axes[1], render, "render")):
        ax.specgram(sig, NFFT=1024, Fs=sr, noverlap=512)
        ax.set_ylabel(f"{title}\nHz")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _write_report(config, stats, corpus, finals) -> None:
    lines = ["# Phase 2 render report", ""]
    n_ok = int((stats["status"] == C.RENDER_OK).sum())
    lines += [
        f"- Devices verified: **{len(stats)}**",
        f"- Passed: **{n_ok}**  |  Failed: **{len(stats) - n_ok}**",
        f"- Final device list: **{len(finals)}** -> `{config.devices_final_path.name}`",
        "",
        "## Split durations (clean corpus)",
        "",
    ]
    for split in C.SPLITS:
        dur = float(corpus.loc[corpus["split"] == split, "duration_s"].sum())
        lines.append(f"- {split}: {dur / 60.0:.1f} min")
    lines += ["", "## Per-device", "",
              "| device_id | status | n_files | out_peak_db | output_scale | reason |",
              "|---|---|---|---|---|---|"]
    for r in stats.itertuples(index=False):
        lines.append(f"| {int(r.device_id):04d} | {r.status} | {r.n_files} | "
                     f"{r.out_peak_db:.2f} | {r.output_scale:.4f} | {r.reason} |")
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("report -> %s", config.report_path)


def _read_clean(config, corpus, file_id):
    row = corpus[corpus["file_id"].astype(str) == str(file_id)]
    if row.empty:
        return None
    split = row.iloc[0]["split"]
    path = config.clean_split_dir(split) / f"{file_id}.{config.output_format}"
    if not path.is_file():
        return None
    sig, _ = audio_io.read_audio(path)
    return sig


def _read_render(config, device_id, file_id):
    path = config.device_render_dir(device_id) / f"{file_id}.{config.output_format}"
    if not path.is_file():
        return None
    sig, _ = audio_io.read_audio(path)
    return sig
