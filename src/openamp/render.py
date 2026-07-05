"""Stage: ``oa3k render run`` — render the clean corpus through every device (spec §4).

Rendering is **whole-file, then sliced** — never per-clip — so there are no
clip-boundary transients. The core is :func:`chunked_forward`, a pure driver that
streams a file through a causal model in chunks while carrying R samples of
left-context and discarding the warmup region, giving output **bit-identical to a
single full-file forward pass**. That function is unit-tested against a numpy FIR
stand-in; the surrounding :func:`run` job (GPU load, disk checks, manifest
bookkeeping, resume) uses the real NAM adapter in :mod:`data.nam_render`.

Model contract for :func:`chunked_forward`: ``model_fn`` is **causal with receptive
field ≤ R** and returns a **right-aligned** mono float32 array (its last output
sample corresponds to its last input sample). The driver keeps the last
``end-pos`` samples of each chunk's output, so both same-length nets (output length
== input length) and "valid" nets (output shorter by the warmup) work unchanged.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from t3k.probe import peak_dbfs, rms_dbfs

from . import audio_io
from . import constants as C
from . import manifests
from .config import Phase2Config

log = logging.getLogger("oa3k.render")

ModelFn = Callable[[np.ndarray], np.ndarray]

# Rough on-disk size of one second of 24-bit/48 kHz mono FLAC (~0.6 compression of
# 144 kB/s PCM). Used only for the up-front free-space guard (spec §4.4).
FLAC_BYTES_PER_SECOND = 90_000

# Threads for the CPU/disk-bound FLAC encode+hash pass. FLAC encode is the
# per-device cost that rivals the GPU forward, and libsndfile releases the GIL, so
# fanning it out overlaps the codec with disk and keeps the GPU from starving.
DEFAULT_IO_WORKERS = min(8, os.cpu_count() or 4)

_SENTINEL = object()


# --- Pure, unit-tested core ----------------------------------------------------
def chunked_forward(model_fn: ModelFn, x: np.ndarray, receptive_field: int,
                    chunk_samples: int) -> np.ndarray:
    """Render ``x`` through ``model_fn`` in chunks with left-context (spec §4.1.3).

    Bit-identical to ``model_fn(x)`` for a same-length causal model with the given
    receptive field. See the module docstring for the model contract.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    length = x.size
    if length == 0:
        return x.copy()
    R = max(int(receptive_field), 0)
    chunk = max(int(chunk_samples), 1)

    # Prepend R zeros so chunk 0 sees the same implicit zero left-context a
    # full-file causal forward would; xp[R + i] == x[i].
    xp = np.concatenate([np.zeros(R, dtype=np.float32), x])
    out = np.empty(length, dtype=np.float32)

    pos = 0
    while pos < length:
        end = min(pos + chunk, length)
        seg = xp[pos:end + R]                       # [pos-R … end) of x, zero-padded
        y = np.asarray(model_fn(seg), dtype=np.float32).reshape(-1)
        take = end - pos
        if y.size < take:
            raise ValueError(
                f"model_fn returned {y.size} samples for a {seg.size}-sample input; "
                f"need at least {take} (see chunked_forward contract)."
            )
        out[pos:end] = y[-take:]                    # keep the fully-conditioned tail
        pos = end
    return out


def output_scale(peak: float, ceiling: float = C.OUTPUT_PEAK_CEILING) -> float:
    """Per-device scalar so the global peak sits at ``ceiling`` (spec §4.2)."""
    peak = float(peak)
    if peak > ceiling:
        return ceiling / peak
    return 1.0


def _read_devices_file(path: str) -> str:
    """Read a ``@file`` device list into the comma/range syntax (``#`` = comment)."""
    tokens: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tokens.append(line)
    return ",".join(tokens)


def parse_device_selection(arg: str | None, all_ids: Sequence[int]) -> list[int]:
    """Parse ``--devices`` ("0-9", "0,2,5", "3", or "@file") against available ids."""
    if arg is None or arg.strip() == "":
        return list(all_ids)
    arg = arg.strip()
    if arg.startswith("@"):
        arg = _read_devices_file(arg[1:])
    available = set(int(i) for i in all_ids)
    picked: set[int] = set()
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            picked.update(range(int(lo), int(hi) + 1))
        else:
            picked.add(int(part))
    return sorted(picked & available)


def estimate_bytes(total_clean_seconds: float, n_devices: int) -> int:
    """Estimated render footprint for the disk guard (spec §4.4)."""
    return int(total_clean_seconds * n_devices * FLAC_BYTES_PER_SECOND)


def device_is_done(config: Phase2Config, device_id: int, file_ids: Iterable[str],
                   renders_df: pd.DataFrame) -> bool:
    """A device is done when every corpus file has an ``render_ok`` row and its
    render exists on disk (spec §4.3 — resumable/idempotent)."""
    if renders_df.empty:
        return False
    sub = renders_df[(renders_df["device_id"] == device_id)
                     & (renders_df["status"] == C.RENDER_OK)]
    have = set(sub["file_id"].astype(str))
    for fid in file_ids:
        if str(fid) not in have:
            return False
        if not config.device_render_dir(device_id).joinpath(
                f"{fid}.{config.output_format}").is_file():
            return False
    return True


# --- Job orchestration (needs GPU + NAM + data; documented, not run in CI) ------
def run(config: Phase2Config, *, devices: str | None = None, dry_run: bool = False,
        device: str = "cuda", flush_every: int = 10,
        io_workers: int = 0) -> pd.DataFrame:
    """Render every selected device's files. Resumable and idempotent.

    ``io_workers`` sizes the CPU/disk FLAC pass; ``0`` (or negative) auto-picks
    :data:`DEFAULT_IO_WORKERS`."""
    from . import nam_render  # local import keeps torch/NAM out of module load

    if io_workers <= 0:
        io_workers = DEFAULT_IO_WORKERS

    config.ensure_dirs()
    manifest = manifests.read(config.phase1_manifest_path,
                              ["device_id", "file_path", "architecture", "sample_rate"])
    if manifest.empty:
        raise RuntimeError(
            f"Phase 1 manifest not found or empty: {config.phase1_manifest_path}. "
            "Run Phase 1 (`t3k finalize`) first.")
    corpus = manifests.read(config.corpus_manifest_path, manifests.CORPUS_COLUMNS)
    if corpus.empty:
        raise RuntimeError("Empty corpus. Run `oa3k corpus prepare` first.")

    manifest = manifest.sort_values("device_id").reset_index(drop=True)
    all_ids = [int(d) for d in manifest["device_id"].tolist()]
    selected = parse_device_selection(devices, all_ids)

    clean_files = _load_clean_files(config, corpus)
    total_clean_seconds = float(corpus["duration_s"].sum())

    _check_disk(config, total_clean_seconds, len(selected))

    renders = manifests.read(config.renders_manifest_path, manifests.RENDERS_COLUMNS)
    rows: list[dict] = renders.to_dict("records") if not renders.empty else []

    for count, device_id in enumerate(selected, start=1):
        row = manifest[manifest["device_id"] == device_id].iloc[0]
        file_ids = [f["file_id"] for f in clean_files]
        if device_is_done(config, device_id, file_ids, pd.DataFrame(rows)
                          if rows else renders):
            log.info("device %04d already done; skipping", device_id)
            continue
        if dry_run:
            log.info("[dry-run] would render device %04d (%s)", device_id, row["file_path"])
            continue

        try:
            new_rows = _render_device(config, int(device_id), str(row["file_path"]),
                                      clean_files, nam_render, device, io_workers)
        except Exception as exc:  # load error / NaN / etc.
            log.warning("device %04d failed: %s", device_id, exc)
            new_rows = [_failed_row(int(device_id), f, str(exc)) for f in clean_files]

        rows = [r for r in rows if int(r["device_id"]) != int(device_id)]  # replace
        rows.extend(new_rows)

        if count % flush_every == 0:
            manifests.write(pd.DataFrame(rows, columns=manifests.RENDERS_COLUMNS),
                            config.renders_manifest_path)
            log.info("rendered %d/%d devices", count, len(selected))

    out = pd.DataFrame(rows, columns=manifests.RENDERS_COLUMNS)
    manifests.write(out, config.renders_manifest_path)
    n_failed = int((out["status"] == C.RENDER_FAILED).sum()) if not out.empty else 0
    log.info("render complete: %d devices, %d failed rows", len(selected), n_failed)
    return out


def _load_clean_files(config: Phase2Config, corpus: pd.DataFrame) -> list[dict]:
    files = []
    for r in corpus.itertuples(index=False):
        path = config.clean_split_dir(r.split) / f"{r.file_id}.{config.output_format}"
        files.append({"file_id": r.file_id, "split": r.split, "path": path})
    return files


def _prefetch(items: Sequence[dict], load_fn: Callable[[dict], object], *,
              workers: int = 2, ahead: int = 4) -> Iterator[tuple[dict, object]]:
    """Yield ``(item, load_fn(item))`` in order, running ``load_fn`` up to ``ahead``
    items in advance on ``workers`` background threads.

    Used to hide FLAC decode behind the GPU forward: the render loop consumes
    already-decoded audio while the next files decode. Order is preserved and the
    bounded look-ahead caps how much decoded audio is resident at once. A load
    error surfaces at the point that item would have been consumed (same ordering
    as a plain serial loop)."""
    it = iter(items)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        pending: deque[tuple[dict, object]] = deque()
        for _ in range(max(1, ahead)):
            nxt = next(it, _SENTINEL)
            if nxt is _SENTINEL:
                break
            pending.append((nxt, ex.submit(load_fn, nxt)))
        while pending:
            item, fut = pending.popleft()
            nxt = next(it, _SENTINEL)
            if nxt is not _SENTINEL:
                pending.append((nxt, ex.submit(load_fn, nxt)))
            yield item, fut.result()


def _render_device(config: Phase2Config, device_id: int, nam_path: str,
                   clean_files: list[dict], nam_render, device: str,
                   io_workers: int = DEFAULT_IO_WORKERS) -> list[dict]:
    model = nam_render.load_model(nam_path, device=device)
    resampled = model.sample_rate != config.sample_rate

    # First pass: render everything on the GPU, tracking the device's global peak
    # (spec §4.2). File decode is prefetched on worker threads so it overlaps the
    # forward instead of stalling it; the GPU forward itself stays serial and
    # bit-identical (only the surrounding I/O moves off the critical path).
    outputs: dict[str, np.ndarray] = {}
    global_peak = 0.0
    for f, (clean, _sr) in _prefetch(clean_files, lambda f: audio_io.read_audio(f["path"])):
        y = _render_signal(model, clean, config, resampled)
        if not np.all(np.isfinite(y)):
            raise ValueError(f"non-finite output on file {f['file_id']}")
        outputs[f["file_id"]] = y
        global_peak = max(global_peak, float(np.max(np.abs(y))) if y.size else 0.0)

    scale = output_scale(global_peak)

    # Second pass: scale, encode, hash. Pure CPU/disk with no GPU dependency, so
    # fan it across threads — FLAC encode is the per-device cost that rivals the
    # forward. An order-preserving map keeps the emitted rows deterministic, and
    # each file writes an independent path so the outputs stay bit-identical.
    def finalize(f: dict) -> dict:
        y = (outputs[f["file_id"]] * scale).astype(np.float32)
        out_path = config.device_render_dir(device_id) / f"{f['file_id']}.{config.output_format}"
        audio_io.write_flac(out_path, y, config.sample_rate)
        return {
            "device_id": device_id, "file_id": f["file_id"], "split": f["split"],
            "path": str(out_path), "output_scale": round(float(scale), 8),
            "out_rms_db": round(rms_dbfs(y), 3), "out_peak_db": round(peak_dbfs(y), 3),
            "render_sha256": _sha256(out_path), "rendered_at": _now(),
            "nam_sha256": model.nam_sha256, "status": C.RENDER_OK,
            "resampled": bool(resampled), "fail_reason": pd.NA,
        }

    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as ex:
        return list(ex.map(finalize, clean_files))


def _render_signal(model, clean: np.ndarray, config: Phase2Config, resampled: bool) -> np.ndarray:
    if not resampled:
        return chunked_forward(model.forward, clean, model.receptive_field, config.chunk_samples)
    # Rare path (spec §4.1.4): resample in -> model rate -> out.
    x = audio_io.resample(clean, config.sample_rate, model.sample_rate)
    chunk = int(round(config.chunk_seconds * model.sample_rate))
    y = chunked_forward(model.forward, x, model.receptive_field, chunk)
    return audio_io.resample(y, model.sample_rate, config.sample_rate)


def _check_disk(config: Phase2Config, total_clean_seconds: float, n_devices: int) -> None:
    need = estimate_bytes(total_clean_seconds, n_devices) * C.DISK_SAFETY_FACTOR
    free = shutil.disk_usage(config.renders_dir).free
    if free < need:
        raise RuntimeError(
            f"Insufficient disk: need ~{need / 1e9:.0f} GB (1.3x estimate), "
            f"have {free / 1e9:.0f} GB free at {config.renders_dir}.")
    log.info("disk check ok: est %.0f GB, free %.0f GB", need / 1e9, free / 1e9)


def _failed_row(device_id: int, f: dict, reason: str) -> dict:
    return {
        "device_id": device_id, "file_id": f["file_id"], "split": f["split"],
        "path": pd.NA, "output_scale": pd.NA, "out_rms_db": pd.NA, "out_peak_db": pd.NA,
        "render_sha256": pd.NA, "rendered_at": _now(), "nam_sha256": pd.NA,
        "status": C.RENDER_FAILED, "resampled": pd.NA, "fail_reason": reason[:300],
    }


def _sha256(path):
    from .corpus import sha256_file
    return sha256_file(path)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
