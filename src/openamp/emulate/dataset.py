"""Training pairs for amp emulation: clean input -> a device's rendered output.

Consumes the Phase 2 corpus + renders (:mod:`openamp.corpus`): the clean FLACs
under ``clean/{split}/`` and each device's aligned renders under
``renders/{device:04d}/`` (same length, sample-aligned — verify enforces it). One
item is a random ``(device, file, window)`` draw: a 2 s clip of the render as the
target, and the *same* clip of the clean input **prefixed with the receptive
field of real left-context** from the source file, so the causal model is fully
warmed and the loss is computed only on conditioned samples (spec §4.1, warmup).

The per-device **embedding-table index is global and stable**: it is derived once
from the sorted render-ok device set (:func:`build_device_index`) so a device's
row is identical across train/val/test and can be saved for Phase 5 to extend.

I/O is a seek-read per clip (:func:`openamp.dsp.audio.seek_read`) — batch 16
through a real TCN is compute-heavy enough that the simple path keeps pace; add
a preloaded crop cache only if a run turns out I/O-bound.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from openamp.core import constants as C
from openamp.core import manifest as manifests
from openamp.core.util import sha256_file


def render_ok_devices(config) -> list[int]:
    """Sorted device ids with at least one ``render_ok`` render (any split)."""
    renders = manifests.read_manifest(config.renders_manifest_path, manifests.RENDERS_COLUMNS)
    if renders.empty:
        return []
    ok = renders[renders["status"] == C.RENDER_OK]
    return sorted(int(d) for d in ok["device_id"].unique())


def load_or_create_holdout(config, frac: float, seed: int) -> list[int]:
    """Sorted device ids held out of one-to-many training (Phase 5 enrollment set).

    Drawn once as a seeded sample of the render-ok set and persisted to
    ``emulate_holdout.txt``; the file, not the RNG, is the source of truth, so
    the holdout stays fixed even if more devices are rendered later. ``frac <= 0``
    disables the holdout (returns ``[]`` without touching the file).
    """
    if frac <= 0:
        return []
    path = Path(config.emulate_holdout_path)
    if path.is_file():
        return sorted(int(line) for line in path.read_text(encoding="utf-8").splitlines()
                      if line.strip() and not line.lstrip().startswith("#"))
    pool = render_ok_devices(config)
    n = max(1, round(frac * len(pool)))
    rng = np.random.default_rng(seed)
    ids = sorted(int(d) for d in rng.choice(pool, size=n, replace=False))
    header = [
        f"# Open-Amp3000 emulate holdout — {len(ids)} of {len(pool)} render-ok devices "
        f"(frac {frac:g}, seed {seed})",
        "# excluded from one-to-many training; Phase 5 enrolls these as unseen devices",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + [str(d) for d in ids]) + "\n", encoding="utf-8")
    return ids


def build_device_index(config, single_device: int | None = None, *,
                       exclude: "set[int] | list[int]" = ()
                       ) -> tuple[list[int], dict[int, int]]:
    """Canonical ``(device_ids, {device_id: row})`` for the embedding table.

    ``single_device >= 0`` builds a one-row table for the one-to-one baseline
    (``exclude`` is ignored there — Phase 5 trains one-to-one on held-out
    devices); otherwise the table spans every render-ok device not in
    ``exclude``, in a fixed sorted order.
    """
    if single_device is not None and single_device >= 0:
        ids = [int(single_device)]
    else:
        drop = set(exclude)
        ids = [d for d in render_ok_devices(config) if d not in drop]
    return ids, {d: i for i, d in enumerate(ids)}


def manifest_signature(config) -> str:
    """SHA-256 of the device manifest the table indexes (provenance for Phase 5)."""
    path = config.devices_final_path if config.devices_final_path.is_file() \
        else config.renders_manifest_path
    return sha256_file(path) if Path(path).is_file() else ""


class EmulationDataset(Dataset):
    """Random ``(clean-with-context, render-clip, device_row)`` pairs for a split.

    ``__getitem__`` returns a dict:
      - ``input``:  ``[receptive_field + clip]`` clean audio (real left-context).
      - ``target``: ``[clip]`` render audio (the warmed region the loss covers).
      - ``device_idx``: the global embedding-table row for that device.

    The train loop feeds ``input`` through the causal model and compares
    ``output[..., receptive_field:]`` against ``target``. Bump ``seed`` per epoch
    (and rebuild the loader) for fresh window positions.
    """

    def __init__(self, config, split: str, *, receptive_field: int,
                 id_to_idx: dict[int, int], clip_samples: int | None = None,
                 pairs_per_epoch: int = 50_000, seed: int = 1234):
        from openamp.dsp import audio as audio_io

        self.config = config
        self.split = split
        self.receptive_field = int(receptive_field)
        self.clip_samples = int(clip_samples if clip_samples is not None
                                else config.clip_samples)
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.seed = int(seed)
        self.id_to_idx = dict(id_to_idx)

        corpus = manifests.read_manifest(config.corpus_manifest_path, manifests.CORPUS_COLUMNS)
        renders = manifests.read_manifest(config.renders_manifest_path, manifests.RENDERS_COLUMNS)
        if corpus.empty or renders.empty:
            raise RuntimeError("Emulation needs a rendered corpus. Run `openamp corpus` "
                               "and `openamp render` first.")

        # Clean source files of this split + their true length (shared by every
        # device's aligned render, so we probe length once per file, not per device).
        files = corpus[corpus["split"] == split]
        self.files: list[dict] = []
        for r in files.itertuples(index=False):
            clean_path = config.clean_split_dir(split) / f"{r.file_id}.{config.output_format}"
            if not clean_path.is_file():
                continue
            n = audio_io.audio_info(clean_path)[0]
            if n >= self.clip_samples:                # need at least one full clip
                self.files.append({"file_id": str(r.file_id), "n": int(n)})
        if not self.files:
            raise RuntimeError(f"No usable clean files for split '{split}'.")

        # Devices available in this split that are also in the (global) table.
        ok = renders[(renders["split"] == split) & (renders["status"] == C.RENDER_OK)]
        present = sorted({int(d) for d in ok["device_id"].unique()} & set(self.id_to_idx))
        if not present:
            raise RuntimeError(f"No render-ok devices for split '{split}'.")
        self.device_ids = present
        self.rows = np.asarray([self.id_to_idx[d] for d in present], dtype=np.int64)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _read_pair(self, device_id: int, file: dict, rng) -> tuple[np.ndarray, np.ndarray]:
        from openamp.dsp import audio as audio_io

        R, clip = self.receptive_field, self.clip_samples
        n = file["n"]
        c = int(rng.integers(0, n - clip + 1))        # clip start in the source file
        clean_path = self.config.clean_split_dir(self.split) / \
            f"{file['file_id']}.{self.config.output_format}"
        render_path = self.config.device_render_dir(device_id) / \
            f"{file['file_id']}.{self.config.output_format}"

        # Input: clean[c-R : c+clip] with zero left-pad when the clip starts < R in.
        read_start = c - R
        pad = 0
        if read_start < 0:
            pad, read_start = -read_start, 0
        clean = audio_io.seek_read(clean_path, read_start, R + clip - pad)
        if pad:
            clean = np.concatenate([np.zeros(pad, dtype=np.float32), clean])
        # Target: render[c : c+clip] (the fully-warmed region the loss covers).
        target = audio_io.seek_read(render_path, c, clip)
        return clean.astype(np.float32, copy=False), target.astype(np.float32, copy=False)

    def item_arrays(self, i: int) -> dict:
        """One draw, tolerant of transient I/O failures: the render store lives on
        NFS, where a rare read flake would otherwise kill a multi-hour run. Each
        failed attempt logs, backs off briefly, and redraws (the rng advances, so
        a genuinely bad file is also routed around); four consecutive failures
        still raise — that is an outage, not a flake."""
        rng = np.random.default_rng((self.seed, int(i)))
        last_err: Exception | None = None
        for attempt in range(4):
            d = int(rng.integers(0, len(self.device_ids)))
            device_id = self.device_ids[d]
            file = self.files[int(rng.integers(0, len(self.files)))]
            try:
                clean, target = self._read_pair(device_id, file, rng)
            except Exception as e:
                last_err = e
                print(f"[dataset] read failed (attempt {attempt + 1}/4) "
                      f"device={device_id} file={file['file_id']}: {e!r}", flush=True)
                time.sleep(0.5 * (attempt + 1))
                continue
            return {"input": clean, "target": target, "device_idx": int(self.rows[d])}
        raise RuntimeError(
            f"4 consecutive read failures in split '{self.split}' "
            f"(last: {last_err!r}) — check the render store / filesystem")

    def __getitem__(self, i: int) -> dict:
        a = self.item_arrays(i)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(a["input"])),
            "target": torch.from_numpy(np.ascontiguousarray(a["target"])),
            "device_idx": torch.tensor(a["device_idx"], dtype=torch.long),
        }
