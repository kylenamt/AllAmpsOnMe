"""Training pairs for amp emulation: clean input -> a device's rendered output.

- Consumes the corpus + renders (:mod:`openamp.corpus`): clean FLACs under
  ``clean/{split}/`` and each device's aligned renders under
  ``renders/{device:04d}/`` (same length, sample-aligned — verify enforces it).
- One item is a random ``(device, file, window)`` draw: a clip of the render as
  the target, and the *same* clip of the clean input **prefixed with the
  receptive field of real left-context** from the source file, so the causal
  model is fully warmed and the loss is computed only on conditioned samples.
- The per-device **embedding-table index is global and stable**: derived once
  from the sorted render-ok device set
  (:func:`~openamp.data.corpus_index.build_device_index`) so a device's row is
  identical across train/val/test.
- I/O is a seek-read per clip (:func:`openamp.dsp.audio.seek_read`).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import Dataset

from openamp.core import constants as C
from openamp.core import manifest as manifests
from openamp.data.corpus_index import CorpusPaths

__all__ = ["EmulationDataset", "read_with_retry", "ITEM_READ_ATTEMPTS"]


ITEM_READ_ATTEMPTS = 4          # consecutive failures before a read is called an outage


def read_with_retry(draw, read, *, split: str, attempts: int = ITEM_READ_ATTEMPTS):
    """Draw one item and read it, tolerating transient store failures.

    - The render store lives on NFS, where a rare read flake would otherwise kill
      a multi-hour run.
    - ``draw()`` returns ``(slot, tag)``: an opaque item identity and a short
      string for the log. ``read(slot)`` returns the item or raises.
    - First retry re-reads the **same** slot after a brief backoff — a passing NFS
      hiccup clears on its own, and the val set (fixed seed, never redrawn across
      epochs) stays the intended item instead of silently swapping in an unrelated
      one. Only once that same-item retry has also failed do we redraw, to route
      around a file that's genuinely bad rather than just unlucky.
    - ``attempts`` consecutive failures still raise — that's an outage, not a flake.

    Shared by :class:`EmulationDataset` and :class:`ToneSegmentDataset`; the
    same-item-then-redraw policy is a bug fix, pinned by tests in
    ``tests/test_emulate.py``.
    """
    slot, tag = draw()
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            return read(slot)
        except Exception as e:
            last_err = e
            print(f"[dataset] read failed (attempt {attempt + 1}/{attempts}) "
                  f"{tag}: {e!r}", flush=True)
            time.sleep(0.5 * (attempt + 1))
            if attempt >= 1:        # same-item retry also failed -- could be a bad file
                slot, tag = draw()
    raise RuntimeError(
        f"{attempts} consecutive read failures in split '{split}' "
        f"(last: {last_err!r}) — check the render store / filesystem")


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

    def __init__(self, paths: CorpusPaths, split: str, *, receptive_field: int,
                 id_to_idx: dict[int, int], clip_samples: int,
                 pairs_per_epoch: int = 50_000, seed: int = 1234,
                 sources: "tuple[str, ...] | list[str]" = ()):
        from openamp.dsp import audio as audio_io

        self.paths = paths
        self.split = split
        self.receptive_field = int(receptive_field)
        self.clip_samples = int(clip_samples)
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.seed = int(seed)
        self.id_to_idx = dict(id_to_idx)
        self.sources = tuple(str(s) for s in sources)   # () = every source

        corpus = manifests.read_manifest(paths.corpus_manifest, manifests.CORPUS_COLUMNS)
        renders = manifests.read_manifest(paths.renders_manifest, manifests.RENDERS_COLUMNS)
        if corpus.empty or renders.empty:
            raise RuntimeError("Emulation needs a rendered corpus. Run `openamp corpus` "
                               "and `openamp render` first.")

        # Clean source files of this split + their true length (shared by every
        # device's aligned render, so we probe length once per file, not per device).
        files = corpus[corpus["split"] == split]
        if self.sources:                              # e.g. sweep-only encoder training
            files = files[files["source"].isin(self.sources)]
        self.files: list[dict] = []
        for r in files.itertuples(index=False):
            clean_path = paths.clean_file(split, r.file_id)
            if not clean_path.is_file():
                continue
            n = audio_io.audio_info(clean_path)[0]
            if n >= self.clip_samples:                # need at least one full clip
                self.files.append({"file_id": str(r.file_id), "n": int(n)})
        if not self.files:
            src = f" (sources={list(self.sources)})" if self.sources else ""
            raise RuntimeError(f"No usable clean files for split '{split}'{src}.")

        # Devices available in this split that are also in the (global) table.
        ok = renders[(renders["split"] == split) & (renders["status"] == C.RENDER_OK)]
        present = sorted({int(d) for d in ok["device_id"].unique()} & set(self.id_to_idx))
        if not present:
            raise RuntimeError(f"No render-ok devices for split '{split}'.")
        self.device_ids = present
        self.rows = np.asarray([self.id_to_idx[d] for d in present], dtype=np.int64)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _clean_path(self, file: dict) -> Path:
        return self.paths.clean_file(self.split, file["file_id"])

    def _read_pair(self, device_id: int, file: dict, rng) -> tuple[np.ndarray, np.ndarray]:
        c = int(rng.integers(0, file["n"] - self.clip_samples + 1))   # clip start
        return self._read_window(device_id, file, c)

    def _read_window(self, device_id: int, file: dict, c: int) -> tuple[np.ndarray, np.ndarray]:
        """The pair at a *given* clip start ``c`` (the draw itself lives in the caller)."""
        from openamp.dsp import audio as audio_io

        R, clip = self.receptive_field, self.clip_samples
        clean_path = self._clean_path(file)
        render_path = self.paths.render_file(device_id, file["file_id"])

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
        """One ``(device, file, window)`` draw, tolerant of transient I/O failures.

        The draw itself lives here; the flake policy lives in
        :func:`read_with_retry` (same item twice, then redraw, then raise).
        """
        rng = np.random.default_rng((self.seed, int(i)))

        def draw():
            d = int(rng.integers(0, len(self.device_ids)))
            file = self.files[int(rng.integers(0, len(self.files)))]
            return (d, file), f"device={self.device_ids[d]} file={file['file_id']}"

        def read(slot):
            d, file = slot
            clean, target = self._read_pair(self.device_ids[d], file, rng)
            return {"input": clean, "target": target, "device_idx": int(self.rows[d])}

        return read_with_retry(draw, read, split=self.split)

    def __getitem__(self, i: int) -> dict:
        a = self.item_arrays(i)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(a["input"])),
            "target": torch.from_numpy(np.ascontiguousarray(a["target"])),
            "device_idx": torch.tensor(a["device_idx"], dtype=torch.long),
        }
