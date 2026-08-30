"""Every device evaluated on the same fixed grid of windows.

Split out of :mod:`openamp.data.emulation`: same item contract, but the windows
are drawn once and replayed for every device, which is what makes per-device ESR
comparable *between* devices.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from openamp.data.corpus_index import CorpusPaths
from openamp.data.emulation import EmulationDataset

__all__ = ["DeviceGridDataset"]


class DeviceGridDataset(EmulationDataset):
    """Every device evaluated on the **same** fixed grid of test windows.

    - Per-device ESR is only comparable *between* devices if the audio is held
      constant — with independent random draws a device can look good or bad on
      window luck alone. So the grid of ``(file, clip start)`` positions is
      drawn once from ``seed`` and replayed for every device; item ``i`` is
      ``(device i // n_windows, window i % n_windows)``, so one sequential pass
      walks the whole device x window matrix device-major, letting a caller
      accumulate per-device sums as batches arrive.
    - Windows are pre-filtered on the **clean** signal's RMS (``silence_dbfs``):
      a near-silent 2 s window carries no amp behaviour, and screening on the
      shared clean side (not each device's render) keeps the grid identical
      across devices.
    """

    def __init__(self, paths: CorpusPaths, split: str, *, receptive_field: int,
                 id_to_idx: dict[int, int], device_ids: "list[int] | None" = None,
                 clip_samples: int, n_windows: int = 128,
                 seed: int = 1234, silence_dbfs: float = -50.0):
        super().__init__(paths, split, receptive_field=receptive_field,
                         id_to_idx=id_to_idx, clip_samples=clip_samples,
                         pairs_per_epoch=1, seed=seed)
        if device_ids is not None:                     # keep only what was asked for
            want = {int(d) for d in device_ids}
            keep = [i for i, d in enumerate(self.device_ids) if d in want]
            if not keep:
                raise RuntimeError(f"None of the requested devices have "
                                   f"render-ok files in split '{split}'.")
            self.device_ids = [self.device_ids[i] for i in keep]
            self.rows = self.rows[keep]
        self.n_windows = int(n_windows)
        self.windows = self._draw_windows(self.n_windows, silence_dbfs)
        self.n_windows = len(self.windows)             # may fall short of the request

    def _draw_windows(self, n_windows: int, silence_dbfs: float) -> list[tuple[int, int]]:
        """``[(file index, clip start), ...]``, seeded and screened for silence."""
        from openamp.dsp import audio as audio_io

        rng = np.random.default_rng(self.seed)
        floor = 10.0 ** (silence_dbfs / 20.0)
        out: list[tuple[int, int]] = []
        for _ in range(8 * n_windows):                 # bounded: silence is a minority
            if len(out) >= n_windows:
                break
            f = int(rng.integers(0, len(self.files)))
            file = self.files[f]
            c = int(rng.integers(0, file["n"] - self.clip_samples + 1))
            try:
                x = audio_io.seek_read(self._clean_path(file), c, self.clip_samples)
            except Exception as e:  # noqa: BLE001  — a bad file must not sink the grid
                print(f"[eval] skipping window {file['file_id']}@{c}: {e!r}", flush=True)
                continue
            if float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) >= floor:
                out.append((f, c))
        if not out:
            raise RuntimeError(f"No non-silent test windows in split '{self.split}'.")
        if len(out) < n_windows:
            print(f"[eval] only {len(out)} of {n_windows} windows cleared the "
                  f"{silence_dbfs:g} dBFS floor", flush=True)
        return out

    def __len__(self) -> int:
        return len(self.device_ids) * self.n_windows

    def item_arrays(self, i: int) -> dict:
        """One grid cell, retrying the *same* window on transient I/O failures.

        Unlike the training dataset this never redraws: the grid is the point, so
        a window that stays unreadable raises rather than silently making one
        device's ESR mean over different audio than its neighbours'.
        """
        d, w = divmod(int(i), self.n_windows)
        device_id = self.device_ids[d]
        f, c = self.windows[w]
        file = self.files[f]
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                clean, target = self._read_window(device_id, file, c)
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[eval] read failed (attempt {attempt + 1}/4) device={device_id} "
                      f"file={file['file_id']}@{c}: {e!r}", flush=True)
                time.sleep(0.5 * (attempt + 1))
                continue
            return {"input": clean, "target": target,
                    "device_idx": int(self.rows[d]), "device_id": int(device_id)}
        raise RuntimeError(f"4 consecutive read failures for device {device_id} "
                           f"file {file['file_id']}@{c} (last: {last_err!r})")

    def __getitem__(self, i: int) -> dict:
        a = self.item_arrays(i)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(a["input"])),
            "target": torch.from_numpy(np.ascontiguousarray(a["target"])),
            "device_idx": torch.tensor(a["device_idx"], dtype=torch.long),
            "device_id": torch.tensor(a["device_id"], dtype=torch.long),
        }
