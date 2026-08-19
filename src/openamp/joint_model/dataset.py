"""Joint-training samples: a generator window plus a *disjoint* reference segment.

One item carries both halves of the joint model's input:

- ``input`` / ``target`` — the generator's window A, exactly what
  :class:`~openamp.emulate.dataset.EmulationDataset` already yields (clean with
  real left-context, and the warmed render region the loss covers).
- ``ref`` — window B, a ``(dry, wet)`` pair of the **same device** taken from a
  **different corpus file**, which the encoder turns into the embedding used to
  render window A.

**The disjointness is the whole mechanism, not a detail.** If B overlapped A the
encoder could pass the answer forward and the embedding would carry content
instead of tone. Drawing B from a different ``file_id`` is a much harder
barrier than a gap within one file. The same-file path with an enforced gap
exists only for single-file corpora (``sources: [sweep]``) and is strictly
the weaker guarantee.

Note that the dry side is shared across every device in the corpus, so a
reference is only informative through its wet channel — the encoder cannot
identify an amp by recognizing the music, because every amp is rendered from the
same music.

Everything else — manifests, source filtering, device rows, the NFS read-retry
policy — comes from the parent unchanged. Uniform device draws rather than the
encoder's P x K episodes: with no contrastive term there is nothing that needs
two segments of the same amp in a batch, and more distinct amps per batch is
strictly better for the reconstruction gradient.
"""

from __future__ import annotations

import numpy as np
import torch

from openamp.emulate.dataset import EmulationDataset, read_with_retry

__all__ = ["JointDataset", "REF_MODES"]

# "audio" yields a (dry, wet) reference segment for a trained encoder; "device" yields
# the device id, for an encoder that looks up a precomputed fingerprint.
REF_MODES = ("audio", "device")


class JointDataset(EmulationDataset):
    """``EmulationDataset`` plus a disjoint ``(dry, wet)`` reference segment.

    ``__getitem__`` returns the parent's ``input`` / ``target`` / ``device_idx``
    and adds ``ref`` ``[2, ref_samples]``.

    ``ref_different_file`` (default) draws B from another file of the same split.
    When the split holds only one usable file it falls back to a same-file draw
    separated from A's *read* span (which reaches back a receptive field) by at
    least ``ref_min_gap`` samples on whichever side has room.
    """

    def __init__(self, config, split: str, *, receptive_field: int,
                 id_to_idx: dict[int, int], ref_samples: int,
                 clip_samples: int | None = None, ref_different_file: bool = True,
                 ref_min_gap: int = 0, pairs_per_epoch: int = 50_000,
                 seed: int = 1234, sources: "tuple[str, ...] | list[str]" = (),
                 ref_mode: str = "audio"):
        super().__init__(config, split, receptive_field=receptive_field,
                         id_to_idx=id_to_idx, clip_samples=clip_samples,
                         pairs_per_epoch=pairs_per_epoch, seed=seed, sources=sources)
        self.ref_samples = int(ref_samples)
        self.ref_min_gap = int(ref_min_gap)
        self.ref_different_file = bool(ref_different_file)
        if ref_mode not in REF_MODES:
            raise ValueError(f"ref_mode must be one of {REF_MODES}, got {ref_mode!r}")
        self.ref_mode = str(ref_mode)

        if self.ref_mode == "device":
            # The reference IS the amp's identity: a device id the encoder resolves to
            # a precomputed fingerprint. Nothing to read and nothing to keep disjoint --
            # that fingerprint is pooled from a whole separate capture render, so it
            # cannot overlap window A no matter where A lands.
            self.ref_files, self.same_file_mode, self._ref_pos = [], False, {}
            return

        # Reference candidates are drawn from the files the parent already probed,
        # so a file must hold a whole clip *and* a whole reference to be one. With
        # the usual clip == ref length that is no restriction at all.
        self.ref_files = [f for f in self.files if f["n"] >= self.ref_samples]
        if not self.ref_files:
            raise RuntimeError(
                f"No file in split '{split}' holds a {self.ref_samples}-sample reference "
                f"(longest is {max(f['n'] for f in self.files)}). Lower joint.ref_samples.")

        # Only one file to draw from means every reference is a same-file draw,
        # whatever the flag says.
        self.same_file_mode = (not self.ref_different_file) or len(self.ref_files) < 2
        if self.same_file_mode:
            # A and B must both fit with the gap, for *any* clip start A lands on.
            # The binding case is a clip just past the point where a before-window
            # stops fitting; solving both sides for that c gives this bound.
            need = (self.receptive_field + self.clip_samples
                    + 2 * self.ref_samples + 2 * self.ref_min_gap)
            keep = [f for f in self.files if f["n"] >= need]
            if not keep:
                raise RuntimeError(
                    f"Same-file references need {need} samples per file "
                    f"(receptive field + clip + 2x reference + 2x gap); the longest file "
                    f"in split '{split}' has {max(f['n'] for f in self.files)}. Shorten "
                    f"joint.ref_samples or joint.ref_min_gap.")
            self.files = keep                       # dropped, never silently clamped
            self.ref_files = keep
        self._ref_pos = {f["file_id"]: i for i, f in enumerate(self.ref_files)}

    # --- Reference draw ---------------------------------------------------------
    def _draw_reference(self, file: dict, c: int, rng) -> tuple[dict, int]:
        """``(file, start)`` for window B, disjoint from window A."""
        if not self.same_file_mode:
            # Uniform over every file *except* A's: draw from one fewer slot and
            # step over the excluded index, rather than rejection-sampling.
            excl = self._ref_pos.get(file["file_id"])
            n = len(self.ref_files)
            j = int(rng.integers(0, n - 1 if excl is not None else n))
            if excl is not None and j >= excl:
                j += 1
            ref_file = self.ref_files[j]
            return ref_file, int(rng.integers(0, ref_file["n"] - self.ref_samples + 1))
        return file, self._draw_same_file_start(file, c, rng)

    def _draw_same_file_start(self, file: dict, c: int, rng) -> int:
        """A start whose window clears A's read span by ``ref_min_gap`` either side.

        A occupies ``[c - receptive_field, c + clip)``: the left-context is part of
        what the generator sees, so the reference has to clear that too, not just
        the target region.
        """
        L, gap = self.ref_samples, self.ref_min_gap
        spans = []
        before_hi = c - self.receptive_field - gap - L
        if before_hi >= 0:
            spans.append((0, before_hi))
        after_lo, after_hi = c + self.clip_samples + gap, file["n"] - L
        if after_lo <= after_hi:
            spans.append((after_lo, after_hi))
        if not spans:                               # excluded by the ctor's length check
            raise RuntimeError(f"no disjoint reference position in {file['file_id']} "
                               f"for clip start {c}")
        widths = np.array([hi - lo + 1 for lo, hi in spans], dtype=np.float64)
        lo, hi = spans[int(rng.choice(len(spans), p=widths / widths.sum()))]
        return int(rng.integers(lo, hi + 1))

    def _read_reference(self, device_id: int, file: dict, c: int) -> np.ndarray:
        """``[2, ref_samples]`` — the clean source and this device's render of it."""
        from openamp.dsp import audio as audio_io

        dry = audio_io.seek_read(self._clean_path(file), c, self.ref_samples)
        render_path = self.config.device_render_dir(device_id) / \
            f"{file['file_id']}.{self.config.output_format}"
        wet = audio_io.seek_read(render_path, c, self.ref_samples)
        return np.stack([dry, wet], axis=0).astype(np.float32, copy=False)

    # --- Item -------------------------------------------------------------------
    def item_arrays(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, int(i)))

        def draw():
            # Window A is drawn first and the reference consumes the RNG only
            # afterwards, so A is bit-identical between the two ref_modes at one seed.
            # That is the experimental control when comparing conditioning sources.
            d = int(rng.integers(0, len(self.device_ids)))
            file = self.files[int(rng.integers(0, len(self.files)))]
            c = int(rng.integers(0, file["n"] - self.clip_samples + 1))
            if self.ref_mode == "device":
                return ((d, file, c, None, 0),
                        f"device={self.device_ids[d]} file={file['file_id']} ref=id")
            ref_file, c_ref = self._draw_reference(file, c, rng)
            return ((d, file, c, ref_file, c_ref),
                    f"device={self.device_ids[d]} file={file['file_id']} "
                    f"ref={ref_file['file_id']}")

        def read(slot):
            d, file, c, ref_file, c_ref = slot
            device_id = self.device_ids[d]
            clean, target = self._read_window(device_id, file, c)
            ref = (np.int64(device_id) if self.ref_mode == "device"
                   else self._read_reference(device_id, ref_file, c_ref))
            return {"input": clean, "target": target, "ref": ref,
                    "device_idx": int(self.rows[d])}

        return read_with_retry(draw, read, split=self.split)

    def __getitem__(self, i: int) -> dict:
        a = self.item_arrays(i)
        ref = (torch.tensor(int(a["ref"]), dtype=torch.long) if self.ref_mode == "device"
               else torch.from_numpy(np.ascontiguousarray(a["ref"])))
        return {
            "input": torch.from_numpy(np.ascontiguousarray(a["input"])),
            "target": torch.from_numpy(np.ascontiguousarray(a["target"])),
            "ref": ref,
            "device_idx": torch.tensor(a["device_idx"], dtype=torch.long),
        }
