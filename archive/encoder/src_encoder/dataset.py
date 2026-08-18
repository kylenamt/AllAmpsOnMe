"""Datasets for the tone encoder: ``(dry, wet)`` segments grouped by capture.

Two datasets and the time-split policy they share:

- :class:`ToneSegmentDataset` — training. Yields a P x K *episode* structure
  (P captures x K segments) built into the item index rather than a Sampler, so
  the SupCon positives exist by construction and every item stays a pure function
  of ``(seed, i)``.
- :class:`CaptureGridDataset` — evaluation. Every device on the *same* fixed grid
  of window positions, so capture embeddings are comparable between devices.
- :func:`time_split_regions` — where the train/val cut in time falls, and which
  part of each file is usable at all. Not a detail; see its docstring.

**This module depends on** :mod:`openamp.emulate.dataset` for corpus plumbing —
:class:`~openamp.emulate.dataset.EmulationDataset` resolves the manifests, filters
by source, probes file lengths and maps device ids to table rows, and
:func:`~openamp.emulate.dataset.read_with_retry` carries the NFS retry policy.
That is the one seam between the encoder and the emulators; nothing flows the
other way, and no emulator module imports anything from ``openamp.encoder``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from openamp.core import constants as C
from openamp.core import manifest as manifests
from openamp.emulate.dataset import EmulationDataset, read_with_retry


def device_labels(config, device_ids: "list[int] | None" = None) -> dict:
    """Identity + coarse-gain labels per device, for the encoder's supervised probes.

    The repo's identity hierarchy, which the contrastive setup leans on:

    - ``device_id`` is the **capture**: one ``.nam`` = one amp at one knob setting.
      This is the contrastive grouping key (the spec's ``capture_id``).
    - ``tone_id`` is the **amp**: one TONE3000 upload holds several captures of the
      same amp at different settings, so it groups captures the way an "amp id"
      should. This is what the amp probe classifies.
    - ``(make, model)`` is deliberately **not** used as amp identity — ``model`` is
      derived from the settings string, so it is ~1:1 with ``device_id`` and would
      make the amp probe a restatement of the contrastive task.

    Returns ``{"tone_idx", "gain_idx", "tone_ids", "n_tones", "n_gain"}`` where the
    two ``*_idx`` maps are ``{device_id: contiguous class}``. ``gain_idx`` is ``-1``
    for the ``unknown`` bucket (a CE ``ignore_index``), since ``gain_bucket`` is a
    keyword heuristic and a wrong label is worse than no label.
    """
    df = manifests.read_manifest(config.manifest_path, manifests.ALL_COLUMNS)
    keep = None if device_ids is None else {int(d) for d in device_ids}
    tone_of: dict[int, int] = {}
    gain_of: dict[int, str] = {}
    if not df.empty and "device_id" in df.columns:
        df = df.dropna(subset=["device_id"])
        for did, tone, bucket in zip(df["device_id"], df["tone_id"], df["gain_bucket"]):
            did = int(did)
            if keep is not None and did not in keep:
                continue
            tone_of[did] = int(tone) if tone == tone and tone is not None else did
            gain_of[did] = str(bucket) if bucket == bucket and bucket is not None \
                else C.GAIN_UNKNOWN
    # A device missing from the manifest still needs a class: give it its own tone.
    for did in (keep or set()):
        tone_of.setdefault(did, did)
        gain_of.setdefault(did, C.GAIN_UNKNOWN)

    tone_order = sorted(set(tone_of.values()))
    tone_rank = {t: i for i, t in enumerate(tone_order)}
    gain_rank = {b: i for i, b in enumerate(C.GAIN_BUCKETS)}
    return {
        "tone_idx": {d: tone_rank[t] for d, t in tone_of.items()},
        "gain_idx": {d: gain_rank.get(b, -1) for d, b in gain_of.items()},
        "tone_ids": tone_of,
        "n_tones": len(tone_order),
        "n_gain": len(C.GAIN_BUCKETS),
    }


# --- Targets and the time-split policy -----------------------------------------
CREST_SCALE_DB = 20.0           # divisor that puts the crest target in an O(1) range
DRY_CACHE_BYTES = 256 << 20     # per-worker budget for cached dry files (256 MiB)


def crest_db(x: np.ndarray, eps: float = 1e-12) -> float:
    """Crest factor in dB: ``20*log10(peak / rms)``. 0 for a silent buffer."""
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if peak <= eps or rms <= eps:
        return 0.0
    return 20.0 * float(np.log10(peak / rms))


def time_split_regions(val_frac: float, *, mode: str = "strided", blocks: int = 12,
                       within: tuple = (0.0, 1.0)):
    """``(train_regions, val_regions)`` as tuples of ``(lo, hi)`` file fractions.

    ``within`` bounds the part of each file the encoder may use **at all**; the
    train/val split is then taken inside it. This exists because the two questions
    are separate and were conflated once already: *where* val is taken decides
    whether the metric is honest, but *what material is in the pool* decides how
    good the encoder gets. Measured 2026-08-12 on ``data_mixed`` at 404 devices —
    training over the whole file, versus over ``(0, 0.85)`` with the decayed tail
    excluded, cost retrieval top-1 0.973 -> 0.948, amp-id probe 0.894 -> 0.661 and
    crest R2 0.762 -> 0.607. The tail carries little about what an amp does and
    displaces budget that better audio would have used.

    The encoder splits train/val **in time** over the same captures — the device
    axis is already spent on ``emulate_holdout.txt``. *Where* in time the cut falls
    turns out to matter a great deal:

    - ``"tail"`` holds out the last ``val_frac`` of every file. It is the obvious
      split and it is wrong for a capture signal, because the signal is not
      level-stationary. Measured on ``data_mixed`` ``sweep_train.flac``: the first
      85% averages -20.98 dBFS and the last 15% averages -35.30 dBFS, and its first
      two windows sit at -57 dBFS, near the render noise floor. An amp is a
      level-dependent nonlinearity, so validating down there does not measure
      "held-out audio" — it measures a quieter operating point, where every amp is
      closer to linear and therefore closer to every other amp. The 2026-08-11
      ``sweep_base`` run was split this way; its val SupCon and crest MSE read
      pessimistically for that reason and not because the encoder was weak.
    - ``"strided"`` (default) cuts the file into ``blocks`` equal blocks and gives
      evenly-spaced ones to val, so both sides span the file and see the same level
      distribution.

    Blocks never overlap, so no val sample is ever trained on. Audio just either
    side of a block boundary is still correlated; that is inherent to splitting a
    single 171 s file in time, and it is why the *device* holdout — not this split —
    carries the generalization claim.
    """
    f = float(val_frac)
    if not 0.0 < f < 1.0:
        raise ValueError(f"time_val_frac must be in (0, 1), got {val_frac}")
    w_lo, w_hi = float(within[0]), float(within[1])
    if not 0.0 <= w_lo < w_hi <= 1.0:
        raise ValueError(f"usable_region must satisfy 0 <= lo < hi <= 1, got {tuple(within)}")

    def _place(regs) -> tuple:
        """Map fractions of the usable window back onto fractions of the whole file."""
        span = w_hi - w_lo
        return tuple((w_lo + a * span, w_lo + b * span) for a, b in regs)

    if mode == "tail":
        return _place(((0.0, 1.0 - f),)), _place(((1.0 - f, 1.0),))
    if mode != "strided":
        raise ValueError(f"unknown time_split_mode {mode!r} (expected 'strided' or 'tail')")

    n = int(blocks)
    if n < 2:
        raise ValueError(f"time_split_blocks must be >= 2, got {blocks}")
    n_val = min(n - 1, max(1, round(n * f)))
    # Evenly spaced val blocks: the j-th sits at the centre of its 1/n_val share, so
    # train and val alternate across the whole file rather than clumping at one end.
    val = {int((j + 0.5) * n / n_val) for j in range(n_val)}

    def _merge(pick) -> tuple:
        """Contiguous runs of chosen blocks, merged — longer runs mean more windows."""
        out, start = [], None
        for i in range(n):
            if i in pick and start is None:
                start = i
            elif i not in pick and start is not None:
                out.append((start / n, i / n))
                start = None
        if start is not None:
            out.append((start / n, 1.0))
        return tuple(out)

    return _place(_merge(set(range(n)) - val)), _place(_merge(val))


def region_level_db(ds, *, max_files: int = 4) -> float:
    """RMS dBFS of the clean audio a :class:`ToneSegmentDataset` can draw from.

    The check that ``time_split_regions`` actually worked. Train and val levels
    have to land close together, or val is measuring the amp at a different
    operating point and its numbers are not comparable to train's. Reads the clean
    side only: every device shares the dry file, so one pass characterizes the
    split for all of them.
    """
    from openamp.dsp import audio as audio_io

    total, n = 0.0, 0
    for f in ds.files[:max_files]:
        for lo, hi in f["spans"]:
            a = audio_io.seek_read(ds._clean_path(f), int(lo), int(hi - lo))
            a = np.asarray(a, dtype=np.float64)
            total += float(np.square(a).sum())
            n += a.size
    if n == 0:
        return float("nan")
    rms = math.sqrt(total / n)
    return 20.0 * math.log10(rms) if rms > 0 else float("-inf")


class ToneSegmentDataset(EmulationDataset):
    """``(dry, wet)`` segments for contrastive tone encoding, batched P captures x K segments.

    One item is a 2-channel segment: the clean source and the *same* window of that
    device's render. No left-context prefix — the encoder is non-causal and pools
    over the whole segment — so this reuses the parent's reader with
    ``receptive_field=0``.

    **The P x K batch structure is built into the index, not a Sampler.** Item ``i``
    decodes to ``(episode, device slot, repeat)``; a plain
    ``DataLoader(shuffle=False, drop_last=True, batch_size=P*K)`` therefore yields
    exactly ``P`` distinct captures with ``K`` segments each. That is what SupCon
    needs (positives inside the batch), it stays deterministic under any worker
    count, and it matches the repo's seed-and-index idiom — items are a pure
    function of ``(seed, i)``, never of shared RNG state.

    ``regions`` restricts window draws to one or more ``(lo, hi)`` fractions of each
    file, which is how train and val are split *in time* over the same devices: the
    device axis is already spoken for by ``emulate_holdout.txt`` (unseen amps for
    retrieval), so a second, orthogonal split is needed to watch for overfitting.
    It takes a *list* of regions rather than one so the split can be interleaved
    across the file — see :func:`time_split_regions` for why a single tail region
    silently changes what val measures.

    ``__getitem__`` returns ``x`` ``[2, T]`` (dry, wet), the capture row
    ``device_idx``, the real ``device_id``, the amp class ``tone_idx``, the gain
    class ``gain_idx`` (-1 = unknown), and the ``crest`` regression target.
    """

    def __init__(self, config, split: str, *, id_to_idx: dict[int, int],
                 segment_samples: int, sources=(), regions: tuple = ((0.0, 1.0),),
                 episodes: int = 400, devices_per_batch: int = 16,
                 segments_per_device: int = 4, labels: dict | None = None,
                 seed: int = 1234, undo_output_scale: bool = False,
                 augment: bool = False, noise_p: float = 0.0,
                 noise_snr_db: tuple = (30.0, 60.0), time_mask_p: float = 0.0,
                 time_mask_frac: float = 0.05, cache_dry: bool = True):
        super().__init__(config, split, receptive_field=0, id_to_idx=id_to_idx,
                         clip_samples=segment_samples, pairs_per_epoch=0, seed=seed,
                         sources=sources)
        self.segment_samples = int(segment_samples)
        self.episodes = int(episodes)
        self.P = int(devices_per_batch)
        self.K = int(segments_per_device)
        if self.K < 2:
            raise ValueError("segments_per_device must be >= 2 (SupCon needs positives)")
        if self.P > len(self.device_ids):
            raise ValueError(f"devices_per_batch {self.P} exceeds the {len(self.device_ids)} "
                             f"devices available in split '{split}'")

        self.regions = tuple((float(a), float(b)) for a, b in regions)
        if not self.regions:
            raise ValueError("regions must not be empty")
        for a, b in self.regions:
            if not 0.0 <= a < b <= 1.0:
                raise ValueError(f"each region must satisfy 0 <= lo < hi <= 1, got {(a, b)}")
        # Keep only the spans that still hold a whole segment, and only the files
        # that have at least one. A short block is dropped, never silently clamped.
        usable = []
        for f in self.files:
            spans = [(int(a * f["n"]), int(b * f["n"])) for a, b in self.regions]
            spans = [(a, b) for a, b in spans if b - a >= self.segment_samples]
            if spans:
                starts = np.array([b - a - self.segment_samples + 1 for a, b in spans],
                                  dtype=np.float64)
                usable.append({**f, "spans": spans, "span_w": starts / starts.sum()})
        if not usable:
            raise RuntimeError(
                f"No file in split '{split}' has a {self.segment_samples}-sample segment "
                f"inside region {self.regions}. Shorten encoder.segment_samples, lower "
                f"encoder.time_split_blocks, or widen the region.")
        self.files = usable

        labels = labels if labels is not None else device_labels(config, self.device_ids)
        self.tone_idx = np.asarray([labels["tone_idx"].get(d, 0) for d in self.device_ids],
                                   dtype=np.int64)
        self.gain_idx = np.asarray([labels["gain_idx"].get(d, -1) for d in self.device_ids],
                                   dtype=np.int64)
        self.n_tones = int(labels["n_tones"])
        self.n_gain = int(labels["n_gain"])

        self.undo_output_scale = bool(undo_output_scale)
        self.scales = self._load_scales() if self.undo_output_scale else {}
        self.augment = bool(augment)
        self.noise_p = float(noise_p)
        self.noise_snr_db = (float(noise_snr_db[0]), float(noise_snr_db[1]))
        self.time_mask_p = float(time_mask_p)
        self.time_mask_frac = float(time_mask_frac)
        self.cache_dry = bool(cache_dry)
        self._dry: dict[str, np.ndarray] = {}       # per-worker, filled lazily
        self._dry_bytes = 0

    # --- setup helpers ----------------------------------------------------------
    def _load_scales(self) -> dict:
        """``{device_id: output_scale}`` so peak-normalized renders can be undone."""
        r = manifests.read_manifest(self.config.renders_manifest_path,
                                    manifests.RENDERS_COLUMNS)
        ok = r[(r["split"] == self.split) & (r["status"] == C.RENDER_OK)]
        out = {}
        for did, s in zip(ok["device_id"], ok["output_scale"]):
            try:
                s = float(s)
            except (TypeError, ValueError):
                continue
            if s > 0:
                out[int(did)] = s
        return out

    def __len__(self) -> int:
        return self.episodes * self.P * self.K

    def batch_size(self) -> int:
        """Segments per batch — pass this to the DataLoader or the P x K split breaks."""
        return self.P * self.K

    def _draw_window(self, rng) -> tuple[dict, int]:
        """A ``(file, start)`` drawn uniformly over every legal start position.

        Spans are weighted by how many starts they hold, not picked uniformly: with
        an interleaved split the blocks differ in length, and picking per-span would
        oversample the short ones.
        """
        file = self.files[int(rng.integers(0, len(self.files)))]
        j = int(rng.choice(len(file["spans"]), p=file["span_w"]))
        lo, hi = file["spans"][j]
        return file, int(rng.integers(lo, hi - self.segment_samples + 1))

    # --- reading ----------------------------------------------------------------
    def _dry_array(self, file: dict) -> np.ndarray | None:
        """The whole clean file, cached per worker, or ``None`` if not cacheable.

        In sweep-only training every item in every batch reads the *same* dry file,
        so seek-reading it 64x per step is pure waste; one 171 s cache entry is
        33 MB. Bounded by ``DRY_CACHE_BYTES`` so an EGDB run degrades to seek-reads
        rather than exhausting RAM across ``num_workers`` copies.
        """
        if not self.cache_dry:
            return None
        fid = file["file_id"]
        hit = self._dry.get(fid)
        if hit is not None:
            return hit
        need = int(file["n"]) * 4
        if self._dry_bytes + need > DRY_CACHE_BYTES:
            return None
        from openamp.dsp import audio as audio_io
        arr = audio_io.seek_read(self._clean_path(file), 0, int(file["n"]))
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._dry[fid] = arr
        self._dry_bytes += need
        return arr

    def _read_segment(self, device_id: int, file: dict, c: int):
        """The ``(dry, wet)`` pair at segment start ``c``."""
        from openamp.dsp import audio as audio_io

        T = self.segment_samples
        cached = self._dry_array(file)
        dry = cached[c:c + T] if cached is not None \
            else audio_io.seek_read(self._clean_path(file), c, T)
        wet = audio_io.seek_read(
            self.config.device_render_dir(device_id) / f"{file['file_id']}.{self.config.output_format}",
            c, T)
        dry = np.ascontiguousarray(dry, dtype=np.float32)
        wet = np.ascontiguousarray(wet, dtype=np.float32)
        if self.undo_output_scale:
            wet = wet / np.float32(self.scales.get(int(device_id), 1.0))
        return dry, wet

    # --- augmentation (see encoder.train for why the list is this short) --------
    def _augment(self, dry: np.ndarray, wet: np.ndarray, rng) -> tuple:
        if rng.random() < self.noise_p:
            # Converter/room noise lands on the WET side only — the dry reference is
            # a file, not a recording.
            snr = rng.uniform(*self.noise_snr_db)
            sig = float(np.sqrt(np.mean(np.square(wet, dtype=np.float64))))
            if sig > 0:
                sigma = sig * (10.0 ** (-snr / 20.0))
                wet = wet + rng.standard_normal(wet.shape).astype(np.float32) * np.float32(sigma)
        if rng.random() < self.time_mask_p and self.time_mask_frac > 0:
            # Same span in BOTH channels: masking only one would fake a transfer
            # relation that no amp has.
            span = max(1, int(self.time_mask_frac * len(wet)))
            s = int(rng.integers(0, max(1, len(wet) - span)))
            dry, wet = dry.copy(), wet.copy()
            dry[s:s + span] = 0.0
            wet[s:s + span] = 0.0
        return dry, wet

    # --- items ------------------------------------------------------------------
    def _episode_devices(self, episode: int) -> np.ndarray:
        """``P`` distinct device slots for one batch (distinct => real SupCon negatives)."""
        rng = np.random.default_rng((self.seed, int(episode)))
        return rng.permutation(len(self.device_ids))[:self.P]

    def item_arrays(self, i: int) -> dict:
        per_batch = self.P * self.K
        episode, within = divmod(int(i), per_batch)
        slot, rep = divmod(within, self.K)
        d = int(self._episode_devices(episode)[slot])
        device_id = int(self.device_ids[d])
        rng = np.random.default_rng((self.seed, int(episode), int(slot), int(rep)))

        def draw():
            file, c = self._draw_window(rng)
            return (file, c), f"device={device_id} file={file['file_id']}"

        def read(slot_):
            file, c = slot_
            dry, wet = self._read_segment(device_id, file, c)
            if self.augment:
                dry, wet = self._augment(dry, wet, rng)
            return {
                "dry": dry, "wet": wet,
                "device_idx": int(self.rows[d]), "device_id": device_id,
                "tone_idx": int(self.tone_idx[d]), "gain_idx": int(self.gain_idx[d]),
                "crest": (crest_db(wet) - crest_db(dry)) / CREST_SCALE_DB,
            }

        return read_with_retry(draw, read, split=self.split)

    def __getitem__(self, i: int) -> dict:
        a = self.item_arrays(i)
        return {
            "x": torch.from_numpy(np.stack([a["dry"], a["wet"]], axis=0)),
            "device_idx": torch.tensor(a["device_idx"], dtype=torch.long),
            "device_id": torch.tensor(a["device_id"], dtype=torch.long),
            "tone_idx": torch.tensor(a["tone_idx"], dtype=torch.long),
            "gain_idx": torch.tensor(a["gain_idx"], dtype=torch.long),
            "crest": torch.tensor(a["crest"], dtype=torch.float32),
        }


class CaptureGridDataset(ToneSegmentDataset):
    """Every device on the **same** fixed grid of segment positions, device-major.

    The evaluation counterpart to :class:`ToneSegmentDataset`, mirroring what
    :class:`DeviceGridDataset` does for ESR: capture embeddings are only comparable
    *between* devices if the audio is held constant, so the ``(file, start)`` grid is
    drawn once and replayed for every device. Item ``i`` is
    ``(device i // n_windows, window i % n_windows)``, so one sequential pass walks
    the device x window matrix and a caller can aggregate per capture as batches
    arrive. No augmentation, ever.
    """

    def __init__(self, config, split: str, *, id_to_idx, segment_samples,
                 n_windows: int = 8, **kw):
        kw.pop("augment", None)
        kw.pop("episodes", None)
        # P/K are meaningless here (the grid replaces the episode structure); these
        # are the smallest values the parent's validity checks accept.
        kw.setdefault("devices_per_batch", 1)
        kw.setdefault("segments_per_device", 2)
        super().__init__(config, split, id_to_idx=id_to_idx,
                         segment_samples=segment_samples, augment=False, episodes=0, **kw)
        self.n_windows = int(n_windows)
        rng = np.random.default_rng(self.seed)
        self.windows = [self._draw_window(rng) for _ in range(self.n_windows)]

    def __len__(self) -> int:
        return len(self.device_ids) * self.n_windows

    def item_arrays(self, i: int) -> dict:
        d, w = divmod(int(i), self.n_windows)
        device_id = int(self.device_ids[d])
        file, c = self.windows[w]

        def draw():
            return (file, c), f"device={device_id} file={file['file_id']}"

        def read(slot_):
            f_, c_ = slot_
            dry, wet = self._read_segment(device_id, f_, c_)
            return {
                "dry": dry, "wet": wet,
                "device_idx": int(self.rows[d]), "device_id": device_id,
                "tone_idx": int(self.tone_idx[d]), "gain_idx": int(self.gain_idx[d]),
                "crest": (crest_db(wet) - crest_db(dry)) / CREST_SCALE_DB,
            }

        return read_with_retry(draw, read, split=self.split)
