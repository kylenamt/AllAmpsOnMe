"""Conditioning from a frozen audio codec's frames, pooled by a trained attention head.

Same interface contract as FingerprintEncoder/WaveNetEncoder (.embedding_dim,
__call__(ref, mask), .pooled_dim, .capacity_report()), so JointModel drives it
unchanged. The one thing this changes relative to FingerprintEncoder: the
pooled vector's source is a *trained* CrossAttentionPool
(openamp.encoder.probe) over the codec's raw per-frame latents, instead of a
fixed mean+std reduction computed once, offline, with no gradient.

Everything else about FingerprintEncoder's design carries over unchanged:

Lookup, not on-the-fly encoding. openamp.encoder.extract precomputes each
device's frames once from the full 171 s capture render; joint.ref_seconds and
the reference-disjointness machinery are inert under this encoder exactly as
they are under enc_kind="fingerprint", and for the same reason -- a short
reference window is not what these frames were extracted from.

No dry-vector subtraction. openamp.encoder.extract keeps wet frames only (see
its docstring): a device's dry frames were never extracted, so there is
nothing to subtract, and fp_preprocess is not offered here.

Frames are a buffer, not on this module directly -- FrameStore memmaps them
lazily from disk per openamp.encoder.probe's own reasoning (12.64 GiB across
450 devices, page-cached after the first epoch rather than GPU- or
process-resident). Only the pool and the projection below are parameters, so
the joint trainer's "encoder" optimizer group means exactly "pool + adapter",
the frame-pooling analogue of "the adapter" under enc_kind="fingerprint".

Cost this arm accepts that enc_kind="fingerprint" does not: FingerprintEncoder
looks up an 8 KB precomputed vector per device; this needs a [~14700, 1024]
fp16 tensor (~29 MiB) per device, ~966 MiB per batch at batch_size=32, to
produce a [B, 512] embedding. That gather happens in forward(), on the main
thread, and the GPU idles through it -- measured 0-12% utilization between
steps, ~68 smp/s overall against ~50 for an otherwise-identical fingerprint
run at the same batch size.

Moving the read into the DataLoader workers (a JointDataset ref_mode yielding
[T, D] per item) was tried and is *worse*, ~40 smp/s: a ~30 MiB item then
crosses worker-to-parent shared memory and pin_memory, which costs more than
the single-threaded gather it was meant to overlap. The bottleneck is the
volume itself, not where it is read, so the only real lever is moving fewer
frames per step.

frame_window (joint.enc_frame_window) is that lever, taken deliberately, and
its point is *not* the throughput -- that is a side effect. With the window
off, one device id maps to exactly one fixed [14728, 1024] tensor, so there
are 405 distinct encoder inputs in the whole training set and memorizing them
is available to the encoder as a strategy. Drawing a random contiguous window
each step makes one amp many inputs instead of one, which is what removes that
option.

Contiguous, not scattered: DAC's encoder receptive field measures 181 ms
(7,978 samples at 44.1 kHz, by gradient probe), so consecutive frames overlap
~94% and the ~14,728 frames hold only ~920 independent views. A scattered draw
would re-sample the same stationary average; a contiguous span is a genuinely
different look at the amp.

At eval the window is the centre one, deterministic and the *same length* as
training. Scoring on all 14,728 frames after training on a short window would
change the pool's softmax distribution -- it is taken over the frame axis --
and hand the generator an embedding drawn from a distribution it never saw.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from openamp.models import make_activation, normalize_activation
from openamp.encoder.probe import CrossAttentionPool, FrameStore

__all__ = ["FrameAttentionEncoder"]


class FrameAttentionEncoder(nn.Module):
    """``[B]`` device ids ``-> [B, embedding_dim]``.

    The adapter copies FingerprintEncoder's projection head shape
    (``Linear(pooled, h) -> act -> Linear(h, De)``, ``proj_hidden=0`` deriving
    as ``pooled_dim // 4``), so this arm differs from the fingerprint arm in
    exactly one place: a trained ``CrossAttentionPool`` over raw frames stands
    where a frozen mean+std vector was looked up.
    """

    def __init__(self, *, embedding_dim: int, frames: FrameStore,
                 n_queries: int = 4, d_model: int = 64, proj_hidden: int = 128,
                 activation: str = "tanh", normalize: bool = True,
                 frame_window: int = 0):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.normalize = bool(normalize)
        self.frame_window = int(frame_window)
        if self.frame_window < 0:
            raise ValueError(f"frame_window must be >= 0 (0 = all frames), got "
                             f"{self.frame_window}")
        if self.frame_window > frames.n_frames:
            raise ValueError(f"frame_window {self.frame_window} exceeds the store's "
                             f"{frames.n_frames} frames per device")
        self.activation = normalize_activation(activation, field="enc_activation")
        self.frames = frames

        ids = np.asarray(frames.device_ids, dtype=np.int64)
        if ids.min() < 0:
            raise ValueError("frame store must not contain a dry id (extract.py writes "
                             "wet frames only)")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("duplicate device ids in the frame store")
        lut = torch.full((int(ids.max()) + 1,), -1, dtype=torch.long)
        lut[torch.from_numpy(ids)] = torch.arange(len(ids), dtype=torch.long)
        self.register_buffer("id_to_row", lut)
        # Only for the `rows_for` bounds check and error messages -- the frames
        # themselves live in `frames`, memmapped, never a tensor buffer.
        self.register_buffer("device_ids", torch.from_numpy(ids))

        self.pool = CrossAttentionPool(frames.latent_dim, n_queries, d_model)
        hidden = int(proj_hidden) if int(proj_hidden) > 0 else max(1, self.pool.pooled_dim // 4)
        self.proj = nn.Sequential(nn.Linear(self.pool.pooled_dim, hidden),
                                  make_activation(self.activation),
                                  nn.Linear(hidden, self.embedding_dim))

    # --- Construction -------------------------------------------------------------
    @staticmethod
    def frames_sha256(path: "str | Path") -> str:
        """Content hash of meta.json, for checkpoint provenance.

        Not a hash of the 12.64 GiB of frame data itself: meta.json already
        records every device's normalization gain and the render manifest it
        was computed from, so a change to the underlying frames changes this
        hash too, at the cost of a blind spot for a shard corrupted without
        touching meta.json.
        """
        return hashlib.sha256((Path(path) / "meta.json").read_bytes()).hexdigest()

    @classmethod
    def from_config(cls, jcfg, embedding_dim: int) -> "FrameAttentionEncoder":
        """Mirrors FingerprintEncoder.from_config: ``embedding_dim`` comes from
        the *generator's* config, so the interface width has one source of truth."""
        store = FrameStore(jcfg.fp_frames_path)
        return cls(embedding_dim=embedding_dim, frames=store,
                   n_queries=jcfg.enc_pool_queries, d_model=jcfg.enc_pool_dim,
                   proj_hidden=jcfg.enc_proj_hidden, activation=jcfg.enc_activation,
                   normalize=jcfg.enc_normalize, frame_window=jcfg.enc_frame_window)

    # --- Forward ------------------------------------------------------------------
    def rows_for(self, device_ids: torch.Tensor) -> torch.Tensor:
        """Validates ids against the frame store, raising on anything unknown."""
        ids = device_ids.reshape(-1).long()
        n = self.id_to_row.numel()
        bad = (ids < 0) | (ids >= n)
        rows = torch.where(bad, torch.full_like(ids, -1),
                           self.id_to_row[ids.clamp(0, n - 1)])
        missing = bad | (rows < 0)
        if bool(missing.any()):
            unknown = sorted(set(ids[missing].tolist()))
            raise RuntimeError(
                f"no frames for device(s) {unknown}: this run's corpus and "
                f"{self.frames.root} disagree. A silent fallback here would condition "
                f"the generator on some other amp.")
        return ids

    def window_starts(self, n: int) -> list[int] | None:
        """Per-item first frame of the window, or ``None`` when windowing is off.

        Random while training (that randomness *is* the augmentation), the
        centre window at eval so a val/holdout number is reproducible epoch to
        epoch rather than re-rolled. Drawn from torch's global CPU generator,
        so ``cfg.seed`` reproduces a run.
        """
        if self.frame_window <= 0:
            return None
        span = self.frames.n_frames - self.frame_window
        if not self.training:
            return [span // 2] * n
        return torch.randint(0, span + 1, (n,)).tolist()

    def forward(self, ref: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B]`` device ids ``-> [B, embedding_dim]``.

        The frame gather happens here, on the calling thread, and it is this
        arm's throughput bottleneck -- see the module docstring.

        ``mask`` is accepted and ignored: it exists in the interface for the
        WaveNet encoder's variable-length references, and there is no
        reference here -- every device's frame count is fixed by extraction.
        """
        ids = self.rows_for(ref).tolist()
        frames = self.frames.batch(ids, ref.device,
                                   window=self.frame_window or None,
                                   starts=self.window_starts(len(ids))).float()
        e = self.proj(self.pool(frames))
        return F.normalize(e, dim=-1) if self.normalize else e

    # --- Reporting ----------------------------------------------------------------
    @property
    def pooled_dim(self) -> int:
        return int(self.pool.pooled_dim)

    @property
    def receptive_field(self) -> int:
        """The whole fingerprinted signal. Not a constraint on window A -- the
        frames are precomputed from a different file entirely."""
        return 0

    @property
    def n_devices(self) -> int:
        return int(self.device_ids.numel())

    def capacity_report(self) -> dict:
        return {"head": "dac_frames", "pooled_dim": self.pooled_dim,
                "embedding_dim": self.embedding_dim,
                "ceiling": min(self.pooled_dim, self.embedding_dim),
                "n_devices": self.n_devices,
                "n_queries": self.pool.q.shape[0], "d_model": self.pool.d_model,
                "frame_window": self.frame_window,
                "n_frames": int(self.frames.n_frames)}
