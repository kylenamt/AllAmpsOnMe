"""Conditioning from a frozen audio codec's fingerprint, looked up by device id.

Same interface contract as WaveNetEncoder (.embedding_dim, __call__(ref, mask),
.pooled_dim, .capacity_report()), so JointModel drives it unchanged. Only the
pooled vector's source differs: a frozen codec's pooled latent over the whole
capture signal, instead of a trained WaveNet backbone's pool over a short window.

Lookup, not on-the-fly encoding: a codec fingerprint needs long audio to be a
reliable statistic (DAC own-nearest accuracy drops from 99.8% at 85s to 17.8%
at 12s), well above a joint reference window's 2-6s. The fingerprint is
precomputed once per amp by scripts/encode_fingerprints.py and looked up here;
joint.ref_seconds and the reference-disjointness machinery are inert under
this encoder -- there is no reference window to draw.

The fingerprint is a buffer, not a parameter: nn.Module.parameters() returns
only the adapter, so the codec is frozen by construction and the joint
trainer's "encoder" optimizer group means exactly "the adapter".

Rows are keyed by device id over every amp in the npz, including held-out
ones -- not leakage, since the training dataset only ever draws training ids
and a buffer can't receive gradient. Keying by id (not the run's id_to_idx)
is what lets a held-out amp be conditioned at evaluation with no table swap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from openamp.core.config import ConfigError

__all__ = ["FingerprintEncoder", "FINGERPRINT_PREPROCESS", "DRY_ID"]

# Must match scripts/encode_fingerprints.py: the dry signal is fingerprinted as its
# own "device" and is the null every amp is measured against.
DRY_ID = -1

FINGERPRINT_PREPROCESS = ("dry_l2", "dry", "l2", "raw")


class FingerprintEncoder(nn.Module):
    """``[B] device ids -> [B, embedding_dim]``.

    The adapter deliberately copies ``WaveNetEncoder``'s projection head shape
    (a single ``Linear(pooled, De)``) so that this arm differs from a
    WaveNet-encoder arm in exactly one place: the provenance of the pooled
    vector.
    """

    ref_mode = "device"                                  # a device id to look up

    def __init__(self, *, embedding_dim: int, fingerprints: np.ndarray,
                 device_ids: np.ndarray, dry: np.ndarray,
                 preprocess: str = "dry_l2", normalize: bool = True,
                 layernorm: bool = False, eps: float = 1e-12,
                 grad_checkpoint: bool = False):
        super().__init__()
        # Accepted and ignored: this arm has no layer array whose activations
        # could be recomputed. Part of the contract so the knob stays uniform
        # across encoders rather than becoming a kind test in build_joint.
        del grad_checkpoint
        fp = np.asarray(fingerprints, dtype=np.float32)
        ids = np.asarray(device_ids, dtype=np.int64)
        dry_v = np.asarray(dry, dtype=np.float32).reshape(-1)
        if fp.ndim != 2:
            raise ValueError(f"fingerprints must be [N, D], got {fp.shape}")
        if len(ids) != len(fp):
            raise ValueError(f"{len(ids)} device ids for {len(fp)} fingerprints")
        if dry_v.shape[0] != fp.shape[1]:
            raise ValueError(
                f"dry vector is {dry_v.shape[0]}-d but fingerprints are {fp.shape[1]}-d")
        if preprocess not in FINGERPRINT_PREPROCESS:
            raise ValueError(f"preprocess must be one of {FINGERPRINT_PREPROCESS}, "
                             f"got {preprocess!r}")
        if ids.min() < 0:
            raise ValueError("device_ids must not contain the dry id; pass it as `dry`")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("duplicate device ids in the fingerprint set")

        self.embedding_dim = int(embedding_dim)
        self.preprocess = str(preprocess)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        self.register_buffer("fingerprints", torch.from_numpy(fp))
        self.register_buffer("dry", torch.from_numpy(dry_v))
        self.register_buffer("device_ids", torch.from_numpy(ids))
        # Dense id -> row table. A dict would be a Python-level lookup in the hot
        # path; more importantly a missing id must *raise*, because a negative index
        # into a tensor silently returns the last amp.
        lut = torch.full((int(ids.max()) + 1,), -1, dtype=torch.long)
        lut[torch.from_numpy(ids)] = torch.arange(len(ids), dtype=torch.long)
        self.register_buffer("id_to_row", lut)

        d_fp = fp.shape[1]
        layers: list[nn.Module] = []
        if layernorm:
            # Per-sample, learned affine, no corpus statistics -- composable with any
            # `preprocess`. Off by default: DAC's vector is mean(D/2) + std(D/2), two
            # halves on different scales, and one LayerNorm across the concatenation
            # mixes them.
            layers.append(nn.LayerNorm(d_fp))
        # An adapter, not a stage: one Linear whose only job is to change width
        # from the fingerprint's to De. No hidden layer, so nothing here narrows
        # the fingerprint before the generator sees it.
        layers.append(nn.Linear(d_fp, self.embedding_dim))
        self.proj = nn.Sequential(*layers)

    # --- Construction -------------------------------------------------------------
    @staticmethod
    def load_pooled(path: "str | Path") -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """``(device_ids, pooled, dry, meta)`` from a fingerprint run directory.

        The dry row (``device_id == DRY_ID``) is split out of the matrix rather than
        left in it: it is a reference, not an amp, and must never be retrievable as
        one.
        """
        run = Path(path)
        npz = run / "pooled.npz"
        if not npz.is_file():
            raise FileNotFoundError(
                f"no pooled.npz under {run} -- run scripts/encode_fingerprints.py first")
        with np.load(npz) as z:
            ids, pooled = np.asarray(z["device_ids"]), np.asarray(z["pooled"])
        dry_mask = ids == DRY_ID
        if not dry_mask.any():
            raise ValueError(f"{npz} has no dry row (device_id {DRY_ID}); it is the "
                             f"null every fingerprint is measured against")
        dry = pooled[dry_mask][0]
        meta_path = run / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        return ids[~dry_mask], pooled[~dry_mask], dry, meta

    @staticmethod
    def fingerprint_sha256(path: "str | Path") -> str:
        """Content hash of the pooled.npz, for checkpoint provenance."""
        return hashlib.sha256((Path(path) / "pooled.npz").read_bytes()).hexdigest()

    @classmethod
    def from_spec(cls, spec, embedding_dim: int, **extra) -> "FingerprintEncoder":
        """Mirrors ``WaveNetEncoder.from_spec``, but ``path`` is opened first: the
        matrix this conditions on comes off disk, not out of the config."""
        kwargs = dict(spec.kwargs)
        ids, pooled, dry, _ = cls.load_pooled(cls._path(kwargs.pop("path", "")))
        return cls(embedding_dim=embedding_dim, fingerprints=pooled, device_ids=ids,
                   dry=dry, **kwargs, **extra)

    @staticmethod
    def _path(path) -> str:
        if not str(path).strip():
            raise ConfigError("encoder kind 'fingerprint' needs joint.encoder.path, "
                              "the directory holding pooled.npz")
        return str(path)

    @classmethod
    def provenance(cls, spec) -> tuple[str, dict]:
        path = cls._path(spec.kwargs.get("path", ""))
        return cls.fingerprint_sha256(path), cls.load_pooled(path)[3]

    # --- Forward ------------------------------------------------------------------
    def preprocess_fp(self, fp: torch.Tensor) -> torch.Tensor:
        """``[B, D_fp] -> [B, D_fp]``. No learned parameters, no corpus statistics.

        Only the fixed dry vector and per-sample arithmetic are used. Per-dimension
        standardization over the amp population is deliberately not offered: it
        rescales each axis by how much this particular corpus happened to vary along
        it, which is a property of the sample rather than of an amp, and would measure
        a new amp in units derived from amps it has nothing to do with.
        """
        if self.preprocess in ("dry", "dry_l2"):
            fp = fp - self.dry
        if self.preprocess in ("l2", "dry_l2"):
            fp = F.normalize(fp, dim=-1, eps=self.eps)
        return fp

    def rows_for(self, device_ids: torch.Tensor) -> torch.Tensor:
        """Fingerprint rows for real device ids, raising on anything unknown."""
        ids = device_ids.reshape(-1).long()
        n = self.id_to_row.numel()
        bad = (ids < 0) | (ids >= n)
        # Clamp into range *before* indexing: an out-of-range id would raise
        # IndexError from the gather itself, and a negative one would quietly wrap
        # to the last amp. Both are caught by `bad` and reported together below.
        rows = torch.where(bad, torch.full_like(ids, -1),
                           self.id_to_row[ids.clamp(0, n - 1)])
        missing = bad | (rows < 0)
        if bool(missing.any()):
            unknown = sorted(set(ids[missing].tolist()))
            raise RuntimeError(
                f"no fingerprint for device(s) {unknown}: this run's corpus and the "
                f"fingerprint set disagree. A silent fallback here would condition the "
                f"generator on some other amp.")
        return rows

    def forward(self, ref: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B]`` device ids ``-> [B, embedding_dim]``.

        ``mask`` is accepted and ignored: it exists in the interface for the WaveNet
        encoder's variable-length references, and there is no reference here.
        """
        fp = self.fingerprints[self.rows_for(ref)]
        e = self.proj(self.preprocess_fp(fp))
        return F.normalize(e, dim=-1) if self.normalize else e

    # --- Reporting ----------------------------------------------------------------
    @property
    def pooled_dim(self) -> int:
        return int(self.fingerprints.shape[1])

    @property
    def receptive_field(self) -> int:
        """The whole fingerprinted signal. Not a constraint on window A — the
        fingerprint is precomputed from a different file entirely."""
        return 0

    @property
    def n_fingerprints(self) -> int:
        return int(self.fingerprints.shape[0])

    def capacity_report(self) -> dict:
        return {"head": "fingerprint", "pooled_dim": self.pooled_dim,
                "embedding_dim": self.embedding_dim,
                "ceiling": min(self.pooled_dim, self.embedding_dim),
                "n_fingerprints": self.n_fingerprints,
                "preprocess": self.preprocess}
