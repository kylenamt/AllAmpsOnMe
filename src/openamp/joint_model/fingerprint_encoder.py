"""Conditioning from a frozen audio codec's fingerprint, looked up by device id.

Same interface contract as :class:`~openamp.joint_model.wavenet_encoder.WaveNetEncoder`
— ``.embedding_dim``, ``__call__(ref, mask)``, ``.pooled_dim``, ``.capacity_report()``
— so :class:`~openamp.joint_model.model.JointModel` drives it unchanged. Only where
the pooled vector comes from changes: a frozen codec's pooled latent over the whole
171 s capture signal, instead of a trained WaveNet backbone's pool over a 2 s window.

**Why a lookup and not an on-the-fly encode.** A codec fingerprint is a statistic over
frames, and short audio makes it a noisy one. Measured on 450 amps with DAC, matching
an amp to itself against 449 competitors (sibling captures excluded):

    audio per side   85 s   43 s   22 s   12 s
    own-nearest      99.8%  88.7%  61.1%  17.8%

A joint reference window is 2-6 s, an order of magnitude below where this becomes
reliable, so encoding one on the fly would condition the generator on noise. The
fingerprint is therefore precomputed once per amp from the full capture render by
``scripts/encode_fingerprints.py`` and looked up here. ``joint.ref_seconds`` and the
reference-disjointness machinery are inert under this encoder — there is no reference
window to draw.

**The fingerprint is a buffer, not a parameter.** ``nn.Module.parameters()`` therefore
returns only the adapter, which is what makes the codec frozen by construction rather
than by a ``requires_grad_(False)`` somebody can forget, and what makes the joint
trainer's ``"encoder"`` optimizer group mean exactly "the adapter".

**Rows are keyed by device id, over every amp in the npz — including held-out ones.**
That is deliberate and it is not leakage: the training dataset only ever draws training
ids, and a buffer cannot receive gradient in any case. Keying by id rather than by the
run's ``id_to_idx`` row is what lets a held-out amp be conditioned at evaluation with no
table swap (contrast ``emulate.enroll._swap_embedding``, which must install a new table
because trained rows exist only for seen devices — here nothing is per-device trained,
so there is nothing to swap).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from openamp.emulate.wavenet import make_activation, normalize_activation

__all__ = ["FingerprintEncoder", "FINGERPRINT_PREPROCESS", "DRY_ID"]

# Must match scripts/encode_fingerprints.py: the dry signal is fingerprinted as its
# own "device" and is the null every amp is measured against.
DRY_ID = -1

FINGERPRINT_PREPROCESS = ("dry_l2", "dry", "l2", "raw")


class FingerprintEncoder(nn.Module):
    """``[B] device ids -> [B, embedding_dim]``.

    The adapter deliberately copies ``WaveNetEncoder``'s projection head shape
    (``Linear(pooled, h) -> act -> Linear(h, De)``, ``proj_hidden=0`` deriving as
    ``pooled_dim // 4``) so that this arm differs from a WaveNet-encoder arm in
    exactly one place: the provenance of the pooled vector.
    """

    def __init__(self, *, embedding_dim: int, fingerprints: np.ndarray,
                 device_ids: np.ndarray, dry: np.ndarray,
                 preprocess: str = "dry_l2", proj_hidden: int = 128,
                 activation: str = "tanh", normalize: bool = True,
                 layernorm: bool = False, eps: float = 1e-12):
        super().__init__()
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
        self.activation = normalize_activation(activation, field="enc_activation")

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
        hidden = int(proj_hidden) if int(proj_hidden) > 0 else max(1, d_fp // 4)
        layers: list[nn.Module] = []
        if layernorm:
            # Per-sample, learned affine, no corpus statistics -- composable with any
            # `preprocess`. Off by default: DAC's vector is mean(D/2) + std(D/2), two
            # halves on different scales, and one LayerNorm across the concatenation
            # mixes them.
            layers.append(nn.LayerNorm(d_fp))
        layers += [nn.Linear(d_fp, hidden), make_activation(self.activation),
                   nn.Linear(hidden, self.embedding_dim)]
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
    def from_config(cls, jcfg, embedding_dim: int) -> "FingerprintEncoder":
        """Mirrors ``WaveNetEncoder.from_config``: ``embedding_dim`` comes from the
        *generator's* config, so the interface width has one source of truth."""
        ids, pooled, dry, _ = cls.load_pooled(jcfg.fp_path)
        return cls(embedding_dim=embedding_dim, fingerprints=pooled, device_ids=ids,
                   dry=dry, preprocess=jcfg.fp_preprocess,
                   proj_hidden=jcfg.enc_proj_hidden, activation=jcfg.enc_activation,
                   normalize=jcfg.enc_normalize, layernorm=jcfg.fp_layernorm)

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
