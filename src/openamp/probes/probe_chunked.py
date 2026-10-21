"""A chunked Conformer encoder on the plain complex spectrogram, trained first by a
contrastive loss to tell real amps apart.

**Input.** One STFT (``N_FFT`` 2048, ``HOP`` 512, Hann, centred) of dry and of
wet. Every frame carries dry re, dry im, wet re, wet im of all 1025 linear bins
-- 4100 numbers, no log, no compression, no scaling -- so the network can
compare dry and wet loudness and phase at each frequency. No mel, no raw-audio
branch.

**Chunks.** One frame sees ``N_FFT`` = 2048 samples, its receptive field (RF);
that is 4 frames. A chunk is ``n`` RF (4n frames), neighbours overlap by
0.25 n RF (n frames), so a new chunk starts every 3n frames. The last chunk is
pulled back to end where the audio ends: nothing dropped, nothing padded. At
``n`` = 47 a chunk is 96,256 samples (2.005 s) and 84.4 s gives 56 chunks.

**Chunk encoder**, the same weights for every chunk: Linear 4100 -> width,
Conformer blocks, mean over frames -> ``h``. Beside it, the chunk's dB
spectrum (dry ‖ wet, 2050 bins), standardized by a fixed per-bin mean / std,
averaged over frames -> ``s``.

**Pool over chunks**: multi-head attention with one learned query. Keys from
``s`` through a small MLP, values ``W_v h``. Each head's weights are a softmax
over the chunks; the head returns the weighted mean and weighted std of its
values (variance floored at ``VAR_FLOOR``). No ``W_q`` (the query is already
learned) and no ``W_o`` (the next Linear does its job). Linear 2·width ->
embedding, L2 normalize.

**Phase 1: contrastive.** Each step draws one length ``k`` (1 to ``K_max``
chunks, uniform; ``K_max`` chunks is ``--max-seconds``) and two crops A and B of
the reference that never overlap. The same two crops for every amp in the step,
so the music is identical across amps and only the amp tells a match from a
miss. InfoNCE both ways on cosine / ``--tau``: each amp's A must pick its own B
among the step's amps, and B its A. No projection head. A step holds
``--batch`` real amps, played online by their ``.nam`` at the ``output_scale``
their corpus renders carried, plus ``--synth-per-batch`` made-up amps.

**Made-up amps** (:class:`ShapedCloud`): an embedding drawn from a Gaussian with
the training table rows' mean and covariance, the covariance's eigenvalues
raised to ``--synth-shape`` (1 = the rows' own shape, 0 = the same spread in
every direction; the total variance is kept), then scaled to the rows' length.
The frozen generator (``--run``) plays both crops, in bf16, with real left
context. Each one is drawn once and never reused.

**Learning rate**: linear warmup from 0 over ``--warmup-steps``, then a cosine
down to ``--min-lr-frac`` of ``--lr`` at the last step. A pure function of the
step, so a resumed run continues the same curve.

**Memory.** Chunks are encoded in checkpointed groups of ``--ckpt-group``: only
each group's outputs are kept, and its inside is recomputed in the backward
pass. Same result, more compute.

**Validation**: every val amp, A = the reference's first ``--max-seconds``,
B = its last. ``best_val`` is the InfoNCE loss there; top-1 retrieval is
reported both ways (chance is 1 / val amps). The same is reported, never used
to pick the checkpoint, for ``--synth-val`` fixed made-up amps from the same
sampler.

**Resuming.** Every draw is seeded by ``(seed, stream, step)``, so
``--resume <out>/last.pt`` continues the identical stream.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from openamp.core.config import load_config
from openamp.corpus.render import _render_signal
from openamp.dsp import audio as audio_io
from openamp.dsp import nam as nam_io
from openamp.evaluation import load_model, resolve_device
from openamp.models import count_parameters
from openamp.probes.probe_conformer import ConformerBlock
from openamp.probes.probe_synth import (DB_EPS, HOP, N_FFT, SKIP_SECONDS, captures_by_sha,
                                        save_atomic, step_rng)
from openamp.training import GRAD_CLIP_NORM

N_BINS = N_FFT // 2 + 1         # 1025 linear bins, 23.4 Hz apart at 48 kHz
VAR_FLOOR = 1e-5                # pool variance floor, so the std's gradient stays finite
VAL_GROUP = 8                   # val crops encoded at once

RESUME_FIXED = ("run", "sweep", "captures", "renders", "seed", "val_frac", "batch",
                "steps", "max_seconds", "tau", "lr", "warmup_steps", "min_lr_frac",
                "weight_decay", "embedding_dim", "chunk_n",
                "width", "n_blocks", "n_heads", "ff_ratio", "conv_kernel", "pool_heads",
                "key_hidden", "synth_per_batch", "synth_shape", "synth_val")


# --- Chunks ---------------------------------------------------------------------------
def chunk_geometry(chunk_n: int) -> tuple[int, int]:
    """``(chunk, step)`` in samples: a chunk is ``chunk_n`` RF = 4n frames, the
    overlap 0.25n RF = n frames, so chunks start 3n frames apart."""
    return chunk_n * N_FFT, 3 * chunk_n * HOP


def chunk_starts(n: int, chunk: int, step: int) -> list[int]:
    """Start sample of every chunk of an ``n``-sample signal; the last one ends at ``n``."""
    if n < chunk:
        raise ValueError(f"signal has {n} samples, a chunk needs {chunk}")
    starts = list(range(0, n - chunk + 1, step))
    if starts[-1] + chunk < n:
        starts.append(n - chunk)
    return starts


def crop_samples(k: int, chunk: int, step: int) -> int:
    """Length of a crop that ``k`` chunks tile exactly."""
    return chunk + (k - 1) * step


# --- Encoder --------------------------------------------------------------------------
class ChunkedEncoder(nn.Module):
    """``(dry [B, n], wet [B, n]) -> [B, embedding_dim]`` unit vectors."""

    def __init__(self, embedding_dim: int, *, chunk_n: int, width: int, n_blocks: int,
                 n_heads: int, ff_ratio: int, conv_kernel: int, pool_heads: int,
                 key_hidden: int):
        super().__init__()
        if width % pool_heads:
            raise ValueError(f"width {width} is not divisible by {pool_heads} pool heads")
        self.chunk, self.step = chunk_geometry(chunk_n)
        self.pool_heads = int(pool_heads)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("key_mean", torch.zeros(2 * N_BINS))
        self.register_buffer("key_std", torch.ones(2 * N_BINS))
        self.inp = nn.Linear(4 * N_BINS, width)
        self.blocks = nn.ModuleList(ConformerBlock(width, n_heads, ff_ratio, conv_kernel)
                                    for _ in range(n_blocks))
        self.w_v = nn.Linear(width, width)
        self.key_mlp = nn.Sequential(nn.Linear(2 * N_BINS, key_hidden), nn.GELU(),
                                     nn.Linear(key_hidden, width))
        self.query = nn.Parameter(0.02 * torch.randn(width))
        self.out = nn.Linear(2 * width, embedding_dim)
        self.ckpt_group = 0         # > 0: encode chunks in checkpointed groups of this many

    def spectra(self, x: torch.Tensor) -> torch.Tensor:
        """``[N, m]`` -> ``[N, T, N_BINS]`` complex, frame ``t`` centred on sample ``t * HOP``."""
        return torch.stft(x.float(), N_FFT, HOP, window=self.window, center=True,
                          return_complex=True).transpose(1, 2)

    def db(self, X: torch.Tensor) -> torch.Tensor:
        return 10.0 * torch.log10(X.real ** 2 + X.imag ** 2 + DB_EPS)

    def chunk_features(self, dry: torch.Tensor,
                       wet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[N, chunk]`` dry and wet -> ``h [N, width]`` and ``s [N, 2 * N_BINS]``."""
        with torch.autocast(dry.device.type, enabled=False):
            X, Y = self.spectra(dry), self.spectra(wet)
            z = torch.cat([X.real, X.imag, Y.real, Y.imag], dim=-1)
            s = ((torch.cat([self.db(X), self.db(Y)], dim=-1) - self.key_mean)
                 / self.key_std).mean(1)
        h = self.inp(z)
        for block in self.blocks:
            h = block(h)
        return h.mean(1), s

    def pool(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """``h [B, K, width]``, ``s [B, K, 2 * N_BINS]`` -> ``[B, 2 * width]``."""
        B, K, d = h.shape
        H, hd = self.pool_heads, d // self.pool_heads
        v = self.w_v(h).view(B, K, H, hd)
        k = self.key_mlp(s).view(B, K, H, hd)
        with torch.autocast(h.device.type, enabled=False):
            v, k = v.float(), k.float()
            score = torch.einsum("bkhd,hd->bkh", k, self.query.float().view(H, hd))
            w = torch.softmax(score / math.sqrt(hd), dim=1)              # over the chunks
            mu = torch.einsum("bkh,bkhd->bhd", w, v)
            var = torch.einsum("bkh,bkhd->bhd", w, v * v) - mu * mu
            sigma = var.clamp_min(VAR_FLOOR).sqrt()
        return torch.cat([mu.flatten(1), sigma.flatten(1)], dim=-1)

    def forward(self, dry: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
        B, n = wet.shape
        starts = torch.as_tensor(chunk_starts(n, self.chunk, self.step), device=wet.device)
        idx = starts[:, None] + torch.arange(self.chunk, device=wet.device)    # [K, chunk]
        K = len(starts)
        dry_c = dry[:, idx].reshape(B * K, self.chunk)
        wet_c = wet[:, idx].reshape(B * K, self.chunk)
        g = self.ckpt_group
        if g and torch.is_grad_enabled():
            parts = [checkpoint(self.chunk_features, dry_c[i:i + g], wet_c[i:i + g],
                                use_reentrant=False) for i in range(0, B * K, g)]
            h, s = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        else:
            h, s = self.chunk_features(dry_c, wet_c)
        e = self.out(self.pool(h.view(B, K, -1), s.view(B, K, -1)))
        return F.normalize(e.float(), dim=-1)


# --- Training -------------------------------------------------------------------------
class ShapedCloud:
    """Random embeddings around a set of table rows: a Gaussian with the rows' mean
    and covariance, the covariance's eigenvalues raised to ``shape`` and rescaled to
    the same total (1 = the rows' own shape, 0 = equal in every direction); each
    draw is then scaled to ``radius``, the length every row of an
    ``embed_normalize`` table has."""

    def __init__(self, rows: np.ndarray, radius: float, shape: float):
        rows = np.asarray(rows, dtype=np.float64)
        self.mu = rows.mean(0)
        lam, self.V = np.linalg.eigh(np.cov(rows.T))
        lam = lam.clip(min=0.0)
        sc = lam ** shape
        self.sd = np.sqrt(sc * lam.sum() / sc.sum())
        self.radius = float(radius)

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        e = self.mu + (rng.standard_normal((n, len(self.mu))) * self.sd) @ self.V.T
        e = self.radius * e / np.linalg.norm(e, axis=1, keepdims=True)
        return e.astype(np.float32)


def crop_pair(rng: np.random.Generator, n: int, L: int) -> tuple[int, int]:
    """Starts ``(a, b)`` of two ``L``-sample crops of ``[0, n)`` that do not overlap,
    uniform over all such pairs; which one is A is a coin flip."""
    free = n - 2 * L
    if free < 0:
        raise ValueError(f"two {L}-sample crops do not fit side by side in {n} samples")
    u1, u2 = np.sort(rng.integers(0, free + 1, size=2))
    first, second = int(u1), int(u2) + L
    return (first, second) if rng.random() < 0.5 else (second, first)


def lr_factor(step: int, steps: int, warmup: int, min_frac: float) -> float:
    """Linear warmup over ``warmup`` steps, then a cosine to ``min_frac`` at ``steps``."""
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(steps - warmup - 1, 1)
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def info_nce(ea: torch.Tensor, eb: torch.Tensor, tau: float) -> tuple[torch.Tensor, dict]:
    """InfoNCE both ways: row ``i`` of ``ea`` must pick row ``i`` of ``eb``, and back."""
    logits = ea.float() @ eb.float().T / tau
    target = torch.arange(len(ea), device=ea.device)
    loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
    with torch.no_grad():
        top1_ab = float((logits.argmax(1) == target).float().mean())
        top1_ba = float((logits.argmax(0) == target).float().mean())
    return loss, {"top1_ab": top1_ab, "top1_ba": top1_ba}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2_n10"),
                    help="emulate run: its table gives the amp list, the train/val "
                         "split (probe_synth's) and the made-up draws' shape; its "
                         "frozen generator plays the made-up amps")
    ap.add_argument("--sweep", type=Path,
                    default=Path("data_sweep_a1/clean/train/sweep_train.flac"),
                    help="the reference: the T3K capture signal's train region")
    ap.add_argument("--captures", type=Path, default=Path("data/captures"))
    ap.add_argument("--renders", type=Path, default=Path("data/manifests/renders.parquet"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/encoder/encoder_probe_chunked"))
    ap.add_argument("--batch", type=int, default=32, help="real amps per step")
    ap.add_argument("--synth-per-batch", type=int, default=0,
                    help="made-up amps per step, on top of --batch")
    ap.add_argument("--synth-shape", type=float, default=1.0,
                    help="made-up draws: covariance eigenvalues ** this (1 = the "
                         "table rows' shape, 0 = equal in every direction)")
    ap.add_argument("--synth-val", type=int, default=76,
                    help="fixed made-up amps scored beside the real val amps")
    ap.add_argument("--gen-chunk", type=int, default=4,
                    help="made-up amps the generator plays at once")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=21,
                    help="steps between validations (21 = about one pass over the "
                         "training amps at 32 per step)")
    ap.add_argument("--max-seconds", type=float, default=84.4,
                    help="longest crop; also the val crop length")
    ap.add_argument("--tau", type=float, default=0.1, help="InfoNCE temperature, fixed")
    ap.add_argument("--embedding-dim", type=int, default=512)
    ap.add_argument("--chunk-n", type=int, default=47,
                    help="chunk = n x 2048 samples (47 = 2.005 s), overlap n/4 of that")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ff-ratio", type=int, default=4)
    ap.add_argument("--conv-kernel", type=int, default=15,
                    help="Conformer depthwise conv, in frames (15 = 160 ms)")
    ap.add_argument("--pool-heads", type=int, default=8)
    ap.add_argument("--key-hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4, help="peak learning rate")
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--min-lr-frac", type=float, default=0.01,
                    help="the cosine ends at this fraction of --lr")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--ckpt-group", type=int, default=112,
                    help="chunks per checkpointed group (memory only; same result)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", type=Path, default=None,
                    help="last.pt of a stopped run: continue its exact stream to --steps")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(resolve_device(args.device))
    cfg = load_config()
    args.out.mkdir(parents=True, exist_ok=True)
    arg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    ck = None
    if args.resume is not None:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        diff = {k: (ck["args"][k], arg_dict[k]) for k in RESUME_FIXED
                if ck["args"][k] != arg_dict[k]}
        if diff:
            raise SystemExit(f"--resume: these settings differ from the run being "
                             f"resumed: {diff}")

    # --- Amps and the split ---------------------------------------------------------
    tbl = torch.load(args.run / "embedding.pt", map_location="cpu", weights_only=False)
    ids = np.array([int(d) for d in tbl["device_ids"]])
    rend = pd.read_parquet(args.renders, columns=["device_id", "output_scale", "nam_sha256"])
    per_dev = rend.groupby("device_id").agg(scale=("output_scale", "first"),
                                            n_scale=("output_scale", "nunique"),
                                            sha=("nam_sha256", "first"))
    if (per_dev["n_scale"] > 1).any():
        raise SystemExit(f"{args.renders}: an amp with more than one output_scale")
    scales = {int(d): float(s) for d, s in per_dev["scale"].items()}
    found = captures_by_sha(args.captures, set(per_dev["sha"]))
    nam_paths = {int(d): found[h] for d, h in per_dev["sha"].items() if h in found}
    missing = [int(d) for d in ids if d not in nam_paths]
    if missing:
        raise SystemExit(f"no .nam under {args.captures} for {len(missing)} table "
                         f"amp(s), e.g. {missing[:5]}")

    # probe_synth's split, exactly: permutation(len(ids)) from a fresh default_rng(seed).
    perm = np.random.default_rng(args.seed).permutation(len(ids))
    n_val = max(1, int(round(args.val_frac * len(ids))))
    val_sel, train_sel = perm[:n_val], perm[n_val:]
    val_ids = ids[val_sel]
    if args.batch > len(train_sel):
        raise SystemExit(f"--batch {args.batch} > {len(train_sel)} training amps")

    # --- Reference and crop lengths -------------------------------------------------
    x, sr = audio_io.read_audio(args.sweep)
    x = np.asarray(x, dtype=np.float32)
    if sr != cfg.sample_rate:
        raise SystemExit(f"{args.sweep} is {sr} Hz, the corpus is {cfg.sample_rate} Hz")
    skip = int(round(SKIP_SECONDS * sr))
    n_keep = len(x) - skip
    L_max = int(round(args.max_seconds * sr))
    chunk, cstep = chunk_geometry(args.chunk_n)
    k_max = len(chunk_starts(L_max, chunk, cstep))
    if 2 * L_max > n_keep:
        raise SystemExit(f"two {args.max_seconds:g} s crops do not fit side by side in "
                         f"the reference's {n_keep / sr:.1f} s")

    # --- Frozen generator and the made-up draws -------------------------------------
    model, _ = load_model(args.run, str(dev))
    model.requires_grad_(False)
    model.eval()
    R = model.receptive_field
    table = tbl["embedding"].float().numpy()
    cloud = ShapedCloud(table[train_sel], float(np.linalg.norm(table, axis=1).mean()),
                        args.synth_shape)
    x_pad_gen = torch.from_numpy(np.concatenate([np.zeros(R, np.float32), x])).to(dev)

    @torch.no_grad()
    def synth_crops(E: torch.Tensor, g: int, L: int) -> torch.Tensor:
        """``[S, De]`` -> ``[S, L]``: the generator on ``x[g : g + L]`` with ``R``
        samples of real left context."""
        seg = x_pad_gen[g:g + R + L]
        out = []
        for s in range(0, len(E), args.gen_chunk):
            e = E[s:s + args.gen_chunk]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(model.forward_emb(seg[None, None].expand(len(e), -1, -1),
                                             e)[:, 0, R:].float())
        return torch.cat(out)

    def crop_len(k: int) -> int:
        return L_max if k == k_max else crop_samples(k, chunk, cstep)

    nams = {}
    t0 = time.time()
    for d in ids:
        nams[int(d)] = nam_io.load_model(nam_paths[int(d)], device=str(dev))
        if nams[int(d)].sample_rate != cfg.sample_rate:
            raise SystemExit(f"amp {d}: .nam at {nams[int(d)].sample_rate} Hz; crop "
                             f"rendering assumes {cfg.sample_rate}")
    R_nam = max(m.receptive_field for m in nams.values())
    x_pad_nam = np.concatenate([np.zeros(R_nam, np.float32), x])
    print(f"[chunk] loaded {len(nams)} .nam captures ({time.time() - t0:.0f} s); "
          f"largest receptive field {R_nam}", flush=True)

    def real_full(d: int) -> torch.Tensor:
        """``[n_keep]`` the amp's ``.nam`` on the whole reference, at its learned level."""
        y = _render_signal(nams[int(d)], x, cfg, False) * scales[int(d)]
        return torch.from_numpy(y[skip:].astype(np.float32))

    def real_crop(d: int, g: int, L: int) -> np.ndarray:
        """The amp on ``x[g : g + L]``, with its own receptive field of real context."""
        m = nams[int(d)]
        seg = x_pad_nam[R_nam - m.receptive_field + g: R_nam + g + L]
        return (m.forward(seg)[-L:] * scales[int(d)]).astype(np.float32)

    # --- Encoder --------------------------------------------------------------------
    encoder = ChunkedEncoder(
        args.embedding_dim, chunk_n=args.chunk_n, width=args.width, n_blocks=args.n_blocks,
        n_heads=args.n_heads, ff_ratio=args.ff_ratio, conv_kernel=args.conv_kernel,
        pool_heads=args.pool_heads, key_hidden=args.key_hidden).to(dev)
    dry_full = torch.from_numpy(x[skip:]).to(dev)

    # The key standardization: dB per bin over the reference's frames, dry from the
    # reference, wet from the real training amps. Val amps are rendered and kept.
    if ck is None:
        t0 = time.time()
        s1 = torch.zeros(N_BINS, dtype=torch.float64)
        s2 = torch.zeros_like(s1)
        n_frames = 0
        with torch.no_grad():
            dry_db = encoder.db(encoder.spectra(dry_full[None]))[0].double().cpu()
            for i, j in enumerate(train_sel):
                w = encoder.db(encoder.spectra(real_full(ids[j]).to(dev)[None]))[0]
                w = w.double().cpu()
                s1 += w.sum(0)
                s2 += (w * w).sum(0)
                n_frames += w.shape[0]
                if (i + 1) % 100 == 0 or i + 1 == len(train_sel):
                    print(f"[chunk] rendered {i + 1}/{len(train_sel)} training amps on the "
                          f"reference ({time.time() - t0:.0f} s)", flush=True)
        wet_mean = s1 / n_frames
        wet_std = (s2 / n_frames - wet_mean ** 2).clamp_min(0.0).sqrt()
        encoder.key_mean.copy_(torch.cat([dry_db.mean(0), wet_mean]).float())
        encoder.key_std.copy_(torch.cat([dry_db.std(0), wet_std]).float().clamp_min(1e-3))
    wet_val = [real_full(d) for d in val_ids]
    wet_val_a = torch.stack([w[:L_max] for w in wet_val])                   # [V, L_max]
    wet_val_b = torch.stack([w[-L_max:] for w in wet_val])
    del wet_val
    if args.synth_val:
        E_sv = torch.from_numpy(cloud.draw(step_rng(args.seed, 6, 0), args.synth_val)).to(dev)
        synth_val_a = synth_crops(E_sv, skip, L_max).cpu()
        synth_val_b = synth_crops(E_sv, skip + n_keep - L_max, L_max).cpu()
    print(f"[chunk] rendered {len(val_ids)} val amps and {args.synth_val} made-up val "
          f"amps on the reference", flush=True)

    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best_val, best_step, best_state, best_metrics = 0, float("inf"), -1, None, {}
    if ck is not None:
        encoder.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start, best_val, best_step = ck["step"] + 1, ck["best_val"], ck["best_step"]
        best_state, best_metrics = ck["best_state"], ck["best_metrics"]
        print(f"[chunk] resumed {args.resume} at step {start} (best val {best_val:.4f} "
              f"at step {best_step + 1})")
    encoder.ckpt_group = args.ckpt_group
    n_params = count_parameters(encoder)
    print(f"[chunk] {len(ids)} amps: {len(train_sel)} train / {len(val_sel)} val  "
          f"reference {n_keep / sr:.1f} s after the skip")
    print(f"[chunk] per step: {args.batch} real + {args.synth_per_batch} made-up amps "
          f"(shape {args.synth_shape:g}) x 2 crops that do not overlap, "
          f"k = 1..{k_max} chunks of {chunk} samples ({chunk / sr:.3f} s, step "
          f"{cstep / sr:.3f} s), longest {args.max_seconds:g} s; InfoNCE tau {args.tau:g}")
    print(f"[chunk] encoder: complex STFT 4x{N_BINS} -> {args.width} -> {args.n_blocks} "
          f"conformer/{args.n_heads}h k{args.conv_kernel} -> mean over frames -> pool "
          f"{args.pool_heads} heads (keys from dB, {args.key_hidden} hidden) -> "
          f"{args.embedding_dim}   {n_params:,} params", flush=True)

    @torch.no_grad()
    def encode(dry: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
        """``[n]`` dry on the GPU, ``[N, n]`` wet on the CPU -> ``[N, De]``."""
        encoder.eval()
        out = []
        for s in range(0, len(wet), VAL_GROUP):
            w = wet[s:s + VAL_GROUP].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(encoder(dry[None].expand(len(w), -1), w))
        return torch.cat(out)

    def validate() -> dict:
        ea = encode(dry_full[:L_max], wet_val_a)
        eb = encode(dry_full[-L_max:], wet_val_b)
        loss, m = info_nce(ea, eb, args.tau)
        out = {"loss": float(loss), **m}
        if args.synth_val:
            loss, m = info_nce(encode(dry_full[:L_max], synth_val_a),
                               encode(dry_full[-L_max:], synth_val_b), args.tau)
            out.update({"synth_loss": float(loss), "synth_top1_ab": m["top1_ab"],
                        "synth_top1_ba": m["top1_ba"]})
        return out

    def checkpoint_state(step: int) -> dict:
        return {"state_dict": encoder.state_dict(), "opt": opt.state_dict(), "step": step,
                "best_val": best_val, "best_step": best_step, "best_state": best_state,
                "best_metrics": best_metrics, "args": arg_dict, "device_ids": ids.tolist()}

    # --- Train ----------------------------------------------------------------------
    perms: dict[int, np.ndarray] = {}

    def step_amps(step: int) -> list[int]:
        """Consecutive slices of one seeded permutation of the training amps per pass."""
        out = []
        for p in range(step * args.batch, (step + 1) * args.batch):
            ep, j = divmod(p, len(train_sel))
            if ep not in perms:
                perms[ep] = step_rng(args.seed, 1, ep).permutation(train_sel)
            out.append(int(ids[perms[ep][j]]))
        return out

    acc = {"loss": 0.0, "top1_ab": 0.0, "top1_ba": 0.0, "n": 0}
    t0, t_render = time.time(), 0.0
    for step in range(start, args.steps):
        encoder.train()
        lr = args.lr * lr_factor(step, args.steps, args.warmup_steps, args.min_lr_frac)
        for group in opt.param_groups:
            group["lr"] = lr
        rng = step_rng(args.seed, 3, step)
        k = int(rng.integers(1, k_max + 1))
        L = crop_len(k)
        a, b = crop_pair(rng, n_keep, L)
        amps = step_amps(step)
        tr0 = time.time()
        S = args.synth_per_batch
        E = (torch.from_numpy(cloud.draw(step_rng(args.seed, 2, step), S)).to(dev)
             if S else None)
        views = []
        for g in (a, b):
            views.append(torch.from_numpy(np.stack([real_crop(d, skip + g, L)
                                                    for d in amps])).to(dev))
            if S:
                views.append(synth_crops(E, skip + g, L))
        wet = torch.cat(views)
        del views
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_render += time.time() - tr0
        B = args.batch + S
        dry = torch.cat([dry_full[a:a + L][None].expand(B, -1),
                         dry_full[b:b + L][None].expand(B, -1)])

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            e = encoder(dry, wet)
        loss, m = info_nce(e[:B], e[B:], args.tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP_NORM)
        opt.step()
        acc["loss"] += float(loss)
        acc["top1_ab"] += m["top1_ab"]
        acc["top1_ba"] += m["top1_ba"]
        acc["n"] += 1
        del wet, dry, e, loss

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            v = validate()
            star = v["loss"] < best_val - 1e-6
            if star:
                best_val, best_step, best_metrics = v["loss"], step, v
                best_state = {kk: vv.detach().clone() for kk, vv in
                              encoder.state_dict().items()}
            n = max(acc["n"], 1)
            elapsed = time.time() - t0
            print(f"  s{step + 1:05d}  train loss {acc['loss'] / n:.4f} top1 "
                  f"{acc['top1_ab'] / n:.3f}/{acc['top1_ba'] / n:.3f}  val loss "
                  f"{v['loss']:.4f} top1 A>B {v['top1_ab']:.3f} B>A {v['top1_ba']:.3f}  "
                  + (f"made-up val loss {v['synth_loss']:.4f} top1 {v['synth_top1_ab']:.3f}/"
                     f"{v['synth_top1_ba']:.3f}  " if args.synth_val else "") +
                  f"best {best_val:.4f}  lr {lr:.2e}  {elapsed / 60:.1f} min, render "
                  f"{t_render / max(elapsed, 1e-9):.0%}" + ("  *" if star else ""),
                  flush=True)
            acc = {kk: 0 if kk == "n" else 0.0 for kk in acc}
            save_atomic(checkpoint_state(step), args.out / "last.pt")
    train_min = (time.time() - t0) / 60
    print(f"[chunk] trained to step {args.steps} in {train_min:.1f} min; best val "
          f"{best_val:.4f} at step {best_step + 1}")

    report = {
        "n_params": int(n_params), "n_train": int(len(train_sel)), "n_val": int(len(val_sel)),
        "steps": int(args.steps), "batch": int(args.batch),
        "synth_per_batch": int(args.synth_per_batch), "synth_shape": args.synth_shape,
        "tau": args.tau,
        "max_seconds": args.max_seconds, "k_max": int(k_max),
        "best_val_loss": best_val, "best_step": int(best_step + 1),
        "val_top1_ab": best_metrics.get("top1_ab"), "val_top1_ba": best_metrics.get("top1_ba"),
        "chance_top1": 1.0 / len(val_sel), "train_minutes": train_min,
        "synth_val": int(args.synth_val), "val_synth_loss": best_metrics.get("synth_loss"),
        "val_synth_top1_ab": best_metrics.get("synth_top1_ab"),
        "val_synth_top1_ba": best_metrics.get("synth_top1_ba"),
    }
    torch.save({"state_dict": encoder.state_dict(), "best_state": best_state,
                "args": arg_dict, "device_ids": ids.tolist(),
                "val_ids": val_ids.tolist(), "report": report},
               args.out / "probe_chunked.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[chunk] wrote {args.out}/probe_chunked.pt and report.json")


if __name__ == "__main__":
    main()
