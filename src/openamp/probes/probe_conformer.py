"""A raw-audio + log-mel Conformer encoder, trained by audio loss through the frozen decoder.

    OPENAMP_DATA_DIR=data python -m openamp.probes.probe_conformer \\
        --out results/encoder/encoder_probe_conformer

then ``python -m openamp.probes.probe_esr --probe <out>/probe_conformer.pt`` to score it.

Two changes from :mod:`openamp.probes.probe_synth`, made together:

- **Audio loss, not cosine.** No table row is a target. The encoder hears the
  amp play the reference, emits ``e``, and the frozen generator plays a
  *different* clip (EGDB guitar DI) with that ``e``; the loss is the decoder's
  own training objective (``esr_mrstft``, the run's ``loss:`` section) against
  the amp's real output on that clip. The two signals differ so that ``e`` can
  only carry the amp, not the clip.
- **A new encoder** (:class:`ConformerEncoder`). The old one read each 11 ms
  log-mel token alone, as an unordered set: it saw how loud each band got, never
  the wave shape and never a token's neighbours.

**Amps per step.** ``--batch`` amps, ``--synth-per-batch`` of them made up:
an embedding drawn as probe_synth draws it (:class:`TableCloud`, the training
rows' mean and covariance -- the table shapes the draw, it is never a target),
whose reference and clip are both played by the frozen generator, online, never
reused. The rest are real training amps, a seeded permutation per pass; their
reference and clip are both played by their ``.nam``, scaled by the
``output_scale`` their corpus renders carried -- which *is* how ``openamp
render`` made those renders, so the targets match the stored files (ESR ~1e-7,
their quantization) without reading them. Only clean audio is read from disk.

**One reference crop per step**, ``--ref-seconds`` long, at a random start in
the reference after its first ``SKIP_SECONDS``, shared by every amp in the step.
Sharing it is what makes the per-amp loss weight below well defined.

**Per-amp loss weight.** The decoder trained with ``device_loss_weighting``
(ESR weight ``power ** -strength`` per amp, from the manifest). A made-up amp
has no manifest row, so here every amp's power is its wet power on the shared
reference crop -- the same content for all, so the ratio between two amps is
their level difference. Weights are normalized to mean 1 per batch and reach the
ESR term only, as in :func:`openamp.losses.device_loss_weights`.

**Validation** reads the encoder over each val amp's *whole* reference (the
scoring condition) and scores the generator on ``--val-clips`` fixed EGDB
val-split clips per amp. ``best_val`` is the mean loss there; it picks the
checkpoint whose predictions are written.

**Output** is what :mod:`openamp.probes.probe_esr` reads from a probe_synth run:
``pred`` (every table amp, the encoder over its whole reference, rescaled to the
table's radius) and ``pred_sim`` (the val amps, hearing the generator's render
of their own row). The split is probe_synth's, rebuilt from ``seed`` and
``val_frac``.

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
from openamp.data import CorpusPaths, EmulationDataset
from openamp.data.emulation import read_with_retry
from openamp.dsp import audio as audio_io
from openamp.dsp import nam as nam_io
from openamp.encoders.refine import CrossAttentionPool2Block
from openamp.evaluation import load_model, resolve_device
from openamp.losses import build_loss, preemph_esr
from openamp.models import count_parameters
from openamp.probes.probe_synth import (HOP, N_MELS, SKIP_SECONDS, LogMel, TableCloud,
                                        captures_by_sha, save_atomic, step_rng)
from openamp.spec import spec_from_checkpoint
from openamp.training import GRAD_CLIP_NORM

# Raw branch: 4*4*4*4*2 = 512 = HOP, so one raw frame per log-mel token.
RAW_STRIDES = (4, 4, 4, 4, 2)
RAW_KERNELS = (8, 8, 8, 8, 4)
POWER_EPS = 1e-12

# --weight-decay, --lr-schedule, --warmup and --min-lr may change on --resume: the
# schedule is a function of the step alone, so a resumed run joins the new one there.
RESUME_FIXED = ("run", "sweep", "captures", "renders", "seed", "val_frac", "batch",
                "synth_per_batch", "ref_seconds", "clip_seconds", "val_clips",
                "eval_every", "lr", "width", "n_blocks", "n_heads",
                "ff_ratio", "conv_kernel", "n_queries", "d_model", "pool_heads",
                "mlp_ratio")


# --- Encoder --------------------------------------------------------------------------
class RawFrames(nn.Module):
    """``[B, 2, n]`` dry+wet ``-> [B, T, width]``, frame ``t`` centred on sample
    ``t * HOP`` -- the same centre as torchaudio's (centred) STFT frame ``t``.

    Dry and wet share channels from the first layer, so the first conv already
    sees the input->output mapping. LayerNorm over channels after every conv
    (per frame, so it does not depend on how long the input is). No BatchNorm.
    """

    def __init__(self, width: int):
        super().__init__()
        chans = [2] + [int(width)] * len(RAW_STRIDES)
        self.convs = nn.ModuleList(nn.Conv1d(chans[i], chans[i + 1], k, s)
                                   for i, (k, s) in enumerate(zip(RAW_KERNELS, RAW_STRIDES)))
        self.norms = nn.ModuleList(nn.LayerNorm(int(width)) for _ in RAW_STRIDES)
        self.receptive_field = 1 + sum((k - 1) * math.prod(RAW_STRIDES[:i])
                                       for i, k in enumerate(RAW_KERNELS))

    def forward(self, x: torch.Tensor, n_frames: int) -> torch.Tensor:
        h = F.pad(x, (self.receptive_field // 2, self.receptive_field))
        for conv, norm in zip(self.convs, self.norms):
            h = F.gelu(norm(conv(h).transpose(1, 2)).transpose(1, 2))
        if h.shape[-1] < n_frames:
            raise ValueError(f"raw branch gave {h.shape[-1]} frames, need {n_frames}")
        return h[:, :, :n_frames].transpose(1, 2)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary positions on ``[B, H, T, hd]``: each pair of dims turned by an angle
    proportional to the frame index."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class RotarySelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        if d % n_heads or (d // n_heads) % 2:
            raise ValueError(f"width {d} / {n_heads} heads must give an even head size")
        self.h, self.hd = int(n_heads), d // int(n_heads)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.register_buffer("inv_freq", 1.0 / 10000 ** (torch.arange(0, self.hd, 2).float()
                                                          / self.hd), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        ang = torch.arange(T, device=x.device).float()[:, None] * self.inv_freq[None]
        cos, sin = ang.cos(), ang.sin()
        q, k = rotate(q, cos, sin).to(v.dtype), rotate(k, cos, sin).to(v.dtype)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class ConformerBlock(nn.Module):
    """Gulati et al. 2020: half feed-forward, self-attention, conv module, half
    feed-forward, final LayerNorm. Pre-norm residuals. The conv module's
    BatchNorm is a LayerNorm here; no dropout."""

    def __init__(self, d: int, n_heads: int, ff_ratio: int, kernel: int):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"conv kernel must be odd, got {kernel}")

        def ff():
            return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff_ratio * d), nn.SiLU(),
                                 nn.Linear(ff_ratio * d, d))
        self.ff1, self.ff2 = ff(), ff()
        self.attn_norm = nn.LayerNorm(d)
        self.attn = RotarySelfAttention(d, n_heads)
        self.conv_norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.dw_norm = nn.LayerNorm(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.out_norm = nn.LayerNorm(d)

    def conv_module(self, x: torch.Tensor) -> torch.Tensor:
        h = F.glu(self.pw1(self.conv_norm(x).transpose(1, 2)), dim=1)
        h = self.dw_norm(self.dw(h).transpose(1, 2)).transpose(1, 2)
        return self.pw2(F.silu(h)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(self.attn_norm(x))
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class ConformerEncoder(nn.Module):
    """``(dry [B, n], wet [B, n]) -> [B, De]`` unit vectors.

    1. Per ~10.7 ms frame: the raw branch (:class:`RawFrames`) next to the dry
       and wet 128-band log-mel, the log-mel standardized by one fixed mean/std
       (``feat_mean``/``feat_std``, set from the real training amps). Level is
       never normalized away on the log-mel side.
    2. One Linear to ``width``, then ``n_blocks`` :class:`ConformerBlock`.
    3. probe_synth's pool (two cross-attention blocks, learned queries, no
       positions), one Linear to ``De``, L2-normalized.

    ``raw_chunk`` > 0 runs the raw branch under activation checkpointing, that many
    batch items at a time, when gradients are on: its first layers hold ~[B, width,
    n / 4] activations, which on a long reference do not fit next to the rest of
    the step. Items are independent in that branch, so the result is unchanged.
    """

    def __init__(self, embedding_dim: int, sample_rate: int, *, width: int, n_blocks: int,
                 n_heads: int, ff_ratio: int, conv_kernel: int, n_queries: int,
                 d_model: int, pool_heads: int, mlp_ratio: int, raw_chunk: int = 0):
        super().__init__()
        self.raw_chunk = int(raw_chunk)
        self.logmel = LogMel(sample_rate, 0)
        self.register_buffer("feat_mean", torch.zeros(2 * N_MELS))
        self.register_buffer("feat_std", torch.ones(2 * N_MELS))
        self.raw = RawFrames(width)
        self.fuse = nn.Linear(width + 2 * N_MELS, width)
        self.blocks = nn.ModuleList(ConformerBlock(width, n_heads, ff_ratio, conv_kernel)
                                    for _ in range(n_blocks))
        self.pool = CrossAttentionPool2Block(width, n_queries, d_model, n_heads=pool_heads,
                                             mlp_ratio=mlp_ratio, positional=False)
        self.proj = nn.Linear(self.pool.pooled_dim, embedding_dim)

    def mel_features(self, dry: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
        """``[B, T, 2 * N_MELS]`` dB, not yet standardized. In fp32 even under
        autocast: the mel filterbank is a matmul, which autocast would run in bf16."""
        with torch.autocast(dry.device.type, enabled=False):
            return torch.cat([self.logmel(dry.float()), self.logmel(wet.float())], dim=-1)

    def forward(self, dry: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
        mel = (self.mel_features(dry, wet) - self.feat_mean) / self.feat_std
        x = torch.stack([dry, wet], dim=1)
        if self.raw_chunk > 0 and torch.is_grad_enabled():
            raw = torch.cat([checkpoint(self.raw, c, mel.shape[1], use_reentrant=False)
                             for c in x.split(self.raw_chunk)])
        else:
            raw = self.raw(x, mel.shape[1])
        h = self.fuse(torch.cat([raw, mel.to(raw.dtype)], dim=-1))
        for block in self.blocks:
            h = block(h)
        return F.normalize(self.proj(self.pool(h)).float(), dim=-1)


# --- Data -----------------------------------------------------------------------------
def read_clean(ds: EmulationDataset, file: dict, c: int) -> np.ndarray:
    """``clean[c - R : c + clip]``, zero left-pad where the clip starts < R in --
    the input half of :meth:`EmulationDataset._read_window`, without the render.
    Clips are drawn from the dataset's file list exactly as it draws them."""
    R, clip = ds.receptive_field, ds.clip_samples
    start, pad = c - R, 0
    if start < 0:
        pad, start = -start, 0
    x = audio_io.seek_read(ds._clean_path(file), start, R + clip - pad)
    if pad:
        x = np.concatenate([np.zeros(pad, dtype=np.float32), x])
    return x.astype(np.float32, copy=False)


class StepClips(torch.utils.data.Dataset):
    """Item ``step``: that step's real amp ids and every amp's EGDB clip.

    ``x`` ``[batch, R + clip]`` clean input, real amps first, where ``R`` is
    the dataset's receptive field. Only clean audio is read: every target is
    played later, on the GPU -- a real amp's by its ``.nam``, a made-up amp's by
    the generator.
    """

    def __init__(self, ds: EmulationDataset, train_sel: np.ndarray, ids: np.ndarray,
                 n_real: int, n_synth: int, seed: int):
        self.ds, self.train_sel, self.ids = ds, train_sel, ids
        self.n_real, self.n_synth, self.seed = n_real, n_synth, seed
        self.perms: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return 1 << 30

    def real_ids(self, step: int) -> np.ndarray:
        """Consecutive slices of one seeded permutation of the training amps per pass."""
        out = []
        for p in range(step * self.n_real, (step + 1) * self.n_real):
            ep, j = divmod(p, len(self.train_sel))
            if ep not in self.perms:
                self.perms[ep] = step_rng(self.seed, 1, ep).permutation(self.train_sel)
            out.append(self.perms[ep][j])
        return self.ids[np.asarray(out, dtype=np.int64)]

    def __getitem__(self, step: int) -> dict:
        rng = step_rng(self.seed, 4, step)
        ds, clip = self.ds, self.ds.clip_samples

        def one():
            def draw():
                f = ds.files[int(rng.integers(0, len(ds.files)))]
                return (f, int(rng.integers(0, f["n"] - clip + 1))), f"file={f['file_id']}"
            return read_with_retry(draw, lambda s: read_clean(ds, *s), split=ds.split)

        x = np.stack([one() for _ in range(self.n_real + self.n_synth)])
        return {"step": step, "real_ids": torch.from_numpy(self.real_ids(step)),
                "x": torch.from_numpy(x)}


def amp_weights(power: torch.Tensor, strength: float) -> torch.Tensor:
    """``power ** -strength``, mean 1 -- see the module docstring."""
    w = power.clamp_min(POWER_EPS) ** -strength
    return w / w.mean()


# --- Main -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2_n10"),
                    help="emulate run: the frozen generator, its loss, and the table "
                         "whose rows shape the made-up draws")
    ap.add_argument("--sweep", type=Path,
                    default=Path("data_sweep_a1/clean/train/sweep_train.flac"),
                    help="the reference: the T3K capture signal's train region")
    ap.add_argument("--captures", type=Path, default=Path("data/captures"))
    ap.add_argument("--renders", type=Path, default=Path("data/manifests/renders.parquet"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/encoder/encoder_probe_conformer"))
    ap.add_argument("--batch", type=int, default=32, help="amps per step, real + made-up")
    ap.add_argument("--synth-per-batch", type=int, default=28)
    ap.add_argument("--steps", type=int, default=17_100)
    ap.add_argument("--eval-every", type=int, default=171,
                    help="steps between validations (171 = about one pass over the "
                         "training amps at 4 real per step)")
    ap.add_argument("--ref-seconds", type=float, default=2.0,
                    help="reference crop the encoder hears per training step")
    ap.add_argument("--clip-seconds", type=float, default=2.0,
                    help="EGDB clip the loss is scored on (the decoder trained on 2.0)")
    ap.add_argument("--val-clips", type=int, default=8,
                    help="fixed EGDB val-split clips per val amp")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ff-ratio", type=int, default=4)
    ap.add_argument("--conv-kernel", type=int, default=15,
                    help="Conformer depthwise conv, in frames (15 = 160 ms)")
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--pool-heads", type=int, default=4)
    ap.add_argument("--mlp-ratio", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                    help="after --warmup: hold --lr, or cosine-decay it to --min-lr at --steps")
    ap.add_argument("--warmup", type=int, default=0, help="steps of linear LR warmup")
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--render-chunk", type=int, default=4,
                    help="amps the generator plays the whole reference for at once")
    ap.add_argument("--raw-chunk", type=int, default=0,
                    help="checkpoint the raw branch this many amps at a time (0: off); "
                         "for a long --ref-seconds")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", type=Path, default=None,
                    help="last.pt of a stopped run: continue its exact stream to --steps")
    args = ap.parse_args()
    if not 0 <= args.synth_per_batch < args.batch:
        raise SystemExit("--synth-per-batch must be in [0, --batch)")

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

    # --- Frozen generator, its loss, its table --------------------------------------
    model, gen_ck = load_model(args.run, str(dev))
    model.requires_grad_(False)
    model.eval()
    R = model.receptive_field
    spec = spec_from_checkpoint(gen_ck)
    lossfn = build_loss(spec.loss).to(dev)
    strength = float(spec.loss.device_loss_weighting)
    tbl = torch.load(args.run / "embedding.pt", map_location="cpu", weights_only=False)
    table = tbl["embedding"].float().numpy()
    ids = np.array([int(d) for d in tbl["device_ids"]])
    embedding_dim = table.shape[1]
    scale = float(np.linalg.norm(table, axis=1).mean())

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
    n_synth = args.synth_per_batch
    n_real = args.batch - n_synth

    # --- Reference ------------------------------------------------------------------
    x, sr = audio_io.read_audio(args.sweep)
    x = np.asarray(x, dtype=np.float32)
    if sr != cfg.sample_rate:
        raise SystemExit(f"{args.sweep} is {sr} Hz, the corpus is {cfg.sample_rate} Hz")
    skip = int(round(SKIP_SECONDS * sr))
    L_ref = int(round(args.ref_seconds * sr))
    L_clip = int(round(args.clip_seconds * sr))
    n_keep = len(x) - skip
    if n_keep < L_ref:
        raise SystemExit(f"reference has {n_keep} samples after the skip, crop is {L_ref}")
    dry_full = torch.from_numpy(x[skip:]).to(dev)
    x_pad_gen = torch.from_numpy(np.concatenate([np.zeros(R, np.float32), x])).to(dev)

    nams = {}
    t0 = time.time()
    for i, d in enumerate(ids):
        nams[int(d)] = nam_io.load_model(nam_paths[int(d)], device=str(dev))
        if nams[int(d)].sample_rate != cfg.sample_rate:
            raise SystemExit(f"amp {d}: .nam at {nams[int(d)].sample_rate} Hz; crop "
                             f"rendering assumes {cfg.sample_rate}")
    R_nam = max(m.receptive_field for m in nams.values())
    x_pad_nam = np.concatenate([np.zeros(R_nam, np.float32), x])
    print(f"[conf] loaded {len(nams)} .nam captures ({time.time() - t0:.0f} s); "
          f"largest receptive field {R_nam}", flush=True)

    def real_full(d: int) -> torch.Tensor:
        """``[n_keep]`` the amp's ``.nam`` on the whole reference, at its learned level."""
        m = nams[int(d)]
        y = _render_signal(m, x, cfg, False) * scales[int(d)]
        return torch.from_numpy(y[skip:].astype(np.float32))

    def real_crop(d: int, g: int) -> np.ndarray:
        """The amp on ``x[g : g + L_ref]``, with its own receptive field of real context."""
        m = nams[int(d)]
        seg = x_pad_nam[R_nam - m.receptive_field + g: R_nam + g + L_ref]
        return (m.forward(seg)[-L_ref:] * scales[int(d)]).astype(np.float32)

    # EGDB clips are read with enough left context for the generator and every .nam.
    R_clip = max(R, R_nam)

    def real_clip(d: int, xc: np.ndarray) -> np.ndarray:
        """``[R_clip + L_clip]`` clean -> ``[L_clip]``: the amp's ``.nam`` on the clip."""
        m = nams[int(d)]
        return (m.forward(xc[R_clip - m.receptive_field:])[-L_clip:]
                * scales[int(d)]).astype(np.float32)

    @torch.no_grad()
    def gen_play(xin: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """``[S, R + n]`` input, ``[S, De]`` embeddings -> ``[S, n]``, bf16, no grad."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            return model.forward_emb(xin[:, None], E)[:, 0, R:].float()

    @torch.no_grad()
    def gen_full(E: torch.Tensor) -> torch.Tensor:
        """``[S, De]`` -> ``[S, n_keep]``: the generator on the whole reference."""
        out = []
        for s in range(0, len(E), args.render_chunk):
            e = E[s:s + args.render_chunk]
            out.append(gen_play(x_pad_gen[None].expand(len(e), -1), e)[:, skip:].cpu())
        return torch.cat(out)

    # --- Encoder --------------------------------------------------------------------
    encoder = ConformerEncoder(
        embedding_dim, sr, width=args.width, n_blocks=args.n_blocks, n_heads=args.n_heads,
        ff_ratio=args.ff_ratio, conv_kernel=args.conv_kernel, n_queries=args.n_queries,
        d_model=args.d_model, pool_heads=args.pool_heads, mlp_ratio=args.mlp_ratio,
        raw_chunk=args.raw_chunk).to(dev)

    # Real amps on the whole reference: the training amps for the log-mel
    # standardization (dry from the reference, wet from them), kept for the val amps.
    t0 = time.time()
    s1 = torch.zeros(N_MELS, dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    n_frames = 0
    with torch.no_grad():
        dry_mel = encoder.logmel(dry_full[None])[0]                         # [T, M]
        for i, j in enumerate(train_sel):
            w = encoder.logmel(real_full(ids[j]).to(dev)[None])[0].double().cpu()
            s1 += w.sum(0)
            s2 += (w * w).sum(0)
            n_frames += w.shape[0]
            if (i + 1) % 100 == 0 or i + 1 == len(train_sel):
                print(f"[conf] rendered {i + 1}/{len(train_sel)} training amps on the "
                      f"reference ({time.time() - t0:.0f} s)", flush=True)
    wet_mean = s1 / n_frames
    wet_std = (s2 / n_frames - wet_mean ** 2).clamp_min(0.0).sqrt()
    encoder.feat_mean.copy_(torch.cat([dry_mel.mean(0).double().cpu(), wet_mean]).float())
    encoder.feat_std.copy_(torch.cat([dry_mel.std(0).double().cpu(), wet_std])
                           .float().clamp_min(1e-3))
    wet_val = torch.stack([real_full(d) for d in val_ids])                  # [V, n_keep]
    p_val = (wet_val.double() ** 2).mean(1).float()
    print(f"[conf] rendered {len(val_ids)} val amps on the reference", flush=True)

    # --- EGDB clips: training stream, and fixed val clips ---------------------------
    paths = CorpusPaths.from_config(cfg)
    train_ds = EmulationDataset(paths, "train", receptive_field=R_clip,
                                id_to_idx={int(ids[j]): int(j) for j in train_sel},
                                clip_samples=L_clip, pairs_per_epoch=1)
    val_ds = EmulationDataset(paths, "val", receptive_field=R_clip,
                              id_to_idx={int(d): i for i, d in enumerate(val_ids)},
                              clip_samples=L_clip, pairs_per_epoch=1)
    vx, vt, va = [], [], []
    for a, d in enumerate(val_ids):
        rng = step_rng(args.seed, 5, a)
        for _ in range(args.val_clips):
            def draw():
                f = val_ds.files[int(rng.integers(0, len(val_ds.files)))]
                return (f, int(rng.integers(0, f["n"] - L_clip + 1))), f"device={d}"
            xi = read_with_retry(draw, lambda s: read_clean(val_ds, *s), split="val")
            vx.append(xi[R_clip - R:])
            vt.append(real_clip(d, xi))
            va.append(a)
    vx, vt = torch.from_numpy(np.stack(vx)), torch.from_numpy(np.stack(vt))
    va = torch.as_tensor(va)
    print(f"[conf] val: {len(val_ids)} amps x {args.val_clips} EGDB val clips", flush=True)

    cloud = TableCloud(table[train_sel], scale)
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best_val, best_step, best_state = 0, float("inf"), -1, None
    if ck is not None:
        encoder.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start, best_val, best_step = ck["step"] + 1, ck["best_val"], ck["best_step"]
        for group in opt.param_groups:
            group["weight_decay"] = args.weight_decay
        best_state = ck["best_state"]
        print(f"[conf] resumed {args.resume} at step {start} (best val {best_val:.4f} "
              f"at step {best_step + 1})")
    n_params = count_parameters(encoder)
    print(f"[conf] {len(ids)} amps: {len(train_sel)} train / {len(val_sel)} val  "
          f"decoder={args.run.name}  radius {scale:.4f}  loss={spec.loss.name} "
          f"(amp weighting {strength:g})")
    print(f"[conf] per step: {n_real} real + {n_synth} made-up amps, reference crop "
          f"{args.ref_seconds:g} s, loss clip {args.clip_seconds:g} s")
    print(f"[conf] encoder: raw {RAW_STRIDES} (rf {encoder.raw.receptive_field}) + log-mel "
          f"-> {args.width} -> {args.n_blocks} conformer/{args.n_heads}h k{args.conv_kernel} "
          f"-> pool {args.n_queries}x{args.d_model}x2blk -> {embedding_dim}   "
          f"{n_params:,} params", flush=True)

    @torch.no_grad()
    def encode_full(wet: torch.Tensor) -> torch.Tensor:
        """``[N, n_keep]`` wet on the whole reference -> ``[N, De]`` unit vectors."""
        encoder.eval()
        out = []
        for w in wet:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(encoder(dry_full[None], w.to(dev)[None]))
        return torch.cat(out)

    @torch.no_grad()
    def validate() -> dict:
        E = scale * encode_full(wet_val)                                     # [V, De]
        w_amp = amp_weights(p_val, strength).to(dev)
        tot = esr_t = stft_t = 0.0
        nb = 0
        for s in range(0, len(vx), args.batch):
            a = va[s:s + args.batch].to(dev)
            y = gen_play(vx[s:s + args.batch].to(dev), E[a])
            w = w_amp[a] / w_amp[a].mean()
            loss, parts = lossfn(y[:, None], vt[s:s + args.batch].to(dev)[:, None], w)
            tot += float(loss)
            esr_t += parts["esr"]
            stft_t += parts["stft"]
            nb += 1
        return {"loss": tot / nb, "esr": esr_t / nb, "stft": stft_t / nb}

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        if args.lr_schedule == "constant":
            return args.lr
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    def checkpoint(step: int) -> dict:
        return {"state_dict": encoder.state_dict(), "opt": opt.state_dict(), "step": step,
                "best_val": best_val, "best_step": best_step, "best_state": best_state,
                "args": arg_dict, "device_ids": ids.tolist()}

    # --- Train ----------------------------------------------------------------------
    clips = StepClips(train_ds, train_sel, ids, n_real, n_synth, args.seed)
    loader = torch.utils.data.DataLoader(
        clips, batch_size=None, sampler=range(start, args.steps),
        num_workers=args.num_workers, pin_memory=dev.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None)

    acc = {"loss": 0.0, "real": 0.0, "synth": 0.0, "n": 0}
    t0, t_wait, t_render = time.time(), 0.0, 0.0
    tw = time.time()
    for item in loader:
        step = int(item["step"])
        t_wait += time.time() - tw
        encoder.train()
        g = skip + int(step_rng(args.seed, 3, step).integers(0, n_keep - L_ref + 1))
        dry = torch.from_numpy(x[g:g + L_ref]).to(dev)
        xc = item["x"]
        xin = xc[:, R_clip - R:].to(dev, non_blocking=True)
        wet_ref, tgt = [], []
        if n_real:
            rid = [int(d) for d in item["real_ids"]]
            wet_ref.append(torch.from_numpy(np.stack([real_crop(d, g) for d in rid])).to(dev))
            tgt.append(torch.from_numpy(np.stack(
                [real_clip(d, xc[i].numpy()) for i, d in enumerate(rid)])).to(dev))
        if n_synth:
            tr0 = time.time()
            E = torch.from_numpy(cloud.draw(step_rng(args.seed, 2, step), n_synth)).to(dev)
            seg = x_pad_gen[g:g + R + L_ref]
            wet_ref.append(gen_play(seg[None].expand(n_synth, -1), E))
            tgt.append(gen_play(xin[n_real:], E))
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_render += time.time() - tr0
        wet_ref, tgt = torch.cat(wet_ref), torch.cat(tgt)
        w = amp_weights((wet_ref.double() ** 2).mean(1).float(), strength)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            e = scale * encoder(dry[None].expand(args.batch, -1), wet_ref)
            y = model.forward_emb(xin[:, None], e)[:, :, R:]
        loss, _ = lossfn(y.float(), tgt[:, None], w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP_NORM)
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        opt.step()

        with torch.no_grad():
            yf = y.float()
            acc["loss"] += float(loss)
            if n_real:
                acc["real"] += float(preemph_esr(yf[:n_real], tgt[:n_real, None],
                                                 lossfn.coeff, weights=w[:n_real]))
            if n_synth:
                acc["synth"] += float(preemph_esr(yf[n_real:], tgt[n_real:, None],
                                                  lossfn.coeff, weights=w[n_real:]))
            acc["n"] += 1

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            va_m = validate()
            star = va_m["loss"] < best_val - 1e-6
            if star:
                best_val, best_step = va_m["loss"], step
                best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            n = max(acc["n"], 1)
            elapsed = time.time() - t0
            print(f"  s{step + 1:05d}  train loss {acc['loss'] / n:.4f}  esr real "
                  f"{acc['real'] / n:.4f} made-up {acc['synth'] / n:.4f}  val loss "
                  f"{va_m['loss']:.4f} (esr {va_m['esr']:.4f} stft {va_m['stft']:.4f}) "
                  f"best {best_val:.4f}  lr {lr_at(step):.2e}  {elapsed / 60:.1f} min, "
                  f"data wait "
                  f"{t_wait / max(elapsed, 1e-9):.0%} render "
                  f"{t_render / max(elapsed, 1e-9):.0%}" + ("  *" if star else ""),
                  flush=True)
            acc = {k: 0 if k == "n" else 0.0 for k in acc}
            save_atomic(checkpoint(step), args.out / "last.pt")
        tw = time.time()
    train_min = (time.time() - t0) / 60
    print(f"[conf] trained to step {args.steps} in {train_min:.1f} min; best val "
          f"{best_val:.4f} at step {best_step + 1}")

    # --- Predict every amp, and the val amps once more from the generator's render ---
    encoder.load_state_dict(best_state)
    pred_all = np.zeros((len(ids), embedding_dim), np.float32)
    t0 = time.time()
    for i, d in enumerate(ids):
        pred_all[i] = (scale * encode_full(real_full(d)[None]))[0].cpu().numpy()
        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            print(f"[conf] predicted {i + 1}/{len(ids)} amps ({time.time() - t0:.0f} s)",
                  flush=True)
    Ev = torch.from_numpy(table[val_sel]).to(dev)
    sim_unit = encode_full(gen_full(Ev))
    pred_sim = (scale * sim_unit).cpu().numpy()
    cos_real = F.cosine_similarity(torch.from_numpy(pred_all[val_sel]).to(dev), Ev).mean()
    cos_sim = F.cosine_similarity(sim_unit, Ev).mean()
    print(f"[conf] val cosine to the table row (not trained on): hearing the real amp "
          f"{float(cos_real):.3f}, hearing the generator's render of its row "
          f"{float(cos_sim):.3f}")

    report = {
        "n_params": int(n_params), "n_train": int(len(train_sel)), "n_val": int(len(val_sel)),
        "steps": int(args.steps), "real_per_batch": int(n_real),
        "synth_per_batch": int(n_synth), "made_up_amps": int(args.steps * n_synth),
        "ref_seconds": args.ref_seconds, "clip_seconds": args.clip_seconds,
        "best_val_loss": best_val, "best_step": int(best_step + 1),
        "val_cos_real": float(cos_real), "val_cos_sim": float(cos_sim),
        "embedding_dim": int(embedding_dim), "target_radius": scale,
        "train_minutes": train_min,
    }
    torch.save({"state_dict": encoder.state_dict(), "args": arg_dict,
                "device_ids": ids.tolist(),
                "pred": {int(d): pred_all[i] for i, d in enumerate(ids)},
                "pred_sim": {int(ids[j]): pred_sim[i] for i, j in enumerate(val_sel)},
                "report": report},
               args.out / "probe_conformer.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[conf] wrote {args.out}/probe_conformer.pt and report.json")


if __name__ == "__main__":
    main()
