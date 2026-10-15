"""Train probe_attn's encoder head on real amps plus made-up amps rendered online.

    OPENAMP_DATA_DIR=data python -m openamp.probes.probe_synth --synth-per-batch 16 \\
        --out results/encoder/encoder_probe_synth_e1

then ``python -m openamp.probes.probe_esr --probe <out>/probe_synth.pt`` to score it.

The encoder has 684 amps to learn from, and the amp-count curve (2026-10-01:
171 / 342 / 684 training amps -> best held-out cosine 0.39 / 0.44 / 0.53) says
more amps help and have not levelled off. This makes more. Any embedding is an
amp the frozen generator can play, and for a made-up one the right answer is
known exactly -- it is the embedding that was drawn.

**Made-up amps.** Each step draws ``--synth-per-batch`` embeddings from a
Gaussian with the mean and covariance of the encoder's *training* rows (never
the val rows), scales each to the table's one radius, and renders the reference
through the frozen generator -- online, in bf16, no gradient, never stored or
reused. Shaped like the table, a draw puts 0.92 of its energy in the directions
the generator's weights respond to, as the real rows do; an isotropic draw puts
0.50 there, so half its target would be inaudible. bf16 against fp32 changes a
render by ESR ~1e-4.

**What the encoder hears.** The T3K capture's train region (``sweep_train``,
171 s) minus everything before its first chirp at 2.1 s: two single-sample blips
and silence. Rendering an amp from its own table row, the generator is off by
78% of the gap to the nearest other real amp on the blips, against 6% on the
guitar, 19% on the chirps and 27% on the noise bursts -- the kept audio is where
made-up renders look like real ones.

Real amps are rendered from their ``.nam`` at startup, the way ``openamp render``
rendered the corpus (:func:`openamp.corpus.render._render_signal`), and scaled by
the ``output_scale`` their EGDB renders carried: that is the level the generator
learned to produce. Each ``.nam`` is found by the sha256 its training render
recorded, not by a manifest path: on 2026-10-01 ``data_sweep_a1``'s manifest
pointed 366 of 844 amps at their A2 file instead.

**Tokens.** One per STFT frame (window 2048, hop 512, ~11 ms): the dry frame's
128 log-mel bands next to the wet frame's -- 256 numbers saying "this went in,
this came out". The dry is the same for every amp; it is here because the pool
reads tokens as an unordered set, and the dry half is what tells an output
frame which input it answers. Level is kept as is (no loudness normalization);
each of the 256 numbers is standardized with one fixed mean/std, taken from the
real training amps, so every amp goes through the same transform.

**Encoder.** probe_attn's head on these tokens: the two-block cross-attention
pool with no positions, one Linear to the table width, L2-normalized. Training
reads a random ``--keep-frac`` of each amp's tokens per step; validation and the
predictions read all of them.

**E0 / E1.** ``--synth-per-batch 0`` is E0, real amps only; 16 of 32 is E1.
Steps, batch, loss (1 - cos), optimizer and the val schedule are the same
either way, so the made-up amps are the one difference. The default 17,100
steps of 32 amps is the DAC probe's budget, 800 epochs over 684 amps.

**Scoring** is :mod:`openamp.probes.probe_esr`. The checkpoint carries the
encoder's vector for every amp (``pred``), and for the val amps a second one
(``pred_sim``) from hearing the generator's render of their own table row
instead of the real one -- the gap between the two is what training on made-up
audio costs on real audio.

**Resuming.** Every draw (which real amps, which made-up embeddings, which
tokens) is seeded by ``(seed, step)``, so ``--resume <out>/last.pt`` continues
the identical stream. ``last.pt`` is rewritten at every validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from openamp.core.config import load_config
from openamp.core.util import sha256_file
from openamp.corpus.render import _render_signal
from openamp.dsp import audio as audio_io
from openamp.dsp import nam as nam_io
from openamp.encoders.refine import CrossAttentionPool2Block
from openamp.evaluation import load_model, resolve_device
from openamp.models import count_parameters
from openamp.probes.probe_attn import cosine_loss
from openamp.training import GRAD_CLIP_NORM

N_FFT = 2048            # 42.7 ms at 48 kHz
HOP = 512               # 10.7 ms between tokens
N_MELS = 128
DB_EPS = 1e-10          # power floor, -100 dB
SKIP_SECONDS = 2.1      # sweep_train before its first chirp: blips at 0.5 s and 1.5 s

# Settings a resume must share with the run it continues: they shape the stream
# of draws or the network. --steps, --render-chunk and --device may change.
RESUME_FIXED = ("run", "sweep", "captures", "renders", "seed", "val_frac", "batch",
                "synth_per_batch", "keep_frac", "eval_every", "lr", "weight_decay",
                "n_queries", "d_model", "n_heads", "mlp_ratio")


class TokenEncoder(nn.Module):
    """``[B, T, D]`` tokens ``-> [B, De]`` unit vectors.

    :class:`~openamp.encoders.refine.AttentionEncoder`'s head with the frame-store
    lookup taken out: the same two-block pool and single-Linear adapter, fed
    tokens the caller built. No positions -- each token's dry half carries its
    place in the reference.
    """

    def __init__(self, in_dim: int, embedding_dim: int, *, n_queries: int,
                 d_model: int, n_heads: int, mlp_ratio: int):
        super().__init__()
        self.pool = CrossAttentionPool2Block(in_dim, n_queries, d_model, n_heads=n_heads,
                                             mlp_ratio=mlp_ratio, positional=False)
        self.proj = nn.Linear(self.pool.pooled_dim, embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.pool(tokens)), dim=-1)


class LogMel(nn.Module):
    """``[B, n]`` audio ``-> [B, T, N_MELS]`` dB, frames before ``first_frame`` dropped."""

    def __init__(self, sample_rate: int, first_frame: int):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
        self.first_frame = int(first_frame)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        p = self.mel(y.float())                                       # [B, N_MELS, T]
        return (10.0 * torch.log10(p + DB_EPS)).transpose(1, 2)[:, self.first_frame:]


def first_kept_frame(skip_samples: int) -> int:
    """The first STFT frame (centred, as torchaudio's) that starts at or after
    ``skip_samples``."""
    return math.ceil((skip_samples + N_FFT // 2) / HOP)


class TableCloud:
    """Random embeddings shaped like a set of table rows.

    A Gaussian with the rows' mean and covariance; each draw is then scaled to
    ``radius``, the one length every row of an ``embed_normalize`` table has.
    """

    def __init__(self, rows: np.ndarray, radius: float):
        rows = np.asarray(rows, dtype=np.float64)
        self.mu = rows.mean(0)
        cov = np.cov(rows.T)
        self.chol = np.linalg.cholesky(cov + 1e-9 * np.eye(len(cov)))
        self.radius = float(radius)

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        e = self.mu + rng.standard_normal((n, len(self.mu))) @ self.chol.T
        e = self.radius * e / np.linalg.norm(e, axis=1, keepdims=True)
        return e.astype(np.float32)


def step_rng(seed: int, stream: int, step: int) -> np.random.Generator:
    """One generator per ``(stream, step)``, so a resumed run redraws exactly what
    it would have."""
    return np.random.default_rng((seed, stream, step))


def captures_by_sha(captures_dir: Path, wanted: set[str]) -> dict[str, Path]:
    """``{sha256: path}`` for each wanted hash among the ``.nam`` files under
    ``captures_dir``."""
    found: dict[str, Path] = {}
    for p in sorted(Path(captures_dir).rglob("*.nam")):
        h = sha256_file(p)
        if h in wanted and h not in found:
            found[h] = p
    return found


@torch.no_grad()
def render_real(ids, nam_paths: dict, scales: dict, x: np.ndarray, cfg,
                logmel: LogMel, dev) -> torch.Tensor:
    """``[N, T, N_MELS]`` fp16 on the CPU: each amp's ``.nam`` on the reference.

    Rendered as ``openamp render`` renders the corpus, then scaled by the
    ``output_scale`` the generator's training renders carried for that amp.
    """
    out, t0 = [], time.time()
    for i, d in enumerate(ids):
        m = nam_io.load_model(nam_paths[d], device=str(dev))
        y = (_render_signal(m, x, cfg, m.sample_rate != cfg.sample_rate)
             * scales[d]).astype(np.float32)
        out.append(logmel(torch.from_numpy(y).to(dev)[None])[0].half().cpu())
        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            print(f"[synth] rendered {i + 1}/{len(ids)} real amps "
                  f"({time.time() - t0:.0f} s)", flush=True)
    return torch.stack(out)


@torch.no_grad()
def render_synth(model, x_pad: torch.Tensor, E: torch.Tensor, chunk: int,
                 logmel: LogMel) -> torch.Tensor:
    """``[S, De]`` embeddings ``-> [S, T, N_MELS]`` dB through the frozen generator.

    ``x_pad`` is ``[1, 1, R + n]``: the reference behind ``R`` zeros, the same
    left context the corpus renderer gives a file's first sample.
    """
    R = model.receptive_field
    feats = []
    for s in range(0, len(E), chunk):
        e = E[s:s + chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x_pad.is_cuda):
            y = model.forward_emb(x_pad.expand(len(e), -1, -1), e)[:, 0, R:]
        feats.append(logmel(y.float()))
    return torch.cat(feats)


def make_tokens(wet: torch.Tensor, dry: torch.Tensor, mean: torch.Tensor,
                std: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
    """Wet ``[B, T, M]`` + dry ``[T, M]`` -> standardized ``[B, T', 2M]``.

    ``idx`` ``[B, k]`` keeps those tokens per amp (``T' = k``); ``None`` keeps all.
    """
    if idx is not None:
        wet = wet.gather(1, idx[..., None].expand(-1, -1, wet.shape[-1]))
        d = dry[idx]
    else:
        d = dry.expand(wet.shape[0], -1, -1)
    return (torch.cat([d, wet.float()], dim=-1) - mean) / std


def standardizer(W: torch.Tensor, train_sel: np.ndarray,
                 dry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-number mean and std of the tokens: dry from the reference, wet from the
    real training amps (accumulated in chunks -- the float copy would be ~5 GB)."""
    s1 = torch.zeros(W.shape[-1], dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    n = 0
    for s in range(0, len(train_sel), 64):
        w = W[torch.from_numpy(train_sel[s:s + 64])].double()
        s1 += w.sum((0, 1))
        s2 += (w * w).sum((0, 1))
        n += w.shape[0] * w.shape[1]
    wet_mean = s1 / n
    wet_std = (s2 / n - wet_mean ** 2).clamp_min(0.0).sqrt()
    mean = torch.cat([dry.mean(0).double().cpu(), wet_mean]).float()
    std = torch.cat([dry.std(0).double().cpu(), wet_std]).float().clamp_min(1e-3)
    return mean, std


def save_atomic(obj, path: Path) -> None:
    """Write then rename, so a job killed mid-save leaves the previous file intact."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2_n10"),
                    help="emulate run: the frozen generator (checkpoint.pt) and the "
                         "target table (embedding.pt)")
    ap.add_argument("--sweep", type=Path,
                    default=Path("data_sweep_a1/clean/train/sweep_train.flac"),
                    help="the reference: the T3K capture signal's train region")
    ap.add_argument("--captures", type=Path, default=Path("data/captures"),
                    help="directory searched (recursively) for each amp's .nam, "
                         "matched by the sha256 in --renders")
    ap.add_argument("--renders", type=Path,
                    default=Path("data/manifests/renders.parquet"),
                    help="the generator's training renders manifest: per-amp "
                         "output_scale (the level it learned) and nam_sha256")
    ap.add_argument("--out", type=Path, default=Path("results/encoder/encoder_probe_synth"))
    ap.add_argument("--synth-per-batch", type=int, default=16,
                    help="made-up amps per step; 0 = real amps only (E0)")
    ap.add_argument("--batch", type=int, default=32,
                    help="amps per step, real + made-up")
    ap.add_argument("--steps", type=int, default=17_100,
                    help="optimizer steps (17,100 x 32 = the DAC probe's 800 epochs "
                         "over 684 amps)")
    ap.add_argument("--eval-every", type=int, default=22,
                    help="steps between validations (22 = about one pass over the "
                         "training amps at 32 real per step)")
    ap.add_argument("--render-chunk", type=int, default=4,
                    help="made-up amps rendered at once (~0.8 GB of GPU memory each)")
    ap.add_argument("--keep-frac", type=float, default=0.5,
                    help="fraction of an amp's tokens one training step reads")
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--mlp-ratio", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", type=Path, default=None,
                    help="last.pt of a stopped run: continue its exact stream to --steps")
    args = ap.parse_args()
    if not 0 <= args.synth_per_batch < args.batch:
        raise SystemExit("--synth-per-batch must be in [0, --batch)")
    if not 0.0 < args.keep_frac <= 1.0:
        raise SystemExit(f"--keep-frac must be in (0, 1], got {args.keep_frac}")

    torch.manual_seed(args.seed)
    dev = torch.device(resolve_device(args.device))
    cfg = load_config()
    args.out.mkdir(parents=True, exist_ok=True)
    arg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    ck = None
    if args.resume is not None:              # checked now, before minutes of rendering
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        diff = {k: (ck["args"][k], arg_dict[k]) for k in RESUME_FIXED
                if ck["args"][k] != arg_dict[k]}
        if diff:
            raise SystemExit(f"--resume: these settings differ from the run being "
                             f"resumed: {diff}")

    # --- Frozen generator and its table ------------------------------------------------
    model, _ = load_model(args.run, str(dev))
    model.requires_grad_(False)
    R = model.receptive_field
    tbl = torch.load(args.run / "embedding.pt", map_location="cpu", weights_only=False)
    table = tbl["embedding"].float().numpy()
    table_ids = [int(d) for d in tbl["device_ids"]]

    rend = pd.read_parquet(args.renders, columns=["device_id", "output_scale", "nam_sha256"])
    per_dev = rend.groupby("device_id").agg(scale=("output_scale", "first"),
                                            n_scale=("output_scale", "nunique"),
                                            sha=("nam_sha256", "first"))
    if (per_dev["n_scale"] > 1).any():
        raise SystemExit(f"{args.renders}: an amp with more than one output_scale")
    scales = {int(d): float(s) for d, s in per_dev["scale"].items()}
    found = captures_by_sha(args.captures, set(per_dev["sha"]))
    nam_paths = {int(d): found[h] for d, h in per_dev["sha"].items() if h in found}

    # Every table amp, or the split below is not probe_attn's and E0 stops being
    # comparable with the DAC probe -- so a missing capture is an error, not a drop.
    missing = [d for d in table_ids if d not in nam_paths]
    if missing:
        raise SystemExit(f"no .nam under {args.captures} for {len(missing)} table "
                         f"amp(s), e.g. {missing[:5]}")
    ids = np.array(table_ids)
    Y = torch.from_numpy(table).float()
    embedding_dim = Y.shape[1]
    scale = float(Y.norm(dim=-1).mean())

    # The same split probe_attn draws: permutation(len(ids)) from a fresh
    # default_rng(seed) before any other draw, over the table's own order.
    perm = np.random.default_rng(args.seed).permutation(len(ids))
    n_val = max(1, int(round(args.val_frac * len(ids))))
    val_sel, train_sel = perm[:n_val], perm[n_val:]
    n_synth = args.synth_per_batch
    n_real = args.batch - n_synth

    # --- Reference and real amps --------------------------------------------------------
    x, sr = audio_io.read_audio(args.sweep)
    if sr != cfg.sample_rate:
        raise SystemExit(f"{args.sweep} is {sr} Hz, the corpus is {cfg.sample_rate} Hz")
    first = first_kept_frame(int(round(SKIP_SECONDS * sr)))
    logmel = LogMel(sr, first).to(dev)
    dry = logmel(torch.from_numpy(x).to(dev)[None])[0]                # [T, M]
    n_tok = dry.shape[0]
    n_keep = max(1, int(round(args.keep_frac * n_tok)))
    print(f"[synth] reference {args.sweep} ({len(x) / sr:.1f} s): {n_tok} tokens after "
          f"dropping the first {first} frames (< {SKIP_SECONDS} s); training reads "
          f"{n_keep} per amp per step", flush=True)
    W = render_real(ids, nam_paths, scales, x, cfg, logmel, dev)              # [N, T, M]
    W_val = W[torch.from_numpy(val_sel)]

    # One fixed standardization for every amp; a resumed run keeps its own exactly.
    mean, std = ((ck["feat_mean"], ck["feat_std"]) if ck is not None
                 else standardizer(W, train_sel, dry))
    mean, std = mean.to(dev), std.to(dev)

    cloud = TableCloud(table[train_sel], scale)
    x_pad = torch.from_numpy(np.concatenate([np.zeros(R, np.float32), x])).to(dev)[None, None]
    Y_dev = Y.to(dev)

    # --- Encoder ------------------------------------------------------------------------
    encoder = TokenEncoder(2 * N_MELS, embedding_dim, n_queries=args.n_queries,
                           d_model=args.d_model, n_heads=args.n_heads,
                           mlp_ratio=args.mlp_ratio).to(dev)
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best_val, best_step, best_state = 0, float("inf"), -1, None
    if ck is not None:
        encoder.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start, best_val, best_step = ck["step"] + 1, ck["best_val"], ck["best_step"]
        best_state = ck["best_state"]
        print(f"[synth] resumed {args.resume} at step {start} (best val {best_val:.4f} "
              f"at step {best_step + 1})")
    n_params = count_parameters(encoder)
    print(f"[synth] {len(ids)} amps: {len(train_sel)} train / {len(val_sel)} val  "
          f"target={args.run.name}  radius {scale:.4f}")
    print(f"[synth] per step: {n_real} real + {n_synth} made-up amps, "
          f"{args.steps} steps -> {args.steps * n_synth:,} made-up amps in all")
    print(f"[synth] encoder: tokens {2 * N_MELS} -> pool {args.n_queries}x{args.d_model}"
          f"x2blk/{args.n_heads}h (no positions) -> {embedding_dim}   {n_params:,} params",
          flush=True)

    @torch.no_grad()
    def predict(wet: torch.Tensor) -> torch.Tensor:
        """``[N, T, M]`` wet features -> ``[N, De]`` unit vectors, every token read."""
        encoder.eval()
        return torch.cat([encoder(make_tokens(wet[s:s + args.batch].to(dev), dry, mean,
                                              std, None))
                          for s in range(0, len(wet), args.batch)])

    def validate() -> dict:
        pred = predict(W_val)
        loss, cos = cosine_loss(pred, Y_dev[torch.from_numpy(val_sel).to(dev)])
        return {"loss": float(loss), "cos": cos}

    def checkpoint(step: int) -> dict:
        return {"state_dict": encoder.state_dict(), "opt": opt.state_dict(), "step": step,
                "best_val": best_val, "best_step": best_step, "best_state": best_state,
                "args": arg_dict, "device_ids": ids.tolist(),
                "feat_mean": mean.cpu(), "feat_std": std.cpu()}

    # --- Train --------------------------------------------------------------------------
    perms: dict[int, np.ndarray] = {}

    def real_batch(step: int) -> np.ndarray:
        """Consecutive slices of one seeded permutation of the training amps per pass."""
        out = []
        for p in range(step * n_real, (step + 1) * n_real):
            ep, j = divmod(p, len(train_sel))
            if ep not in perms:
                perms[ep] = step_rng(args.seed, 1, ep).permutation(train_sel)
            out.append(perms[ep][j])
        return np.asarray(out, dtype=np.int64)

    acc = {"real_loss": 0.0, "real_cos": 0.0, "n_real": 0,
           "synth_loss": 0.0, "synth_cos": 0.0, "n_synth": 0}
    t0, t_render = time.time(), 0.0
    for step in range(start, args.steps):
        encoder.train()
        wet, tgt = [], []
        if n_real:
            b = real_batch(step)
            wet.append(W[torch.from_numpy(b)].to(dev, non_blocking=True).float())
            tgt.append(Y_dev[torch.from_numpy(b).to(dev)])
        if n_synth:
            tr0 = time.time()
            E = torch.from_numpy(cloud.draw(step_rng(args.seed, 2, step), n_synth)).to(dev)
            wet.append(render_synth(model, x_pad, E, args.render_chunk, logmel))
            tgt.append(E)
            if dev.type == "cuda":
                torch.cuda.synchronize()     # or the GPU's queued render is billed later
            t_render += time.time() - tr0
        wet, tgt = torch.cat(wet), torch.cat(tgt)
        rng = step_rng(args.seed, 3, step)
        idx = torch.from_numpy(np.stack([np.sort(rng.choice(n_tok, n_keep, replace=False))
                                         for _ in range(len(wet))])).to(dev)
        pred = encoder(make_tokens(wet, dry, mean, std, idx))
        loss, _ = cosine_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP_NORM)
        opt.step()

        with torch.no_grad():
            cos = F.cosine_similarity(pred, F.normalize(tgt, dim=-1), dim=-1)
        for part, c in (("real", cos[:n_real]), ("synth", cos[n_real:])):
            acc[f"{part}_loss"] += float((1.0 - c).sum())
            acc[f"{part}_cos"] += float(c.sum())
            acc[f"n_{part}"] += len(c)

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            va = validate()
            star = va["loss"] < best_val - 1e-6
            if star:
                best_val, best_step = va["loss"], step
                best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            txt = [f"  s{step + 1:05d}"]
            for part, label in (("real", "real"), ("synth", "made-up")):
                n = max(acc[f"n_{part}"], 1)
                if acc[f"n_{part}"]:
                    txt.append(f"{label} {acc[f'{part}_loss'] / n:.4f} "
                               f"(cos {acc[f'{part}_cos'] / n:.3f})")
            elapsed = time.time() - t0
            txt.append(f"val {va['loss']:.4f} (cos {va['cos']:.3f}) best {best_val:.4f}")
            txt.append(f"{elapsed / 60:.1f} min, render {t_render / max(elapsed, 1e-9):.0%}")
            print("  ".join(txt) + ("  *" if star else ""), flush=True)
            acc = {k: 0 if k.startswith("n_") else 0.0 for k in acc}
            save_atomic(checkpoint(step), args.out / "last.pt")
    train_min = (time.time() - t0) / 60
    print(f"[synth] trained to step {args.steps} in {train_min:.1f} min; best val "
          f"{best_val:.4f} at step {best_step + 1}")

    # --- Predict every amp, and the val amps once more from the generator's render ---
    encoder.load_state_dict(best_state)
    pred_all = (scale * predict(W)).cpu().numpy()
    Ev = Y_dev[torch.from_numpy(val_sel).to(dev)]
    sim_unit = predict(render_synth(model, x_pad, Ev, args.render_chunk, logmel))
    pred_sim = (scale * sim_unit).cpu().numpy()
    cos_real = F.cosine_similarity(torch.from_numpy(pred_all[val_sel]).to(dev), Ev).mean()
    cos_sim = F.cosine_similarity(sim_unit, Ev).mean()
    print(f"[synth] val cosine at best: hearing the real amp {float(cos_real):.3f}, "
          f"hearing the generator's render of its row {float(cos_sim):.3f}")

    report = {
        "n_params": int(n_params), "n_train": int(len(train_sel)), "n_val": int(len(val_sel)),
        "steps": int(args.steps), "real_per_batch": int(n_real),
        "synth_per_batch": int(n_synth), "made_up_amps": int(args.steps * n_synth),
        "best_val_loss": best_val, "best_step": int(best_step + 1),
        "val_cos_real": float(cos_real), "val_cos_sim": float(cos_sim),
        "n_tokens": int(n_tok), "keep_tokens": int(n_keep),
        "embedding_dim": int(embedding_dim), "target_radius": scale,
        "train_minutes": train_min,
    }
    torch.save({"state_dict": encoder.state_dict(), "args": arg_dict,
                "device_ids": ids.tolist(),
                "pred": {int(d): pred_all[i] for i, d in enumerate(ids)},
                "pred_sim": {int(ids[j]): pred_sim[i] for i, j in enumerate(val_sel)},
                "feat_mean": mean.cpu(), "feat_std": std.cpu(), "report": report},
               args.out / "probe_synth.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[synth] wrote {args.out}/probe_synth.pt and report.json")


if __name__ == "__main__":
    main()
