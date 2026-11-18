"""The sweep-aware refinement network (run r1): a vector improved one pass at a time.

    OPENAMP_DATA_DIR=data python -m openamp.probes.probe_refine \\
        --out results/encoder/encoder_probe_refine_r1

Spec: the "Sweep-Aware Refinement Network" page (19 Nov 2026), with a small nudge
(``NUDGE``) measuring each candidate and the start range measured from c1.

**One pass** (any number of them; the first starts from frozen c1's guess)::

    guess   = generator(sweep, u)                      forward only
    d_1..8  = RefineNet(recording, guess, u, log sweep ESR)    unit, tangent to u
    D_i     = (pre(generator(sweep, normalize(u + h d_i))) - pre(guess)) / h
    c       = argmin || pre(recording) - pre(guess) - sum c_i D_i ||^2      8 unknowns
    move    = sum c_i d_i

At use the full, half and quarter move are tested on the sweep; the best is kept only
if it lowers the sweep ESR, else the amp stops. All test-step renders are fp32 with
TF32 off, so the small nudge is not lost to rounding.

**RefineNet** hears the whole sweep (or, in training, a crop starting on a 4 s
boundary, where log-mel frames line up with the whole sweep's), never the dry audio:

* mel branch, per log-mel cell: A = (recording - average amp) / spread,
  B = (recording - guess) / spread, C = the dry sweep's log-mel (one mean/std).
  Average and spread over the training amps' own recordings of the sweep, stored once;
  spread floored at ``SPREAD_FLOOR_DB``.
* wave branch: 5 strided convolutions on [recording, error / RMS(error)].
* join (Linear), 2 Conformer blocks, then one mean per second of the sweep plus a
  learned tag for that second; a 4-layer Transformer reads those pieces with the
  current vector, the log sweep ESR and 8 learned question items; each question item
  becomes one candidate direction.

**Training.** Generator and c1 frozen; RefineNet from random weights. Per step 16 real
+ 16 made-up amps, one crop, one 2 s EGDB clip per amp, one pass. Starts: made-up
amps at their true vector turned by a random angle in [0, theta_max] (1 in 8 at
exactly the truth); real amps, half the same way around their table vector, a quarter
from c1's guess on the crop, a quarter from where the network left that amp last
time (c1's guess the first time, and with chance 1 in 10). theta_max = the 95th
percentile of c1's angle to the table on c1's val amps. Renders and the mix solve run
without gradients; the loss (the generator's own) is taken at normalize(u + move) on
the EGDB clip, the gradient reaching the directions with c held fixed. No step-size
test in training.

**Validation**: c1's 76 val amps from c1's guess on the whole sweep, through
``--val-passes`` with the keep rule; loss on c1's fixed EGDB val clips; best checkpoint
by the loss after ``--select-pass``. **Report**: the generator's holdout through the
same passes; the unfold stage's report half scored with its own scorer, plus the
feedback check (the network fed another amp's recording; the test step still uses the
amp's own). ``--score <out>`` scores on a machine with the corpus renders.
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
from openamp.evaluation import load_model, resolve_device
from openamp.losses import build_loss, pre_emphasis, preemph_esr
from openamp.models import count_parameters
from openamp.probes.probe_conformer import (ConformerBlock, ConformerEncoder, RawFrames,
                                            StepClips, amp_weights, read_clean)
from openamp.probes.probe_synth import (HOP, N_MELS, SKIP_SECONDS, LogMel, TableCloud,
                                        captures_by_sha, save_atomic, step_rng)
from openamp.spec import spec_from_checkpoint
from openamp.training import GRAD_CLIP_NORM

N_CAND = 8                  # candidate directions per pass
NUDGE = 0.02                # radians: how far each candidate is nudged to measure it
STEP_SIZES = (1.0, 0.5, 0.25)
SPREAD_FLOOR_DB = 1.0       # smallest spread of amps per cell (silent cells have ~0)
EXACT_SHARE = 1 / 8         # noisy starts placed exactly at the truth
RESET_CHANCE = 0.1          # saved starts reset to c1's guess
THETA_PERCENTILE = 95
CROP_GRID_SECONDS = 4       # 4 s = 375 log-mel frames: crop frames line up with the sweep's
ESR_FLOOR = 1e-6            # log(ESR + floor) for the error-level input
C1_MODEL_ARGS = ("width", "n_blocks", "n_heads", "ff_ratio", "conv_kernel", "n_queries",
                 "d_model", "pool_heads", "mlp_ratio")
C1_SHARED = ("run", "sweep", "captures", "renders", "seed")
RESUME_FIXED = ("init", "run", "sweep", "captures", "renders", "seed", "batch",
                "synth_per_batch", "steps", "eval_every", "ref_seconds", "clip_seconds",
                "val_clips", "val_passes", "select_pass", "lr", "warmup", "min_lr",
                "weight_decay", "width", "n_blocks", "n_heads", "ff_ratio", "conv_kernel",
                "d_model", "n_layers", "reader_heads")


# --- Network --------------------------------------------------------------------------
def piece_index(n_frames: int, n_samples: int, sr: int) -> tuple[torch.Tensor, int]:
    """``([n_frames] piece of each frame, n_pieces)``: frame ``t`` (centred on sample
    ``t * HOP``) belongs to second ``t * HOP // sr``; the last frame or two past the
    end join the last piece."""
    n_pieces = math.ceil(n_samples / sr)
    idx = torch.clamp(torch.arange(n_frames) * HOP // sr, max=n_pieces - 1)
    return idx, n_pieces


class RefineNet(nn.Module):
    """``(recording, guess [B, n], u [B, De], log_esr [B], start_s)`` -> ``[B, k, De]``
    unit candidate directions, each tangent to its ``u``.

    ``start_s`` is where the audio starts, in whole seconds after the sweep's lead-in;
    a multiple of ``CROP_GRID_SECONDS`` so its log-mel frames are the sweep's own.
    """

    def __init__(self, embedding_dim: int, sample_rate: int, n_frames: int, n_pieces: int, *,
                 width: int, n_blocks: int, n_heads: int, ff_ratio: int, conv_kernel: int,
                 d_model: int, n_layers: int, reader_heads: int, n_cand: int = N_CAND,
                 raw_chunk: int = 0):
        super().__init__()
        self.sr, self.raw_chunk, self.n_cand = int(sample_rate), int(raw_chunk), int(n_cand)
        self.logmel = LogMel(sample_rate, 0)
        # Stored once (set_stats): [n_frames, N_MELS] each, for the whole sweep.
        self.register_buffer("dry_mel", torch.zeros(n_frames, N_MELS))
        self.register_buffer("amp_mean", torch.zeros(n_frames, N_MELS))
        self.register_buffer("amp_spread", torch.ones(n_frames, N_MELS))
        self.raw = RawFrames(width)
        self.fuse = nn.Linear(width + 3 * N_MELS, width)
        self.blocks = nn.ModuleList(ConformerBlock(width, n_heads, ff_ratio, conv_kernel)
                                    for _ in range(n_blocks))
        self.piece = nn.Linear(width, d_model)
        self.tags = nn.Embedding(n_pieces, d_model)
        self.vec_in = nn.Linear(embedding_dim, d_model)
        self.err_in = nn.Linear(1, d_model)
        self.questions = nn.Parameter(0.02 * torch.randn(n_cand, d_model))
        layer = nn.TransformerEncoderLayer(d_model, reader_heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.reader = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, embedding_dim)

    @torch.no_grad()
    def set_stats(self, dry_db: torch.Tensor, mean_db: torch.Tensor, spread_db: torch.Tensor):
        self.dry_mel.copy_((dry_db - dry_db.mean()) / dry_db.std())
        self.amp_mean.copy_(mean_db)
        self.amp_spread.copy_(spread_db.clamp_min(SPREAD_FLOOR_DB))

    def mel(self, y: torch.Tensor) -> torch.Tensor:
        """``[B, n]`` -> ``[B, T, N_MELS]`` dB, fp32 even under autocast."""
        with torch.autocast(y.device.type, enabled=False):
            return self.logmel(y.float())

    def forward(self, rec: torch.Tensor, guess: torch.Tensor, u: torch.Tensor,
                log_esr: torch.Tensor, start_s: int) -> torch.Tensor:
        B, n = rec.shape
        if start_s % CROP_GRID_SECONDS:
            raise ValueError(f"start {start_s} s is not on the {CROP_GRID_SECONDS} s grid")
        m_rec, m_guess = self.mel(rec), self.mel(guess)
        T = m_rec.shape[1]
        f0 = start_s * self.sr // HOP
        if f0 + T > self.dry_mel.shape[0]:
            raise ValueError(f"frames {f0}..{f0 + T} run past the stored sweep")
        mean, spread = self.amp_mean[f0:f0 + T], self.amp_spread[f0:f0 + T]
        mel = torch.cat([(m_rec - mean) / spread, (m_rec - m_guess) / spread,
                         self.dry_mel[f0:f0 + T].expand(B, -1, -1)], dim=-1)
        err = rec - guess
        err = err / err.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        x = torch.stack([rec, err], dim=1)
        if self.raw_chunk > 0 and torch.is_grad_enabled():
            raw = torch.cat([checkpoint(self.raw, c, T, use_reentrant=False)
                             for c in x.split(self.raw_chunk)])
        else:
            raw = self.raw(x, T)
        h = self.fuse(torch.cat([raw, mel.to(raw.dtype)], dim=-1))
        for block in self.blocks:
            h = block(h)
        idx, P = piece_index(T, n, self.sr)
        idx = idx.to(h.device)
        sums = h.new_zeros(B, P, h.shape[-1], dtype=torch.float32).index_add_(1, idx, h.float())
        counts = torch.bincount(idx, minlength=P).clamp_min(1).float()
        tag_ids = torch.arange(start_s, start_s + P, device=h.device)
        pieces = self.piece(sums / counts[None, :, None]) + self.tags(tag_ids)[None]
        tokens = torch.cat([pieces, self.vec_in(u.to(pieces.dtype))[:, None],
                            self.err_in(log_esr.to(pieces.dtype)[:, None])[:, None],
                            self.questions.to(pieces.dtype).expand(B, -1, -1)], dim=1)
        q = self.norm(self.reader(tokens)[:, -self.n_cand:])
        return tangent_unit(self.out(q).float(), u.float())


def tangent_unit(d: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``[B, k, De]`` directions with their part along ``u`` ``[B, De]`` removed, unit
    length: a move along ``u`` itself changes nothing after normalizing."""
    u = F.normalize(u, dim=-1)
    d = d - (d * u[:, None]).sum(-1, keepdim=True) * u[:, None]
    return F.normalize(d, dim=-1)


# --- The test step: no learned weights ------------------------------------------------
def sweep_esr(rec_p: torch.Tensor, y_p: torch.Tensor) -> torch.Tensor:
    """Per-row ESR of pre-emphasized ``y_p`` against ``rec_p`` ``[N, n]``."""
    return (rec_p - y_p).double().pow(2).sum(-1) / rec_p.double().pow(2).sum(-1).clamp_min(1e-30)


@torch.no_grad()
def solve_mix(play, rec_p: torch.Tensor, y0_p: torch.Tensor, u: torch.Tensor,
              d: torch.Tensor, coeff: float, chunk: int, h: float = NUDGE) -> torch.Tensor:
    """``[B, k]`` mix of the directions ``d`` ``[B, k, De]`` that best cancels each amp's
    remaining error, from one nudged render per direction.

    ``play(units [S, De]) -> [S, n]`` plays the span with unit vectors (fp32); ``rec_p``,
    ``y0_p`` ``[B, n - 1]`` the pre-emphasized recording and current guess.
    """
    B, k, _ = d.shape
    per = max(1, int(chunk) // k)
    out = []
    for s in range(0, B, per):
        ub, db = u[s:s + per].float(), d[s:s + per].float()
        Y = play(F.normalize(ub[:, None] + h * db, dim=-1).reshape(-1, ub.shape[-1]))
        D = ((pre_emphasis(Y, coeff).view(len(ub), k, -1) - y0_p[s:s + per, None]) / h).double()
        r = (rec_p[s:s + per] - y0_p[s:s + per]).double()
        A = D @ D.transpose(1, 2)                                       # [b, k, k]
        b = (D @ r[:, :, None])[..., 0]                                 # [b, k]
        lam = 1e-6 * A.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-30)
        A = A + lam[:, None, None] * torch.eye(k, dtype=A.dtype, device=A.device)
        out.append(torch.linalg.solve(A, b).float())
    return torch.cat(out)


@torch.no_grad()
def keep_best(play, rec_p: torch.Tensor, u: torch.Tensor, move: torch.Tensor,
              e0: torch.Tensor, coeff: float):
    """Test the full, half and quarter ``move`` on the span for each amp; keep the best
    if it lowers the sweep ESR. Returns ``(u, esr, y, improved [B] bool)``; ``y`` is the
    render at the kept vector, or ``None`` where nothing improved."""
    B = len(u)
    best_e, best_u = e0.clone(), u.clone()
    best_y = [None] * B
    for s in STEP_SIZES:
        cand = F.normalize(u + s * move, dim=-1)
        Y = play(cand)
        e = sweep_esr(rec_p, pre_emphasis(Y, coeff))
        for i in range(B):
            if e[i] < best_e[i]:
                best_e[i], best_u[i], best_y[i] = e[i], cand[i], Y[i]
    return best_u, best_e, best_y, best_e < e0


@torch.no_grad()
def refine(net: RefineNet, play, rec: torch.Tensor, u0: torch.Tensor, passes: int,
           coeff: float, chunk: int, *, net_rec: torch.Tensor | None = None):
    """Use-time passes over whole recordings, one amp at a time.

    ``rec`` ``[N, n]`` (CPU), ``u0`` ``[N, De]``; ``play`` plays the whole sweep.
    ``net_rec``, if given, is what the *network* hears in place of each amp's own
    recording (the feedback check); the test step always uses ``rec``. Returns
    ``(us [passes + 1, N, De], esr [passes + 1, N])``: an amp that stopped keeps its
    vector and ESR.
    """
    net.eval()
    dev = u0.device
    N = len(u0)
    us = torch.zeros(passes + 1, N, u0.shape[-1])
    es = torch.zeros(passes + 1, N, dtype=torch.float64)
    for i in range(N):
        r = rec[i].to(dev)[None]
        r_p = pre_emphasis(r, coeff)
        hear = r if net_rec is None else net_rec[i].to(dev)[None]
        u = u0[i:i + 1].float()
        y = play(u)
        e = sweep_esr(r_p, pre_emphasis(y, coeff))
        us[0, i], es[0, i] = u[0].cpu(), e[0].cpu()
        stopped = False
        for p in range(1, passes + 1):
            if not stopped:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    d = net(hear, y, u, torch.log(e.float() + ESR_FLOOR), 0)
                y_p = pre_emphasis(y, coeff)
                c = solve_mix(play, r_p, y_p, u, d, coeff, chunk)
                move = (c[:, :, None] * d).sum(1)
                u_new, e_new, y_new, ok = keep_best(play, r_p, u, move, e, coeff)
                if bool(ok[0]):
                    u, e, y = u_new, e_new, y_new[0][None]
                else:
                    stopped = True
            us[p, i], es[p, i] = u[0].cpu(), e[0].cpu()
    return us, es


# --- Scoring the holdout vectors ------------------------------------------------------
def score_report(out_dir: Path, device: str = "cuda") -> dict | None:
    """Score a finished run's report-amp vectors with the unfold stage's scorer."""
    from openamp.enroll.fit import TEST_PAIRS_PER_DEVICE
    from openamp.training.unfold import score

    out_dir = Path(out_dir)
    ck = torch.load(out_dir / "probe_refine.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    rep_ids = [int(d) for d in ck["report_ids"]]
    paths = CorpusPaths.from_config(load_config())
    missing = [d for d in rep_ids if not paths.render_file(d, "x").parent.is_dir()]
    if missing:
        print(f"[rf] report not scored: no corpus renders here for {len(missing)} of "
              f"{len(rep_ids)} report amps (e.g. {missing[:3]}). Run\n"
              f"       python -m openamp.probes.probe_refine --score {out_dir}\n"
              f"     on a machine that has them.", flush=True)
        return None
    dev = torch.device(resolve_device(device))
    gen, gen_ck = load_model(Path(a["run"]), str(dev))
    gen.requires_grad_(False)
    gen.eval()
    spec = spec_from_checkpoint(gen_ck)
    clip = int(round(a["clip_seconds"] * spec.sample_rate))
    row = {d: i for i, d in enumerate(int(d) for d in ck["holdout_ids"])}
    sel = torch.as_tensor([row[d] for d in rep_ids])

    def esr(u: torch.Tensor) -> np.ndarray:
        return score(gen, gen.project_embedding(u.to(dev)), rep_ids, paths, "test",
                     TEST_PAIRS_PER_DEVICE, seed=spec.seed + 7, clip=clip, batch_size=16,
                     num_workers=a["num_workers"], dev=dev)

    cols = {"device_id": rep_ids}
    summary = {}
    for p, u in ck["holdout_u"].items():
        cols[f"pass{p}"] = esr(u[sel])
        summary[f"report_pass{p}_median"] = float(np.nanmedian(cols[f"pass{p}"]))
        summary[f"report_pass{p}_mean"] = float(np.nanmean(cols[f"pass{p}"]))
        print(f"[rf] report amps after pass {p}: median {summary[f'report_pass{p}_median']:.4f} "
              f"mean {summary[f'report_pass{p}_mean']:.4f}" + ("  (c1 alone)" if p == 0 else ""),
              flush=True)
    sp = ck["swap_pass"]
    cols["swapped"] = esr(ck["report_u_swap"])
    summary["report_swapped_median"] = float(np.nanmedian(cols["swapped"]))
    summary["report_swapped_mean"] = float(np.nanmean(cols["swapped"]))
    print(f"[rf] feedback check, network fed another amp's recording, after pass {sp}: "
          f"median {summary['report_swapped_median']:.4f} mean "
          f"{summary['report_swapped_mean']:.4f} (must be worse than pass {sp})", flush=True)
    pd.DataFrame(cols).to_csv(out_dir / "report_per_device.csv", index=False)
    rp = out_dir / "report.json"
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.is_file() else {}
    report.update(summary)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[rf] wrote {out_dir}/report_per_device.csv and report.json")
    return summary


# --- Main -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", type=Path, default=None,
                    help="score a finished run's report amps (needs the corpus renders) and exit")
    ap.add_argument("--init", type=Path,
                    default=Path("results/encoder/encoder_probe_conformer_c1/probe_conformer.pt"),
                    help="the c1 run: its frozen encoder gives every first guess")
    ap.add_argument("--run", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2_n10"))
    ap.add_argument("--sweep", type=Path,
                    default=Path("data_sweep_a1/clean/train/sweep_train.flac"))
    ap.add_argument("--captures", type=Path, default=Path("data/captures"))
    ap.add_argument("--renders", type=Path, default=Path("data/manifests/renders.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/encoder/encoder_probe_refine"))
    ap.add_argument("--batch", type=int, default=32, help="amps per step, real + made-up")
    ap.add_argument("--synth-per-batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ref-seconds", type=int, default=84,
                    help="whole seconds of sweep per training step (crop on a 4 s grid)")
    ap.add_argument("--clip-seconds", type=float, default=2.0)
    ap.add_argument("--val-clips", type=int, default=8)
    ap.add_argument("--val-passes", default="1,2,3,5,10")
    ap.add_argument("--select-pass", type=int, default=3)
    ap.add_argument("--report-passes", default="0,1,2,3,5,10")
    ap.add_argument("--swap-passes", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ff-ratio", type=int, default=4)
    ap.add_argument("--conv-kernel", type=int, default=15)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--reader-heads", type=int, default=8)
    ap.add_argument("--render-chunk", type=int, default=16,
                    help="vectors the generator plays at once (whole sweep or crop)")
    ap.add_argument("--raw-chunk", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()
    if args.score is not None:
        score_report(args.score, args.device)
        return
    if not 0 <= args.synth_per_batch < args.batch:
        raise SystemExit("--synth-per-batch must be in [0, --batch)")
    val_passes = sorted({int(p) for p in args.val_passes.split(",")})
    report_passes = sorted({0} | {int(p) for p in args.report_passes.split(",")})
    if args.select_pass not in val_passes:
        raise SystemExit("--select-pass must be one of --val-passes")

    # The test step's renders must not round the nudge away: no TF32 anywhere fp32 runs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
            raise SystemExit(f"--resume: these settings differ from the run being resumed: {diff}")
    init = torch.load(args.init, map_location="cpu", weights_only=False)
    ia = init["args"]
    diff = {k: (ia[k], arg_dict[k]) for k in C1_SHARED if ia[k] != arg_dict[k]}
    if diff:
        raise SystemExit(f"--init was trained with different settings: {diff}")
    val_frac = float(ia["val_frac"])

    # --- Frozen generator, its loss, its table --------------------------------------
    model, gen_ck = load_model(args.run, str(dev))
    model.requires_grad_(False)
    model.eval()
    R = model.receptive_field
    spec = spec_from_checkpoint(gen_ck)
    lossfn = build_loss(spec.loss).to(dev)
    coeff = float(lossfn.coeff)
    strength = float(spec.loss.device_loss_weighting)
    tbl = torch.load(args.run / "embedding.pt", map_location="cpu", weights_only=False)
    table = tbl["embedding"].float().numpy()
    ids = np.array([int(d) for d in tbl["device_ids"]])
    embedding_dim = table.shape[1]
    scale = float(np.linalg.norm(table, axis=1).mean())
    table_unit = torch.from_numpy(table / np.linalg.norm(table, axis=1, keepdims=True))
    row_of = {int(d): j for j, d in enumerate(ids)}
    holdout = sorted(int(d) for d in gen_ck["holdout_ids"])

    rend = pd.read_parquet(args.renders, columns=["device_id", "output_scale", "nam_sha256"])
    per_dev = rend.groupby("device_id").agg(scale=("output_scale", "first"),
                                            n_scale=("output_scale", "nunique"),
                                            sha=("nam_sha256", "first"))
    if (per_dev["n_scale"] > 1).any():
        raise SystemExit(f"{args.renders}: an amp with more than one output_scale")
    scales = {int(d): float(s) for d, s in per_dev["scale"].items()}
    found = captures_by_sha(args.captures, set(per_dev["sha"]))
    nam_paths = {int(d): found[h] for d, h in per_dev["sha"].items() if h in found}
    missing = [int(d) for d in list(ids) + holdout if int(d) not in nam_paths]
    if missing:
        raise SystemExit(f"no .nam under {args.captures} for {len(missing)} amp(s), e.g. {missing[:5]}")

    perm = np.random.default_rng(args.seed).permutation(len(ids))     # c1's split
    n_val = max(1, int(round(val_frac * len(ids))))
    val_sel, train_sel = perm[:n_val], perm[n_val:]
    val_ids = ids[val_sel]
    n_synth = args.synth_per_batch
    n_real = args.batch - n_synth
    n_noisy, n_c1 = n_real // 2, n_real // 4

    # --- The sweep ------------------------------------------------------------------
    x, sr = audio_io.read_audio(args.sweep)
    x = np.asarray(x, dtype=np.float32)
    if sr != cfg.sample_rate:
        raise SystemExit(f"{args.sweep} is {sr} Hz, the corpus is {cfg.sample_rate} Hz")
    if (CROP_GRID_SECONDS * sr) % HOP:
        raise SystemExit(f"{CROP_GRID_SECONDS} s is not a whole number of frames at {sr} Hz")
    skip = int(round(SKIP_SECONDS * sr))
    n_keep = len(x) - skip
    L_ref = int(args.ref_seconds) * sr
    L_clip = int(round(args.clip_seconds * sr))
    starts = list(range(0, int((n_keep - L_ref) // sr) + 1, CROP_GRID_SECONDS))
    if not starts:
        raise SystemExit(f"the sweep has {n_keep / sr:.1f} s after the lead-in; crop is {args.ref_seconds} s")
    dry_full = torch.from_numpy(x[skip:]).to(dev)
    x_pad_gen = torch.from_numpy(np.concatenate([np.zeros(R, np.float32), x])).to(dev)
    seg_full = x_pad_gen[skip:]                                   # [R + n_keep]
    logmel = LogMel(sr, 0).to(dev)
    n_frames_full = n_keep // HOP + 1
    _, n_pieces_full = piece_index(n_frames_full, n_keep, sr)

    def load_nam(d: int):
        m = nam_io.load_model(nam_paths[int(d)], device=str(dev))
        if m.sample_rate != cfg.sample_rate:
            raise SystemExit(f"amp {d}: .nam at {m.sample_rate} Hz")
        return m

    t0 = time.time()
    nams = {int(d): load_nam(d) for d in ids}
    R_nam = max(m.receptive_field for m in nams.values())
    x_pad_nam = np.concatenate([np.zeros(R_nam, np.float32), x])
    print(f"[rf] loaded {len(nams)} .nam captures ({time.time() - t0:.0f} s)", flush=True)

    def real_full(m, d: int) -> torch.Tensor:
        y = _render_signal(m, x, cfg, False) * scales[int(d)]
        return torch.from_numpy(y[skip:].astype(np.float32))

    def real_crop(d: int, g: int) -> np.ndarray:
        m = nams[int(d)]
        seg = x_pad_nam[R_nam - m.receptive_field + g: R_nam + g + L_ref]
        return (m.forward(seg)[-L_ref:] * scales[int(d)]).astype(np.float32)

    R_clip = max(R, R_nam)

    def real_clip(d: int, xc: np.ndarray) -> np.ndarray:
        m = nams[int(d)]
        return (m.forward(xc[R_clip - m.receptive_field:])[-L_clip:] * scales[int(d)]).astype(np.float32)

    @torch.no_grad()
    def play16(xin: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """``[S, R + n]`` input, effective ``[S, De]`` -> ``[S, n]``, bf16 (EGDB clips)."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            return model.forward_emb(xin[:, None], E)[:, 0, R:].float()

    def player(seg: torch.Tensor):
        """``play(units [S, De]) -> [S, n]``: the generator on one shared span, fp32."""
        @torch.no_grad()
        def play(units: torch.Tensor) -> torch.Tensor:
            out = []
            for c in units.split(args.render_chunk):
                with torch.autocast(dev.type, enabled=False):
                    out.append(model.forward_emb(seg[None].expand(len(c), -1)[:, None],
                                                 scale * c.float())[:, 0, R:])
            return torch.cat(out)
        return play

    play_full = player(seg_full)

    # --- c1, frozen -----------------------------------------------------------------
    c1 = ConformerEncoder(embedding_dim, sr, **{k: ia[k] for k in C1_MODEL_ARGS}).to(dev)
    c1.load_state_dict(init["state_dict"])
    c1.requires_grad_(False)
    c1.eval()
    del init

    @torch.no_grad()
    def c1_guess(dry: torch.Tensor, wet: torch.Tensor, chunk: int) -> torch.Tensor:
        out = []
        for s in range(0, len(wet), chunk):
            w = wet[s:s + chunk].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(c1(dry[None].expand(len(w), -1), w).float())
        return torch.cat(out)

    wet_val = torch.stack([real_full(nams[int(d)], d) for d in val_ids])     # [V, n_keep]
    u0_val = c1_guess(dry_full, wet_val, 1)
    print(f"[rf] rendered {len(val_ids)} val amps on the sweep, c1 guessed each", flush=True)

    # --- The network, and what it stores once ---------------------------------------
    sizes = dict(width=args.width, n_blocks=args.n_blocks, n_heads=args.n_heads,
                 ff_ratio=args.ff_ratio, conv_kernel=args.conv_kernel, d_model=args.d_model,
                 n_layers=args.n_layers, reader_heads=args.reader_heads)
    net = RefineNet(embedding_dim, sr, n_frames_full, n_pieces_full, **sizes,
                    raw_chunk=args.raw_chunk).to(dev)
    if ck is None:
        t0 = time.time()
        with torch.no_grad():
            s1 = torch.zeros(n_frames_full, N_MELS, dtype=torch.float64, device=dev)
            s2 = torch.zeros_like(s1)
            for j in train_sel:
                d = int(ids[j])
                mdb = logmel(real_full(nams[d], d).to(dev)[None])[0].double()
                s1 += mdb
                s2 += mdb ** 2
            mean = s1 / len(train_sel)
            spread = (s2 / len(train_sel) - mean ** 2).clamp_min(0).sqrt()
            net.set_stats(logmel(dry_full[None])[0], mean.float(), spread.float())
        theta = torch.arccos((u0_val.cpu() * table_unit[val_sel]).sum(-1).clamp(-1, 1))
        theta_max = float(np.percentile(theta.numpy(), THETA_PERCENTILE))
        print(f"[rf] stored the sweep's log-mel and the average / spread of the "
              f"{len(train_sel)} training amps on it ({time.time() - t0:.0f} s); c1's angle "
              f"to the table on its val amps: median {math.degrees(float(theta.median())):.1f}"
              f" deg, {THETA_PERCENTILE}th pct {math.degrees(theta_max):.1f} deg = theta_max",
              flush=True)
    else:
        theta_max = float(ck["theta_max"])

    # --- EGDB clips -----------------------------------------------------------------
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
    p_val = (wet_val.double() ** 2).mean(1).float()
    print(f"[rf] val: {len(val_ids)} amps x {args.val_clips} EGDB val clips", flush=True)

    cloud = TableCloud(table[train_sel], scale)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best_val, best_step, best_state = 0, float("inf"), -1, None
    saved: dict[int, torch.Tensor] = {}
    if ck is not None:
        net.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start, best_val, best_step = ck["step"] + 1, ck["best_val"], ck["best_step"]
        best_state, saved = ck["best_state"], ck["saved"]
        print(f"[rf] resumed {args.resume} at step {start} (best val {best_val:.4f} at "
              f"step {best_step + 1})")
    n_params = count_parameters(net)
    print(f"[rf] {len(ids)} amps: {len(train_sel)} train / {len(val_sel)} val  "
          f"decoder={args.run.name}  radius {scale:.4f}  loss={spec.loss.name}")
    print(f"[rf] per step: {n_real} real ({n_noisy} noisy, {n_c1} c1, "
          f"{n_real - n_noisy - n_c1} saved) + {n_synth} made-up amps, crop "
          f"{args.ref_seconds} s on a {CROP_GRID_SECONDS} s grid ({len(starts)} places), "
          f"loss clip {args.clip_seconds:g} s, {N_CAND} candidates, nudge {NUDGE}")
    print(f"[rf] RefineNet from random weights, {n_params:,} params; theta_max "
          f"{math.degrees(theta_max):.1f} deg", flush=True)

    @torch.no_grad()
    def clip_loss(u: torch.Tensor) -> dict:
        E = scale * u
        w_amp = amp_weights(p_val, strength).to(dev)
        tot = esr_t = 0.0
        nb = 0
        for s in range(0, len(vx), args.batch):
            a = va[s:s + args.batch].to(dev)
            y = play16(vx[s:s + args.batch].to(dev), E[a])
            w = w_amp[a] / w_amp[a].mean()
            loss, parts = lossfn(y[:, None], vt[s:s + args.batch].to(dev)[:, None], w)
            tot += float(loss)
            esr_t += parts["esr"]
            nb += 1
        return {"loss": tot / nb, "esr": esr_t / nb}

    def validate() -> tuple[list[dict], dict]:
        us, es = refine(net, play_full, wet_val, u0_val, max(val_passes), coeff, args.render_chunk)
        rows = [dict(clip_loss(us[p].to(dev)), sweep=float(es[p].median()), p=p)
                for p in [0] + val_passes]
        moved = {p: int((es[p] < es[p - 1]).sum()) for p in val_passes if p >= 1}
        return rows, moved

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    def run_state(step: int) -> dict:
        return {"state_dict": net.state_dict(), "opt": opt.state_dict(), "step": step,
                "best_val": best_val, "best_step": best_step, "best_state": best_state,
                "saved": saved, "theta_max": theta_max, "args": arg_dict,
                "device_ids": ids.tolist()}

    def unit_tangent_noise(rng: np.random.Generator, u_true: torch.Tensor) -> torch.Tensor:
        """Each row of ``u_true`` turned by an angle in [0, theta_max] (exactly 0 for a
        share of ``EXACT_SHARE``) toward a random direction."""
        n = len(u_true)
        th = rng.uniform(0.0, theta_max, n)
        th[rng.random(n) < EXACT_SHARE] = 0.0
        g = torch.from_numpy(rng.standard_normal((n, u_true.shape[1])).astype(np.float32))
        w = tangent_unit(g[:, None], u_true)[:, 0]
        th = torch.from_numpy(th.astype(np.float32))[:, None]
        return F.normalize(torch.cos(th) * u_true + torch.sin(th) * w, dim=-1)

    # --- Train ----------------------------------------------------------------------
    clips = StepClips(train_ds, train_sel, ids, n_real, n_synth, args.seed)
    loader = torch.utils.data.DataLoader(
        clips, batch_size=None, sampler=range(start, args.steps),
        num_workers=args.num_workers, pin_memory=dev.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None)

    def fresh_acc() -> dict:
        return {"loss": 0.0, "n": 0, "real0": 0.0, "real1": 0.0, "syn0": 0.0, "syn1": 0.0}

    acc = fresh_acc()
    t0, t_wait, t_render = time.time(), 0.0, 0.0
    tw = time.time()
    step = start - 1
    for item in loader:
        step = int(item["step"])
        t_wait += time.time() - tw
        net.train()
        rng = step_rng(args.seed, 3, step)
        s0 = int(starts[int(rng.integers(0, len(starts)))])
        g = skip + s0 * sr
        seg = x_pad_gen[g:g + R + L_ref]
        play = player(seg)
        xc = item["x"]
        xin = xc[:, R_clip - R:].to(dev, non_blocking=True)
        dry = torch.from_numpy(x[g:g + L_ref]).to(dev)
        srng = step_rng(args.seed, 6, step)

        tr0 = time.time()
        rid = [int(d) for d in item["real_ids"]]
        rec = [torch.from_numpy(np.stack([real_crop(d, g) for d in rid])).to(dev)]
        tgt = [torch.from_numpy(np.stack([real_clip(d, xc[i].numpy()) for i, d in enumerate(rid)])).to(dev)]
        u_true_real = table_unit[[row_of[d] for d in rid]]
        u_real = unit_tangent_noise(srng, u_true_real).to(dev)              # noisy by default
        guess_c1 = c1_guess(dry, rec[0][n_noisy:], 4)
        u_real[n_noisy:] = guess_c1
        for i in range(n_noisy + n_c1, n_real):
            d = rid[i]
            if d in saved and srng.random() >= RESET_CHANCE:
                u_real[i] = saved[d].to(dev)
        if n_synth:
            E = torch.from_numpy(cloud.draw(step_rng(args.seed, 2, step), n_synth)).to(dev)
            u_true_syn = F.normalize(E, dim=-1)
            rec.append(play(u_true_syn))
            tgt.append(play16(xin[n_real:], E))
            u = torch.cat([u_real, unit_tangent_noise(srng, u_true_syn.cpu()).to(dev)])
        else:
            u = u_real
        rec, tgt = torch.cat(rec), torch.cat(tgt)
        w = amp_weights((rec.double() ** 2).mean(1).float(), strength)
        y0 = play(u)
        rec_p, y0_p = pre_emphasis(rec, coeff), pre_emphasis(y0, coeff)
        e0 = sweep_esr(rec_p, y0_p).float()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_render += time.time() - tr0

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            d_dirs = net(rec, y0, u, torch.log(e0 + ESR_FLOOR), s0)
        tr0 = time.time()
        c = solve_mix(play, rec_p, y0_p, u, d_dirs.detach(), coeff, args.render_chunk)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_render += time.time() - tr0
        u_new = F.normalize(u + (c[:, :, None] * d_dirs).sum(1), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            y = model.forward_emb(xin[:, None], scale * u_new)[:, :, R:]
        loss, _ = lossfn(y.float(), tgt[:, None], w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        opt.step()

        with torch.no_grad():
            y_start = play16(xin, scale * u)[:, None]
            yf = y.float()
            acc["real0"] += float(preemph_esr(y_start[:n_real], tgt[:n_real, None], coeff, weights=w[:n_real]))
            acc["real1"] += float(preemph_esr(yf[:n_real], tgt[:n_real, None], coeff, weights=w[:n_real]))
            if n_synth:
                acc["syn0"] += float(preemph_esr(y_start[n_real:], tgt[n_real:, None], coeff, weights=w[n_real:]))
                acc["syn1"] += float(preemph_esr(yf[n_real:], tgt[n_real:, None], coeff, weights=w[n_real:]))
            for i, d in enumerate(rid):
                saved[d] = u_new[i].detach().cpu()
        acc["loss"] += float(loss.detach())
        acc["n"] += 1

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            tv = time.time()
            rows, moved = validate()
            sel_loss = next(r["loss"] for r in rows if r["p"] == args.select_pass)
            star = sel_loss < best_val - 1e-6
            if star:
                best_val, best_step = sel_loss, step
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            n = max(acc["n"], 1)
            elapsed = time.time() - t0

            def arrows(key):
                return ">".join(f"{r[key]:.3f}" for r in rows)
            print(f"  s{step + 1:05d}  train loss {acc['loss'] / n:.4f}  clip esr start>after: "
                  f"real {acc['real0'] / n:.3f}>{acc['real1'] / n:.3f} made-up "
                  f"{acc['syn0'] / n:.3f}>{acc['syn1'] / n:.3f}  val by pass "
                  f"{','.join(str(r['p']) for r in rows)}: loss {arrows('loss')} esr "
                  f"{arrows('esr')} sweep {arrows('sweep')} moved "
                  f"{','.join(str(v) for v in moved.values())}/{len(val_ids)}  best "
                  f"{best_val:.4f}  lr {lr_at(step):.2e}  {elapsed / 60:.1f} min (val "
                  f"{(time.time() - tv) / 60:.1f}), data wait {t_wait / max(elapsed, 1e-9):.0%} "
                  f"render {t_render / max(elapsed, 1e-9):.0%}" + ("  *" if star else ""),
                  flush=True)
            acc = fresh_acc()
            save_atomic(run_state(step), args.out / "last.pt")
        tw = time.time()
    train_min = (time.time() - t0) / 60
    print(f"[rf] trained to step {args.steps} in {train_min:.1f} min; best val "
          f"{best_val:.4f} at step {best_step + 1}")

    # --- The generator's holdout: vectors after the report passes, best checkpoint ---
    from openamp.training.unfold import split_holdout

    net.load_state_dict(best_state)
    del nams
    t0 = time.time()
    wet_h = torch.stack([real_full(load_nam(d), d) for d in holdout])
    u0_h = c1_guess(dry_full, wet_h, 1)
    us_h, es_h = refine(net, play_full, wet_h, u0_h, max(report_passes), coeff, args.render_chunk)
    _, rep_ids = split_holdout(holdout)
    rows_r = torch.as_tensor([holdout.index(d) for d in rep_ids])
    nxt = rows_r.roll(-1)
    us_s, _ = refine(net, play_full, wet_h[rows_r], u0_h[rows_r.to(dev)], args.swap_passes,
                     coeff, args.render_chunk, net_rec=wet_h[nxt])
    print(f"[rf] {len(holdout)} holdout amps through {max(report_passes)} passes "
          f"({time.time() - t0:.0f} s); median sweep ESR by pass "
          + " ".join(f"{p}:{float(es_h[p].median()):.4f}" for p in report_passes), flush=True)
    report = {"init": str(args.init), "n_params": int(n_params), "theta_max_deg": math.degrees(theta_max),
              "n_train": int(len(train_sel)), "n_val": int(len(val_sel)), "steps": int(args.steps),
              "best_val_loss": best_val, "best_step": int(best_step + 1),
              "holdout_sweep_esr_median": {p: float(es_h[p].median()) for p in report_passes},
              "train_minutes": train_min}
    torch.save({"state_dict": net.state_dict(), "args": arg_dict, "device_ids": ids.tolist(),
                "val_ids": val_ids.tolist(), "holdout_ids": holdout,
                "holdout_u": {p: us_h[p] for p in report_passes},
                "holdout_sweep_esr": {p: es_h[p] for p in report_passes},
                "report_ids": rep_ids, "report_u_swap": us_s[-1], "swap_pass": args.swap_passes,
                "theta_max": theta_max, "report": report},
               args.out / "probe_refine.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[rf] wrote {args.out}/probe_refine.pt and report.json")
    score_report(args.out, args.device)


if __name__ == "__main__":
    main()
