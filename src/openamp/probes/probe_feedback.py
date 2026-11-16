"""A feedback encoder: c1's guess, corrected by forward passes that hear their own error.

    OPENAMP_DATA_DIR=data python -m openamp.probes.probe_feedback \\
        --out results/encoder/encoder_probe_feedback_f1

**The idea.** A blind encoder (:mod:`openamp.probes.probe_conformer`, run c1) hears an
amp play the reference and guesses its vector in one shot. Here that guess is only the
start. The frozen generator plays the *same* reference with the guess -- a forward pass,
no gradients -- and because the reference is one fixed signal, the amp's recording and
the generator's line up sample by sample: their difference is the amp error alone. A
correction network hears that error and moves the vector; this repeats ``--passes``
times with the same weights::

    u_0 = c1(dry, wet)                                        unit vector, c1 frozen
    err = wet - generator(reference, radius * u_{t-1})        forward only
    u_t = normalize(u_{t-1} + F(dry, wet, err, u_{t-1}))      t = 1 .. passes

**The correction network** (:class:`FeedbackEncoder`) is c1's network and starts from
c1's weights (``--init``), with four changes: ``err`` is a third raw channel and a third
log-mel (standardized by wet's mean and std); the current vector enters through one
Linear, added to every frame after the fuse layer; and the output layer emits the
correction, not the vector. Every new weight starts at zero, so an untrained network
returns a zero correction and the result *is* c1's guess.

**Training** is c1's recipe (the generator's own loss on a different 2 s EGDB clip;
real amps played by their ``.nam``, made-up amps by the generator, never reused; one
reference crop per step shared by every amp; per-amp loss weight from the wet power on
that crop). New here: frozen c1 hears the step's crop and gives ``u_0``; the loss is
taken after every pass and averaged; no gradient flows from one pass into the previous
one (``u_{t-1}`` is a constant), so memory is one pass however many there are.

**Validation** (c1's val amps, never trained on by c1 or here): c1 and the correction
network hear the whole reference; the loss on c1's fixed EGDB val clips is reported
after pass 0 (c1 alone) and after every pass. ``best_val`` is the loss after the last
pass; it picks the checkpoint.

**Report.** The best checkpoint's vectors after every pass are written for the
generator's holdout amps (seen by neither network). The unfold stage's *report* half
(:func:`openamp.training.unfold.split_holdout`) is then scored with that stage's own
scorer, so the numbers read directly against an unfold run's -- together with the
feedback check: the same passes with every amp fed another amp's guessed wet, which
must score worse or the network is ignoring the feedback. Scoring reads the corpus
renders; where they are missing (a cluster holding only the captures) it is skipped,
and ``--score <out>`` runs it later on a machine that has them.

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
from openamp.probes.probe_conformer import (RAW_KERNELS, RAW_STRIDES, ConformerBlock,
                                            ConformerEncoder, RawFrames, StepClips,
                                            amp_weights, read_clean)
from openamp.probes.probe_synth import (N_MELS, SKIP_SECONDS, LogMel, TableCloud,
                                        captures_by_sha, save_atomic, step_rng)
from openamp.spec import spec_from_checkpoint
from openamp.training import GRAD_CLIP_NORM

# The network's own sizes come from --init (c1's), never from flags.
C1_MODEL_ARGS = ("width", "n_blocks", "n_heads", "ff_ratio", "conv_kernel", "n_queries",
                 "d_model", "pool_heads", "mlp_ratio")
# --init must share these with this run (and its --val-frac is taken over): its val
# amps are then this run's val amps.
C1_SHARED = ("run", "sweep", "captures", "renders", "seed")
RESUME_FIXED = ("init", "run", "sweep", "captures", "renders", "seed", "batch",
                "synth_per_batch", "passes", "steps", "eval_every", "ref_seconds",
                "clip_seconds", "val_clips", "lr", "warmup", "min_lr", "weight_decay")
TRAIN_RENDER = 16       # amps the generator plays a crop for at once (what c1's step did)


# --- The correction network -----------------------------------------------------------
class FeedbackEncoder(nn.Module):
    """``(dry, wet, err [B, n], u [B, De]) -> [B, De]`` correction to the unit vector.

    c1's :class:`~openamp.probes.probe_conformer.ConformerEncoder` with ``err`` as a
    third raw channel and a third log-mel, the current vector added to every frame
    after the fuse layer, and an output layer that emits a correction (not
    normalized). ``e_in`` and ``out`` start at zero; :meth:`load_c1` fills the rest.
    """

    def __init__(self, embedding_dim: int, sample_rate: int, *, width: int, n_blocks: int,
                 n_heads: int, ff_ratio: int, conv_kernel: int, n_queries: int,
                 d_model: int, pool_heads: int, mlp_ratio: int, raw_chunk: int = 0):
        super().__init__()
        self.raw_chunk = int(raw_chunk)
        self.logmel = LogMel(sample_rate, 0)
        self.register_buffer("feat_mean", torch.zeros(3 * N_MELS))
        self.register_buffer("feat_std", torch.ones(3 * N_MELS))
        self.raw = RawFrames(width)
        # RawFrames reads dry + wet; here its first conv also reads err.
        self.raw.convs[0] = nn.Conv1d(3, int(width), RAW_KERNELS[0], RAW_STRIDES[0])
        self.fuse = nn.Linear(width + 3 * N_MELS, width)
        self.e_in = nn.Linear(embedding_dim, width)
        self.blocks = nn.ModuleList(ConformerBlock(width, n_heads, ff_ratio, conv_kernel)
                                    for _ in range(n_blocks))
        self.pool = CrossAttentionPool2Block(width, n_queries, d_model, n_heads=pool_heads,
                                             mlp_ratio=mlp_ratio, positional=False)
        self.out = nn.Linear(self.pool.pooled_dim, embedding_dim)
        for layer in (self.e_in, self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def load_c1(self, c1_state: dict) -> None:
        """Start from a c1 encoder's weights; whatever c1 does not have stays zero.

        The err columns of the first raw conv and of the fuse layer are zeroed, and
        err's log-mel takes wet's mean and std. c1's output layer (``proj``) is
        dropped: it emitted the vector itself.
        """
        new = {k: v.clone() for k, v in self.state_dict().items()}
        for k, v in c1_state.items():
            if k.startswith("proj."):
                continue
            if k not in new:
                raise KeyError(f"c1 weight {k!r} has no place in the feedback encoder")
            if k == "raw.convs.0.weight":                   # [width, dry | wet | err, k]
                new[k].zero_()
                new[k][:, :2] = v
            elif k == "fuse.weight":                        # [width, raw | dry | wet | err]
                new[k].zero_()
                new[k][:, :v.shape[1]] = v
            elif k in ("feat_mean", "feat_std"):            # dry | wet -> dry | wet | wet
                new[k] = torch.cat([v, v[N_MELS:]])
            elif new[k].shape != v.shape:
                raise ValueError(f"c1 weight {k!r} is {tuple(v.shape)}, the feedback "
                                 f"encoder's is {tuple(new[k].shape)}")
            else:
                new[k] = v.clone()
        self.load_state_dict(new)

    def features(self, dry: torch.Tensor, wet: torch.Tensor, err: torch.Tensor,
                 u: torch.Tensor) -> torch.Tensor:
        """``[B, pooled_dim]``: everything before the output layer."""
        with torch.autocast(dry.device.type, enabled=False):    # the mel matmul stays fp32
            mel = torch.cat([self.logmel(dry.float()), self.logmel(wet.float()),
                             self.logmel(err.float())], dim=-1)
        mel = (mel - self.feat_mean) / self.feat_std
        x = torch.stack([dry, wet, err], dim=1)
        if self.raw_chunk > 0 and torch.is_grad_enabled():
            raw = torch.cat([checkpoint(self.raw, c, mel.shape[1], use_reentrant=False)
                             for c in x.split(self.raw_chunk)])
        else:
            raw = self.raw(x, mel.shape[1])
        h = self.fuse(torch.cat([raw, mel.to(raw.dtype)], dim=-1))
        h = h + self.e_in(u.to(h.dtype))[:, None]
        for block in self.blocks:
            h = block(h)
        return self.pool(h)

    def forward(self, dry: torch.Tensor, wet: torch.Tensor, err: torch.Tensor,
                u: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(dry, wet, err, u)).float()


def step_vector(u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """One correction: ``normalize(u + delta)``, back on the unit sphere."""
    return F.normalize(u.float() + delta.float(), dim=-1)


@torch.no_grad()
def refine(net: FeedbackEncoder, play, dry: torch.Tensor, wet: torch.Tensor,
           u0: torch.Tensor, passes: int, *, swap: bool = False) -> list[torch.Tensor]:
    """``[u_0, u_1, ..., u_passes]`` for ``N`` amps over one shared ``dry`` ``[n]``.

    ``wet`` ``[N, n]`` (CPU), ``u0`` ``[N, De]`` unit vectors on the network's device,
    ``play(u) -> [N, n]`` (CPU) the generator on ``dry`` with those vectors. Amps go
    through the network one at a time: ``n`` may be the whole reference.

    ``swap`` is the feedback check: amp ``i`` is fed amp ``i + 1``'s guessed wet in
    place of its own. A network that uses the error must then do worse.
    """
    net.eval()
    dev = u0.device
    us, u = [u0], u0
    for _ in range(int(passes)):
        guess = play(u)
        if swap:
            guess = guess.roll(-1, 0)
        delta = []
        for i in range(len(u)):
            w = wet[i].to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                delta.append(net(dry[None], w[None], (w - guess[i].to(dev))[None],
                                 u[i:i + 1]))
        u = step_vector(u, torch.cat(delta))
        us.append(u)
    return us


# --- Scoring the holdout vectors ------------------------------------------------------
def score_report(out_dir: Path, device: str = "cuda") -> dict | None:
    """Score a finished run's report-amp vectors with the unfold stage's scorer.

    Reads ``<out_dir>/probe_feedback.pt``; writes ``report_per_device.csv`` and adds
    the medians and means to ``report.json``. Returns ``None`` (after saying so) when
    the corpus renders are not on this machine.
    """
    from openamp.enroll.fit import TEST_PAIRS_PER_DEVICE
    from openamp.training.unfold import score

    out_dir = Path(out_dir)
    ck = torch.load(out_dir / "probe_feedback.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    rep_ids = [int(d) for d in ck["report_ids"]]
    cfg = load_config()
    paths = CorpusPaths.from_config(cfg)
    missing = [d for d in rep_ids if not paths.render_file(d, "x").parent.is_dir()]
    if missing:
        print(f"[fb] report not scored: no corpus renders here for {len(missing)} of "
              f"{len(rep_ids)} report amps (e.g. {missing[:3]}). Run\n"
              f"       python -m openamp.probes.probe_feedback --score {out_dir}\n"
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
        # The unfold stage's own call: its seed offset, its clip count, its batch.
        return score(gen, gen.project_embedding(u.to(dev)), rep_ids, paths, "test",
                     TEST_PAIRS_PER_DEVICE, seed=spec.seed + 7, clip=clip, batch_size=16,
                     num_workers=a["num_workers"], dev=dev)

    cols = {"device_id": rep_ids}
    summary = {}
    for t, u in enumerate(ck["holdout_u"]):
        cols[f"pass{t}"] = esr(u[sel])
        summary[f"report_pass{t}_median"] = float(np.nanmedian(cols[f"pass{t}"]))
        summary[f"report_pass{t}_mean"] = float(np.nanmean(cols[f"pass{t}"]))
        print(f"[fb] report amps after pass {t}: median {summary[f'report_pass{t}_median']:.4f} "
              f"mean {summary[f'report_pass{t}_mean']:.4f}"
              + ("  (c1 alone)" if t == 0 else ""), flush=True)
    cols["swapped"] = esr(ck["report_u_swap"])
    summary["report_swapped_median"] = float(np.nanmedian(cols["swapped"]))
    summary["report_swapped_mean"] = float(np.nanmean(cols["swapped"]))
    print(f"[fb] feedback check, fed another amp's guessed wet: median "
          f"{summary['report_swapped_median']:.4f} mean "
          f"{summary['report_swapped_mean']:.4f} (must be worse than the last pass)",
          flush=True)

    pd.DataFrame(cols).to_csv(out_dir / "report_per_device.csv", index=False)
    rp = out_dir / "report.json"
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.is_file() else {}
    report.update(summary)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[fb] wrote {out_dir}/report_per_device.csv and report.json")
    return summary


# --- Main -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", type=Path, default=None,
                    help="score a finished run's report amps (needs the corpus renders) "
                         "and exit")
    ap.add_argument("--init", type=Path,
                    default=Path("results/encoder/encoder_probe_conformer_c1/"
                                 "probe_conformer.pt"),
                    help="the c1 run: its frozen encoder gives every first guess, and "
                         "its weights and sizes start the correction network")
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
                    default=Path("results/encoder/encoder_probe_feedback"))
    ap.add_argument("--passes", type=int, default=3, help="corrections per amp")
    ap.add_argument("--batch", type=int, default=32, help="amps per step, real + made-up")
    ap.add_argument("--synth-per-batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=171,
                    help="steps between validations (c1's)")
    ap.add_argument("--ref-seconds", type=float, default=84.4,
                    help="reference crop c1 and the correction network hear per step")
    ap.add_argument("--clip-seconds", type=float, default=2.0,
                    help="EGDB clip the loss is scored on (the decoder trained on 2.0)")
    ap.add_argument("--val-clips", type=int, default=8,
                    help="fixed EGDB val-split clips per val amp")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50, help="steps of linear LR warmup")
    ap.add_argument("--min-lr", type=float, default=1e-5,
                    help="the cosine decay's floor, reached at --steps")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--render-chunk", type=int, default=4,
                    help="amps the generator plays the whole reference for at once")
    ap.add_argument("--raw-chunk", type=int, default=4,
                    help="checkpoint the raw branch this many amps at a time (0: off)")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", type=Path, default=None,
                    help="last.pt of a stopped run: continue its exact stream to --steps")
    args = ap.parse_args()
    if args.score is not None:
        score_report(args.score, args.device)
        return
    if not 0 <= args.synth_per_batch < args.batch:
        raise SystemExit("--synth-per-batch must be in [0, --batch)")
    if args.passes < 1:
        raise SystemExit("--passes must be >= 1")

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

    init = torch.load(args.init, map_location="cpu", weights_only=False)
    ia = init["args"]
    diff = {k: (ia[k], arg_dict[k]) for k in C1_SHARED if ia[k] != arg_dict[k]}
    if diff:
        raise SystemExit(f"--init was trained with different settings: {diff}")
    val_frac = float(ia["val_frac"])
    arg_dict["val_frac"] = val_frac
    arg_dict.update({k: ia[k] for k in C1_MODEL_ARGS})

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
        raise SystemExit(f"no .nam under {args.captures} for {len(missing)} amp(s), "
                         f"e.g. {missing[:5]}")

    # c1's split, exactly: permutation(len(ids)) from a fresh default_rng(seed).
    perm = np.random.default_rng(args.seed).permutation(len(ids))
    n_val = max(1, int(round(val_frac * len(ids))))
    val_sel, train_sel = perm[:n_val], perm[n_val:]
    val_ids = ids[val_sel]
    n_synth = args.synth_per_batch
    n_real = args.batch - n_synth
    T = int(args.passes)

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

    def load_nam(d: int):
        m = nam_io.load_model(nam_paths[int(d)], device=str(dev))
        if m.sample_rate != cfg.sample_rate:
            raise SystemExit(f"amp {d}: .nam at {m.sample_rate} Hz; crop rendering "
                             f"assumes {cfg.sample_rate}")
        return m

    t0 = time.time()
    nams = {int(d): load_nam(d) for d in ids}
    R_nam = max(m.receptive_field for m in nams.values())
    x_pad_nam = np.concatenate([np.zeros(R_nam, np.float32), x])
    print(f"[fb] loaded {len(nams)} .nam captures ({time.time() - t0:.0f} s); "
          f"largest receptive field {R_nam}", flush=True)

    def real_full(m, d: int) -> torch.Tensor:
        """``[n_keep]`` the amp's ``.nam`` on the whole reference, at its learned level."""
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
    def play_full(u: torch.Tensor) -> torch.Tensor:
        """``[S, De]`` unit vectors -> ``[S, n_keep]`` (CPU): the generator on the
        whole reference."""
        out = []
        for s in range(0, len(u), args.render_chunk):
            e = scale * u[s:s + args.render_chunk]
            out.append(gen_play(x_pad_gen[None].expand(len(e), -1), e)[:, skip:].cpu())
        return torch.cat(out)

    # --- The two encoders -----------------------------------------------------------
    sizes = {k: ia[k] for k in C1_MODEL_ARGS}
    c1 = ConformerEncoder(embedding_dim, sr, **sizes).to(dev)
    c1.load_state_dict(init["state_dict"])
    c1.requires_grad_(False)
    c1.eval()
    net = FeedbackEncoder(embedding_dim, sr, **sizes, raw_chunk=args.raw_chunk).to(dev)
    net.load_c1(init["state_dict"])
    del init

    @torch.no_grad()
    def c1_guess(dry: torch.Tensor, wet: torch.Tensor, chunk: int) -> torch.Tensor:
        """``[N, n]`` wet over one ``dry`` ``[n]`` -> ``[N, De]`` unit vectors."""
        out = []
        for s in range(0, len(wet), chunk):
            w = wet[s:s + chunk].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(c1(dry[None].expand(len(w), -1), w).float())
        return torch.cat(out)

    # Val amps on the whole reference, and c1's guess for each (pass 0).
    wet_val = torch.stack([real_full(nams[int(d)], d) for d in val_ids])   # [V, n_keep]
    p_val = (wet_val.double() ** 2).mean(1).float()
    u0_val = c1_guess(dry_full, wet_val, 1)
    print(f"[fb] rendered {len(val_ids)} val amps on the reference, c1 guessed each",
          flush=True)

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
    print(f"[fb] val: {len(val_ids)} amps x {args.val_clips} EGDB val clips", flush=True)

    cloud = TableCloud(table[train_sel], scale)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best_val, best_step, best_state = 0, float("inf"), -1, None
    if ck is not None:
        net.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start, best_val, best_step = ck["step"] + 1, ck["best_val"], ck["best_step"]
        best_state = ck["best_state"]
        print(f"[fb] resumed {args.resume} at step {start} (best val {best_val:.4f} "
              f"at step {best_step + 1})")
    n_params = count_parameters(net)
    print(f"[fb] {len(ids)} amps: {len(train_sel)} train / {len(val_sel)} val  "
          f"decoder={args.run.name}  radius {scale:.4f}  loss={spec.loss.name} "
          f"(amp weighting {strength:g})")
    print(f"[fb] per step: {n_real} real + {n_synth} made-up amps, reference crop "
          f"{args.ref_seconds:g} s, loss clip {args.clip_seconds:g} s, {T} passes")
    print(f"[fb] first guess: {args.init} (frozen); correction network starts from its "
          f"weights, {n_params:,} params", flush=True)

    @torch.no_grad()
    def clip_loss(u: torch.Tensor) -> dict:
        """c1's validation: the generator's loss on the fixed val clips at ``u``."""
        E = scale * u
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

    def validate() -> list[dict]:
        """The val loss after pass 0 (c1 alone), 1, ..., T."""
        return [clip_loss(u) for u in refine(net, play_full, dry_full, wet_val, u0_val, T)]

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    def run_state(step: int) -> dict:
        return {"state_dict": net.state_dict(), "opt": opt.state_dict(), "step": step,
                "best_val": best_val, "best_step": best_step, "best_state": best_state,
                "args": arg_dict, "device_ids": ids.tolist()}

    def arrows(v) -> str:
        return ">".join(f"{q:.3f}" for q in v)

    # --- Train ----------------------------------------------------------------------
    clips = StepClips(train_ds, train_sel, ids, n_real, n_synth, args.seed)
    loader = torch.utils.data.DataLoader(
        clips, batch_size=None, sampler=range(start, args.steps),
        num_workers=args.num_workers, pin_memory=dev.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None)

    def fresh_acc() -> dict:
        return {"loss": 0.0, "real": np.zeros(T + 1), "synth": np.zeros(T + 1), "n": 0}

    acc = fresh_acc()
    t0, t_wait, t_render = time.time(), 0.0, 0.0
    tw = time.time()
    for item in loader:
        step = int(item["step"])
        t_wait += time.time() - tw
        net.train()
        g = skip + int(step_rng(args.seed, 3, step).integers(0, n_keep - L_ref + 1))
        dry = torch.from_numpy(x[g:g + L_ref]).to(dev)
        seg = x_pad_gen[g:g + R + L_ref]
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
            wet_ref.append(gen_play(seg[None].expand(n_synth, -1), E))
            tgt.append(gen_play(xin[n_real:], E))
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_render += time.time() - tr0
        wet_ref, tgt = torch.cat(wet_ref), torch.cat(tgt)
        w = amp_weights((wet_ref.double() ** 2).mean(1).float(), strength)
        dry_b = dry[None].expand(args.batch, -1)

        def note(t: int, yf: torch.Tensor) -> None:
            """Pass ``t``'s ESR on the step's clips, real and made-up apart."""
            if n_real:
                acc["real"][t] += float(preemph_esr(yf[:n_real], tgt[:n_real, None],
                                                    lossfn.coeff, weights=w[:n_real]))
            if n_synth:
                acc["synth"][t] += float(preemph_esr(yf[n_real:], tgt[n_real:, None],
                                                     lossfn.coeff, weights=w[n_real:]))

        u = c1_guess(dry, wet_ref, max(args.raw_chunk, 1))            # pass 0, no grad
        note(0, gen_play(xin, scale * u)[:, None])
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for t in range(1, T + 1):
            tr0 = time.time()
            guess = torch.cat([gen_play(seg[None].expand(len(c), -1), scale * c)
                               for c in u.split(TRAIN_RENDER)])
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_render += time.time() - tr0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                u_new = step_vector(u, net(dry_b, wet_ref, wet_ref - guess, u))
                y = model.forward_emb(xin[:, None], scale * u_new)[:, :, R:]
            loss, _ = lossfn(y.float(), tgt[:, None], w)
            (loss / T).backward()                  # this pass's graph is freed here
            step_loss += float(loss.detach()) / T
            with torch.no_grad():
                note(t, y.float())
            u = u_new.detach()                     # the next pass starts from a constant
        torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        opt.step()
        acc["loss"] += step_loss
        acc["n"] += 1

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            va_m = validate()
            star = va_m[-1]["loss"] < best_val - 1e-6
            if star:
                best_val, best_step = va_m[-1]["loss"], step
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            n = max(acc["n"], 1)
            elapsed = time.time() - t0
            print(f"  s{step + 1:05d}  train loss {acc['loss'] / n:.4f}  esr by pass: real "
                  f"{arrows(acc['real'] / n)} made-up {arrows(acc['synth'] / n)}  val loss "
                  f"{arrows(m['loss'] for m in va_m)} (esr {arrows(m['esr'] for m in va_m)}) "
                  f"best {best_val:.4f}  lr {lr_at(step):.2e}  {elapsed / 60:.1f} min, "
                  f"data wait {t_wait / max(elapsed, 1e-9):.0%} render "
                  f"{t_render / max(elapsed, 1e-9):.0%}" + ("  *" if star else ""),
                  flush=True)
            acc = fresh_acc()
            save_atomic(run_state(step), args.out / "last.pt")
        tw = time.time()
    train_min = (time.time() - t0) / 60
    print(f"[fb] trained to step {args.steps} in {train_min:.1f} min; best val "
          f"{best_val:.4f} at step {best_step + 1}")

    # --- The generator's holdout amps: vectors after every pass, best checkpoint -----
    from openamp.training.unfold import split_holdout

    net.load_state_dict(best_state)
    final_val = validate()
    del nams
    wet_h = []
    t0 = time.time()
    for d in holdout:
        wet_h.append(real_full(load_nam(d), d))
    wet_h = torch.stack(wet_h)                                              # [H, n_keep]
    u0_h = c1_guess(dry_full, wet_h, 1)
    us_h = refine(net, play_full, dry_full, wet_h, u0_h, T)
    _, rep_ids = split_holdout(holdout)
    rep_rows = torch.as_tensor([holdout.index(d) for d in rep_ids])
    u_swap = refine(net, play_full, dry_full, wet_h[rep_rows], u0_h[rep_rows.to(dev)], T,
                    swap=True)[-1]
    print(f"[fb] {len(holdout)} holdout amps through {T} passes ({time.time() - t0:.0f} s)",
          flush=True)

    report = {
        "init": str(args.init), "n_params": int(n_params), "n_train": int(len(train_sel)),
        "n_val": int(len(val_sel)), "passes": T, "steps": int(args.steps),
        "real_per_batch": int(n_real), "synth_per_batch": int(n_synth),
        "ref_seconds": args.ref_seconds, "clip_seconds": args.clip_seconds,
        "best_val_loss": best_val, "best_step": int(best_step + 1),
        "val_loss_by_pass": [m["loss"] for m in final_val],
        "val_esr_by_pass": [m["esr"] for m in final_val],
        "embedding_dim": int(embedding_dim), "target_radius": scale,
        "train_minutes": train_min,
    }
    torch.save({"state_dict": net.state_dict(), "args": arg_dict,
                "device_ids": ids.tolist(), "val_ids": val_ids.tolist(),
                "holdout_ids": holdout, "holdout_u": [u.cpu() for u in us_h],
                "report_ids": rep_ids, "report_u_swap": u_swap.cpu(),
                "report": report},
               args.out / "probe_feedback.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[fb] wrote {args.out}/probe_feedback.pt and report.json")
    score_report(args.out, args.device)


if __name__ == "__main__":
    main()
