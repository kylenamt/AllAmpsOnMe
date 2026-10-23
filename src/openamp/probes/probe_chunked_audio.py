"""probe_chunked, phase 2: the chunked encoder trained by audio loss through the
frozen generator, starting from a phase-1 (contrastive) checkpoint.

    OPENAMP_DATA_DIR=data python -m openamp.probes.probe_chunked_audio \\
        --init results/encoder/encoder_probe_chunked_b128/probe_chunked.pt \\
        --out results/encoder/encoder_probe_chunked_audio

then ``python -m openamp.probes.probe_esr --probe <out>/probe_chunked_audio.pt``.

**Start.** The encoder (:class:`~openamp.probes.probe_chunked.ChunkedEncoder`)
is built from ``--init``'s settings and loads its ``best_state``, the key
standardization included. ``--init`` must share this run's ``--run``, ``--seed``
and ``--val-frac``, so its val amps are this run's val amps: never trained on in
either phase.

**Loss** (:mod:`openamp.probes.probe_conformer`'s recipe): the encoder hears an
amp play a crop of the reference and emits a unit vector; times the table's
radius, that is ``e``. The frozen generator plays a *different* clip (EGDB guitar
DI, ``--clip-seconds``) with ``e``, scored against the amp's real output on that
clip by the generator's own loss with its STFT weight set to ``--stft-weight``
(0 = pre-emphasized ESR alone). Per-amp ESR weight ``power ** -strength`` from
the amp's wet power on the shared reference crop, mean 1 per batch.

**Amps per step**: ``--batch`` real training amps (a seeded permutation per
pass), reference crop and clip played by their ``.nam`` at their renders'
``output_scale``; plus ``--synth-per-batch`` made-up amps from
:class:`~openamp.probes.probe_chunked.ShapedCloud` (``--synth-shape``), both
played by the generator, each drawn once.

**Reference crop**: one length ``k`` per step, 1 to ``K_max`` chunks uniform, as
in phase 1, at a random start after the skip; the same crop for every amp in the
step.

**Learning rate**: phase 1's -- linear warmup over ``--warmup-steps``, then a
cosine to ``--min-lr-frac`` of ``--lr`` at the last step.

**Validation**: the encoder hears each val amp over the *whole* reference (after
the skip); the generator plays ``--val-clips`` fixed EGDB val clips per amp.
``best_val`` is the loss there (the ESR and STFT parts are both reported).

**Output** is what :mod:`openamp.probes.probe_esr` reads: ``pred`` (every table
amp, the encoder over its whole reference, times the radius) and ``pred_sim``
(the val amps, hearing the generator's render of their own row).

**Resuming.** Every draw is seeded by ``(seed, stream, step)``, so
``--resume <out>/last.pt`` continues the identical stream.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from openamp.core.config import load_config
from openamp.corpus.render import _render_signal
from openamp.data import CorpusPaths, EmulationDataset
from openamp.data.emulation import read_with_retry
from openamp.dsp import audio as audio_io
from openamp.dsp import nam as nam_io
from openamp.evaluation import load_model, resolve_device
from openamp.losses import build_loss, preemph_esr
from openamp.models import count_parameters
from openamp.probes.probe_chunked import (ChunkedEncoder, ShapedCloud, chunk_geometry,
                                          chunk_starts, crop_samples, lr_factor)
from openamp.probes.probe_conformer import StepClips, amp_weights, read_clean
from openamp.probes.probe_synth import SKIP_SECONDS, captures_by_sha, save_atomic, step_rng
from openamp.spec import spec_from_checkpoint
from openamp.training import GRAD_CLIP_NORM

MODEL_ARGS = ("embedding_dim", "chunk_n", "width", "n_blocks", "n_heads", "ff_ratio",
              "conv_kernel", "pool_heads", "key_hidden", "max_seconds")
RESUME_FIXED = ("init", "run", "sweep", "captures", "renders", "seed", "val_frac", "batch",
                "synth_per_batch", "synth_shape", "steps", "clip_seconds", "val_clips",
                "stft_weight", "lr", "warmup_steps", "min_lr_frac", "weight_decay")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=Path, required=True,
                    help="phase-1 probe_chunked.pt: its settings and best_state")
    ap.add_argument("--run", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2_n10"),
                    help="emulate run: the frozen generator, its loss, and the table "
                         "(amp list, split, radius, made-up draws' shape)")
    ap.add_argument("--sweep", type=Path,
                    default=Path("data_sweep_a1/clean/train/sweep_train.flac"),
                    help="the reference: the T3K capture signal's train region")
    ap.add_argument("--captures", type=Path, default=Path("data/captures"))
    ap.add_argument("--renders", type=Path, default=Path("data/manifests/renders.parquet"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/encoder/encoder_probe_chunked_audio"))
    ap.add_argument("--batch", type=int, default=64, help="real amps per step")
    ap.add_argument("--synth-per-batch", type=int, default=64,
                    help="made-up amps per step, on top of --batch")
    ap.add_argument("--synth-shape", type=float, default=0.5,
                    help="made-up draws: covariance eigenvalues ** this (1 = the "
                         "table rows' shape, 0 = equal in every direction)")
    ap.add_argument("--gen-chunk", type=int, default=4,
                    help="made-up amps the generator plays at once, no grad")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=21)
    ap.add_argument("--clip-seconds", type=float, default=2.0,
                    help="EGDB clip the loss is scored on (the decoder trained on 2.0)")
    ap.add_argument("--val-clips", type=int, default=8,
                    help="fixed EGDB val-split clips per val amp")
    ap.add_argument("--stft-weight", type=float, default=0.0,
                    help="the generator loss's STFT weight (0 = pre-emphasized ESR only)")
    ap.add_argument("--lr", type=float, default=3e-4, help="peak learning rate")
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--min-lr-frac", type=float, default=0.01,
                    help="the cosine ends at this fraction of --lr")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--ckpt-group", type=int, default=112,
                    help="chunks per checkpointed group (memory only; same result)")
    ap.add_argument("--num-workers", type=int, default=6)
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
    init = torch.load(args.init, map_location="cpu", weights_only=False)
    ia = init["args"]
    diff = {k: (ia[k], arg_dict[k]) for k in ("run", "seed", "val_frac") if ia[k] != arg_dict[k]}
    if diff:
        raise SystemExit(f"--init was trained with a different split: {diff}")
    arg_dict.update({k: ia[k] for k in MODEL_ARGS})

    # --- Frozen generator, its loss, its table --------------------------------------
    model, gen_ck = load_model(args.run, str(dev))
    model.requires_grad_(False)
    model.eval()
    R = model.receptive_field
    spec = spec_from_checkpoint(gen_ck)
    lossfn = build_loss(spec.loss.with_kwargs(stft_weight=args.stft_weight)).to(dev)
    strength = float(spec.loss.device_loss_weighting)
    tbl = torch.load(args.run / "embedding.pt", map_location="cpu", weights_only=False)
    table = tbl["embedding"].float().numpy()
    ids = np.array([int(d) for d in tbl["device_ids"]])
    scale = float(np.linalg.norm(table, axis=1).mean())
    if table.shape[1] != ia["embedding_dim"]:
        raise SystemExit(f"--init embeds to {ia['embedding_dim']}, the table is "
                         f"{table.shape[1]} wide")

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
    S = args.synth_per_batch
    B = args.batch + S

    # --- Reference and crop lengths -------------------------------------------------
    x, sr = audio_io.read_audio(args.sweep)
    x = np.asarray(x, dtype=np.float32)
    if sr != cfg.sample_rate:
        raise SystemExit(f"{args.sweep} is {sr} Hz, the corpus is {cfg.sample_rate} Hz")
    skip = int(round(SKIP_SECONDS * sr))
    n_keep = len(x) - skip
    L_max = int(round(ia["max_seconds"] * sr))
    L_clip = int(round(args.clip_seconds * sr))
    chunk, cstep = chunk_geometry(ia["chunk_n"])
    k_max = len(chunk_starts(L_max, chunk, cstep))
    if L_max > n_keep:
        raise SystemExit(f"a {ia['max_seconds']:g} s crop does not fit in the reference's "
                         f"{n_keep / sr:.1f} s")
    dry_full = torch.from_numpy(x[skip:]).to(dev)
    x_pad_gen = torch.from_numpy(np.concatenate([np.zeros(R, np.float32), x])).to(dev)

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
    print(f"[chunk-audio] loaded {len(nams)} .nam captures ({time.time() - t0:.0f} s); "
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
        out = []
        for s in range(0, len(E), args.gen_chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(model.forward_emb(xin[s:s + args.gen_chunk, None],
                                             E[s:s + args.gen_chunk])[:, 0, R:].float())
        return torch.cat(out)

    # --- Encoder --------------------------------------------------------------------
    encoder = ChunkedEncoder(
        ia["embedding_dim"], chunk_n=ia["chunk_n"], width=ia["width"],
        n_blocks=ia["n_blocks"], n_heads=ia["n_heads"], ff_ratio=ia["ff_ratio"],
        conv_kernel=ia["conv_kernel"], pool_heads=ia["pool_heads"],
        key_hidden=ia["key_hidden"]).to(dev)
    encoder.load_state_dict(init["best_state"])
    del init

    wet_val = torch.stack([real_full(d) for d in val_ids])                  # [V, n_keep]
    p_val = (wet_val.double() ** 2).mean(1).float()
    print(f"[chunk-audio] rendered {len(val_ids)} val amps on the reference", flush=True)

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
    print(f"[chunk-audio] val: {len(val_ids)} amps x {args.val_clips} EGDB val clips",
          flush=True)

    cloud = ShapedCloud(table[train_sel], scale, args.synth_shape)
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best_val, best_step, best_state, best_metrics = 0, float("inf"), -1, None, {}
    if ck is not None:
        encoder.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start, best_val, best_step = ck["step"] + 1, ck["best_val"], ck["best_step"]
        best_state, best_metrics = ck["best_state"], ck["best_metrics"]
        print(f"[chunk-audio] resumed {args.resume} at step {start} (best val "
              f"{best_val:.4f} at step {best_step + 1})")
    encoder.ckpt_group = args.ckpt_group
    n_params = count_parameters(encoder)
    print(f"[chunk-audio] {len(ids)} amps: {len(train_sel)} train / {len(val_sel)} val  "
          f"decoder={args.run.name}  radius {scale:.4f}  loss={spec.loss.name} with STFT "
          f"weight {args.stft_weight:g} (amp weighting {strength:g})")
    print(f"[chunk-audio] init {args.init}; per step: {args.batch} real + {S} made-up amps "
          f"(shape {args.synth_shape:g}), reference crop k = 1..{k_max} chunks (longest "
          f"{ia['max_seconds']:g} s), loss clip {args.clip_seconds:g} s; val hears the "
          f"whole {n_keep / sr:.1f} s; encoder {n_params:,} params", flush=True)

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
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                y = model.forward_emb(vx[s:s + args.batch].to(dev)[:, None],
                                      E[a])[:, :, R:].float()
            w = w_amp[a] / w_amp[a].mean()
            loss, parts = lossfn(y, vt[s:s + args.batch].to(dev)[:, None], w)
            tot += float(loss)
            esr_t += parts["esr"]
            stft_t += parts["stft"]
            nb += 1
        return {"loss": tot / nb, "esr": esr_t / nb, "stft": stft_t / nb}

    def checkpoint_state(step: int) -> dict:
        return {"state_dict": encoder.state_dict(), "opt": opt.state_dict(), "step": step,
                "best_val": best_val, "best_step": best_step, "best_state": best_state,
                "best_metrics": best_metrics, "args": arg_dict, "device_ids": ids.tolist()}

    # --- Train ----------------------------------------------------------------------
    clips = StepClips(train_ds, train_sel, ids, args.batch, S, args.seed)
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
        lr = args.lr * lr_factor(step, args.steps, args.warmup_steps, args.min_lr_frac)
        for group in opt.param_groups:
            group["lr"] = lr
        rng = step_rng(args.seed, 3, step)
        k = int(rng.integers(1, k_max + 1))
        L = crop_len(k)
        g = int(rng.integers(0, n_keep - L + 1))
        xc = item["x"]
        xin = xc[:, R_clip - R:].to(dev, non_blocking=True)
        tr0 = time.time()
        rid = [int(d) for d in item["real_ids"]]
        wet_ref = [torch.from_numpy(np.stack([real_crop(d, skip + g, L) for d in rid])).to(dev)]
        tgt = [torch.from_numpy(np.stack(
            [real_clip(d, xc[i].numpy()) for i, d in enumerate(rid)])).to(dev)]
        if S:
            E = torch.from_numpy(cloud.draw(step_rng(args.seed, 2, step), S)).to(dev)
            seg = x_pad_gen[skip + g:skip + g + R + L]
            wet_ref.append(gen_play(seg[None].expand(S, -1), E))
            tgt.append(gen_play(xin[args.batch:], E))
        wet_ref, tgt = torch.cat(wet_ref), torch.cat(tgt)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_render += time.time() - tr0
        w = amp_weights((wet_ref.double() ** 2).mean(1).float(), strength)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            e = scale * encoder(dry_full[g:g + L][None].expand(B, -1), wet_ref)
            y = model.forward_emb(xin[:, None], e)[:, :, R:]
        del wet_ref
        loss, _ = lossfn(y.float(), tgt[:, None], w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP_NORM)
        opt.step()

        with torch.no_grad():
            yf = y.float()
            acc["loss"] += float(loss)
            acc["real"] += float(preemph_esr(yf[:args.batch], tgt[:args.batch, None],
                                             lossfn.coeff, weights=w[:args.batch]))
            if S:
                acc["synth"] += float(preemph_esr(yf[args.batch:], tgt[args.batch:, None],
                                                  lossfn.coeff, weights=w[args.batch:]))
            acc["n"] += 1
        del e, y, yf, tgt, loss

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            v = validate()
            star = v["loss"] < best_val - 1e-6
            if star:
                best_val, best_step, best_metrics = v["loss"], step, v
                best_state = {kk: vv.detach().clone() for kk, vv in
                              encoder.state_dict().items()}
            n = max(acc["n"], 1)
            elapsed = time.time() - t0
            print(f"  s{step + 1:05d}  train loss {acc['loss'] / n:.4f}  esr real "
                  f"{acc['real'] / n:.4f} made-up {acc['synth'] / n:.4f}  val loss "
                  f"{v['loss']:.4f} (esr {v['esr']:.4f} stft {v['stft']:.4f})  best "
                  f"{best_val:.4f}  lr {lr:.2e}  {elapsed / 60:.1f} min, data wait "
                  f"{t_wait / max(elapsed, 1e-9):.0%} render "
                  f"{t_render / max(elapsed, 1e-9):.0%}" + ("  *" if star else ""),
                  flush=True)
            acc = {kk: 0 if kk == "n" else 0.0 for kk in acc}
            save_atomic(checkpoint_state(step), args.out / "last.pt")
        tw = time.time()
    train_min = (time.time() - t0) / 60
    print(f"[chunk-audio] trained to step {args.steps} in {train_min:.1f} min; best val "
          f"{best_val:.4f} at step {best_step + 1}")

    # --- Predict every amp, and the val amps once more from the generator's render ---
    encoder.load_state_dict(best_state)
    pred_all = np.zeros((len(ids), table.shape[1]), np.float32)
    t0 = time.time()
    for i, d in enumerate(ids):
        pred_all[i] = (scale * encode_full(real_full(d)[None]))[0].cpu().numpy()
        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            print(f"[chunk-audio] predicted {i + 1}/{len(ids)} amps "
                  f"({time.time() - t0:.0f} s)", flush=True)
    Ev = torch.from_numpy(table[val_sel]).to(dev)
    sim_unit = torch.cat([encode_full(gen_play(x_pad_gen[None], Ev[i:i + 1])[:, skip:].cpu())
                          for i in range(len(Ev))])
    pred_sim = (scale * sim_unit).cpu().numpy()
    cos_real = F.cosine_similarity(torch.from_numpy(pred_all[val_sel]).to(dev), Ev).mean()
    cos_sim = F.cosine_similarity(sim_unit, Ev).mean()
    print(f"[chunk-audio] val cosine to the table row (not trained on): hearing the real "
          f"amp {float(cos_real):.3f}, hearing the generator's render of its row "
          f"{float(cos_sim):.3f}")

    report = {
        "init": str(args.init), "n_params": int(n_params), "n_train": int(len(train_sel)),
        "n_val": int(len(val_sel)), "steps": int(args.steps), "batch": int(args.batch),
        "synth_per_batch": int(S), "synth_shape": args.synth_shape,
        "max_seconds": ia["max_seconds"], "k_max": int(k_max),
        "clip_seconds": args.clip_seconds, "stft_weight": args.stft_weight,
        "best_val_loss": best_val, "best_step": int(best_step + 1),
        "val_esr": best_metrics.get("esr"), "val_stft": best_metrics.get("stft"),
        "val_cos_real": float(cos_real), "val_cos_sim": float(cos_sim),
        "embedding_dim": int(table.shape[1]), "target_radius": scale,
        "train_minutes": train_min,
    }
    torch.save({"state_dict": encoder.state_dict(), "best_state": best_state,
                "args": arg_dict, "device_ids": ids.tolist(), "val_ids": val_ids.tolist(),
                "pred": {int(d): pred_all[i] for i, d in enumerate(ids)},
                "pred_sim": {int(ids[j]): pred_sim[i] for i, j in enumerate(val_sel)},
                "report": report},
               args.out / "probe_chunked_audio.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[chunk-audio] wrote {args.out}/probe_chunked_audio.pt and report.json")


if __name__ == "__main__":
    main()
