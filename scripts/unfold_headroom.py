"""How far do a few plain enrollment steps get, from the encoder's guess?

    OPENAMP_DATA_DIR=data python scripts/unfold_headroom.py

The question before building a deep-unfolding encoder (encoder guess, then K
learned gradient steps through the frozen generator): does a *plain*, unlearned
version of those steps already close the gap? Learned steps should only do
better, so this is a floor on what unfolding can reach at a given K.

Per held-out amp, independently, from two starting points:

- ``encoder`` -- the frozen-decoder encoder's zero-shot vector for that amp.
- ``uniform`` -- enrollment's own default start (``_swap_embedding``'s U(-a, a)).

One step = one Adam update of that amp's single vector, on a batch of
``--batch-size`` fresh training clips of that amp. Enrollment's settings
otherwise: Adam, constant ``--lr``, the run's grad clip, the run's
device-loss weighting over the enrolled cohort -- but **ESR only** (stft_weight
0), the loss an unfolded encoder would step on. Both starts see the identical
sequence of batches, so the start is the only difference between them.

The vectors are kept at each step in ``--checkpoints`` and scored afterwards
with :func:`openamp.probes.probe.holdout_esr` -- the test-split, silence-gated
per-window ESR that ``enroll`` reports, so step ``max`` is comparable to the
run's own enrollment (mean 0.0231 / median 0.0089 for
delta_a2_tanh_r32_512_v2, after ~46 epochs).
"""
from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch

from openamp.core.config import load_config
from openamp.data import CorpusPaths, EmulationDataset, make_loader
from openamp.encoders import build_encoder
from openamp.enroll.fit import (ENROLL_INIT_A, TEST_PAIRS_PER_DEVICE, _enroll_spec,
                                _resolve_enroll_ids, _swap_embedding)
from openamp.evaluation import load_model
from openamp.losses import build_loss, device_loss_weights
from openamp.probes.probe import holdout_esr
from openamp.spec import EncoderSpec, spec_from_checkpoint
from openamp.training import GRAD_CLIP_NORM

INITS = ("encoder", "uniform")


def encoder_vectors(enc_run: Path, frames: Path, ids: list[int], dev) -> np.ndarray:
    """``[len(ids), De]`` zero-shot vectors from a joint run's encoder, eval mode.

    Raw encoder output (unit length). The generator's own ``project_embedding``
    puts it on the trained sphere when it is installed as a table row -- the same
    vector ``ScaledJointModel.embed`` hands the frozen decoder.
    """
    ck = torch.load(enc_run / "checkpoint.pt", map_location="cpu", weights_only=False)
    spec = spec_from_checkpoint(ck)
    es = spec.joint.encoder
    # The checkpoint names a host-local copy of the store; read the shared one.
    es = EncoderSpec(es.kind, {**es.kwargs, "frames_path": str(frames)})
    gen_dim = int(ck["generator"]["embedding.weight"].shape[1])
    enc = build_encoder(es, gen_dim)
    enc.load_state_dict(ck["encoder"])
    enc.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), 8):
            out.append(enc(torch.tensor(ids[i:i + 8], device=dev)).float().cpu())
    print(f"[encoder] {enc_run.name} epoch {int(ck['epoch'])} (val_esr "
          f"{float(ck['val_esr']):.5f}), frozen decoder {ck.get('frozen_decoder')}")
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2"),
                    help="Emulate run supplying the frozen generator.")
    ap.add_argument("--encoder-run", type=Path,
                    default=Path("results/joint/frozen_delta_a2_tanh_r32_512_v2_b32"),
                    help="Frozen-decoder joint run supplying the zero-shot guesses.")
    ap.add_argument("--frames", type=Path, default=Path("results/dac_frames/sweep_train"))
    ap.add_argument("--out", type=Path, default=Path("results/encoder/unfold_headroom"))
    ap.add_argument("--checkpoints", default="0,1,5,10,25,50",
                    help="Steps at which the vectors are scored; the last is the budget.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-pairs", type=int, default=TEST_PAIRS_PER_DEVICE)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoints = sorted({int(s) for s in args.checkpoints.split(",")})
    n_steps = checkpoints[-1]
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model, ck = load_model(args.run, str(dev))
    enc_ck = torch.load(args.encoder_run / "checkpoint.pt", map_location="cpu",
                        weights_only=False)
    if Path(str(enc_ck.get("frozen_decoder", ""))).resolve() != args.run.resolve():
        raise SystemExit(f"{args.encoder_run} was trained against "
                         f"{enc_ck.get('frozen_decoder')!r}, not {args.run}")
    del enc_ck
    spec = _enroll_spec(ck, 0.0, args.batch_size)
    lossfn = build_loss(spec.loss).to(dev)
    ids, skipped = _resolve_enroll_ids(cfg, ck, None)
    paths = CorpusPaths.from_config(cfg)
    strength = float(spec.loss.device_loss_weighting)
    dev_w = (device_loss_weights(paths.renders_manifest, ids, strength)
             if strength > 0.0 else np.ones(len(ids)))

    starts = {"encoder": encoder_vectors(args.encoder_run, args.frames, ids, dev)}
    _swap_embedding(model, 1, dev)          # freeze everything; one trainable row
    a = ENROLL_INIT_A
    g = torch.Generator().manual_seed(args.seed)
    starts["uniform"] = (torch.rand(len(ids), model.embedding_dim, generator=g)
                         * 2 * a - a).numpy()

    R = model.receptive_field
    clip = int(round(spec.data.clip_seconds * cfg.sample_rate))
    train_all = EmulationDataset(paths, "train", receptive_field=R,
                                 id_to_idx={d: i for i, d in enumerate(ids)},
                                 clip_samples=clip,
                                 pairs_per_epoch=n_steps * args.batch_size,
                                 seed=args.seed)
    print(f"[headroom] {len(ids)} amps (skipped {len(skipped)}) x {len(INITS)} starts "
          f"x {n_steps} steps, batch {args.batch_size}, Adam lr {args.lr:g}, "
          f"ESR only, scored at {checkpoints}, test_pairs {args.test_pairs}")

    # snaps[init][step] -> [n_amps, De]
    snaps = {s: {k: np.empty_like(starts["encoder"]) for k in checkpoints} for s in INITS}
    emb = model.embedding.weight
    t0 = time.time()
    for i, d in enumerate(ids):
        ds = copy.copy(train_all)           # this amp only; same files, same draws
        ds.device_ids, ds.rows = [d], np.zeros(1, dtype=np.int64)
        loader = make_loader(ds, args.batch_size, spec.training.num_workers,
                             shuffle=False, seed=args.seed, drop_last=False)
        batches = [(b["input"], b["target"].unsqueeze(1)) for b in loader]
        w = torch.full((args.batch_size,), float(dev_w[i]), device=dev)
        di = torch.zeros(args.batch_size, dtype=torch.long, device=dev)
        for init in INITS:
            with torch.no_grad():
                emb.copy_(torch.from_numpy(starts[init][i]).to(dev)[None])
            opt = torch.optim.Adam([emb], lr=args.lr)
            if 0 in snaps[init]:
                snaps[init][0][i] = emb.detach()[0].cpu().numpy()
            for step, (inp, target) in enumerate(batches, start=1):
                inp, target = inp.to(dev), target.to(dev)
                opt.zero_grad(set_to_none=True)
                out = model(inp, di[:len(inp)])[..., R:]
                loss, _ = lossfn(out.float(), target, weights=w[:len(inp)])
                loss.backward()
                torch.nn.utils.clip_grad_norm_([emb], GRAD_CLIP_NORM)
                opt.step()
                if step in snaps[init]:
                    snaps[init][step][i] = emb.detach()[0].cpu().numpy()
        el = time.time() - t0
        print(f"  amp {i + 1:>3d}/{len(ids)} device {d}  "
              f"{el / 60:.1f} min, ~{el / (i + 1) * (len(ids) - i - 1) / 60:.0f} min left",
              flush=True)
    torch.save({"device_ids": ids, "checkpoints": checkpoints, "snaps": snaps,
                "args": vars(args)}, args.out / "vectors.pt")
    del model
    torch.cuda.empty_cache()

    report = args.out / "report.csv"
    per_dev = args.out / "per_device.csv"
    with report.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["init", "step", "esr_mean", "esr_median"])
    with per_dev.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["init", "step", "device_id", "test_esr"])
    print(f"[headroom] fitting done in {(time.time() - t0) / 60:.1f} min; scoring")
    for init in INITS:
        for k in checkpoints:
            r = holdout_esr(cfg, args.run, {d: snaps[init][k][j] for j, d in enumerate(ids)},
                            device=args.device, test_pairs=args.test_pairs)
            with report.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([init, k, f"{r['mean']:.6f}", f"{r['median']:.6f}"])
            with per_dev.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows([init, k, d, f"{v:.6f}"]
                                         for d, v in r["per_device"].items())
            print(f"  {init:8} step {k:>4d}  test ESR mean {r['mean']:8.4f}  "
                  f"median {r['median']:8.4f}   ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    print(f"[headroom] -> {report}")


if __name__ == "__main__":
    main()
