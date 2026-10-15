"""Can one capture do both jobs? Unfold on the NAM capture file instead of corpus clips.

    OPENAMP_DATA_DIR=data python scripts/unfold_capture_test.py

The encoder already reads each amp's render of the T3K capture signal (its DAC
frames). Deployed, that same recording is a (dry, wet) pair, so the unfolded
steps could fit on it too -- one capture for everything. This measures it.

Fixed: the trained unfolders (prototype runs ``results/unfold/unfold_k{5,10}_v2_b32``),
the encoder guess, the 42 report amps, and the scoring set (their EGDB *test*
files, 200 clips each -- exactly the runs' report). Only the source of the 16
fit clips changes:

- ``egdb``      -- the amp's EGDB train clips, as in training (reproduces the report)
- ``capture``   -- random windows anywhere in ``sweep_train`` (the capture's NAM
                   train region, 171 s: ~7 s of blips/chirps/noise, then guitar DI)
- ``capture_di``-- windows inside its guitar-DI body only

The capture renders come from ``data_sweep_a1`` (the A1 captures the generator
learned -- checked here by ``nam_sha256`` against ``data/``) and are rescaled to
``data/``'s per-amp ``output_scale``, the level convention the generator trained
on. Window starts leave ``R`` samples of real left-context, as every dataset does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from openamp.core.config import load_config
from openamp.data import AmpClipsDataset, CorpusPaths, EmulationDataset, make_loader
from openamp.dsp import audio as audio_io
from openamp.enroll.fit import TEST_PAIRS_PER_DEVICE
from openamp.enroll.unfold import Unfolder
from openamp.evaluation import load_model
from openamp.losses import build_loss
from openamp.spec import spec_from_checkpoint
from openamp.training.unfold import score

# sweep_train = T3K-sweep-v3.wav[480000:8688000]; the guitar-DI body starts at
# 828000 in the source file (t3k-v3 layout), i.e. 348000 into this one.
DI_START = 828_000 - 480_000
CONDITIONS = ("egdb", "capture", "capture_di")


def capture_clips(clean: np.ndarray, wet: np.ndarray, lo: int, n: int, R: int, clip: int,
                  rng: np.random.Generator):
    """``n`` random (input-with-context, target) windows whose loss region starts >= lo."""
    starts = rng.integers(max(lo, R), len(clean) - clip + 1, size=n)
    x = np.stack([clean[c - R:c + clip] for c in starts])
    t = np.stack([wet[c:c + clip] for c in starts])
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(t.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path,
                    default=[Path("results/unfold/unfold_k5_v2_b32"),
                             Path("results/unfold/unfold_k10_v2_b32")])
    ap.add_argument("--generator", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2"))
    ap.add_argument("--sweep-dir", type=Path, default=Path("data_sweep_a1"))
    ap.add_argument("--out", type=Path, default=Path("results/unfold/capture_test"))
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config()
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    gen, gck = load_model(args.generator, "cuda")
    gen.requires_grad_(False)
    spec = spec_from_checkpoint(gck)
    R, clip = gen.receptive_field, spec.clip_samples
    coeff = float(build_loss(spec.loss).coeff)
    holdout = sorted(int(d) for d in gck["holdout_ids"])
    rep_ids = holdout[1::2]                                 # the runs' report amps
    paths = CorpusPaths.from_config(cfg)

    # --- capture renders: the right amps, at the generator's level convention ------
    r_data = pd.read_parquet(paths.renders_manifest)
    r_cap = pd.read_parquet(args.sweep_dir / "manifests" / "renders.parquet")
    r_cap = r_cap[r_cap["file_id"] == "sweep_train"].set_index("device_id")
    ref = r_data.groupby("device_id").agg(sha=("nam_sha256", "first"),
                                          scale=("output_scale", "first"))
    bad = [d for d in rep_ids if r_cap.loc[d, "nam_sha256"] != ref.loc[d, "sha"]]
    if bad:
        raise SystemExit(f"capture renders are not the generator's amps for {bad}")
    clean, sr = audio_io.read_audio(args.sweep_dir / "clean" / "train" / "sweep_train.flac")

    # --- fit clips per condition --------------------------------------------------
    base = EmulationDataset(paths, "train", receptive_field=R,
                            id_to_idx={d: i for i, d in enumerate(holdout)},
                            clip_samples=clip, pairs_per_epoch=1, seed=args.seed)
    egdb = AmpClipsDataset(base, rep_ids, args.clips, seed=args.seed + 2)   # = the runs'
    loader = make_loader(egdb, 1, args.num_workers, shuffle=False, seed=0, drop_last=False)
    fits = {c: ([], []) for c in CONDITIONS}
    for j, (d, b) in enumerate(zip(rep_ids, loader)):
        fits["egdb"][0].append(b["xf"][0]); fits["egdb"][1].append(b["tf"][0])
        wet, _ = audio_io.read_audio(args.sweep_dir / "renders" / f"{d:04d}" /
                                     "sweep_train.flac")
        wet = wet * (ref.loc[d, "scale"] / r_cap.loc[d, "output_scale"])
        for cond, lo in (("capture", 0), ("capture_di", DI_START)):
            x, t = capture_clips(clean, wet, lo, args.clips, R, clip,
                                 np.random.default_rng((args.seed, j, len(cond))))
            fits[cond][0].append(x); fits[cond][1].append(t)
    fits = {c: (torch.stack(x), torch.stack(t)) for c, (x, t) in fits.items()}
    print(f"[capture] {len(rep_ids)} report amps x {args.clips} fit clips; capture "
          f"{len(clean) / sr:.0f} s, DI body from {DI_START / sr:.2f} s", flush=True)

    enc = torch.load(args.runs[0] / "encoder_vectors.pt", weights_only=False)
    e0 = gen.project_embedding(torch.from_numpy(
        np.stack([enc[d] for d in rep_ids])).to(dev)).detach()

    rows = []
    for run in args.runs:
        st = torch.load(run / "checkpoint.pt" if (run / "checkpoint.pt").is_file()
                        else run / "best.pt", map_location=dev, weights_only=False)
        state = st.get("unfolder", st.get("model"))
        unf = Unfolder(gen, int(state["V"].shape[0])).to(dev)
        unf.load_state_dict(state)
        for cond in CONDITIONS:
            x, t = fits[cond]
            out = []
            with torch.no_grad():
                for i in range(0, len(rep_ids), 6):
                    e, _, _ = unf(gen, e0[i:i + 6], x[i:i + 6].to(dev), t[i:i + 6].to(dev),
                                  coeff, amp_dtype=torch.bfloat16)
                    out.append(e)
            esr = score(gen, torch.cat(out), rep_ids, paths, "test", TEST_PAIRS_PER_DEVICE,
                        seed=cfg.seed + 7, clip=clip, batch_size=args.clips,
                        num_workers=args.num_workers, dev=dev)
            rows += [{"run": run.name, "steps": unf.steps, "fit_clips": cond,
                      "device_id": d, "test_esr": float(v)} for d, v in zip(rep_ids, esr)]
            print(f"  K={unf.steps:<3d} fit on {cond:11s} report median "
                  f"{np.nanmedian(esr):.4f} mean {np.nanmean(esr):.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_device.csv", index=False)
    summary = (df.groupby(["steps", "fit_clips"])["test_esr"]
               .agg(["median", "mean"]).reset_index())
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()},
                                                   indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
