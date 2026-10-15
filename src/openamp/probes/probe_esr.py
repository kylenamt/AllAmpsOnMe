"""Wire a trained probe encoder to the frozen decoder; ESR for seen vs unseen amps.

    OPENAMP_DATA_DIR=data python -m openamp.probes.probe_esr

:mod:`openamp.probes.probe_attn` fits an :class:`AttentionEncoder` to delta
v2's embedding table and, at the very end, renders only the enrolled amps
through the frozen generator (its ``[GATE]``). This does the other half of the
picture: it reloads a finished ``probe_attn.pt``, predicts an embedding for
*every* amp, and measures ESR through that same frozen generator for three
groups *separately*, each against its **oracle** vector:

- ``enc-val``   -- table amps held out of the *encoder's* fit. The generator
  trained on them, so their table row is an exact oracle, and the encoder has
  never seen them. This is the group that says whether a val cosine converts
  into usable audio, and it needs no enrollment to interpret.
- ``enc-train`` -- amps the encoder fitted (subsampled to ``enc-val``'s count,
  so the two means are over the same number of devices). The memorization
  reference.
- ``held-out``  -- amps neither half ever saw, against their enrolled vectors.
  Skipped when the run has no ``enroll/`` directory.

The encoder's train/val split is not stored explicitly; it is rebuilt from the
seed and ``val_frac`` in the checkpoint's saved args, which reproduces it
exactly.

"Encoder connected to the decoder": this encoder is a pure lookup -- one
deterministic embedding per amp -- so predicting every amp's vector once and
installing it as the generator's embedding table is exactly the encoder->
decoder forward, and it renders once per amp instead of once per test window.

Two ESR definitions, both reported (docs/decisions.md S6):

- ``test`` -- mean/median over devices of per-window raw ratios, silence-gated.
  Comparable to enrollment's ``test_esr_mean`` and probe_attn's ``[GATE]``.
- ``val``  -- device-weighted pooled pre-emphasized ESR. The definition delta
  v2's own ``val_esr`` (0.0509) is in.

Plus the training loss's MRSTFT term and the combined ``esr_mrstft`` value, so
the full ``loss:`` objective delta v2 was trained on is covered. One forward
pass per (group x source) produces all of it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from openamp.core.config import load_config
from openamp.data import CorpusPaths, EmulationDataset, make_loader
from openamp.encoders import AttentionEncoder, FrameStore
from openamp.evaluation import SILENCE_DBFS, load_model, resolve_device
from openamp.evaluation.metrics import pooled_esr_terms, silence_floor, windowed_esr
from openamp.losses import build_loss
from openamp.spec import spec_from_checkpoint

DEFAULT_PROBE = Path("results/encoder/encoder_probe_attn_nopos_allwd0p1/probe_attn.pt")
DEFAULT_RUN = Path("results/emulate/delta_a2_tanh_r32_512_v2")


def load_encoder(probe_pt: Path, store: FrameStore, embedding_dim: int, dev):
    """Rebuild the AttentionEncoder from a probe_attn.pt and its saved args."""
    d = torch.load(probe_pt, map_location="cpu", weights_only=False)
    a = d["args"]
    enc = AttentionEncoder(
        embedding_dim=embedding_dim, frames=store,
        n_queries=int(a["n_queries"]), d_model=int(a["d_model"]),
        n_heads=int(a["n_heads"]), mlp_ratio=int(a["mlp_ratio"]),
        positional=not bool(a.get("no_positional", False)), normalize=True).to(dev)
    enc.load_state_dict(d["state_dict"])
    enc.eval()
    return enc, d


@torch.no_grad()
def predict_embeddings(enc: AttentionEncoder, ids: list[int], scale: float,
                       dev, batch: int) -> dict[int, np.ndarray]:
    """``{device_id: scale * unit_vector}`` -- the same rescale probe_attn uses."""
    out: dict[int, np.ndarray] = {}
    for s in range(0, len(ids), batch):
        b = ids[s:s + batch]
        e = scale * enc(torch.as_tensor(b, dtype=torch.long, device=dev))
        for d, row in zip(b, e.float().cpu().numpy()):
            out[int(d)] = row
    return out


@torch.no_grad()
def measure(cfg, model, orig_embedding, group_ids: list[int],
            vec_by_id: dict[int, np.ndarray], *, split: str, test_pairs: int,
            lossfn, spec, dev) -> dict:
    """Install ``vec_by_id`` as the table, render the group's ``split``, one pass.

    Returns both ESR definitions (mean + median over devices) plus the MRSTFT
    term and the combined ``esr_mrstft`` loss (batch means, as training logs).
    ``model.embedding`` is restored to ``orig_embedding`` before returning.
    """
    ids = sorted(group_ids)
    idx = {d: i for i, d in enumerate(ids)}
    n = len(ids)
    emb = nn.Embedding(n, model.embedding_dim).to(dev)
    emb.weight.copy_(torch.from_numpy(np.stack([vec_by_id[d] for d in ids])).to(dev))
    model.requires_grad_(False)
    model.embedding = emb
    model.eval()

    R = model.receptive_field
    clip = int(round(spec.data.clip_seconds * cfg.sample_rate))
    ds = EmulationDataset(CorpusPaths.from_config(cfg), split, receptive_field=R,
                          id_to_idx=idx, clip_samples=clip,
                          pairs_per_epoch=test_pairs * n, seed=cfg.seed + 7)
    loader = make_loader(ds, spec.training.batch_size, spec.training.num_workers,
                         shuffle=False, seed=cfg.seed + 7, drop_last=False)

    floor = silence_floor(clip, SILENCE_DBFS)
    coeff = lossfn.coeff
    num_t = torch.zeros(n, dtype=torch.float64, device=dev)   # test: raw+gate
    den_t = torch.zeros(n, dtype=torch.float64, device=dev)
    num_v = torch.zeros(n, dtype=torch.float64, device=dev)   # val: pooled preemph
    den_v = torch.zeros(n, dtype=torch.float64, device=dev)
    esr_sum = stft_sum = comb_sum = 0.0
    nb = 0
    try:
        for batch in loader:
            inp = batch["input"].to(dev, non_blocking=True)
            target = batch["target"].to(dev, non_blocking=True)      # [B, clip]
            di = batch["device_idx"].to(dev, non_blocking=True)
            out = model(inp, di)[..., R:]                            # [B, 1, clip]
            o2 = out.squeeze(1)
            ratios, keep = windowed_esr(o2, target, floor)
            if ratios.numel():
                num_t.index_add_(0, di[keep], ratios.double())
                den_t.index_add_(0, di[keep], torch.ones_like(ratios).double())
            n_i, d_i = pooled_esr_terms(o2, target, coeff=coeff)
            num_v.index_add_(0, di, n_i.double())
            den_v.index_add_(0, di, d_i.double())
            _, parts = lossfn(out.float(), target.unsqueeze(1))
            esr_sum += parts["esr"]; stft_sum += parts["stft"]
            comb_sum += parts["esr"] + lossfn.stft_weight * parts["stft"]
            nb += 1
    finally:
        model.embedding = orig_embedding

    test_dev = (num_t / den_t).cpu().numpy()
    test_dev[~np.isfinite(test_dev)] = np.nan
    val_dev = (num_v / den_v).cpu().numpy()
    val_dev[~np.isfinite(val_dev)] = np.nan
    return {
        "n": n,
        "test_esr_mean": float(np.nanmean(test_dev)),
        "test_esr_median": float(np.nanmedian(test_dev)),
        "val_esr_mean": float(np.nanmean(val_dev)),
        "val_esr_median": float(np.nanmedian(val_dev)),
        "mrstft": stft_sum / max(nb, 1),
        "esr_mrstft_loss": comb_sum / max(nb, 1),
        "per_device_test": {int(d): float(test_dev[i]) for i, d in enumerate(ids)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=Path, default=DEFAULT_PROBE,
                    help="probe_attn.pt or probe_synth.pt from a finished run")
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN,
                    help="emulate run: frozen generator + trained/enrolled tables")
    ap.add_argument("--frames", type=Path, default=None,
                    help="override the frame store (default: the probe run's own)")
    ap.add_argument("--out", type=Path, default=Path("results/encoder/encoder_probe_esr"))
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--test-pairs", type=int, default=100,
                    help="render windows per device per group (runtime knob)")
    ap.add_argument("--train-sample", type=int, default=76,
                    help="amps drawn from the encoder's training set, so its ESR "
                         "mean is over the same count as enc-val's. 0 = all of them.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(resolve_device(args.device))
    cfg = load_config()
    args.out.mkdir(parents=True, exist_ok=True)

    # --- Frozen decoder + the two oracle tables ---------------------------------
    model, ck = load_model(args.run, str(dev))
    model.requires_grad_(False)
    spec = spec_from_checkpoint(ck)
    lossfn = build_loss(spec.loss).to(dev)
    orig_embedding = model.embedding

    tr = torch.load(args.run / "embedding.pt", map_location="cpu", weights_only=False)
    T = tr["embedding"].float().numpy()
    tr_row = {int(d): i for i, d in enumerate(tr["device_ids"])}
    # Runs without an enrollment (embed-noise variants, say) still have both
    # encoder groups; only the held-out group needs enrolled vectors.
    en_path = args.run / "enroll" / "enrolled_embeddings.pt"
    if en_path.is_file():
        en = torch.load(en_path, map_location="cpu", weights_only=False)
        E = en["embedding"].float().numpy()
        en_row = {int(d): i for i, d in enumerate(en["device_ids"])}
    else:
        E, en_row = None, {}

    # --- Encoder --------------------------------------------------------------------
    d0 = torch.load(args.probe, map_location="cpu", weights_only=False)
    a0 = d0["args"]
    if "pred" in d0:
        # probe_synth: its encoder reads rendered audio, not a frame store, so the
        # checkpoint carries the predictions -- one vector per amp, already
        # rescaled -- plus the val amps' vectors from the generator's own render.
        enc_e = {int(d): np.asarray(v, dtype=np.float32) for d, v in d0["pred"].items()}
        sim_e = {int(d): np.asarray(v, dtype=np.float32) for d, v in d0["pred_sim"].items()}
        has, positional, source = enc_e.__contains__, False, f"audio={a0['sweep']}"
    else:
        frames_path = args.frames or Path(a0["frames"])
        store = FrameStore(frames_path)
        enc, _ = load_encoder(args.probe, store, model.embedding_dim, dev)
        enc_e, sim_e = None, {}
        has, positional = store.__contains__, enc.pool.positional
        source = f"frames={frames_path}"

    # The encoder's own train/val split of the table amps, rebuilt exactly:
    # probe_attn draws it as permutation(len(ids)) from a fresh default_rng(seed)
    # before any other draw, and saves that same `ids` order in the checkpoint.
    # Reporting the two apart matters because the generator trained on both, so
    # a single "trained" number is 90% amps the encoder memorized.
    saved_ids = [int(d) for d in d0["device_ids"]]
    sperm = np.random.default_rng(int(a0["seed"])).permutation(len(saved_ids))
    vfrac = float(a0["val_frac"])
    n_val = max(1, int(round(vfrac * len(saved_ids)))) if vfrac > 0 else 0
    val_ids = [saved_ids[i] for i in sperm[:n_val] if has(saved_ids[i])]
    train_all = [saved_ids[i] for i in sperm[n_val:] if has(saved_ids[i])]
    # Size-matched to the val group: participation-ratio-style sample bias aside,
    # an ESR mean over 684 amps and one over 76 are not the same measurement.
    n_tr = len(train_all) if args.train_sample <= 0 else min(args.train_sample,
                                                            len(train_all))
    train_ids = list(np.random.default_rng(0).choice(train_all, n_tr, replace=False)) \
        if n_tr < len(train_all) else train_all
    train_ids = [int(d) for d in train_ids]
    holdout_ids = [d for d in en_row if has(d)]
    scale = float(np.linalg.norm(T, axis=1).mean())
    print(f"[esr]  probe={args.probe.parent.name}  decoder={args.run.name}  {source}")
    print(f"[esr]  encoder positional={positional}  De={model.embedding_dim}  "
          f"|e| rescale={scale:.4f}")
    print(f"[esr]  groups: {len(train_ids)} enc-train (of {len(train_all)}, "
          f"the encoder fitted these), {len(val_ids)} enc-val (held out from the "
          f"encoder, oracle row exists), {len(holdout_ids)} held-out "
          f"(seen by neither)")
    print(f"[esr]  split={args.split}  test_pairs={args.test_pairs}/device")

    if enc_e is None:
        enc_e = predict_embeddings(enc, train_ids + val_ids + holdout_ids, scale, dev,
                                   spec.training.batch_size)

    # --- {enc-train, enc-val, held-out} x {encoder e, oracle e} -------------------
    jobs = [
        ("enc-val", "encoder", val_ids, {d: enc_e[d] for d in val_ids}),
        ("enc-val", "oracle", val_ids, {d: T[tr_row[d]] for d in val_ids}),
    ]
    if sim_e:
        jobs += [("enc-val", "encoder-sim", val_ids, {d: sim_e[d] for d in val_ids})]
    jobs += [
        ("enc-train", "encoder", train_ids, {d: enc_e[d] for d in train_ids}),
        ("enc-train", "oracle", train_ids, {d: T[tr_row[d]] for d in train_ids}),
    ]
    if holdout_ids:
        jobs += [
            ("held-out", "encoder", holdout_ids, {d: enc_e[d] for d in holdout_ids}),
            ("held-out", "oracle", holdout_ids, {d: E[en_row[d]] for d in holdout_ids}),
        ]
    t0 = time.time()
    report: dict = {"probe": str(args.probe), "run": str(args.run),
                    "split": args.split, "test_pairs": args.test_pairs,
                    "positional": bool(positional), "results": {}}
    for group, source, gids, vecs in jobs:
        print(f"[esr]  rendering {group:8s} / {source:7s} ({len(gids)} amps) ...",
              flush=True)
        m = measure(cfg, model, orig_embedding, gids, vecs, split=args.split,
                    test_pairs=args.test_pairs, lossfn=lossfn, spec=spec, dev=dev)
        report["results"][f"{group}/{source}"] = m

    # --- Table -------------------------------------------------------------------
    def row(label, m):
        return (f"  {label:20s}  test {m['test_esr_mean']:.4f}/{m['test_esr_median']:.4f}"
                f"   val {m['val_esr_mean']:.4f}/{m['val_esr_median']:.4f}"
                f"   mrstft {m['mrstft']:.4f}   loss {m['esr_mrstft_loss']:.4f}")

    R = report["results"]
    groups = [g for g in ("enc-val", "enc-train", "held-out")
              if f"{g}/encoder" in R]
    print(f"\n[esr]  ESR mean/median, on the {args.split} split "
          f"(done in {(time.time() - t0) / 60:.1f} min)")
    print(f"  {'group / source':20s}  {'test (raw,gated)':17s}  {'val (pooled preemph)':20s}")
    for g in groups:
        for src in ("encoder", "encoder-sim", "oracle"):
            if f"{g}/{src}" in R:
                print(row(f"{g}/{src}", R[f"{g}/{src}"]))
    for g in groups:
        o = R[f"{g}/oracle"]
        for src in ("encoder", "encoder-sim"):
            if f"{g}/{src}" not in R:
                continue
            e = R[f"{g}/{src}"]
            gap = (e["test_esr_mean"] / o["test_esr_mean"] if o["test_esr_mean"] > 0
                   else float("nan"))
            print(f"  {g}: {src} is {gap:.1f}x the oracle's test ESR "
                  f"({e['test_esr_mean']:.4f} vs {o['test_esr_mean']:.4f})")
            e["gap_vs_oracle"] = gap

    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float),
                                          encoding="utf-8")
    print(f"\n[esr]  wrote {args.out}/report.json")


if __name__ == "__main__":
    main()
