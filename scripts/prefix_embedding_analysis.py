"""Is the nested-dropout table ordered, and does a cut vector still sound right?

    OPENAMP_DATA_DIR=data python scripts/prefix_embedding_analysis.py            # geometry only
    OPENAMP_DATA_DIR=data python scripts/prefix_embedding_analysis.py --esr      # + ESR curves

Compares ``delta_a2_tanh_r32_512_v2_prefix`` (trained with random prefix cuts,
``embed_prefix_sizes``) against its control ``delta_a2_tanh_r32_512_v2`` (same
everything, no cuts). Both tables are the effective rows ``scale * normalize``.

Geometry (CPU, all 760 table rows; enrolled rows too when ``--enrolled``):

- ``energy_k``  -- share of the rows' squared length that sits in the first k
  dims. Unordered = k/512.
- ``var_k``     -- share of the between-amp variance in the first k dims (the
  part that tells amps apart; ``energy`` also counts the shared mean).
- ``pca_k``     -- share of variance the best k directions (top-k PCs) hold.
  ``var_k / pca_k`` = 1 means the first k dims are as good as PCA.
- ``mantel_k``  -- correlation of amp-to-amp distances using only the first k
  dims vs all 512: does the prefix already get the neighbourhoods right?

ESR (GPU, ``--esr``): render the test split through each run's frozen
generator with each table row cut to its first k dims (tail zeroed, no
renormalize -- exactly the training-time cut; see ``render_esr``). For the control, also the
fair comparator: project onto the table's top-k PCs, radius restored (the
``trunc`` shape of scripts/cos_esr_curve.py). Same devices, pairs and seed for
every condition.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

KS = (8, 16, 32, 64, 128, 256, 512)
PREFIX = Path("results/emulate/delta_a2_tanh_r32_512_v2_prefix")
CONTROL = Path("results/emulate/delta_a2_tanh_r32_512_v2")


def load_rows(path: Path) -> tuple[np.ndarray, list[int]]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["embedding"].float().numpy().astype(np.float64), [int(x) for x in d["device_ids"]]


def dists(E: np.ndarray) -> np.ndarray:
    sq = (E * E).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * E @ E.T, 0.0))


def mantel(A: np.ndarray, B: np.ndarray) -> float:
    iu = np.triu_indices(len(A), 1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def geometry(E: np.ndarray) -> dict:
    """Ordering stats for one table (rows = amps)."""
    n, dim = E.shape
    energy = (E ** 2).sum(0)
    C = E - E.mean(0)
    var = (C ** 2).sum(0)
    s = np.linalg.svd(C, compute_uv=False) ** 2
    pr = float(s.sum() ** 2 / (s ** 2).sum())
    Dfull = dists(E)
    out = {"radius_mean": float(np.linalg.norm(E, axis=1).mean()),
           "participation_ratio": pr,
           "pcs_for_90pct": int(np.searchsorted(np.cumsum(s) / s.sum(), 0.90) + 1),
           "pcs_for_99pct": int(np.searchsorted(np.cumsum(s) / s.sum(), 0.99) + 1),
           "per_k": {}}
    for k in KS:
        out["per_k"][k] = {
            "energy_k": float(energy[:k].sum() / energy.sum()),
            "var_k": float(var[:k].sum() / var.sum()),
            "pca_k": float(s[:k].sum() / s.sum()),
            "prefix_radius": float(np.linalg.norm(E[:, :k], axis=1).mean()),
            "mantel_k": mantel(dists(E[:, :k]), Dfull) if k < dim else 1.0}
    # Per-dim variance, binned, for the tail question.
    bins = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512)]
    out["var_per_dim_by_bin"] = {f"{a}-{b}": float(var[a:b].mean() / var.mean())
                                 for a, b in bins}
    return out


@torch.no_grad()
def render_esr(cfg, run_dir: Path, pred: dict[int, np.ndarray], *,
               device: str, test_pairs: int) -> dict:
    """``openamp.probes.probe.holdout_esr`` with one change: the vectors go in as-is.

    holdout_esr installs vectors in the *raw* table slot, so an embed_normalize
    run re-projects them (``scale * normalize``). That round-trips a full row
    exactly, but stretches a cut row back to full length -- the opposite of
    the training-time cut. Here the projection is switched off after install,
    so the generator sees exactly the vector given.
    """
    from torch import nn
    from openamp.data import CorpusPaths, EmulationDataset, make_loader
    from openamp.evaluation import SILENCE_DBFS, load_model, per_device_esr, resolve_device
    from openamp.spec import spec_from_checkpoint

    dev = torch.device(resolve_device(device))
    model, ck = load_model(run_dir, str(dev))
    spec = spec_from_checkpoint(ck)
    ids = sorted(pred)
    idx = {d: i for i, d in enumerate(ids)}

    emb = nn.Embedding(len(ids), model.embedding_dim).to(dev)
    emb.weight.copy_(torch.from_numpy(np.stack([pred[d] for d in ids])).to(dev))
    model.requires_grad_(False)
    model.embedding = emb
    model.embed_normalize = False          # the one change: no re-projection
    model.eval()

    R = model.receptive_field
    clip = int(round(spec.data.clip_seconds * cfg.sample_rate))
    ds = EmulationDataset(CorpusPaths.from_config(cfg), "test", receptive_field=R,
                          id_to_idx=idx, clip_samples=clip,
                          pairs_per_epoch=test_pairs * len(ids), seed=cfg.seed + 7)
    loader = make_loader(ds, spec.training.batch_size, spec.training.num_workers,
                         shuffle=False, seed=cfg.seed + 7, drop_last=False)
    per_dev = per_device_esr(model, loader, dev, len(ids),
                             silence_dbfs=SILENCE_DBFS, clip=clip)
    return {"per_device": {int(d): float(per_dev[i]) for i, d in enumerate(ids)},
            "mean": float(np.nanmean(per_dev)), "median": float(np.nanmedian(per_dev))}


def cut(E: np.ndarray, k: int) -> np.ndarray:
    P = E.copy()
    P[:, k:] = 0.0
    return P


def pca_trunc(E: np.ndarray, table: np.ndarray, k: int) -> np.ndarray:
    mu = table.mean(0)
    _, _, Vt = np.linalg.svd(table - mu, full_matrices=False)
    rec = ((E - mu) @ Vt[:k].T) @ Vt[:k] + mu
    return rec * (np.linalg.norm(E, axis=1, keepdims=True)
                  / np.linalg.norm(rec, axis=1, keepdims=True))


def print_geometry(name: str, g: dict) -> None:
    print(f"\n[{name}]  radius {g['radius_mean']:.4f}  participation ratio "
          f"{g['participation_ratio']:.1f}  PCs for 90% {g['pcs_for_90pct']}  "
          f"99% {g['pcs_for_99pct']}")
    print("     k   energy_k   var_k   pca_k  var/pca  mantel_k  prefix_radius")
    for k, r in g["per_k"].items():
        print(f"  {k:4d}   {r['energy_k']:7.3f}  {r['var_k']:6.3f}  {r['pca_k']:6.3f}"
              f"   {r['var_k'] / r['pca_k']:6.3f}   {r['mantel_k']:6.3f}   "
              f"{r['prefix_radius']:.4f}")
    print("  per-dim variance vs the all-dim average, by block: " + "  ".join(
        f"{b}:{v:.2f}" for b, v in g["var_per_dim_by_bin"].items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PREFIX / "embedding_analysis")
    ap.add_argument("--enrolled", action="store_true",
                    help="Also analyse <run>/enroll/enrolled_embeddings.pt (held-out amps).")
    ap.add_argument("--esr", action="store_true", help="Render the ESR curves (GPU).")
    ap.add_argument("--esr-source", choices=("table", "enrolled"), default="table",
                    help="Which rows to cut for the ESR curve.")
    ap.add_argument("--n-devices", type=int, default=100,
                    help="Table amps rendered per condition (random, seed 0).")
    ap.add_argument("--test-pairs", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    Ep, ids_p = load_rows(PREFIX / "embedding.pt")
    Ec, ids_c = load_rows(CONTROL / "embedding.pt")
    assert ids_p == ids_c, "the two tables must list the same amps in the same order"

    report = {"prefix": geometry(Ep), "control": geometry(Ec)}
    print_geometry("prefix  table", report["prefix"])
    print_geometry("control table", report["control"])
    report["mantel_prefix_vs_control"] = mantel(dists(Ep), dists(Ec))
    print(f"\n[same map?] Mantel r, prefix table vs control table distances: "
          f"{report['mantel_prefix_vs_control']:.3f}")

    if args.enrolled:
        for tag, run in (("prefix", PREFIX), ("control", CONTROL)):
            V, _ = load_rows(run / "enroll" / "enrolled_embeddings.pt")
            report[f"{tag}_enrolled"] = geometry(V)
            print_geometry(f"{tag} enrolled", report[f"{tag}_enrolled"])

    (args.out / "geometry.json").write_text(json.dumps(report, indent=2))
    if not args.esr:
        return

    from openamp.core.config import load_config

    cfg = load_config()
    if args.esr_source == "table":
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(len(ids_p), min(args.n_devices, len(ids_p)), replace=False))
        dev_ids = [ids_p[i] for i in pick]
        src = {"prefix": Ep[pick], "control": Ec[pick]}
    else:
        src, dev_ids = {}, None
        for tag, run in (("prefix", PREFIX), ("control", CONTROL)):
            V, vids = load_rows(run / "enroll" / "enrolled_embeddings.pt")
            order = np.argsort(vids)
            src[tag], got = V[order], [vids[i] for i in order]
            assert dev_ids is None or got == dev_ids
            dev_ids = got
    tables = {"prefix": Ep, "control": Ec}

    conds = [("prefix", "cut", k) for k in KS] \
        + [("control", "pca", k) for k in KS[:-1]] \
        + [("control", "cut", k) for k in KS]
    csv_path = args.out / f"esr_{args.esr_source}.csv"
    with csv_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(["run", "shape", "k", "cos_to_full", "esr_mean",
                                 "esr_median", "minutes"])
    print(f"\n[esr] {len(dev_ids)} {args.esr_source} amps x {args.test_pairs} test pairs")
    rows = []
    for tag, shape, k in conds:
        t0 = time.time()
        V = src[tag]
        P = cut(V, k) if shape == "cut" else pca_trunc(V, tables[tag], k)
        cos = float(np.mean(np.sum(P * V, 1) / (np.linalg.norm(P, axis=1)
                                                 * np.linalg.norm(V, axis=1))))
        esr = render_esr(cfg, PREFIX if tag == "prefix" else CONTROL,
                         {d: P[i].astype(np.float32) for i, d in enumerate(dev_ids)},
                         device=args.device, test_pairs=args.test_pairs)
        mins = (time.time() - t0) / 60
        row = {"run": tag, "shape": shape, "k": k, "cos_to_full": cos,
               "esr_mean": esr["mean"], "esr_median": esr["median"], "minutes": mins,
               "per_device": esr["per_device"]}
        rows.append(row)
        print(f"  {tag:8} {shape:4} k={k:4d}  cos {cos:.3f}  ESR mean {esr['mean']:.4f}  "
              f"median {esr['median']:.4f}  ({mins:.1f} min)", flush=True)
        with csv_path.open("a", newline="") as fh:
            csv.writer(fh).writerow([tag, shape, k, f"{cos:.4f}", f"{esr['mean']:.6f}",
                                     f"{esr['median']:.6f}", f"{mins:.2f}"])
        (args.out / f"esr_{args.esr_source}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
