"""Train a cross-attention pooler to predict a trained embedding table; report whether it placed.

    OPENAMP_DATA_DIR=data python -m openamp.probes.probe

Reads the DAC frames :mod:`openamp.probes.extract` wrote and the embedding table
an ``openamp emulate`` run already trained, and learns the map between them. The
generator is not in the training loop at all -- it is loaded once, frozen, at the
end, to turn predicted embeddings into the only number that decides anything.

**Why regress the table instead of training against reconstruction.** Placement,
not recognition, is where every encoder in this project has failed: ``joint/proto``
reaches holdout retrieval top-1 0.978 with a holdout ESR of 0.762, against 0.254
for a vector fitted directly on those amps. Retrieval measures whether two clips
of one amp land together; it says nothing about whether they land in the *right
place*. Supervising the table directly makes placement the training signal rather
than something hoped for downstream, and it costs minutes instead of hours.

**The two heads.** The trained table's vectors have mean norm 6.49; the vectors
enrollment fitted for the 45 held-out amps have mean norm 18.43, and both render
well. Magnitude is therefore the axis where seen and unseen demonstrably
disagree, so it is predicted separately (log-magnitude scalar) from direction
(unit vector, cosine loss). If magnitude is what fails to transfer, that keeps
the failure out of the direction and lets each be scored on its own.

**No L2 normalization of the output.** The generator here is *frozen* and was
trained against table vectors of norm ~6.5; projecting onto the unit sphere would
under-drive it. (``enc_normalize: true`` is right in the joint arm only because
the generator co-adapts there.)

**Overfitting is controlled by capacity alone.** There are 405 training amps and
one deterministic input each -- no augmentation, by design -- so the pooler is
kept small (~230k params against 405 x 512 = 207,360 supervised values) and
early-stopped on a validation split held out of the 405. The 45 enrolled amps are
never touched until the final report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from openamp.core.config import Config, load_config
from openamp.data import CorpusPaths, EmulationDataset, make_loader
from openamp.encoders import CrossAttentionPool, FrameStore
from openamp.evaluation import (SILENCE_DBFS, load_model, per_device_esr,
                                resolve_device)
from openamp.spec import spec_from_checkpoint

# What mean+std pooling of the same DAC latents achieves, measured over the same
# 405 devices. The pooler has to beat these or the input was never the problem.
MEANSTD_PR = 2.84
MEANSTD_TOP1 = 0.272
MEANSTD_MANTEL = -0.023

# The enroll run's own numbers on the 45 held-out amps, i.e. what a vector fitted
# directly on each amp achieves with this generator. The placement ceiling.
ENROLL_TEST_MEAN = 0.0364
ENROLL_TEST_MEDIAN = 0.0124
NO_FIT_BASELINE = 1.512


# --- Model ----------------------------------------------------------------------
class EmbeddingRegressor(nn.Module):
    """``[B, T, D] frames -> [B, embedding_dim]``, as ``magnitude * direction``."""

    def __init__(self, *, in_dim: int = 1024, embedding_dim: int = 512,
                 n_queries: int = 4, d_model: int = 64, hidden: int = 128):
        super().__init__()
        self.pool = CrossAttentionPool(in_dim, n_queries, d_model)
        self.trunk = nn.Sequential(nn.Linear(self.pool.pooled_dim, hidden), nn.Tanh())
        self.dir_head = nn.Linear(hidden, embedding_dim)
        self.mag_head = nn.Linear(hidden, 1)

    def forward(self, frames: torch.Tensor):
        """Returns ``(e, direction, magnitude)`` -- the parts stay separable so the
        report can say which of the two failed."""
        pooled = self.pool(frames)
        h = self.trunk(pooled)
        direction = F.normalize(self.dir_head(h), dim=-1)
        magnitude = torch.exp(self.mag_head(h)).squeeze(-1)
        return magnitude.unsqueeze(-1) * direction, direction, magnitude


def regression_loss(direction, magnitude, target, mag_weight: float):
    """Cosine on direction + squared error in log-magnitude, reported separately."""
    tgt_mag = target.norm(dim=-1).clamp_min(1e-8)
    tgt_dir = target / tgt_mag.unsqueeze(-1)
    cos = F.cosine_similarity(direction, tgt_dir, dim=-1)
    dir_loss = (1.0 - cos).mean()
    mag_loss = ((magnitude.clamp_min(1e-8).log() - tgt_mag.log()) ** 2).mean()
    return dir_loss + mag_weight * mag_loss, float(dir_loss), float(mag_loss), float(cos.mean())


# --- Geometry diagnostics -----------------------------------------------------------
def participation_ratio(x: np.ndarray) -> float:
    """Effective dimensionality of a cloud. Biased by sample count — only ever
    compare it against a figure computed on the same number of points."""
    xc = np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean(0)
    ev = np.linalg.svd(xc, compute_uv=False) ** 2
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def _cos_dist(x: np.ndarray) -> np.ndarray:
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return 1.0 - xn @ xn.T


def geometry_vs_table(pred: np.ndarray, table: np.ndarray) -> dict:
    """How the predicted/pooled cloud's geometry compares with the target table's."""
    Dp, Dt = _cos_dist(pred), _cos_dist(table)
    iu = np.triu_indices(len(pred), 1)
    mantel = float(np.corrcoef(Dp[iu], Dt[iu])[0, 1])
    out = {"n": int(len(pred)), "participation_ratio": participation_ratio(pred),
           "mantel_r": mantel}
    for k in (1, 5):
        np_, nt = np.argsort(Dp, 1)[:, 1:k + 1], np.argsort(Dt, 1)[:, 1:k + 1]
        out[f"top{k}_overlap"] = float(
            np.mean([len(set(a) & set(b)) / k for a, b in zip(np_, nt)]))
    return out


# --- Held-out ESR through the frozen generator ---------------------------------------
@torch.no_grad()
def holdout_esr(cfg: Config, run_dir: Path, pred: dict[int, np.ndarray], *,
                device: str, test_pairs: int) -> dict:
    """Install predicted vectors as the embedding table and measure test ESR.

    Uses the same per-window raw-ratio semantics with the silence gate that
    ``enroll`` reports ``test_esr_mean`` under, so the numbers are directly
    comparable rather than merely similar.
    """
    dev = torch.device(resolve_device(device))
    model, ck = load_model(run_dir, str(dev))
    spec = spec_from_checkpoint(ck)
    ids = sorted(pred)
    idx = {d: i for i, d in enumerate(ids)}

    emb = nn.Embedding(len(ids), model.embedding_dim).to(dev)
    with torch.no_grad():
        emb.weight.copy_(torch.from_numpy(np.stack([pred[d] for d in ids])).to(dev))
    model.requires_grad_(False)
    model.embedding = emb
    model.eval()

    R = model.receptive_field
    clip = int(round(spec.data.clip_seconds * cfg.sample_rate))
    ds = EmulationDataset(CorpusPaths.from_config(cfg), "test", receptive_field=R,
                          id_to_idx=idx, clip_samples=clip,
                          pairs_per_epoch=test_pairs * len(ids), seed=cfg.seed + 7)
    loader = make_loader(ds, spec.training.batch_size, spec.training.num_workers,
                         shuffle=False,
                          seed=cfg.seed + 7, drop_last=False)
    per_dev = per_device_esr(model, loader, dev, len(ids),
                              silence_dbfs=SILENCE_DBFS, clip=clip)
    return {"per_device": {int(d): float(per_dev[i]) for i, d in enumerate(ids)},
            "mean": float(np.nanmean(per_dev)), "median": float(np.nanmedian(per_dev))}


# --- Main -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path,
                    default=Path("results/dac_frames/sweep_train"))
    ap.add_argument("--run", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512"),
                    help="Emulate run supplying the target table and the frozen generator.")
    ap.add_argument("--out", type=Path, default=Path("results/encoder/encoder_probe"))
    ap.add_argument("--n-queries", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mag-weight", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32, help="amps per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-esr", action="store_true",
                    help="Geometry only — skip loading the generator.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(resolve_device(args.device))
    cfg = load_config()
    args.out.mkdir(parents=True, exist_ok=True)

    # --- Targets -------------------------------------------------------------------
    tbl = torch.load(args.run / "embedding.pt", map_location="cpu", weights_only=False)
    table = tbl["embedding"].float().numpy()
    table_ids = [int(d) for d in tbl["device_ids"]]
    enrolled_path = args.run / "enroll" / "enrolled_embeddings.pt"
    enr = torch.load(enrolled_path, map_location="cpu", weights_only=False) \
        if enrolled_path.is_file() else None

    store = FrameStore(args.frames)
    keep = [i for i, d in enumerate(table_ids) if d in store]
    if len(keep) < len(table_ids):
        print(f"[probe] {len(table_ids) - len(keep)} target device(s) have no frames "
              f"— dropped")
    ids = np.array([table_ids[i] for i in keep])
    Y = torch.from_numpy(table[keep]).float()
    embedding_dim = Y.shape[1]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ids))
    n_val = max(1, int(round(args.val_frac * len(ids))))
    val_sel, train_sel = perm[:n_val], perm[n_val:]

    model = EmbeddingRegressor(in_dim=store.latent_dim, embedding_dim=embedding_dim,
                               n_queries=args.n_queries, d_model=args.d_model,
                               hidden=args.hidden).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[probe] frames {args.frames}  {len(ids)} amps x "
          f"[{store.n_frames}, {store.latent_dim}] fp16  target={args.run.name}")
    print(f"[probe] pooler n_q={args.n_queries} d={args.d_model} -> pooled "
          f"{model.pool.pooled_dim} -> De={embedding_dim}   {n_params:,} params "
          f"vs {len(train_sel) * embedding_dim:,} supervised values")
    print(f"[probe] split {len(train_sel)} train / {len(val_sel)} val  "
          f"(the {0 if enr is None else len(enr['device_ids'])} enrolled amps are "
          f"untouched until the report)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Y_dev = Y.to(dev)

    def run_split(sel, train: bool):
        model.train(train)
        order = rng.permutation(sel) if train else sel
        tot = {"loss": 0.0, "dir": 0.0, "mag": 0.0, "cos": 0.0}
        n = 0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            frames = store.batch(ids[b], dev).float()
            tgt = Y_dev[b]
            with torch.set_grad_enabled(train):
                _, direction, magnitude = model(frames)
                loss, dl, ml, cs = regression_loss(direction, magnitude, tgt,
                                                   args.mag_weight)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            k = len(b)
            tot["loss"] += float(loss) * k; tot["dir"] += dl * k
            tot["mag"] += ml * k; tot["cos"] += cs * k
            n += k
            del frames
        return {k: v / max(n, 1) for k, v in tot.items()}

    # --- Train ----------------------------------------------------------------------
    best_val, best_state, since, t0 = float("inf"), None, 0, time.time()
    for epoch in range(args.epochs):
        tr = run_split(train_sel, True)
        with torch.no_grad():
            va = run_split(val_sel, False)
        if va["loss"] < best_val - 1e-6:
            best_val, since = va["loss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if epoch % 10 == 0 or since == 0 or epoch == args.epochs - 1:
            print(f"  e{epoch:03d} train {tr['loss']:.4f} (cos {tr['cos']:.3f}) "
                  f"val {va['loss']:.4f} (cos {va['cos']:.3f} dir {va['dir']:.4f} "
                  f"mag {va['mag']:.4f}) best {best_val:.4f}"
                  f"{'  *' if since == 0 else ''}", flush=True)
        if since >= args.patience:
            print(f"[probe] no val improvement for {since} epochs — stopping")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[probe] trained in {(time.time() - t0) / 60:.1f} min, best val {best_val:.4f}")

    # --- Predict everything ----------------------------------------------------------
    @torch.no_grad()
    def predict(device_ids):
        model.eval()
        E, P = [], []
        for s in range(0, len(device_ids), args.batch):
            b = device_ids[s:s + args.batch]
            frames = store.batch(b, dev).float()
            e, _, _ = model(frames)
            E.append(e.float().cpu().numpy())
            P.append(model.pool(frames).float().cpu().numpy())
            del frames
        return np.concatenate(E), np.concatenate(P)

    pred_seen, pooled_seen = predict(list(ids))
    report: dict = {
        "n_params": int(n_params), "n_train": int(len(train_sel)),
        "n_val": int(len(val_sel)), "best_val_loss": best_val,
        "pooled_dim": int(model.pool.pooled_dim), "embedding_dim": int(embedding_dim),
    }

    # 1. Did the pooler widen the input past what mean+std managed?
    report["pooled_geometry"] = geometry_vs_table(pooled_seen, table[keep])
    report["predicted_geometry"] = geometry_vs_table(pred_seen, table[keep])
    g, gp = report["pooled_geometry"], report["predicted_geometry"]
    print(f"\n[geom]  n={g['n']} (same count mean+std's numbers were computed on)")
    print(f"[geom]  pooled    PR {g['participation_ratio']:7.2f}  "
          f"mantel {g['mantel_r']:+.3f}  top1 {g['top1_overlap']:.3f}")
    print(f"[geom]  predicted PR {gp['participation_ratio']:7.2f}  "
          f"mantel {gp['mantel_r']:+.3f}  top1 {gp['top1_overlap']:.3f}")
    print(f"[geom]  mean+std  PR {MEANSTD_PR:7.2f}  mantel {MEANSTD_MANTEL:+.3f}  "
          f"top1 {MEANSTD_TOP1:.3f}   <- the bar")
    print(f"[geom]  target table PR {participation_ratio(table[keep]):.2f}")

    # 2. The held-out amps: direction and magnitude, scored apart.
    if enr is not None:
        enr_row = {int(d): i for i, d in enumerate(enr["device_ids"])}
        h_ids = [d for d in sorted(enr_row) if d in store]
        h_tgt = enr["embedding"].float().numpy()[[enr_row[d] for d in h_ids]]
        pred_hold, _ = predict(h_ids)
        pm = np.linalg.norm(pred_hold, axis=1)
        tm = np.linalg.norm(h_tgt, axis=1)
        cos = np.sum(pred_hold / pm[:, None] * h_tgt / tm[:, None], axis=1)
        report["holdout_embedding"] = {
            "n": len(h_ids), "cos_mean": float(cos.mean()),
            "pred_norm_mean": float(pm.mean()), "enrolled_norm_mean": float(tm.mean()),
            "log_mag_rmse": float(np.sqrt(np.mean((np.log(pm) - np.log(tm)) ** 2))),
        }
        r = report["holdout_embedding"]
        print(f"\n[hold]  vs enrolled: cos {r['cos_mean']:+.3f}  "
              f"|pred| {r['pred_norm_mean']:.2f} vs |enrolled| "
              f"{r['enrolled_norm_mean']:.2f}  log-mag rmse {r['log_mag_rmse']:.3f}")

        # 3. The gate.
        if not args.skip_esr:
            print(f"[hold]  rendering {len(h_ids)} held-out amps through the frozen "
                  f"generator ...", flush=True)
            esr = holdout_esr(cfg, args.run, {d: pred_hold[i] for i, d in enumerate(h_ids)},
                              device=args.device, test_pairs=args.test_pairs)
            report["holdout_esr"] = esr
            print(f"\n[GATE]  predicted-e test ESR  mean {esr['mean']:.4f}  "
                  f"median {esr['median']:.4f}")
            print(f"[GATE]  enrollment ceiling      mean {ENROLL_TEST_MEAN:.4f}  "
                  f"median {ENROLL_TEST_MEDIAN:.4f}")
            print(f"[GATE]  no-fit baseline         mean {NO_FIT_BASELINE:.4f}")
            ratio = esr["mean"] / ENROLL_TEST_MEAN
            print(f"[GATE]  {ratio:.1f}x the ceiling — "
                  + ("placement works" if ratio < 2 else
                     "recognizes but does not place" if esr["mean"] < NO_FIT_BASELINE
                     else "no better than not fitting at all"))

    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "device_ids": ids.tolist(), "report": report}, args.out / "probe.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float),
                                          encoding="utf-8")
    print(f"\n[probe] wrote {args.out}/report.json")


if __name__ == "__main__":
    main()
