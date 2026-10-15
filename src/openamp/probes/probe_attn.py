"""Train the package's own AttentionEncoder to predict a trained embedding table.

    OPENAMP_DATA_DIR=data python -m openamp.probes.probe_attn

Same question as :mod:`openamp.probes.probe` -- does a pooler over DAC frames
land an amp in the *right place*, supervised directly against an
already-trained embedding table rather than hoped for downstream -- but with
the package's actual :class:`~openamp.encoders.refine.AttentionEncoder`
(refinement blocks + positional frames) standing in for the probe's own
smaller single-block pool. ``probe.py`` answers "can cross-attention over
frames recover the table at all"; this answers "does the *shipped* head do
it," so a bad number here is about that head, not about pooling in general.

**Defaults to delta v2** (``delta_a2_tanh_r32_512_v2``, the same run
:mod:`scripts.frozen_decoder_joint` freezes): it is the one emulate run
trained with ``embed_normalize: true``, so every row of its table is
``embed_scale * normalize(raw_row)`` -- one global radius, not 405 different
ones. That is what makes fitting *this* encoder to *this* table a clean
comparison: ``AttentionEncoder(normalize=True)`` already ends in
``F.normalize``, i.e. it can only ever emit one radius too. Pointed at a run
without ``embed_normalize``, the fit below still runs (direction against the
table's own mean radius), but the constant-magnitude assumption the encoder
embodies no longer matches the target, and the script prints the target
table's coefficient of variation so that mismatch is visible rather than
silently absorbed into a worse cosine number. (``probe.py``'s direction +
log-magnitude split exists for exactly that non-constant case; it is not
needed here because v2's table is already single-radius.)

**Training reads a random subset of the frames** (``--keep-frac r``): per amp,
per step, ``round(r * n_frames)`` frame indices drawn without replacement and
gathered straight off the shard, in place of the whole 14,728-frame render.
With the whole render one device id maps to exactly one fixed encoder input,
so the training set holds as many distinct inputs as it holds amps and
memorizing them is a strategy available to the encoder -- the
``nopos_allwd0p1`` fit took it (train cos 0.980, cosine to the enrolled amps'
own vectors 0.381). A fresh draw each step makes one amp many inputs instead
of one. Scattered, not contiguous: this is dropout at ``1 - r`` over the frame
axis, deliberately *not*
:class:`~openamp.encoders.frames.FrameAttentionEncoder`'s ``frame_window``,
whose span is unbroken. A positional pool is still given each kept frame's
**absolute** index, so dropping frames changes what the pool reads, never
where it thinks it is reading.

**Validation and the report always read the whole render** -- ``--keep-frac``
is a training-time draw only. There are two validation groups and they are not
interchangeable: the held-out table amps (``--val-frac``) are the only thing
early stopping and the best-checkpoint pick ever look at, while the enrolled
amps are scored against their enrolled vectors every epoch purely as a watched
number. Nothing selects on them, so the ``[GATE]`` ESR over those same amps
stays a figure no model choice was made against.

**Prefix-weighted loss** (``--prefix-sizes``), for a table trained with
``embed_prefix_sizes`` (nested dropout), whose leading dims carry the coarse
amp and whose generator accepts cut vectors. Plain ``1 - cos`` treats all dims
alike; this instead averages the squared error of every cut ``[:k]`` of the
(normalized, then cut) prediction against the same cut of the target, which is
one per-dim weight stepping down from 1 (first block) to ``1/len(sizes)``
(last block) -- see :func:`prefix_loss`. With the full size alone it *is*
``1 - cos``.

Like ``probe.py``: the generator is not in the training loop, only loaded once
at the end, frozen, to turn predicted embeddings into a held-out ESR. Unlike
``probe.py``, the enrollment ceiling it gates against is read live from
``<run>/enroll/metrics.json`` rather than hardcoded -- ``probe.py``'s
``ENROLL_TEST_MEAN``/``ENROLL_TEST_MEDIAN`` are delta_a2_tanh_r32_512's
numbers (its own default ``--run``), not v2's, and reusing them here would
silently compare against the wrong run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from openamp.core.config import load_config
from openamp.encoders import AttentionEncoder, FrameStore
from openamp.evaluation import resolve_device
from openamp.models import count_parameters
from openamp.probes.probe import (MEANSTD_MANTEL, MEANSTD_PR, MEANSTD_TOP1,
                                  geometry_vs_table, holdout_esr, participation_ratio)
from openamp.training import GRAD_CLIP_NORM


def cosine_loss(pred_unit: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float]:
    """``1 - cos(pred, target direction)``. ``pred_unit`` is already on the unit
    sphere -- ``AttentionEncoder(normalize=True)``'s last op -- so nothing here
    renormalizes it; only the target's own direction is extracted."""
    tgt_dir = F.normalize(target, dim=-1)
    cos = F.cosine_similarity(pred_unit, tgt_dir, dim=-1)
    return (1.0 - cos).mean(), float(cos.mean())


def prefix_weights(sizes: tuple[int, ...], dim: int) -> torch.Tensor:
    """Per-dim weight ``w_j = #{k in sizes : k > j} / len(sizes)`` (0-indexed j).

    Summing the squared error over every prefix ``[:k]`` counts dim ``j`` once
    per prefix that contains it, so the nested sum is exactly one weighted sum:
    1 on the first ``sizes[0]`` dims, stepping down by ``1/len(sizes)`` at each
    size, ``1/len(sizes)`` on the last block.
    """
    j = torch.arange(dim)
    return (torch.as_tensor(sizes).unsqueeze(1) > j).sum(0).float() / len(sizes)


def prefix_loss(pred_unit: torch.Tensor, target: torch.Tensor, scale: float,
                weights: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Mean over prefix sizes of ``0.5 * |scale*pred[:k] - target[:k]|^2 / scale^2``.

    The prediction is normalized over all dims *first* and only then cut, so
    each term is the distance between the cut vectors the generator would see
    (it was trained on cuts with no renormalize). With sizes ``(dim,)`` alone
    this is ``1 - cos`` exactly, i.e. :func:`cosine_loss`, since both vectors
    sit at the one radius. The reported cosine is the full-vector one either way.
    """
    err = (scale * pred_unit - target) ** 2 / (2.0 * scale ** 2)
    cos = F.cosine_similarity(pred_unit, target, dim=-1)
    return (err * weights).sum(-1).mean(), float(cos.mean())


def enroll_ceiling(run_dir: Path) -> dict:
    """The run's own enrollment numbers: a vector fitted directly per held-out amp.
    Read live, not hardcoded -- see the module docstring."""
    path = Path(run_dir) / "enroll" / "metrics.json"
    if not path.is_file():
        return {}
    m = json.loads(path.read_text())
    return {k: m[k] for k in ("test_esr_mean", "test_esr_median", "n_enrolled") if k in m}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path,
                    default=Path("results/dac_frames/sweep_train"))
    ap.add_argument("--run", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2"),
                    help="Emulate run supplying the target table (embedding.pt) and "
                         "the frozen generator for the final ESR gate.")
    ap.add_argument("--out", type=Path, default=Path("results/encoder/encoder_probe_attn"))
    # Same defaults scripts.frozen_decoder_joint uses for this class, so a
    # table-regressed AttentionEncoder and a joint-trained one are comparable.
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--mlp-ratio", type=int, default=4)
    ap.add_argument("--no-positional", action="store_true",
                    help="drop the sinusoidal positions: the pool then reads the "
                         "frames as an unordered bag")
    ap.add_argument("--resume", type=Path, default=None,
                    help="probe_attn.pt to warm-start the encoder weights from; "
                         "the optimizer restarts fresh, and every other setting "
                         "(lr, weight_decay, epochs, ...) comes from this CLI")
    ap.add_argument("--batch", type=int, default=32, help="amps per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40,
                    help="stop once val has not improved for this many epochs; "
                         "0 = never stop early, run every epoch of --epochs. The "
                         "best-val weights are what the report uses either way.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--keep-frac", type=float, default=1.0,
                    help="fraction of a render's frames one training step reads, "
                         "drawn without replacement per amp per step (1.0 = the "
                         "whole render). Validation and the report always read "
                         "every frame.")
    ap.add_argument("--prefix-sizes", default="",
                    help="comma-separated prefix sizes, e.g. 8,16,32,64,128,256,512: "
                         "train on the prefix-weighted error (prefix_loss) instead of "
                         "1 - cos, for a table trained with embed_prefix_sizes. Empty "
                         "= the plain cosine loss. Val and best-checkpoint selection "
                         "use the same loss.")
    ap.add_argument("--test-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-esr", action="store_true",
                    help="Geometry only -- skip loading the generator.")
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

    # The radius every unit-output encoder is later rescaled onto, and how far
    # the table actually is from having just one -- see the module docstring.
    norms = Y.norm(dim=-1)
    scale, cv = float(norms.mean()), float(norms.std() / norms.mean().clamp_min(1e-8))
    print(f"[probe] target radius mean={scale:.4f} cv={cv:.4f} " +
          ("(embed_normalize: one radius)" if cv < 0.01 else
           "(varies — direction-only fit cannot recover magnitude; see probe.py "
           "for a direction+magnitude head)"))

    # The enrolled amps are a validation group now -- scored every epoch, never
    # selected on -- so their targets are built here rather than at the report.
    if enr is not None:
        enr_row = {int(d): i for i, d in enumerate(enr["device_ids"])}
        h_ids = [d for d in sorted(enr_row) if d in store]
        h_tgt = enr["embedding"].float().numpy()[[enr_row[d] for d in h_ids]]
        h_ids_arr = np.array(h_ids)
    else:
        h_ids, h_tgt, h_ids_arr = [], None, None

    if not 0.0 < args.keep_frac <= 1.0:
        raise SystemExit(f"--keep-frac must be in (0, 1], got {args.keep_frac}")
    keep_frames = (None if args.keep_frac >= 1.0
                   else max(1, int(round(args.keep_frac * store.n_frames))))

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ids))
    # --val-frac 0 -> no held-out split: every table amp is a training target,
    # nothing to early-stop on, and the final (overfit) weights are what the
    # report uses. Any positive fraction keeps the old behaviour.
    n_val = max(1, int(round(args.val_frac * len(ids)))) if args.val_frac > 0 else 0
    have_val = n_val > 0
    val_sel, train_sel = perm[:n_val], perm[n_val:]

    encoder = AttentionEncoder(
        embedding_dim=embedding_dim, frames=store, n_queries=args.n_queries,
        d_model=args.d_model, n_heads=args.n_heads, mlp_ratio=args.mlp_ratio,
        positional=not args.no_positional, normalize=True).to(dev)
    if args.resume is not None:
        ckd = torch.load(args.resume, map_location="cpu", weights_only=False)
        ra = ckd.get("args", {})
        # positional lives in a non-persistent buffer, so load_state_dict would
        # NOT catch a mismatch -- the other structural knobs it would.
        if bool(ra.get("no_positional", False)) != bool(args.no_positional):
            raise SystemExit(
                f"--resume: checkpoint was positional={not ra.get('no_positional', False)}, "
                f"this run is positional={not args.no_positional} -- structural, cannot warm-start")
        encoder.load_state_dict(ckd["state_dict"])
        print(f"[probe] resumed encoder weights from {args.resume} "
              f"(fresh optimizer; lr={args.lr:g} wd={args.weight_decay:g} "
              f"epochs={args.epochs} -- all from this CLI)")
    n_params = count_parameters(encoder)
    cap = encoder.capacity_report()
    print(f"[probe] frames {args.frames}  {len(ids)} amps x "
          f"[{store.n_frames}, {store.latent_dim}] fp16  target={args.run.name}")
    print(f"[probe] head={cap['head']} pool={cap['n_queries']}x{cap['d_model']}"
          f"x{cap['n_blocks']}blk/{cap['n_heads']}h positional={cap['positional']} "
          f"-> pooled={cap['pooled_dim']} -> De={embedding_dim}   {n_params:,} params "
          f"vs {len(train_sel) * embedding_dim:,} supervised values")
    print(f"[probe] split {len(train_sel)} train / {len(val_sel)} val"
          f"{' (no val — train on all, keep final weights)' if not have_val else ''}  "
          f"(+ {len(h_ids)} enrolled amps watched every epoch, never selected on)")
    print(f"[probe] a training step reads "
          + (f"{keep_frames} of {store.n_frames} frames per amp, redrawn each "
             f"step (--keep-frac {args.keep_frac:g})" if keep_frames else
             f"all {store.n_frames} frames — one fixed input per amp")
          + f"; val and the report read all {store.n_frames}")

    sizes = tuple(sorted(int(s) for s in args.prefix_sizes.split(",") if s.strip()))
    if sizes and not all(1 <= s <= embedding_dim for s in sizes):
        raise SystemExit(f"--prefix-sizes must lie in [1, {embedding_dim}], got {sizes}")
    weights = prefix_weights(sizes, embedding_dim).to(dev) if sizes else None
    if weights is None:
        print("[probe] loss: 1 - cos")
    else:
        print(f"[probe] loss: prefix-weighted error over sizes {list(sizes)} -- per-dim "
              "weight " + ", ".join(f"{a + 1}-{b}: {float(weights[a]):.3f}"
                                    for a, b in zip((0,) + sizes[:-1], sizes)))

    def loss_fn(pred_unit, target):
        if weights is None:
            return cosine_loss(pred_unit, target)
        return prefix_loss(pred_unit, target, scale, weights)

    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    Y_dev = Y.to(dev)

    H_dev = None if h_tgt is None else torch.from_numpy(h_tgt).float().to(dev)

    def encode(ids_t: torch.Tensor, n_keep: int | None) -> torch.Tensor:
        """``[B]`` device ids ``-> [B, De]`` unit vectors.

        ``n_keep=None`` is the encoder's own forward over the whole render.
        Otherwise ``n_keep`` frame indices are drawn per amp and the pool is
        called on those frames directly, with the drawn indices passed as
        positions -- a kept frame is read at the place it occupies in the
        render, not at its rank within the draw. Indices are sorted, which
        the pool does not care about but the page cache does.
        """
        if n_keep is None:
            return encoder(ids_t)
        dids = encoder.rows_for(ids_t).tolist()
        idx = np.stack([np.sort(rng.choice(store.n_frames, n_keep, replace=False))
                        for _ in dids])
        frames = store.batch(dids, dev, indices=idx).float()
        pos = torch.as_tensor(idx, device=dev) if encoder.pool.positional else None
        e = encoder.proj(encoder.pool(frames, None, pos))
        return F.normalize(e, dim=-1) if encoder.normalize else e

    def run_split(dev_ids, targets, sel, train: bool, n_keep: int | None = None):
        encoder.train(train)
        order = rng.permutation(sel) if train else sel
        tot = {"loss": 0.0, "cos": 0.0}
        n = 0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            ids_t = torch.as_tensor(dev_ids[b], dtype=torch.long, device=dev)
            tgt = targets[b]
            with torch.set_grad_enabled(train):
                pred = encode(ids_t, n_keep if train else None)
                loss, cos = loss_fn(pred, tgt)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP_NORM)
                opt.step()
            k = len(b)
            tot["loss"] += float(loss) * k; tot["cos"] += cos * k
            n += k
        return {k: v / max(n, 1) for k, v in tot.items()}

    # --- Train ----------------------------------------------------------------------
    best_val, best_state, since, t0 = float("inf"), None, 0, time.time()
    best_enr = None
    enr_sel = None if h_ids_arr is None else np.arange(len(h_ids_arr))
    for epoch in range(args.epochs):
        tr = run_split(ids, Y_dev, train_sel, True, keep_frames)
        with torch.no_grad():
            va = run_split(ids, Y_dev, val_sel, False) if have_val else None
            ve = (run_split(h_ids_arr, H_dev, enr_sel, False)
                  if enr_sel is not None else None)
        if have_val:
            if va["loss"] < best_val - 1e-6:
                best_val, since = va["loss"], 0
                best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
                best_enr = None if ve is None else ve["cos"]
            else:
                since += 1
        if epoch % 10 == 0 or since == 0 or epoch == args.epochs - 1:
            vtxt = f"val {va['loss']:.4f} (cos {va['cos']:.3f}) best {best_val:.4f}  " \
                   if have_val else ""
            etxt = f"enr cos {ve['cos']:+.3f}" if ve is not None else ""
            print(f"  e{epoch:03d} train {tr['loss']:.4f} (cos {tr['cos']:.3f}) "
                  f"{vtxt}{etxt}{'  *' if since == 0 and have_val else ''}", flush=True)
        if have_val and args.patience > 0 and since >= args.patience:
            print(f"[probe] no val improvement for {since} epochs — stopping")
            break
    if best_state is not None:
        encoder.load_state_dict(best_state)
    tail = f"best val {best_val:.4f}" if have_val else f"final train loss {tr['loss']:.4f}"
    print(f"[probe] trained in {(time.time() - t0) / 60:.1f} min, {tail}")

    # --- Predict everything ----------------------------------------------------------
    @torch.no_grad()
    def predict(device_ids):
        encoder.eval()
        E = []
        for s in range(0, len(device_ids), args.batch):
            b = device_ids[s:s + args.batch]
            ids_t = torch.as_tensor(b, dtype=torch.long, device=dev)
            E.append((scale * encoder(ids_t)).float().cpu().numpy())
        return np.concatenate(E)

    pred_seen = predict(list(ids))
    report: dict = {
        "n_params": int(n_params), "n_train": int(len(train_sel)),
        "n_val": int(len(val_sel)),
        "best_val_loss": best_val if have_val else None,
        "final_train_loss": float(tr["loss"]),
        "loss": "prefix" if weights is not None else "cosine",
        "prefix_sizes": list(sizes),
        "keep_frac": float(args.keep_frac),
        "keep_frames": int(keep_frames) if keep_frames else int(store.n_frames),
        "n_val_enrolled": int(len(h_ids)),
        "val_enrolled_cos_at_best": best_enr,
        "pooled_dim": int(cap["pooled_dim"]), "embedding_dim": int(embedding_dim),
        "target_radius_mean": scale, "target_radius_cv": cv,
    }

    # 1. Did the head place amps better than mean+std pooling of the same frames?
    report["predicted_geometry"] = geometry_vs_table(pred_seen, table[keep])
    gp = report["predicted_geometry"]
    print(f"\n[geom]  n={gp['n']} (same count mean+std's numbers were computed on)")
    print(f"[geom]  predicted PR {gp['participation_ratio']:7.2f}  "
          f"mantel {gp['mantel_r']:+.3f}  top1 {gp['top1_overlap']:.3f}")
    print(f"[geom]  mean+std  PR {MEANSTD_PR:7.2f}  mantel {MEANSTD_MANTEL:+.3f}  "
          f"top1 {MEANSTD_TOP1:.3f}   <- the bar")
    print(f"[geom]  target table PR {participation_ratio(table[keep]):.2f}")

    # 2. The held-out amps: cosine to the enrolled direction, and the ESR gate.
    if h_tgt is not None:
        pred_hold = predict(h_ids)
        pm, tm = np.linalg.norm(pred_hold, axis=1), np.linalg.norm(h_tgt, axis=1)
        cos = np.sum(pred_hold / pm[:, None] * h_tgt / tm[:, None], axis=1)
        report["holdout_embedding"] = {
            "n": len(h_ids), "cos_mean": float(cos.mean()),
            "pred_norm_mean": float(pm.mean()), "enrolled_norm_mean": float(tm.mean()),
        }
        r = report["holdout_embedding"]
        print(f"\n[hold]  vs enrolled: cos {r['cos_mean']:+.3f}  "
              f"|pred| {r['pred_norm_mean']:.2f} vs |enrolled| "
              f"{r['enrolled_norm_mean']:.2f}")

        if not args.skip_esr:
            print(f"[hold]  rendering {len(h_ids)} held-out amps through the frozen "
                  f"generator ...", flush=True)
            esr = holdout_esr(cfg, args.run, {d: pred_hold[i] for i, d in enumerate(h_ids)},
                              device=args.device, test_pairs=args.test_pairs)
            report["holdout_esr"] = esr
            ceiling = enroll_ceiling(args.run)
            print(f"\n[GATE]  predicted-e test ESR  mean {esr['mean']:.4f}  "
                  f"median {esr['median']:.4f}")
            if ceiling:
                report["enroll_ceiling"] = ceiling
                print(f"[GATE]  enrollment ceiling ({ceiling.get('n_enrolled')} amps, "
                      f"{args.run.name})  mean {ceiling['test_esr_mean']:.4f}  "
                      f"median {ceiling['test_esr_median']:.4f}")
                ratio = esr["mean"] / ceiling["test_esr_mean"]
                print(f"[GATE]  {ratio:.1f}x the ceiling")
            else:
                print(f"[GATE]  no enroll/metrics.json under {args.run} — no ceiling "
                      f"to compare against")

    torch.save({"state_dict": encoder.state_dict(), "args": vars(args),
                "device_ids": ids.tolist(), "report": report}, args.out / "probe_attn.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float),
                                          encoding="utf-8")
    print(f"\n[probe] wrote {args.out}/report.json")


if __name__ == "__main__":
    main()
