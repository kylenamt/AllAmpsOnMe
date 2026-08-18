"""The diagnostics gate for a trained tone encoder — run this before wiring it into anything.

Six checks. The first two decide whether the direction is alive at all; the rest
say whether the two-branch design earned its keep.

======================  =======================================================
Check                   Pass condition
======================  =======================================================
k-NN retrieval          Top-1 on **unseen captures** far above chance (1/N).
Dry-channel ablation    Zeroing dry must **hurt**. See below.
Branch redundancy       Mean |corr| between branches low — else the split is
                        decorative and the honest response is to collapse to
                        one branch, not to tune the weights.
Amp probe <- static     High. Trained on the train devices' held-out time
                        region, because held-out *devices* carry ``tone_id``s
                        the probe has never seen and could not classify.
Gain probe <- dynamic   High, **and higher than the same probe from static** —
                        that ordering is the actual claim, not the raw number.
t-SNE                   Clusters by capture (eyeball; coords are written out).
======================  =======================================================

**Why the ablation is not optional.** In sweep-only training every device is fed
the *same* dry file, so the dry channel carries no capture identity whatsoever.
An encoder could therefore ignore it completely, solve SupCon from the wet
channel alone, and still post a perfect retrieval score — while learning "what
this recording sounds like" instead of "what this amp does to a signal". Zeroing
dry at eval time is what separates those two hypotheses. It is the same role
``val_esr_shuffled`` plays for the emulators (``docs/decisions.md`` §5).

Writes ``diagnostics.json``, ``capture_embeddings.npz``, ``tsne.csv`` and
``tsne.png`` into the run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from openamp.core.config import Config, EncoderConfig
from openamp.emulate.dataset import build_device_index, device_names
from openamp.encoder.dataset import (CaptureGridDataset, device_labels,
                                     time_split_regions)
from openamp.encoder.model import aggregate_capture
from openamp.encoder.train import load_encoder, make_loader

EVAL_WINDOWS = 8            # segments per capture on the shared grid
TSNE_PERPLEXITY = 15.0


def _resolve_device(name: str) -> str:
    return name if (name != "cuda" or torch.cuda.is_available()) else "cpu"


@torch.no_grad()
def embed_captures(model, cfg: Config, ncfg: EncoderConfig, device_ids: list[int],
                   *, dev, split: str = "train", regions=((0.0, 1.0),),
                   n_windows: int = EVAL_WINDOWS, zero_dry: bool = False,
                   labels: dict | None = None) -> dict:
    """Per-segment and per-capture embeddings for ``device_ids`` on a shared grid.

    The window grid is drawn once and replayed for every device — same reasoning as
    :class:`~openamp.emulate.dataset.DeviceGridDataset`: with independent random
    windows a capture can look distinctive or bland on window luck alone, and
    between-device comparisons are the whole point here.
    """
    id_to_idx = {d: i for i, d in enumerate(device_ids)}
    ds = CaptureGridDataset(cfg, split, id_to_idx=id_to_idx,
                            segment_samples=ncfg.segment_samples, sources=ncfg.sources,
                            regions=regions, n_windows=n_windows,
                            labels=labels, seed=cfg.seed,
                            undo_output_scale=ncfg.undo_output_scale)
    loader = make_loader_eval(ds, ncfg, n_windows)
    seg_s, seg_d, seg_dev, seg_tone, seg_gain, seg_crest = [], [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(dev, non_blocking=True)
        if zero_dry:
            x = x.clone()
            x[:, 0, :] = 0.0                     # keep the channel, remove the content
        _, s, d = model(x)
        seg_s.append(s.float().cpu())
        seg_d.append(d.float().cpu())
        seg_dev.append(batch["device_id"])
        seg_tone.append(batch["tone_idx"])
        seg_gain.append(batch["gain_idx"])
        seg_crest.append(batch["crest"])
    seg_s = torch.cat(seg_s)
    seg_d = torch.cat(seg_d)
    seg_dev = torch.cat(seg_dev).numpy()
    out_ids, cap_vecs = [], []
    for did in device_ids:
        m = seg_dev == int(did)
        if not m.any():
            continue
        out_ids.append(int(did))
        cap_vecs.append(aggregate_capture(seg_s[m], seg_d[m]))
    return {
        "segment_static": seg_s.numpy(), "segment_dynamic": seg_d.numpy(),
        "segment_device": seg_dev,
        "segment_tone": torch.cat(seg_tone).numpy(),
        "segment_gain": torch.cat(seg_gain).numpy(),
        "segment_crest": torch.cat(seg_crest).numpy(),
        "capture_device": np.asarray(out_ids, dtype=np.int64),
        "capture_embedding": torch.stack(cap_vecs).numpy() if cap_vecs else np.zeros((0, 0)),
    }


def make_loader_eval(ds, ncfg: EncoderConfig, n_windows: int):
    """Sequential loader; batch = one capture's windows so aggregation stays device-major."""
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=max(1, int(n_windows)), shuffle=False,
                      num_workers=ncfg.num_workers, drop_last=False, pin_memory=True)


# --- The individual checks ------------------------------------------------------
def branch_redundancy(static: np.ndarray, dynamic: np.ndarray) -> dict:
    """Mean |Pearson r| between every static and dynamic feature, over captures.

    High means the dynamic branch re-derived the static one and the two-branch
    split is buying nothing — the cross-covariance term exists to prevent exactly
    this, and this number is how you find out whether it worked.
    """
    if static.shape[0] < 3:
        return {"mean_abs_corr": float("nan"), "max_abs_corr": float("nan"), "n": 0}
    a = static - static.mean(0)
    b = dynamic - dynamic.mean(0)
    a = a / (a.std(0) + 1e-8)
    b = b / (b.std(0) + 1e-8)
    c = np.abs(a.T @ b) / a.shape[0]
    return {"mean_abs_corr": float(c.mean()), "max_abs_corr": float(c.max()),
            "n": int(static.shape[0])}


def knn_retrieval(seg_vec: np.ndarray, seg_device: np.ndarray,
                  cap_vec: np.ndarray, cap_device: np.ndarray) -> dict:
    """Query with one segment, retrieve against the capture gallery. Top-1 / top-5.

    Chance is ``1 / n_captures``; the gate is that top-1 sits far above it. The
    per-device ranks are returned too, so a single broken capture (device 486 is
    a near-silent capture parked in the holdout for *exclusion*) shows up as
    itself rather than quietly dragging the mean.
    """
    if cap_vec.size == 0 or seg_vec.size == 0:
        return {"top1": float("nan"), "top5": float("nan"), "n_captures": 0, "chance": 0.0}
    # Explicit id -> gallery-column map: a device with no usable segments is simply
    # absent from the gallery, and a positional guess would then score every query
    # against the wrong capture.
    col = {int(d): i for i, d in enumerate(cap_device)}
    keep = np.array([int(d) in col for d in seg_device])
    if not keep.any():
        return {"top1": float("nan"), "top5": float("nan"),
                "n_captures": int(cap_vec.shape[0]), "chance": 0.0}
    seg_vec, seg_device = seg_vec[keep], seg_device[keep]
    q = seg_vec / (np.linalg.norm(seg_vec, axis=1, keepdims=True) + 1e-12)
    g = cap_vec / (np.linalg.norm(cap_vec, axis=1, keepdims=True) + 1e-12)
    sim = q @ g.T                                   # [n_segments, n_captures]
    order = np.argsort(-sim, axis=1)
    truth = np.asarray([col[int(d)] for d in seg_device])
    rank = np.argmax(order == truth[:, None], axis=1)
    per_device = {}
    for did in np.unique(seg_device):
        m = seg_device == did
        per_device[int(did)] = {"top1": float((rank[m] == 0).mean()),
                                "mean_rank": float(rank[m].mean())}
    return {"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()),
            "mean_rank": float(rank.mean()), "n_captures": int(cap_vec.shape[0]),
            "n_queries": int(seg_vec.shape[0]),
            "chance": 1.0 / max(1, cap_vec.shape[0]), "per_device": per_device}


def _linear_probe_cls(x_tr, y_tr, x_te, y_te, *, n_class: int, steps: int = 300,
                      lr: float = 0.05) -> float:
    """Accuracy of a multinomial logistic probe on frozen features."""
    keep_tr, keep_te = y_tr >= 0, y_te >= 0
    if keep_tr.sum() < 2 or keep_te.sum() < 1:
        return float("nan")
    xt = torch.from_numpy(x_tr[keep_tr]).float()
    yt = torch.from_numpy(y_tr[keep_tr]).long()
    xe = torch.from_numpy(x_te[keep_te]).float()
    ye = torch.from_numpy(y_te[keep_te]).long()
    lin = torch.nn.Linear(xt.shape[1], int(n_class))
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    lossfn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        lossfn(lin(xt), yt).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(xe).argmax(-1) == ye).float().mean())


def _linear_probe_reg(x_tr, y_tr, x_te, y_te) -> float:
    """R^2 of a closed-form ridge regression on frozen features (no tuning to hide in)."""
    if len(y_tr) < 3 or len(y_te) < 2:
        return float("nan")
    a = np.concatenate([x_tr, np.ones((x_tr.shape[0], 1))], axis=1)
    w = np.linalg.solve(a.T @ a + 1e-3 * np.eye(a.shape[1]), a.T @ y_tr)
    b = np.concatenate([x_te, np.ones((x_te.shape[0], 1))], axis=1)
    pred = b @ w
    ss_res = float(np.sum((y_te - pred) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _split_by_index(n: int, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = max(1, int((1.0 - frac) * n))
    return idx[:cut], idx[cut:]


def _tsne(x: np.ndarray, seed: int) -> np.ndarray | None:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return None
    if x.shape[0] < 5:
        return None
    p = min(TSNE_PERPLEXITY, max(2.0, (x.shape[0] - 1) / 3.0))
    return TSNE(n_components=2, perplexity=p, init="pca",
                random_state=int(seed)).fit_transform(x)


def _write_tsne(run_dir: Path, coords, ids, names) -> None:
    import csv

    with (run_dir / "tsne.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["device_id", "x", "y", "name"])
        for did, (x, y) in zip(ids, coords):
            w.writerow([int(did), f"{x:.4f}", f"{y:.4f}", names.get(int(did), "")])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(ids)), cmap="turbo", s=28)
    for did, (x, y) in zip(ids, coords):
        ax.annotate(str(int(did)), (x, y), fontsize=5, alpha=0.6)
    ax.set_title("Tone embeddings, held-out captures (t-SNE)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(run_dir / "tsne.png")
    plt.close(fig)


# --- The gate -------------------------------------------------------------------
def diagnose(cfg: Config, run_dir: Path, *, device: str = "cuda",
             n_windows: int = EVAL_WINDOWS) -> dict:
    """Run all six checks against a finished encoder run; writes + returns the report."""
    run_dir = Path(run_dir)
    dev = torch.device(_resolve_device(device))
    model, ck = load_encoder(run_dir, str(dev))
    ncfg = EncoderConfig(**ck["encoder_cfg"])
    train_ids = [int(d) for d in ck["device_ids"]]
    held_ids = sorted(int(d) for d in ck.get("holdout_ids", []))

    sig = ck.get("manifest_sha256", "")
    from openamp.emulate.dataset import manifest_signature
    live = manifest_signature(cfg)
    if sig and live and sig != live:
        print(f"[encode-eval] WARNING manifest sha256 differs from the checkpoint's "
              f"({sig[:8]} != {live[:8]}) — are you pointing OPENAMP_DATA_DIR at the "
              f"corpus this run trained on?", flush=True)

    # Held-out devices must actually be rendered in this data dir.
    present = set(build_device_index(cfg)[0])
    held_ids = [d for d in held_ids if d in present]
    labels = device_labels(cfg, sorted(set(train_ids) | set(held_ids)))
    # The probes must run on the same val regions the run trained against, so this
    # comes from the checkpoint's config, not the live one. Printed because an old
    # checkpoint predating time_split_mode falls back to the dataclass default and
    # would otherwise be scored on a different slice of the file than it was trained on.
    _, val_regions = time_split_regions(ncfg.time_val_frac, mode=ncfg.time_split_mode,
                                        blocks=ncfg.time_split_blocks,
                                        within=tuple(ncfg.usable_region))
    print(f"[encode-eval] run={ck.get('name', run_dir.name)} "
          f"train_devices={len(train_ids)} held_out={len(held_ids)} "
          f"windows={n_windows} dev={dev}", flush=True)
    print(f"[encode-eval] probe region: {ncfg.time_split_mode} split, "
          f"{[(round(a, 3), round(b, 3)) for a, b in val_regions]}", flush=True)

    # 1-2. Retrieval on UNSEEN captures, with and without the dry channel.
    held = embed_captures(model, cfg, ncfg, held_ids, dev=dev, n_windows=n_windows,
                          labels=labels)
    held_ablate = embed_captures(model, cfg, ncfg, held_ids, dev=dev,
                                 n_windows=n_windows, zero_dry=True, labels=labels)
    seg512 = np.concatenate([held["segment_static"], held["segment_dynamic"]], axis=1)
    retrieval = knn_retrieval(seg512, held["segment_device"],
                              held["capture_embedding"], held["capture_device"])
    seg512_ab = np.concatenate([held_ablate["segment_static"],
                                held_ablate["segment_dynamic"]], axis=1)
    retrieval_ab = knn_retrieval(seg512_ab, held_ablate["segment_device"],
                                 held_ablate["capture_embedding"],
                                 held_ablate["capture_device"])

    # 3. Branch redundancy over held-out captures.
    half = held["capture_embedding"].shape[1] // 2 if held["capture_embedding"].size else 0
    redundancy = branch_redundancy(held["capture_embedding"][:, :half],
                                   held["capture_embedding"][:, half:]) if half else {}

    # 4-5. Linear probes on the TRAIN devices' held-out time region. Held-out
    # devices carry tone_ids the probe has never seen, so amp ID cannot be scored
    # there — the split that answers "is this information present" is over time.
    probe = embed_captures(model, cfg, ncfg, train_ids, dev=dev, regions=val_regions,
                           n_windows=n_windows, labels=labels)
    n = probe["segment_static"].shape[0]
    tr, te = _split_by_index(n, 0.3, cfg.seed)
    s, d = probe["segment_static"], probe["segment_dynamic"]
    tone, gain, crest = probe["segment_tone"], probe["segment_gain"], probe["segment_crest"]
    # Class count comes from the labels built HERE, not from the checkpoint.
    # ``labels`` is indexed over train + held-out devices, a strictly wider set than
    # the one the run trained on, so ``ck["n_tones"]`` is too small and the two
    # tone_idx spaces are not interchangeable. These probes are fit fresh anyway, so
    # only the eval-time indexing matters — but never compare an index across the two.
    n_tone_cls = int(labels["n_tones"])
    n_gain_cls = int(labels["n_gain"])
    probes = {
        "amp_from_static": _linear_probe_cls(s[tr], tone[tr], s[te], tone[te],
                                             n_class=n_tone_cls),
        "amp_from_dynamic": _linear_probe_cls(d[tr], tone[tr], d[te], tone[te],
                                              n_class=n_tone_cls),
        "gain_from_dynamic": _linear_probe_cls(d[tr], gain[tr], d[te], gain[te],
                                               n_class=n_gain_cls),
        "gain_from_static": _linear_probe_cls(s[tr], gain[tr], s[te], gain[te],
                                              n_class=n_gain_cls),
        "crest_r2_from_dynamic": _linear_probe_reg(d[tr], crest[tr], d[te], crest[te]),
        "crest_r2_from_static": _linear_probe_reg(s[tr], crest[tr], s[te], crest[te]),
    }
    # The dry-channel claim, stated on the dynamics probe as well as retrieval.
    ab = embed_captures(model, cfg, ncfg, train_ids, dev=dev, regions=val_regions,
                        n_windows=n_windows, zero_dry=True, labels=labels)
    probes["crest_r2_from_dynamic_nodry"] = _linear_probe_reg(
        ab["segment_dynamic"][tr], crest[tr], ab["segment_dynamic"][te], crest[te])

    # 6. t-SNE over held-out captures.
    coords = _tsne(held["capture_embedding"], cfg.seed)
    if coords is not None:
        _write_tsne(run_dir, coords, held["capture_device"], device_names(cfg))

    report = {
        "run": ck.get("name", run_dir.name),
        "n_train_devices": len(train_ids), "n_held_out": len(held_ids),
        "n_windows": int(n_windows), "embedding_dim": int(ncfg.embedding_dim),
        "best_val_supcon": float(ck.get("best_val_supcon", float("nan"))),
        "retrieval": retrieval,
        "retrieval_no_dry": {k: v for k, v in retrieval_ab.items() if k != "per_device"},
        "branch_redundancy": redundancy,
        "probes": probes,
        "tsne": "tsne.csv" if coords is not None else "unavailable (sklearn missing)",
        "manifest_sha256": live,
    }
    (run_dir / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "capture_embeddings.npz",
                        device_ids=held["capture_device"],
                        embedding=held["capture_embedding"])
    _print_report(report)
    return report


def _verdict(ok: bool | None) -> str:
    return "?" if ok is None else ("PASS" if ok else "FAIL")


def _print_report(r: dict) -> None:
    ret, ab = r["retrieval"], r["retrieval_no_dry"]
    p, red = r["probes"], r["branch_redundancy"]
    chance = ret.get("chance", 0.0)
    print("\n=== tone encoder diagnostics =========================================")
    print(f"run {r['run']}  |  {r['n_held_out']} unseen captures  |  "
          f"{r['n_train_devices']} train devices  |  dim {r['embedding_dim']}")
    print(f"  k-NN retrieval (unseen)   top1 {ret.get('top1', float('nan')):.3f}  "
          f"top5 {ret.get('top5', float('nan')):.3f}  chance {chance:.3f}   "
          f"[{_verdict(ret.get('top1', 0) > 10 * chance)}]")
    print(f"  ... with dry zeroed       top1 {ab.get('top1', float('nan')):.3f}  "
          f"(retrieval may survive; the dynamics probe is the real tell)")
    print(f"  branch redundancy         mean|r| {red.get('mean_abs_corr', float('nan')):.3f}  "
          f"max {red.get('max_abs_corr', float('nan')):.3f}   "
          f"[{_verdict(red.get('mean_abs_corr', 1) < 0.3)}]")
    print(f"  amp id   <- static {p['amp_from_static']:.3f}   "
          f"(dynamic {p['amp_from_dynamic']:.3f})")
    print(f"  gain     <- dynamic {p['gain_from_dynamic']:.3f}   "
          f"(static {p['gain_from_static']:.3f})   "
          f"[{_verdict(p['gain_from_dynamic'] > p['gain_from_static'])}]")
    print(f"  crest R2 <- dynamic {p['crest_r2_from_dynamic']:.3f}   "
          f"(static {p['crest_r2_from_static']:.3f}, "
          f"no-dry {p['crest_r2_from_dynamic_nodry']:.3f})   "
          f"[{_verdict(p['crest_r2_from_dynamic'] > p['crest_r2_from_dynamic_nodry'])}]")
    print("======================================================================\n")


__all__ = ["diagnose", "embed_captures", "knn_retrieval", "branch_redundancy",
           "EVAL_WINDOWS"]
