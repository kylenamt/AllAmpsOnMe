"""Embedding-geometry helpers: k-NN retrieval and a t-SNE writer.

Lifted verbatim from the archived ``openamp.encoder.evaluate`` when that package
was retired, because :mod:`openamp.joint_model.evaluate` is the only remaining
consumer. Unchanged so the numbers stay comparable with runs made before the move.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TSNE_PERPLEXITY = 15.0

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
