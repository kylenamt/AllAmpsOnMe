"""Frozen-encoder probes on external effect datasets (Phase 3 spec §4.2).

Embed every clip with the frozen encoder (mean over 2 s crops), then measure how
linearly separable effects are with two simple probes:

- **KNN** (k=5, cosine) and **MLP** (single hidden layer) on the embeddings.
- Controls: the same probes on a **randomly initialized** encoder, and on
  **MFCC mean/std** spectral features (context baseline).

Splits are leak-free where a dataset defines groups (:mod:`eval.adapters`): a
grouped 80/20 split keeps a clean source (note/phrase/DI file) on one side.
``evaluate_dataset`` returns a dict consumed by :mod:`eval.report`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


# --- Embedding extraction ------------------------------------------------------
def _load_crops(path: str, sample_rate: int, crop_samples: int, max_crops: int):
    from .embed import _to_mono_48k, crops_from_signal
    sig = _to_mono_48k(path, sample_rate)
    return crops_from_signal(sig, crop_samples, max_crops)


def embed_dataset(encoder, dataset, *, device: str = "cpu", crop_samples: int = 96_000,
                  max_crops: int = 8, clip_batch: int = 64, workers: int = 8,
                  progress: bool = True) -> np.ndarray:
    """Return ``[N, embed_dim]`` mean embeddings, one row per clip (clip order)."""
    import torch

    encoder.eval()
    paths = [c.path for c in dataset.clips]
    sr = dataset.sample_rate
    out = np.empty((len(paths), encoder.embed_dim), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in range(0, len(paths), clip_batch):
            chunk = paths[start:start + clip_batch]
            crop_lists = list(ex.map(
                lambda p: _load_crops(p, sr, crop_samples, max_crops), chunk))
            counts = [len(c) for c in crop_lists]
            allcrops = np.concatenate(crop_lists, axis=0)
            with torch.no_grad():
                emb = encoder(torch.from_numpy(allcrops).to(device)).cpu().numpy()
            pos = 0
            for j, n in enumerate(counts):
                out[start + j] = emb[pos:pos + n].mean(axis=0)
                pos += n
            if progress and (start // clip_batch) % 10 == 0:
                print(f"    embed {min(start + clip_batch, len(paths))}/{len(paths)}",
                      flush=True)
    return out


def mfcc_features(dataset, *, n_mfcc: int = 20, crop_samples: int = 96_000,
                  max_crops: int = 8, workers: int = 8) -> np.ndarray:
    """MFCC mean+std per clip -> ``[N, 2*n_mfcc]`` (spectral baseline, spec §4.2)."""
    import librosa

    from .embed import _to_mono_48k, crops_from_signal

    sr = dataset.sample_rate

    def feat(path: str) -> np.ndarray:
        sig = _to_mono_48k(path, sr)
        crops = crops_from_signal(sig, crop_samples, max_crops)
        m = librosa.feature.mfcc(y=crops.mean(axis=0), sr=sr, n_mfcc=n_mfcc)
        return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(feat, [c.path for c in dataset.clips]))
    return np.stack(rows)


# --- Splits --------------------------------------------------------------------
def grouped_split(labels: np.ndarray, groups: np.ndarray, *, test_size: float = 0.2,
                  seed: int = 1234) -> tuple[np.ndarray, np.ndarray]:
    """80/20 split keeping each group wholly on one side; falls back to stratified.

    Retries a few seeds so both sides carry every class; if grouping makes that
    impossible (too few groups), falls back to a stratified per-class split.
    """
    from sklearn.model_selection import (GroupShuffleSplit,
                                          StratifiedShuffleSplit)

    n_classes = len(np.unique(labels))
    if len(np.unique(groups)) >= max(5, n_classes + 1):
        for s in range(seed, seed + 10):
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
            tr, te = next(gss.split(labels, labels, groups))
            if len(np.unique(labels[tr])) == n_classes and len(np.unique(labels[te])) == n_classes:
                return tr, te
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(sss.split(labels, labels))
    return tr, te


# --- Probes --------------------------------------------------------------------
def knn_probe(x_tr, y_tr, x_te, y_te, k: int = 5) -> tuple[float, np.ndarray]:
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=min(k, len(x_tr)), metric="cosine")
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    return float((pred == y_te).mean()), pred


def mlp_probe(x_tr, y_tr, x_te, y_te, *, hidden: int = 128, epochs: int = 100,
              lr: float = 1e-3, seed: int = 1234, device: str = "cpu"
              ) -> tuple[float, np.ndarray]:
    """Single-hidden-layer MLP on standardized embeddings; returns (acc, preds)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    xtr = torch.tensor((x_tr - mu) / sd, dtype=torch.float32, device=device)
    xte = torch.tensor((x_te - mu) / sd, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    n_classes = int(max(y_tr.max(), y_te.max())) + 1

    net = nn.Sequential(nn.Linear(x_tr.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, n_classes)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(net(xtr), ytr).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(xte).argmax(1).cpu().numpy()
    return float((pred == y_te).mean()), pred


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m


# --- Per-dataset evaluation ----------------------------------------------------
def evaluate_dataset(dataset, trained_encoder, random_encoder, cfg, *,
                     device: str = "cpu", include_mfcc: bool = True) -> dict:
    """Run KNN+MLP probes for trained/random encoders (+MFCC) on one dataset."""
    ecfg = cfg.eval
    crop_n = int(round(ecfg.crop_seconds * dataset.sample_rate))
    y = dataset.labels
    groups = dataset.groups
    tr, te = grouped_split(y, groups, test_size=ecfg.test_size, seed=cfg.seed)
    n_classes = len(dataset.class_names)
    chance = float(np.bincount(y[te], minlength=n_classes).max() / len(te))

    print(f"  [{dataset.name}] N={len(y)} classes={n_classes} "
          f"train/test={len(tr)}/{len(te)} chance={chance:.3f}")

    def probes_for(x):
        knn_acc, _ = knn_probe(x[tr], y[tr], x[te], y[te], k=ecfg.knn_k)
        mlp_acc, pred = mlp_probe(x[tr], y[tr], x[te], y[te], hidden=ecfg.mlp_hidden,
                                  epochs=ecfg.mlp_epochs, lr=ecfg.mlp_lr,
                                  seed=cfg.seed, device=device)
        return knn_acc, mlp_acc, pred

    print("    embedding (trained encoder)...", flush=True)
    x_tr_emb = embed_dataset(trained_encoder, dataset, device=device,
                             crop_samples=crop_n, max_crops=ecfg.max_crops_per_clip)
    knn_t, mlp_t, pred_t = probes_for(x_tr_emb)

    print("    embedding (random encoder)...", flush=True)
    x_rand = embed_dataset(random_encoder, dataset, device=device,
                           crop_samples=crop_n, max_crops=ecfg.max_crops_per_clip)
    knn_r, mlp_r, _ = probes_for(x_rand)

    result = {
        "dataset": dataset.name, "note": dataset.note,
        "n_clips": int(len(y)), "n_classes": n_classes,
        "class_names": list(dataset.class_names),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "chance": chance,
        "trained": {"knn": knn_t, "mlp": mlp_t},
        "random": {"knn": knn_r, "mlp": mlp_r},
        "confusion": confusion_matrix(y[te], pred_t, n_classes).tolist(),
    }
    if include_mfcc:
        print("    MFCC baseline...", flush=True)
        x_mfcc = mfcc_features(dataset, crop_samples=crop_n,
                               max_crops=ecfg.max_crops_per_clip)
        knn_m, mlp_m, _ = probes_for(x_mfcc)
        result["mfcc"] = {"knn": knn_m, "mlp": mlp_m}
    return result


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    import torch

    from models.encoder import build_encoder
    from train.config import ModelConfig, load_config

    from . import adapters
    from .embed import load_encoder

    ap = argparse.ArgumentParser(description="Phase 3 frozen-encoder probe eval")
    ap.add_argument("dataset", choices=["egfxset", "gfx", "egdb"], help="which adapter")
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-mfcc", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    trained, meta = load_encoder(args.ckpt, args.device)
    # Random-init control: identical architecture, untrained weights (spec §4.2).
    random_enc = build_encoder(ModelConfig(**meta_model_cfg(args.ckpt))).to(args.device).eval()

    build = {"egfxset": adapters.egfxset_adapter, "gfx": adapters.gfx_adapter,
             "egdb": adapters.egdb_amp_adapter}[args.dataset]
    ds = build(args.root) if args.root else build()
    res = evaluate_dataset(ds, trained, random_enc, cfg, device=args.device,
                           include_mfcc=not args.no_mfcc)
    print(json.dumps(res, indent=2))


def meta_model_cfg(ckpt_path) -> dict:
    """Read the stored model-config dict from a checkpoint (arch for the control)."""
    import torch

    from .embed import default_ckpt
    ck = torch.load(ckpt_path or default_ckpt(), map_location="cpu", weights_only=False)
    return ck["model_cfg"]


if __name__ == "__main__":
    main()
