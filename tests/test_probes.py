"""Probe primitives: leak-free split, KNN/MLP on separable data, confusion."""

import numpy as np
import pytest

from eval.probes import confusion_matrix, grouped_split


def _separable(n_classes=4, per=40, d=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(n_classes, d))
    X, y, g = [], [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.normal(scale=0.3, size=(per, d)))
        y += [c] * per
        g += [f"src{c}_{i}" for i in range(per)]          # unique groups
    return np.vstack(X).astype(np.float32), np.asarray(y), np.asarray(g, dtype=object)


def test_grouped_split_is_leak_free_and_covers_classes():
    _, y, g = _separable()
    tr, te = grouped_split(y, g, test_size=0.25, seed=1)
    assert set(g[tr]).isdisjoint(set(g[te]))               # no group on both sides
    assert set(y[tr]) == set(y[te]) == set(y)              # every class both sides
    assert 0 < len(te) < len(y)


def test_grouped_split_falls_back_when_too_few_groups():
    y = np.array([0, 0, 0, 1, 1, 1])
    g = np.array(["a", "a", "a", "a", "a", "a"], dtype=object)   # one group
    tr, te = grouped_split(y, g, test_size=0.34, seed=0)
    assert len(tr) + len(te) == 6 and len(set(tr) & set(te)) == 0


def test_confusion_matrix_counts():
    m = confusion_matrix([0, 1, 2, 2], [0, 1, 2, 1], n_classes=3)
    assert m[0, 0] == 1 and m[1, 1] == 1 and m[2, 2] == 1 and m[2, 1] == 1
    assert m.sum() == 4


def test_knn_probe_high_on_separable():
    pytest.importorskip("sklearn")
    from eval.probes import knn_probe
    X, y, g = _separable()
    tr, te = grouped_split(y, g, seed=2)
    acc, pred = knn_probe(X[tr], y[tr], X[te], y[te], k=5)
    assert acc > 0.9 and pred.shape == y[te].shape


def test_mlp_probe_high_on_separable():
    pytest.importorskip("torch")
    from eval.probes import mlp_probe
    X, y, g = _separable()
    tr, te = grouped_split(y, g, seed=3)
    acc, _ = mlp_probe(X[tr], y[tr], X[te], y[te], hidden=32, epochs=200, seed=0)
    assert acc > 0.9
