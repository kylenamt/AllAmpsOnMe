"""NT-Xent loss + in-batch retrieval: correctness on constructed embeddings."""

import math

import pytest

torch = pytest.importorskip("torch")

from train.objectives import nt_xent_loss, retrieval_top1


def test_retrieval_perfect_when_views_match():
    # Each pair's two views are identical and pairs are far apart -> retrieval 1.0.
    base = torch.eye(8) * 10.0                  # 8 near-orthogonal anchors
    assert retrieval_top1(base, base.clone()) == 1.0


def test_retrieval_chance_for_random():
    torch.manual_seed(0)
    n = 64
    z1, z2 = torch.randn(n, 32), torch.randn(n, 32)
    acc = retrieval_top1(z1, z2)
    # Chance = 1/(2n-1); random embeddings stay well below a loose ceiling.
    assert acc < 0.2


def test_nt_xent_lower_for_aligned_than_random():
    torch.manual_seed(0)
    n, d = 32, 16
    z = torch.randn(n, d)
    aligned = nt_xent_loss(z, z.clone(), temperature=0.1).item()
    z2 = torch.randn(n, d)
    misaligned = nt_xent_loss(z, z2, temperature=0.1).item()
    assert aligned < misaligned


def test_nt_xent_is_finite_and_positive():
    torch.manual_seed(1)
    loss = nt_xent_loss(torch.randn(16, 8), torch.randn(16, 8), 0.1)
    assert torch.isfinite(loss) and loss.item() > 0


def test_nt_xent_symmetric_in_views():
    torch.manual_seed(2)
    z1, z2 = torch.randn(16, 8), torch.randn(16, 8)
    a = nt_xent_loss(z1, z2, 0.1).item()
    b = nt_xent_loss(z2, z1, 0.1).item()
    assert math.isclose(a, b, rel_tol=1e-5)
