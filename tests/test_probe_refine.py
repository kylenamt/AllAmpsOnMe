"""The refinement network (probe_refine): its size, its directions, the per-second
pieces, and the test step (mix solve, keep rule) on a toy linear "generator".

torch-guarded like the other model tests; tiny sizes, CPU."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

import torch.nn.functional as F

from openamp.losses import pre_emphasis
from openamp.models import count_parameters
from openamp.probes.probe_refine import (RefineNet, keep_best, piece_index, refine,
                                         solve_mix, sweep_esr, tangent_unit)
from openamp.probes.probe_synth import HOP, N_MELS

SR, DE = 48_000, 8
SIZES = dict(width=16, n_blocks=1, n_heads=2, ff_ratio=2, conv_kernel=3, d_model=16,
             n_layers=1, reader_heads=2)
COEFF = 0.85


def _net(seconds=8, n_cand=4):
    n = seconds * SR
    torch.manual_seed(0)
    net = RefineNet(DE, SR, n // HOP + 1, seconds, **SIZES, n_cand=n_cand).eval()
    net.set_stats(torch.randn(n // HOP + 1, N_MELS), torch.randn(n // HOP + 1, N_MELS),
                  torch.rand(n // HOP + 1, N_MELS) * 5)
    return net


def test_page_sizes_give_the_page_parameter_count():
    n_keep = 8_107_200
    net = RefineNet(512, SR, n_keep // HOP + 1, 169, width=128, n_blocks=2, n_heads=4,
                    ff_ratio=4, conv_kernel=15, d_model=256, n_layers=4, reader_heads=8)
    assert count_parameters(net) == 4_794_880


def test_pieces_are_whole_seconds_and_the_last_frames_join_the_last_piece():
    n = 8 * SR
    idx, P = piece_index(n // HOP + 1, n, SR)
    assert P == 8
    assert idx[0] == 0 and idx[-1] == 7
    assert int((idx == 0).sum()) in (93, 94)          # 48000 / 512 = 93.75 frames a second
    idx_full, P_full = piece_index(8_107_200 // HOP + 1, 8_107_200, SR)
    assert P_full == 169 and idx_full[-1] == 168


def test_directions_are_unit_and_tangent_and_depend_on_the_crop_start():
    net = _net()
    g = torch.Generator().manual_seed(1)
    rec, guess = torch.randn(2, 4 * SR, generator=g), torch.randn(2, 4 * SR, generator=g)
    u = F.normalize(torch.randn(2, DE, generator=g), dim=-1)
    le = torch.tensor([-1.0, -2.0])
    with torch.no_grad():
        d0 = net(rec, guess, u, le, 0)
        d4 = net(rec, guess, u, le, 4)
    assert d0.shape == (2, 4, DE)
    assert torch.allclose(d0.norm(dim=-1), torch.ones(2, 4), atol=1e-5)
    assert (d0 * u[:, None]).sum(-1).abs().max() < 1e-5
    assert not torch.allclose(d0, d4)                  # the stored stats and tags move with it
    with pytest.raises(ValueError):
        net(rec, guess, u, le, 2)                      # off the 4 s grid


def test_tangent_unit_removes_the_part_along_u():
    u = F.normalize(torch.randn(3, DE), dim=-1)
    d = tangent_unit(torch.randn(3, 5, DE) + 3 * u[:, None], u)
    assert (d * u[:, None]).sum(-1).abs().max() < 1e-5


class Toy:
    """A linear "generator" on the sphere's tangent: y = M @ normalize(units)."""

    def __init__(self, n=4000, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.M = torch.randn(n, DE, generator=g, dtype=torch.float64)

    def __call__(self, units):
        return (F.normalize(units.double(), dim=-1) @ self.M.T).float()


def test_solve_mix_recovers_the_move_when_the_target_lies_in_the_span():
    play = Toy()
    torch.manual_seed(2)
    u = F.normalize(torch.randn(2, DE), dim=-1)
    d = tangent_unit(torch.randn(2, 3, DE), u)
    c_true = torch.tensor([[0.02, -0.01, 0.015], [0.0, 0.03, -0.02]])
    target = F.normalize(u + (c_true[:, :, None] * d).sum(1), dim=-1)
    rec_p = pre_emphasis(play(target), COEFF)
    y0_p = pre_emphasis(play(u), COEFF)
    c = solve_mix(play, rec_p, y0_p, u, d, COEFF, chunk=6)
    assert torch.allclose(c, c_true, atol=2e-3)


def test_keep_best_never_raises_the_error_and_reports_when_it_stopped():
    play = Toy()
    torch.manual_seed(3)
    u = F.normalize(torch.randn(2, DE), dim=-1)
    target = F.normalize(u + 0.05 * torch.randn(2, DE), dim=-1)
    rec_p = pre_emphasis(play(target), COEFF)
    e0 = sweep_esr(rec_p, pre_emphasis(play(u), COEFF))
    good = target - u
    bad = -good
    u1, e1, y1, ok = keep_best(play, rec_p, u, torch.stack([good[0], bad[1]]), e0, COEFF)
    assert bool(ok[0]) and not bool(ok[1])
    assert e1[0] < e0[0] and e1[1] == e0[1]
    assert torch.equal(u1[1], u[1]) and y1[1] is None


def test_refine_with_the_network_never_raises_the_sweep_error():
    net = _net(seconds=4, n_cand=4)
    play = Toy(n=4 * SR)
    torch.manual_seed(4)
    target = F.normalize(torch.randn(2, DE), dim=-1)
    rec = play(target)
    u0 = F.normalize(target + 0.3 * torch.randn(2, DE), dim=-1)
    us, es = refine(net, play, rec, u0, 3, COEFF, chunk=8)
    assert us.shape == (4, 2, DE) and es.shape == (4, 2)
    assert (es[1:] <= es[:-1] + 1e-12).all()
    assert (es[-1] < es[0]).all()                      # 4 random tangent directions still help
