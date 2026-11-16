"""The feedback encoder (probe_feedback): what it takes over from c1, and that an
untrained one leaves c1's guess alone.

torch-guarded like the other model tests; tiny sizes, CPU."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

import torch.nn.functional as F

from openamp.probes.probe_conformer import ConformerEncoder
from openamp.probes.probe_feedback import FeedbackEncoder, refine, step_vector
from openamp.probes.probe_synth import N_MELS

SR, DE, N = 48_000, 8, 4096
SIZES = dict(width=16, n_blocks=1, n_heads=2, ff_ratio=2, conv_kernel=3, n_queries=2,
             d_model=8, pool_heads=2, mlp_ratio=2)


def _c1():
    torch.manual_seed(0)
    c1 = ConformerEncoder(DE, SR, **SIZES).eval()
    with torch.no_grad():                       # a c1 run's own standardization
        c1.feat_mean.copy_(torch.randn(2 * N_MELS))
        c1.feat_std.copy_(torch.rand(2 * N_MELS) + 0.5)
    return c1


def _net(c1):
    net = FeedbackEncoder(DE, SR, **SIZES).eval()
    net.load_c1(c1.state_dict())
    return net


def _audio(B=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    dry = 0.3 * torch.randn(B, N, generator=g)
    wet = torch.tanh(2.0 * dry)
    err = 0.1 * torch.randn(B, N, generator=g)
    u = F.normalize(torch.randn(B, DE, generator=g), dim=-1)
    return dry, wet, err, u


def test_load_c1_copies_the_body_and_zeroes_what_is_new():
    c1 = _c1()
    net = _net(c1)
    old, new = c1.state_dict(), net.state_dict()
    for k, v in old.items():
        if k.startswith("proj.") or k in ("raw.convs.0.weight", "fuse.weight",
                                          "feat_mean", "feat_std"):
            continue
        assert torch.equal(new[k], v), k
    assert torch.equal(new["raw.convs.0.weight"][:, :2], old["raw.convs.0.weight"])
    assert not new["raw.convs.0.weight"][:, 2].any()
    n_old = old["fuse.weight"].shape[1]
    assert torch.equal(new["fuse.weight"][:, :n_old], old["fuse.weight"])
    assert not new["fuse.weight"][:, n_old:].any()
    # err's log-mel is standardized by wet's mean and std
    for k in ("feat_mean", "feat_std"):
        assert torch.equal(new[k][:2 * N_MELS], old[k])
        assert torch.equal(new[k][2 * N_MELS:], old[k][N_MELS:])
    for k in ("e_in.weight", "e_in.bias", "out.weight", "out.bias"):
        assert not new[k].any(), k


def test_an_untrained_network_is_c1_alone():
    c1 = _c1()
    net = _net(c1)
    dry, wet, err, u = _audio()
    with torch.no_grad():
        # zero correction, whatever the error and the current vector are
        assert not net(dry, wet, err, u).any()
        # and everything before the output layer is still c1's own body
        again = F.normalize(c1.proj(net.features(dry, wet, err, u)), dim=-1)
        assert torch.allclose(again, c1(dry, wet), atol=1e-6)

    def play(v):                                 # any render: the correction is zero
        return torch.zeros(len(v), N)

    us = refine(net, play, dry[0], wet, u, passes=3)
    assert len(us) == 4
    for v in us:
        assert torch.allclose(v, u, atol=1e-6)


def test_refine_feeds_each_amp_its_own_error_and_swap_feeds_its_neighbours():
    c1 = _c1()
    net = _net(c1)
    torch.manual_seed(2)
    with torch.no_grad():                        # a "trained" network: nothing left at zero
        for p in (net.e_in.weight, net.e_in.bias, net.out.weight, net.out.bias):
            p.normal_(0.0, 0.05)
        net.raw.convs[0].weight[:, 2].normal_(0.0, 0.05)
        net.fuse.weight[:, SIZES["width"] + 2 * N_MELS:].normal_(0.0, 0.05)
    dry, wet, _, u = _audio()
    g = torch.Generator().manual_seed(3)
    guesses = 0.2 * torch.randn(len(u), N, generator=g)

    def play(v):
        return guesses

    with torch.no_grad():
        own = torch.cat([net(dry[:1], wet[i:i + 1], (wet[i] - guesses[i])[None], u[i:i + 1])
                         for i in range(len(u))])
        nxt = torch.cat([net(dry[:1], wet[i:i + 1],
                             (wet[i] - guesses[(i + 1) % len(u)])[None], u[i:i + 1])
                         for i in range(len(u))])
    assert torch.allclose(refine(net, play, dry[0], wet, u, 1)[1], step_vector(u, own),
                          atol=1e-6)
    assert torch.allclose(refine(net, play, dry[0], wet, u, 1, swap=True)[1],
                          step_vector(u, nxt), atol=1e-6)
    assert not torch.allclose(own, nxt, atol=1e-4)   # the error input does reach the output
    assert torch.allclose(step_vector(u, own).norm(dim=-1), torch.ones(len(u)), atol=1e-6)
