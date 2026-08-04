"""Emulation models (FiLM-TCN + A2 FiLM-WaveNet): receptive field / causality /
conditioning, arch dispatch, training-pair construction (alignment + real
left-context warmup), losses, and the size-comparison CSV. torch-guarded like
the other model tests; the dataset cases additionally need soundfile (they
write tiny real FLACs in tmp_path); the A2 reference check needs NAM."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openamp.core.config import EmulateConfig, load_config
from openamp.emulate.models import build_model
from openamp.emulate.tcn import FiLMTCN, count_parameters
from openamp.emulate.wavenet import (A2_DILATIONS, A2_KERNEL_SIZES, DeltaWaveNet,
                                     DeltaWaveNetLayer, FiLMWaveNet, MLPFiLMWaveNet,
                                     film_output_linear)

# The two A2-topology archs differ only in the FiLM generator, so every WaveNet
# case below is parametrized over both: cond_hidden 0 = film_wavenet's single
# Linear, > 0 = mlpfilm_wavenet's per-layer MLP.
COND_HIDDEN = [0, 6]


# --- Model ---------------------------------------------------------------------
def test_receptive_field_matches_formula():
    # paper default: 1 + 2 blocks * (k-1) * sum(1,2,4,...,128) = 1 + 2*2*255 = 1021
    assert FiLMTCN.compute_receptive_field(2, 8, 3, 2) == 1021
    m = FiLMTCN.from_config(EmulateConfig(), n_devices=5)
    assert m.receptive_field == 1021


def test_forward_shape_and_from_config():
    m = FiLMTCN.from_config(EmulateConfig(channels=8, layers_per_block=4, blocks=1),
                            n_devices=6)
    x = torch.randn(3, 4000)
    y = m(x, torch.tensor([0, 1, 2]))
    assert y.shape == (3, 1, 4000)                    # [B, T] -> [B, 1, T], same length
    assert y.shape[-1] == x.shape[-1]


def test_causal_output_independent_of_future():
    torch.manual_seed(0)
    m = FiLMTCN(n_devices=2, blocks=1, layers_per_block=4, channels=8).eval()
    R = m.receptive_field
    x = torch.randn(1, 3000)
    xb = x.clone(); xb[..., 2000:] += 5.0             # perturb only the future
    ya = m(x, torch.tensor([0]))
    yb = m(xb, torch.tensor([0]))
    # samples before the perturbation (minus the receptive field) must be identical
    assert torch.allclose(ya[..., :2000 - R], yb[..., :2000 - R], atol=1e-5)


def test_conditioning_changes_output():
    torch.manual_seed(0)
    m = FiLMTCN(n_devices=4, blocks=1, layers_per_block=4, channels=8).eval()
    x = torch.randn(1, 2000)
    y0 = m(x, torch.tensor([0]))
    y1 = m(x, torch.tensor([3]))
    assert not torch.allclose(y0, y1)                 # the embedding actually steers


def test_param_count_scales_with_config():
    base = count_parameters(FiLMTCN.from_config(EmulateConfig(), n_devices=10))
    wider = count_parameters(FiLMTCN.from_config(EmulateConfig(channels=32), n_devices=10))
    bigger_emb = count_parameters(
        FiLMTCN.from_config(EmulateConfig(embedding_dim=256), n_devices=10))
    assert wider > base and bigger_emb > base


# --- FiLM-WaveNet (the NAM A2 capture topology) ----------------------------------
def _small_wavenet(cond_hidden=0, **kw):
    """A shrunken schedule (R=18) so causality tests stay fast.

    ``cond_hidden`` 0 picks :class:`FiLMWaveNet`, > 0 :class:`MLPFiLMWaveNet`.
    """
    kw.setdefault("n_devices", 4)
    cls = FiLMWaveNet if cond_hidden <= 0 else MLPFiLMWaveNet
    if cond_hidden > 0:
        kw["cond_hidden"] = cond_hidden
    return cls(channels=4, kernel_sizes=(3, 3, 3), dilations=(1, 2, 4),
               head_kernel=4, **kw)


def test_wavenet_receptive_field_matches_a2_captures():
    # 3 kernel-6 runs over (1,3,7,17,41,101,239) + two kernel-15 layers + head k16:
    # 1 + 3*5*409 + 14*(1+13) + 15 = 6347 — the R of every A2 capture in the corpus.
    assert FiLMWaveNet.compute_receptive_field() == 6347
    m = FiLMWaveNet.from_config(EmulateConfig(arch="film_wavenet"), n_devices=4)
    assert m.receptive_field == 6347
    assert FiLMWaveNet.compute_receptive_field((3, 3), (1, 2), head_kernel=4) == 10


def test_wavenet_forward_shape_and_from_config():
    m = FiLMWaveNet.from_config(EmulateConfig(arch="film_wavenet", wn_channels=4),
                                n_devices=6)
    x = torch.randn(3, 4000)
    y = m(x, torch.tensor([0, 1, 2]))
    assert y.shape == (3, 1, 4000)                    # [B, T] -> [B, 1, T], same length


def test_wavenet_causal_output_independent_of_future():
    torch.manual_seed(0)
    m = _small_wavenet(n_devices=2).eval()
    R = m.receptive_field
    x = torch.randn(1, 3000)
    xb = x.clone(); xb[..., 2000:] += 5.0             # perturb only the future
    ya = m(x, torch.tensor([0]))
    yb = m(xb, torch.tensor([0]))
    assert torch.allclose(ya[..., :2000 - R], yb[..., :2000 - R], atol=1e-6)


@pytest.mark.parametrize("cond_hidden", COND_HIDDEN)
def test_wavenet_conditioning_changes_output_and_only_through_film(cond_hidden):
    torch.manual_seed(0)
    m = _small_wavenet(cond_hidden).eval()
    x = torch.randn(1, 2000)
    # near-identity FiLM init still conditions (the sanity-#2 shuffle check needs it)
    assert not torch.allclose(m(x, torch.tensor([0])), m(x, torch.tensor([3])))
    # FiLM is the *only* path from the embedding, for either generator: zeroing the
    # final Linear makes it emit its bias verbatim, so the net is unconditioned.
    # This is exactly the precondition export's fold neutralization relies on.
    with torch.no_grad():
        for layer in m.layers:
            film_output_linear(layer.film).weight.zero_()
    assert torch.allclose(m(x, torch.tensor([0])), m(x, torch.tensor([3])))


@pytest.mark.parametrize("cls,kw", [(FiLMWaveNet, {}),
                                    (MLPFiLMWaveNet, {"cond_hidden": 32})])
def test_wavenet_core_param_count_matches_a2_export(cls, kw):
    # Everything except the two one-to-many additions (embedding table + FiLMs)
    # must count exactly what an A2 capture exports: 12,146 weights incl.
    # head_scale — which is a fixed buffer here (trained, it collapses to the
    # silence solution), so the trainable core is one less. Both archs: this is
    # the guarantee that a folded model is still a capture the plugin can run.
    m = cls(n_devices=10, **kw)
    film = sum(p.numel() for layer in m.layers for p in layer.film.parameters())
    core = count_parameters(m) - m.embedding.weight.numel() - film
    assert core == 12_145
    assert not m.head_scale.requires_grad
    assert float(m.head_scale) == pytest.approx(0.02)
    wider = cls(n_devices=10, channels=16, **kw)
    assert count_parameters(wider) > count_parameters(m)


def test_mlpfilm_generator_shape_and_rejects_degenerate_hidden():
    m = MLPFiLMWaveNet(n_devices=4, channels=8, embedding_dim=16, cond_hidden=5)
    film = m.layers[0].film
    assert (film.fc1.weight.shape, film.fc2.weight.shape) == ((5, 16), (16, 5))
    assert film_output_linear(film) is film.fc2
    # H=16 is parameter-matched to film_wavenet's Linear at the real E=256/C=8.
    per_layer = sum(p.numel() for p in
                    MLPFiLMWaveNet(n_devices=4, embedding_dim=256,
                                   cond_hidden=16).layers[0].film.parameters())
    assert per_layer == 4_384                         # vs the Linear's 4,112
    # cond_hidden 0 is film_wavenet, not a degenerate mlpfilm_wavenet.
    with pytest.raises(ValueError, match="cond_hidden"):
        MLPFiLMWaveNet(n_devices=4, cond_hidden=0)
    with pytest.raises(ValueError, match="cond_activation"):
        MLPFiLMWaveNet(n_devices=4, cond_hidden=4, cond_activation="relu")


# --- Delta WaveNet (conditioning moved into the conv weights) --------------------
def _small_delta_wavenet(rank=3, **kw):
    """The same shrunken schedule as :func:`_small_wavenet`, delta-conditioned."""
    kw.setdefault("n_devices", 4)
    return DeltaWaveNet(channels=4, kernel_sizes=(3, 3, 3), dilations=(1, 2, 4),
                        head_kernel=4, delta_rank=rank, **kw)


def test_delta_wavenet_shape_causality_and_from_config():
    torch.manual_seed(0)
    m = DeltaWaveNet.from_config(
        EmulateConfig(arch="delta_wavenet", wn_channels=4, delta_rank=3), n_devices=6)
    assert m.delta_rank == 3 and m.receptive_field == 6347   # the A2 schedule, unchanged
    assert m(torch.randn(3, 4000), torch.tensor([0, 1, 2])).shape == (3, 1, 4000)

    s = _small_delta_wavenet(n_devices=2).eval()
    R = s.receptive_field
    x = torch.randn(1, 3000)
    xb = x.clone(); xb[..., 2000:] += 5.0             # perturb only the future
    assert torch.allclose(s(x, torch.tensor([0]))[..., :2000 - R],
                          s(xb, torch.tensor([0]))[..., :2000 - R], atol=1e-6)


def test_delta_wavenet_weights_are_per_sample():
    """A training batch mixes devices, so each sample must be convolved with *its
    own* deltaed kernel — that is what the grouped conv in the layer buys. A shared
    weight (e.g. taking row 0 for the whole batch) would pass every other test in
    this file and quietly train every device toward one amp."""
    torch.manual_seed(0)
    m = _small_delta_wavenet().eval()
    x = torch.randn(3, 1200)
    batched = m(x, torch.tensor([0, 2, 3]))
    for b, row in enumerate([0, 2, 3]):
        alone = m(x[b:b + 1], torch.tensor([row]))
        assert torch.allclose(batched[b:b + 1], alone, atol=1e-6)
    # ...and the same audio through two different devices must actually differ.
    same = m(x[:1].expand(2, -1), torch.tensor([0, 3]))
    assert not torch.allclose(same[0], same[1])


def test_delta_conditioning_flows_only_through_the_generator():
    torch.manual_seed(0)
    m = _small_delta_wavenet().eval()
    x = torch.randn(1, 2000)
    assert not torch.allclose(m(x, torch.tensor([0])), m(x, torch.tensor([3])))
    # Zeroing the scale makes the delta exactly zero for any embedding, so the net
    # falls back to its shared A2 convs — export's fold neutralization relies on it.
    with torch.no_grad():
        for layer in m.layers:
            layer.delta.scale.zero_()
    assert torch.allclose(m(x, torch.tensor([0])), m(x, torch.tensor([3])))


def test_delta_generator_shape_accounting_and_rejects_degenerate_rank():
    m = DeltaWaveNet(n_devices=4, channels=8, embedding_dim=16, delta_rank=5,
                     kernel_sizes=(6,), dilations=(1,))
    gen = m.layers[0].delta
    assert gen.coeff.weight.shape == (5, 16)
    # One row per rank direction: flattened kernel (C*C*K), bias (C), mixin gain (C).
    assert gen.basis.shape == (5, 8 * 8 * 6 + 8 + 8)
    dw, db, dm = gen(m.embedding.weight[:2])
    assert dw.shape == (2, 8, 8, 6) and db.shape == dm.shape == (2, 8)

    # At the real E=256 / C=8, rank 6 is the closest parameter match to
    # film_wavenet's 23 Linears — the reason delta_a2_r6_256.yaml exists next to
    # delta_a2_256.yaml (rank 5 undershoots by 14%, rank 8 is the 1.38x default).
    def gen_params(model, attr):
        return sum(p.numel() for l in model.layers for p in getattr(l, attr).parameters())

    assert gen_params(DeltaWaveNet(n_devices=4, embedding_dim=256, delta_rank=6),
                      "delta") == 97_601        # 16,263*6 + one scale per layer
    assert gen_params(FiLMWaveNet(n_devices=4, embedding_dim=256), "film") == 94_576

    with pytest.raises(ValueError, match="delta_rank"):
        DeltaWaveNet(n_devices=4, delta_rank=0)
    with pytest.raises(ValueError, match="delta_rank"):
        build_model(EmulateConfig(arch="delta_wavenet", delta_rank=-1), 3)


def test_delta_magnitude_lives_only_in_scale():
    """Regression guard on what killed the first delta_a2_256 run.

    With magnitude spread diffusely through ``basis``, ``coeff`` and ``basis`` can
    trade it back and forth at zero cost, and the optimizer took that trade: by epoch
    12 the delta was 5.1x the base kernel, i.e. the shared filter had been shrunk to
    irrelevance and each device's conv rebuilt from its own residual — the per-device
    weight table this arch exists to avoid. Normalizing the rows makes ``basis``
    direction-only, so a layer's deviation from the shared kernel is exactly one
    number that can be watched and decayed.
    """
    torch.manual_seed(0)
    m = _small_delta_wavenet()
    gen, emb = m.layers[0].delta, m.embedding.weight
    before = gen(emb)[0].clone()

    with torch.no_grad():
        gen.basis.mul_(37.0)                          # rescaling directions...
    assert torch.allclose(gen(emb)[0], before, atol=1e-5)   # ...changes nothing

    with torch.no_grad():
        gen.scale.mul_(2.0)                           # scale is the only size knob
    assert torch.allclose(gen(emb)[0], 2.0 * before, atol=1e-5)


def test_delta_starts_as_a_small_fraction_of_the_shared_kernel():
    # The residual must start as a residual, at every kernel size in the schedule —
    # the A2 layers are kernel 6 and 15, whose fan-in inits differ. scale is solved
    # for at construction (DeltaWeightGen._calibrate), not derived, so pin the result.
    torch.manual_seed(0)
    m = DeltaWaveNet(n_devices=64, channels=8, embedding_dim=256, delta_rank=8)
    for idx in (0, 14, 22):                            # layer 14 is one of the k=15s
        layer = m.layers[idx]
        dw, _, _ = layer.delta(m.embedding.weight)
        ratio = float(dw.detach().std() / layer.conv.weight.detach().std())
        assert 0.015 < ratio < 0.06, f"layer {idx} (k={layer.kernel_size}): {ratio}"


def test_delta_family_contains_the_film_hook():
    """The claim the delta_a2 configs rest on: everything film_wavenet can do at this
    hook, delta_wavenet can do too — so a null result against nam_a2_256 is about the
    hook, not about something the arch cannot express. Build the delta that
    reproduces a given (gamma, beta) and check the two layers agree sample-for-sample.
    """
    from openamp.emulate.wavenet import FiLMWaveNetLayer

    torch.manual_seed(0)
    C, K, D, E = 4, 3, 2, 6
    film = FiLMWaveNetLayer(C, K, D, E).eval()
    delta = DeltaWaveNetLayer(C, K, D, E, rank=1).eval()
    for name in ("conv", "input_mixer", "layer1x1"):
        getattr(delta, name).load_state_dict(getattr(film, name).state_dict())

    emb = torch.randn(1, E)
    gamma, beta = film.film(emb).chunk(2, dim=-1)
    g = (gamma[0] - 1.0)
    want = torch.cat([
        (g[:, None, None] * film.conv.weight).reshape(-1),          # dW
        g * film.conv.bias + beta[0],                               # db
        g * film.input_mixer.weight.reshape(-1),                    # dm
    ])
    with torch.no_grad():
        # basis row 0 carries the direction, coeff's bias its length: forward()
        # normalizes the row, so the constant coefficient has to restore the norm.
        delta.delta.scale.fill_(1.0)
        delta.delta.basis[0] = want
        delta.delta.coeff.weight.zero_()
        delta.delta.coeff.bias.fill_(float(want.norm()))

    x = torch.randn(1, C, 200)
    cond = torch.randn(1, 1, 200)
    for a, b in zip(film(x, cond, emb), delta(x, cond, emb)):
        assert torch.allclose(a, b, atol=1e-6)


def test_delta_core_param_count_matches_a2_export():
    # Same guarantee as the FiLM archs: everything except the embedding table and
    # the generator is exactly an A2 capture's 12,146 weights (head_scale is a fixed
    # buffer here, so the trainable core is one less). This is what makes a folded
    # delta model still a capture the plugin can run.
    m = DeltaWaveNet(n_devices=10)
    gen = sum(p.numel() for layer in m.layers for p in layer.delta.parameters())
    assert count_parameters(m) - m.embedding.weight.numel() - gen == 12_145
    assert count_parameters(DeltaWaveNet(n_devices=10, delta_rank=16)) > count_parameters(m)


def test_delta_step_moves_both_the_shared_conv_and_the_embedding():
    """One optimizer step must move the shared kernel *and* the device embedding:
    the delta path is what carries gradient back to the table, and a zero-initialized
    basis would silently cut it (see init_delta_zero)."""
    torch.manual_seed(0)
    m = _small_delta_wavenet(n_devices=2)
    before = (m.layers[0].conv.weight.detach().clone(),
              m.embedding.weight.detach().clone(),
              m.layers[0].delta.coeff.weight.detach().clone())
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    m(torch.randn(2, 600), torch.tensor([0, 1])).pow(2).mean().backward()
    opt.step()
    after = (m.layers[0].conv.weight, m.embedding.weight, m.layers[0].delta.coeff.weight)
    for b, a in zip(before, after):
        assert torch.linalg.norm(b - a) > 1e-5


def test_wavenet_activation_option_swaps_the_nonlinearity():
    torch.manual_seed(0)
    leaky = _small_wavenet(n_devices=2).eval()
    tanh = _small_wavenet(n_devices=2, activation="tanh").eval()
    assert leaky.activation == "leakyrelu"
    assert all(isinstance(l.act, torch.nn.LeakyReLU) for l in leaky.layers)
    assert all(isinstance(l.act, torch.nn.Tanh) for l in tanh.layers)
    # Both activations are parameter-free, so the two nets have identical
    # state_dicts — the reason wn_activation must be resume-structural.
    tanh.load_state_dict(leaky.state_dict())
    assert count_parameters(leaky) == count_parameters(tanh)
    x = torch.randn(1, 1500) * 3.0                    # loud enough to saturate tanh
    assert not torch.allclose(leaky(x, torch.tensor([0])), tanh(x, torch.tensor([0])))


def test_wavenet_activation_from_config_and_validation():
    from openamp.emulate.models import arch_summary

    m = FiLMWaveNet.from_config(
        EmulateConfig(arch="film_wavenet", wn_channels=4, wn_activation="TanH"), 3)
    assert m.activation == "tanh"                     # case-insensitive
    assert isinstance(m.layers[-1].act, torch.nn.Tanh)
    default = FiLMWaveNet.from_config(EmulateConfig(arch="film_wavenet"), 3)
    assert default.activation == "leakyrelu"          # the capture default
    # The comparison CSV tags a non-capture activation onto the topology column.
    assert arch_summary(EmulateConfig(arch="film_wavenet"))["blocks_x_layers"] == "a2-23L"
    assert arch_summary(EmulateConfig(arch="film_wavenet", wn_activation="tanh")
                        )["blocks_x_layers"] == "a2-23L-tanh"
    # The generator is tagged the same way, so delta runs are distinguishable in
    # the existing comparison table without a new column.
    assert arch_summary(EmulateConfig(arch="delta_wavenet", delta_rank=6)
                        )["blocks_x_layers"] == "a2-23L-delta6"
    assert arch_summary(EmulateConfig(arch="delta_wavenet", delta_rank=8,
                                      wn_activation="tanh")
                        )["blocks_x_layers"] == "a2-23L-delta8-tanh"
    with pytest.raises(ValueError, match="wn_activation"):
        FiLMWaveNet(n_devices=1, activation="relu")
    with pytest.raises(ValueError, match="wn_activation"):
        build_model(EmulateConfig(arch="film_wavenet", wn_activation="silu"), 3)


def test_build_model_dispatches_on_arch():
    assert isinstance(build_model(EmulateConfig(), 5), FiLMTCN)
    # `type is` not isinstance: MLPFiLMWaveNet subclasses FiLMWaveNet, so isinstance
    # would no longer distinguish the two archs.
    assert type(build_model(EmulateConfig(arch="film_wavenet"), 5)) is FiLMWaveNet
    mlp = build_model(EmulateConfig(arch="mlpfilm_wavenet", cond_hidden=8), 5)
    assert type(mlp) is MLPFiLMWaveNet and mlp.cond_hidden == 8
    # film_wavenet must ignore cond_hidden entirely — that is what freezes every
    # existing run at the linear generator regardless of the config default.
    plain = build_model(EmulateConfig(arch="film_wavenet", cond_hidden=8), 5)
    assert isinstance(plain.layers[0].film, torch.nn.Linear)
    delta = build_model(EmulateConfig(arch="delta_wavenet", delta_rank=3), 5)
    assert type(delta) is DeltaWaveNet and delta.delta_rank == 3
    assert isinstance(delta.layers[0], DeltaWaveNetLayer)
    with pytest.raises(ValueError, match="arch"):
        build_model(EmulateConfig(arch="nope"), 5)


def test_load_model_rebuilds_arch_and_defaults_legacy_to_tcn(tmp_path):
    from dataclasses import asdict

    from openamp.emulate.evaluate import load_model

    torch.manual_seed(0)
    ecfg = EmulateConfig(arch="film_wavenet", wn_channels=4)
    m = build_model(ecfg, 3).eval()
    run = tmp_path / "wn"; run.mkdir()
    torch.save({"model": m.state_dict(), "emulate_cfg": asdict(ecfg),
                "device_ids": [1, 2, 3]}, run / "checkpoint.pt")
    loaded, ck = load_model(run)
    assert isinstance(loaded, FiLMWaveNet)
    x = torch.randn(1, 500)
    assert torch.allclose(loaded(x, torch.tensor([1])), m(x, torch.tensor([1])))

    # pre-arch checkpoint (no arch/wn_channels keys) must come back as a FiLM-TCN
    legacy = asdict(EmulateConfig(channels=8, blocks=1, layers_per_block=2))
    legacy.pop("arch"); legacy.pop("wn_channels")
    t = FiLMTCN(n_devices=2, channels=8, blocks=1, layers_per_block=2)
    run2 = tmp_path / "legacy"; run2.mkdir()
    torch.save({"model": t.state_dict(), "emulate_cfg": legacy,
                "device_ids": [7, 9]}, run2 / "checkpoint.pt")
    loaded2, _ = load_model(run2)
    assert isinstance(loaded2, FiLMTCN)

    # a pre-mlpfilm film_wavenet checkpoint (no cond_* keys) must still rebuild
    # with the linear generator, whatever EmulateConfig now defaults cond_hidden to
    pre = asdict(EmulateConfig(arch="film_wavenet", wn_channels=4))
    pre.pop("cond_hidden"); pre.pop("cond_activation")
    run3 = tmp_path / "pre_mlp"; run3.mkdir()
    torch.save({"model": m.state_dict(), "emulate_cfg": pre,
                "device_ids": [1, 2, 3]}, run3 / "checkpoint.pt")
    loaded3, _ = load_model(run3)
    assert type(loaded3) is FiLMWaveNet
    assert isinstance(loaded3.layers[0].film, torch.nn.Linear)


def test_mlpfilm_checkpoint_roundtrip_and_cross_arch_load_is_loud(tmp_path):
    from dataclasses import asdict

    from openamp.emulate.evaluate import load_model

    torch.manual_seed(0)
    ecfg = EmulateConfig(arch="mlpfilm_wavenet", wn_channels=4, embedding_dim=8,
                         cond_hidden=6)
    m = build_model(ecfg, 3).eval()
    run = tmp_path / "mlp"; run.mkdir()
    torch.save({"model": m.state_dict(), "emulate_cfg": asdict(ecfg),
                "device_ids": [1, 2, 3]}, run / "checkpoint.pt")
    loaded, _ = load_model(run)
    assert type(loaded) is MLPFiLMWaveNet and loaded.cond_hidden == 6
    x = torch.randn(1, 500)
    assert torch.allclose(loaded(x, torch.tensor([1])), m(x, torch.tensor([1])))

    # The two archs have incompatible state_dicts, so a mixed load must fail
    # loudly rather than silently reinterpret weights.
    plain = FiLMWaveNet(n_devices=3, channels=4, embedding_dim=8)
    with pytest.raises(RuntimeError, match="state_dict"):
        plain.load_state_dict(m.state_dict())


def test_delta_checkpoint_roundtrip_and_cross_arch_load_is_loud(tmp_path):
    from dataclasses import asdict

    from openamp.emulate.evaluate import load_model

    torch.manual_seed(0)
    ecfg = EmulateConfig(arch="delta_wavenet", wn_channels=4, embedding_dim=8,
                         delta_rank=3)
    m = build_model(ecfg, 3).eval()
    run = tmp_path / "delta"; run.mkdir()
    torch.save({"model": m.state_dict(), "emulate_cfg": asdict(ecfg),
                "device_ids": [1, 2, 3]}, run / "checkpoint.pt")
    loaded, _ = load_model(run)
    assert type(loaded) is DeltaWaveNet and loaded.delta_rank == 3
    x = torch.randn(1, 500)
    assert torch.allclose(loaded(x, torch.tensor([1])), m(x, torch.tensor([1])))

    # delta has no `film` and the FiLM archs have no `delta`, so either direction
    # of a cross-arch load must fail rather than reinterpret weights.
    plain = FiLMWaveNet(n_devices=3, channels=4, embedding_dim=8)
    with pytest.raises(RuntimeError, match="state_dict"):
        plain.load_state_dict(m.state_dict())
    with pytest.raises(RuntimeError, match="state_dict"):
        m.load_state_dict(plain.state_dict())


@pytest.mark.parametrize("cond_hidden", COND_HIDDEN)
@pytest.mark.parametrize("activation,nam_activation", [
    ("leakyrelu", {"name": "LeakyReLU", "negative_slope": 0.01}),
    ("tanh", "Tanh"),
])
def test_wavenet_matches_nam_a2_reference(activation, nam_activation, cond_hidden):
    """Weight-for-weight against NAM's own WaveNet: with FiLM at exact identity,
    our causal net must reproduce NAM's valid-conv output sample-for-sample.

    Both ``wn_activation`` options are checked, because both must stay a real A2
    capture after export folds FiLM away — that is what makes the plugin able to
    run them with NeuralAmpModelerCore. Both FiLM generators are checked for the
    same reason: the MLP is folded away too, so it must leave the identical net."""
    _wavenet = pytest.importorskip("nam.models.wavenet")

    torch.manual_seed(0)
    nam_net = _wavenet.WaveNet.init_from_config({
        "layers_configs": [{
            "input_size": 1, "condition_size": 1, "channels": 8,
            "kernel_sizes": list(A2_KERNEL_SIZES), "dilations": list(A2_DILATIONS),
            "activation": nam_activation,
            "head": {"out_channels": 1, "kernel_size": 16, "bias": True},
        }],
        "head": None, "head_scale": 0.02,
    }).eval()
    ours = (FiLMWaveNet(n_devices=1, activation=activation) if cond_hidden <= 0 else
            MLPFiLMWaveNet(n_devices=1, activation=activation,
                           cond_hidden=cond_hidden)).eval()

    core = nam_net._net                               # wrapper -> raw valid-conv net
    la = core._layer_arrays[0]
    with torch.no_grad():
        ours.rechannel.weight.copy_(la._rechannel.weight)
        for mine, theirs in zip(ours.layers, la._layers):
            mine.conv.weight.copy_(theirs._conv.weight)
            mine.conv.bias.copy_(theirs._conv.bias)
            mine.input_mixer.weight.copy_(theirs._input_mixer.weight)
            mine.layer1x1.weight.copy_(theirs._layer1x1.weight)
            mine.layer1x1.bias.copy_(theirs._layer1x1.bias)
            film_output_linear(mine.film).weight.zero_()   # exact identity FiLM
        ours.head_rechannel.weight.copy_(la._head_rechannel.weight)
        ours.head_rechannel.bias.copy_(la._head_rechannel.bias)

    R = ours.receptive_field
    x = torch.randn(1, 1, R + 1200) * 0.2
    with torch.no_grad():
        y_nam = core(x)                               # valid conv: [1, 1, 1201]
        y_ours = ours(x.squeeze(1), torch.tensor([0]))[..., R - 1:]
    assert y_nam.shape == y_ours.shape
    assert torch.allclose(y_ours, y_nam, atol=1e-6)


# --- Losses --------------------------------------------------------------------
def test_esr_zero_for_identical_and_large_for_random():
    from openamp.emulate.train import esr, preemph_esr

    t = torch.randn(4, 1, 2000)
    assert esr(t, t).item() == pytest.approx(0.0, abs=1e-6)
    assert preemph_esr(t, t, 0.85).item() == pytest.approx(0.0, abs=1e-6)
    assert esr(torch.randn(4, 1, 2000), t).item() > 0.5


def test_emulation_loss_orders_aligned_below_random():
    pytest.importorskip("auraloss")
    from openamp.emulate.train import EmulationLoss

    L = EmulationLoss(0.85, 1.0)
    t = torch.randn(2, 1, 8192)                       # >= STFT window
    good, _ = L(t, t)
    bad, _ = L(torch.randn(2, 1, 8192), t)
    assert good.item() < bad.item()
    assert good.item() == pytest.approx(0.0, abs=1e-5)


# --- Dataset: tiny real corpus + renders on disk -------------------------------
def _build_corpus(cfg, files, devices):
    """Write clean + (identity) render FLACs and their manifests in tmp_path.

    ``files`` is ``[(file_id, split, signal), ...]``; each render equals its clean
    source, so a correctly-aligned pair must satisfy ``target == input[R:]``.
    """
    import pandas as pd

    from openamp.core import constants as C
    from openamp.core import manifest as manifests
    from openamp.dsp import audio as audio_io

    cfg.ensure_dirs()
    corpus_rows, render_rows = [], []
    for fid, split, sig in files:
        cdir = cfg.clean_split_dir(split)
        cdir.mkdir(parents=True, exist_ok=True)
        audio_io.write_flac(cdir / f"{fid}.{cfg.output_format}", sig, cfg.sample_rate)
        corpus_rows.append({"file_id": fid, "split": split, "source": C.SOURCE_EGDB,
                            "orig_path": "", "duration_s": len(sig) / cfg.sample_rate,
                            "applied_gain_db": 0.0, "sha256": ""})
        for d in devices:
            rdir = cfg.device_render_dir(d)
            rdir.mkdir(parents=True, exist_ok=True)
            rp = rdir / f"{fid}.{cfg.output_format}"
            audio_io.write_flac(rp, sig, cfg.sample_rate)          # identity render
            render_rows.append({"device_id": d, "file_id": fid, "split": split,
                                "path": str(rp), "status": C.RENDER_OK})
    manifests.write_manifest(pd.DataFrame(corpus_rows, columns=manifests.CORPUS_COLUMNS),
                             cfg.corpus_manifest_path)
    manifests.write_manifest(
        manifests.ensure_schema(pd.DataFrame(render_rows), manifests.RENDERS_COLUMNS),
        cfg.renders_manifest_path)


def _dataset(cfg, split, *, R, clip, **kw):
    from openamp.emulate.dataset import EmulationDataset, build_device_index

    _, id_to_idx = build_device_index(cfg)
    return EmulationDataset(cfg, split, receptive_field=R, id_to_idx=id_to_idx,
                            clip_samples=clip, **kw)


def test_device_index_is_sorted_and_stable(tmp_path):
    pytest.importorskip("soundfile")
    from openamp.emulate.dataset import build_device_index, manifest_signature

    cfg = load_config(data_dir=tmp_path)
    sig = np.linspace(-0.4, 0.4, 300, dtype=np.float32)
    _build_corpus(cfg, [("f0", "train", sig)], devices=[5, 2, 9])
    ids, idx = build_device_index(cfg)
    assert ids == [2, 5, 9] and idx == {2: 0, 5: 1, 9: 2}
    # one-to-one baseline -> a single-row table
    assert build_device_index(cfg, single_device=5) == ([5], {5: 0})
    assert len(manifest_signature(cfg)) == 64          # sha-256 hex of the manifest


def test_holdout_excluded_persisted_and_bypassed(tmp_path):
    pytest.importorskip("soundfile")
    from openamp.emulate.dataset import build_device_index, load_or_create_holdout

    cfg = load_config(data_dir=tmp_path)
    sig = np.linspace(-0.4, 0.4, 300, dtype=np.float32)
    _build_corpus(cfg, [("f0", "train", sig)], devices=list(range(10)))

    held = load_or_create_holdout(cfg, 0.2, seed=1)
    assert len(held) == 2 and cfg.emulate_holdout_path.is_file()
    # the file, not the RNG, is the source of truth: other frac/seed reads it back
    assert load_or_create_holdout(cfg, 0.5, seed=99) == held
    # held-out devices leave the embedding table; rows stay dense over the rest
    ids, idx = build_device_index(cfg, exclude=held)
    assert set(ids) == set(range(10)) - set(held)
    assert sorted(idx.values()) == list(range(8))
    # frac <= 0 disables the holdout even when the file exists
    assert load_or_create_holdout(cfg, 0.0, seed=1) == []
    # single-device baseline ignores exclude (Phase 5 targets held-out devices)
    assert build_device_index(cfg, single_device=held[0], exclude=held) == \
        ([held[0]], {held[0]: 0})


def test_pairs_are_aligned_and_warmed(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    R, clip = 20, 100
    # A long, strictly-positive ramp so an unpadded draw has input[0] != 0 and
    # left-context joins the clip with a constant step (no zero-crossing ambiguity).
    sig = np.linspace(0.1, 0.9, R + clip + 400, dtype=np.float32)
    _build_corpus(cfg, [("f0", "train", sig)], devices=[0, 1])
    ds = _dataset(cfg, "train", R=R, clip=clip, pairs_per_epoch=32, seed=7)

    saw_real_context = False
    for i in range(16):
        a = ds.item_arrays(i)
        assert a["input"].shape == (R + clip,)
        assert a["target"].shape == (clip,)
        assert 0 <= a["device_idx"] < 2
        # identity render => the warmed region of the input equals the target.
        assert np.allclose(a["input"][R:], a["target"], atol=1e-4)
        if a["input"][0] > 1e-3:                        # fully unpadded (real context)
            saw_real_context = True
            # ramp is contiguous: left-context joins the clip with a constant step
            assert np.allclose(np.diff(a["input"]), a["input"][1] - a["input"][0], atol=1e-4)
    assert saw_real_context                            # real left-context was read


def test_warmup_zero_pads_at_file_start(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    R, clip = 20, 100
    # File exactly one clip long -> every draw starts at 0 -> left-context is zeros.
    sig = np.linspace(-0.5, 0.5, clip, dtype=np.float32)
    _build_corpus(cfg, [("f0", "train", sig)], devices=[0])
    ds = _dataset(cfg, "train", R=R, clip=clip, pairs_per_epoch=8, seed=1)
    a = ds.item_arrays(0)
    assert np.allclose(a["input"][:R], 0.0)            # zero-padded warmup
    assert np.allclose(a["input"][R:], a["target"], atol=1e-4)


# --- Per-device validation: one shared window grid, one ESR per amp -------------
def _grid(cfg, *, R, clip, **kw):
    from openamp.emulate.dataset import DeviceGridDataset, build_device_index

    _, id_to_idx = build_device_index(cfg)
    return DeviceGridDataset(cfg, "test", receptive_field=R, id_to_idx=id_to_idx,
                             clip_samples=clip, **kw)


def test_eval_grid_is_shared_across_devices_and_seeded(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    R, clip, n_win = 16, 64, 5
    rng = np.random.default_rng(0)
    files = [(f"f{i}", "test", (0.3 * rng.standard_normal(clip * 20)).astype(np.float32))
             for i in range(2)]
    _build_corpus(cfg, files, devices=[0, 1, 2])

    ds = _grid(cfg, R=R, clip=clip, n_windows=n_win, seed=3)
    assert len(ds) == 3 * n_win
    items = [[ds.item_arrays(d * n_win + w) for w in range(n_win)] for d in range(3)]
    # device-major layout, and every device is scored on the *same* audio
    assert [a["device_id"] for row in items for a in row] == [0] * 5 + [1] * 5 + [2] * 5
    for w in range(n_win):
        for d in (1, 2):
            assert np.array_equal(items[0][w]["input"], items[d][w]["input"])
    # distinct windows (not one position repeated), reproducible from the seed
    assert len({tuple(a["input"]) for a in items[0]}) == n_win
    assert _grid(cfg, R=R, clip=clip, n_windows=n_win, seed=3).windows == ds.windows
    assert _grid(cfg, R=R, clip=clip, n_windows=n_win, seed=4).windows != ds.windows
    # device_ids narrows the rows without touching the grid
    sub = _grid(cfg, R=R, clip=clip, n_windows=n_win, seed=3, device_ids=[0, 2])
    assert sub.device_ids == [0, 2] and len(sub) == 2 * n_win
    assert sub.windows == ds.windows


def test_eval_grid_skips_silent_windows(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    clip = 64
    loud = (0.3 * np.random.default_rng(1).standard_normal(clip * 20)).astype(np.float32)
    _build_corpus(cfg, [("silent", "test", np.zeros(clip * 20, dtype=np.float32)),
                        ("loud", "test", loud)], devices=[0])
    ds = _grid(cfg, R=8, clip=clip, n_windows=8, seed=0)
    assert ds.n_windows == 8
    assert {ds.files[f]["file_id"] for f, _ in ds.windows} == {"loud"}


def _tiny_run(cfg, tmp_path, device_ids, *, clip):
    """A saved checkpoint over ``device_ids`` (tiny net) -> its run dir."""
    from dataclasses import asdict

    ecfg = EmulateConfig(blocks=1, layers_per_block=2, channels=4, embedding_dim=8,
                         clip_seconds=clip / cfg.sample_rate, batch_size=4,
                         num_workers=0)
    torch.manual_seed(0)
    model = build_model(ecfg, len(device_ids)).eval()
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "emulate_cfg": asdict(ecfg),
                "device_ids": list(device_ids),
                "id_to_idx": {d: i for i, d in enumerate(device_ids)},
                "holdout_ids": [], "name": "run", "epoch": 1, "manifest_sha256": ""},
               run / "checkpoint.pt")
    return run, model, ecfg


def test_validate_per_device_writes_a_row_per_amp(tmp_path):
    pytest.importorskip("soundfile")
    pytest.importorskip("auraloss")
    import csv as _csv
    from dataclasses import asdict

    from openamp.emulate.evaluate import (PER_DEVICE_COLUMNS, format_per_device_summary,
                                          validate_per_device)

    cfg = load_config(data_dir=tmp_path)
    clip = 8192                                        # >= the largest STFT window
    rng = np.random.default_rng(0)
    files = [(f"f{i}", "test", (0.3 * rng.standard_normal(clip + 3000)).astype(np.float32))
             for i in range(2)]
    _build_corpus(cfg, files, devices=[0, 1, 2])
    run, model, ecfg = _tiny_run(cfg, tmp_path, [0, 1, 2], clip=clip)

    rows = validate_per_device(cfg, run, device="cpu", n_windows=6)

    assert {r["device_id"] for r in rows} == {0, 1, 2}
    assert all(r["n_windows"] == 6 for r in rows)
    assert all(np.isfinite(r["test_ESR"]) and np.isfinite(r["test_MRSTFT"]) for r in rows)
    assert [r["test_ESR"] for r in rows] == sorted(r["test_ESR"] for r in rows)
    assert all(r["name"] == f"device {r['device_id']}" for r in rows)   # no manifest
    with (run / "per_device_esr.csv").open(newline="") as fh:
        csv_rows = list(_csv.DictReader(fh))
    assert list(csv_rows[0]) == PER_DEVICE_COLUMNS
    assert [r["device_id"] for r in csv_rows] == [str(r["device_id"]) for r in rows]
    assert "devices 3" in format_per_device_summary(rows)

    # The grid is what makes rows comparable: identical embeddings + identity
    # renders must give byte-identical per-device scores, not just close ones.
    with torch.no_grad():
        model.embedding.weight.copy_(model.embedding.weight[:1].expand(3, -1))
    torch.save({"model": model.state_dict(), "emulate_cfg": asdict(ecfg),
                "device_ids": [0, 1, 2], "id_to_idx": {0: 0, 1: 1, 2: 2}, "name": "run"},
               run / "checkpoint.pt")
    tied = validate_per_device(cfg, run, device="cpu", n_windows=6,
                               out_path=tmp_path / "tied.csv")
    assert len({r["test_ESR"] for r in tied}) == 1
    assert (tmp_path / "tied.csv").is_file()


# --- Enrollment (Phase 5): frozen net, new embeddings only ----------------------
def test_enroll_end_to_end(tmp_path):
    pytest.importorskip("soundfile")
    pytest.importorskip("auraloss")
    from dataclasses import asdict

    from openamp.emulate.enroll import enroll
    from openamp.emulate.evaluate import load_model

    cfg = load_config(data_dir=tmp_path)
    clip = 8192                                       # >= the largest STFT window
    rng = np.random.default_rng(0)
    files = [(f"f{i}", split,
              (0.3 * rng.standard_normal(clip + 2000)).astype(np.float32))
             for i, split in enumerate(("train", "val", "test"))]
    _build_corpus(cfg, files, devices=[0, 1, 2, 3])   # identity renders, all splits

    torch.manual_seed(0)
    ecfg = EmulateConfig(blocks=1, layers_per_block=2, channels=4, embedding_dim=8,
                         clip_seconds=clip / cfg.sample_rate, batch_size=4,
                         num_workers=0, val_pairs=8, log_every=10**9)
    m = build_model(ecfg, 2).eval()                   # trained table = devices [0, 1]
    run = tmp_path / "run"; run.mkdir()
    torch.save({"model": m.state_dict(), "emulate_cfg": asdict(ecfg),
                "device_ids": [0, 1], "holdout_ids": [2, 3], "name": "run",
                "epoch": 5, "manifest_sha256": ""}, run / "checkpoint.pt")
    before = {k: v.clone() for k, v in m.state_dict().items()}

    metrics = enroll(cfg, run, pairs=16, epochs=6, lr=5e-2, device="cpu", seed=0,
                     test_pairs=8)

    # optimizing the embedding must actually help vs the table-mean prior
    assert metrics["n_enrolled"] == 2 and metrics["skipped"] == []
    assert metrics["best_val_esr_pooled"] < metrics["init_val_esr_pooled"]

    # the run itself is untouched: reloading gives bit-identical network weights
    reloaded, _ = load_model(run)
    for k, v in before.items():
        assert torch.equal(reloaded.state_dict()[k], v)

    # enrolled_embeddings.pt mirrors the embedding.pt schema + provenance
    ep = torch.load(run / "enroll" / "enrolled_embeddings.pt", weights_only=False)
    assert ep["embedding"].shape == (2, 8)
    assert ep["device_ids"] == [2, 3]
    assert ep["base_run"] == "run" and ep["init"] == "table_mean"
    table_mean = before["embedding.weight"].mean(dim=0)
    assert not torch.allclose(ep["embedding"], table_mean.expand(2, 8))
    assert set(ep["per_device"]) == {2, 3}

    # one CSV row per enrolled device, with finite metrics (test renders exist)
    import csv as _csv
    with (run / "enroll" / "enrollment.csv").open(newline="") as fh:
        rows = {int(r["device_id"]): r for r in _csv.DictReader(fh)}
    assert set(rows) == {2, 3}
    for r in rows.values():
        assert np.isfinite(float(r["val_esr"]))
        assert np.isfinite(float(r["test_esr"]))
        assert np.isfinite(float(r["baseline_test_esr"]))


def test_enrollment_csv_merges_by_device(tmp_path):
    import csv as _csv

    from openamp.emulate.enroll import ENROLLMENT_COLUMNS, _merge_enrollment_rows

    def row(d, esr):
        return {c: 0 for c in ENROLLMENT_COLUMNS} | {"device_id": d, "test_esr": esr}

    path = tmp_path / "enrollment.csv"
    _merge_enrollment_rows(path, [row(2, 0.5), row(3, 0.6)])
    _merge_enrollment_rows(path, [row(2, 0.4)])        # re-run replaces device 2 only
    with path.open(newline="") as fh:
        by = {r["device_id"]: r for r in _csv.DictReader(fh)}
    assert set(by) == {"2", "3"}
    assert float(by["2"]["test_esr"]) == 0.4
    assert float(by["3"]["test_esr"]) == 0.6


def test_enroll_rejects_seen_and_skips_unrenderable(tmp_path):
    pytest.importorskip("soundfile")
    from openamp.emulate.enroll import _resolve_enroll_ids

    cfg = load_config(data_dir=tmp_path)
    sig = np.linspace(-0.4, 0.4, 300, dtype=np.float32)
    _build_corpus(cfg, [("f0", "train", sig), ("f1", "val", sig)], devices=[0, 1, 2])
    ck = {"device_ids": [0, 1], "holdout_ids": []}

    with pytest.raises(RuntimeError, match="trained table"):
        _resolve_enroll_ids(cfg, ck, [0, 2])           # 0 is a seen device
    with pytest.raises(RuntimeError, match="holdout"):
        _resolve_enroll_ids(cfg, ck, None)             # no holdout, no --devices
    with pytest.raises(RuntimeError, match="No enrollable"):
        _resolve_enroll_ids(cfg, ck, [7])              # nothing rendered at all
    ids, skipped = _resolve_enroll_ids(cfg, ck, [2, 7])
    assert ids == [2] and skipped == [7]               # 7 has no renders -> skipped


def test_wet_dry_dataset_and_lag_estimation():
    from openamp.emulate.enroll import WetDryDataset, estimate_lag

    rng = np.random.default_rng(0)
    a = (0.3 * rng.standard_normal(20_000)).astype(np.float32)
    # reamp latency detection: wet trailing / leading the dry by a fixed offset
    late = np.concatenate([np.zeros(37, np.float32), a])[:len(a)]
    early = np.concatenate([a[23:], np.zeros(23, np.float32)])
    assert estimate_lag(a, late, max_lag=100) == 37
    assert estimate_lag(a, early, max_lag=100) == -23

    # identity pair on a ramp: warmed alignment + targets stay inside the region
    R, clip = 16, 400
    ramp = np.arange(20_000, dtype=np.float32)
    ds = WetDryDataset(ramp, ramp, receptive_field=R, clip_samples=clip,
                       region=(1000, 5000), pairs_per_epoch=8, seed=3)
    for i in range(8):
        it = ds.item_arrays(i)
        assert it["input"].shape == (R + clip,)
        assert it["target"].shape == (clip,)
        assert np.allclose(it["input"][R:], it["target"])
        c = int(it["target"][0])                       # ramp value == sample index
        assert 1000 <= c <= 5000 - clip
        assert it["input"][0] == c - R                 # real left-context, no padding
    with pytest.raises(ValueError, match="aligned"):
        WetDryDataset(ramp, ramp[:-1], receptive_field=R, clip_samples=clip)
    with pytest.raises(ValueError, match="region"):
        WetDryDataset(ramp, ramp, receptive_field=R, clip_samples=clip,
                      region=(0, clip - 1))


def test_blip_lag_recovers_latency_through_a_coloring_amp():
    from openamp.emulate.enroll import blip_lag, estimate_lag

    sr = 48_000
    rng = np.random.default_rng(0)
    n = sr                                             # 1 s
    dry = np.zeros(n, dtype=np.float32)
    d = 480                                            # blip at 10 ms
    dry[d:d + 8] = np.array([1, -1, 1, -1, 1, -1, 1, -1], np.float32)  # broadband blip
    dry[4000:] += 0.2 * rng.standard_normal(n - 4000).astype(np.float32)  # 'sweep'

    # 'amp' impulse response like a real one: it starts immediately but ~40 dB
    # down and *ramps* to its peak over ~10-20 samples (a real ADA MP-1 capture
    # measured exactly this), which is what makes a high detection threshold
    # report the ramp instead of the onset. Plus a gentle nonlinearity.
    k = np.arange(80)
    h = ((1 - np.exp(-k / 5.0)) * np.exp(-k / 30.0)).astype(np.float32)
    amp = np.tanh(2.0 * np.convolve(dry, h)[:n]).astype(np.float32)
    L = 137
    wet = np.concatenate([np.zeros(L, np.float32), amp])[:n]

    # blip alignment recovers the interface latency itself, tone-independent:
    # first-arrival is at d+L regardless of the amp's group delay
    assert abs(blip_lag(dry, wet, sample_rate=sr) - L) <= 5
    # a shorter search window still isolates the blip (it is the loudest early)
    assert abs(blip_lag(dry, wet, sample_rate=sr, search_seconds=0.5) - L) <= 5
    # the low default beats a high threshold, which reports the ramp (+18 on the
    # real capture) — guards the default against being raised back
    assert abs(blip_lag(dry, wet, sample_rate=sr) - L) < \
        abs(blip_lag(dry, wet, sample_rate=sr, thresh_frac=0.5) - L)
    _ = estimate_lag                                   # fallback path, tested elsewhere
    with pytest.raises(ValueError, match="silent|no signal"):
        blip_lag(np.zeros(sr, np.float32), wet, sample_rate=sr)


def test_enroll_pair_end_to_end(tmp_path):
    pytest.importorskip("auraloss")
    from dataclasses import asdict

    from openamp.emulate.enroll import enroll_pair, load_pair_model, render_dry

    cfg = load_config(data_dir=tmp_path)
    clip = 8192                                        # >= the largest STFT window
    torch.manual_seed(0)
    ecfg = EmulateConfig(blocks=1, layers_per_block=2, channels=4, embedding_dim=8,
                         clip_seconds=clip / cfg.sample_rate, batch_size=4,
                         num_workers=0, val_pairs=8, log_every=10**9)
    m = build_model(ecfg, 2).eval()
    run = tmp_path / "run"; run.mkdir()
    torch.save({"model": m.state_dict(), "emulate_cfg": asdict(ecfg),
                "device_ids": [0, 1], "holdout_ids": [], "name": "run",
                "epoch": 5, "manifest_sha256": ""}, run / "checkpoint.pt")

    rng = np.random.default_rng(0)
    dry = (0.3 * rng.standard_normal(4 * clip)).astype(np.float32)
    wet = dry.copy()                                   # identity "device"

    metrics = enroll_pair(cfg, run, dry, wet, name="mydevice", pairs=16, epochs=6,
                          lr=5e-2, device="cpu", seed=0, val_frac=0.25, val_pairs=8,
                          sources={"dry": "synthetic", "wet": "synthetic"})

    assert metrics["best_val_esr_pooled"] < metrics["init_val_esr_pooled"]
    assert np.isfinite(metrics["val_esr_render_raw"])

    out = run / "enroll" / "pairs" / "mydevice"
    assert (out / "metrics.json").is_file() and (out / "enroll_log.csv").is_file()
    blob = torch.load(out / "enrolled_pair.pt", weights_only=False)
    assert blob["embedding"].shape == (1, 8)
    assert blob["name"] == "mydevice" and blob["base_run"] == "run"
    assert blob["sources"] == {"dry": "synthetic", "wet": "synthetic"}

    # reload: the enrolled vector is installed and the model renders end-to-end
    m2, blob2 = load_pair_model(run, "mydevice", device="cpu")
    assert torch.equal(m2.embedding.weight.detach().cpu(), blob["embedding"])
    pred = render_dry(m2, dry[:12_000], device="cpu", chunk_samples=5000)
    assert pred.shape == (12_000,)
    # chunked streaming must equal one full-length forward
    with torch.no_grad():
        full = m2(torch.from_numpy(dry[:12_000])[None],
                  torch.tensor([0]))[..., m2.receptive_field:]
    assert np.allclose(pred[m2.receptive_field:], full.squeeze().numpy(), atol=1e-5)


# --- Comparison CSV ------------------------------------------------------------
def test_enroll_pair_batch_size_overrides_the_runs_value(tmp_path):
    """The fit is fp32, so a run's own batch can exceed the card it trained on."""
    pytest.importorskip("auraloss")
    from dataclasses import asdict

    from openamp.emulate.enroll import enroll_pair

    cfg = load_config(data_dir=tmp_path)
    clip = 8192
    torch.manual_seed(0)
    ecfg = EmulateConfig(blocks=1, layers_per_block=2, channels=4, embedding_dim=8,
                         clip_seconds=clip / cfg.sample_rate, batch_size=4,
                         num_workers=0, val_pairs=8, log_every=10**9)
    m = build_model(ecfg, 2).eval()
    run = tmp_path / "run"; run.mkdir()
    torch.save({"model": m.state_dict(), "emulate_cfg": asdict(ecfg),
                "device_ids": [0, 1], "holdout_ids": [], "name": "run",
                "epoch": 5, "manifest_sha256": ""}, run / "checkpoint.pt")

    rng = np.random.default_rng(0)
    dry = (0.3 * rng.standard_normal(4 * clip)).astype(np.float32)
    kw = dict(name="d", pairs=4, epochs=1, lr=5e-2, device="cpu", seed=0,
              val_frac=0.25, val_pairs=4)

    assert enroll_pair(cfg, run, dry, dry.copy(), **kw)["batch_size"] == 4  # run's
    assert enroll_pair(cfg, run, dry, dry.copy(), batch_size=2, **kw)["batch_size"] == 2
    with pytest.raises(ValueError, match="batch_size"):
        enroll_pair(cfg, run, dry, dry.copy(), batch_size=0, **kw)


def test_append_rows_dedups_by_run_name(tmp_path):
    import csv

    from openamp.emulate.evaluate import COMPARISON_COLUMNS, _append_rows

    def row(name, esr):
        return {c: 0 for c in COMPARISON_COLUMNS} | {"run_name": name, "test_ESR_mean": esr}

    path = tmp_path / "comparison.csv"
    _append_rows(path, [row("paper", 0.10), row("embed16", 0.12)])
    _append_rows(path, [row("paper", 0.08)])           # re-run replaces the paper row
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    by = {r["run_name"]: r for r in rows}
    assert set(by) == {"paper", "embed16"}
    assert float(by["paper"]["test_ESR_mean"]) == 0.08