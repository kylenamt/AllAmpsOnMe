"""Morph-plugin export: weight-folding equivalence, bundle round-trip, profile
schema. The fold test is the correctness anchor for the C++ plugin — if folding ==
conditioning here, the plugin only has to reproduce a stock A2 forward.
torch-guarded like the other model tests; no checkpoint or corpus needed
(everything runs on a small random WaveNet)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openamp.emulate.export import (_blob, _unblob, folded_model,
                                    forward_with_embedding)
from openamp.emulate.wavenet import (DeltaWaveNet, FiLMWaveNet, MLPFiLMWaveNet,
                                     TableDeltaWaveNet)

# Small-but-real schedule: mixed kernels like A2, a few hundred samples of RF.
KW = dict(n_devices=4, channels=4, embedding_dim=8,
          kernel_sizes=(3, 3, 5, 3), dilations=(1, 2, 4, 8), head_kernel=4)

# Folding must be exact for every conditioning hook: 0 = film_wavenet's single
# Linear, > 0 = mlpfilm_wavenet's per-layer MLP (the fold math is identical for
# those two — conditioning is per-channel affine either way), "delta" =
# delta_wavenet's low-rank weight residual and "table" = tabledelta_wavenet's free
# per-device one, which both fold by addition instead. Every case is parametrized
# over all four, since all four must leave a stock A2 capture.
COND_HIDDEN = [0, 6, "delta", "table"]


def _model(seed=0, cond_hidden=0, **kw):
    torch.manual_seed(seed)
    if cond_hidden == "table":
        # embedding_dim is derived from the schedule for this arch, not passed.
        m = TableDeltaWaveNet(**{k: v for k, v in KW.items() if k != "embedding_dim"},
                              **kw).eval()
        # Zero is the training init; a zero delta would make the fold test vacuous.
        torch.nn.init.normal_(m.embedding.weight, std=0.2)
        return m
    if cond_hidden == "delta":
        m = DeltaWaveNet(**KW, delta_rank=3, **kw).eval()
        # Un-neutralize the near-zero training init: a zero delta would make the
        # fold test vacuous, and a trained net's is arbitrary.
        for l in m.layers:
            for p in l.delta.parameters():
                torch.nn.init.normal_(p, std=0.3)
        return m
    cls = FiLMWaveNet if cond_hidden <= 0 else MLPFiLMWaveNet
    if cond_hidden > 0:
        kw["cond_hidden"] = cond_hidden
    m = cls(**KW, **kw).eval()
    # Un-neutralize FiLM: trained nets have arbitrary gamma/beta, and identity
    # FiLM would make the fold test vacuous.
    for l in m.layers:
        for p in l.film.parameters():
            torch.nn.init.normal_(p, std=0.3)
    return m


@pytest.mark.parametrize("cond_hidden", COND_HIDDEN)
@torch.no_grad()
def test_fold_matches_film_on_table_rows(cond_hidden):
    m = _model(cond_hidden=cond_hidden)
    x = torch.randn(2, 1500)
    for row in range(KW["n_devices"]):
        idx = torch.tensor([row, row])
        want = m(x, idx)
        got = folded_model(m, m.embedding.weight[row])(x, torch.tensor([0, 0]))
        assert torch.allclose(want, got, atol=1e-5), \
            f"row {row}: max err {(want - got).abs().max().item():.2e}"


@pytest.mark.parametrize("cond_hidden", COND_HIDDEN)
@torch.no_grad()
def test_fold_matches_film_on_arbitrary_morph_points(cond_hidden):
    # Morph points are convex (and extrapolated) mixes — never seen as rows.
    # With an MLP generator these are no longer linear in (gamma, beta), so the
    # fold has to be re-evaluated per point; it must still be exact.
    m = _model(1, cond_hidden=cond_hidden)
    x = torch.randn(1, 1200)
    e0, e1 = m.embedding.weight[0], m.embedding.weight[1]
    for alpha in (-0.5, 0.25, 0.5, 1.5):
        e = (1 - alpha) * e0 + alpha * e1
        want = forward_with_embedding(m, x, e)
        got = folded_model(m, e)(x, torch.tensor([0]))
        assert torch.allclose(want, got, atol=1e-5)


@pytest.mark.parametrize("cond_hidden", COND_HIDDEN)
@torch.no_grad()
def test_forward_with_embedding_matches_forward_on_rows(cond_hidden):
    # The unfolded reference itself must equal the model's own forward.
    m = _model(2, cond_hidden=cond_hidden)
    x = torch.randn(2, 800)
    idx = torch.tensor([1, 3])
    want = m(x, idx)
    for b in range(2):
        got = forward_with_embedding(m, x[b:b + 1], m.embedding.weight[idx[b]])
        # batch-1 vs batch-2 convs may take different GEMM paths: tight, not exact
        assert torch.allclose(want[b:b + 1], got, atol=1e-7)


@pytest.mark.parametrize("cond_hidden", COND_HIDDEN)
@torch.no_grad()
def test_fold_matches_film_under_tanh_activation(cond_hidden):
    """Folding rewrites conv/mixin weights *upstream* of the nonlinearity, so a
    ``wn_activation: tanh`` run must fold exactly like the LeakyReLU one — and the
    bundle must name it in NAM's spelling so the plugin picks the right one."""
    from openamp.emulate.wavenet import NAM_ACTIVATION_NAMES

    m = _model(5, cond_hidden=cond_hidden, activation="tanh")
    assert NAM_ACTIVATION_NAMES[m.activation] == "Tanh"   # what export_bundle writes
    x = torch.randn(1, 1200) * 2.0                        # into tanh saturation
    for row in range(KW["n_devices"]):
        e = m.embedding.weight[row]
        want = forward_with_embedding(m, x, e)
        got = folded_model(m, e)(x, torch.tensor([0]))
        assert torch.allclose(want, got, atol=1e-5)


@pytest.mark.parametrize("cond_hidden", COND_HIDDEN)
@torch.no_grad()
def test_folded_model_ignores_device_idx(cond_hidden):
    m = _model(3, cond_hidden=cond_hidden)
    x = torch.randn(1, 600)
    f = folded_model(m, m.embedding.weight[2])
    assert torch.equal(f(x, torch.tensor([0])), f(x, torch.tensor([3])))


@pytest.mark.parametrize("parts", [(True, False, False), (False, True, True),
                                   (False, False, False)])
@torch.no_grad()
def test_fold_honours_the_tabledelta_part_mask(parts):
    """A masked capture must play what the masked model plays. If the fold ignored
    the mask, a listening test done with kernel-only conditioning would export a
    capture with the full residual baked in — silently a different amp."""
    m = _model(7, cond_hidden="table")
    m.set_delta_parts(**dict(zip(("kernel", "bias", "mixin"), parts)))
    x = torch.randn(1, 1200)
    for row in range(KW["n_devices"]):
        e = m.embedding.weight[row]
        want = forward_with_embedding(m, x, e)
        got = folded_model(m, e)(x, torch.tensor([0]))
        assert torch.allclose(want, got, atol=1e-5), \
            f"parts={parts} row={row}: {(want - got).abs().max().item():.2e}"


def test_bundle_tabledelta_ships_no_generator(tmp_path):
    """A profile row *is* the delta for this arch, so the bundle carries only the
    slice widths — and must omit every generator key, so a reader that predates it
    dies on a missing key rather than folding garbage."""
    import json

    from openamp.core.config import EmulateConfig
    from openamp.emulate import export as ex
    from openamp.emulate.models import build_model

    ecfg = EmulateConfig(arch="tabledelta_wavenet", wn_channels=4)
    mm = build_model(ecfg, 2).eval()
    run = tmp_path / "table"
    run.mkdir()
    torch.save({"model": mm.state_dict(), "emulate_cfg": ecfg.__dict__.copy(),
                "device_ids": [1, 2], "id_to_idx": {1: 0, 2: 1},
                "sample_rate": 48000, "name": "table"}, run / "checkpoint.pt")
    ex.export_bundle(run, run / "b.json")
    b = json.loads((run / "b.json").read_text(encoding="utf-8"))

    assert b["aaom_bundle"] == 1 and b["arch"]["type"] == "tabledelta_wavenet_a2"
    assert b["arch"]["delta_parts"] == ["kernel", "bias", "mixin"]
    assert "delta_rank" not in b["arch"] and "cond_hidden" not in b["arch"]
    l0 = b["weights"]["layers"][0]
    assert not {"film_w", "film_b", "film_layers", "delta_coeff_w", "delta_basis"} & set(l0)
    C, K = mm.channels, mm.kernel_sizes[0]
    assert l0["delta_split"] == [C * C * K, C, C]
    # The split widths must actually tile a profile row, or the plugin can't slice it.
    assert sum(sum(lay["delta_split"]) for lay in b["weights"]["layers"]) == \
        len(b["profiles"][0]["embedding"]) == mm.embedding_dim


def test_blob_roundtrip_exact():
    t = torch.randn(3, 4, 5)
    a = _unblob(_blob(t))
    assert a.dtype == np.float32 and a.shape == (3, 4, 5)
    assert np.array_equal(a, t.numpy().astype(np.float32))


def test_export_bundle_and_profile_from_fake_run(tmp_path):
    # Fake a run dir: a checkpoint with exactly the keys export reads.
    from openamp.core.config import EmulateConfig
    from openamp.emulate import export as ex

    m = _model(4)
    ecfg = EmulateConfig(arch="film_wavenet", wn_channels=KW["channels"],
                         embedding_dim=KW["embedding_dim"])
    # from_config rebuilds with the A2 schedule; store our small schedule's
    # weights under a matching model instead: monkeypatch via direct save/load
    # is overkill — just verify against a real from_config-shaped model.
    m2 = FiLMWaveNet.from_config(ecfg, n_devices=3).eval()
    ck = {"model": m2.state_dict(), "emulate_cfg": ecfg.__dict__.copy(),
          "device_ids": [11, 22, 33], "id_to_idx": {11: 0, 22: 1, 33: 2},
          "holdout_ids": [], "manifest_sha256": "x", "sample_rate": 48000,
          "receptive_field": m2.receptive_field, "name": "fake", "epoch": 1,
          "val_esr": 0.5}
    run = tmp_path / "run"
    run.mkdir()
    torch.save(ck, run / "checkpoint.pt")

    out = run / "export" / "bundle.json"
    s = ex.export_bundle(run, out, profile_device_ids=[22])
    assert out.is_file() and s["arch"]["embedding_dim"] == KW["embedding_dim"]
    assert s["arch"]["activation"] == "LeakyReLU"      # NAM's spelling, for the plugin
    assert s["profiles"] == ["Table mean", "device 22"]

    import json
    b = json.loads(out.read_text(encoding="utf-8"))
    assert len(b["weights"]["layers"]) == len(m2.layers)
    # Round-trip one weight exactly and check FiLM matrices ship per layer.
    w0 = _unblob(b["weights"]["layers"][0]["conv_w"])
    assert np.array_equal(w0, m2.layers[0].conv.weight.detach().numpy())
    assert _unblob(b["weights"]["layers"][0]["film_w"]).shape == \
        (2 * m2.channels, m2.embedding_dim)
    # Built-in profile == the table row it came from.
    assert np.allclose(b["profiles"][1]["embedding"],
                       m2.embedding.weight[1].detach().numpy(), atol=1e-6)

    p = ex.export_profile(run, device_id=22)
    assert p["aaom_profile"] == 1 and p["dim"] == KW["embedding_dim"]
    assert p["run"] == b["run"]["sha8"] and len(p["embedding"]) == p["dim"]
    with pytest.raises(RuntimeError):
        ex.export_profile(run, device_id=22, mean=True)   # exactly one source
    with pytest.raises(RuntimeError):
        ex.export_profile(run, device_id=99)              # not in the table

    # export_profiles: trained rows only, in table-index order, matching schema.
    ps = ex.export_profiles(run)
    assert [q["name"] for q in ps] == ["device 11", "device 22", "device 33"]
    assert all(q["aaom_profile"] == 1 and q["dim"] == KW["embedding_dim"]
               and q["run"] == b["run"]["sha8"] for q in ps)
    assert np.allclose(ps[1]["embedding"],
                       m2.embedding.weight[1].detach().numpy(), atol=1e-6)
    # Each row equals export_profile for the same device (shared _profile path).
    assert ps[1] == ex.export_profile(run, device_id=22)
    # --mean prepends the table mean; device rows still follow in order.
    pm = ex.export_profiles(run, mean=True)
    assert [q["name"] for q in pm] == ["Table mean", "device 11", "device 22",
                                       "device 33"]
    assert np.allclose(pm[0]["embedding"],
                       m2.embedding.weight.mean(0).detach().numpy(), atol=1e-6)

    # names map by device_id; ids missing from the map keep the generic label.
    named = ex.export_profiles(run, names={22: "Marshall JCM800"})
    assert [q["name"] for q in named] == ["device 11", "Marshall JCM800", "device 33"]
    assert ex.export_profile(run, device_id=22, names={22: "Marshall JCM800"})["name"] \
        == "Marshall JCM800"
    # An explicit name still wins over the names map.
    assert ex.export_profile(run, device_id=22, name="Custom",
                             names={22: "Marshall JCM800"})["name"] == "Custom"


def test_bundle_conditioning_payload_is_arch_specific(tmp_path):
    """The plugin discriminates the conditioning payload on ``arch.type``, and every
    non-legacy arch *omits* film_w/film_b so a reader that predates it dies on a
    missing key instead of folding garbage. The film_wavenet third of this test is
    the regression guard on the already-shipped contract."""
    import json

    from openamp.core.config import EmulateConfig
    from openamp.emulate import export as ex

    def _bundle(arch, **extra):
        ecfg = EmulateConfig(arch=arch, wn_channels=KW["channels"],
                             embedding_dim=KW["embedding_dim"], **extra)
        from openamp.emulate.models import build_model
        mm = build_model(ecfg, 2).eval()
        run = tmp_path / arch
        run.mkdir()
        torch.save({"model": mm.state_dict(), "emulate_cfg": ecfg.__dict__.copy(),
                    "device_ids": [1, 2], "id_to_idx": {1: 0, 2: 1},
                    "sample_rate": 48000, "name": arch},
                   run / "checkpoint.pt")
        ex.export_bundle(run, run / "b.json")
        return mm, json.loads((run / "b.json").read_text(encoding="utf-8"))

    # film_wavenet: unchanged, ships the two matrices and no film_layers.
    _, b = _bundle("film_wavenet")
    assert b["aaom_bundle"] == 1 and b["arch"]["type"] == "film_wavenet_a2"
    assert "cond_hidden" not in b["arch"]
    l0 = b["weights"]["layers"][0]
    assert "film_w" in l0 and "film_b" in l0 and "film_layers" not in l0

    # mlpfilm_wavenet: the MLP stack, input->output, and no legacy keys.
    mm, b = _bundle("mlpfilm_wavenet", cond_hidden=5, cond_activation="tanh")
    assert b["aaom_bundle"] == 1 and b["arch"]["type"] == "mlpfilm_wavenet_a2"
    assert b["arch"]["cond_hidden"] == 5
    assert b["arch"]["cond_activation"] == "Tanh"     # NAM's spelling
    l0 = b["weights"]["layers"][0]
    assert "film_w" not in l0 and "film_b" not in l0
    assert [_unblob(e["w"]).shape for e in l0["film_layers"]] == \
        [(5, KW["embedding_dim"]), (2 * mm.channels, 5)]
    assert [_unblob(e["b"]).shape for e in l0["film_layers"]] == \
        [(5,), (2 * mm.channels,)]
    # Weights ship in input->output order, exactly as the re-fold consumes them.
    assert np.array_equal(_unblob(l0["film_layers"][0]["w"]),
                          mm.layers[0].film.fc1.weight.detach().numpy())
    assert np.array_equal(_unblob(l0["film_layers"][1]["w"]),
                          mm.layers[0].film.fc2.weight.detach().numpy())

    # delta_wavenet: the weight-residual generator, no FiLM payload at all.
    mm, b = _bundle("delta_wavenet", delta_rank=3)
    assert b["aaom_bundle"] == 1 and b["arch"]["type"] == "delta_wavenet_a2"
    assert b["arch"]["delta_rank"] == 3 and "cond_hidden" not in b["arch"]
    C, K = mm.channels, mm.kernel_sizes[0]
    l0 = b["weights"]["layers"][0]
    assert not {"film_w", "film_b", "film_layers"} & set(l0)
    assert _unblob(l0["delta_coeff_w"]).shape == (3, KW["embedding_dim"])
    assert _unblob(l0["delta_coeff_b"]).shape == (3,)
    # basis rows: one flattened kernel (C-order, like conv_w), bias, mixin gain.
    assert _unblob(l0["delta_basis"]).shape == (3, C * C * K + 2 * C)
    # What ships is scale * normalize(basis), NOT the raw parameter: the training-time
    # split of direction from magnitude must stay invisible to the plugin, whose
    # re-fold is a plain `coeff(e) @ delta_basis` matmul.
    gen = mm.layers[0].delta
    assert not np.array_equal(_unblob(l0["delta_basis"]), gen.basis.detach().numpy())
    assert np.allclose(_unblob(l0["delta_basis"]),
                       gen.effective_basis().detach().numpy(), atol=1e-6)
    # The end-to-end property that matters: reproducing the model's own delta from
    # nothing but the bundle's three matrices.
    e = mm.embedding.weight[1].detach().numpy()
    from_bundle = (_unblob(l0["delta_coeff_w"]) @ e + _unblob(l0["delta_coeff_b"])) \
        @ _unblob(l0["delta_basis"])
    dw, db, dm = gen(mm.embedding.weight[1])
    want = np.concatenate([dw.reshape(-1).detach().numpy(),
                           db.reshape(-1).detach().numpy(),
                           dm.reshape(-1).detach().numpy()])
    assert np.allclose(from_bundle, want, atol=1e-5)
