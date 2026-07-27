"""Morph-plugin export: FiLM weight-folding equivalence, bundle round-trip,
profile schema. The fold test is the correctness anchor for the C++ plugin —
if folding == FiLM here, the plugin only has to reproduce a stock A2 forward.
torch-guarded like the other model tests; no checkpoint or corpus needed
(everything runs on a small random FiLMWaveNet)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openamp.emulate.export import (_blob, _unblob, folded_model,
                                    forward_with_embedding)
from openamp.emulate.wavenet import FiLMWaveNet

# Small-but-real schedule: mixed kernels like A2, a few hundred samples of RF.
KW = dict(n_devices=4, channels=4, embedding_dim=8,
          kernel_sizes=(3, 3, 5, 3), dilations=(1, 2, 4, 8), head_kernel=4)


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    m = FiLMWaveNet(**KW, **kw).eval()
    # Un-neutralize FiLM: trained nets have arbitrary gamma/beta, and identity
    # FiLM would make the fold test vacuous.
    for l in m.layers:
        torch.nn.init.normal_(l.film.weight, std=0.3)
        torch.nn.init.normal_(l.film.bias, std=0.3)
    return m


@torch.no_grad()
def test_fold_matches_film_on_table_rows():
    m = _model()
    x = torch.randn(2, 1500)
    for row in range(KW["n_devices"]):
        idx = torch.tensor([row, row])
        want = m(x, idx)
        got = folded_model(m, m.embedding.weight[row])(x, torch.tensor([0, 0]))
        assert torch.allclose(want, got, atol=1e-5), \
            f"row {row}: max err {(want - got).abs().max().item():.2e}"


@torch.no_grad()
def test_fold_matches_film_on_arbitrary_morph_points():
    # Morph points are convex (and extrapolated) mixes — never seen as rows.
    m = _model(1)
    x = torch.randn(1, 1200)
    e0, e1 = m.embedding.weight[0], m.embedding.weight[1]
    for alpha in (-0.5, 0.25, 0.5, 1.5):
        e = (1 - alpha) * e0 + alpha * e1
        want = forward_with_embedding(m, x, e)
        got = folded_model(m, e)(x, torch.tensor([0]))
        assert torch.allclose(want, got, atol=1e-5)


@torch.no_grad()
def test_forward_with_embedding_matches_forward_on_rows():
    # The unfolded reference itself must equal the model's own forward.
    m = _model(2)
    x = torch.randn(2, 800)
    idx = torch.tensor([1, 3])
    want = m(x, idx)
    for b in range(2):
        got = forward_with_embedding(m, x[b:b + 1], m.embedding.weight[idx[b]])
        # batch-1 vs batch-2 convs may take different GEMM paths: tight, not exact
        assert torch.allclose(want[b:b + 1], got, atol=1e-7)


@torch.no_grad()
def test_fold_matches_film_under_tanh_activation():
    """Folding rewrites conv/mixin weights *upstream* of the nonlinearity, so a
    ``wn_activation: tanh`` run must fold exactly like the LeakyReLU one — and the
    bundle must name it in NAM's spelling so the plugin picks the right one."""
    from openamp.emulate.wavenet import NAM_ACTIVATION_NAMES

    m = _model(5, activation="tanh")
    assert NAM_ACTIVATION_NAMES[m.activation] == "Tanh"   # what export_bundle writes
    x = torch.randn(1, 1200) * 2.0                        # into tanh saturation
    for row in range(KW["n_devices"]):
        e = m.embedding.weight[row]
        want = forward_with_embedding(m, x, e)
        got = folded_model(m, e)(x, torch.tensor([0]))
        assert torch.allclose(want, got, atol=1e-5)


@torch.no_grad()
def test_folded_model_ignores_device_idx():
    m = _model(3)
    x = torch.randn(1, 600)
    f = folded_model(m, m.embedding.weight[2])
    assert torch.equal(f(x, torch.tensor([0])), f(x, torch.tensor([3])))


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
