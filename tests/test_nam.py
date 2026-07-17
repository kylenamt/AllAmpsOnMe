"""Loader tests for both capture architectures.

A2 exports a ``SlimmableContainer`` of width variants; A1 exports a bare WaveNet
in a pre-0.7 schema the current NAM no longer parses. Both must load, so both are
built here from real NAM exports rather than checked-in fixtures.
"""

import copy
import json

import numpy as np
import pytest

from openamp.dsp import nam as N

torch = pytest.importorskip("torch")
_wavenet = pytest.importorskip("nam.models.wavenet")


def _tiny_export() -> dict:
    """A real (tiny) .nam export dict, straight from NAM's own exporter."""
    net = _wavenet.WaveNet.init_from_config({
        "layers_configs": [{
            "input_size": 1, "condition_size": 1, "channels": 2, "kernel_size": 2,
            "dilations": [1, 2], "activation": "Tanh",
            "head": {"out_channels": 1, "kernel_size": 1, "bias": False},
        }],
        "head_config": None, "head_scale": 0.02,
    })
    return net._get_export_dict()


def _as_legacy(export: dict) -> dict:
    """Rewrite a current-schema WaveNet export into the pre-0.7 (A1) schema.

    The inverse of what the loader's upgrade step does, so a round-trip through
    both must reproduce the original net exactly.
    """
    legacy = copy.deepcopy(export)
    layer = legacy["config"]["layers"][0]
    head = layer.pop("head")
    for key in ("kernel_sizes", "bottleneck", "head1x1", "layer1x1", "groups_input",
                "groups_input_mixin", "conv_pre_film", "conv_post_film",
                "input_mixin_pre_film", "input_mixin_post_film", "activation_pre_film",
                "activation_post_film", "layer1x1_post_film", "head1x1_post_film",
                "gating_mode", "secondary_activation", "slimmable"):
        layer.pop(key, None)
    layer.update({
        "gated": False, "kernel_size": 2, "activation": "Tanh",
        "head_size": head["out_channels"], "head_bias": head["bias"],
    })
    legacy["version"] = "0.5.4"
    return legacy


def _container(narrow: dict, wide: dict) -> dict:
    """An A2 container over two width variants, narrow one listed first."""
    return {
        "version": "0.7.0", "architecture": N.CONTAINER_ARCH, "sample_rate": 48000,
        "weights": [], "metadata": {},
        "config": {"submodels": [{"max_value": 0.5, "model": narrow},
                                 {"max_value": 1, "model": wide}]},
    }


def _write(tmp_path, name, data) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _signal() -> np.ndarray:
    return (np.random.default_rng(0).standard_normal(2048) * 0.2).astype(np.float32)


def test_full_width_submodel_picks_the_widest_not_the_first():
    narrow, wide = {"architecture": "narrow"}, {"architecture": "wide"}
    data = _container(narrow, wide)
    assert N.is_container(data)
    assert N.full_width_submodel(data) is wide


def test_full_width_submodel_rejects_empty_container():
    with pytest.raises(N.NamBackendError):
        N.full_width_submodel({"architecture": N.CONTAINER_ARCH, "config": {"submodels": []}})


def test_parse_nam_reads_weights_through_the_container(tmp_path):
    export = _tiny_export()
    path = _write(tmp_path, "a2.nam", _container(export, export))
    summary = N.parse_nam(path)

    assert summary["architecture"] == N.CONTAINER_ARCH
    assert summary["n_submodels"] == 2
    # The container itself carries no top-level weights; the variants hold them.
    assert summary["has_weights"] is True


def test_a2_container_renders_through_its_full_width_variant(tmp_path):
    wide = _tiny_export()
    narrow = _tiny_export()  # different random init -> different output
    a2 = _write(tmp_path, "a2.nam", _container(narrow, wide))
    wide_alone = _write(tmp_path, "wide.nam", wide)

    x = _signal()
    out = N.default_load_and_render(a2, x, 48000)

    assert out.shape == x.shape and np.isfinite(out).all()
    np.testing.assert_array_equal(out, N.default_load_and_render(wide_alone, x, 48000))


def test_legacy_a1_export_still_loads_and_renders_identically(tmp_path):
    export = _tiny_export()
    current = _write(tmp_path, "current.nam", export)
    legacy = _write(tmp_path, "legacy.nam", _as_legacy(export))

    x = _signal()
    # Same net, two schemas: the upgrade must be lossless, not merely loadable.
    np.testing.assert_array_equal(N.default_load_and_render(legacy, x, 48000),
                                  N.default_load_and_render(current, x, 48000))


def test_gated_legacy_export_is_refused_rather_than_guessed(tmp_path):
    legacy = _as_legacy(_tiny_export())
    legacy["config"]["layers"][0]["gated"] = True
    path = _write(tmp_path, "gated.nam", legacy)

    with pytest.raises(N.NamBackendError, match="gated"):
        N.default_load_and_render(path, _signal(), 48000)


def test_load_model_reports_receptive_field_for_both_architectures(tmp_path):
    export = _tiny_export()
    a1 = _write(tmp_path, "a1.nam", _as_legacy(export))
    a2 = _write(tmp_path, "a2.nam", _container(export, export))

    for path in (a1, a2):
        model = N.load_model(path, device="cpu")
        assert model.receptive_field > 0
        assert model.sample_rate == 48000
        assert model.forward(np.zeros(512, dtype=np.float32)).shape == (512,)
