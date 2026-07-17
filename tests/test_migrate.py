import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import make_candidate

from openamp.acquire import migrate as mig
from openamp.core import manifest
from openamp.core.manifest import STATUS_VALIDATED


class FakeClient:
    def __init__(self, a2_models=None, fail_urls=()):
        self.a2_models = a2_models or {}
        self.fail_urls = set(fail_urls)
        self.downloaded = []

    def list_models(self, tone_id, architecture=None):
        assert architecture == 2, "migration must ask for A2 explicitly"
        return self.a2_models.get(tone_id, [])

    def download_model(self, url, dest):
        if url in self.fail_urls:
            raise RuntimeError("boom")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        # Shaped like a real A2 export: validate_one parses the JSON before it
        # ever reaches the render backend.
        data = json.dumps({
            "version": "0.7.0", "architecture": "SlimmableContainer",
            "sample_rate": 48000, "weights": [], "metadata": {},
            "config": {"submodels": [
                {"max_value": 1, "model": {"architecture": "WaveNet",
                                           "config": {"layers": []}, "weights": [0.1]}},
            ]},
        }).encode()
        Path(dest).write_bytes(data)
        self.downloaded.append(url)
        return hashlib.sha256(data).hexdigest(), len(data)


def _a2(model_id, name):
    return {"id": model_id, "name": name, "sample_rate": 48000,
            "model_url": f"https://t3k/models/{model_id}.nam"}


def _device(device_id, tone_id, model_id, model_name):
    return make_candidate(tone_id=tone_id, model_id=model_id, model_name=model_name,
                          architecture="A1", device_id=device_id,
                          file_path=f"/old/{model_id}.nam", status=STATUS_VALIDATED)


def _df(rows):
    return manifest.ensure_schema(pd.DataFrame(rows))


def _render_ok(path, signal, sample_rate):
    """Stand-in for the NAM backend: a plausible, quiet-in-silence output."""
    out = np.asarray(signal, dtype=np.float32) * 0.5
    return out


def test_swaps_to_the_a2_twin_and_keeps_the_device_id(settings):
    df = _df([_device(7, tone_id=1, model_id=100, model_name="JCM800 Lead")])
    client = FakeClient({1: [_a2(200, "JCM800 Lead")]})

    out = mig.migrate(client, df, settings, render_fn=_render_ok)
    row = out.iloc[0]

    assert row["device_id"] == 7          # renders on disk are addressed by this
    assert row["architecture"] == "A2"
    assert row["model_id"] == 200
    assert row["status"] == STATUS_VALIDATED
    assert Path(row["file_path"]).is_file()
    assert bool(row["architecture_fallback"]) is False


def test_device_without_an_a2_twin_keeps_its_a1_capture(settings):
    """Models on one tone are the same amp at different gain settings, so an
    unmatched name must not fall back to a sibling."""
    df = _df([_device(7, tone_id=1, model_id=100, model_name="JCM800 Lead")])
    client = FakeClient({1: [_a2(200, "JCM800 Crunch"), _a2(201, "JCM800 Clean")]})

    out = mig.migrate(client, df, settings, render_fn=_render_ok)
    row = out.iloc[0]

    assert row["architecture"] == "A1"
    assert row["model_id"] == 100
    assert row["file_path"] == "/old/100.nam"
    assert bool(row["architecture_fallback"]) is True
    assert client.downloaded == []


def test_a2_capture_that_fails_validation_is_discarded(settings):
    df = _df([_device(7, tone_id=1, model_id=100, model_name="JCM800 Lead")])
    client = FakeClient({1: [_a2(200, "JCM800 Lead")]})

    def render_silent(path, signal, sample_rate):   # fails the "audible" check
        return np.zeros_like(np.asarray(signal, dtype=np.float32))

    out = mig.migrate(client, df, settings, render_fn=render_silent)
    row = out.iloc[0]

    assert row["architecture"] == "A1"
    assert bool(row["architecture_fallback"]) is True
    assert not (settings.captures_dir / "1" / "200.nam").exists()


def test_a1_manifest_is_backed_up_before_the_swap(settings):
    df = _df([_device(7, tone_id=1, model_id=100, model_name="JCM800 Lead")])
    client = FakeClient({1: [_a2(200, "JCM800 Lead")]})

    mig.migrate(client, df, settings, render_fn=_render_ok)

    backup = pd.read_parquet(settings.manifests_dir / "manifest.a1.parquet")
    assert backup.iloc[0]["architecture"] == "A1"
    assert backup.iloc[0]["model_id"] == 100


def test_two_devices_never_collapse_onto_one_a2_capture(settings):
    """Same-named models under one tone must not both claim the same A2 file --
    that would merge two device_ids onto a single device."""
    df = _df([_device(7, tone_id=1, model_id=100, model_name="JCM800 Lead"),
              _device(8, tone_id=1, model_id=101, model_name="JCM800 Lead")])
    client = FakeClient({1: [_a2(200, "JCM800 Lead")]})

    out = mig.migrate(client, df, settings, render_fn=_render_ok)

    first, second = out.iloc[0], out.iloc[1]
    assert first["architecture"] == "A2" and first["model_id"] == 200
    assert second["architecture"] == "A1" and second["model_id"] == 101
    assert bool(second["architecture_fallback"]) is True
    assert out["model_id"].nunique() == 2


def test_already_migrated_devices_are_left_alone(settings):
    row = _device(7, tone_id=1, model_id=200, model_name="JCM800 Lead")
    row["architecture"] = "A2"
    client = FakeClient({1: [_a2(200, "JCM800 Lead")]})

    out = mig.migrate(client, _df([row]), settings, render_fn=_render_ok)

    assert client.downloaded == []
    assert out.iloc[0]["architecture"] == "A2"
