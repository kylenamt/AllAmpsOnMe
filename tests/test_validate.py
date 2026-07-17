import shutil

import numpy as np
import pytest
from conftest import FIXTURES

from openamp.dsp import audio as probe_mod, nam as nam_backend
from openamp.acquire import validate
from openamp.core.manifest import STATUS_REJECTED, STATUS_VALIDATED

# The NAM fixtures are not committed (see tests/fixtures/README.md); skip the
# cases that need them rather than failing on a fresh checkout.
needs_nam_fixture = pytest.mark.skipif(
    not (FIXTURES / "tiny_a2.nam").exists(),
    reason="tests/fixtures/tiny_a2.nam not present (see tests/fixtures/README.md)")


def _probe(sr=8000):
    return probe_mod.generate_probe(sr)


def test_check_output_pass():
    p = _probe()
    assert validate._check_output(p.signal.copy(), p) is None


def test_check_output_non_finite():
    p = _probe()
    bad = p.signal.copy()
    bad[10] = np.inf
    assert validate._check_output(bad, p) == "non_finite_output"


def test_check_output_exploding():
    p = _probe()
    assert validate._check_output(p.signal * 20.0, p) == "exploding_output"


def test_check_output_silent():
    p = _probe()
    assert validate._check_output(np.zeros_like(p.signal), p) == "silent_output"


def test_check_output_noise_in_silence():
    p = _probe()
    out = p.signal.copy()
    out[p.silence] = 0.3
    assert validate._check_output(out, p) == "noise_in_silence"


@needs_nam_fixture
def test_parse_nam_fixtures():
    a2 = nam_backend.parse_nam(FIXTURES / "tiny_a2.nam")
    assert a2["architecture"] == "WaveNet"
    assert a2["sample_rate"] == 48000
    with pytest.raises(nam_backend.NamBackendError):
        nam_backend.parse_nam(FIXTURES / "broken.nam")


@needs_nam_fixture
def test_validate_one_pass_with_injected_renderer(settings):
    dest = settings.captures_dir / "1" / "10.nam"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "tiny_a2.nam", dest)
    row = {"tone_id": 1, "model_id": 10, "file_path": str(dest),
           "architecture": "A2", "make": "fender", "model": "deluxe"}

    def good_render(path, signal, sr):
        return np.asarray(signal, dtype=np.float32)

    fields = validate.validate_one(row, settings, good_render)
    assert fields["status"] == STATUS_VALIDATED
    assert (settings.captures_dir / "1" / "10.probe.npy").is_file()


@needs_nam_fixture
def test_validate_one_render_failure_rejects(settings):
    dest = settings.captures_dir / "1" / "11.nam"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "tiny_a2.nam", dest)
    row = {"tone_id": 1, "model_id": 11, "file_path": str(dest), "architecture": "A2"}

    def boom(path, signal, sr):
        raise RuntimeError("cannot load")

    fields = validate.validate_one(row, settings, boom)
    assert fields["status"] == STATUS_REJECTED
    assert fields["reject_reason"] == "load_or_render_failed"


def test_validate_one_missing_file(settings):
    fields = validate.validate_one({"file_path": "/nope/x.nam"}, settings,
                                   lambda *a: np.zeros(1, np.float32))
    assert fields["status"] == STATUS_REJECTED
    assert fields["reject_reason"] == "file_missing"
