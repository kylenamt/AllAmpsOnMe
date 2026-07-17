"""Verification tests: pure signal-sanity + completeness checks."""

import numpy as np
import pandas as pd

from openamp.corpus import verify
from openamp.core import constants as C
from openamp.core import manifest as manifests


def test_is_pass_through():
    x = np.random.default_rng(0).standard_normal(4096).astype(np.float32)
    assert verify.is_pass_through(x, x.copy())                 # identical -> pass-through
    assert not verify.is_pass_through(x, np.tanh(3.0 * x))     # clearly processed


def test_is_silent():
    assert verify.is_silent(np.zeros(1000, np.float32))
    assert not verify.is_silent(0.1 * np.ones(1000, np.float32))


def test_has_non_finite():
    x = np.ones(10, np.float32)
    assert not verify.has_non_finite(x)
    x[3] = np.nan
    assert verify.has_non_finite(x)
    x[3] = np.inf
    assert verify.has_non_finite(x)


def test_signal_flags_clean_pass():
    rng = np.random.default_rng(1)
    inp = rng.standard_normal(8192).astype(np.float32)
    out = np.tanh(2.0 * inp).astype(np.float32)
    flags = verify.signal_flags(inp, out)
    assert flags == {"non_finite": False, "silent": False, "pass_through": False}


def test_aligned():
    assert verify.aligned(96_000, 96_000)
    assert not verify.aligned(96_000, 95_999)


def test_missing_pairs():
    renders = pd.DataFrame([
        {"device_id": 0, "file_id": "f0", "status": C.RENDER_OK},
        {"device_id": 0, "file_id": "f1", "status": C.RENDER_FAILED},
        {"device_id": 1, "file_id": "f0", "status": C.RENDER_OK},
    ], columns=manifests.RENDERS_COLUMNS)
    miss = verify.missing_pairs(renders, [0, 1], ["f0", "f1"])
    # device 0/f1 failed (not ok), device 1/f1 absent.
    assert (0, "f1") in miss and (1, "f1") in miss
    assert (0, "f0") not in miss and (1, "f0") not in miss


def test_missing_pairs_empty_renders():
    empty = manifests.ensure_schema(pd.DataFrame(), manifests.RENDERS_COLUMNS)
    assert verify.missing_pairs(empty, [0], ["f0"]) == [(0, "f0")]
