"""Phase 2 render-core tests: the chunked forward must be bit-identical to a full
forward, plus output-scale, device-selection, and resume bookkeeping."""

import numpy as np
import pytest

from data import render
from data import constants as C
from data import manifests
from data.config import load_config


def make_fir(h):
    """A same-length causal FIR model: y[n] = sum_k h[k]*x[n-k], x[<0]=0.

    Accumulates taps in a fixed order so a given output index rounds identically
    regardless of input length — the property chunked_forward relies on.
    """
    h = np.asarray(h, dtype=np.float64)

    def model_fn(sig):
        sig = np.asarray(sig, dtype=np.float64)
        y = h[0] * sig
        for k in range(1, len(h)):
            y[k:] += h[k] * sig[:-k]
        return y.astype(np.float32)

    return model_fn, len(h) - 1


@pytest.mark.parametrize("R", [0, 1, 5, 37, 200])
def test_chunked_forward_bit_identical_to_full(R):
    rng = np.random.default_rng(R)
    model_fn, rf = make_fir(rng.standard_normal(R + 1))
    assert rf == R
    # Small signals stress every chunk size (incl. sample-by-sample); large signals
    # only need realistic chunks — the tiny-chunk cases already cover the seams.
    cases = [
        (1, (1,)),
        (33, (1, 7, 33, 40)),
        (4096, (1, 7, 1000, 4096, 5000)),
        (40_000, (1000, 9973, 40_000, 40_050)),
    ]
    for length, chunks in cases:
        x = rng.standard_normal(length).astype(np.float32)
        full = model_fn(x)
        for chunk in chunks:
            out = render.chunked_forward(model_fn, x, R, chunk)
            assert out.dtype == np.float32
            assert np.array_equal(out, full), f"R={R} len={length} chunk={chunk}"


def test_chunked_forward_empty():
    model_fn, _ = make_fir([1.0])
    assert render.chunked_forward(model_fn, np.array([], np.float32), 0, 10).size == 0


def test_chunked_forward_rejects_too_short_model_output():
    def bad(sig):
        return np.zeros(1, dtype=np.float32)  # fewer than one chunk's worth of samples

    with pytest.raises(ValueError):
        render.chunked_forward(bad, np.ones(100, np.float32), 4, 10)


@pytest.mark.parametrize("workers,ahead", [(1, 1), (2, 4), (4, 3), (8, 8)])
def test_prefetch_preserves_order_and_completeness(workers, ahead):
    items = [{"i": i} for i in range(50)]
    out = list(render._prefetch(items, lambda d: d["i"] * 10,
                                workers=workers, ahead=ahead))
    assert [it for it, _ in out] == items           # same objects, same order
    assert [v for _, v in out] == [i * 10 for i in range(50)]


def test_prefetch_empty():
    assert list(render._prefetch([], lambda d: d)) == []


def test_prefetch_error_surfaces_in_order():
    """A load failure must appear where that item would be consumed — not early
    (look-ahead runs it sooner) and not swallowed."""
    items = [{"i": i} for i in range(20)]

    def load(d):
        if d["i"] == 7:
            raise ValueError("boom")
        return d["i"]

    seen = []
    with pytest.raises(ValueError):
        for it, _ in render._prefetch(items, load, workers=4, ahead=4):
            seen.append(it["i"])
    assert seen == list(range(7))                    # 0..6 delivered, then raise


def test_output_scale_boundaries():
    assert render.output_scale(0.5) == 1.0
    assert render.output_scale(0.999) == 1.0            # exactly at ceiling: no scaling
    s = render.output_scale(2.0)
    assert abs(2.0 * s - C.OUTPUT_PEAK_CEILING) < 1e-9   # peak lands on the ceiling


def test_parse_device_selection():
    ids = list(range(10))
    assert render.parse_device_selection(None, ids) == ids
    assert render.parse_device_selection("0-3", ids) == [0, 1, 2, 3]
    assert render.parse_device_selection("0,2,5", ids) == [0, 2, 5]
    assert render.parse_device_selection("8-100", ids) == [8, 9]   # clamped to available


def test_parse_device_selection_from_file(tmp_path):
    ids = list(range(10))
    f = tmp_path / "subset.txt"
    f.write_text("# a comment\n0\n2-4  # inline comment\n7\n", encoding="utf-8")
    assert render.parse_device_selection(f"@{f}", ids) == [0, 2, 3, 4, 7]


def test_estimate_bytes_scales():
    one = render.estimate_bytes(600.0, 1)
    assert render.estimate_bytes(600.0, 10) == 10 * one


def test_device_is_done(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    cfg.ensure_dirs()
    import pandas as pd

    df = pd.DataFrame([
        {"device_id": 0, "file_id": "egdb_0000", "status": C.RENDER_OK},
        {"device_id": 0, "file_id": "egdb_0001", "status": C.RENDER_OK},
    ], columns=manifests.RENDERS_COLUMNS)

    # No files on disk yet -> not done even though rows exist.
    assert not render.device_is_done(cfg, 0, ["egdb_0000", "egdb_0001"], df)

    ddir = cfg.device_render_dir(0)
    ddir.mkdir(parents=True, exist_ok=True)
    for fid in ("egdb_0000", "egdb_0001"):
        (ddir / f"{fid}.{cfg.output_format}").write_bytes(b"x")
    assert render.device_is_done(cfg, 0, ["egdb_0000", "egdb_0001"], df)
    # A missing file_id row -> not done.
    assert not render.device_is_done(cfg, 0, ["egdb_0000", "egdb_9999"], df)
