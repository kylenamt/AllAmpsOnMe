"""Tone encoder: shapes / receptive field / anti-aliasing, the contrastive losses,
and the (dry, wet) segment dataset's P x K batch structure.

torch-guarded like the other model tests; the dataset cases additionally need
soundfile (they write tiny real FLACs in tmp_path). Architectures are shrunken but
real — no mocks — so an assertion about causality-free padding or pooling width is
an assertion about the code that actually runs.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from openamp.core.config import EncoderConfig, load_config  # noqa: E402
from openamp.encoder.dataset import time_split_regions  # noqa: E402
from openamp.encoder.model import (BlurPool1d, Branch, ToneEncoder,  # noqa: E402
                                     aggregate_capture, count_parameters)
from openamp.encoder.train import (cross_covariance, supcon,  # noqa: E402
                                           supcon_floor)

# A shrunken encoder: same structure, ~200x fewer params, so tests stay fast.
SMALL = dict(stem_channels=(8, 8, 16, 16), static_dilations=(1, 2),
             dynamic_dilations=(1, 2, 4), head_dim=12, proj_hidden=12, proj_dim=6,
             norm_groups=4)
SMALL_T = 4096          # divisible by the 256x stem downsample


# --- Model ---------------------------------------------------------------------
def test_receptive_field_matches_the_spec_numbers():
    # Each block is 2 convs of k=3 at one dilation, so RF = 1 + 2*sum((k-1)*d).
    # static  (1,2,4):        1 + 4*7   = 29   frames = 155 ms at 187.5 Hz
    # dynamic (1..128):       1 + 4*255 = 1021 frames = 5.45 s
    assert Branch.compute_receptive_field((1, 2, 4)) == 29
    assert Branch.compute_receptive_field((1, 2, 4, 8, 16, 32, 64, 128)) == 1021
    m = ToneEncoder()
    assert m.static.receptive_field == 29
    assert m.dynamic.receptive_field == 1021
    # The dynamic branch must span essentially a whole segment, or the branch is
    # wasted. It lands 3 frames short of the 1024-frame segment (1021 vs 1024, 99.7%)
    # — that is the spec's own pairing of 262,144 samples with these dilations, and
    # the shortfall is the centre frame plus symmetric padding, not a bug.
    frames = 262_144 // m.downsample
    assert frames == 1024
    assert m.dynamic.receptive_field >= 0.99 * frames
    assert m.static.receptive_field < 0.05 * frames        # static stays local


def test_stem_downsamples_exactly_256x():
    m = ToneEncoder().eval()
    h = m.encode_frames(torch.randn(2, 2, 262_144))
    assert m.downsample == 256
    assert h.shape == (2, 128, 1024)          # (B, C, T/256) -> 187.5 Hz frame rate


def test_forward_shapes_and_per_branch_unit_norm():
    m = ToneEncoder(**SMALL).eval()
    emb, s, d = m(torch.randn(3, 2, SMALL_T))
    assert s.shape == (3, 12) and d.shape == (3, 12)
    assert emb.shape == (3, 24) == (3, m.embedding_dim)
    # Each branch is normalized SEPARATELY (that is what stops one dominating by
    # norm), so the concat has norm sqrt(2), not 1.
    assert torch.allclose(s.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(d.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(emb.norm(dim=-1), torch.full((3,), 2 ** 0.5), atol=1e-5)
    assert torch.allclose(emb, torch.cat([s, d], dim=-1))


def test_dynamic_pooling_is_four_statistics_wide():
    m = ToneEncoder(**SMALL)
    # static pools C features (mean); dynamic pools 4C (mean, std, |mean(d)|, std(d)).
    assert m.static.head[0].in_features == 16
    assert m.dynamic.head[0].in_features == 64


def test_dynamic_branch_sees_envelope_that_static_pooling_destroys():
    """A ramp and its time-reverse have the same mean but opposite rate of change."""
    m = ToneEncoder(**SMALL).eval()
    h = torch.linspace(-1, 1, 64).view(1, 1, 64).repeat(1, 16, 1)
    fwd, rev = m.dynamic.pool(h, None), m.dynamic.pool(h.flip(-1), None)
    # mean and std blocks (the first 2C) are identical...
    assert torch.allclose(fwd[:, :32], rev[:, :32], atol=1e-5)
    # ...and the delta blocks carry real, non-zero signal that the mean has no room for.
    assert fwd[:, 32:48].abs().max() > 1e-3
    mean_pool = m.static.pool(h, None)
    assert torch.allclose(mean_pool, m.static.pool(h.flip(-1), None), atol=1e-5)


def test_blurpool_kernel_is_fixed_unit_gain():
    bp = BlurPool1d(4, stride=4)
    assert bp.kernel.requires_grad is False            # a buffer, not a parameter
    assert "kernel" in dict(bp.named_buffers())
    assert bp.kernel.shape == (4, 1, 5)                # depthwise
    assert pytest.approx(1.0, abs=1e-6) == float(bp.kernel[0, 0].sum())
    # Unit DC gain: a constant signal passes through unchanged (interior samples).
    out = bp(torch.ones(1, 4, 64))
    assert torch.allclose(out[..., 1:-1], torch.ones_like(out[..., 1:-1]), atol=1e-6)


def test_blurpool_attenuates_nyquist_that_naive_striding_would_alias():
    """+1,-1,+1,... is exactly the content a stride-4 decimation folds back down.

    The binomial kernel has a zero at Nyquist ((1-4+6-4+1)/16 = 0), so the interior
    is annihilated; the first and last outputs keep an edge artifact from the zero
    padding, which is why the check skips them.
    """
    bp = BlurPool1d(1, stride=4)
    nyq = torch.tensor([1.0, -1.0]).repeat(64).view(1, 1, -1)
    assert float(bp(nyq)[..., 1:-1].abs().max()) < 1e-6   # blurred: gone
    assert float(nyq[..., ::4].abs().max()) == 1.0        # naive striding: full scale


def test_encoder_rejects_wrong_channel_count():
    m = ToneEncoder(**SMALL)
    with pytest.raises(ValueError, match="2 channels"):
        m(torch.randn(2, 3, SMALL_T))


def test_from_config_round_trips_and_is_size_driven():
    ncfg = EncoderConfig(stem_channels=[8, 8, 16, 16], static_dilations=[1, 2],
                         dynamic_dilations=[1, 2, 4], head_dim=12, proj_hidden=12,
                         proj_dim=6, norm_groups=4)
    m = ToneEncoder.from_config(ncfg)
    assert m.embedding_dim == ncfg.embedding_dim == 24
    assert m.static.receptive_field == Branch.compute_receptive_field([1, 2])
    big = ToneEncoder.from_config(EncoderConfig())
    assert count_parameters(big) > count_parameters(m)


def test_stem_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        ToneEncoder(stem_channels=(8, 16), stem_kernels=(3, 3, 3), stem_strides=(2, 2))


def test_aggregate_capture_renormalizes_each_half():
    s = F.normalize(torch.randn(7, 16), dim=-1)
    d = F.normalize(torch.randn(7, 16), dim=-1)
    v = aggregate_capture(s, d)
    assert v.shape == (32,)
    assert pytest.approx(1.0, abs=1e-5) == float(v[:16].norm())
    assert pytest.approx(1.0, abs=1e-5) == float(v[16:].norm())
    # Static is a plain mean; dynamic adds 0.5*std, so identical segments differ only
    # in that the dynamic spread term vanishes.
    one = F.normalize(torch.randn(1, 16), dim=-1)
    v1 = aggregate_capture(one.repeat(4, 1), one.repeat(4, 1))
    assert torch.allclose(v1[:16], v1[16:], atol=1e-5)


# --- Losses ---------------------------------------------------------------------
def test_supcon_ranks_clustered_below_random_below_shuffled():
    torch.manual_seed(0)
    labels = torch.arange(4).repeat_interleave(4)          # 4 captures x 4 segments
    centers = F.normalize(torch.randn(4, 16), dim=-1)
    tight = F.normalize(centers[labels] + 1e-3 * torch.randn(16, 16), dim=-1)
    scattered = F.normalize(torch.randn(16, 16), dim=-1)
    clustered = float(supcon(tight, labels, 0.1))
    random_z = float(supcon(scattered, labels, 0.1))
    shuffled = float(supcon(tight, labels[torch.randperm(16)], 0.1))
    assert clustered < random_z < shuffled


def test_supcon_floor_is_log_k_minus_1_not_zero():
    """The number the --overfit sanity check must be read against."""
    torch.manual_seed(0)
    for k in (2, 4, 8):
        labels = torch.arange(4).repeat_interleave(k)
        # Perfectly collapsed positives, near-orthogonal captures in high dim.
        centers = F.normalize(torch.randn(4, 512), dim=-1)
        z = F.normalize(centers[labels], dim=-1)
        assert float(supcon(z, labels, 0.1)) == pytest.approx(supcon_floor(k), abs=0.05)
    assert supcon_floor(4) == pytest.approx(np.log(3))


def test_supcon_ignores_items_with_no_positive():
    z = F.normalize(torch.randn(8, 16), dim=-1)
    assert float(supcon(z, torch.arange(8), 0.1)) == 0.0   # every label unique
    # A batch where only some items have partners scores on those items alone.
    mixed = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6])
    assert float(supcon(z, mixed, 0.1)) > 0.0


def test_supcon_is_finite_and_differentiable_for_identical_rows():
    z = F.normalize(torch.ones(4, 8), dim=-1).requires_grad_()
    loss = supcon(z, torch.tensor([0, 0, 1, 1]), 0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_cross_covariance_is_high_for_copies_and_low_for_independent():
    torch.manual_seed(0)
    a = torch.randn(256, 32)
    assert float(cross_covariance(a, a)) > 5 * float(cross_covariance(a, torch.randn(256, 32)))
    # Scale-invariant: it standardizes per feature first, so a louder branch is not
    # penalized for being loud.
    assert float(cross_covariance(a, 100 * a)) == pytest.approx(
        float(cross_covariance(a, a)), rel=1e-3)


# --- Dataset: tiny real corpus, P x K structure ---------------------------------
def _build_sweep_corpus(cfg, devices, *, seconds=4.0, sr=48_000, identity=True,
                        tail_quiet_db=0.0):
    """Clean + render FLACs with a `sweep` source, so `sources: [sweep]` matches.

    ``tail_quiet_db`` attenuates the last 20% of the clean file, reproducing the
    real capture signal's shape: the level falls off toward the end, which is what
    makes a tail split validate at the wrong operating point.
    """
    import pandas as pd

    from openamp.core import constants as C
    from openamp.core import manifest as manifests
    from openamp.dsp import audio as audio_io

    cfg.ensure_dirs()
    n = int(seconds * sr)
    rng = np.random.default_rng(0)
    sig = (0.3 * rng.standard_normal(n)).astype(np.float32)
    if tail_quiet_db:
        sig[int(0.8 * n):] *= np.float32(10.0 ** (-abs(tail_quiet_db) / 20.0))
    corpus_rows, render_rows, man_rows = [], [], []
    cdir = cfg.clean_split_dir("train")
    cdir.mkdir(parents=True, exist_ok=True)
    audio_io.write_flac(cdir / f"sweep_train.{cfg.output_format}", sig, sr)
    corpus_rows.append({"file_id": "sweep_train", "split": "train",
                        "source": C.SOURCE_SWEEP, "orig_path": "",
                        "duration_s": seconds, "applied_gain_db": 0.0, "sha256": ""})
    for d in devices:
        rdir = cfg.device_render_dir(d)
        rdir.mkdir(parents=True, exist_ok=True)
        # A per-device gain makes captures actually distinguishable; identity=True
        # keeps wet == dry so alignment is provable.
        wet = sig if identity else np.tanh(sig * (1.0 + d)).astype(np.float32)
        rp = rdir / f"sweep_train.{cfg.output_format}"
        audio_io.write_flac(rp, wet, sr)
        render_rows.append({"device_id": d, "file_id": "sweep_train", "split": "train",
                            "path": str(rp), "status": C.RENDER_OK, "output_scale": 1.0})
        man_rows.append({"device_id": d, "tone_id": d // 2,      # 2 captures per amp
                         "gain_bucket": C.GAIN_BUCKETS[d % 3]})
    manifests.write_manifest(pd.DataFrame(corpus_rows, columns=manifests.CORPUS_COLUMNS),
                             cfg.corpus_manifest_path)
    manifests.write_manifest(
        manifests.ensure_schema(pd.DataFrame(render_rows), manifests.RENDERS_COLUMNS),
        cfg.renders_manifest_path)
    manifests.write_manifest(
        manifests.ensure_schema(pd.DataFrame(man_rows), manifests.ALL_COLUMNS),
        cfg.manifest_path)


def _seg_dataset(cfg, **kw):
    from openamp.emulate.dataset import build_device_index
    from openamp.encoder.dataset import ToneSegmentDataset

    ids, id_to_idx = build_device_index(cfg)
    kw.setdefault("segment_samples", 24_000)
    kw.setdefault("sources", ("sweep",))
    return ToneSegmentDataset(cfg, "train", id_to_idx=id_to_idx, seed=7, **kw)


def test_batch_is_p_distinct_captures_of_k_segments(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=list(range(8)))
    ds = _seg_dataset(cfg, episodes=3, devices_per_batch=4, segments_per_device=3)

    assert len(ds) == 3 * 4 * 3 and ds.batch_size() == 12
    for episode in range(3):
        items = [ds.item_arrays(episode * 12 + j) for j in range(12)]
        devs = [it["device_id"] for it in items]
        # K consecutive items share a capture (the SupCon positives)...
        for slot in range(4):
            block = devs[slot * 3:(slot + 1) * 3]
            assert len(set(block)) == 1
        # ...and the P slots are distinct captures (real negatives, not accidental
        # positives that would make the loss meaningless).
        assert len({devs[s * 3] for s in range(4)}) == 4


def test_items_are_a_pure_function_of_seed_and_index(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1, 2, 3])
    a = _seg_dataset(cfg, episodes=2, devices_per_batch=2, segments_per_device=2)
    b = _seg_dataset(cfg, episodes=2, devices_per_batch=2, segments_per_device=2)
    for i in range(8):
        x, y = a.item_arrays(i), b.item_arrays(i)
        assert x["device_id"] == y["device_id"]
        assert np.array_equal(x["wet"], y["wet"])
    # A different seed must actually move the windows (else "fresh windows per
    # epoch" in the trainer is a no-op).
    a.seed = 99
    assert not np.array_equal(a.item_arrays(0)["wet"], b.item_arrays(0)["wet"])


def test_dry_and_wet_are_the_same_window(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1], identity=True)   # render == clean
    ds = _seg_dataset(cfg, episodes=2, devices_per_batch=2, segments_per_device=2)
    for i in range(8):
        it = ds.item_arrays(i)
        assert it["dry"].shape == it["wet"].shape == (24_000,)
        assert np.allclose(it["dry"], it["wet"], atol=1e-4)   # aligned, no offset
    item = ds[0]
    assert item["x"].shape == (2, 24_000)                     # (dry, wet) stacked
    assert torch.allclose(item["x"][0], item["x"][1], atol=1e-4)


def test_regions_split_train_and_val_in_time(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1], seconds=8.0)
    n = int(8.0 * 48_000)
    tr = _seg_dataset(cfg, episodes=4, devices_per_batch=2, segments_per_device=2,
                      regions=((0.0, 0.85),))
    va = _seg_dataset(cfg, episodes=4, devices_per_batch=2, segments_per_device=2,
                      regions=((0.85, 1.0),))
    assert tr.files[0]["spans"] == [(0, int(0.85 * n))]
    assert va.files[0]["spans"] == [(int(0.85 * n), n)]
    assert tr.files[0]["spans"][0][1] <= va.files[0]["spans"][0][0]   # no overlap
    with pytest.raises(ValueError, match="region must satisfy"):
        _seg_dataset(cfg, episodes=1, devices_per_batch=2, segments_per_device=2,
                     regions=((0.9, 0.1),))


def test_region_too_short_for_a_segment_is_an_error_not_a_silent_clamp(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1], seconds=4.0)
    with pytest.raises(RuntimeError, match="segment inside region"):
        _seg_dataset(cfg, episodes=1, devices_per_batch=2, segments_per_device=2,
                     segment_samples=24_000, regions=((0.99, 1.0),))


def test_interleaved_regions_draw_windows_from_every_block(tmp_path):
    """A multi-region dataset must actually use all its blocks, not just the first."""
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1], seconds=12.0)
    n = int(12.0 * 48_000)
    regions = ((0.0, 0.25), (0.5, 0.75))
    ds = _seg_dataset(cfg, episodes=40, devices_per_batch=2, segments_per_device=2,
                      segment_samples=24_000, regions=regions)
    assert len(ds.files[0]["spans"]) == 2
    rng = np.random.default_rng(0)
    starts = [ds._draw_window(rng)[1] for _ in range(200)]
    in_a = [c for c in starts if c < int(0.25 * n)]
    in_b = [c for c in starts if int(0.5 * n) <= c < int(0.75 * n)]
    assert len(in_a) + len(in_b) == len(starts)     # never lands between the blocks
    assert len(in_a) > 20 and len(in_b) > 20        # both blocks are really sampled


def test_short_blocks_are_dropped_not_clamped(tmp_path):
    """A block too short for a segment must vanish, never yield a truncated window."""
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1], seconds=12.0)
    ds = _seg_dataset(cfg, episodes=4, devices_per_batch=2, segments_per_device=2,
                      segment_samples=24_000, regions=((0.0, 0.25), (0.9, 0.92)))
    assert len(ds.files[0]["spans"]) == 1            # 0.24 s block cannot hold a 0.5 s segment
    for i in range(8):
        assert ds.item_arrays(i)["wet"].shape == (24_000,)


# --- The time-split policy ------------------------------------------------------
def test_strided_split_spreads_val_over_the_file_and_tail_does_not(tmp_path):
    """The reason the default is `strided`: a tail split samples one end only."""
    tr, va = time_split_regions(0.15, mode="tail")
    assert tr == ((0.0, 0.85),) and va == ((0.85, 1.0),)

    tr, va = time_split_regions(0.15, mode="strided", blocks=12)
    assert len(va) == 2                               # round(12 * 0.15) = 2 blocks
    assert va == ((3 / 12, 4 / 12), (9 / 12, 10 / 12))
    # Train covers the rest, contiguous runs merged, and the two sides never overlap.
    assert tr == ((0.0, 3 / 12), (4 / 12, 9 / 12), (10 / 12, 1.0))
    assert sum(b - a for a, b in tr) + sum(b - a for a, b in va) == pytest.approx(1.0)
    for a, b in va:
        assert all(b <= c or d <= a for c, d in tr)
    # Unlike the tail split, val samples both halves of the file.
    assert va[0][1] < 0.5 < va[1][0]


def test_level_check_catches_a_tail_split_on_a_decaying_signal(tmp_path):
    """The regression test for the real bug: a tail split validated 14 dB too quiet.

    Nothing about the *shapes* was wrong, which is why it went unnoticed — only the
    audio's level distribution differed, so the guard has to measure the audio.
    """
    pytest.importorskip("soundfile")
    from openamp.encoder.dataset import region_level_db

    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1], seconds=20.0, tail_quiet_db=20.0)
    kw = dict(episodes=2, devices_per_batch=2, segments_per_device=2,
              segment_samples=48_000)

    tr_r, va_r = time_split_regions(0.15, mode="tail")
    tail_gap = abs(region_level_db(_seg_dataset(cfg, regions=tr_r, **kw))
                   - region_level_db(_seg_dataset(cfg, regions=va_r, **kw)))
    tr_r, va_r = time_split_regions(0.15, mode="strided", blocks=10)
    strided_gap = abs(region_level_db(_seg_dataset(cfg, regions=tr_r, **kw))
                      - region_level_db(_seg_dataset(cfg, regions=va_r, **kw)))

    from openamp.encoder.train import SPLIT_LEVEL_WARN_DB

    assert tail_gap > SPLIT_LEVEL_WARN_DB       # the run header would warn, correctly
    assert strided_gap < SPLIT_LEVEL_WARN_DB    # ...and the interleaved split would not
    assert strided_gap < tail_gap / 3


def test_resume_is_refused_across_a_split_change_including_legacy_checkpoints():
    """A checkpoint predating the knob ran `tail`; resuming it as `strided` would
    silently redefine val_supcon and corrupt the best-val bookkeeping."""
    from openamp.encoder import train as et

    legacy = {"segment_samples": 262_144, "time_val_frac": 0.15}    # no split keys
    with pytest.raises(RuntimeError, match="time_split_mode 'tail' -> 'strided'"):
        et._check_resume_compatible(legacy, EncoderConfig(time_split_mode="strided"))
    # Same run, unchanged config: allowed.
    et._check_resume_compatible(legacy, EncoderConfig(time_split_mode="tail"))
    # And a block-count change is caught too, since it moves the val region.
    current = {"time_split_mode": "strided", "time_split_blocks": 10}
    with pytest.raises(RuntimeError, match="time_split_blocks"):
        et._check_resume_compatible(current, EncoderConfig(time_split_blocks=12))


def test_usable_region_bounds_the_pool_independently_of_where_val_falls():
    """Two separate concerns, and conflating them cost a training run.

    `usable_region` decides what audio the encoder may see at all; the split mode
    decides which part of that is val. The tail split got the pool right and the
    metric wrong; a plain strided split got the metric right and the pool wrong.
    """
    tr, va = time_split_regions(0.15, mode="strided", blocks=12, within=(0.0, 0.85))
    assert all(0.0 <= a < b <= 0.85 + 1e-9 for a, b in tr + va)     # nothing past the bound
    assert len(va) == 2 and va[0][1] < 0.5 < va[1][0]               # still interleaved
    assert sum(b - a for a, b in tr) + sum(b - a for a, b in va) == pytest.approx(0.85)
    # The bound applies to `tail` too, so the two knobs compose rather than clash.
    tr, va = time_split_regions(0.15, mode="tail", within=(0.0, 0.85))
    assert tr[0] == pytest.approx((0.0, 0.7225)) and va[0] == pytest.approx((0.7225, 0.85))
    # Whole file is the default and must stay a no-op.
    assert time_split_regions(0.15, mode="strided", blocks=12) == \
        time_split_regions(0.15, mode="strided", blocks=12, within=(0.0, 1.0))
    with pytest.raises(ValueError, match="usable_region"):
        time_split_regions(0.15, within=(0.9, 0.2))


def test_split_arithmetic_stays_valid_at_the_edges():
    for frac in (0.05, 0.15, 0.3, 0.5, 0.9):
        for blocks in (2, 3, 12, 31):
            tr, va = time_split_regions(frac, mode="strided", blocks=blocks)
            assert tr and va, (frac, blocks)          # neither side is ever empty
            assert sum(b - a for a, b in tr) + sum(b - a for a, b in va) \
                == pytest.approx(1.0)
    with pytest.raises(ValueError, match="time_split_blocks"):
        time_split_regions(0.15, mode="strided", blocks=1)
    with pytest.raises(ValueError, match="unknown time_split_mode"):
        time_split_regions(0.15, mode="middle")
    with pytest.raises(ValueError, match="time_val_frac"):
        time_split_regions(0.0)


def test_sources_filter_selects_the_sweep_file(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1])
    assert _seg_dataset(cfg, episodes=1, devices_per_batch=2,
                        segments_per_device=2).files[0]["file_id"] == "sweep_train"
    with pytest.raises(RuntimeError, match="sources"):
        _seg_dataset(cfg, episodes=1, devices_per_batch=2, segments_per_device=2,
                     sources=("egdb",))


def test_labels_use_tone_id_for_amps_and_ignore_unknown_gain(tmp_path):
    pytest.importorskip("soundfile")
    from openamp.core import constants as C
    from openamp.encoder.dataset import device_labels

    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=list(range(8)))
    lab = device_labels(cfg, list(range(8)))
    # tone_id = d // 2 in the fixture: 8 captures, 4 amps. That two-level structure
    # is the point — the amp probe must not be a restatement of the capture id.
    assert lab["n_tones"] == 4
    assert lab["tone_idx"][0] == lab["tone_idx"][1] != lab["tone_idx"][2]
    assert set(lab["gain_idx"].values()) <= set(range(len(C.GAIN_BUCKETS)))
    # An unknown bucket becomes -1 so CrossEntropyLoss(ignore_index=-1) drops it,
    # rather than inventing a fourth class from a keyword heuristic's failure.
    assert device_labels(cfg, [999])["gain_idx"][999] == -1


def test_crest_target_is_zero_for_an_identity_render(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1], identity=True)
    ds = _seg_dataset(cfg, episodes=2, devices_per_batch=2, segments_per_device=2)
    assert abs(ds.item_arrays(0)["crest"]) < 1e-3      # wet == dry -> no crest change


def test_crest_db_matches_hand_computation():
    from openamp.encoder.dataset import crest_db

    x = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)   # peak == rms
    assert crest_db(x) == pytest.approx(0.0, abs=1e-6)
    y = np.zeros(100, dtype=np.float32)
    y[0] = 1.0                                                # peak 1, rms 1/10
    assert crest_db(y) == pytest.approx(20.0, abs=1e-4)
    assert crest_db(np.zeros(10, dtype=np.float32)) == 0.0    # silence, no NaN


def test_augmentation_touches_only_the_wet_channel(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1], identity=True)
    plain = _seg_dataset(cfg, episodes=2, devices_per_batch=2, segments_per_device=2)
    noisy = _seg_dataset(cfg, episodes=2, devices_per_batch=2, segments_per_device=2,
                         augment=True, noise_p=1.0, noise_snr_db=(20.0, 20.0),
                         time_mask_p=0.0)
    a, b = plain.item_arrays(0), noisy.item_arrays(0)
    assert np.array_equal(a["dry"], b["dry"])          # the dry reference is a file
    assert not np.array_equal(a["wet"], b["wet"])      # noise is a recording artifact


def test_time_mask_zeroes_the_same_span_in_both_channels(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1], identity=True)
    ds = _seg_dataset(cfg, episodes=2, devices_per_batch=2, segments_per_device=2,
                      augment=True, noise_p=0.0, time_mask_p=1.0, time_mask_frac=0.1)
    it = ds.item_arrays(0)
    # Masking only one channel would fake a transfer relation no amp has, so the
    # zeroed samples must coincide exactly.
    assert np.array_equal(it["dry"] == 0.0, it["wet"] == 0.0)
    assert (it["wet"] == 0.0).sum() >= int(0.1 * len(it["wet"])) - 1


def test_capture_grid_replays_one_window_set_for_every_device(tmp_path):
    pytest.importorskip("soundfile")
    from openamp.emulate.dataset import build_device_index
    from openamp.encoder.dataset import CaptureGridDataset

    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1, 2], identity=False)
    ids, id_to_idx = build_device_index(cfg)
    ds = CaptureGridDataset(cfg, "train", id_to_idx=id_to_idx, segment_samples=24_000,
                            sources=("sweep",), n_windows=4, seed=7)
    assert len(ds) == 3 * 4
    items = [ds.item_arrays(i) for i in range(len(ds))]
    # Device-major: item i is (device i//n_windows, window i%n_windows).
    assert [it["device_id"] for it in items] == [0] * 4 + [1] * 4 + [2] * 4
    # Same audio for every device — otherwise capture vectors differ on window luck,
    # not on tone, and between-device retrieval is meaningless.
    for w in range(4):
        dry = [items[d * 4 + w]["dry"] for d in range(3)]
        assert np.array_equal(dry[0], dry[1]) and np.array_equal(dry[1], dry[2])
    # Windows within a device are actually different positions.
    assert not np.array_equal(items[0]["dry"], items[1]["dry"])


def test_dry_cache_returns_the_same_samples_as_seek_read(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=[0, 1])
    cached = _seg_dataset(cfg, episodes=2, devices_per_batch=2, segments_per_device=2,
                          cache_dry=True)
    direct = _seg_dataset(cfg, episodes=2, devices_per_batch=2, segments_per_device=2,
                          cache_dry=False)
    for i in range(8):
        assert np.array_equal(cached.item_arrays(i)["dry"], direct.item_arrays(i)["dry"])
    assert cached._dry_bytes > 0 and direct._dry_bytes == 0


# --- End to end: the loader yields what the loss expects ------------------------
def test_loader_batch_feeds_the_loss_and_gradients_reach_the_stem(tmp_path):
    pytest.importorskip("soundfile")
    from openamp.encoder.train import EncoderLoss, make_loader

    cfg = load_config(data_dir=tmp_path)
    _build_sweep_corpus(cfg, devices=list(range(6)), identity=False)
    ncfg = EncoderConfig(devices_per_batch=3, segments_per_device=2, num_workers=0,
                         **{k: list(v) if isinstance(v, tuple) else v
                            for k, v in SMALL.items()})
    ds = _seg_dataset(cfg, episodes=2, devices_per_batch=3, segments_per_device=2,
                      segment_samples=SMALL_T)
    batch = next(iter(make_loader(ds, ncfg, workers=0)))
    assert batch["x"].shape == (6, 2, SMALL_T)
    assert len(set(batch["device_idx"].tolist())) == 3

    model = ToneEncoder.from_config(ncfg)
    lossfn = EncoderLoss(ncfg, n_tones=3, n_gain=3)
    _, s, d = model(batch["x"])
    loss, parts = lossfn(s, d, *model.project(s, d), batch)
    assert torch.isfinite(loss)
    assert {"supcon_s", "supcon_d", "xcov", "amp_ce", "dyn", "loss"} <= set(parts)
    loss.backward()
    first_stem_conv = model.stem[0].conv.weight
    assert first_stem_conv.grad is not None and torch.isfinite(first_stem_conv.grad).all()
    assert float(first_stem_conv.grad.abs().sum()) > 0


def _tiny_encoder_cfg(**over):
    kw = dict(devices_per_batch=3, segments_per_device=2, episodes_per_epoch=1,
              segment_samples=SMALL_T, num_workers=0, holdout_frac=0.0,
              warmup_epochs=1, early_stop_patience=99)
    kw.update({k: list(v) if isinstance(v, tuple) else v for k, v in SMALL.items()})
    kw.update(over)
    return EncoderConfig(**kw)


def _scripted_val(monkeypatch, values):
    """Force `evaluate` to return a chosen val_supcon sequence, so checkpoint
    bookkeeping is tested independently of whatever the loss happens to do."""
    from openamp.encoder import train as et

    seq = iter(values)
    monkeypatch.setattr(et, "evaluate", lambda *a, **k: {
        "val_supcon": next(seq), "val_supcon_shuffled": 9.0,
        "val_amp_acc": 0.0, "val_crest_mse": 0.0})


def test_resume_carries_the_true_best_so_a_worse_epoch_cannot_overwrite_it(
        tmp_path, monkeypatch):
    """A resumed run must not overwrite a good checkpoint.pt with a worse epoch.

    Regression: last.pt was written *before* the best-val update, so it recorded the
    previous epoch's worse number. A resume then came back believing its best was
    that stale value, and the next mediocre-but-better-than-stale epoch clobbered a
    genuinely better checkpoint.pt.

    The val sequence is scripted because the bug only shows when the final epoch of
    the first session is an improvement — leaving that to the optimizer makes the
    test pass or fail on trajectory luck rather than on the bookkeeping.
    """
    pytest.importorskip("soundfile")
    from openamp.encoder import train as et

    cfg = load_config(data_dir=tmp_path)
    cfg.results_dir = tmp_path / "results"     # keep runs out of the real results tree
    _build_sweep_corpus(cfg, devices=list(range(6)), seconds=3.0, identity=False)
    cfg.encoder = _tiny_encoder_cfg(epochs=3)

    _scripted_val(monkeypatch, [3.0, 2.0, 1.0])       # best IS the last epoch
    et.train(cfg, name="r", device="cpu")
    run = cfg.encoder_run_dir("r")
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert last["best_val_supcon"] == pytest.approx(1.0)   # not the stale 2.0
    assert last["best_epoch"] == 2

    # Resume into an epoch that is worse than the true best but better than the
    # stale one. checkpoint.pt must survive untouched.
    _scripted_val(monkeypatch, [1.5])
    et.train(cfg, name="r", device="cpu", epochs=4, resume=True)
    best = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert best["best_val_supcon"] == pytest.approx(1.0)
    assert best["val_supcon"] == pytest.approx(1.0)        # 1.5 did NOT overwrite it


def test_best_checkpoint_tracks_the_minimum_across_epochs(tmp_path, monkeypatch):
    """checkpoint.pt is the best epoch, last.pt is the latest — even when they differ."""
    pytest.importorskip("soundfile")
    from openamp.encoder import train as et

    cfg = load_config(data_dir=tmp_path)
    cfg.results_dir = tmp_path / "results"     # keep runs out of the real results tree
    _build_sweep_corpus(cfg, devices=list(range(6)), seconds=3.0, identity=False)
    cfg.encoder = _tiny_encoder_cfg(epochs=3)
    _scripted_val(monkeypatch, [3.0, 1.0, 2.0])           # best is the MIDDLE epoch
    et.train(cfg, name="r2", device="cpu")

    run = cfg.encoder_run_dir("r2")
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 2 and last["val_supcon"] == pytest.approx(2.0)
    assert best["epoch"] == 1 and best["val_supcon"] == pytest.approx(1.0)
    assert last["best_val_supcon"] == best["best_val_supcon"] == pytest.approx(1.0)


def test_resume_refuses_a_changed_structural_knob(tmp_path, monkeypatch):
    """Loading old weights into a differently-shaped network must fail loudly."""
    pytest.importorskip("soundfile")
    from openamp.encoder import train as et

    cfg = load_config(data_dir=tmp_path)
    cfg.results_dir = tmp_path / "results"     # keep runs out of the real results tree
    _build_sweep_corpus(cfg, devices=list(range(6)), seconds=3.0, identity=False)
    cfg.encoder = _tiny_encoder_cfg(epochs=1)
    _scripted_val(monkeypatch, [3.0])
    et.train(cfg, name="r3", device="cpu")

    cfg.encoder = _tiny_encoder_cfg(epochs=2, head_dim=16)   # was 12
    _scripted_val(monkeypatch, [2.0])
    with pytest.raises(RuntimeError, match="structural knobs changed"):
        et.train(cfg, name="r3", device="cpu", resume=True)


def test_dyn_probe_none_drops_the_term(tmp_path):
    from openamp.encoder.train import EncoderLoss

    ncfg = EncoderConfig(dyn_probe="none", head_dim=12)
    lossfn = EncoderLoss(ncfg, n_tones=3, n_gain=3)
    s = F.normalize(torch.randn(4, 12), dim=-1)
    d = F.normalize(torch.randn(4, 12), dim=-1)
    batch = {"device_idx": torch.tensor([0, 0, 1, 1]),
             "tone_idx": torch.tensor([0, 0, 1, 1]),
             "gain_idx": torch.tensor([0, 0, 1, 1]),
             "crest": torch.zeros(4)}
    _, parts = lossfn(s, d, s, d, batch)
    assert parts["dyn"] == 0.0
