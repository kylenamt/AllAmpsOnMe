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
from openamp.emulate.wavenet import (A2_DILATIONS, A2_KERNEL_SIZES, FiLMWaveNet)


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
def _small_wavenet(**kw):
    """A shrunken schedule (R=18) so causality tests stay fast."""
    kw.setdefault("n_devices", 4)
    return FiLMWaveNet(channels=4, kernel_sizes=(3, 3, 3), dilations=(1, 2, 4),
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


def test_wavenet_conditioning_changes_output_and_only_through_film():
    torch.manual_seed(0)
    m = _small_wavenet().eval()
    x = torch.randn(1, 2000)
    # near-identity FiLM init still conditions (the sanity-#2 shuffle check needs it)
    assert not torch.allclose(m(x, torch.tensor([0])), m(x, torch.tensor([3])))
    # FiLM is the *only* path from the embedding: zeroed FiLM weights = unconditioned
    with torch.no_grad():
        for layer in m.layers:
            layer.film.weight.zero_()
    assert torch.allclose(m(x, torch.tensor([0])), m(x, torch.tensor([3])))


def test_wavenet_core_param_count_matches_a2_export():
    # Everything except the two one-to-many additions (embedding table + FiLMs)
    # must count exactly what an A2 capture exports: 12,146 weights incl.
    # head_scale — which is a fixed buffer here (trained, it collapses to the
    # silence solution), so the trainable core is one less.
    m = FiLMWaveNet(n_devices=10)
    film = sum(p.numel() for layer in m.layers for p in layer.film.parameters())
    core = count_parameters(m) - m.embedding.weight.numel() - film
    assert core == 12_145
    assert not m.head_scale.requires_grad
    assert float(m.head_scale) == pytest.approx(0.02)
    wider = FiLMWaveNet(n_devices=10, channels=16)
    assert count_parameters(wider) > count_parameters(m)


def test_build_model_dispatches_on_arch():
    assert isinstance(build_model(EmulateConfig(), 5), FiLMTCN)
    assert isinstance(build_model(EmulateConfig(arch="film_wavenet"), 5), FiLMWaveNet)
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


def test_wavenet_matches_nam_a2_reference():
    """Weight-for-weight against NAM's own WaveNet: with FiLM at exact identity,
    our causal net must reproduce NAM's valid-conv output sample-for-sample."""
    _wavenet = pytest.importorskip("nam.models.wavenet")

    torch.manual_seed(0)
    nam_net = _wavenet.WaveNet.init_from_config({
        "layers_configs": [{
            "input_size": 1, "condition_size": 1, "channels": 8,
            "kernel_sizes": list(A2_KERNEL_SIZES), "dilations": list(A2_DILATIONS),
            "activation": {"name": "LeakyReLU", "negative_slope": 0.01},
            "head": {"out_channels": 1, "kernel_size": 16, "bias": True},
        }],
        "head": None, "head_scale": 0.02,
    }).eval()
    ours = FiLMWaveNet(n_devices=1).eval()

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
            mine.film.weight.zero_()                  # exact identity FiLM
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
