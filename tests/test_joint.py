"""Joint encoder + FiLM generator: the encoder's shapes and non-causality, the
generator's new embedding-vector path, and the dataset's A/B disjointness.

torch-guarded like the other model tests; the dataset cases additionally need
soundfile (they write tiny real FLACs in tmp_path). Architectures are shrunken but
real — no mocks — so an assertion about centered padding or masked pooling is an
assertion about the code that actually runs.

The load-bearing tests here are the ones about *disjointness* and *gradient flow*:
if window B could overlap window A the embedding would carry content, and if the
gradient did not reach the encoder it would never learn at all. Both failures are
silent in the training loss.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openamp.core.config import load_config  # noqa: E402
from openamp.spec import (DataSpec, EncoderSpec, JointSpec,  # noqa: E402
                          LossSpec, ModelSpec, RefSpec, RunSpec, TrainingSpec)
from openamp.models import FiLMWaveNet  # noqa: E402
from openamp.models.joint import JointModel, build_joint  # noqa: E402
from openamp.encoders import AttentiveStatsPool, WaveNetEncoder  # noqa: E402

# A shrunken encoder schedule: same structure, tiny receptive field, fast.
SMALL = dict(kernel_sizes=(3, 3, 3), dilations=(1, 2, 4), channels=4,
             attn_hidden=8, proj_hidden=8)


def _jspec(tmp_path, *, model=None, encoder=None, ref=None, **training):
    """A joint RunSpec for the tests that need a whole experiment."""
    return RunSpec(
        name="run", data_dir=tmp_path / "data", results_dir=tmp_path / "results",
        seed=1234,
        model=model or ModelSpec("film_wavenet", {"channels": 4, "embedding_dim": 16}),
        data=DataSpec(dataset="joint", clip_seconds=training.pop("clip_seconds", 2.0)),
        loss=LossSpec(kwargs={"preemph_coeff": 0.85, "stft_weight": 1.0}),
        training=TrainingSpec(num_workers=0, **training),
        joint=JointSpec(encoder=encoder or EncoderSpec(
                            "wavenet", {"channels": 4, "attn_hidden": 4,
                                        "proj_hidden": 4, "activation": "tanh"}),
                        ref=ref or RefSpec()))


# --- Encoder -------------------------------------------------------------------
def test_encoder_maps_any_reference_length_to_one_vector():
    m = WaveNetEncoder(embedding_dim=16, **SMALL)
    for length in (2048, 4096, 7777):            # pooling is length-invariant
        e = m(torch.randn(3, 2, length))
        assert e.shape == (3, 16)
        assert torch.isfinite(e).all()


def test_encoder_receptive_field_has_no_head_term():
    # RF = 1 + sum((k-1)*d); the generator's adds (head_kernel - 1) on top.
    assert WaveNetEncoder.compute_receptive_field((3, 3, 3), (1, 2, 4)) == 1 + 2 * 7
    assert WaveNetEncoder(embedding_dim=8, **SMALL).receptive_field == 15


def test_encoder_rejects_the_wrong_channel_count():
    m = WaveNetEncoder(embedding_dim=8, **SMALL)
    with pytest.raises(ValueError, match="channels"):
        m(torch.randn(2, 3, 1024))               # 3 channels, not (dry, wet)


def test_encoder_is_non_causal():
    """A perturbation late in the reference must reach an *earlier* frame.

    This is the property that distinguishes the encoder from the generator it
    feeds: the reference is offline, so a centered conv is allowed and sees both
    sides. With one k=3 d=1 layer the reach is exactly one frame either way.
    """
    m = WaveNetEncoder(embedding_dim=4, kernel_sizes=(3,), dilations=(1,), channels=4,
                       attn_hidden=4, proj_hidden=4).eval()
    x = torch.zeros(1, 2, 32)
    with torch.no_grad():
        base = m.encode_frames(x)
        x[0, 0, 16] = 1.0                        # perturb one sample at t=16
        bumped = m.encode_frames(x)
    delta = (bumped - base).abs().sum(dim=1)[0]  # per-frame change
    assert delta[15] > 0, "a causal encoder would leave the earlier frame untouched"
    assert delta[17] > 0
    assert delta[13] == 0, "reach should stop at one frame for k=3, d=1"


def test_l2_toggle_controls_the_embedding_norm():
    kw = dict(embedding_dim=16, **SMALL)
    raw = WaveNetEncoder(normalize=False, **kw)(torch.randn(4, 2, 2048))
    unit = WaveNetEncoder(normalize=True, **kw)(torch.randn(4, 2, 2048))
    assert torch.allclose(unit.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert not torch.allclose(raw.norm(dim=-1), torch.ones(4), atol=1e-3)


def test_grad_checkpointing_changes_nothing_but_memory():
    torch.manual_seed(0)
    plain = WaveNetEncoder(embedding_dim=8, grad_checkpoint=False, **SMALL)
    torch.manual_seed(0)
    ckpt = WaveNetEncoder(embedding_dim=8, grad_checkpoint=True, **SMALL)
    ckpt.load_state_dict(plain.state_dict())
    x = torch.randn(2, 2, 2048)
    plain.train(); ckpt.train()
    assert torch.allclose(plain(x), ckpt(x), atol=1e-6)


# --- Attentive stats pooling ---------------------------------------------------
def test_uniform_attention_reduces_to_plain_mean_and_std():
    """With zeroed scores every frame is weighted equally, so the pooled stats
    must equal the ordinary (population) mean and std — the sanity anchor for the
    weighted formula."""
    pool = AttentiveStatsPool(channels=5, hidden=4)
    for p in pool.score.parameters():
        torch.nn.init.zeros_(p)
    h = torch.randn(3, 5, 64)
    out = pool(h)
    assert torch.allclose(out[:, :5], h.mean(dim=-1), atol=1e-5)
    assert torch.allclose(out[:, 5:], h.std(dim=-1, unbiased=False), atol=1e-4)


def test_mask_excludes_frames_from_both_moments():
    """The mask is applied to the attention logits, so a masked frame gets weight
    exactly zero — mean *and* std. Masking the moments after the fact is the
    version that silently leaves the std unmasked."""
    pool = AttentiveStatsPool(channels=3, hidden=4)
    for p in pool.score.parameters():
        torch.nn.init.zeros_(p)
    h = torch.randn(1, 3, 40)
    mask = torch.zeros(1, 40, dtype=torch.bool)
    mask[0, :10] = True
    out = pool(h, mask)
    valid = h[..., :10]
    assert torch.allclose(out[:, :3], valid.mean(dim=-1), atol=1e-5)
    assert torch.allclose(out[:, 3:], valid.std(dim=-1, unbiased=False), atol=1e-4)
    # And the junk beyond the mask really is ignored.
    h2 = h.clone()
    h2[..., 10:] = 1e3
    assert torch.allclose(pool(h2, mask), out, atol=1e-4)


def test_mask_length_must_match_the_features():
    pool = AttentiveStatsPool(channels=2, hidden=4)
    with pytest.raises(ValueError, match="mask length"):
        pool(torch.randn(1, 2, 40), torch.ones(1, 33, dtype=torch.bool))


# --- Multitap head -------------------------------------------------------------
# The whole point of this head is that pooled width stops tracking backbone width,
# so most of these assert on `pooled_dim` rather than on the embedding: the
# embedding was never the thing that was too narrow.
MULTITAP = dict(SMALL, head="multitap", taps=(0, 1, 2), tap_stride=2,
                expand_dim=8, n_heads=2)


def test_multitap_pooled_width_is_independent_of_backbone_width():
    """K * 2 * E, whatever `channels` is — the bottleneck this head removes."""
    for channels in (4, 16):
        m = WaveNetEncoder(embedding_dim=16, **dict(MULTITAP, channels=channels))
        assert m.pooled_dim == 2 * 2 * 8 == 32
        assert m.capacity_report()["pooled_dim"] == 32
    # ...and the old head's is exactly 2C, which is the thing being escaped.
    assert WaveNetEncoder(embedding_dim=16, **SMALL).pooled_dim == 2 * 4


def test_multitap_capacity_report_names_the_real_ceiling():
    m = WaveNetEncoder(embedding_dim=16, **dict(MULTITAP, expand_dim=64))
    cap = m.capacity_report()
    assert cap == {"head": "multitap", "pooled_dim": 256, "embedding_dim": 16,
                   "ceiling": 16}
    # Pooled well above De is the configuration we want: the projection is then
    # the deliberate bottleneck, not the pool.
    assert cap["pooled_dim"] > cap["embedding_dim"]


def test_multitap_maps_any_reference_length_to_one_vector():
    m = WaveNetEncoder(embedding_dim=16, normalize=True, **MULTITAP).eval()
    for length in (2048, 4096, 7777):            # incl. a length tap_stride splits
        e = m(torch.randn(3, 2, length))
        assert e.shape == (3, 16)
        assert torch.isfinite(e).all()
        assert torch.allclose(e.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_multitap_reads_every_tap():
    """Perturbing the trunk at one tap must move `e`. If the concat silently
    dropped a tap, the extra depth would be costing memory for nothing."""
    m = WaveNetEncoder(embedding_dim=8, **MULTITAP).eval()
    x = torch.randn(1, 2, 512)
    with torch.no_grad():
        base = m(x)
        for i in range(len(m.layers)):
            saved = m.layers[i].layer1x1.weight.detach().clone()
            m.layers[i].layer1x1.weight.mul_(1.5)
            assert not torch.allclose(m(x), base, atol=1e-6), f"tap {i} unused"
            m.layers[i].layer1x1.weight.copy_(saved)


def test_multitap_rejects_taps_outside_the_layer_array():
    with pytest.raises(ValueError, match="enc_taps"):
        WaveNetEncoder(embedding_dim=8, **dict(MULTITAP, taps=(0, 99)))
    with pytest.raises(ValueError, match="enc_taps"):
        WaveNetEncoder(embedding_dim=8, **dict(MULTITAP, taps=()))


def test_multitap_mask_excludes_padded_frames():
    """Junk beyond the mask must not reach `e`.

    In eval() mode: BatchNorm then uses fixed running stats, so each frame is
    normalized independently and the pool's -inf masking is the only thing that
    decides what counts. In train() mode the batch statistics would see the junk —
    see MultiTapStatsHead's docstring.
    """
    m = WaveNetEncoder(embedding_dim=8, **dict(MULTITAP, tap_stride=1)).eval()
    x = torch.randn(2, 2, 256)
    mask = torch.zeros(2, 256, dtype=torch.bool)
    mask[:, :100] = True
    with torch.no_grad():
        base = m(x, mask)
        x2 = x.clone()
        x2[..., 140:] = 1e3          # clear of the taps' reach into valid frames
        assert torch.allclose(m(x2, mask), base, atol=1e-4)


def test_multitap_mask_is_pooled_alongside_the_features():
    """With tap_stride > 1 the mask has to be decimated too, or it desynchronizes
    from the features and the length check fires."""
    m = WaveNetEncoder(embedding_dim=8, **dict(MULTITAP, tap_stride=4)).eval()
    mask = torch.zeros(2, 256, dtype=torch.bool)
    mask[:, :128] = True
    e = m(torch.randn(2, 2, 256), mask)
    assert e.shape == (2, 8) and torch.isfinite(e).all()
    with pytest.raises(ValueError, match="mask length"):
        m(torch.randn(2, 2, 256), torch.ones(2, 33, dtype=torch.bool))


def test_multitap_expand_norm_options_all_build_and_run():
    for norm in ("batch", "group", "none"):
        m = WaveNetEncoder(embedding_dim=8, **dict(MULTITAP, expand_norm=norm))
        assert torch.isfinite(m(torch.randn(2, 2, 512))).all()
    with pytest.raises(ValueError, match="enc_expand_norm"):
        WaveNetEncoder(embedding_dim=8, **dict(MULTITAP, expand_norm="layer"))


def test_multitap_derives_proj_hidden_from_the_pooled_width():
    """proj_hidden 0 means "a quarter of whatever pooling emits" — the only
    default that still makes sense once pooled_dim is not a fixed 2*enc_channels."""
    m = WaveNetEncoder(embedding_dim=16, **dict(MULTITAP, proj_hidden=0))
    assert m.head.proj[0].in_features == m.pooled_dim == 32
    assert m.head.proj[0].out_features == 32 // 4
    explicit = WaveNetEncoder(embedding_dim=16, **dict(MULTITAP, proj_hidden=7))
    assert explicit.head.proj[0].out_features == 7


def test_multitap_gradient_reaches_the_first_layer():
    torch.manual_seed(0)
    model = JointModel(WaveNetEncoder(embedding_dim=8, **MULTITAP), _small_generator())
    model.generator.embedding.weight.requires_grad_(False)
    out = model(torch.randn(2, 512), torch.randn(2, 2, 1024))
    out.pow(2).mean().backward()
    first_conv = model.encoder.layers[0].conv.weight
    assert first_conv.grad is not None and first_conv.grad.abs().sum() > 0, \
        "no gradient reached the encoder through the multitap head"


def test_stats_head_is_still_the_default():
    """Existing joint checkpoints load under the default, so it must not move."""
    m = WaveNetEncoder(embedding_dim=8, **SMALL)
    assert m.head_kind == "stats"
    assert hasattr(m, "pool") and not hasattr(m, "head")
    assert WaveNetEncoder(embedding_dim=8).head_kind == "stats"


# --- Generator: the new embedding-vector path ----------------------------------
def _small_generator(**kw):
    return FiLMWaveNet(n_devices=4, channels=4, embedding_dim=8,
                       kernel_sizes=(3, 3), dilations=(1, 2), head_kernel=2, **kw)


def test_forward_emb_is_exactly_the_table_forward():
    """``forward`` must be ``forward_emb`` behind a lookup — not a parallel copy
    that can drift. Bit-identical, not merely close."""
    torch.manual_seed(0)
    m = _small_generator().eval()
    x = torch.randn(3, 512)
    idx = torch.tensor([0, 2, 1])
    with torch.no_grad():
        assert torch.equal(m(x, idx), m.forward_emb(x, m.embedding(idx)))


def test_forward_emb_carries_gradient_into_the_embedding():
    m = _small_generator()
    e = torch.randn(2, 8, requires_grad=True)
    m.forward_emb(torch.randn(2, 256), e).sum().backward()
    assert e.grad is not None and e.grad.abs().sum() > 0


def test_export_reference_forward_matches_the_model():
    """export.forward_with_embedding is now a wrapper, so it cannot drift."""
    from openamp.export import forward_with_embedding

    m = _small_generator().eval()
    x = torch.randn(2, 256)
    e = torch.randn(8)
    with torch.no_grad():
        want = m.forward_emb(x, e.reshape(1, -1).expand(2, -1))
    assert torch.equal(forward_with_embedding(m, x, e), want)


def test_joint_model_refuses_a_width_mismatch():
    with pytest.raises(ValueError, match="embedding width mismatch"):
        JointModel(WaveNetEncoder(embedding_dim=7, **SMALL), _small_generator())


def test_joint_forward_and_gradients_reach_both_halves():
    torch.manual_seed(0)
    model = JointModel(WaveNetEncoder(embedding_dim=8, **SMALL), _small_generator())
    model.generator.embedding.weight.requires_grad_(False)
    out = model(torch.randn(2, 512), torch.randn(2, 2, 1024))
    assert out.shape == (2, 1, 512)
    out.pow(2).mean().backward()

    first_conv = model.encoder.layers[0].conv.weight
    assert first_conv.grad is not None and first_conv.grad.abs().sum() > 0, \
        "no gradient reached the encoder — it would never learn"
    assert model.generator.rechannel.weight.grad.abs().sum() > 0
    # The lookup table is the thing the encoder replaces; it must not train.
    assert model.generator.embedding.weight.grad is None


def test_build_joint_freezes_the_generator_table(tmp_path):
    m = build_joint(_jspec(tmp_path), 5)
    assert not m.generator.embedding.weight.requires_grad
    assert m.encoder.embedding_dim == m.generator.embedding_dim == 16


# --- Config --------------------------------------------------------------------
def test_encoder_validates_its_numbers():
    """Under v2 the encoder section is the constructor's own arguments, so these
    are the constructor's checks rather than a separate config-layer copy."""
    small_taps = {"head": "multitap", "taps": (0, 1)}
    for kw, msg in [({"head": "nope"}, "enc_head"),
                    ({**small_taps, "taps": ()}, "enc_taps"),
                    ({**small_taps, "taps": (99,)}, "enc_taps"),
                    ({**small_taps, "expand_norm": "layer"}, "enc_expand_norm"),
                    ({"activation": "relu"}, "enc_activation")]:
        with pytest.raises(ValueError, match=msg):
            WaveNetEncoder(embedding_dim=8, **{**SMALL, **kw})
    # 0 is legal for proj_hidden — it is the "derive it" sentinel, not a mistake.
    enc = WaveNetEncoder(embedding_dim=8, **{**SMALL, "proj_hidden": 0})
    assert enc.proj[0].out_features == max(1, SMALL["channels"] // 2)


def test_joint_section_round_trips_through_yaml(tmp_path):
    import yaml

    from openamp.spec import load_run_spec

    p = tmp_path / "j.yaml"
    p.write_text(yaml.safe_dump(
        {"model": {"arch": "film_wavenet"},
         "joint": {"encoder": {"kind": "wavenet", "channels": 24},
                   "ref": {"seconds": 3.0}}}))
    spec = load_run_spec(p)
    assert spec.joint.encoder.kwargs["channels"] == 24
    assert spec.joint.ref.seconds == 3.0
    assert spec.joint.enc_lr == JointSpec().enc_lr      # omitted keys keep defaults
    assert spec.name == "j"                             # still the file stem


# --- Dataset -------------------------------------------------------------------
def _build_corpus(cfg, devices, *, n_files=4, seconds=3.0, sr=48_000, splits=("train",)):
    """A multi-file corpus, so different-file references have somewhere to go.

    Every split is written in one call: the manifests are single files that
    :func:`write_manifest` replaces wholesale, so building them one split at a
    time would leave only the last one on disk.

    The signal stays inside +-0.95 because the FLACs are PCM_24, which clips at
    +-1.0 — an unclipped fixture would put a different dry on disk than the one
    the renders were computed from, and every (dry, wet) assertion would be
    comparing the wrong pair. The real corpus is normalized to -18 dBFS RMS with
    a -1 dBFS peak ceiling, so it never lands near this.
    """
    import pandas as pd

    from openamp.core import constants as C
    from openamp.core import manifest as manifests
    from openamp.dsp import audio as audio_io

    cfg.ensure_dirs()
    n = int(seconds * sr)
    rng = np.random.default_rng(0)
    corpus_rows, render_rows, man_rows = [], [], []
    for split in splits:
        cdir = cfg.clean_split_dir(split)
        cdir.mkdir(parents=True, exist_ok=True)
        sigs = {}
        for f in range(n_files):
            fid = f"egdb_{split[:2]}{f:04d}"
            sig = np.clip(0.3 * rng.standard_normal(n), -0.95, 0.95).astype(np.float32)
            sigs[fid] = sig
            audio_io.write_flac(cdir / f"{fid}.{cfg.output_format}", sig, sr)
            corpus_rows.append({"file_id": fid, "split": split, "source": C.SOURCE_EGDB,
                                "orig_path": "", "duration_s": seconds,
                                "applied_gain_db": 0.0, "sha256": ""})
        for d in devices:
            rdir = cfg.device_render_dir(d)
            rdir.mkdir(parents=True, exist_ok=True)
            for fid, sig in sigs.items():
                wet = np.tanh(sig * (1.0 + d)).astype(np.float32)   # device character
                rp = rdir / f"{fid}.{cfg.output_format}"
                audio_io.write_flac(rp, wet, sr)
                render_rows.append({"device_id": d, "file_id": fid, "split": split,
                                    "path": str(rp), "status": C.RENDER_OK,
                                    "output_scale": 1.0})
    for d in devices:
        man_rows.append({"device_id": d, "tone_id": d // 2,
                         "gain_bucket": C.GAIN_BUCKETS[d % 3]})
    manifests.write_manifest(pd.DataFrame(corpus_rows, columns=manifests.CORPUS_COLUMNS),
                             cfg.corpus_manifest_path)
    manifests.write_manifest(
        manifests.ensure_schema(pd.DataFrame(render_rows), manifests.RENDERS_COLUMNS),
        cfg.renders_manifest_path)
    manifests.write_manifest(
        manifests.ensure_schema(pd.DataFrame(man_rows), manifests.ALL_COLUMNS),
        cfg.manifest_path)


def _joint_dataset(cfg, **kw):
    from openamp.data import CorpusPaths, build_device_index
    from openamp.data import JointDataset

    _, id_to_idx = build_device_index(CorpusPaths.from_config(cfg))
    kw.setdefault("receptive_field", 64)
    kw.setdefault("clip_samples", 8_000)
    kw.setdefault("ref_samples", 8_000)
    return JointDataset(CorpusPaths.from_config(cfg), "train", id_to_idx=id_to_idx, seed=7, **kw)


def test_reference_comes_from_a_different_file(tmp_path):
    """The A/B barrier. If this ever regresses, the embedding can carry content
    and every downstream leakage number becomes meaningless."""
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_corpus(cfg, devices=list(range(4)), n_files=5)
    ds = _joint_dataset(cfg)
    assert not ds.same_file_mode
    for i in range(60):
        rng = np.random.default_rng((ds.seed, i))
        d = int(rng.integers(0, len(ds.device_ids)))
        file = ds.files[int(rng.integers(0, len(ds.files)))]
        c = int(rng.integers(0, file["n"] - ds.clip_samples + 1))
        ref_file, _ = ds._draw_reference(file, c, rng)
        assert ref_file["file_id"] != file["file_id"]


def test_every_other_file_is_reachable_as_a_reference(tmp_path):
    """The index-shifting draw must stay uniform over the other files, not skip one."""
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_corpus(cfg, devices=[0], n_files=4)
    ds = _joint_dataset(cfg)
    target = ds.files[0]
    rng = np.random.default_rng(0)
    seen = {ds._draw_reference(target, 0, rng)[0]["file_id"] for _ in range(200)}
    assert seen == {f["file_id"] for f in ds.ref_files} - {target["file_id"]}


def test_same_file_reference_clears_the_window_and_the_gap(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_corpus(cfg, devices=[0, 1], n_files=1, seconds=6.0)
    gap = 4_000
    ds = _joint_dataset(cfg, ref_min_gap=gap)
    assert ds.same_file_mode                     # one file -> no other file to use
    rng = np.random.default_rng(0)
    file = ds.files[0]
    for _ in range(200):
        c = int(rng.integers(0, file["n"] - ds.clip_samples + 1))
        start = ds._draw_same_file_start(file, c, rng)
        # A's read span is [c - receptive_field, c + clip); B must clear it by `gap`.
        assert (start + ds.ref_samples + gap <= c - ds.receptive_field
                or start >= c + ds.clip_samples + gap)
        assert 0 <= start <= file["n"] - ds.ref_samples


def test_same_file_mode_drops_files_that_cannot_hold_both(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_corpus(cfg, devices=[0], n_files=1, seconds=0.6)
    with pytest.raises(RuntimeError, match="Same-file references need"):
        _joint_dataset(cfg, ref_different_file=False, ref_min_gap=4_000)


def test_items_are_a_pure_function_of_seed_and_index(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_corpus(cfg, devices=list(range(3)), n_files=4)
    a, b = _joint_dataset(cfg), _joint_dataset(cfg)
    for i in (0, 5, 19):
        x, y = a[i], b[i]
        for k in ("input", "target", "ref", "device_idx"):
            assert torch.equal(x[k], y[k]), k


def test_item_shapes_and_reference_channel_order(tmp_path):
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_corpus(cfg, devices=[0, 1], n_files=3)
    ds = _joint_dataset(cfg, receptive_field=64, clip_samples=8_000, ref_samples=6_000)
    it = ds[0]
    assert it["input"].shape == (64 + 8_000,)
    assert it["target"].shape == (8_000,)
    assert it["ref"].shape == (2, 6_000)
    # Channel 0 is the dry source, which the corpus SHARES across every device;
    # channel 1 is this device's render. So the same window read for two devices
    # must agree on channel 0 and differ on channel 1 — that asymmetry is the
    # whole reason the encoder cannot identify an amp by recognizing the music.
    a = ds._read_reference(int(ds.device_ids[0]), ds.files[0], 0)
    b = ds._read_reference(int(ds.device_ids[1]), ds.files[0], 0)
    assert np.array_equal(a[0], b[0])
    assert not np.allclose(a[1], b[1])


def test_reference_reads_the_right_device(tmp_path):
    """A reference must be *this* amp's render, not another's — otherwise every
    example is silently mislabelled and the encoder learns nothing."""
    pytest.importorskip("soundfile")
    cfg = load_config(data_dir=tmp_path)
    _build_corpus(cfg, devices=[0, 3], n_files=2)
    ds = _joint_dataset(cfg)
    file = ds.files[0]
    for d in ds.device_ids:
        ref = ds._read_reference(int(d), file, 0)
        dry = ref[0]
        assert np.allclose(ref[1], np.tanh(dry * (1.0 + d)), atol=1e-4)


# --- End to end ----------------------------------------------------------------
def test_train_writes_a_run_and_logs_the_tripwire(tmp_path):
    """A whole tiny run: the loop, the per-epoch collapse tripwire, checkpoints."""
    pytest.importorskip("soundfile")
    import csv
    import json

    from openamp.training import joint as joint_train
    from openamp.training.joint import LOG_COLUMNS

    spec = _jspec(tmp_path, clip_seconds=0.1, batch_size=4, epochs=2, val_pairs=8,
                  amp=False, log_every=1,
                  ref=RefSpec(seconds=0.1)).replace(name="t")
    spec = spec.replace(data=DataSpec(dataset="joint", clip_seconds=0.1,
                                      pairs_per_epoch=16, holdout_frac=0.25))
    _build_corpus(load_config(data_dir=spec.data_dir), devices=list(range(4)),
                  n_files=3, splits=("train", "val"))

    m = joint_train.train(spec, device="cpu")
    run = spec.run_dir
    assert (run / "checkpoint.pt").is_file() and (run / "last.pt").is_file()
    assert (run / "resolved_config.yaml").is_file()
    assert (run / "config_history.jsonl").is_file()
    assert json.loads((run / "metrics.json").read_text())["embedding_dim"] == 16
    assert m["epochs_run"] == 2 and m["n_devices"] == 3 and m["n_holdout"] == 1

    with (run / "train_log.csv").open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == LOG_COLUMNS
    assert len(rows) == 3                        # header + 2 epochs
    # The tripwire columns must actually be populated, not left blank.
    for r in rows[1:]:
        assert float(r[LOG_COLUMNS.index("val_esr_shuffled")]) >= 0
        assert float(r[LOG_COLUMNS.index("emb_spread")]) >= 0


def test_resume_is_refused_when_the_encoder_shape_changed(tmp_path):
    from openamp.training.joint import _check_resume_compatible

    def spec(**enc):
        return _jspec(tmp_path, encoder=EncoderSpec("wavenet", enc))

    base = {"channels": 16, "normalize": False}
    ck = {"spec": spec(**base).to_dict()}
    _check_resume_compatible(ck, spec(**base), None, None)
    with pytest.raises(RuntimeError, match="channels"):
        _check_resume_compatible(ck, spec(**{**base, "channels": 32}), None, None)
    # Parameter-free but semantics-changing: must be refused too.
    with pytest.raises(RuntimeError, match="normalize"):
        _check_resume_compatible(ck, spec(**{**base, "normalize": True}), None, None)


def test_resume_is_refused_when_the_head_changed(tmp_path):
    """The head knobs postdate every checkpoint written so far, so `k in saved` is
    false for all of them — without the legacy defaults the guard would wave a head
    swap straight through into a strict load_state_dict failure far from the cause.
    """
    from openamp.training.joint import _check_resume_compatible

    def spec(**enc):
        return _jspec(tmp_path, encoder=EncoderSpec("wavenet", {"channels": 16, **enc}))

    ck = {"spec": spec().to_dict()}                   # pre-multitap checkpoint
    _check_resume_compatible(ck, spec(), None, None)
    for kw, msg in [({"head": "multitap"}, "head"),
                    ({"taps": [1, 2]}, "taps"),
                    ({"expand_dim": 128}, "expand_dim"),
                    ({"n_heads": 4}, "n_heads"),
                    ({"expand_norm": "group"}, "expand_norm")]:
        with pytest.raises(RuntimeError, match=msg):
            _check_resume_compatible(ck, spec(**kw), None, None)


def test_resume_tolerates_a_yaml_list_tuple_round_trip(tmp_path):
    """enc_taps survives torch.save/yaml as either list or tuple; the guard must
    not read that as a structural change and refuse a legitimate resume."""
    from openamp.training.joint import _check_resume_compatible

    saved = _jspec(tmp_path, encoder=EncoderSpec(
        "wavenet", {"channels": 16, "taps": (6, 13, 22)}))
    ck = {"spec": saved.to_dict()}
    _check_resume_compatible(ck, _jspec(tmp_path, encoder=EncoderSpec(
        "wavenet", {"channels": 16, "taps": [6, 13, 22]})), None, None)


# --- Fingerprint conditioning ---------------------------------------------------
def _write_fingerprints(tmp_path, device_ids, *, dim=32, seed=0):
    """A synthetic fingerprint run dir: pooled.npz with a dry row, plus meta.json."""
    import json

    run = tmp_path / "fp" / "sweep_train"
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = np.asarray([-1] + [int(d) for d in device_ids], dtype=np.int64)
    pooled = rng.standard_normal((len(ids), dim)).astype(np.float32)
    np.savez(run / "pooled.npz", device_ids=ids, pooled=pooled)
    (run / "meta.json").write_text(json.dumps({"encoder": "synthetic", "latent_dim": dim}))
    return run, ids, pooled


def _fp_encoder(tmp_path, device_ids, *, embedding_dim=16, dim=32, **kw):
    from openamp.encoders import FingerprintEncoder

    run, _, _ = _write_fingerprints(tmp_path, device_ids, dim=dim)
    ids, pooled, dry, _ = FingerprintEncoder.load_pooled(run)
    return FingerprintEncoder(embedding_dim=embedding_dim, fingerprints=pooled,
                              device_ids=ids, dry=dry, **kw), run


def test_fingerprint_encoder_maps_device_ids_to_vectors(tmp_path):
    enc, _ = _fp_encoder(tmp_path, [0, 3, 7], embedding_dim=16)
    e = enc(torch.tensor([0, 7, 3, 0], dtype=torch.long))
    assert e.shape == (4, 16) and torch.isfinite(e).all()
    # normalize defaults on: e lands on the unit sphere.
    assert torch.allclose(e.norm(dim=-1), torch.ones(4), atol=1e-5)
    # The same id twice gives the same vector — conditioning is deterministic.
    assert torch.equal(e[0], e[3])


def test_fingerprints_are_frozen_and_only_the_adapter_trains(tmp_path):
    enc, _ = _fp_encoder(tmp_path, [0, 1, 2])
    before = enc.fingerprints.clone()
    names = {n for n, _ in enc.named_parameters()}
    assert names and all(n.startswith("proj.") for n in names), names

    opt = torch.optim.SGD(enc.parameters(), lr=1.0)
    enc(torch.tensor([0, 1, 2], dtype=torch.long)).sum().backward()
    opt.step()
    # The codec's output is a buffer: no gradient, and untouched by an optimizer step.
    assert enc.fingerprints.grad is None
    assert torch.equal(enc.fingerprints, before)


def test_unknown_device_id_raises_instead_of_reading_another_amp(tmp_path):
    """A negative or absent id must not silently index some other amp's row."""
    enc, _ = _fp_encoder(tmp_path, [0, 1, 5])
    for bad in ([2], [99], [-1]):
        with pytest.raises(RuntimeError, match="no fingerprint"):
            enc(torch.tensor(bad, dtype=torch.long))


def test_holdout_ids_resolve_to_their_own_fingerprints(tmp_path):
    """Rows are keyed by device id, not by a run's training index.

    This is what makes zero-shot evaluation possible: an amp absent from the run's
    id_to_idx must still return *its own* vector.
    """
    from openamp.encoders import FingerprintEncoder

    run, ids, pooled = _write_fingerprints(tmp_path, [0, 1, 2, 3], dim=8)
    fids, fp, dry, _ = FingerprintEncoder.load_pooled(run)
    enc = FingerprintEncoder(embedding_dim=4, fingerprints=fp, device_ids=fids,
                             dry=dry, preprocess="raw", normalize=False)
    row = int(np.where(fids == 3)[0][0])
    got = enc.fingerprints[enc.rows_for(torch.tensor([3], dtype=torch.long))][0]
    assert torch.allclose(got, torch.from_numpy(fp[row]))
    # And the dry row is not reachable as an amp.
    assert -1 not in fids.tolist()


def test_preprocess_modes_do_what_they_say(tmp_path):
    from openamp.encoders import FingerprintEncoder

    run, _, _ = _write_fingerprints(tmp_path, [0, 1], dim=8)
    ids, fp, dry, _ = FingerprintEncoder.load_pooled(run)
    kw = dict(embedding_dim=4, fingerprints=fp, device_ids=ids, dry=dry, normalize=False)
    x = torch.from_numpy(fp[:1])

    raw = FingerprintEncoder(preprocess="raw", **kw)
    assert torch.equal(raw.preprocess_fp(x), x)

    d = FingerprintEncoder(preprocess="dry", **kw)
    assert torch.allclose(d.preprocess_fp(x), x - torch.from_numpy(dry))

    dl2 = FingerprintEncoder(preprocess="dry_l2", **kw)
    y = dl2.preprocess_fp(x)
    assert torch.allclose(y.norm(dim=-1), torch.ones(1), atol=1e-5)
    assert float((y * (x - torch.from_numpy(dry))).sum()) > 0


def test_device_mode_yields_ids_reads_no_reference_and_keeps_window_a(tmp_path):
    """The experimental control: only the conditioning changes, never window A."""
    pytest.importorskip("soundfile")
    from openamp.data import CorpusPaths, JointDataset

    cfg = load_config(data_dir=tmp_path)
    _build_corpus(cfg, devices=[0, 1, 2], n_files=3, splits=("train",))
    paths = CorpusPaths.from_config(cfg)
    common = dict(receptive_field=8, id_to_idx={0: 0, 1: 1, 2: 2},
                  ref_samples=256, clip_samples=128, pairs_per_epoch=6, seed=7)
    audio = JointDataset(paths, "train", ref_mode="audio", **common)
    dev = JointDataset(paths, "train", ref_mode="device", **common)

    assert dev.ref_files == [] and dev.same_file_mode is False
    for i in range(6):
        a, d = audio[i], dev[i]
        assert d["ref"].dtype == torch.long and d["ref"].dim() == 0
        assert int(d["ref"]) in (0, 1, 2)
        # Window A is drawn before the reference, so it is bit-identical.
        assert torch.equal(a["input"], d["input"])
        assert torch.equal(a["target"], d["target"])
        assert torch.equal(a["device_idx"], d["device_idx"])


def test_build_joint_selects_the_fingerprint_encoder(tmp_path):
    from openamp.encoders import FingerprintEncoder
    from openamp.models.joint import build_joint

    run, _, _ = _write_fingerprints(tmp_path, list(range(4)), dim=32)
    spec = _jspec(tmp_path, encoder=EncoderSpec(
        "fingerprint", {"path": str(run), "proj_hidden": 4, "normalize": False}))

    model = build_joint(spec, 4)
    assert isinstance(model.encoder, FingerprintEncoder)
    assert model.encoder.embedding_dim == 16
    assert not model.generator.embedding.weight.requires_grad
    assert model.encoder.capacity_report()["head"] == "fingerprint"


def test_encoder_kind_and_its_path_are_validated(tmp_path):
    """Both checks moved to where the value is used: the registry rejects an
    unknown kind, and the arm that opens a file rejects an empty path."""
    from openamp.core.config import ConfigError
    from openamp.encoders import FingerprintEncoder, build_encoder

    with pytest.raises(ValueError, match="kind"):
        build_encoder(EncoderSpec("codec"), 8)
    with pytest.raises(ValueError, match="preprocess"):
        FingerprintEncoder(embedding_dim=8, fingerprints=np.zeros((2, 4), np.float32),
                           device_ids=np.asarray([1, 2]), dry=np.zeros(4, np.float32),
                           preprocess="zscore")
    with pytest.raises(ConfigError, match="path"):
        FingerprintEncoder.from_spec(EncoderSpec("fingerprint", {"path": ""}), 8)


def test_resume_is_refused_when_the_fingerprint_knobs_change(tmp_path):
    """Including path/preprocess, which change no shape at all."""
    from openamp.training.joint import _check_resume_compatible

    base = {"path": "/a", "preprocess": "dry_l2", "layernorm": False}
    spec = _jspec(tmp_path, encoder=EncoderSpec("fingerprint", base))
    ck = {"spec": spec.to_dict()}
    _check_resume_compatible(ck, spec, None, None)              # identical: fine

    for kind, field, value in (("fingerprint", "path", "/b"),
                               ("fingerprint", "preprocess", "raw"),
                               ("wavenet", None, None),
                               ("fingerprint", "layernorm", True)):
        changed = dict(base)
        if field:
            changed[field] = value
        else:
            changed = {}
        with pytest.raises(Exception):
            _check_resume_compatible(
                ck, _jspec(tmp_path, encoder=EncoderSpec(kind, changed)), None, None)


def test_a_key_absent_on_one_side_is_a_change(tmp_path):
    """The v1 guard needed a table of legacy defaults so a knob added after a
    checkpoint was written could not be waved through. Comparing the whole
    encoder section gets that for free: absent on one side and set on the other
    is a difference, and absent on both means both build the same default."""
    from openamp.training.joint import _check_resume_compatible

    bare = _jspec(tmp_path, encoder=EncoderSpec("wavenet", {"channels": 4}))
    ck = {"spec": bare.to_dict()}
    _check_resume_compatible(ck, bare, None, None)              # both omit head: fine
    with pytest.raises(RuntimeError, match="head"):
        _check_resume_compatible(ck, _jspec(tmp_path, encoder=EncoderSpec(
            "wavenet", {"channels": 4, "head": "multitap"})), None, None)


# --- Frame windowing (joint.enc_frame_window) -----------------------------------
def _write_frames(tmp_path, device_ids, *, n_frames=200, dim=16):
    """A synthetic dac_frames run dir. Frame t holds the value t in every column,
    so a returned window says exactly which slice was read."""
    import json

    run = tmp_path / "frames" / "sweep_train"
    (run / "frames").mkdir(parents=True, exist_ok=True)
    for d in device_ids:
        np.save(run / "frames" / f"{int(d):04d}.npy",
                np.broadcast_to(np.arange(n_frames, dtype=np.float16)[:, None],
                                (n_frames, dim)).copy())
    (run / "meta.json").write_text(json.dumps(
        {"device_ids": [int(d) for d in device_ids], "n_frames": n_frames,
         "latent_dim": dim}))
    return run


def _frame_encoder(tmp_path, device_ids, **kw):
    from openamp.encoders import FrameStore
    from openamp.encoders import FrameAttentionEncoder

    store = FrameStore(_write_frames(tmp_path, device_ids))
    kw.setdefault("embedding_dim", 16)
    kw.setdefault("n_queries", 2)
    kw.setdefault("d_model", 8)
    kw.setdefault("proj_hidden", 8)
    return FrameAttentionEncoder(frames=store, **kw), store


def test_frame_store_window_reads_the_requested_slice(tmp_path):
    from openamp.encoders import FrameStore

    store = FrameStore(_write_frames(tmp_path, [3, 7, 11]))
    assert store.batch([3, 7, 11], "cpu").shape == (3, 200, 16)   # window off: all frames
    w = store.batch([3, 7, 11], "cpu", window=50, starts=[0, 10, 150])
    assert w.shape == (3, 50, 16)
    assert [float(w[i, 0, 0]) for i in range(3)] == [0.0, 10.0, 150.0]
    assert [float(w[i, -1, 0]) for i in range(3)] == [49.0, 59.0, 199.0]


def test_frame_store_names_the_start_that_overruns(tmp_path):
    """np.stack would report only 'arrays must have the same shape'."""
    from openamp.encoders import FrameStore

    store = FrameStore(_write_frames(tmp_path, [3, 7]))
    with pytest.raises(ValueError, match="overruns"):
        store.batch([3, 7], "cpu", window=50, starts=[0, 190])
    with pytest.raises(ValueError, match="overruns"):
        store.batch([3, 7], "cpu", window=50, starts=[-1, 0])
    with pytest.raises(ValueError, match="starts has"):
        store.batch([3, 7], "cpu", window=50, starts=[0])


def test_frame_window_makes_one_device_many_inputs(tmp_path):
    """The point of the knob: the same id must not resolve to one fixed tensor."""
    enc, _ = _frame_encoder(tmp_path, [3, 7, 11], frame_window=50)
    ref = torch.tensor([3, 7, 11])
    enc.train()
    assert not torch.allclose(enc(ref), enc(ref))


def test_frame_window_is_deterministic_and_centred_in_eval(tmp_path):
    """A val/holdout number has to be reproducible epoch to epoch, and drawn from
    the same window length training used — the pool's softmax runs over frames."""
    enc, _ = _frame_encoder(tmp_path, [3, 7, 11], frame_window=50)
    ref = torch.tensor([3, 7, 11])
    enc.eval()
    assert torch.allclose(enc(ref), enc(ref))
    assert enc.window_starts(3) == [75, 75, 75]         # (200 - 50) // 2


def test_frame_window_off_is_the_original_path(tmp_path):
    enc, _ = _frame_encoder(tmp_path, [3, 7, 11])
    ref = torch.tensor([3, 7, 11])
    assert enc.frame_window == 0 and enc.window_starts(3) is None
    enc.train()
    assert torch.allclose(enc(ref), enc(ref))          # no draw, so no variation
    assert enc.capacity_report()["frame_window"] == 0


def test_frame_window_must_fit_the_shards(tmp_path):
    with pytest.raises(ValueError, match="exceeds"):
        _frame_encoder(tmp_path, [3, 7], frame_window=201)
    with pytest.raises(ValueError, match="frame_window"):
        _frame_encoder(tmp_path, [3, 7], frame_window=-1)


def test_frame_window_is_structural_on_resume(tmp_path):
    """Changing what the encoder reads mid-run would silently change the method."""
    from openamp.training.joint import _check_resume_compatible

    base = {"frames_path": "/a", "frame_window": 1700}
    spec = _jspec(tmp_path, encoder=EncoderSpec("dac_frames", base))
    ck = {"spec": spec.to_dict()}
    _check_resume_compatible(ck, spec, None, None)
    with pytest.raises(Exception, match="frame_window"):
        _check_resume_compatible(ck, _jspec(tmp_path, encoder=EncoderSpec(
            "dac_frames", {**base, "frame_window": 800})), None, None)


# --- Embedding jitter (emulate.embed_noise_frac) ---------------------------------
def test_jitter_is_off_at_zero():
    from openamp.training.joint import jitter

    e = torch.randn(4, 64)
    assert torch.equal(jitter(e, 0.0, True), e)
    assert torch.equal(jitter(e, 0.0, False), e)


def test_jitter_stays_on_the_sphere_and_scales_with_frac():
    """enc_normalize puts e on the unit sphere, so jitter must move it *along* the
    sphere: an additive perturbation would move |e| instead, an axis the generator
    never sees vary. frac is read against 1/sqrt(De), so ||de|| ~= frac."""
    from openamp.training.joint import jitter

    torch.manual_seed(0)
    e = torch.nn.functional.normalize(torch.randn(256, 512), dim=-1)
    for frac in (0.05, 0.15, 0.3):
        j = jitter(e, frac, True)
        assert torch.allclose(j.norm(dim=-1), torch.ones(256), atol=1e-5)
        assert abs(float((j - e).norm(dim=-1).mean()) - frac) < 0.1 * frac


def test_jitter_without_normalize_scales_with_the_batch_std():
    from openamp.training.joint import jitter

    torch.manual_seed(0)
    e = torch.randn(256, 512) * 4.0
    j = jitter(e, 0.15, False)
    ratio = float(((j - e).std() / e.std()))
    assert abs(ratio - 0.15) < 0.02


def test_jitter_carries_gradient_to_the_clean_embedding():
    """Noise must not become a gradient path of its own — e still has to train."""
    from openamp.training.joint import jitter

    e = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1).requires_grad_(True)
    jitter(e, 0.2, True).sum().backward()
    assert e.grad is not None and torch.isfinite(e.grad).all() and e.grad.abs().sum() > 0


# --- LR schedule ----------------------------------------------------------------
def _two_group_opt():
    """The joint trainer's shape: two groups 8x apart, as the configs run them."""
    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    return torch.optim.AdamW([{"params": [a], "lr": 4.0e-3, "name": "generator"},
                              {"params": [b], "lr": 5.0e-4, "name": "encoder"}])


def test_cosine_floors_each_group_at_a_fraction_of_its_own_lr():
    """The reason this is a LambdaLR: CosineAnnealingLR's eta_min is one scalar."""
    from openamp.training import build_scheduler, step_scheduler

    tcfg = TrainingSpec(scheduler="cosine", cosine_min_lr_frac=0.01)
    opt = _two_group_opt()
    sched = build_scheduler(opt, tcfg, max_epochs=300)
    assert [g["lr"] for g in opt.param_groups] == [4.0e-3, 5.0e-4]
    for _ in range(300):
        step_scheduler(sched, 0.5)
    assert [pytest.approx(g["lr"]) for g in opt.param_groups] == [4.0e-5, 5.0e-6]


def test_cosine_decreases_monotonically_and_ignores_the_metric():
    from openamp.training import build_scheduler, step_scheduler

    opt = _two_group_opt()
    sched = build_scheduler(opt, TrainingSpec(scheduler="cosine"), max_epochs=50)
    lrs = []
    for epoch in range(51):
        lrs.append(opt.param_groups[0]["lr"])
        # A metric that gets *worse* must not change the curve.
        step_scheduler(sched, 0.5 + epoch)
    assert all(b < a for a, b in zip(lrs, lrs[1:]))


def test_cosine_resume_replays_to_the_same_lr():
    """Scheduler state is not checkpointed; start_epoch has to reconstruct it."""
    from openamp.training import build_scheduler, step_scheduler

    tcfg = TrainingSpec(scheduler="cosine", cosine_min_lr_frac=0.01)
    opt = _two_group_opt()
    sched = build_scheduler(opt, tcfg, max_epochs=300)
    for _ in range(137):
        step_scheduler(sched, 0.5)
    fresh = _two_group_opt()
    build_scheduler(opt=fresh, ecfg=tcfg, max_epochs=300, start_epoch=137)
    assert ([pytest.approx(g["lr"]) for g in fresh.param_groups]
            == [g["lr"] for g in opt.param_groups])


def test_plateau_is_still_the_default_and_reads_the_metric():
    from openamp.training import build_scheduler, step_scheduler

    opt = _two_group_opt()
    sched = build_scheduler(opt, TrainingSpec(), max_epochs=300)
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    for _ in range(6):                       # patience 3, factor 0.5, no improvement
        step_scheduler(sched, 0.5)
    assert [pytest.approx(g["lr"]) for g in opt.param_groups] == [2.0e-3, 2.5e-4]


@pytest.mark.parametrize("changes", [dict(scheduler="cosinne"),
                                     dict(scheduler="cosine", cosine_min_lr_frac=-0.1),
                                     dict(scheduler="cosine", cosine_min_lr_frac=1.5)])
def test_scheduler_config_is_validated(changes):
    from openamp.training import build_scheduler

    with pytest.raises(ValueError):
        build_scheduler(_two_group_opt(), TrainingSpec(**changes), max_epochs=10)
