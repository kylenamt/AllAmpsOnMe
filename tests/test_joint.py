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
from dataclasses import asdict

torch = pytest.importorskip("torch")

from openamp.core.config import JointConfig, load_config  # noqa: E402
from openamp.emulate.wavenet import FiLMWaveNet  # noqa: E402
from openamp.joint_model.model import JointModel, build_joint  # noqa: E402
from openamp.joint_model.wavenet_encoder import (AttentiveStatsPool,  # noqa: E402
                                                 WaveNetEncoder)

# A shrunken encoder schedule: same structure, tiny receptive field, fast.
SMALL = dict(kernel_sizes=(3, 3, 3), dilations=(1, 2, 4), channels=4,
             attn_hidden=8, proj_hidden=8)


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
    assert JointConfig().enc_head == "stats"


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
    from openamp.emulate.export import forward_with_embedding

    m = _small_generator().eval()
    x = torch.randn(2, 256)
    e = torch.randn(8)
    with torch.no_grad():
        want = m.forward_emb(x, e.reshape(1, -1).expand(2, -1))
    assert torch.equal(forward_with_embedding(m, x, e), want)


def test_table_delta_still_forwards_after_losing_its_override():
    """TableDeltaWaveNet's forward was a copy of the base body; it now inherits.
    Its per-layer slicing lives in layer_conditioning, which the base uses."""
    from openamp.emulate.wavenet import TableDeltaWaveNet

    m = TableDeltaWaveNet(n_devices=3, channels=4, kernel_sizes=(3, 3),
                          dilations=(1, 2), head_kernel=2).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 256), torch.tensor([0, 2]))
    assert y.shape == (2, 1, 256) and torch.isfinite(y).all()


# --- JointModel ----------------------------------------------------------------
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


def test_build_joint_freezes_the_table_and_refuses_tabledelta():
    cfg = load_config(data_dir="/tmp")
    cfg.emulate.arch = "film_wavenet"
    cfg.emulate.wn_channels = 4
    cfg.emulate.embedding_dim = 16
    cfg.joint = JointConfig(enc_channels=4, enc_attn_hidden=4, enc_proj_hidden=4)
    m = build_joint(cfg.emulate, cfg.joint, 5)
    assert not m.generator.embedding.weight.requires_grad
    assert m.encoder.embedding_dim == m.generator.embedding_dim == 16

    cfg.emulate.arch = "tabledelta_wavenet"
    with pytest.raises(ValueError, match="cannot be driven by an encoder"):
        build_joint(cfg.emulate, cfg.joint, 5)


# --- Config --------------------------------------------------------------------
def test_joint_config_validates_its_numbers():
    for kw, msg in [({"ref_seconds": 0}, "ref_seconds"),
                    ({"ref_min_gap_seconds": -1}, "ref_min_gap_seconds"),
                    ({"enc_channels": 0}, "enc_channels"),
                    ({"enc_lr": 0}, "enc_lr"),
                    ({"sources": ["nope"]}, "sources"),
                    ({"enc_head": "nope"}, "enc_head"),
                    ({"enc_taps": []}, "enc_taps"),
                    ({"enc_taps": [3, 1]}, "enc_taps"),
                    ({"enc_taps": [1, 1]}, "enc_taps"),
                    ({"enc_tap_stride": 0}, "enc_tap_stride"),
                    ({"enc_expand_dim": 0}, "enc_expand_dim"),
                    ({"enc_n_heads": 0}, "enc_n_heads"),
                    ({"enc_expand_norm": "layer"}, "enc_expand_norm"),
                    ({"enc_proj_hidden": -1}, "enc_proj_hidden")]:
        with pytest.raises(Exception, match=msg):
            JointConfig(**kw)
    # 0 is legal for proj_hidden — it is the "derive it" sentinel, not a mistake.
    assert JointConfig(enc_proj_hidden=0).enc_proj_hidden == 0


def test_joint_section_round_trips_through_yaml(tmp_path):
    import yaml

    p = tmp_path / "j.yaml"
    p.write_text(yaml.safe_dump({"joint": {"enc_channels": 24, "ref_seconds": 3.0}}))
    cfg = load_config(p, data_dir=tmp_path)
    assert cfg.joint.enc_channels == 24 and cfg.joint.ref_seconds == 3.0
    assert cfg.joint.enc_lr == JointConfig().enc_lr      # omitted keys keep defaults


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
    from openamp.emulate.dataset import build_device_index
    from openamp.joint_model.dataset import JointDataset

    _, id_to_idx = build_device_index(cfg)
    kw.setdefault("receptive_field", 64)
    kw.setdefault("clip_samples", 8_000)
    kw.setdefault("ref_samples", 8_000)
    return JointDataset(cfg, "train", id_to_idx=id_to_idx, seed=7, **kw)


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

    from openamp.joint_model import train as joint_train
    from openamp.joint_model.train import LOG_COLUMNS

    cfg = load_config(data_dir=tmp_path)
    cfg.results_dir = tmp_path / "results"
    _build_corpus(cfg, devices=list(range(4)), n_files=3, splits=("train", "val"))
    e = cfg.emulate
    e.arch, e.wn_channels, e.embedding_dim = "film_wavenet", 4, 16
    e.clip_seconds, e.batch_size, e.pairs_per_epoch = 0.1, 4, 16
    e.epochs, e.val_pairs, e.num_workers, e.amp = 2, 8, 0, False
    e.holdout_frac, e.log_every = 0.25, 1
    cfg.joint = JointConfig(ref_seconds=0.1, enc_channels=4, enc_attn_hidden=4,
                            enc_proj_hidden=4)

    m = joint_train.train(cfg, name="t", device="cpu")
    run = cfg.joint_run_dir("t")
    assert (run / "checkpoint.pt").is_file() and (run / "last.pt").is_file()
    assert (run / "config.yaml").is_file() and (run / "config_history.jsonl").is_file()
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
    from openamp.joint_model.train import _check_resume_compatible

    ecfg = load_config(data_dir=tmp_path).emulate
    ck = {"emulate_cfg": {"arch": "film_wavenet", "embedding_dim": 256},
          "joint_cfg": {"enc_channels": 16, "enc_normalize": False}}
    ecfg.arch, ecfg.embedding_dim = "film_wavenet", 256
    _check_resume_compatible(ck, ecfg, JointConfig(enc_channels=16), None, None)
    with pytest.raises(RuntimeError, match="enc_channels"):
        _check_resume_compatible(ck, ecfg, JointConfig(enc_channels=32), None, None)
    # Parameter-free but semantics-changing: must be refused too.
    with pytest.raises(RuntimeError, match="enc_normalize"):
        _check_resume_compatible(ck, ecfg, JointConfig(enc_normalize=True), None, None)


def test_resume_is_refused_when_the_head_changed(tmp_path):
    """The head knobs postdate every checkpoint written so far, so `k in saved` is
    false for all of them — without the legacy defaults the guard would wave a head
    swap straight through into a strict load_state_dict failure far from the cause.
    """
    from openamp.joint_model.train import _check_resume_compatible

    ecfg = load_config(data_dir=tmp_path).emulate
    ecfg.arch, ecfg.embedding_dim = "film_wavenet", 256
    ck = {"emulate_cfg": {"arch": "film_wavenet", "embedding_dim": 256},
          "joint_cfg": {"enc_channels": 16}}          # pre-multitap checkpoint
    _check_resume_compatible(ck, ecfg, JointConfig(), None, None)
    for kw, msg in [({"enc_head": "multitap"}, "enc_head"),
                    ({"enc_taps": [1, 2]}, "enc_taps"),
                    ({"enc_expand_dim": 128}, "enc_expand_dim"),
                    ({"enc_n_heads": 4}, "enc_n_heads"),
                    ({"enc_expand_norm": "group"}, "enc_expand_norm")]:
        with pytest.raises(RuntimeError, match=msg):
            _check_resume_compatible(ck, ecfg, JointConfig(**kw), None, None)


def test_resume_tolerates_a_yaml_list_tuple_round_trip():
    """enc_taps survives torch.save/yaml as either list or tuple; the guard must
    not read that as a structural change and refuse a legitimate resume."""
    from openamp.joint_model.train import _check_resume_compatible

    ecfg = load_config(data_dir="/tmp").emulate
    ecfg.arch, ecfg.embedding_dim = "film_wavenet", 256
    ck = {"emulate_cfg": {"arch": "film_wavenet", "embedding_dim": 256},
          "joint_cfg": {"enc_channels": 16, "enc_taps": (6, 13, 22)}}
    _check_resume_compatible(ck, ecfg, JointConfig(enc_taps=[6, 13, 22]), None, None)


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
    from openamp.joint_model.fingerprint_encoder import FingerprintEncoder

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
    from openamp.joint_model.fingerprint_encoder import FingerprintEncoder

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
    from openamp.joint_model.fingerprint_encoder import FingerprintEncoder

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
    from openamp.joint_model.dataset import JointDataset

    cfg = load_config(data_dir=tmp_path)
    _build_corpus(cfg, devices=[0, 1, 2], n_files=3, splits=("train",))
    common = dict(receptive_field=8, id_to_idx={0: 0, 1: 1, 2: 2},
                  ref_samples=256, clip_samples=128, pairs_per_epoch=6, seed=7)
    audio = JointDataset(cfg, "train", ref_mode="audio", **common)
    dev = JointDataset(cfg, "train", ref_mode="device", **common)

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
    from openamp.joint_model.fingerprint_encoder import FingerprintEncoder
    from openamp.joint_model.model import build_joint

    run, _, _ = _write_fingerprints(tmp_path, list(range(4)), dim=32)
    cfg = load_config(data_dir=tmp_path)
    e = cfg.emulate
    e.arch, e.wn_channels, e.embedding_dim = "film_wavenet", 4, 16
    j = JointConfig(enc_kind="fingerprint", fp_path=str(run), enc_proj_hidden=4)

    model = build_joint(e, j, 4)
    assert isinstance(model.encoder, FingerprintEncoder)
    assert model.encoder.embedding_dim == 16
    assert not model.generator.embedding.weight.requires_grad
    assert model.encoder.capacity_report()["head"] == "fingerprint"


def test_joint_config_validates_the_fingerprint_fields(tmp_path):
    from openamp.core.config import ConfigError

    with pytest.raises(ConfigError, match="enc_kind"):
        JointConfig(enc_kind="codec")
    with pytest.raises(ConfigError, match="fp_preprocess"):
        JointConfig(fp_preprocess="zscore")
    with pytest.raises(ConfigError, match="fp_path"):
        JointConfig(enc_kind="fingerprint", fp_path="")


def test_resume_is_refused_when_the_fingerprint_knobs_change(tmp_path):
    """Including fp_path/fp_preprocess, which change no shape at all."""
    from openamp.joint_model.train import _check_resume_compatible

    ecfg = load_config(data_dir=tmp_path).emulate
    base = dict(enc_kind="fingerprint", fp_path="/a", fp_preprocess="dry_l2",
                fp_layernorm=False)
    ck = {"emulate_cfg": asdict(ecfg), "joint_cfg": asdict(JointConfig(**base))}
    _check_resume_compatible(ck, ecfg, JointConfig(**base), None, None)   # identical: fine

    for field, value in (("fp_path", "/b"), ("fp_preprocess", "raw"),
                         ("enc_kind", "wavenet"), ("fp_layernorm", True)):
        changed = dict(base)
        changed[field] = value
        if field == "enc_kind":
            changed["fp_path"] = ""
        with pytest.raises(Exception):
            _check_resume_compatible(ck, ecfg, JointConfig(**changed), None, None)


def test_pre_fingerprint_checkpoints_still_resume(tmp_path):
    """A checkpoint written before these knobs existed must load under the defaults."""
    from openamp.joint_model.train import _check_resume_compatible

    cfg = load_config(data_dir=tmp_path)
    jcfg = JointConfig()
    saved = asdict(jcfg)
    for k in ("enc_kind", "fp_path", "fp_preprocess", "fp_layernorm",
              "enc_weight_decay"):
        saved.pop(k, None)                       # as an older run's config.yaml would be
    ck = {"emulate_cfg": asdict(cfg.emulate), "joint_cfg": saved}
    _check_resume_compatible(ck, cfg.emulate, jcfg, None, None)
