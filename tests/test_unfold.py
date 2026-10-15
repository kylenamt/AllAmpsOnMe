"""Deep-unfolded enrollment: the closed-form gradient, the learned step, the
unfold section of the run spec, the per-amp dataset, and a tiny end-to-end run.

torch-guarded like the other model tests; the dataset and training cases also
write tiny FLACs, so they need soundfile."""

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openamp.core.config import ConfigError
from openamp.enroll.unfold import Unfolder, coeff_grads, per_amp_esr
from openamp.models import DeltaWaveNet, FiLMWaveNet
from openamp.spec import (DataSpec, EncoderSpec, JointSpec, LossSpec, ModelSpec,
                          RunSpec, TrainingSpec, UnfoldSpec, load_run_spec)

SMALL = dict(channels=4, kernel_sizes=(3, 3, 3), dilations=(1, 2, 4), head_kernel=4)
COEFF = 0.85


def _gen(**kw):
    torch.manual_seed(0)
    kw.setdefault("n_devices", 4)
    kw.setdefault("embedding_dim", 8)
    return DeltaWaveNet(delta_rank=3, **SMALL, **kw).eval().requires_grad_(False)


def _clips(gen, A=3, n=2, T=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = 0.3 * torch.randn(A, n, gen.receptive_field + T, generator=g)
    t = torch.tanh(2.0 * x[..., gen.receptive_field:])
    return x, t


def _e0(gen, A=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return gen.project_embedding(torch.randn(A, gen.embedding_dim, generator=g))


# --- The closed form --------------------------------------------------------------
def test_coeff_grads_through_W_is_exactly_the_embedding_gradient():
    gen = _gen(embed_normalize=False)
    x, t = _clips(gen)
    e = _e0(gen).requires_grad_(True)
    A, n = x.shape[:2]
    y = gen.forward_emb(x.reshape(A * n, -1), e.repeat_interleave(n, 0))
    per_amp_esr(y[..., gen.receptive_field:].view(A, n, -1), t, COEFF).sum().backward()

    g, _ = coeff_grads(gen, x, t, e, COEFF)
    unf = Unfolder(gen, steps=1)
    closed = torch.einsum("alr,lrd->ad", g, unf.W)          # sum_l W_l^T g_l
    assert torch.allclose(closed, e.grad, rtol=1e-4, atol=1e-7)


def test_an_untrained_step_is_normalized_gradient_descent():
    gen = _gen(embed_normalize=True)
    x, t = _clips(gen)
    e0 = _e0(gen)
    unf = Unfolder(gen, steps=1)
    unf.log_s.data.fill_(np.log(0.05))
    with torch.no_grad():
        e1, _, _ = unf(gen, e0, x, t, COEFF)

    e = e0.clone().requires_grad_(True)
    A, n = x.shape[:2]
    y = gen.forward_emb(x.reshape(A * n, -1), e.repeat_interleave(n, 0))
    per_amp_esr(y[..., gen.receptive_field:].view(A, n, -1), t, COEFF).sum().backward()
    g, _ = coeff_grads(gen, x, t, e0, COEFF)
    scale = g.flatten(1).norm(dim=1, keepdim=True)          # what normalize divides by
    expected = gen.project_embedding(e0 - 0.05 * e.grad / scale)
    assert torch.allclose(e1, expected, rtol=1e-4, atol=1e-6)


def test_amps_are_stepped_independently_of_their_batch():
    gen = _gen(embed_normalize=True)
    x, t = _clips(gen, A=3)
    e0 = _e0(gen, A=3)
    unf = Unfolder(gen, steps=3)
    with torch.no_grad():
        together, _, _ = unf(gen, e0, x, t, COEFF)
        alone, _, _ = unf(gen, e0[1:2], x[1:2], t[1:2], COEFF)
    assert torch.allclose(together[1:2], alone, rtol=1e-4, atol=1e-6)


def test_steps_stay_on_the_generators_sphere():
    gen = _gen(embed_normalize=True)
    x, t = _clips(gen)
    with torch.no_grad():
        eK, _, _ = Unfolder(gen, steps=3)(gen, _e0(gen), x, t, COEFF)
    assert torch.allclose(eK.norm(dim=1), gen.embed_scale.expand(len(eK)), atol=1e-5)


def test_calibrate_sets_the_first_step_length():
    gen = _gen(embed_normalize=True)
    x, t = _clips(gen)
    e0 = _e0(gen)
    unf = Unfolder(gen, steps=2)
    target = unf.calibrate(gen, e0, x, t, COEFF, 0.03)
    assert target == pytest.approx(0.03 * float(e0.norm(dim=1).mean()), rel=1e-6)
    g, _ = coeff_grads(gen, x, t, e0, COEFF)
    with torch.no_grad():
        step = unf.log_s[0].exp() * unf.direction(0, Unfolder.normalize(g))
    assert float(step.norm(dim=1).mean()) == pytest.approx(target, rel=1e-4)


def test_V_and_the_step_sizes_get_gradient_and_the_generator_does_not():
    gen = _gen(embed_normalize=True)
    x, t = _clips(gen)
    unf = Unfolder(gen, steps=2)
    eK, _, _ = unf(gen, _e0(gen), x, t, COEFF)
    A, n = x.shape[:2]
    y = gen.forward_emb(x.reshape(A * n, -1), eK.repeat_interleave(n, 0))
    per_amp_esr(y[..., gen.receptive_field:].view(A, n, -1), t, COEFF).mean().backward()
    assert unf.V.grad is not None and float(unf.V.grad.abs().sum()) > 0
    assert unf.log_s.grad is not None and float(unf.log_s.grad.abs().sum()) > 0
    assert all(p.grad is None for p in gen.parameters())


def test_untrained_V_starts_at_W_and_drift_reads_zero():
    unf = Unfolder(_gen(), steps=4)
    assert unf.V.shape == (4, 3, 3, 8)                      # [K, L, rank, D]
    assert torch.equal(unf.V[2], unf.W)
    assert unf.drift() == 0.0


def test_unfolding_refuses_a_film_generator():
    gen = FiLMWaveNet(n_devices=2, embedding_dim=8, **SMALL)
    with pytest.raises(ValueError, match="delta_wavenet"):
        Unfolder(gen, steps=1)


# --- The run spec -----------------------------------------------------------------
def test_unfold_section_round_trips_and_names_its_stage(tmp_path):
    cfg = tmp_path / "k5.yaml"
    cfg.write_text("unfold:\n  generator_run: g\n  encoder_run: e\n  steps: 5\n"
                   "training:\n  lr: 3.0e-4\n")
    spec = load_run_spec(cfg, results_dir=tmp_path / "results")
    assert spec.is_unfold and not spec.is_joint and spec.stage == "unfold"
    assert spec.run_dir == tmp_path / "results" / "unfold" / "k5"
    assert spec.unfold == UnfoldSpec(generator_run="g", encoder_run="e", steps=5)
    assert RunSpec.from_dict(spec.to_dict()) == spec


def test_unfold_section_rejects_unknown_keys_and_a_joint_section(tmp_path):
    with pytest.raises(ConfigError, match="unfold: unknown"):
        UnfoldSpec.from_dict({"stpes": 5})
    with pytest.raises(ConfigError, match="not both"):
        RunSpec(name="x", data_dir=tmp_path, results_dir=tmp_path,
                joint=JointSpec(), unfold=UnfoldSpec())


def test_holdout_split_is_disjoint_and_covers_every_amp():
    from openamp.training.unfold import split_holdout

    sel, rep = split_holdout([9, 2, 7, 4, 5])
    assert sel == [2, 5, 9] and rep == [4, 7]


# --- The per-amp dataset and a whole run -----------------------------------------
TRAIN, HOLD = [0, 1, 2, 3], [4, 5, 6, 7]
CLIP = 1024


def _base(spec, gen):
    from openamp.data import EmulationDataset

    ids = TRAIN + HOLD
    return EmulationDataset(spec.paths, "train", receptive_field=gen.receptive_field,
                            id_to_idx={d: i for i, d in enumerate(ids)},
                            clip_samples=CLIP, pairs_per_epoch=1, seed=0)


def _data_spec(tmp_path):
    return RunSpec(name="data", data_dir=tmp_path / "data",
                   results_dir=tmp_path / "results")


def test_amp_clips_are_one_amp_and_a_pure_function_of_seed_epoch_index(
        tmp_path, write_corpus):
    pytest.importorskip("soundfile")
    from openamp.data import AmpClipsDataset

    spec = _data_spec(tmp_path)
    write_corpus(spec, TRAIN + HOLD, samples=8000, splits=("train",))
    gen = _gen()
    ds = AmpClipsDataset(_base(spec, gen), [2, 5], 3, seed=7)
    a, b = ds[1], ds[1]
    assert a["xf"].shape == (3, gen.receptive_field + CLIP) and a["ts"].shape == (3, CLIP)
    assert int(a["amp"]) == 1
    assert all(torch.equal(a[k], b[k]) for k in ("xf", "tf", "xs", "ts"))
    assert not torch.equal(a["xf"], a["xs"])                 # fit and score clips differ
    # renders are tanh(clean * (1 + id)): every clip is amp 5's, none another amp's
    for x, t in ((a["xf"], a["tf"]), (a["xs"], a["ts"])):
        assert torch.allclose(t, torch.tanh(x[:, gen.receptive_field:] * 6.0), atol=1e-4)
    ds.epoch = 1
    assert not torch.equal(ds[1]["xf"], a["xf"])


def _write_frames(root, device_ids, *, n_frames=40, dim=16):
    rng = np.random.default_rng(0)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    for d in device_ids:
        np.save(root / "frames" / f"{int(d):04d}.npy",
                rng.standard_normal((n_frames, dim)).astype(np.float16))
    (root / "meta.json").write_text(json.dumps(
        {"device_ids": [int(d) for d in device_ids], "n_frames": n_frames,
         "latent_dim": dim}))
    return root


def _write_runs(tmp_path):
    """A frozen delta generator (emulate checkpoint) and a frozen dac_frames encoder
    (joint checkpoint), laid out the way the trainer reads finished runs."""
    from openamp.encoders import build_encoder
    from openamp.models import build_model

    gspec = RunSpec(name="gen", data_dir=tmp_path / "data",
                    results_dir=tmp_path / "results",
                    model=ModelSpec("delta_wavenet", {**SMALL, "embedding_dim": 8,
                                                      "delta_rank": 3,
                                                      "embed_normalize": True}),
                    data=DataSpec(clip_seconds=CLIP / 48_000),
                    loss=LossSpec(kwargs={"preemph_coeff": COEFF, "stft_weight": 0.05}))
    torch.manual_seed(0)
    gen = build_model(gspec.model, len(TRAIN))
    gdir = tmp_path / "results" / "emulate" / "gen"
    gdir.mkdir(parents=True)
    torch.save({"model": gen.state_dict(), "spec": gspec.to_dict(), "device_ids": TRAIN,
                "holdout_ids": HOLD, "manifest_sha256": "", "name": "gen", "epoch": 0},
               gdir / "checkpoint.pt")

    frames = _write_frames(tmp_path / "frames", TRAIN + HOLD)
    es = EncoderSpec("dac_frames", {"frames_path": str(frames), "n_queries": 2,
                                    "d_model": 8, "normalize": True})
    enc = build_encoder(es, 8)
    edir = tmp_path / "results" / "joint" / "enc"
    edir.mkdir(parents=True)
    torch.save({"spec": gspec.replace(name="enc", joint=JointSpec(encoder=es)).to_dict(),
                "encoder": enc.state_dict(), "generator": gen.state_dict()},
               edir / "checkpoint.pt")
    return gdir, edir, gen.state_dict()


def _uspec(tmp_path, gdir, edir, **unfold):
    return RunSpec(name="u", data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                   unfold=UnfoldSpec(generator_run=str(gdir), encoder_run=str(edir),
                                     **{"steps": 2, "clips": 2, **unfold}),
                   training=TrainingSpec(batch_size=2, epochs=2, num_workers=0, amp=False,
                                         val_pairs=4, early_stop_patience=5, log_every=1,
                                         lr=1e-3))


def test_train_writes_a_run_reports_unseen_amps_and_resumes(tmp_path, write_corpus):
    pytest.importorskip("soundfile")
    pytest.importorskip("auraloss")
    import pandas as pd

    from openamp.enroll.unfold import load_unfold
    from openamp.training.unfold import train

    write_corpus(_data_spec(tmp_path), TRAIN + HOLD, samples=6000,
                 splits=("train", "val", "test"))
    gdir, edir, gen_before = _write_runs(tmp_path)
    spec = _uspec(tmp_path, gdir, edir)
    m = train(spec, device="cpu")

    run = spec.run_dir
    assert {"checkpoint.pt", "last.pt", "train_log.csv", "metrics.json",
            "resolved_config.yaml", "report_per_device.csv",
            "encoder_vectors.pt"} <= {p.name for p in run.iterdir()}
    log = pd.read_csv(run / "train_log.csv")
    assert list(log["epoch"]) == [-1, 0, 1]
    assert m["n_train"] == 4 and m["n_select"] == 2 and m["n_report"] == 2
    assert m["epochs_run"] == 2 and np.isfinite(m["report_median"])
    assert {"report_encoder_median", "report_untrained_median"} <= set(m)
    rep = pd.read_csv(run / "report_per_device.csv")
    assert sorted(rep["device_id"]) == [5, 7]               # odd holdout positions

    # the generator is untouched, and the run reads back with its own spec
    unf, gen, ck = load_unfold(run)
    assert all(torch.equal(gen.state_dict()[k], v) for k, v in gen_before.items())
    assert unf.steps == 2 and ck["spec"]["model"]["arch"] == "delta_wavenet"

    # resume continues from last.pt and appends; changing K is refused
    train(spec.replace(training=replace(spec.training, epochs=3)), device="cpu",
          resume=True)
    assert list(pd.read_csv(run / "train_log.csv")["epoch"]) == [-1, 0, 1, 2]
    with pytest.raises(RuntimeError, match="unfold.steps"):
        train(_uspec(tmp_path, gdir, edir, steps=3), device="cpu", resume=True)


def test_amp_clips_can_score_on_another_split_and_scale_targets(tmp_path, write_corpus):
    pytest.importorskip("soundfile")
    from openamp.data import AmpClipsDataset, EmulationDataset

    spec = _data_spec(tmp_path)
    write_corpus(spec, TRAIN + HOLD, samples=8000, splits=("train", "val"))
    gen = _gen()
    val = EmulationDataset(spec.paths, "val", receptive_field=gen.receptive_field,
                           id_to_idx={d: i for i, d in enumerate(TRAIN + HOLD)},
                           clip_samples=CLIP, pairs_per_epoch=1, seed=0)
    plain = AmpClipsDataset(_base(spec, gen), [5], 3, seed=7)[0]
    split = AmpClipsDataset(_base(spec, gen), [5], 3, seed=7, score_base=val,
                            gains={5: 2.0})[0]
    assert torch.allclose(split["tf"], 2.0 * plain["tf"])    # same fit draws, scaled
    assert torch.equal(split["xf"], plain["xf"])
    val_files = {f["file_id"] for f in val.files}
    assert val_files and not torch.equal(split["xs"], plain["xs"])
    assert torch.allclose(split["ts"], 2.0 * torch.tanh(
        split["xs"][:, gen.receptive_field:] * 6.0), atol=1e-4)


def _stamp_captures(paths, sha_of, scale=1.0):
    """Give a fixture corpus the per-amp capture hash and output_scale real renders carry."""
    import pandas as pd

    r = pd.read_parquet(paths.renders_manifest)
    r["nam_sha256"] = r["device_id"].map(sha_of)
    r["output_scale"] = scale
    r.to_parquet(paths.renders_manifest)


def test_train_can_take_its_clips_from_another_corpus(tmp_path, write_corpus):
    pytest.importorskip("soundfile")
    pytest.importorskip("auraloss")
    from openamp.data import CorpusPaths
    from openamp.training.unfold import train

    write_corpus(_data_spec(tmp_path), TRAIN + HOLD, samples=6000,
                 splits=("train", "val", "test"))
    capture = CorpusPaths.from_data_dir(tmp_path / "capture")
    write_corpus(capture, TRAIN + HOLD, samples=6000, splits=("train", "val"))
    _stamp_captures(_data_spec(tmp_path).paths, str)
    _stamp_captures(capture, str, scale=0.5)
    gdir, edir, _ = _write_runs(tmp_path)
    spec = _uspec(tmp_path, gdir, edir, clips_data_dir=str(tmp_path / "capture"),
                  fit_split="train", loss_split="val")
    m = train(spec.replace(training=replace(spec.training, epochs=1)), device="cpu")
    assert np.isfinite(m["report_median"])

    # a corpus that rendered a different capture for any amp is refused
    _stamp_captures(capture, lambda d: "other" if d == 5 else str(d))
    with pytest.raises(RuntimeError, match="different capture"):
        train(spec.replace(name="u2"), device="cpu")
