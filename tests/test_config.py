"""The one Config: env/yaml load, defaults, sections, derived paths, validation."""

from pathlib import Path

import pytest

from openamp.core.config import Config, ConfigError, load_config
from openamp.spec import load_run_spec


# --- Defaults & derived paths --------------------------------------------------
def test_defaults_and_derived_paths(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    assert cfg.sample_rate == 48_000
    assert cfg.clip_samples == 96_000
    assert cfg.chunk_samples == int(cfg.chunk_seconds * cfg.sample_rate)
    assert cfg.egdb_dir == tmp_path / "raw" / "egdb"
    assert cfg.nam_sweep_path.name == "v3_0_0.wav"
    assert cfg.corpus_manifest_path == tmp_path / "manifests" / "corpus.parquet"
    assert cfg.manifest_path == tmp_path / "manifests" / "manifest.parquet"
    assert cfg.captures_dir == tmp_path / "captures"
    assert cfg.device_render_dir(7) == tmp_path / "renders" / "0007"
    assert cfg.clean_split_dir("train") == tmp_path / "clean" / "train"


def test_result_paths_have_no_phase_names():
    cfg = Config(data_dir="/tmp/oa", results_dir="/tmp/oa/results")
    assert cfg.qa_dir.name == "qa"
    assert cfg.render_report_path.name == "render_report.md"
    assert cfg.emulate_run_dir("paper") == cfg.emulate_dir / "paper"
    assert cfg.emulate_comparison_path.name == "comparison.csv"


# --- Sections load from yaml ---------------------------------------------------
def test_corpus_and_render_from_yaml(tmp_path):
    p = tmp_path / "openamp.yaml"
    p.write_text(
        "seed: 99\n"
        "sample_rate: 48000\n"
        "corpus:\n"
        "  minutes_total: 12.5\n"
        "  clip_seconds: 2.0\n"
        "  output_format: flac\n"
        "  split_ratios:\n    train: 0.7\n    val: 0.2\n    test: 0.1\n"
        "render:\n  chunk_seconds: 5\n",
        encoding="utf-8",
    )
    cfg = load_config(p, data_dir=tmp_path)
    assert cfg.seed == 99
    assert cfg.minutes_total == 12.5
    assert cfg.chunk_seconds == 5
    assert cfg.split_ratios == {"train": 0.7, "val": 0.2, "test": 0.1}


def test_run_spec_defaults_and_overrides(tmp_path):
    """Training config is a RunSpec now, read from its own per-run file."""
    p = tmp_path / "nam_a2_256.yaml"
    p.write_text("model:\n  arch: film_wavenet\n", encoding="utf-8")
    spec = load_run_spec(p)
    assert spec.name == "nam_a2_256"            # the file stem, as it always was
    assert spec.model.arch == "film_wavenet"
    assert spec.training.batch_size == 16       # omitted keys keep their defaults

    p.write_text("name: custom\ndata_dir: data_mixed\n"
                 "model:\n  arch: film_wavenet\n  channels: 16\n"
                 "  embedding_dim: 256\ntraining:\n  batch_size: 8\n",
                 encoding="utf-8")
    spec = load_run_spec(p)
    assert spec.name == "custom"                # a config may name itself
    assert spec.data_dir.name == "data_mixed"   # ...and its corpus
    assert spec.model.kwargs == {"channels": 16, "embedding_dim": 256}
    assert spec.training.batch_size == 8



# --- Validation & rejection ----------------------------------------------------
def test_bad_ratios_rejected(tmp_path):
    with pytest.raises(ConfigError):
        Config(data_dir=tmp_path, split_ratios={"train": 0.5, "val": 0.2, "test": 0.1})


def test_missing_split_key_rejected(tmp_path):
    with pytest.raises(ConfigError):
        Config(data_dir=tmp_path, split_ratios={"train": 0.9, "val": 0.1})


def test_unknown_section_key_raises(tmp_path):
    p = tmp_path / "openamp.yaml"
    p.write_text("corpus:\n  nonsense: 1\n", encoding="utf-8")
    load_config(p, data_dir=tmp_path)           # corpus keys are read selectively

    # A run spec is stricter: model/loss kwargs are opaque by design, but a
    # mistyped *section* or training knob must not be silently ignored.
    p.write_text("nonsense:\n  x: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_run_spec(p)
    p.write_text("training:\n  nonsense: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="training"):
        load_run_spec(p)


# --- Environment ---------------------------------------------------------------
def test_require_key_raises_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAMP_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)            # no .env present
    with pytest.raises(ConfigError):
        load_config(data_dir=tmp_path, require_key=True)


def test_legacy_key_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("OPENAMP_API_KEY", raising=False)
    monkeypatch.setenv("T3K_PUBLISHABLE_KEY", "legacy")
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("WARNING"):
        load_config(data_dir=tmp_path)
    assert any("OPENAMP_API_KEY" in r.message for r in caplog.records)


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAMP_API_KEY", "pub_xyz")
    monkeypatch.setenv("OPENAMP_SEED", "7")
    monkeypatch.chdir(tmp_path)
    cfg = load_config(data_dir=tmp_path)
    assert cfg.publishable_key == "pub_xyz"
    assert cfg.seed == 7


# --- Shipped repo config -------------------------------------------------------
def test_repo_config_yaml_is_valid():
    repo_yaml = Path(__file__).resolve().parents[1] / "configs" / "openamp.yaml"
    cfg = load_config(repo_yaml, data_dir="/tmp/openamp-test")
    assert abs(sum(cfg.split_ratios.values()) - 1.0) < 1e-9
    assert cfg.minutes_total > 0


def test_every_repo_run_config_loads():
    """Every configs/{emulate,joint}/*.yaml must resolve to a RunSpec."""
    root = Path(__file__).resolve().parents[1] / "configs"
    files = [f for f in sorted(root.glob("emulate/*.yaml"))
             + sorted(root.glob("joint/*.yaml")) if f.stem != "enroll"]
    specs = [load_run_spec(f) for f in files]
    assert len(specs) >= 30
    for spec, f in zip(specs, files):
        assert spec.model.arch in ("film_wavenet", "mlpfilm_wavenet", "delta_wavenet")
        assert spec.training.batch_size > 0
        assert (spec.joint is not None) == (f.parent.name == "joint")
