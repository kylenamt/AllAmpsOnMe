"""One configuration object for the whole pipeline.

The pipeline runs as a single process, so it reads a single :class:`Config`:

- **API access** (publishable key, base URL, token path, rate limit) comes from
  the environment, optionally via a git-ignored ``.env`` file (``OPENAMP_*`` vars).
- **Pipeline knobs** (corpus, render, model, train, eval) come from
  ``configs/openamp.yaml``; ``model``/``train``/``eval`` are nested sub-sections.
- **Every path** is derived from one ``data_dir`` (inputs/manifests/renders) and
  one ``results_dir`` (reports/checkpoints), so a run is fully reproducible from
  the single ``seed``.

``load_config()`` merges ``.env`` + yaml into a :class:`Config`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from openamp.core import constants as C

log = logging.getLogger("openamp.config")

# --- Fixed API facts -----------------------------------------------------------
DEFAULT_BASE_URL = "https://www.tone3000.com/api/v1"
DEFAULT_REDIRECT_URI = "http://localhost:3001"
DEFAULT_RATE_LIMIT_RPM = 80          # sit under the server's 100 req/min
DEFAULT_SEED = 1234
SELECT_TARGET = 430                  # headroom over 400 for validation/dedup losses
FINAL_TARGET = 400
DEFAULT_ARCHITECTURE = 2             # A2 (Slimmable NAM) is the canonical capture


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


# --- Nested pipeline sections (yaml) -------------------------------------------
@dataclass
class EmulateConfig:
    """One-to-many amp emulation: FiLM-conditioned model + its training (spec §4).

    A self-contained sub-pipeline: every architecture size is a plain knob here,
    so exploring sizes is copy-a-config-and-change-numbers. Read from the
    ``emulate:`` yaml section; a per-run file under ``configs/emulate/<name>.yaml``
    overrides just this section and the run is named after the file stem.
    """

    # --- Architecture ------------------------------------------------------------
    # "film_tcn" (paper FiLM-TCN) | "film_wavenet" (the corpus's own NAM A2
    # WaveNet topology, FiLM-conditioned; see openamp/emulate/wavenet.py).
    arch: str = "film_tcn"

    # --- FiLM-TCN model (fully parametric; the paper default is the values below)
    blocks: int = 2                  # dilation resets per block
    layers_per_block: int = 8
    channels: int = 16
    kernel_size: int = 3
    dilation_growth: int = 2

    # --- FiLM-WaveNet model (kernel/dilation schedule is the fixed A2 one) ------
    wn_channels: int = 8             # A2 full-width value; the width sweep knob

    # --- Shared model knobs ------------------------------------------------------
    embedding_dim: int = 64          # per-device embedding, FiLM at every layer
    # One-to-one baseline: >=0 trains a single-device model on that device_id
    # (same class, n_devices=1) for the one-to-many gap reference.
    single_device: int = -1

    # --- Training (one script; Adam + reduce-on-plateau to val plateau) ---------
    # Fraction of render-ok devices held out of training entirely, so Phase 5 can
    # enroll them as truly unseen (reference code uses a 90/10 device split). The
    # drawn set is persisted to `emulate_holdout.txt` and read back thereafter;
    # <= 0 disables the holdout for a run. Does not apply to `single_device`.
    holdout_frac: float = 0.1
    clip_seconds: float = 2.0        # training clip length (must match corpus grid)
    batch_size: int = 16
    pairs_per_epoch: int = 50_000    # dataset length: random (device, file, window) draws
    epochs: int = 60                 # max epochs; early-stopped at val plateau
    lr: float = 5e-4
    weight_decay: float = 0.0
    plateau_patience: int = 3        # ReduceLROnPlateau patience (epochs)
    plateau_factor: float = 0.5
    early_stop_patience: int = 8     # stop after N epochs with no val-ESR improvement
    stft_weight: float = 1.0         # multi-resolution STFT term weight (ESR term = 1.0)
    preemph: float = 0.85            # 1st-order pre-emphasis coefficient for ESR
    num_workers: int = 6
    amp: bool = True                 # CUDA mixed precision (STFT loss forced fp32)
    amp_dtype: str = "fp16"          # "fp16" | "bf16"; bf16 has fp32's range (no overflow) at fp16's memory cost
    val_pairs: int = 2000            # #pairs used for each val-ESR estimate
    log_every: int = 50


# --- The one config ------------------------------------------------------------
@dataclass
class Config:
    """Resolved runtime configuration for the whole pipeline."""

    data_dir: Path
    results_dir: Path = field(default=None)  # type: ignore[assignment]
    seed: int = DEFAULT_SEED
    sample_rate: int = C.SAMPLE_RATE

    # Corpus / render knobs (yaml ``corpus:`` / ``render:`` sections).
    minutes_total: float = 40.0
    split_ratios: dict = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    clip_seconds: float = C.CLIP_SECONDS
    output_format: str = "flac"
    chunk_seconds: float = C.DEFAULT_CHUNK_SECONDS

    # API access (environment / ``.env``).
    publishable_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    redirect_uri: str = DEFAULT_REDIRECT_URI
    token_path: Path = field(default_factory=lambda: Path("./.openamp_tokens.json"))
    rate_limit_rpm: int = DEFAULT_RATE_LIMIT_RPM
    select_target: int = SELECT_TARGET
    final_target: int = FINAL_TARGET
    # NAM capture architecture to acquire: 2 (A2, canonical) or 1 (legacy A1).
    # Must be passed to BOTH `/tones/search` and `/models` -- the models endpoint
    # defaults to A1 and will silently hand back A1 captures otherwise.
    architecture: int = DEFAULT_ARCHITECTURE

    # Nested emulation sub-section (yaml).
    emulate: EmulateConfig = field(default_factory=EmulateConfig)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir) if self.results_dir is not None \
            else Path.cwd() / "results"
        self.token_path = Path(self.token_path)
        self._validate()

    def _validate(self) -> None:
        s = self.split_ratios
        if set(s) != set(C.SPLITS):
            raise ConfigError(f"split_ratios must have keys {C.SPLITS}, got {sorted(s)}")
        if abs(sum(s.values()) - 1.0) > 1e-6:
            raise ConfigError(f"split_ratios must sum to 1.0, got {sum(s.values())}")
        if self.minutes_total <= 0:
            raise ConfigError("minutes_total must be positive")
        if self.architecture not in (1, 2):
            raise ConfigError(
                f"architecture must be 1 (A1) or 2 (A2), got {self.architecture}")

    # --- Derived counts ---------------------------------------------------------
    @property
    def clip_samples(self) -> int:
        return int(round(self.clip_seconds * self.sample_rate))

    @property
    def chunk_samples(self) -> int:
        return int(round(self.chunk_seconds * self.sample_rate))

    # --- Input / manifest / render paths (under data_dir) -----------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def egdb_dir(self) -> Path:
        return self.raw_dir / "egdb"

    @property
    def nam_sweep_path(self) -> Path:
        return self.raw_dir / "nam_sweep" / C.NAM_SWEEP_FILENAME

    @property
    def captures_dir(self) -> Path:
        return self.data_dir / "captures"

    @property
    def clean_dir(self) -> Path:
        return self.data_dir / "clean"

    @property
    def renders_dir(self) -> Path:
        return self.data_dir / "renders"

    @property
    def manifests_dir(self) -> Path:
        return self.data_dir / "manifests"

    def clean_split_dir(self, split: str) -> Path:
        return self.clean_dir / split

    def device_render_dir(self, device_id: int) -> Path:
        return self.renders_dir / f"{int(device_id):04d}"

    # Manifest files.
    @property
    def candidates_path(self) -> Path:
        return self.manifests_dir / "candidates.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.manifests_dir / C.ACQUISITION_MANIFEST

    @property
    def rejected_path(self) -> Path:
        return self.manifests_dir / "rejected.parquet"

    @property
    def corpus_manifest_path(self) -> Path:
        return self.manifests_dir / C.CORPUS_MANIFEST

    @property
    def clips_manifest_path(self) -> Path:
        return self.manifests_dir / C.CLIPS_MANIFEST

    @property
    def renders_manifest_path(self) -> Path:
        return self.manifests_dir / C.RENDERS_MANIFEST

    @property
    def devices_final_path(self) -> Path:
        return self.manifests_dir / C.DEVICES_FINAL_MANIFEST

    @property
    def render_subset_path(self) -> Path:
        return self.manifests_dir / C.RENDER_SUBSET

    @property
    def emulate_holdout_path(self) -> Path:
        return self.manifests_dir / C.EMULATE_HOLDOUT

    # --- Result / report / checkpoint paths (under results_dir) -----------------
    @property
    def qa_dir(self) -> Path:
        return self.results_dir / "qa"

    @property
    def render_report_path(self) -> Path:
        return self.results_dir / "render_report.md"

    # --- Emulation foundation-model runs (Phase 4) under results_dir ------------
    @property
    def emulate_dir(self) -> Path:
        return self.results_dir / "emulate"

    def emulate_run_dir(self, name: str) -> Path:
        return self.emulate_dir / name

    @property
    def emulate_comparison_path(self) -> Path:
        return self.emulate_dir / "comparison.csv"

    @property
    def emulate_demos_dir(self) -> Path:
        return self.emulate_dir / "demos"

    def ensure_dirs(self) -> None:
        for d in (self.manifests_dir, self.captures_dir, self.clean_dir,
                  self.renders_dir, self.results_dir, self.qa_dir):
            d.mkdir(parents=True, exist_ok=True)


# --- Loading -------------------------------------------------------------------
def default_config_path(project_root: Path | None = None) -> Path:
    return (project_root or Path.cwd()) / "configs" / "openamp.yaml"


def _load_dotenv(path: Path) -> None:
    """Minimal, dependency-free ``.env`` loader.

    Only sets keys not already in ``os.environ`` (the real environment wins).
    Supports ``KEY=VALUE`` lines, ``#`` comments, optional surrounding quotes.
    """
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _coerce(dc_cls, values: dict):
    """Build a dataclass from a dict, rejecting unknown keys, keeping defaults."""
    known = {f.name for f in dc_cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"{dc_cls.__name__}: unknown keys {sorted(unknown)}")
    return dc_cls(**{k: v for k, v in values.items() if k in known})


def _read_yaml(path: Path) -> dict:
    import yaml  # local import; pyyaml is a listed dep

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML ({exc})") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    return loaded


def load_config(path: Path | None = None, *, data_dir: Path | None = None,
                require_key: bool = False) -> Config:
    """Build the one :class:`Config` from ``.env`` + ``configs/openamp.yaml``.

    ``data_dir`` overrides the yaml/env data root (used by tests). ``require_key``
    raises if the API key is absent (stages that actually call the API set it;
    metadata-only stages can run without it).
    """
    _load_dotenv(Path.cwd() / ".env")

    y: dict = {}
    cfg_path = path or default_config_path()
    if cfg_path and Path(cfg_path).is_file():
        y = _read_yaml(Path(cfg_path))

    corpus = y.get("corpus", {})
    render = y.get("render", {})
    acquire = y.get("acquire", {})

    key = os.environ.get("OPENAMP_API_KEY", "").strip()
    if not key and os.environ.get("T3K_PUBLISHABLE_KEY", "").strip():
        log.warning("T3K_PUBLISHABLE_KEY is set but OPENAMP_API_KEY is not; rename it "
                    "in your .env (and re-run `openamp auth` or rename the token file).")
    if require_key and not key:
        raise ConfigError(
            "OPENAMP_API_KEY is not set. Create a publishable key at tone3000.com -> "
            "Settings -> API Keys and put it in .env (see .env.example).")

    if data_dir is None:
        data_dir = Path(os.environ.get("OPENAMP_DATA_DIR", y.get("data_dir") or "./data"))

    # Build the Config by overriding only what a source (yaml/env) actually
    # provides; every default lives once, on the dataclass fields above. A key is
    # added to ``kw`` only when present, so absent sources fall through to those.
    kw: dict = {
        "data_dir": Path(data_dir).expanduser(),
        "publishable_key": key,
        "emulate": _coerce(EmulateConfig, y.get("emulate", {})),
    }

    def _yaml(mapping: dict, name: str, dest: str, cast) -> None:
        if name in mapping:
            kw[dest] = cast(mapping[name])

    def _env(var: str, dest: str, cast) -> None:
        if var in os.environ:
            kw[dest] = cast(os.environ[var])

    if y.get("results_dir"):
        kw["results_dir"] = Path(y["results_dir"]).expanduser()
    # seed: env wins over yaml.
    _yaml(y, "seed", "seed", int)
    _env("OPENAMP_SEED", "seed", int)
    _yaml(y, "sample_rate", "sample_rate", int)
    _yaml(corpus, "minutes_total", "minutes_total", float)
    _yaml(corpus, "split_ratios", "split_ratios", lambda d: {k: float(v) for k, v in d.items()})
    _yaml(corpus, "clip_seconds", "clip_seconds", float)
    _yaml(corpus, "output_format", "output_format", str)
    _yaml(render, "chunk_seconds", "chunk_seconds", float)
    _yaml(acquire, "architecture", "architecture", int)
    _yaml(acquire, "select_target", "select_target", int)
    _env("OPENAMP_ARCHITECTURE", "architecture", int)
    _env("OPENAMP_BASE_URL", "base_url", lambda s: s.rstrip("/"))
    _env("OPENAMP_REDIRECT_URI", "redirect_uri", lambda s: s.rstrip("/"))
    _env("OPENAMP_TOKEN_PATH", "token_path", lambda s: Path(s).expanduser())
    _env("OPENAMP_RATE_LIMIT_RPM", "rate_limit_rpm", int)

    return Config(**kw)
