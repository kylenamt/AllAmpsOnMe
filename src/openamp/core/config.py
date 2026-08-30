"""The pipeline's configuration: env/.env for API access, configs/openamp.yaml for
acquisition/corpus/render knobs, one data_dir/results_dir for all paths.

Training is *not* here. An experiment is a :class:`~openamp.spec.RunSpec`, read
from its own ``configs/{emulate,joint}/<name>.yaml`` by
:func:`~openamp.spec.load_run_spec`, which carries the corpus it trained on
rather than picking it up from the environment. This object stays what the
acquisition, corpus and render stages need: fixed API facts, split ratios, and
every derived path.
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
DEFAULT_RATE_LIMIT_RPM = 80          # stay under the server's 100 req/min
DEFAULT_SEED = 1234
SELECT_TARGET = 430                  # headroom over 400 for validation/dedup losses
FINAL_TARGET = 400
DEFAULT_ARCHITECTURE = 2             # A2 (Slimmable NAM) is the canonical capture


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


# --- Nested pipeline sections (yaml) -------------------------------------------
@dataclass
class EnrollConfig:
    """Phase 5 enrollment: fit embeddings for unseen devices against a frozen run.

    Read from the ``enroll:`` yaml section; passed to ``emulate-enroll --config``.
    Run dir / ``--devices`` / ``--device`` stay CLI flags (what/where, not how).
    """

    pairs: int = 1000                 # training pairs per device per epoch (optimization budget)
    epochs: int = 30                  # max epochs; early-stopped on val ESR
    lr: float = 1e-2
    weight_decay: float = 0.0         # Adam weight decay on the embedding rows
    early_stop_patience: int = 5
    plateau_patience: int = 2         # ReduceLROnPlateau patience (epochs)
    plateau_factor: float = 0.5
    test_pairs: int = 200             # test pairs per device, for the final test ESR
    stft_weight: float | None = None  # override the run's trained STFT loss weight (None = keep it)
    batch_size: int | None = None     # override the run's batch size (fp32 fit needs ~2x memory; try 8-16)
    init: str = "uniform"             # "uniform" | "table_mean" -- see _swap_embedding
    seed: int | None = None           # override the top-level seed (None = use it)


# --- The one config ------------------------------------------------------------
@dataclass
class Config:
    """Resolved runtime configuration for the whole pipeline."""

    data_dir: Path
    results_dir: Path = field(default=None)  # type: ignore[assignment]
    seed: int = DEFAULT_SEED
    sample_rate: int = C.SAMPLE_RATE

    # Corpus / render knobs (yaml `corpus:` / `render:` sections).
    minutes_total: float = 40.0
    split_ratios: dict = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    clip_seconds: float = C.CLIP_SECONDS
    output_format: str = "flac"
    chunk_seconds: float = C.DEFAULT_CHUNK_SECONDS

    # API access (environment / `.env`).
    publishable_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    redirect_uri: str = DEFAULT_REDIRECT_URI
    token_path: Path = field(default_factory=lambda: Path("./.openamp_tokens.json"))
    rate_limit_rpm: int = DEFAULT_RATE_LIMIT_RPM
    select_target: int = SELECT_TARGET
    final_target: int = FINAL_TARGET
    architecture: int = DEFAULT_ARCHITECTURE   # NAM capture arch to acquire: 2 (A2) or 1 (legacy A1)

    # Nested sub-sections (yaml). Training lives in a RunSpec, not here:
    # openamp.spec.load_run_spec reads configs/{emulate,joint}/<name>.yaml.
    enroll: EnrollConfig = field(default_factory=EnrollConfig)

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

    # --- Joint encoder+generator runs under results_dir (own tree: ESR here is ---
    # --- conditioned on an encoder, not directly comparable to emulate/'s) ------
    @property
    def joint_dir(self) -> Path:
        return self.results_dir / "joint"

    def joint_run_dir(self, name: str) -> Path:
        return self.joint_dir / name

    def ensure_dirs(self) -> None:
        for d in (self.manifests_dir, self.captures_dir, self.clean_dir,
                  self.renders_dir, self.results_dir, self.qa_dir):
            d.mkdir(parents=True, exist_ok=True)


# --- Loading -------------------------------------------------------------------
def default_config_path(project_root: Path | None = None) -> Path:
    return (project_root or Path.cwd()) / "configs" / "openamp.yaml"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, # comments, real env wins."""
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
    """Build the one Config from .env + configs/openamp.yaml.

    ``data_dir`` overrides the yaml/env data root (used by tests). ``require_key``
    raises if the API key is absent (metadata-only stages can run without it).
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

    # A key lands in kw only when a source (yaml/env) actually provides it;
    # defaults live once on the dataclass fields above.
    kw: dict = {
        "data_dir": Path(data_dir).expanduser(),
        "publishable_key": key,
        "enroll": _coerce(EnrollConfig, y.get("enroll", {})),
    }

    def _yaml(mapping: dict, name: str, dest: str, cast) -> None:
        if name in mapping:
            kw[dest] = cast(mapping[name])

    def _env(var: str, dest: str, cast) -> None:
        if var in os.environ:
            kw[dest] = cast(os.environ[var])

    if y.get("results_dir"):
        kw["results_dir"] = Path(y["results_dir"]).expanduser()
    _yaml(y, "seed", "seed", int)
    _env("OPENAMP_SEED", "seed", int)              # env wins over yaml
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
