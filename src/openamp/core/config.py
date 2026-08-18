"""One configuration object for the whole pipeline.

- **API access** (key, base URL, token path, rate limit): environment / git-ignored ``.env`` (``OPENAMP_*``).
- **Pipeline knobs** (corpus, render, model, train, eval): ``configs/openamp.yaml``; ``model``/``train``/``eval`` are nested sub-sections.
- **Every path** derives from one ``data_dir`` (inputs/manifests/renders) + one ``results_dir`` (reports/checkpoints); a run is fully reproducible from the single ``seed``.
- ``load_config()`` merges ``.env`` + yaml into a :class:`Config`.
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

    - Self-contained sub-pipeline: every architecture size is a plain knob here, so
      exploring sizes is copy-a-config-and-change-numbers.
    - Read from the ``emulate:`` yaml section; a per-run file under
      ``configs/emulate/<name>.yaml`` overrides just this section, run named after
      the file stem.
    """

    # --- Architecture ------------------------------------------------------------
    # "film_tcn" paper FiLM-TCN | "film_wavenet" corpus's own NAM A2 WaveNet
    # topology, FiLM-conditioned | "mlpfilm_wavenet" same A2 topology, small-MLP
    # FiLM generator | "delta_wavenet" same A2 topology, no FiLM: embedding writes
    # a low-rank residual onto each layer's conv weights | "tabledelta_wavenet"
    # that residual free and per-device rather than low-rank (the full-rank
    # control; derives its own embedding_dim, has no capacity knob, and cannot be
    # enrolled). See emulate/wavenet.py.
    arch: str = "film_tcn"

    # --- FiLM-TCN model (fully parametric; paper default = values below) --------
    blocks: int = 2                  # dilation resets per block
    layers_per_block: int = 8
    channels: int = 16
    kernel_size: int = 3
    dilation_growth: int = 2

    # --- FiLM-WaveNet model (kernel/dilation schedule is the fixed A2 one) ------
    wn_channels: int = 8             # A2 full-width value; the width sweep knob
    # Per-layer nonlinearity: "leakyrelu" (slope 0.01, what every A2 capture uses)
    # | "tanh" (NAM's other A2-schema activation, still plugin-playable). Validated
    # in openamp/emulate/wavenet.py.
    wn_activation: str = "leakyrelu"

    # --- MLP-FiLM WaveNet ("mlpfilm_wavenet"): the FiLM generator only -----------
    # Hidden width of the per-layer embedding -> (gamma, beta) MLP (Linear(E,H) ->
    # cond_activation -> Linear(H,2C), independent per layer). Ignored by film_tcn
    # and film_wavenet. At embedding_dim 256 / wn_channels 8: 16 params-matches
    # film_wavenet's single Linear (4,384 vs 4,112/layer), 32 is ~2x, 64 quadruples
    # the plugin bundle.
    cond_hidden: int = 32
    # Nonlinearity inside that MLP: same two options as wn_activation. Parameter-
    # free, so it's resume-structural (see train.py's _STRUCTURAL_KEYS).
    cond_activation: str = "leakyrelu"

    # --- Delta WaveNet ("delta_wavenet"): the weight-residual generator ----------
    # Rank of the per-layer map from the device embedding to a residual on that
    # layer's conv kernel, bias and mixin gain (delta = scale * coeff(e) @
    # normalize(basis), one shared basis of `delta_rank` unit-norm directions).
    # Arch's only capacity knob; ignored by the other archs. At embedding_dim 256 /
    # wn_channels 8: 16,263 * delta_rank + 23 params — rank 6 params-matches
    # film_wavenet's 23 Linears (97,601 vs 94,576), 8 is ~1.36x.
    delta_rank: int = 8

    # --- Shared model knobs ------------------------------------------------------
    # Per-device embedding, consumed at every layer. Ignored by tabledelta_wavenet,
    # which derives its width from the schedule (a device's row is its whole
    # weight residual).
    embedding_dim: int = 64
    # One-to-one baseline: >=0 trains a single-device model on that device_id
    # (same class, n_devices=1), for the one-to-many gap reference.
    single_device: int = -1

    # --- Training (one script; Adam + reduce-on-plateau to val plateau) ---------
    # Fraction of render-ok devices held out of training entirely, so Phase 5 can
    # enroll them as truly unseen (reference: 90/10 device split). Drawn once,
    # persisted to `emulate_holdout.txt`; <= 0 disables. Not applied to
    # `single_device`.
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




@dataclass
class JointConfig:
    """Encoder trained jointly with a FiLM generator (see joint_model/guidelines.md).

    - Read from the ``joint:`` yaml section. A joint run reads **two** sections:
      ``emulate:`` is the generator and every shared training knob (batch size,
      epochs, learning rate, loss weights), and this holds only what is new. Same
      "full copy, never a delta" rule as the other one.
    - Two knobs deliberately do **not** live here, because duplicating them would
      create a second source of truth for one number:
      ``emulate.embedding_dim`` is the width of ``e`` (it is the generator's FiLM
      input — the interface contract), and ``emulate.lr`` is the generator's
      learning rate. Only the *encoder's* learning rate is new.
    """

    # --- Conditioning source ------------------------------------------------------
    # "wavenet" trains an encoder on a short (dry, wet) reference clip. "fingerprint"
    # looks up a precomputed frozen-codec vector by device id, training only a small
    # adapter onto it. The lookup exists because a codec fingerprint is a statistic
    # over frames and needs ~85 s of audio to be reliable (measured own-nearest over
    # 450 amps: 99.8% at 85 s, 61.1% at 22 s, 17.8% at 12 s) -- an order of magnitude
    # more than a reference window holds. Under "fingerprint" the whole reference
    # block below (ref_seconds, ref_different_file, ref_min_gap_seconds) is inert.
    enc_kind: str = "wavenet"
    # Fingerprint run directory holding pooled.npz + meta.json, as written by
    # scripts/encode_fingerprints.py. Required under enc_kind "fingerprint".
    fp_path: str = ""
    # What happens to the pooled vector before the adapter. NO corpus statistics are
    # permitted: every option uses either the single fixed dry vector (device -1, the
    # same signal with no amp) or per-sample arithmetic.
    #   "dry_l2"  subtract the dry null, then L2. Measured on the 450 DAC vectors,
    #             dry subtraction drops mean pairwise cosine 0.886 -> 0.576: most of a
    #             raw fingerprint is "what the codec does to this audio", not "what
    #             this amp does". L2 then does for the fingerprint what enc_normalize
    #             does for e. It discards ||fp - dry||, which is real amp information
    #             (range 1.25-95.9) -- "dry" is the arm that keeps it.
    #   "dry" | "l2" | "raw"
    # Note a linear adapter's bias absorbs a pure shift, so "dry" alone differs from
    # "raw" only when composed with fp_layernorm.
    fp_preprocess: str = "dry_l2"
    # LayerNorm(D_fp) as the adapter's first layer: per-sample, learned affine, no
    # corpus statistics. Off by default because DAC's vector is mean + std
    # concatenated, two halves on different scales that one LayerNorm would mix.
    fp_layernorm: bool = False

    # --- Reference segment (window B) --------------------------------------------
    # Length of the clip the encoder reads. Long enough that pooling averages
    # content away, short enough to fit beside the generator on one card: a
    # WaveNet never downsamples, so this length survives all 23 layers.
    ref_seconds: float = 2.0
    # Draw window B from a different corpus file than window A. This is what stops
    # the embedding carrying content instead of tone, and on a multi-file corpus it
    # is free. Falls back to a gapped same-file draw when a split has one file.
    ref_different_file: bool = True
    ref_min_gap_seconds: float = 1.0   # same-file fallback only
    sources: list = field(default_factory=list)   # [] = every source

    # --- Encoder (see joint_model/wavenet_encoder.py) ----------------------------
    # Pooling emits 2 * enc_channels numbers, so this — not embedding_dim — is the
    # real bottleneck on what can be said about an amp. Guidelines §7 puts it at
    # 16-32; 16 gives a 32-d pooled vector, still twice the spec's default width.
    enc_channels: int = 16
    enc_activation: str = "tanh"       # "leakyrelu" | "tanh", as wn_activation
    enc_attn_hidden: int = 64          # attentive-pooling bottleneck
    # Hidden width of the pooled -> embedding_dim projection. 0 = derive it as
    # pooled_dim // 4, which is the only sane default once pooled_dim stops being
    # a fixed 2*enc_channels.
    enc_proj_hidden: int = 128
    enc_normalize: bool = False        # L2-project e onto the unit sphere

    # --- Pooling head (see joint_model/wavenet_encoder.py) ------------------------
    # "stats" is the original 2*enc_channels readout; "multitap" decouples pooled
    # width from backbone width via taps + expansion + K attention heads. The two
    # have disjoint parameters, so this is structural — a checkpoint written under
    # one cannot load under the other.
    enc_head: str = "stats"
    enc_taps: list = field(default_factory=lambda: [6, 13, 22])   # A2 dilation-run ends
    enc_tap_stride: int = 8            # avg-pool before the head; memory, not modelling
    enc_expand_dim: int = 256          # E: pooled width is enc_n_heads * 2 * E
    enc_n_heads: int = 2               # K
    # BatchNorm subtracts a dataset-level mean, so every per-amp deviation survives
    # into the pool; "group" is batch-independent but subtracts a *per-sample* mean
    # over time, which deletes the statistic the pool then measures.
    enc_expand_norm: str = "batch"     # "batch" | "group" | "none"
    enc_norm_groups: int = 32          # "group" only; clamped to divide enc_expand_dim
    # Recompute encoder activations in backward instead of storing them: ~35%
    # slower, ~2.2 GB cheaper at batch 16 / enc_channels 32.
    grad_checkpoint_encoder: bool = False

    # --- Optimization -------------------------------------------------------------
    # Separate from the generator's lr because the two halves are differently
    # shaped and a dead encoder looks exactly like a working one from ESR alone
    # (guidelines §9). Equal by default; lower this first if the encoder collapses.
    enc_lr: float = 4.0e-3
    # Weight decay for the encoder/adapter group only; < 0 inherits emulate.weight_decay.
    # It exists for the fingerprint arm, whose deepest failure mode is memorization: a
    # WaveNet encoder sees a *different* reference window on every draw, an implicit
    # regularizer, whereas the adapter sees the identical vector for a given amp on
    # every one of pairs_per_epoch draws. Shrinkage is the natural guard, and
    # emulate.weight_decay is 3.17e-7, i.e. effectively none.
    enc_weight_decay: float = -1.0

    def __post_init__(self) -> None:
        if self.ref_seconds <= 0:
            raise ConfigError(f"joint.ref_seconds must be positive, got {self.ref_seconds}")
        if self.ref_min_gap_seconds < 0:
            raise ConfigError("joint.ref_min_gap_seconds must be >= 0, got "
                              f"{self.ref_min_gap_seconds}")
        if self.enc_kind not in ("wavenet", "fingerprint"):
            raise ConfigError("joint.enc_kind must be 'wavenet' or 'fingerprint', got "
                              f"{self.enc_kind!r}")
        if self.fp_preprocess not in ("dry_l2", "dry", "l2", "raw"):
            raise ConfigError("joint.fp_preprocess must be one of ('dry_l2', 'dry', "
                              f"'l2', 'raw'), got {self.fp_preprocess!r}")
        if self.enc_kind == "fingerprint" and not str(self.fp_path).strip():
            raise ConfigError("joint.enc_kind 'fingerprint' needs joint.fp_path, the "
                              "directory holding pooled.npz")
        if self.enc_channels < 1:
            raise ConfigError(f"joint.enc_channels must be >= 1, got {self.enc_channels}")
        if self.enc_attn_hidden < 1:
            raise ConfigError(f"joint.enc_attn_hidden must be >= 1, got {self.enc_attn_hidden}")
        if self.enc_proj_hidden < 0:
            raise ConfigError("joint.enc_proj_hidden must be >= 0 (0 = derive as "
                              f"pooled_dim // 4), got {self.enc_proj_hidden}")
        if self.enc_head not in ("stats", "multitap"):
            raise ConfigError("joint.enc_head must be 'stats' or 'multitap', got "
                              f"{self.enc_head!r}")
        taps = [int(t) for t in self.enc_taps]
        if not taps or any(t < 0 for t in taps):
            raise ConfigError(f"joint.enc_taps must be non-empty layer indices >= 0, "
                              f"got {self.enc_taps}")
        if taps != sorted(set(taps)):
            raise ConfigError("joint.enc_taps must be strictly increasing and unique, "
                              f"got {self.enc_taps}")
        if self.enc_tap_stride < 1:
            raise ConfigError(f"joint.enc_tap_stride must be >= 1, got {self.enc_tap_stride}")
        if self.enc_expand_dim < 1 or self.enc_n_heads < 1:
            raise ConfigError("joint.enc_expand_dim and joint.enc_n_heads must be >= 1")
        if self.enc_expand_norm not in ("batch", "group", "none"):
            raise ConfigError("joint.enc_expand_norm must be one of ('batch', 'group', "
                              f"'none'), got {self.enc_expand_norm!r}")
        if self.enc_norm_groups < 1:
            raise ConfigError(f"joint.enc_norm_groups must be >= 1, got {self.enc_norm_groups}")
        if self.enc_lr <= 0:
            raise ConfigError(f"joint.enc_lr must be positive, got {self.enc_lr}")
        bad = [s for s in self.sources if s not in (C.SOURCE_EGDB, C.SOURCE_SWEEP)]
        if bad:
            raise ConfigError(f"joint.sources must be drawn from "
                              f"{(C.SOURCE_EGDB, C.SOURCE_SWEEP)}, got {bad}")


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
    # Must go to BOTH `/tones/search` and `/models` -- `/models` defaults to A1
    # and silently hands back A1 captures otherwise.
    architecture: int = DEFAULT_ARCHITECTURE

    # Nested emulation sub-section (yaml).
    emulate: EmulateConfig = field(default_factory=EmulateConfig)
    # Nested tone-encoder sub-section (yaml).
    # Nested joint encoder+generator sub-section (yaml).
    joint: JointConfig = field(default_factory=JointConfig)

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

    # --- Tone-encoder runs under results_dir ------------------------------------
    # A separate tree from `emulate/`: encoder runs log a contrastive loss, not
    # ESR, and the two must not end up in one comparison table (decisions.md §6 is
    # the standing warning about cross-comparing metrics that share a name).
    # --- Joint encoder+generator runs under results_dir -------------------------
    # Its own tree again: a joint run logs ESR like `emulate/` does, but that ESR
    # is conditioned on an encoder rather than a table, so the two numbers answer
    # different questions and must not land in one comparison table.
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
    """Minimal, dependency-free ``.env`` loader.

    - Only sets keys not already in ``os.environ`` (the real environment wins).
    - Supports ``KEY=VALUE`` lines, ``#`` comments, optional surrounding quotes.
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

    - ``data_dir`` overrides the yaml/env data root (used by tests).
    - ``require_key`` raises if the API key is absent (stages that call the API
      set it; metadata-only stages can run without it).
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

    # Override only what a source (yaml/env) actually provides; defaults live once
    # on the dataclass fields above. A key lands in ``kw`` only when present.
    kw: dict = {
        "data_dir": Path(data_dir).expanduser(),
        "publishable_key": key,
        "emulate": _coerce(EmulateConfig, y.get("emulate", {})),
        "joint": _coerce(JointConfig, y.get("joint", {})),
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
