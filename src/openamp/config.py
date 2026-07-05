"""Phase 2 configuration (spec §7).

All tunables live in ``configs/phase2.yaml``; this module loads them into a
frozen-ish dataclass and derives every output path from one data root. Randomness
across the whole pipeline is seeded from the single ``seed`` value so a run is
reproducible. The data root follows Phase 1's ``T3K_DATA_DIR`` env var (default
``./data``) so both phases write under the same tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import constants as C

# Defaults mirror configs/phase2.yaml so the pipeline is runnable even if the
# yaml is absent (the yaml simply overrides these).
DEFAULTS = {
    "minutes_total": 40.0,
    "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    "seed": 1234,
    "chunk_seconds": C.DEFAULT_CHUNK_SECONDS,
    "sample_rate": C.SAMPLE_RATE,
    "clip_seconds": C.CLIP_SECONDS,
    "output_format": "flac",
    "results_dirname": "results",
}


class ConfigError(RuntimeError):
    """Raised when the Phase 2 config is malformed."""


@dataclass
class Phase2Config:
    """Resolved Phase 2 runtime configuration."""

    data_dir: Path
    minutes_total: float = DEFAULTS["minutes_total"]
    split_ratios: dict = field(default_factory=lambda: dict(DEFAULTS["split_ratios"]))
    seed: int = DEFAULTS["seed"]
    chunk_seconds: float = DEFAULTS["chunk_seconds"]
    sample_rate: int = DEFAULTS["sample_rate"]
    clip_seconds: float = DEFAULTS["clip_seconds"]
    output_format: str = DEFAULTS["output_format"]
    results_dir: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.results_dir is None:
            self.results_dir = Path.cwd() / DEFAULTS["results_dirname"]
        self.results_dir = Path(self.results_dir)
        self._validate()

    def _validate(self) -> None:
        s = self.split_ratios
        if set(s) != set(C.SPLITS):
            raise ConfigError(f"split_ratios must have keys {C.SPLITS}, got {sorted(s)}")
        total = sum(s.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"split_ratios must sum to 1.0, got {total}")
        if self.minutes_total <= 0:
            raise ConfigError("minutes_total must be positive")

    # --- Derived counts ---------------------------------------------------------
    @property
    def clip_samples(self) -> int:
        return int(round(self.clip_seconds * self.sample_rate))

    @property
    def chunk_samples(self) -> int:
        return int(round(self.chunk_seconds * self.sample_rate))

    # --- Derived paths ----------------------------------------------------------
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
    def clean_dir(self) -> Path:
        return self.data_dir / "clean"

    @property
    def renders_dir(self) -> Path:
        return self.data_dir / "renders"

    @property
    def manifests_dir(self) -> Path:
        return self.data_dir / "manifests"

    @property
    def qa_dir(self) -> Path:
        return self.results_dir / "phase2_qa"

    # Manifest file locations.
    @property
    def phase1_manifest_path(self) -> Path:
        return self.manifests_dir / C.PHASE1_MANIFEST

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
    def report_path(self) -> Path:
        return self.results_dir / "phase2_report.md"

    def clean_split_dir(self, split: str) -> Path:
        return self.clean_dir / split

    def device_render_dir(self, device_id: int) -> Path:
        return self.renders_dir / f"{int(device_id):04d}"

    def ensure_dirs(self) -> None:
        for d in (self.manifests_dir, self.clean_dir, self.renders_dir,
                  self.results_dir, self.qa_dir):
            d.mkdir(parents=True, exist_ok=True)


def default_config_path(project_root: Path | None = None) -> Path:
    root = project_root or Path.cwd()
    return root / "configs" / "phase2.yaml"


def load_config(path: Path | None = None, *, data_dir: Path | None = None) -> Phase2Config:
    """Build :class:`Phase2Config` from ``configs/phase2.yaml`` (if present).

    ``data_dir`` overrides the yaml/env data root (used by tests). Otherwise the
    root is taken from ``T3K_DATA_DIR`` (default ``./data``), matching Phase 1.
    """
    values = dict(DEFAULTS)
    cfg_path = path or default_config_path()
    if cfg_path and Path(cfg_path).is_file():
        import yaml  # local import; pyyaml is a listed dep

        try:
            loaded = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{cfg_path}: invalid YAML ({exc})") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"{cfg_path}: top-level YAML must be a mapping")
        values.update(loaded)

    if data_dir is None:
        data_dir = Path(os.environ.get("T3K_DATA_DIR", values.get("data_dir") or "./data"))
    data_dir = Path(data_dir).expanduser()

    results_dir = values.get("results_dir")
    return Phase2Config(
        data_dir=data_dir,
        minutes_total=float(values["minutes_total"]),
        split_ratios={k: float(v) for k, v in values["split_ratios"].items()},
        seed=int(values["seed"]),
        chunk_seconds=float(values["chunk_seconds"]),
        sample_rate=int(values["sample_rate"]),
        clip_seconds=float(values["clip_seconds"]),
        output_format=str(values["output_format"]),
        results_dir=Path(results_dir).expanduser() if results_dir else None,
    )
