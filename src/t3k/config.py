"""Configuration & settings loading.

All tunables come from environment variables (optionally via a git-ignored
``.env`` file). Secrets (the publishable key) are never written to disk by us;
OAuth tokens are persisted separately by :mod:`t3k.auth`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Fixed API facts (spec §3) --------------------------------------------------
DEFAULT_BASE_URL = "https://www.tone3000.com/api/v1"
DEFAULT_REDIRECT_URI = "http://localhost:3001"

# Client-side rate cap: sit comfortably under the server's 100 req/min (spec §3).
DEFAULT_RATE_LIMIT_RPM = 80

# Selection / quota targets (spec §5.3, §7).
DEFAULT_SEED = 1234
SELECT_TARGET = 430          # headroom over 400 for validation/dedup losses
FINAL_TARGET = 400
NAM_DEFAULT_SAMPLE_RATE = 48_000


def _load_dotenv(path: Path) -> None:
    """Minimal, dependency-free ``.env`` loader.

    Only sets keys that are not already present in ``os.environ`` (real env wins).
    Supports ``KEY=VALUE`` lines, ``#`` comments, optional surrounding quotes.
    ``python-dotenv`` is a listed dependency, but we avoid importing it so that
    the metadata-only stages work even in a bare environment.
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


@dataclass
class Settings:
    """Resolved runtime configuration."""

    publishable_key: str
    base_url: str
    redirect_uri: str
    token_path: Path
    data_dir: Path
    rate_limit_rpm: int
    seed: int
    select_target: int = SELECT_TARGET
    final_target: int = FINAL_TARGET

    # Derived paths.
    captures_dir: Path = field(init=False)
    manifests_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.captures_dir = self.data_dir / "captures"
        self.manifests_dir = self.data_dir / "manifests"

    # Convenience manifest locations (spec §4).
    @property
    def candidates_path(self) -> Path:
        return self.manifests_dir / "candidates.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.manifests_dir / "manifest.parquet"

    @property
    def rejected_path(self) -> Path:
        return self.manifests_dir / "rejected.parquet"

    def ensure_dirs(self) -> None:
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)


def load_settings(project_root: Path | None = None, require_key: bool = False) -> Settings:
    """Build :class:`Settings` from the environment.

    ``require_key`` raises if the publishable key is missing (used by stages that
    actually talk to the API); metadata-inspection stages can run without it.
    """
    root = project_root or Path.cwd()
    _load_dotenv(root / ".env")

    key = os.environ.get("T3K_PUBLISHABLE_KEY", "").strip()
    if require_key and not key:
        raise ConfigError(
            "T3K_PUBLISHABLE_KEY is not set. Create a publishable key at "
            "tone3000.com -> Settings -> API Keys and put it in .env "
            "(see .env.example)."
        )

    token_path = Path(os.environ.get("T3K_TOKEN_PATH", "./.t3k_tokens.json")).expanduser()
    data_dir = Path(os.environ.get("T3K_DATA_DIR", "./data")).expanduser()

    return Settings(
        publishable_key=key,
        base_url=os.environ.get("T3K_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        redirect_uri=os.environ.get("T3K_REDIRECT_URI", DEFAULT_REDIRECT_URI).rstrip("/"),
        token_path=token_path,
        data_dir=data_dir,
        rate_limit_rpm=int(os.environ.get("T3K_RATE_LIMIT_RPM", DEFAULT_RATE_LIMIT_RPM)),
        seed=int(os.environ.get("T3K_SEED", DEFAULT_SEED)),
    )


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""
