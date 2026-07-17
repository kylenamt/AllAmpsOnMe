"""Small shared helpers used across pipeline stages."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (manifest timestamps)."""
    return datetime.now(timezone.utc).isoformat()
