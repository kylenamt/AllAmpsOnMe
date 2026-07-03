#!/usr/bin/env python3
"""CLI entry point (spec §4). Runs the ``t3k`` command group without requiring an
editable install: it adds ``src/`` to sys.path, then delegates to
``t3k.cli.main``. Equivalent to the ``t3k`` console script from ``pip install``.

Usage:
    python scripts/t3k.py <command> [options]
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t3k.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
