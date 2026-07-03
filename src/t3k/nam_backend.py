"""Thin, best-effort adapter over the ``neural-amp-modeler`` package.

Loading an exported ``.nam`` and running inference is the one place that depends
on NAM's internal Python API, which varies across releases. It is isolated here
and dependency-injected into :mod:`t3k.validate` so the rest of the pipeline (and
its tests) never import torch/NAM. If the installed NAM version exposes a
different entry point, adapt ONLY this file (spec §3, §9 — adapt & document).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NamBackendError(RuntimeError):
    """Raised when a .nam file cannot be loaded or rendered."""


def parse_nam(path: str | Path) -> dict:
    """Parse the .nam JSON and return a small summary (no torch needed)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NamBackendError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise NamBackendError(f"{path}: top-level JSON is not an object")
    config = data.get("config") or {}
    return {
        "version": data.get("version"),
        "architecture": data.get("architecture"),
        "sample_rate": data.get("sample_rate") or config.get("sample_rate"),
        "config": config,
        "metadata": data.get("metadata") or {},
        "has_weights": bool(data.get("weights")),
    }


def _import_nam_model(data: dict):
    """Reconstruct a runnable NAM model from parsed .nam data.

    Tries the known construction paths across NAM versions. Kept narrow and
    well-commented so it is easy to pin to a specific version if needed.
    """
    try:
        from nam.models.base import BaseNet  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise NamBackendError(
            "neural-amp-modeler is not installed. Install the validation extras: "
            "`pip install -e .[validate]` (or `pip install -r requirements.txt`)."
        ) from exc

    config = data.get("config") or {}
    weights = data.get("weights")

    # Preferred: a single classmethod that rebuilds arch+weights from the export.
    for ctor_name in ("init_from_config", "from_config"):
        ctor = getattr(BaseNet, ctor_name, None)
        if ctor is None:
            continue
        try:
            net = ctor(config)
        except Exception:
            continue
        _load_weights(net, weights)
        return net

    raise NamBackendError("could not construct a NAM model from the export config")


def _load_weights(net, weights) -> None:
    if weights is None:
        return
    for method in ("import_weights", "set_weights", "load_weights"):
        fn = getattr(net, method, None)
        if callable(fn):
            try:
                fn(weights)
                return
            except Exception:
                continue
    # If none worked, the caller's forward pass will surface a mismatch.


def default_load_and_render(path: str | Path, signal: np.ndarray,
                            sample_rate: int) -> np.ndarray:
    """Load ``path`` and render ``signal`` (mono float32) through it on CPU.

    Returns a float32 array the same length as ``signal``.
    """
    try:
        import torch  # noqa: F401  (local import; heavy)
    except Exception as exc:  # pragma: no cover
        raise NamBackendError("PyTorch is not installed (needed for validation).") from exc

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    net = _import_nam_model(data)

    import torch

    net.eval()
    x = torch.tensor(np.asarray(signal, dtype=np.float32))
    with torch.no_grad():
        try:
            y = net(x)
        except Exception:
            # Some nets expect a (batch, time) or (batch, 1, time) shape.
            y = net(x.reshape(1, -1))
    y = y.detach().cpu().numpy().reshape(-1).astype(np.float32)
    return y
