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


# The exported ``architecture`` string names the concrete net class. NAM's
# ``BaseNet`` is abstract, so construction must go through the subclass.
_ARCH_ALIASES = {
    "wavenet": "WaveNet",
    "lstm": "LSTM",
    "convnet": "ConvNet",
    "linear": "Linear",
}


def _translate_config(arch: str, config: dict) -> dict:
    """Map exported .nam config keys onto the current constructor kwargs.

    Older exports (schema ``version`` 0.5.x) name the WaveNet fields ``layers``
    and ``head``; the installed ``_WaveNet.__init__`` expects ``layers_configs``
    and ``head_config``. Only rename when the target key is absent so newer
    exports pass through untouched.
    """
    config = dict(config)
    if arch == "WaveNet":
        if "layers" in config and "layers_configs" not in config:
            config["layers_configs"] = config.pop("layers")
        if "head" in config and "head_config" not in config:
            config["head_config"] = config.pop("head")
    return config


def _import_nam_model(data: dict):
    """Reconstruct a runnable NAM model from parsed .nam data.

    Selects the concrete architecture class, remaps the export config to the
    installed constructor's kwargs, builds via ``init_from_config``, then loads
    the weights. Isolated here so only this file changes across NAM versions.
    """
    try:
        from nam import models as nam_models  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise NamBackendError(
            "neural-amp-modeler could not be imported. Install the validation "
            "extras (`pip install -e .[validate]`); if it is installed but the "
            "import fails on `pkg_resources`, pin `setuptools<81` (NAM 0.x still "
            f"imports pkg_resources). Underlying error: {exc}"
        ) from exc

    arch = str(data.get("architecture") or "").strip()
    cls = getattr(nam_models, _ARCH_ALIASES.get(arch.lower(), arch), None)
    if cls is None:
        raise NamBackendError(f"unknown/unsupported NAM architecture: {arch!r}")

    config = _translate_config(arch, data.get("config") or {})
    weights = data.get("weights")

    try:
        net = cls.init_from_config(config)
    except Exception as exc:
        raise NamBackendError(
            f"could not construct a {arch} model from the export config: {exc}"
        ) from exc
    _load_weights(net, weights)
    return net


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
