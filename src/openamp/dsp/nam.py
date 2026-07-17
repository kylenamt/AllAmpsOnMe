"""Thin, best-effort adapter over the ``neural-amp-modeler`` package.

Loading an exported ``.nam`` and running inference is the one place that
depends on NAM's internal Python API, which varies across releases. It is
isolated here so the rest of the pipeline (and its tests) never imports
torch/NAM. If the installed NAM version exposes a different entry point,
adapt ONLY this file.

Two entry points:
- :func:`default_load_and_render` — CPU render for the acquisition validate
  stage (5 s probe).
- :func:`load_model` — GPU ``inference_mode`` forward callable + the model's
  finite receptive field R, used by the render stage to process whole files
  chunk-by-chunk without boundary artifacts.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from openamp.core.util import sha256_file

log = logging.getLogger("openamp.nam")


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
    submodels = config.get("submodels") if is_container(data) else None
    return {
        "version": data.get("version"),
        "architecture": data.get("architecture"),
        "sample_rate": data.get("sample_rate") or config.get("sample_rate"),
        "config": config,
        "metadata": data.get("metadata") or {},
        # An A2 container carries no top-level weights: each width variant holds
        # its own. Look inside, or every A2 capture reads as weightless.
        "has_weights": (bool(submodels) and all(s.get("model", {}).get("weights")
                                                for s in submodels)
                        if submodels is not None else bool(data.get("weights"))),
        "n_submodels": len(submodels) if submodels is not None else 0,
    }


# TONE3000 "A2" exports a ``SlimmableContainer``: not a net, but a set of width
# variants of one capture (Slimmable NAM, arXiv:2511.07470). The variants are a
# runtime CPU/quality dial -- each models the same device, the narrow ones less
# accurately. ``max_value`` is the variant's width fraction, so the 1.0 variant
# is the reference full-quality net and is the only one we ever render with.
CONTAINER_ARCH = "SlimmableContainer"


def is_container(data: dict) -> bool:
    return str(data.get("architecture") or "").strip() == CONTAINER_ARCH


def full_width_submodel(data: dict) -> dict:
    """The widest (full-quality) submodel of an A2 container export."""
    submodels = (data.get("config") or {}).get("submodels") or []
    if not submodels:
        raise NamBackendError(f"{CONTAINER_ARCH} export has no submodels")
    widest = max(submodels, key=lambda s: float(s.get("max_value") or 0.0))
    inner = widest.get("model")
    if not isinstance(inner, dict):
        raise NamBackendError(f"{CONTAINER_ARCH} submodel has no 'model' object")
    return inner


def _upgrade_legacy_wavenet(data: dict) -> dict:
    """Rewrite a pre-0.7 (A1) WaveNet export into the schema NAM 0.13 parses.

    The old layer schema carries ``head_size``/``head_bias``; the current
    ``LayerArray`` wants a ``head`` object. The old head conv is a 1x1, so the
    translation is exact -- verified bit-identical against NAM 0.11 output on the
    A1 corpus. ``gated`` has no tested mapping onto the new ``gating_mode``, so a
    gated export is refused rather than silently rendered as something else.
    """
    if data.get("architecture") != "WaveNet":
        return data
    layers = (data.get("config") or {}).get("layers") or []
    if not any("head_size" in layer for layer in layers):
        return data  # already the current schema

    data = copy.deepcopy(data)
    for layer in data["config"]["layers"]:
        if layer.pop("gated", False):
            raise NamBackendError(
                "gated WaveNet exports are not supported (no verified mapping onto "
                "the current 'gating_mode' schema)")
        if "head" not in layer:
            layer["head"] = {
                "out_channels": layer.pop("head_size"),
                "kernel_size": 1,
                "bias": bool(layer.pop("head_bias", False)),
            }
    return data


def _import_nam_model(data: dict):
    """Reconstruct a runnable NAM net from parsed .nam data (A1 or A2).

    A2 containers resolve to their full-width submodel; A1 exports are upgraded
    to the current schema first. Everything then goes through NAM's own
    ``init_from_nam``, so only this file changes across NAM versions.
    """
    try:
        from nam.models import init_from_nam  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise NamBackendError(
            "neural-amp-modeler >=0.13 could not be imported (it provides "
            "`init_from_nam` and the A2 layer schema). Install the package: "
            f"`pip install -e .`. Underlying error: {exc}"
        ) from exc

    if is_container(data):
        data = full_width_submodel(data)
    data = _upgrade_legacy_wavenet(data)

    arch = str(data.get("architecture") or "").strip()
    try:
        return init_from_nam(data)
    except NamBackendError:
        raise
    except KeyError as exc:
        raise NamBackendError(f"unknown/unsupported NAM architecture: {arch!r}") from exc
    except Exception as exc:
        raise NamBackendError(
            f"could not construct a {arch} model from the export config: {exc}"
        ) from exc


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


# --------------------------------------------------------------------------
# GPU rendering: forward callable + receptive field
# --------------------------------------------------------------------------

# Fallback if a model's receptive field can't be read off the net/config. Chosen
# large enough to bound any realistic A1/A2 stack; over-estimating R only costs a
# little extra context compute, never correctness.
DEFAULT_RECEPTIVE_FIELD = 8192

RenderModelFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class LoadedModel:
    forward: RenderModelFn        # mono float32 -> same-length mono float32 (on GPU)
    receptive_field: int
    sample_rate: int
    nam_sha256: str


def _torch(device: str):
    try:
        import torch  # local heavy import
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise NamBackendError(
            "PyTorch is not installed. Install the package: `pip install -e .`"
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise NamBackendError(
            "CUDA is not available but a GPU render was requested. "
            "Install a CUDA torch build or pass device='cpu' (slow)."
        )
    return torch


def _receptive_field(net, config: dict) -> int:
    """Best-effort receptive field R. Prefer the net's own property; fall back
    to a WaveNet dilation sum; finally a safe constant."""
    for attr in ("receptive_field", "_receptive_field"):
        val = getattr(net, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                val = None
        if isinstance(val, (int, np.integer)) and val > 0:
            return int(val)

    # WaveNet-style config: R = 1 + sum over layers of dilation*(kernel_size-1).
    try:
        layers = config.get("layers") or config.get("layers_configs") or []
        total = 0
        for layer in layers if isinstance(layers, list) else []:
            kernel = int(layer.get("kernel_size", 1))
            for d in layer.get("dilations", []) or []:
                total += int(d) * (kernel - 1)
        if total > 0:
            return total + 1
    except Exception:
        pass
    log.warning("could not determine receptive field; using default %d", DEFAULT_RECEPTIVE_FIELD)
    return DEFAULT_RECEPTIVE_FIELD


def load_model(path: str | Path, *, device: str = "cuda",
               default_sample_rate: int = 48_000) -> LoadedModel:
    """Load a ``.nam`` export into a GPU forward callable + receptive field."""
    torch = _torch(device)
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = parse_nam(path)

    net = _import_nam_model(data)  # version-tolerant loader shared with validate
    net.eval()
    net.to(device)

    rf = _receptive_field(net, data.get("config") or {})
    sr = int(summary.get("sample_rate") or default_sample_rate)

    def forward(x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        with torch.inference_mode():
            try:
                y = net(xt)
            except Exception:
                # Some nets expect (batch, time); collapse back afterwards.
                y = net(xt.reshape(1, -1))
        return y.detach().to("cpu").numpy().reshape(-1).astype(np.float32)

    return LoadedModel(forward=forward, receptive_field=rf, sample_rate=sr,
                       nam_sha256=sha256_file(path))
