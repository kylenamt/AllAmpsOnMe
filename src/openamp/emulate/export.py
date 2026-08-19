"""Morph-plugin exports: weight bundle + pasteable embedding profiles.

- The plugin runs the trained FiLM-WaveNet by weight folding: for a fixed
  embedding e, film_wavenet/mlpfilm_wavenet's per-layer FiLM
  (z = gamma*(conv(x)+mixin(clean))+beta) is absorbed into the layer's own
  weights (conv.weight *= gamma, conv.bias = gamma*bias+beta, mixin *= gamma),
  leaving a bit-for-bit stock NAM A2 WaveNet. delta_wavenet folds the same way
  one step upstream (conv.weight += dW(e), etc.) since its generator already
  emits a weight residual. folded_model() is the Python reference for this;
  forward_with_embedding() is the unfolded reference; test_export.py pins the
  two equal to the trained model.
- Two artifacts:
  - Bundle (export_bundle) -- one JSON file with every static weight, per-layer
    FiLM/delta matrices, the A2 schedule, and built-in profiles (tensors as
    base64 float32). Meant to be embedded in the plugin binary.
  - Profile (export_profile) -- the small pasteable JSON users exchange:
    {"aaom_profile", "name", "run", "dim", "embedding"}. run is the checkpoint's
    sha256 prefix, so the plugin can reject profiles from a different network.
- All three A2 archs export (film_wavenet, mlpfilm_wavenet, delta_wavenet),
  discriminated by arch.type. Each non-legacy arch omits film_w/film_b, so a
  reader that predates the arch fails on a missing key instead of folding garbage.
- WaveNet-only: the plugin ships the A2 topology, so exporting film_tcn is a hard error.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

from openamp.core.util import sha256_file
from openamp.emulate.wavenet import (NAM_ACTIVATION_NAMES, DELTA_PART_NAMES,
                                     DeltaWaveNet, FiLMWaveNet, MLPFiLMWaveNet,
                                     TableDeltaWaveNet, TableDeltaWaveNetLayer)

__all__ = ["export_bundle", "export_profile", "export_profiles", "folded_model",
           "forward_with_embedding", "run_sha8",
           "BUNDLE_VERSION", "PROFILE_VERSION"]

BUNDLE_VERSION = 1
PROFILE_VERSION = 1


# --- Small helpers -------------------------------------------------------------
def run_sha8(run_dir: Path) -> str:
    """First 8 hex chars of the checkpoint's sha256 — the profile/bundle key."""
    return sha256_file(Path(run_dir) / "checkpoint.pt")[:8]


def _blob(t: torch.Tensor) -> dict:
    """Tensor -> JSON blob: little-endian float32, C-order, base64."""
    a = np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4")
    return {"shape": list(a.shape), "dtype": "f32",
            "data": base64.b64encode(a.tobytes()).decode("ascii")}


def _unblob(b: dict) -> np.ndarray:
    """JSON blob -> ndarray (the test's round-trip inverse of :func:`_blob`)."""
    a = np.frombuffer(base64.b64decode(b["data"]), dtype="<f4")
    return a.reshape(b["shape"]).copy()


def _require_wavenet(model) -> FiLMWaveNet:
    if not isinstance(model, FiLMWaveNet):     # the other A2 archs subclass it
        raise RuntimeError("morph exports support film_wavenet / mlpfilm_wavenet / "
                           f"delta_wavenet runs only, got {type(model).__name__}")
    return model


# --- Folding + reference forward ------------------------------------------------
@torch.no_grad()
def _fold_layer(layer, emb: torch.Tensor, channels: int) -> None:
    """Fold one layer's conditioning at ``emb`` into its own weights, in place.

    The model neutralizes the generator afterwards (``neutralize_conditioning``),
    so the copy's forward ignores device_idx entirely for *any* value.

    - FiLM (``film_wavenet`` / ``mlpfilm_wavenet``): per-channel affine,
      absorbed by scaling conv/mixin rows and shifting the bias.
    - Delta (``delta_wavenet``): the generator already speaks in weight space,
      so the fold is the addition the layer would have done.
    - Table (``tabledelta_wavenet``): same addition, residual read straight
      from the device's row. The layer's part mask is honoured.
    """
    if isinstance(layer, TableDeltaWaveNetLayer):
        dw, db, dm = layer.split_delta(emb.reshape(1, -1))
        use_kernel, use_bias, use_mixin = layer.delta_parts
        if use_kernel:
            layer.conv.weight.add_(dw[0])
        if use_bias:
            layer.conv.bias.add_(db[0])
        if use_mixin:
            layer.input_mixer.weight.add_(dm[0].reshape(-1, 1, 1))
        return
    if hasattr(layer, "delta"):
        dw, db, dm = layer.delta(emb)                       # [1,C,C,K], [1,C], [1,C]
        layer.conv.weight.add_(dw[0])
        layer.conv.bias.add_(db[0])
        layer.input_mixer.weight.add_(dm[0].reshape(-1, 1, 1))
        return
    gamma, beta = layer.film(emb).chunk(2, dim=-1)          # [C], [C]
    layer.conv.weight.mul_(gamma[:, None, None])
    layer.conv.bias.mul_(gamma).add_(beta)
    layer.input_mixer.weight.mul_(gamma[:, None, None])


@torch.no_grad()
def folded_model(model: FiLMWaveNet, emb: torch.Tensor) -> FiLMWaveNet:
    """Deep copy with the conditioning at ``emb`` folded into the conv weights.

    The copy's conditioning generators are neutralized, so its forward — with
    *any* device_idx — is the plain A2 WaveNet the plugin runs. ``emb`` is a
    ``[embedding_dim]`` vector (any morph point, not just rows).
    """
    _require_wavenet(model)
    import copy
    emb = emb.detach().reshape(-1).to(next(model.parameters()).dtype)
    if emb.numel() != model.embedding_dim:
        raise ValueError(f"embedding has {emb.numel()} dims, "
                         f"model expects {model.embedding_dim}")
    m = copy.deepcopy(model).eval()
    for layer, cond in m.layer_conditioning(emb):
        _fold_layer(layer, cond, m.channels)
    m.neutralize_conditioning()
    return m


@torch.no_grad()
def forward_with_embedding(model: FiLMWaveNet, x: torch.Tensor,
                           emb: torch.Tensor) -> torch.Tensor:
    """Conditioned forward with one explicit embedding vector, not a table row.

    A no-grad convenience wrapper over :meth:`FiLMWaveNet.forward_emb` that
    broadcasts a single ``[E]`` vector across the batch: this is the unfolded
    reference the folded path is tested against, and the semantics of an
    arbitrary morph point ``emb = sum_i w_i * e_i``. Being the same call the
    network itself makes, it cannot drift out of step with ``forward``.
    """
    _require_wavenet(model)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    e = emb.detach().reshape(1, -1).to(x.dtype).expand(x.shape[0], -1)
    return model.forward_emb(x, e)


# --- Bundle --------------------------------------------------------------------
def _cond_blobs(layer) -> dict:
    """One layer's conditioning weights, in the shape the plugin's re-fold expects.

    - film_wavenet: ``film_w``/``film_b`` (day-one keys, byte-identical).
    - mlpfilm_wavenet: ``film_layers`` (Linears in input->output order; apply
      ``cond_activation`` *between* them, not after the last).
    - delta_wavenet: ``delta = (coeff_w @ e + coeff_b) @ delta_basis``, split
      into kernel/bias/mixin residuals (``C*C*K``/``C``/``C`` entries).
    - tabledelta_wavenet: no generator at all -- a profile row *is* the delta,
      so only ``delta_split`` (the slice widths) is needed; profiles are
      correspondingly large (~10k floats vs 256).
    - Every non-legacy arch omits ``film_w``/``film_b``: ``arch.type`` is the
      intended discriminator, and a reader that ignores it dies on a missing
      key instead of silently folding garbage.
    """
    if isinstance(layer, TableDeltaWaveNetLayer):
        return {"delta_split": [layer.n_weight, layer.channels, layer.channels]}
    if hasattr(layer, "delta"):
        return {"delta_coeff_w": _blob(layer.delta.coeff.weight),
                "delta_coeff_b": _blob(layer.delta.coeff.bias),
                "delta_basis": _blob(layer.delta.effective_basis())}
    film = layer.film
    if isinstance(film, torch.nn.Linear):
        return {"film_w": _blob(film.weight), "film_b": _blob(film.bias)}
    return {"film_layers": [{"w": _blob(l.weight), "b": _blob(l.bias)}
                            for l in film.modules()
                            if isinstance(l, torch.nn.Linear)]}


@torch.no_grad()
def export_bundle(run_dir: Path, out_path: Path, *,
                  profile_device_ids: list[int] | None = None,
                  profile_pairs: list[str] | None = None) -> dict:
    """Write the plugin weight bundle for a run; returns the (blob-free) summary.

    Always includes a ``"Table mean"`` profile (the trained table's average —
    the model's best generic-amp prior). ``profile_device_ids`` adds table rows
    by device_id; ``profile_pairs`` adds ``enroll/pairs/<name>`` embeddings.
    """
    from openamp.emulate.evaluate import load_model

    run_dir = Path(run_dir)
    model, ck = load_model(run_dir, "cpu")
    model = _require_wavenet(model)

    layers = [{"conv_w": _blob(l.conv.weight), "conv_b": _blob(l.conv.bias),
               "mixin_w": _blob(l.input_mixer.weight),
               **_cond_blobs(l),
               "x1_w": _blob(l.layer1x1.weight), "x1_b": _blob(l.layer1x1.bias)}
              for l in model.layers]

    profiles = [{"name": "Table mean",
                 "embedding": model.embedding.weight.mean(dim=0).tolist()}]
    id_to_idx = {int(k): int(v) for k, v in ck["id_to_idx"].items()}
    for did in (profile_device_ids or []):
        if int(did) not in id_to_idx:
            raise RuntimeError(f"device {did} is not in the run's trained table")
        row = model.embedding.weight[id_to_idx[int(did)]]
        profiles.append({"name": f"device {int(did)}", "embedding": row.tolist()})
    for name in (profile_pairs or []):
        blob = torch.load(run_dir / "enroll" / "pairs" / name / "enrolled_pair.pt",
                          map_location="cpu", weights_only=False)
        profiles.append({"name": str(blob.get("name", name)),
                         "embedding": blob["embedding"].reshape(-1).tolist()})

    is_table = isinstance(model, TableDeltaWaveNet)
    is_delta = isinstance(model, DeltaWaveNet)
    is_mlp = isinstance(model, MLPFiLMWaveNet)
    bundle = {
        "aaom_bundle": BUNDLE_VERSION,
        "run": {"name": ck.get("name", run_dir.name), "sha8": run_sha8(run_dir),
                "manifest_sha256": ck.get("manifest_sha256", ""),
                "epoch": int(ck.get("epoch", -1)),
                "val_esr": float(ck.get("val_esr", float("nan"))),
                "n_devices": len(ck["device_ids"])},
        "sample_rate": int(ck["sample_rate"]),
        "arch": {"type": ("tabledelta_wavenet_a2" if is_table else
                          "delta_wavenet_a2" if is_delta else
                          "mlpfilm_wavenet_a2" if is_mlp else "film_wavenet_a2"),
                 "channels": model.channels,
                 "embedding_dim": model.embedding_dim,
                 # NAM's own spelling, so the plugin can hand it straight to
                 # NeuralAmpModelerCore; absent in pre-knob bundles = LeakyReLU.
                 "activation": NAM_ACTIVATION_NAMES[model.activation],
                 # Per-arch generator descriptors only, so film_wavenet_a2 bundles
                 # stay byte-identical to every one already shipped.
                 **({"delta_parts": [n for n, keep
                                     in zip(DELTA_PART_NAMES, model.delta_parts)
                                     if keep]} if is_table else
                    {"delta_rank": model.delta_rank} if is_delta else
                    {"cond_hidden": model.cond_hidden,
                     "cond_activation": NAM_ACTIVATION_NAMES[model.cond_activation]}
                    if is_mlp else {}),
                 "kernel_sizes": list(model.kernel_sizes),
                 "dilations": list(model.dilations),
                 "head_kernel": model.head_kernel,
                 "head_scale": float(model.head_scale),
                 "receptive_field": model.receptive_field},
        "weights": {"rechannel_w": _blob(model.rechannel.weight),
                    "layers": layers,
                    "head_w": _blob(model.head_rechannel.weight),
                    "head_b": _blob(model.head_rechannel.bias)},
        "profiles": profiles,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle), encoding="utf-8")

    summary = {k: bundle[k] for k in ("aaom_bundle", "run", "sample_rate", "arch")}
    summary["profiles"] = [p["name"] for p in profiles]
    summary["bytes"] = out_path.stat().st_size
    return summary


# --- Profile -------------------------------------------------------------------
def _profile(emb: torch.Tensor, name: str, *, dim: int, run_sha: str) -> dict:
    """One profile in the pasteable ``aaom_profile`` schema."""
    return {"aaom_profile": PROFILE_VERSION, "name": str(name),
            "run": run_sha, "dim": int(dim),
            "embedding": emb.detach().reshape(-1).tolist()}


@torch.no_grad()
def export_profile(run_dir: Path, *, device_id: int | None = None,
                   mean: bool = False, pair_name: str | None = None,
                   name: str | None = None,
                   names: dict[int, str] | None = None) -> dict:
    """Build one pasteable profile dict from exactly one source.

    Sources (mutually exclusive): ``device_id`` — a trained table row;
    ``mean`` — the table average; ``pair_name`` — an
    ``enroll/pairs/<name>/enrolled_pair.pt`` embedding. ``name`` overrides the
    label outright; otherwise a ``device_id`` row is labelled from ``names``
    (``{device_id: display name}``), falling back to ``device <id>``.
    """
    from openamp.emulate.evaluate import load_model

    run_dir = Path(run_dir)
    if sum(x is not None and x is not False
           for x in (device_id, mean or None, pair_name)) != 1:
        raise RuntimeError("pass exactly one of device_id / mean / pair_name")

    model, ck = load_model(run_dir, "cpu")
    model = _require_wavenet(model)
    if pair_name is not None:
        blob = torch.load(run_dir / "enroll" / "pairs" / pair_name / "enrolled_pair.pt",
                          map_location="cpu", weights_only=False)
        emb, default = blob["embedding"].reshape(-1), str(blob.get("name", pair_name))
    elif mean:
        emb, default = model.embedding.weight.mean(dim=0), "Table mean"
    else:
        id_to_idx = {int(k): int(v) for k, v in ck["id_to_idx"].items()}
        if int(device_id) not in id_to_idx:
            raise RuntimeError(f"device {device_id} is not in the run's trained table")
        emb = model.embedding.weight[id_to_idx[int(device_id)]]
        default = (names or {}).get(int(device_id)) or f"device {int(device_id)}"
    return _profile(emb, name or default, dim=model.embedding_dim,
                    run_sha=run_sha8(run_dir))


@torch.no_grad()
def export_profiles(run_dir: Path, *, mean: bool = False, pairs: bool = False,
                    names: dict[int, str] | None = None) -> list[dict]:
    """Build a list of pasteable profiles — every trained device row by default.

    Rows come out in table-index order, labelled from ``names``
    (``{device_id: display name}``) and falling back to ``device <id>`` for any
    id not present. Opt-in extras: ``mean`` prepends the table-mean profile;
    ``pairs`` appends every ``enroll/pairs/<name>/enrolled_pair.pt`` embedding
    (name-sorted). Each entry has the same ``aaom_profile`` schema as
    :func:`export_profile`.
    """
    from openamp.emulate.evaluate import load_model

    run_dir = Path(run_dir)
    names = names or {}
    model, ck = load_model(run_dir, "cpu")
    model = _require_wavenet(model)
    dim, run_sha = model.embedding_dim, run_sha8(run_dir)

    id_to_idx = {int(k): int(v) for k, v in ck["id_to_idx"].items()}
    profiles = []
    if mean:
        profiles.append(_profile(model.embedding.weight.mean(dim=0), "Table mean",
                                 dim=dim, run_sha=run_sha))
    for did in sorted(id_to_idx, key=id_to_idx.get):        # table-index order
        profiles.append(_profile(model.embedding.weight[id_to_idx[did]],
                                 names.get(did) or f"device {did}",
                                 dim=dim, run_sha=run_sha))
    if pairs:
        for pair_dir in sorted((run_dir / "enroll" / "pairs").glob("*/enrolled_pair.pt")):
            blob = torch.load(pair_dir, map_location="cpu", weights_only=False)
            profiles.append(_profile(blob["embedding"],
                                     str(blob.get("name", pair_dir.parent.name)),
                                     dim=dim, run_sha=run_sha))
    return profiles
