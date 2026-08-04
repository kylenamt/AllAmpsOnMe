"""Morph-plugin exports: weight bundle + pasteable embedding profiles.

The real-time plugin (AllAmpsOnMePlugin) runs the trained FiLM-WaveNet by
*weight folding*: for any fixed embedding ``e``, the per-layer FiLM
``z = gamma * (conv(x) + mixin(clean)) + beta`` (gamma, beta = film(e)) is
absorbed into the layer's own weights::

    conv.weight  *= gamma[:, None, None]
    conv.bias     = gamma * conv.bias + beta
    mixin.weight *= gamma[:, None, None]

leaving a bit-for-bit standard NAM A2 WaveNet — the architecture
NeuralAmpModelerCore already runs in real time. The plugin therefore never
evaluates FiLM in its DSP loop; it re-folds (a few kFLOPs) whenever the
morph dot moves. :func:`folded_model` is the Python reference for that
operation and :func:`forward_with_embedding` the reference for the unfolded
path; ``tests/test_export.py`` pins the two equal to the trained model.

Two artifacts leave this module:

- **Bundle** (:func:`export_bundle`) — one JSON file with everything the
  plugin needs: base weights of every static part, per-layer FiLM matrices,
  the A2 schedule, and built-in profiles (always the table mean). Tensors are
  base64 float32 little-endian, C-order; structure stays human-readable.
  Intended to be embedded in the plugin binary via ``juce_add_binary_data``.
- **Profile** (:func:`export_profile`) — the small pasteable JSON users
  exchange: ``{"aaom_profile": 1, "name", "run", "dim", "embedding"}``.
  ``run`` is the first 8 hex chars of the checkpoint's sha256, so the plugin
  can reject profiles enrolled against a different network.

``delta_wavenet`` folds the same way, one step upstream: its generator emits the
layer's own weight residual, so folding is literally::

    conv.weight  += dW(e)
    conv.bias    += db(e)
    mixin.weight += dm(e)

Same guarantee, same stock-A2 result — and no per-channel structure to preserve,
because the residual is already in weight space.

All three A2 archs export: ``film_wavenet``, ``mlpfilm_wavenet``, ``delta_wavenet``.
Only *how* the conditioning is computed from ``e`` differs, and that is
discriminated by ``arch.type`` (``film_wavenet_a2`` / ``mlpfilm_wavenet_a2`` /
``delta_wavenet_a2``). ``aaom_bundle`` stays 1: the envelope is unchanged, and the
per-arch layer payload is what ``arch.type`` exists to select. An mlpfilm bundle
carries ``film_layers``, a delta bundle ``delta_coeff_w``/``delta_coeff_b``/
``delta_basis``, and both deliberately **omit** ``film_w``/``film_b``, so a reader
that predates the arch fails on a missing key rather than folding garbage.

WaveNet-only: folding into the TCN would work the same way, but the plugin ships
the A2 topology; exporting a ``film_tcn`` run is a hard error.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

from openamp.core.util import sha256_file
from openamp.emulate.wavenet import (NAM_ACTIVATION_NAMES, DeltaWaveNet, FiLMWaveNet,
                                     MLPFiLMWaveNet, init_delta_zero,
                                     init_film_identity)

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

    Then neutralize the generator, so the layer's forward ignores its embedding
    argument entirely — that is what makes the copy a plain A2 layer for *any*
    device_idx. Two shapes of conditioning, one contract:

    - FiLM (``film_wavenet`` / ``mlpfilm_wavenet``): per-channel affine on the
      conv output, absorbed by scaling the rows of conv/mixin and shifting the
      bias. Both generators fold identically; only ``film(emb)`` differs.
    - Delta (``delta_wavenet``): the generator already speaks in weight space, so
      the fold is the addition the layer would have done.
    """
    if hasattr(layer, "delta"):
        dw, db, dm = layer.delta(emb)                       # [1,C,C,K], [1,C], [1,C]
        layer.conv.weight.add_(dw[0])
        layer.conv.bias.add_(db[0])
        layer.input_mixer.weight.add_(dm[0].reshape(-1, 1, 1))
        init_delta_zero(layer.delta)                        # delta -> exact zero
        return
    gamma, beta = layer.film(emb).chunk(2, dim=-1)          # [C], [C]
    layer.conv.weight.mul_(gamma[:, None, None])
    layer.conv.bias.mul_(gamma).add_(beta)
    layer.input_mixer.weight.mul_(gamma[:, None, None])
    init_film_identity(layer.film, channels)                # FiLM -> exact identity


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
    for layer in m.layers:
        _fold_layer(layer, emb, m.channels)
    return m


@torch.no_grad()
def forward_with_embedding(model: FiLMWaveNet, x: torch.Tensor,
                           emb: torch.Tensor) -> torch.Tensor:
    """FiLM forward with an explicit embedding vector instead of a table row.

    Mirrors :meth:`FiLMWaveNet.forward` exactly (same op order); this is the
    unfolded reference the folded path is tested against, and the semantics of
    an arbitrary morph point ``emb = sum_i w_i * e_i``.
    """
    _require_wavenet(model)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    e = emb.detach().reshape(1, -1).to(x.dtype).expand(x.shape[0], -1)
    h = model.rechannel(x)
    head = torch.zeros_like(h)
    for layer in model.layers:
        h, head_term = layer(h, x, e)
        head = head + head_term
    y = model.head_rechannel(torch.nn.functional.pad(head, (model.head_pad, 0)))
    return model.head_scale * y


# --- Bundle --------------------------------------------------------------------
def _cond_blobs(layer) -> dict:
    """One layer's conditioning weights, in the shape the plugin's re-fold expects.

    film_wavenet's single Linear ships as ``film_w``/``film_b`` -- the keys that
    shipped from day one, byte-identical. mlpfilm_wavenet ships ``film_layers``
    (its Linears in input->output order; apply ``arch.cond_activation`` *between*
    them and not after the last). delta_wavenet ships its generator instead:
    ``delta = (coeff_w @ e + coeff_b) @ delta_basis``, split into the kernel residual
    (first ``C*C*K`` entries, C-order like ``conv_w``), the bias residual (next
    ``C``) and the mixin-gain residual (last ``C``). Each non-legacy arch
    deliberately **omits** ``film_w``/``film_b``:
    ``arch.type`` is the intended discriminator, but a reader that predates the arch
    and ignores it then dies on a missing key instead of silently folding garbage
    and playing the wrong amp at full volume.
    """
    if hasattr(layer, "delta"):
        # effective_basis() folds the training-time direction/magnitude split back
        # into one matrix, so the plugin's re-fold is a plain matmul and the bundle
        # format is unchanged by that reparameterization.
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
        "arch": {"type": ("delta_wavenet_a2" if is_delta else
                          "mlpfilm_wavenet_a2" if is_mlp else "film_wavenet_a2"),
                 "channels": model.channels,
                 "embedding_dim": model.embedding_dim,
                 # NAM's own spelling, so the plugin can hand it straight to
                 # NeuralAmpModelerCore; absent in pre-knob bundles = LeakyReLU.
                 "activation": NAM_ACTIVATION_NAMES[model.activation],
                 # Per-arch generator descriptors only, so film_wavenet_a2 bundles
                 # stay byte-identical to every one already shipped.
                 **({"delta_rank": model.delta_rank} if is_delta else
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
