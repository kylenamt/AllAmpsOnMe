"""Characterization harness: pins what the code does today, so the refactor
cannot quietly change it.

Every check compares against a JSON snapshot in ``tests/snapshots/``. Regenerate
with::

    OPENAMP_UPDATE_SNAPSHOTS=1 pytest tests/test_equivalence.py

and only when a behaviour change is intended — the snapshot diff is the review.

Everything runs on CPU, fp32, ``num_workers=0``, AMP off. That is what makes
*exact* equality the bar rather than a tolerance: no cuDNN autotuning picking a
different reduction order, no fp16 accumulation. A GPU run of the same code is
not bit-reproducible even against itself, so it is not snapshotted here.

What each snapshot pins, and why that one:

- ``datasets``  — hashes of the item tensors at fixed ``(seed, index)``. This is
  the RNG-order guard: any change to how a dataset draws a device, file or
  window shows up here, and nothing else would catch it.
- ``models`` / ``encoders`` — hashes of every parameter after seeded
  construction, plus the forward output. Pins init order (which consumes the
  global RNG) and the arithmetic separately.
- ``training``  — the per-epoch CSV, metrics.json, and hashes of the saved
  checkpoint weights for a whole tiny run of both trainers. The weight hashes
  are the precision-carrying assertion: they encode every optimizer step
  exactly, which is stronger than the 6-decimal CSV and cheaper than scraping
  per-step numbers out of stdout.

Time-varying fields (``elapsed_s``, ``train_hours``) are dropped before
comparison, and ``manifest_sha256`` is dropped because it hashes a parquet file
whose bytes are not guaranteed stable across writes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("soundfile")
pytest.importorskip("auraloss")

from openamp.core.config import Config, EmulateConfig, JointConfig  # noqa: E402

SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"
UPDATE = os.environ.get("OPENAMP_UPDATE_SNAPSHOTS") == "1"

SEED = 1234
CLIP = 4096          # >= auraloss's largest window (1200)
REF = 4096
SAMPLES = 24_000     # per fixture file
DEVICES = [0, 1, 2, 3]


# --- Snapshot plumbing ----------------------------------------------------------
def _h(t) -> str:
    """Content hash of a tensor: dtype, shape and exact bytes."""
    a = np.ascontiguousarray(t.detach().cpu().numpy())
    return hashlib.sha256(
        f"{a.dtype}|{a.shape}|".encode() + a.tobytes()).hexdigest()[:16]


def _params(module) -> dict:
    return {k: _h(v) for k, v in module.state_dict().items()}


def check(name: str, payload) -> None:
    """Compare ``payload`` against ``snapshots/<name>.json``, or write it."""
    path = SNAPSHOTS / f"{name}.json"
    text = json.dumps(payload, indent=2, sort_keys=True)
    # Compare the round-tripped payload, not the live one: json turns the integer
    # keys of id_to_idx into strings and tuples into lists, so a fresh payload
    # would never equal one read back from disk.
    payload = json.loads(text)
    if UPDATE or not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        if not UPDATE:
            pytest.skip(f"wrote a new snapshot: {path.name} — rerun to compare")
        return
    want = json.loads(path.read_text(encoding="utf-8"))
    if want != payload:
        diff = [k for k in set(want) | set(payload) if want.get(k) != payload.get(k)]
        pytest.fail(f"{name} changed in {len(diff)} entries: {sorted(diff)[:12]}\n"
                    f"If the change is intended, rerun with "
                    f"OPENAMP_UPDATE_SNAPSHOTS=1 and review the diff.")


def _config(tmp_path, **emulate) -> Config:
    """A Config built here rather than through load_config, so the repo's own
    configs/openamp.yaml cannot leak into a snapshot."""
    cfg = Config(data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                 seed=SEED)
    cfg.emulate = EmulateConfig(
        arch="film_wavenet", wn_channels=4, embedding_dim=8,
        clip_seconds=CLIP / cfg.sample_rate, batch_size=4, pairs_per_epoch=16,
        epochs=2, val_pairs=8, num_workers=0, amp=False, holdout_frac=0.25,
        log_every=10 ** 9, **emulate)
    return cfg


# --- 1. Datasets: the RNG-order guard -------------------------------------------
def test_dataset_items_are_unchanged(tmp_path, write_corpus):
    from openamp.data import (CorpusPaths, DeviceGridDataset, EmulationDataset,
                              JointDataset, WetDryDataset, build_device_index)

    cfg = _config(tmp_path)
    write_corpus(cfg, DEVICES, samples=SAMPLES, splits=("train", "val", "test"))
    paths = CorpusPaths.from_config(cfg)
    _, id_to_idx = build_device_index(paths)
    common = dict(receptive_field=64, id_to_idx=id_to_idx, clip_samples=CLIP)
    out: dict = {}

    def record(tag, ds, indices):
        for i in indices:
            item = ds[i]
            for key, v in sorted(item.items()):
                out[f"{tag}/{i}/{key}"] = _h(v)

    record("emulation", EmulationDataset(paths, "train", pairs_per_epoch=64,
                                         seed=SEED, **common), range(6))
    record("grid", DeviceGridDataset(paths, "test", n_windows=3, seed=SEED, **common),
           range(6))
    for mode in ("audio", "device"):
        record(f"joint_{mode}",
               JointDataset(paths, "train", ref_samples=REF, pairs_per_epoch=64,
                            seed=SEED, ref_mode=mode, **common), range(6))

    rng = np.random.default_rng(0)
    dry = (0.3 * rng.standard_normal(SAMPLES)).astype(np.float32)
    record("wetdry", WetDryDataset(dry, np.tanh(2 * dry).astype(np.float32),
                                   receptive_field=64, clip_samples=CLIP,
                                   pairs_per_epoch=64, seed=SEED), range(4))
    check("datasets", out)


# --- 2. Models and encoders: init order + forward arithmetic ---------------------
@torch.no_grad()
def test_model_init_and_forward_are_unchanged():
    from openamp.models import ARCHS, build_model

    x = torch.linspace(-0.8, 0.8, 512).reshape(1, 1, 512).repeat(2, 1, 1)
    idx = torch.tensor([0, 2])
    out: dict = {}
    for arch in ARCHS:
        for norm in (False, True):
            tag = f"{arch}{'-norm' if norm else ''}"
            torch.manual_seed(0)
            m = build_model(EmulateConfig(arch=arch, wn_channels=4, embedding_dim=8,
                                          cond_hidden=6, delta_rank=3,
                                          embed_normalize=norm), n_devices=4).eval()
            out[f"{tag}/receptive_field"] = m.receptive_field
            out[f"{tag}/embedding_dim"] = m.embedding_dim
            out[f"{tag}/forward"] = _h(m(x, idx))
            for k, v in _params(m).items():
                out[f"{tag}/param/{k}"] = v
    check("models", out)


@torch.no_grad()
def test_encoder_init_and_forward_are_unchanged(tmp_path):
    from openamp.encoders import (FingerprintEncoder, FrameAttentionEncoder,
                                  FrameStore, WaveNetEncoder)

    out: dict = {}

    def record(tag, enc, ref):
        out[f"{tag}/pooled_dim"] = enc.pooled_dim
        out[f"{tag}/receptive_field"] = enc.receptive_field
        out[f"{tag}/forward"] = _h(enc(ref))
        for k, v in _params(enc).items():
            out[f"{tag}/param/{k}"] = v

    ref_audio = torch.linspace(-0.7, 0.7, 2048).reshape(1, 1, 2048).repeat(2, 2, 1)
    for head in ("stats", "multitap"):
        torch.manual_seed(0)
        record(f"wavenet-{head}",
               WaveNetEncoder(embedding_dim=8, channels=4, attn_hidden=4,
                              proj_hidden=4, head=head, taps=(1, 5), expand_dim=8,
                              n_heads=2, tap_stride=4).eval(),
               ref_audio)

    # Fingerprint: a pooled.npz with the dry row extract.py writes.
    fp_dir = tmp_path / "fp"
    fp_dir.mkdir()
    rng = np.random.default_rng(0)
    ids = np.asarray([-1] + DEVICES, dtype=np.int64)
    np.savez(fp_dir / "pooled.npz", device_ids=ids,
             pooled=rng.standard_normal((len(ids), 32)).astype(np.float32))
    dev_ids, pooled, dry, _ = FingerprintEncoder.load_pooled(fp_dir)
    torch.manual_seed(0)
    record("fingerprint",
           FingerprintEncoder(embedding_dim=8, fingerprints=pooled,
                              device_ids=dev_ids, dry=dry, proj_hidden=8).eval(),
           torch.tensor(DEVICES, dtype=torch.long))

    # dac_frames: per-device [T, D] fp16 shards.
    fr_dir = tmp_path / "frames"
    (fr_dir / "frames").mkdir(parents=True)
    for d in DEVICES:
        np.save(fr_dir / "frames" / f"{d:04d}.npy",
                np.broadcast_to(np.arange(64, dtype=np.float16)[:, None],
                                (64, 16)).copy())
    (fr_dir / "meta.json").write_text(json.dumps(
        {"device_ids": DEVICES, "n_frames": 64, "latent_dim": 16}))
    torch.manual_seed(0)
    record("dac_frames",
           FrameAttentionEncoder(embedding_dim=8, frames=FrameStore(fr_dir),
                                 n_queries=2, d_model=8, proj_hidden=8).eval(),
           torch.tensor(DEVICES, dtype=torch.long))
    check("encoders", out)


# --- 3. Training: a whole tiny run of each trainer -------------------------------
_DROP_METRICS = ("train_hours", "manifest_sha256")
_DROP_CKPT = ("manifest_sha256", "fingerprint_sha256", "fingerprint_meta")


def _run_payload(run_dir: Path, weight_keys: tuple[str, ...]) -> dict:
    """CSV (minus elapsed_s), metrics.json, and the best checkpoint's contents."""
    rows = [r.split(",") for r in
            (run_dir / "train_log.csv").read_text().strip().splitlines()]
    drop = rows[0].index("elapsed_s")
    payload = {"log": [r[:drop] + r[drop + 1:] for r in rows]}

    metrics = json.loads((run_dir / "metrics.json").read_text())
    payload["metrics"] = {k: v for k, v in metrics.items() if k not in _DROP_METRICS}

    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    for key in weight_keys:
        for k, v in ck[key].items():
            payload[f"weights/{key}/{k}"] = _h(v)
    payload["checkpoint"] = {
        k: v for k, v in ck.items()
        if k not in weight_keys and k not in _DROP_CKPT
        and k not in ("optim", "scaler")}
    return payload


def test_emulate_training_is_unchanged(tmp_path, write_corpus):
    from openamp.emulate import train as emu_train

    cfg = _config(tmp_path)
    write_corpus(cfg, DEVICES, samples=SAMPLES)
    emu_train.train(cfg, name="eq", device="cpu")
    check("training_emulate", _run_payload(cfg.emulate_run_dir("eq"), ("model",)))


def test_joint_training_is_unchanged(tmp_path, write_corpus):
    from openamp.joint_model import train as joint_train

    cfg = _config(tmp_path)
    cfg.joint = JointConfig(ref_seconds=REF / cfg.sample_rate, enc_channels=4,
                            enc_attn_hidden=4, enc_proj_hidden=4)
    write_corpus(cfg, DEVICES, samples=SAMPLES)
    joint_train.train(cfg, name="eq", device="cpu")
    check("training_joint",
          _run_payload(cfg.joint_run_dir("eq"), ("encoder", "generator")))
