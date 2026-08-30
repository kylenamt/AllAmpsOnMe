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

from openamp.spec import (DataSpec, EncoderSpec, JointSpec,  # noqa: E402
                          LossSpec, ModelSpec, RefSpec, RunSpec, TrainingSpec)

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


SAMPLE_RATE = 48_000


def _spec(tmp_path, **changes) -> RunSpec:
    """A RunSpec built here rather than through load_run_spec, so the repo's own
    configs cannot leak into a snapshot."""
    base = dict(
        name="eq", data_dir=tmp_path / "data", results_dir=tmp_path / "results",
        seed=SEED, sample_rate=SAMPLE_RATE,
        model=ModelSpec("film_wavenet", {"channels": 4, "embedding_dim": 8}),
        data=DataSpec(dataset="emulation", clip_seconds=CLIP / SAMPLE_RATE,
                      pairs_per_epoch=16, holdout_frac=0.25),
        loss=LossSpec(kwargs={"preemph_coeff": 0.85, "stft_weight": 1.0}),
        training=TrainingSpec(batch_size=4, epochs=2, val_pairs=8, num_workers=0,
                              amp=False, log_every=10 ** 9))
    return RunSpec(**{**base, **changes})


# --- 1. Datasets: the RNG-order guard -------------------------------------------
def test_dataset_items_are_unchanged(tmp_path, write_corpus):
    from openamp.data import (DeviceGridDataset, EmulationDataset, JointDataset,
                              WetDryDataset, build_device_index)

    spec = _spec(tmp_path)
    write_corpus(spec, DEVICES, samples=SAMPLES, splits=("train", "val", "test"))
    paths = spec.paths
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
    from openamp.spec import ModelSpec as MS

    x = torch.linspace(-0.8, 0.8, 512).reshape(1, 1, 512).repeat(2, 1, 1)
    idx = torch.tensor([0, 2])
    out: dict = {}
    for arch in ARCHS:
        for norm in (False, True):
            tag = f"{arch}{'-norm' if norm else ''}"
            kw = {"channels": 4, "embedding_dim": 8, "embed_normalize": norm}
            if arch == "mlpfilm_wavenet":
                kw["cond_hidden"] = 6
            if arch == "delta_wavenet":
                kw["delta_rank"] = 3
            torch.manual_seed(0)
            m = build_model(MS(arch, kw), n_devices=4).eval()
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
# The spec carries absolute tmp_path directories, which differ every run.
_DROP_SPEC = ("data_dir", "results_dir")


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
        and k not in ("optim", "scaler", "spec")}
    payload["spec"] = {k: v for k, v in ck["spec"].items() if k not in _DROP_SPEC}
    return payload


def test_emulate_training_is_unchanged(tmp_path, write_corpus):
    from openamp.training import emulate as emu_train

    spec = _spec(tmp_path)
    write_corpus(spec, DEVICES, samples=SAMPLES)
    emu_train.train(spec, device="cpu")
    check("training_emulate", _run_payload(spec.run_dir, ("model",)))


def test_joint_training_is_unchanged(tmp_path, write_corpus):
    from openamp.training import joint as joint_train

    spec = _spec(tmp_path, data=DataSpec(dataset="joint",
                                         clip_seconds=CLIP / SAMPLE_RATE,
                                         pairs_per_epoch=16, holdout_frac=0.25),
                 joint=JointSpec(
                     ref=RefSpec(seconds=REF / SAMPLE_RATE),
                     encoder=EncoderSpec("wavenet", {"channels": 4, "attn_hidden": 4,
                                                     "proj_hidden": 4,
                                                     # v1's JointConfig default;
                                                     # the constructor's is leakyrelu
                                                     "activation": "tanh"})))
    write_corpus(spec, DEVICES, samples=SAMPLES)
    joint_train.train(spec, device="cpu")
    check("training_joint", _run_payload(spec.run_dir, ("encoder", "generator")))
