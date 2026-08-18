"""Fingerprint every amp with a **frozen** pretrained audio encoder.

The idea this serves: every device's render is the same dry capture signal, so any
difference between two devices' latents is caused by the amp and nothing else. No
content to leak, nothing to train — freeze somebody else's encoder, push each render
through it, pool the frames, and see whether the resulting vectors separate the amps.

    python scripts/encode_fingerprints.py --encoder dac     --limit-devices 8   # smoke
    python scripts/encode_fingerprints.py --encoder dac                         # full
    python scripts/encode_fingerprints.py --encoder encodec

Writes ``results/fingerprint/<encoder>/`` — one small pooled vector per device, a
combined ``.npz``, and ``meta.json``. Resumable: a device whose pooled file exists is
skipped, so an interrupted pass picks up where it stopped.

**Frames are pooled in RAM and never written.** The first version of this script cached
the per-frame latents so the pooling choice could be revisited without re-encoding, on
the reasoning that 13 GB was nothing against 7.6 TB free. That was the wrong number: the
volume is shared and this account's quota is 200 GB, already nearly full, so the cache
died at device 347 with ``OSError: Disk quota exceeded``. A device's frames are only
~60 MB in float32, which fits in RAM comfortably, so pooling on the fly costs nothing
and stores 8 KB per device instead of 30 MB. Changing the pooling now means re-encoding
(~16 min per encoder), which is the honest price and a cheap one.

Three decisions worth knowing, each argued at its site below: the corpus is
``data_sweep_a1`` (architecturally consistent), ``output_scale`` is undone before
encoding (level is amp information), and chunks carry real context that is then
trimmed (so the frames are the same ones a single whole-file pass would produce).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openamp.core import constants as C
from openamp.core import manifest as manifests
from openamp.core.config import load_config
from openamp.core.util import sha256_file
from openamp.dsp import audio as audio_io
from openamp.emulate.dataset import render_ok_devices

# Pre-quantization encoder trunks. Both are reconstruction codecs, which is the point:
# they are trained to preserve enough detail to rebuild the waveform, so they cannot
# have discarded timbre. `DacModel.encode()` would hand back the *quantized* codes, so
# both paths call `model.encoder(x)` directly instead.
ENCODERS = {
    "dac":     {"cls": "DacModel",     "id": "descript/dac_44khz",     "sr": 44_100},
    "encodec": {"cls": "EncodecModel", "id": "facebook/encodec_24khz", "sr": 24_000},
}

# The dry signal is identical for every device, so its own fingerprint is the natural
# null: a device whose vector sits on top of it means the encoder did not react to the
# amp at all. Stored under this id alongside the real devices.
DRY_ID = -1

# `sweep_val` is a byte-copy of the capture signal's lead-in: 7.75 s of audio followed
# by 1.25 s of exact digital zeros. Those zero frames are identical across devices, so
# pooling over them pulls every fingerprint toward a shared constant — precisely the
# differences we are trying to measure. Trim to the audible part.
VAL_AUDIBLE_SAMPLES = 372_000       # 7.75 s @ 48 kHz


def load_encoder(name: str, device):
    """Return ``(frozen_module, target_sample_rate, samples_per_frame)``."""
    import torch
    import transformers

    spec = ENCODERS[name]
    model = getattr(transformers, spec["cls"]).from_pretrained(spec["id"])
    model = model.eval().to(device)
    model.requires_grad_(False)

    # Derive the hop from the model rather than trusting a table -- but from a *pair* of
    # lengths, not one. A single probe measures (samples / frames), which carries the
    # encoder's edge offset: DAC turns 44100 samples into 86 frames, and 44100/86 = 513
    # while the true stride product is 512. Differencing two lengths cancels the offset,
    # and the chunk arithmetic below is only correct on the exact stride.
    def frames_for(n: int) -> int:
        with torch.no_grad():
            return _encode(model, torch.zeros(1, 1, n, device=device)).shape[-1]

    n1, delta = spec["sr"], 64 * 1024
    f1, f2 = frames_for(n1), frames_for(n1 + delta)
    hop = int(round(delta / (f2 - f1)))
    # Rounding is not blind: adding exactly 100 hops must add exactly 100 frames. That
    # holds only for the true stride, so a wrong guess fails here rather than silently
    # skewing every chunk boundary downstream.
    if frames_for(n1 + 100 * hop) - f1 != 100:
        raise SystemExit(f"{name}: could not determine the encoder hop (guessed {hop})")
    return model, spec["sr"], hop


def _encode(model, x):
    """``[1, 1, T] -> [C, frames]`` from the encoder trunk, before quantization."""
    z = model.encoder(x)
    if isinstance(z, (tuple, list)):
        z = z[0]
    return z[0]


def encode_signal(model, x: np.ndarray, device, *, hop: int,
                  chunk_frames: int, pad_frames: int) -> np.ndarray:
    """Encode a whole signal in chunks that carry real context; ``[frames, C]`` fp32.

    Each chunk is widened by ``pad_frames`` on both sides and the extra frames are
    dropped from the output, so a boundary sees the same left/right context it would
    have seen in a single full-length pass. Without this the seams pick up the
    encoder's zero-padding: measured on 20 s of noise against a whole-file pass, DAC
    goes from 32% relative error at ``pad_frames=0`` to 1.1e-4 at 16.

    Why chunk at all, when a whole 171 s file *does* fit: DAC peaks at 9.35 GiB of the
    card's 11.5 doing it, and the card is shared. Chunked runs in about a gigabyte.

    One honest caveat. DAC's encoder is pure convolution, so with enough context the
    chunked result is the whole-file result up to float noise. **EnCodec's encoder
    contains an LSTM**, whose context is unbounded, so no finite pad reproduces a
    whole-file pass exactly. Measured: at ``pad_frames=16`` individual frames differ by
    up to 2.5%, but the pooled mean+std vector — the only thing downstream uses — agrees
    with the whole-file pooling to cosine 0.999999, and to 1.000000 by pad 256.
    """
    import torch

    n = len(x)
    step, pad = chunk_frames * hop, pad_frames * hop
    out = []
    for start in range(0, n, step):
        stop = min(start + step, n)
        lo, hi = max(0, start - pad), min(n, stop + pad)
        seg = torch.from_numpy(np.ascontiguousarray(x[lo:hi])).to(device)
        with torch.no_grad():
            z = _encode(model, seg.view(1, 1, -1))
        drop_lo = (start - lo) // hop
        drop_hi = (hi - stop) // hop
        out.append(z[:, drop_lo:z.shape[-1] - drop_hi if drop_hi else None].T.float().cpu())
    return torch.cat(out, dim=0).numpy()


def output_scales(cfg) -> dict[int, float]:
    """``{device_id: output_scale}`` so the per-device peak normalization can be undone."""
    r = manifests.read_manifest(cfg.renders_manifest_path, manifests.RENDERS_COLUMNS)
    out: dict[int, float] = {}
    if r.empty:
        return out
    ok = r[r["status"] == C.RENDER_OK]
    for did, s in zip(ok["device_id"], ok["output_scale"]):
        try:
            s = float(s)
        except (TypeError, ValueError):
            continue
        if s > 0:
            out[int(did)] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data_sweep_a1"),
                    help="All 450 devices here are rendered from A1 captures. "
                         "data_sweep/ mixes 408 A2 with 42 A1, and that split would "
                         "show up in the plots as a cluster with no amp meaning.")
    ap.add_argument("--file", default="sweep_train",
                    choices=["sweep_train", "sweep_val"],
                    help="sweep_train (171 s) is the fingerprint; sweep_val (9 s, "
                         "different audio) is the stability check.")
    ap.add_argument("--out", type=Path, default=Path("results/fingerprint"))
    ap.add_argument("--limit-devices", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-frames", type=int, default=1024,
                    help="~12 s at DAC's 86 Hz; lower it if the card runs out.")
    ap.add_argument("--pad-frames", type=int, default=64,
                    help="Context carried into each chunk and trimmed from its output. "
                         "16 already suffices (see encode_signal); 64 is nearly free.")
    ap.add_argument("--no-undo-scale", action="store_true",
                    help="Keep the renders' peak normalization. Off by default: how "
                         "hard a device pushes IS gain information, so normalizing it "
                         "away deletes part of the fingerprint.")
    ap.add_argument("--force", action="store_true", help="re-encode devices already done")
    args = ap.parse_args()

    import torch
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = load_config(data_dir=args.data_dir)
    devices = render_ok_devices(cfg)
    if not devices:
        raise SystemExit(f"no render_ok devices under {args.data_dir} — wrong data dir?")
    if args.limit_devices:
        devices = devices[:args.limit_devices]

    scales = {} if args.no_undo_scale else output_scales(cfg)
    model, target_sr, hop = load_encoder(args.encoder, dev)

    run = args.out / args.encoder / args.file
    pooled_dir = run / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    trim = VAL_AUDIBLE_SAMPLES if args.file == "sweep_val" else None

    print(f"[fp] {args.encoder}  {ENCODERS[args.encoder]['id']}  "
          f"{target_sr} Hz  hop {hop} smp/frame ({target_sr / hop:.1f} Hz)  {dev}")
    print(f"[fp] {args.data_dir}/{args.file}  {len(devices)} devices  "
          f"undo_scale={not args.no_undo_scale}"
          + (f"  trim={trim} smp" if trim else ""))

    def fingerprint(x48: np.ndarray) -> tuple[np.ndarray, tuple]:
        """Audio -> the pooled vector. Frames live in RAM only, never on disk."""
        if trim is not None:
            x48 = x48[:trim]
        x = audio_io.resample(x48, cfg.sample_rate, target_sr)
        z = encode_signal(model, x, dev, hop=hop,
                          chunk_frames=args.chunk_frames, pad_frames=args.pad_frames)
        # The global pool: mean and std over every frame, concatenated. float64 for the
        # reduction because a 171 s file is ~15k frames and the std is the small term.
        z64 = z.astype(np.float64)
        return (np.concatenate([z64.mean(axis=0), z64.std(axis=0)]).astype(np.float32),
                z.shape)

    # The dry side first: it is the null every device is measured against.
    dry_path = pooled_dir / f"{DRY_ID}.npy"
    if args.force or not dry_path.is_file():
        clean = cfg.clean_split_dir(
            "train" if args.file == "sweep_train" else "val") / f"{args.file}.{cfg.output_format}"
        v, shape = fingerprint(audio_io.read_audio(clean)[0])
        np.save(dry_path, v)
        print(f"[fp] dry  {clean.name}  frames {shape} -> pooled {v.shape}")

    t0, done = time.time(), 0
    for i, did in enumerate(devices):
        dst = pooled_dir / f"{did:04d}.npy"
        if dst.is_file() and not args.force:
            continue
        wet = audio_io.read_audio(
            cfg.device_render_dir(did) / f"{args.file}.{cfg.output_format}")[0]
        wet = wet / np.float32(scales.get(int(did), 1.0))
        v, shape = fingerprint(wet)
        np.save(dst, v)
        done += 1
        if done == 1 or (i + 1) % 25 == 0 or i + 1 == len(devices):
            el = time.time() - t0
            print(f"[fp] {i + 1:4d}/{len(devices)}  device {did:4d}  frames {shape}  "
                  f"{el / max(done, 1):5.1f} s/device  elapsed {el / 60:5.1f} min")

    ids, pooled = [], []
    for did in [DRY_ID] + list(devices):
        p = pooled_dir / (f"{DRY_ID}.npy" if did == DRY_ID else f"{did:04d}.npy")
        if not p.is_file():
            continue
        ids.append(did)
        pooled.append(np.load(p))
    pooled = np.stack(pooled).astype(np.float32)
    ids = np.asarray(ids, dtype=np.int64)

    np.savez(run / "pooled.npz", device_ids=ids, pooled=pooled)
    (run / "meta.json").write_text(json.dumps({
        "encoder": args.encoder, "model_id": ENCODERS[args.encoder]["id"],
        "target_sr": target_sr, "hop": hop, "frame_rate_hz": target_sr / hop,
        "latent_dim": pooled.shape[1] // 2, "pooled_dim": pooled.shape[1],
        "data_dir": str(args.data_dir), "file": args.file,
        "undo_output_scale": not args.no_undo_scale, "trim_samples": trim,
        "chunk_frames": args.chunk_frames, "pad_frames": args.pad_frames,
        "n_devices": int((ids != DRY_ID).sum()), "dry_id": DRY_ID,
        # Provenance, the same way checkpoints carry it: which render set produced this.
        "renders_manifest_sha256": sha256_file(cfg.renders_manifest_path)
        if cfg.renders_manifest_path.is_file() else "",
    }, indent=2), encoding="utf-8")

    finite = np.isfinite(pooled).all(axis=1)
    print(f"\n[fp] wrote {run}/pooled.npz  {pooled.shape}  "
          f"({int((ids != DRY_ID).sum())} devices + dry)")
    if not finite.all():
        print(f"[fp] WARNING: {int((~finite).sum())} vectors contain non-finite values")


if __name__ == "__main__":
    main()
