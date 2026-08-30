"""Fingerprint every amp with a frozen DAC, keeping the **frames** rather than pooling them.

    python -m openamp.encoder.extract --limit-devices 4     # smoke
    python -m openamp.encoder.extract                       # full, ~18 min, 12.6 GiB

Forked from ``scripts/encode_fingerprints.py``. The encoder plumbing is that
script's -- hop derivation, chunking with real context, resumability, provenance
-- and two things change:

**The frames are written, not reduced.** ``encode_signal`` already returns the
whole ``[T, 1024]`` array; ``concat(mean, std)`` was the only thing discarding
it, and it costs almost everything: the pooled cloud over 450 amps has a
participation ratio of 2.84 against the trained embedding table's 52.5. A
trainable pooler needs the frames back.

float16, per-device shards: 14,728 x 1024 is 28.77 MiB per amp, 12.64 GiB over
450. fp32 would be 25.28 GiB against 30 GB of system RAM, so it could not stay
page-resident and every training epoch would go back to NFS. Shards (rather than
one stacked array) keep resume-by-file-existence and avoid holding two copies
during a combine step. The original script's per-frame cache died of a 200 GB
quota; that quota is now 400 G with ~211 GiB free.

**The input is loudness-normalized to -16 LUFS**, where the original deliberately
*undid* the renders' peak normalization on the grounds that level is amp
information. -16 dB LUFS is what DAC itself was trained with (``VolumeNorm.db:
[const, -16]`` in descript-audio-codec's ``conf/base.yml`` and
``conf/final/44khz.yml``), so this puts our audio where the codec's latents are
best conditioned. Note it is LUFS -- ITU-R BS.1770-4, K-weighted and gated --
not dBFS; -24 is ``audiotools.AudioSignal.normalize()``'s library default, which
DAC overrides.

No dry row. The dry signal was the null a pooled fingerprint got subtracted
against; this design has no subtraction, and the pooler sees wet frames only.
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

# Pre-quantization encoder trunk. DAC is a reconstruction codec, which is the point:
# it is trained to preserve enough detail to rebuild the waveform, so it cannot have
# discarded timbre. `DacModel.encode()` would hand back the *quantized* codes, so this
# calls `model.encoder(x)` directly instead.
DAC = {"cls": "DacModel", "id": "descript/dac_44khz", "sr": 44_100}

# DAC's own training target. LUFS, not dBFS.
TARGET_LUFS = -16.0

# `sweep_val` is a byte-copy of the capture signal's lead-in: 7.75 s of audio followed
# by 1.25 s of exact digital zeros. Those zero frames are identical across devices, so
# pooling over them pulls every fingerprint toward a shared constant. Trim to the
# audible part.
VAL_AUDIBLE_SAMPLES = 372_000       # 7.75 s @ 48 kHz


def load_encoder(device):
    """Return ``(frozen_dac, target_sample_rate, samples_per_frame)``."""
    import torch
    import transformers
    model = getattr(transformers, DAC["cls"]).from_pretrained(DAC["id"])
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

    n1, delta = DAC["sr"], 64 * 1024
    f1, f2 = frames_for(n1), frames_for(n1 + delta)
    hop = int(round(delta / (f2 - f1)))
    # Rounding is not blind: adding exactly 100 hops must add exactly 100 frames. That
    # holds only for the true stride, so a wrong guess fails here rather than silently
    # skewing every chunk boundary downstream.
    if frames_for(n1 + 100 * hop) - f1 != 100:
        raise SystemExit(f"could not determine the encoder hop (guessed {hop})")
    return model, DAC["sr"], hop


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


def normalize_lufs(x: np.ndarray, sample_rate: int, target_lufs: float,
                   meter) -> tuple[np.ndarray, float, float]:
    """``(normalized, measured_lufs, applied_gain)`` -- a **pure scalar gain**.

    Deliberately no peak limiting afterwards. A limiter is a nonlinearity, and
    applying it only to the amps that happen to overshoot 1.0 would stamp
    amp-dependent distortion onto the very thing the fingerprint is measuring.
    DAC's encoder takes unbounded float, so an overshoot is harmless here; the
    caller records the post-normalization peak in meta.json so it stays visible
    rather than silently clipped.
    """
    measured = float(meter.integrated_loudness(x.astype(np.float64)))
    if not np.isfinite(measured):        # digital silence -> leave it alone
        return x, measured, 1.0
    gain = float(10.0 ** ((target_lufs - measured) / 20.0))
    return (x * np.float32(gain)).astype(np.float32, copy=False), measured, gain


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data_sweep_a1"),
                    help="All 450 devices here are rendered from A1 captures. "
                         "data_sweep/ mixes 408 A2 with 42 A1, and that split would "
                         "show up as a cluster with no amp meaning.")
    ap.add_argument("--file", default="sweep_train",
                    choices=["sweep_train", "sweep_val"],
                    help="sweep_train (171 s) is the fingerprint; sweep_val (9 s, "
                         "different audio) is the stability check.")
    ap.add_argument("--out", type=Path, default=Path("results/dac_frames"))
    ap.add_argument("--target-lufs", type=float, default=TARGET_LUFS,
                    help="DAC's own training target. LUFS, not dBFS.")
    ap.add_argument("--limit-devices", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-frames", type=int, default=1024,
                    help="~12 s at DAC's 86 Hz; lower it if the card runs out.")
    ap.add_argument("--pad-frames", type=int, default=64,
                    help="Context carried into each chunk and trimmed from its output. "
                         "16 already suffices (see encode_signal); 64 is nearly free.")
    ap.add_argument("--force", action="store_true", help="re-encode devices already done")
    args = ap.parse_args()

    import pyloudnorm
    import torch
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = load_config(data_dir=args.data_dir)
    devices = render_ok_devices(cfg)
    if not devices:
        raise SystemExit(f"no render_ok devices under {args.data_dir} — wrong data dir?")
    if args.limit_devices:
        devices = devices[:args.limit_devices]

    scales = output_scales(cfg)
    model, target_sr, hop = load_encoder(dev)
    meter = pyloudnorm.Meter(cfg.sample_rate)          # BS.1770-4, measured at 48 kHz

    run = args.out / args.file
    frames_dir = run / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    trim = VAL_AUDIBLE_SAMPLES if args.file == "sweep_val" else None

    print(f"[dac] {DAC['id']}  {target_sr} Hz  hop {hop} smp/frame "
          f"({target_sr / hop:.1f} Hz)  {dev}")
    print(f"[dac] {args.data_dir}/{args.file}  {len(devices)} devices  "
          f"target={args.target_lufs:g} LUFS  fp16 shards -> {frames_dir}"
          + (f"  trim={trim} smp" if trim else ""))

    def encode_device(x48: np.ndarray) -> tuple[np.ndarray, dict]:
        """Audio -> ``[T, 1024]`` fp16 frames, plus what we normalized by."""
        if trim is not None:
            x48 = x48[:trim]
        # Normalize at 48 kHz, before resampling: the loudness we are targeting is a
        # property of the signal, and resampling is gain-preserving to well below the
        # precision this matters at.
        x48, measured, gain = normalize_lufs(x48, cfg.sample_rate, args.target_lufs, meter)
        peak = float(np.abs(x48).max()) if x48.size else 0.0
        x = audio_io.resample(x48, cfg.sample_rate, target_sr)
        z = encode_signal(model, x, dev, hop=hop, chunk_frames=args.chunk_frames,
                          pad_frames=args.pad_frames)
        return z.astype(np.float16), {"lufs_in": measured, "gain": gain,
                                      "post_norm_peak": peak}

    t0, done, norm = time.time(), 0, {}
    n_frames = None
    for i, did in enumerate(devices):
        dst = frames_dir / f"{did:04d}.npy"
        if dst.is_file() and not args.force:
            continue
        wet = audio_io.read_audio(
            cfg.device_render_dir(did) / f"{args.file}.{cfg.output_format}")[0]
        wet = wet / np.float32(scales.get(int(did), 1.0))
        z, info = encode_device(wet)
        # Uniform shape is what lets the probe batch these at all; a short render would
        # otherwise surface as a confusing collate error thousands of frames later.
        if n_frames is None:
            n_frames = z.shape[0]
        elif z.shape[0] != n_frames:
            raise SystemExit(f"device {did} gave {z.shape[0]} frames, expected {n_frames} "
                             f"— renders are not the same length")
        tmp = dst.with_suffix(".tmp.npy")
        np.save(tmp, z)
        tmp.replace(dst)                    # atomic: an interrupt never leaves a partial
        norm[int(did)] = info
        done += 1
        if done == 1 or (i + 1) % 25 == 0 or i + 1 == len(devices):
            el = time.time() - t0
            print(f"[dac] {i + 1:4d}/{len(devices)}  device {did:4d}  frames {z.shape}  "
                  f"in {info['lufs_in']:6.1f} LUFS  peak {info['post_norm_peak']:5.2f}  "
                  f"{el / max(done, 1):5.1f} s/device  elapsed {el / 60:5.1f} min",
                  flush=True)

    # Only finished shards: an interrupted run leaves a `<id>.tmp.npy` behind, and
    # that name still matches a naive [0-9]*.npy glob.
    present = sorted(int(p.stem) for p in frames_dir.glob("[0-9]*.npy")
                     if p.stem.isdigit())
    shape = np.load(frames_dir / f"{present[0]:04d}.npy", mmap_mode="r").shape
    (run / "meta.json").write_text(json.dumps({
        "encoder": "dac", "model_id": DAC["id"], "target_sr": target_sr, "hop": hop,
        "frame_rate_hz": target_sr / hop, "latent_dim": int(shape[1]),
        "n_frames": int(shape[0]), "dtype": "float16",
        "data_dir": str(args.data_dir), "file": args.file,
        "target_lufs": args.target_lufs, "loudness_backend": "pyloudnorm BS.1770-4",
        "undo_output_scale": True, "peak_limited": False,
        "trim_samples": trim, "chunk_frames": args.chunk_frames,
        "pad_frames": args.pad_frames, "n_devices": len(present),
        "device_ids": present, "normalization": norm,
        # Provenance, the same way checkpoints carry it: which render set produced this.
        "renders_manifest_sha256": sha256_file(cfg.renders_manifest_path)
        if cfg.renders_manifest_path.is_file() else "",
    }, indent=2), encoding="utf-8")

    gib = len(present) * shape[0] * shape[1] * 2 / 2**30
    over = [d for d, v in norm.items() if v["post_norm_peak"] > 1.0]
    print(f"\n[dac] {len(present)} devices x {shape} fp16 = {gib:.2f} GiB -> {frames_dir}")
    if over:
        print(f"[dac] {len(over)} device(s) peak above 1.0 after normalization "
              f"(not limited, by design): {over[:10]}{' ...' if len(over) > 10 else ''}")


if __name__ == "__main__":
    main()
