"""Render, plot, and listen to ``wet - dry`` for one device's sweep render.

    python scripts/wet_minus_dry.py [--device 0000] [--data-dir data_sweep_a1]

Wet and dry are the same length by construction (the renderer pads the input
by the model's receptive field and trims the output to match), so the
subtraction is sample-for-sample with no alignment step of its own. This
script checks that assumption anyway -- cross-correlating a broadband chunk
(the loudest calibration chirp) at up to +-200 samples of lag -- and stops if
the render disagrees.

The difference is **not** "the distortion" in any clean sense: these are
nonlinear amp models, so wet is not dry-plus-an-independent-signal, and
`wet - dry` is only a linear-system idea applied to a nonlinear one. It is
still a fair question to look and listen: how much of wet's energy and shape
survives once the (comparatively quiet) input is subtracted back out.

Writes three WAVs (dry/wet/diff, each independently peak-normalized to
-1 dBFS so all three are audible -- their relative levels are NOT preserved,
see the printed dBFS stats for those) and five PNGs: a full-file three-row
overview, zooms on the calibration section / one loud-chirp cycle / a guitar
passage, and a spectrogram comparing wet against diff over that passage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import correlate

COLOR = {"dry": "#2a78d6", "wet": "#eb6834", "diff": "#7a5195"}
ALIGN_CHECK = (4.0, 4.95)   # s from the start of the file read -- the loud chirp
CAL_ZOOM = (0.0, 7.2)       # blips + 3 chirps + 4 noise bursts, if this is sweep_train
GUITAR_ZOOM = (90.0, 98.0)
CHIRP_ZOOM = (4.30, 4.34)
NFFT, HOP = 2048, 256


def check_alignment(dry: np.ndarray, wet: np.ndarray, sr: int, *, span, lag_max=200) -> int:
    a, b = int(span[0] * sr), int(span[1] * sr)
    xc = correlate(wet[a:b], dry[a:b], mode="full")
    lags = np.arange(-(b - a) + 1, b - a)
    center = len(xc) // 2
    window = slice(center - lag_max, center + lag_max + 1)
    return int(lags[window][np.argmax(xc[window])])


def dbfs(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-12))


def write_normalized(path: Path, x: np.ndarray, sr: int, target_dbfs: float = -1.0) -> float:
    peak = np.abs(x).max()
    gain = 10 ** (target_dbfs / 20.0) / (peak + 1e-12)
    sf.write(path, (x * gain).astype(np.float32), sr, subtype="PCM_16")
    return gain


def dbfs_spectrogram(x: np.ndarray) -> np.ndarray:
    """``[freq, frames]`` magnitude in dBFS -- 1.0 == a full-scale sine."""
    win = np.hanning(NFFT).astype(np.float64)
    n_frames = 1 + (len(x) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * win, axis=1)).T / (win.sum() / 2.0)
    return 20.0 * np.log10(np.maximum(mag, 1e-9))


def waveform_row(ax, t, x, t0, t1, color, label, ylim=(-1.05, 1.05)):
    m = (t >= t0) & (t < t1)
    ax.plot(t[m], x[m], lw=0.4, color=color)
    ax.set_xlim(t0, t1)
    ax.set_ylim(*ylim)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(alpha=0.25)


def plot_all(out: Path, dry, wet, diff, sr: int, title: str) -> None:
    t = np.arange(len(dry)) / sr
    rows = (("dry", dry), ("wet", wet), ("diff = wet - dry", diff))

    fig, axs = plt.subplots(3, 1, figsize=(24, 8), sharex=True)
    for ax, (name, x) in zip(axs, rows):
        ax.plot(t, x, lw=0.15, color=COLOR[name.split(" ")[0]])
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(name, fontsize=11)
        ax.grid(alpha=0.25)
    axs[-1].set_xlabel("time (s)")
    fig.suptitle(f"{title} -- full render", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=90)
    plt.close(fig)

    for name, span, dpi in (("zoom_calibration", CAL_ZOOM, 100),
                            ("zoom_chirp_40ms", CHIRP_ZOOM, 110),
                            ("zoom_guitar", GUITAR_ZOOM, 100)):
        fig, axs = plt.subplots(3, 1, figsize=(20, 9), sharex=True)
        for ax, (rname, x) in zip(axs, rows):
            waveform_row(ax, t, x, *span, COLOR[rname.split(" ")[0]], rname)
        axs[-1].set_xlabel("time (s)")
        fig.suptitle(f"Zoom: {span[0]:g}-{span[1]:g} s", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=dpi)
        plt.close(fig)

    a, b = int(GUITAR_ZOOM[0] * sr), int(GUITAR_ZOOM[1] * sr)
    S_wet, S_diff = dbfs_spectrogram(wet[a:b]), dbfs_spectrogram(diff[a:b])
    freqs = np.fft.rfftfreq(NFFT, 1.0 / sr)
    tt = np.linspace(*GUITAR_ZOOM, S_wet.shape[1])
    fig, axs = plt.subplots(2, 1, figsize=(20, 8), sharex=True, sharey=True)
    for ax, S, name in zip(axs, (S_wet, S_diff), ("wet", "diff = wet - dry")):
        im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-100, vmax=-10,
                       extent=(tt[0], tt[-1], freqs[0], freqs[-1]))
        ax.set_yscale("log")
        ax.set_ylim(40, 16_000)
        ax.set_ylabel(f"{name}\nHz")
    fig.colorbar(im, ax=axs, label="dBFS", fraction=0.02)
    axs[-1].set_xlabel("time (s)")
    fig.suptitle(f"Spectrogram: wet vs. diff, {GUITAR_ZOOM[0]:g}-{GUITAR_ZOOM[1]:g} s",
                fontsize=13)
    fig.savefig(out / "spectrograms.png", dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="0000", help="4-digit device id")
    ap.add_argument("--data-dir", type=Path, default=Path("data_sweep_a1"))
    ap.add_argument("--file", default="sweep_train", help="render filename stem")
    ap.add_argument("--out", type=Path, default=Path("results/wetdry_probe"))
    args = ap.parse_args()

    dry_path = args.data_dir / "clean" / "train" / f"{args.file}.flac"
    wet_path = args.data_dir / "renders" / args.device / f"{args.file}.flac"
    dry, sr = sf.read(dry_path, dtype="float64")
    wet, sr2 = sf.read(wet_path, dtype="float64")
    if sr != sr2:
        raise SystemExit(f"sample rate mismatch: dry={sr} wet={sr2}")
    n = min(len(dry), len(wet))
    if len(dry) != len(wet):
        print(f"[warn] length mismatch (dry={len(dry)}, wet={len(wet)}); "
              f"truncating both to {n}")
    dry, wet = dry[:n], wet[:n]

    lag = check_alignment(dry, wet, sr, span=ALIGN_CHECK)
    print(f"[align] best lag on the loud chirp: {lag} samples ({1000 * lag / sr:.3f} ms)")
    if abs(lag) > 4:
        raise SystemExit(f"[align] {lag} samples is too large to treat as aligned -- "
                         f"check the render, don't just subtract")

    diff = wet - dry
    for name, x in (("dry", dry), ("wet", wet), ("diff", diff)):
        print(f"[level] {name}: rms={dbfs(x):.1f} dBFS  "
              f"peak={20 * np.log10(np.abs(x).max() + 1e-12):.1f} dBFS")

    title = f"device {args.device}"
    manifest = args.data_dir / "manifests" / "manifest.parquet"
    if manifest.is_file():
        m = pd.read_parquet(manifest)
        row = m[m["device_id_str"] == args.device]
        if len(row):
            r = row.iloc[0]
            title = f"device {args.device} ({r['tone_title']}, {r['gain_bucket']}, {r['architecture']})"
    print(f"[info]  {title}")

    out = args.out / args.device
    out.mkdir(parents=True, exist_ok=True)
    for name, x in (("dry", dry), ("wet", wet), ("diff", diff)):
        gain = write_normalized(out / f"{name}.wav", x, sr)
        print(f"[write] {out / f'{name}.wav'}  (x{gain:.3f} to -1 dBFS peak; "
              f"the three are independently normalized, not level-matched)")
    plot_all(out, dry, wet, diff, sr, title)
    print(f"[write] {out}/overview.png, zoom_calibration.png, zoom_chirp_40ms.png, "
          f"zoom_guitar.png, spectrograms.png")


if __name__ == "__main__":
    main()
