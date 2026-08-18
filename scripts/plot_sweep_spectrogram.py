"""High-resolution annotated spectrogram of the TONE3000 v3 capture signal.

Renders ``data/raw/nam_sweep/T3K-sweep-v3.wav`` with NAM's own region layout drawn
on top — the layout ``scripts/build_sweep_corpus.py`` slices the sweep corpus along
and ``openamp.emulate.enroll.nam_signal_regions`` mirrors:

    0.0-10.0 s   lead-in    discarded by NAM (7.75 s of guitar DI, then silence)
   10.0-181.0 s  train      2 blips, 3 chirps, 4 noise bursts, then guitar DI body
  181.0-190.0 s  val        NAM's validation block — and, in this corpus, ``test``
                            too (NAM defines no test split), and byte-identical to
                            the first 9 s of the lead-in

    python scripts/plot_sweep_spectrogram.py [--out docs/figures/sweep_spectrogram.png]

Two panels: the whole 190 s file, and a zoom on the 10-17 s calibration section,
which is the only genuinely *sweep-like* content in the file (~7 s of 190).
Frequency is log-scaled by max-pooling the linear STFT bins into log-spaced bands,
so the image stays an image (fast, compact) instead of a 36 M-quad pcolormesh.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import FancyArrowPatch, Rectangle

# --- Region layout (samples @ 48 kHz), from build_sweep_corpus.REGIONS ----------
LEAD_IN = (0.0, 10.0)
TRAIN = (10.0, 181.0)
VAL = (181.0, 190.0)

# Sub-structure inside the train region, measured at 50 ms resolution.
BLIPS = (10.5, 11.5)
CHIRPS = (12.1, 14.9)
NOISE = (15.0, 17.0)
DI_BODY = 17.25

# --- Palette (dataviz reference palette, light mode) ----------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
SERIES_1 = "#2a78d6"   # blue   — train
SERIES_2 = "#eb6834"   # orange — val = test
NEUTRAL = "#b8b6ae"    # gray   — lead-in (discarded)
# Annotation ink drawn *on* the spectrogram, which is dark: light-on-dark.
ON_DARK = "#f0efec"

FMIN, FMAX = 20.0, 22_000.0
DB_RANGE = 90.0        # dynamic range shown below the loudest bin


def log_spectrogram(x, sr, n_fft, hop, n_rows=1200):
    """``(dB image [n_rows, frames], log freq edges)`` — linear STFT, log-pooled.

    Each output row is the **max** over the linear bins inside its band, so the
    high end (where the log grid is coarser than the FFT grid) can't alias a
    narrow tone away; rows too narrow to contain a bin fall back to interpolation.
    """
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(x[idx] * win, axis=1).T                  # [bins, frames]
    mag = np.abs(spec).astype(np.float32) / (win.sum() / 2.0)   # -> amplitude, dBFS-ref 1.0
    db = 20.0 * np.log10(np.maximum(mag, 1e-9))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    edges = np.logspace(np.log10(FMIN), np.log10(FMAX), n_rows + 1)
    bounds = np.searchsorted(freqs, edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    out = np.empty((n_rows, db.shape[1]), dtype=np.float32)
    for r in range(n_rows):
        lo, hi = bounds[r], bounds[r + 1]
        if hi > lo:
            out[r] = db[lo:hi].max(axis=0)
        else:                                   # band narrower than one FFT bin
            j = min(max(lo, 1), len(freqs) - 1)
            w = (centers[r] - freqs[j - 1]) / (freqs[j] - freqs[j - 1])
            out[r] = (1.0 - w) * db[j - 1] + w * db[j]
    return out, edges


def draw_spectrogram(ax, img, edges, t0, t1, vmin, vmax):
    im = ax.imshow(img, origin="lower", aspect="auto", cmap="magma",
                   vmin=vmin, vmax=vmax, interpolation="antialiased",
                   extent=(t0, t1, 0.0, float(img.shape[0])))
    n_rows = img.shape[0]

    def row_of(f):                      # frequency -> fractional row index
        return n_rows * (np.log10(f) - np.log10(edges[0])) / \
            (np.log10(edges[-1]) - np.log10(edges[0]))

    ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]
    ax.set_yticks([row_of(f) for f in ticks])
    ax.set_yticklabels([f"{f // 1000}k" if f >= 1000 else str(f) for f in ticks])
    ax.set_ylabel("frequency (Hz)", color=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, length=3)
    return im, row_of


def region_ribbon(ax, t1, *, compact=False):
    """The train/val/lead-in bar. Identity is text-in-place, not color alone."""
    ax.set_xlim(0, t1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bands = [(LEAD_IN, NEUTRAL, "lead-in", "discarded"),
             (TRAIN, SERIES_1, "train", "10.0 - 181.0 s   (171 s)"),
             (VAL, SERIES_2, "val = test", "181.0 - 190.0 s   (9 s)")]
    for (a, b), color, label, sub in bands:
        if b <= 0 or a >= t1:
            continue
        a, b = max(a, 0.0), min(b, t1)
        ax.add_patch(Rectangle((a, 0.18), b - a, 0.64, facecolor=color,
                               edgecolor=SURFACE, linewidth=2.0, clip_on=False))
        wide = (b - a) / t1 > 0.15
        if wide and not compact:
            ax.text((a + b) / 2, 0.5, f"{label}      {sub}", ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
        else:                      # too narrow for an inside label — caption it above
            ax.text((a + b) / 2, 1.02, f"{label}\n{sub.split('  ')[0]}", ha="center",
                    va="bottom", color=color, fontsize=9, fontweight="bold",
                    linespacing=1.3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", type=Path, default=Path("data/raw/nam_sweep/T3K-sweep-v3.wav"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/sweep_spectrogram.png"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    x, sr = sf.read(args.wav, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    dur = len(x) / sr

    full, edges = log_spectrogram(x, sr, n_fft=4096, hop=512)
    z0, z1 = 9.5, 18.0                                  # the calibration section
    zoom, zedges = log_spectrogram(x[int(z0 * sr):int(z1 * sr)], sr, n_fft=2048, hop=96)

    vmax = float(np.ceil(np.percentile(full, 99.99) / 5) * 5)
    vmin = vmax - DB_RANGE

    plt.rcParams.update({"font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig = plt.figure(figsize=(22, 14.5))
    gs = fig.add_gridspec(7, 2, width_ratios=[1, 0.012],
                          height_ratios=[0.62, 0.42, 4.6, 1.15, 0.55, 0.30, 3.4],
                          hspace=0.16, wspace=0.012,
                          left=0.045, right=0.955, top=0.900, bottom=0.050)

    fig.text(0.045, 0.963, "The TONE3000 v3 capture signal, and how NAM splits it",
             fontsize=20, fontweight="bold", color=INK)
    fig.text(0.045, 0.938,
             f"{args.wav}  ·  {dur:.0f} s  ·  {sr // 1000} kHz  ·  "
             "STFT 4096 pt / 512 hop, log-pooled to 1200 bands  ·  magnitude in dBFS",
             fontsize=11, color=INK_2)
    fig.text(0.045, 0.920,
             "Only ~7 s of the 190 s file is sweep-like (blips, chirps, noise at 10-17 s); "
             "everything else is guitar DI. The val block is a byte-for-byte copy of the lead-in.",
             fontsize=11, color=INK_2)

    # --- Panel A: the whole file ------------------------------------------------
    ax_arc = fig.add_subplot(gs[0, 0])       # the "these two are the same audio" tie
    ax_arc.set_xlim(0, dur)
    ax_arc.set_ylim(0, 1)
    ax_arc.axis("off")
    # rad is a fraction of the (very wide) endpoint distance — keep it tiny.
    ax_arc.add_patch(FancyArrowPatch((4.5, 0.34), (185.5, 0.34),
                                     transform=ax_arc.transData,
                                     connectionstyle="arc3,rad=-0.022",
                                     arrowstyle="<|-|>", mutation_scale=11,
                                     linewidth=1.3, color=INK_MUTED, clip_on=False))
    ax_arc.text(95, 0.68, "identical audio  (np.array_equal is True over these 9 s)",
                ha="center", va="center", fontsize=9.5, color=INK_MUTED, style="italic",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=3.0))

    ax_rib = fig.add_subplot(gs[1, 0])
    region_ribbon(ax_rib, dur)

    ax = fig.add_subplot(gs[2, 0], sharex=ax_rib)
    im, row_of = draw_spectrogram(ax, full, edges, 0.0, dur, vmin, vmax)
    for t in (TRAIN[0], VAL[0]):
        ax.axvline(t, color=ON_DARK, linewidth=1.4, alpha=0.85)
    ax.annotate("calibration section — 2 blips, 3 chirps, 4 noise bursts (see below)",
                xy=(13.5, row_of(19_000)), xytext=(30, row_of(19_000)),
                color=ON_DARK, fontsize=10.5, va="center",
                arrowprops=dict(arrowstyle="-|>", color=ON_DARK, linewidth=1.2))
    ax.text(99, row_of(28), "guitar DI body — 17.25 to 181.0 s", ha="center", va="bottom",
            color=ON_DARK, fontsize=10.5, alpha=0.9)
    ax.set_xlim(0, dur)
    ax.set_xticks(np.arange(0, dur + 1, 10))
    ax.tick_params(labelbottom=False)

    cax = fig.add_subplot(gs[2, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("dBFS", color=INK_2)
    cb.outline.set_visible(False)
    cax.tick_params(colors=INK_2, length=3)

    # --- Level envelope ---------------------------------------------------------
    ax_env = fig.add_subplot(gs[3, 0], sharex=ax)
    hop = int(0.05 * sr)
    n = len(x) // hop
    rms = 20.0 * np.log10(np.sqrt(
        (x[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12) + 1e-12)
    t = np.arange(n) * 0.05
    ax_env.fill_between(t, -110, np.maximum(rms, -110), color=SERIES_1,
                        alpha=0.16, linewidth=0)
    ax_env.plot(t, np.maximum(rms, -110), color=SERIES_1, linewidth=0.7)
    ax_env.axvspan(*VAL, color=SERIES_2, alpha=0.13, linewidth=0)
    ax_env.axvspan(*LEAD_IN, color=NEUTRAL, alpha=0.25, linewidth=0)
    ax_env.set_ylim(-110, 0)
    ax_env.set_yticks([-100, -75, -50, -25, 0])
    ax_env.set_ylabel("level (dBFS,\n50 ms RMS)", color=INK_2)
    ax_env.set_xlabel("time (s)", color=INK_2)
    ax_env.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax_env.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax_env.spines[s].set_visible(False)
    ax_env.spines["bottom"].set_color("#d8d7d1")
    ax_env.tick_params(colors=INK_2, length=3)
    ax_env.text(7.6, -104, "digital silence", ha="right", va="bottom",
                fontsize=9, color=INK_MUTED)

    # --- Panel B: the calibration zoom -----------------------------------------
    axz = fig.add_subplot(gs[6, 0])
    top = axz.get_position().y1
    fig.text(0.045, top + 0.030, "Zoom: the calibration section, 9.5 - 18.0 s",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.045, top + 0.011,
             "STFT 2048 pt / 96 hop. This is the whole of the file's sweep content — "
             "and all of it sits in the train region, where no val metric can see it.",
             fontsize=10.5, color=INK_2)

    imz, zrow_of = draw_spectrogram(axz, zoom, zedges, z0, z1, vmin, vmax)
    axz.set_xlim(z0, z1)
    axz.set_xticks(np.arange(10, 18.1, 1.0))
    axz.set_xlabel("time (s)", color=INK_2)
    axz.axvline(TRAIN[0], color=ON_DARK, linewidth=1.4, alpha=0.85)
    axz.text(TRAIN[0] + 0.04, zrow_of(15_500), "train region starts  10.0 s",
             color=ON_DARK, fontsize=10, va="center")

    marks = [(BLIPS[0], BLIPS[1], "2 blips\n-33.9 dBFS pk 0.99"),
             (CHIRPS[0], CHIRPS[1], "3 log chirps  14 Hz -> 15.9 kHz\n-57 / -23 / -9 dBFS"),
             (NOISE[0], NOISE[1], "4 noise bursts\n-60 / -40 / -26 / -14 dBFS"),
             (DI_BODY, z1, "guitar DI body")]
    for a, b, label in marks:
        axz.annotate("", xy=(a, zrow_of(20_500)), xytext=(b, zrow_of(20_500)),
                     arrowprops=dict(arrowstyle="|-|,widthA=0.35,widthB=0.35",
                                     color=ON_DARK, linewidth=1.1))
        axz.text((a + b) / 2, zrow_of(13_000), label, ha="center", va="top",
                 color=ON_DARK, fontsize=9.5, linespacing=1.35)

    caxz = fig.add_subplot(gs[6, 1])
    cbz = fig.colorbar(imz, cax=caxz)
    cbz.set_label("dBFS", color=INK_2)
    cbz.outline.set_visible(False)
    caxz.tick_params(colors=INK_2, length=3)

    for cell in (gs[0, 1], gs[1, 1], gs[3, 1], gs[4, 0], gs[4, 1], gs[5, 0], gs[5, 1]):
        fig.add_subplot(cell).axis("off")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    w, h = fig.get_size_inches()
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, "
          f"{int(w * args.dpi)}x{int(h * args.dpi)} px)")


if __name__ == "__main__":
    main()
