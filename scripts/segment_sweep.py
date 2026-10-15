"""Break the train region of the T3K v3 capture signal into content segments.

    python scripts/segment_sweep.py [--out results/sweep_segments]

Two features per 10 ms frame: **level** (RMS, dBFS) and **spectral centroid**
(the "centre of mass" of the spectrum, in Hz -- high for bright audio, low for
dark). A content change is assumed to happen only at a *pause* or at an
*abrupt level change*, so:

1. **Cut** at every candidate: the centre of each pause (level below
   ``--pause-db`` dBFS for at least ``--min-pause`` ms), and each abrupt level
   step (the average level jumps by at least ``--step-db`` dB between the 250 ms
   before and after). The pause test is absolute on purpose: a relative one
   ("far below its neighbours") calls the -60 dBFS noise burst a pause, since
   it sits between a -9 dBFS chirp and a -40 dBFS burst.
2. **Merge** neighbouring pieces. First, any piece shorter than ``--min-seg``
   joins whichever neighbour it is more alike. Then the most-alike pair merges,
   again and again, until every neighbouring pair differs by at least
   ``--level-db`` dB in loud level (90th-percentile frame level) or
   ``--centroid-oct`` octaves in median centroid (over frames within 20 dB of
   the piece's loud level).

Pauses touching either end of the region are not cuts, and no level step is
cut inside or next to any pause. Times are reported both from the start of the
train region (= ``sweep_train.flac``) and from the start of the raw file.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

HOP = 480           # 10 ms at 48 kHz
N_FFT = 2048
FLOOR_DB = -120.0   # digital silence is clamped here
STEP_SPAN = 25      # frames either side for the step test (250 ms)
ACTIVE_DB = 20.0    # centroid frames: within this many dB of the piece's loud level


def frame_features(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """``(level dBFS, centroid Hz)``, one value per 10 ms frame."""
    n = len(x) // HOP
    level = 10.0 * np.log10((x[:n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-30)
    level = np.maximum(level, FLOOR_DB)
    pad = np.pad(x, (N_FFT // 2, N_FFT // 2))
    win = np.hanning(N_FFT)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    centroid = np.empty(n)
    for i in range(0, n, 2000):
        j = min(n, i + 2000)
        idx = (np.arange(i, j) * HOP + HOP // 2)[:, None] + np.arange(N_FFT)[None, :]
        mag = np.abs(np.fft.rfft(pad[idx] * win, axis=1))
        centroid[i:j] = (mag * freqs).sum(1) / (mag.sum(1) + 1e-20)
    return level, centroid


def find_cuts(level: np.ndarray, *, pause_db: float, min_pause: int,
              step_db: float) -> list[tuple[int, str]]:
    """Candidate cut frames, ``[(frame, "pause" | "step")]``, sorted."""
    n = len(level)
    quiet = level < pause_db
    edges = np.diff(np.r_[0, quiet.astype(int), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    pauses = [(a, b) for a, b in zip(starts, stops) if b - a >= min_pause]
    cuts = [((a + b) // 2, "pause") for a, b in pauses if a > 0 and b < n]

    # Energy average in dB, not the max: a max-filter step is a flat plateau
    # S frames wide that ends (rising) or starts (falling) at the jump, while the
    # energy average peaks on the jump itself in both directions.
    power = np.pad(10.0 ** (level / 10.0), STEP_SPAN, mode="edge")
    avg = 10.0 * np.log10(sliding_window_view(power, STEP_SPAN).mean(1))  # avg[k]: level[k-S:k]
    step = np.zeros(n)
    step[STEP_SPAN:n - STEP_SPAN] = (avg[2 * STEP_SPAN:n] - avg[STEP_SPAN:n - STEP_SPAN])
    mag = np.abs(step)
    in_pause = np.zeros(n, bool)
    for a, b in pauses:
        in_pause[max(0, a - STEP_SPAN):b + STEP_SPAN] = True
    for t in np.flatnonzero((mag >= step_db) & ~in_pause):
        if mag[t] == mag[max(0, t - STEP_SPAN):t + STEP_SPAN + 1].max():
            cuts.append((int(t), "step"))
    cuts.sort()
    # a flat-topped step can tie within its own span; keep the first of each run
    out = []
    for t, kind in cuts:
        if not out or t - out[-1][0] > STEP_SPAN:
            out.append((t, kind))
    return out


def piece_stats(level: np.ndarray, log2c: np.ndarray, a: int, b: int) -> tuple[float, float]:
    """``(loud level dB, median log2 centroid)`` of frames ``[a, b)``."""
    lv = level[a:b]
    loud = float(np.percentile(lv, 90))
    active = lv > loud - ACTIVE_DB
    c = float(np.median(log2c[a:b][active])) if loud > FLOOR_DB and active.any() else np.nan
    return loud, c


def merge(level, log2c, bounds, kinds, *, level_db: float, centroid_oct: float,
          min_seg: int):
    """Absorb pieces shorter than ``min_seg`` frames, then merge most-alike
    neighbours while any pair is within both thresholds."""
    bounds, kinds = list(bounds), list(kinds)
    stats = [piece_stats(level, log2c, bounds[i], bounds[i + 1])
             for i in range(len(bounds) - 1)]

    def distance(i):
        (l0, c0), (l1, c1) = stats[i], stats[i + 1]
        dc = 0.0 if np.isnan(c0) or np.isnan(c1) else abs(c1 - c0)
        return max(abs(l1 - l0) / level_db, dc / centroid_oct)

    def join(i):                                  # piece i absorbs piece i + 1
        del bounds[i + 1], kinds[i + 1]
        stats[i:i + 2] = [piece_stats(level, log2c, bounds[i], bounds[i + 1])]

    while len(stats) > 1:
        lengths = np.diff(bounds)
        k = int(np.argmin(lengths))
        if lengths[k] >= min_seg:
            break
        pairs = [i for i in (k - 1, k) if 0 <= i < len(stats) - 1]
        join(min(pairs, key=distance))

    while len(stats) > 1:
        d = [distance(i) for i in range(len(stats) - 1)]
        i = int(np.argmin(d))
        if d[i] >= 1.0:
            break
        join(i)
    return bounds, kinds, stats


def plot(out: Path, x, sr, level, centroid, bounds, kinds, offset_s, rows=6):
    t = np.arange(len(level)) * HOP / sr
    dur = len(level) * HOP / sr
    spec = np.abs(np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(np.pad(x, (N_FFT // 2, N_FFT)), N_FFT)
        [::HOP][:len(level)] * np.hanning(N_FFT), axis=1)).T
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    edges = np.logspace(np.log10(40), np.log10(20_000), 161)
    band = np.searchsorted(freqs, edges)
    img = np.stack([spec[max(lo, 1):max(hi, lo + 1)].max(0)
                    for lo, hi in zip(band[:-1], band[1:])])
    img = 20 * np.log10(img / (np.hanning(N_FFT).sum() / 2) + 1e-9)

    seg = dur / rows
    fig, axs = plt.subplots(2 * rows, 1, figsize=(26, 4.6 * rows),
                            gridspec_kw={"height_ratios": [2.0, 1.2] * rows})
    for r in range(rows):
        a0, a1 = r * seg, (r + 1) * seg
        m = (t >= a0) & (t < a1)
        ax, axl = axs[2 * r], axs[2 * r + 1]
        ax.imshow(img[:, m], origin="lower", aspect="auto", cmap="magma", vmin=-110,
                  vmax=-10, extent=(t[m][0], t[m][-1], 0, img.shape[0]))
        ax.set_yticks([np.searchsorted(edges, f) for f in (100, 1000, 10_000)],
                      ["100", "1k", "10k"])
        axl.plot(t[m], level[m], lw=0.5, color="C0")
        axl.set_ylim(-100, 0)
        axl.set_ylabel("level dBFS")
        ax2 = axl.twinx()
        ok = m & (level > -65)
        ax2.scatter(t[ok], centroid[ok], s=0.5, color="C1")
        ax2.set_yscale("log")
        ax2.set_ylim(100, 15_000)
        ax2.set_ylabel("centroid Hz", color="C1")
        for a in (ax, axl):
            a.set_xlim(a0, a1)
            a.set_xticks(np.arange(np.ceil(a0), a1, 1))
        ax.tick_params(labelbottom=False)
        axl.grid(alpha=0.3)
        for k, (b, kind) in enumerate(zip(bounds[:-1], kinds)):
            s = b * HOP / sr
            if a0 <= s < a1 and k > 0:
                color = "cyan" if kind == "pause" else "lime"
                ax.axvline(s, color=color, lw=1.6)
                axl.axvline(s, color=color, lw=1.2)
        for k in range(len(bounds) - 1):
            s, e = bounds[k] * HOP / sr, bounds[k + 1] * HOP / sr
            mid = (max(s, a0) + min(e, a1)) / 2
            if s < a1 and e > a0:
                ax.text(mid, img.shape[0] * 0.93, str(k), color="white", ha="center",
                        va="top", fontsize=11, fontweight="bold")
    axs[-1].set_xlabel(f"time from start of train region (s); add {offset_s:g} for the raw "
                       f"file   |   cyan = cut at a pause, lime = cut at a level step")
    fig.tight_layout()
    fig.savefig(out, dpi=60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", type=Path, default=Path("data/raw/nam_sweep/T3K-sweep-v3.wav"))
    ap.add_argument("--start", type=int, default=480_000, help="train region, first sample")
    ap.add_argument("--stop", type=int, default=8_688_000, help="train region, end sample")
    ap.add_argument("--out", type=Path, default=Path("results/sweep_segments"))
    ap.add_argument("--pause-db", type=float, default=-65.0)
    ap.add_argument("--min-pause", type=float, default=50.0, help="ms")
    ap.add_argument("--step-db", type=float, default=10.0)
    ap.add_argument("--min-seg", type=float, default=0.4, help="s")
    ap.add_argument("--level-db", type=float, default=8.0)
    ap.add_argument("--centroid-oct", type=float, default=0.6)
    args = ap.parse_args()

    raw, sr = sf.read(args.wav, dtype="float64")
    x = raw[args.start:args.stop]
    offset_s = args.start / sr
    level, centroid = frame_features(x, sr)
    log2c = np.log2(np.maximum(centroid, 1.0))

    cuts = find_cuts(level, pause_db=args.pause_db,
                     min_pause=int(round(args.min_pause / (1000 * HOP / sr))),
                     step_db=args.step_db)
    bounds = [0] + [c for c, _ in cuts] + [len(level)]
    kinds = ["start"] + [k for _, k in cuts]
    print(f"[cut]   {len(cuts)} candidates "
          f"({sum(k == 'pause' for _, k in cuts)} pauses, "
          f"{sum(k == 'step' for _, k in cuts)} level steps)")
    bounds, kinds, stats = merge(level, log2c, bounds, kinds, level_db=args.level_db,
                                 centroid_oct=args.centroid_oct,
                                 min_seg=int(round(args.min_seg * sr / HOP)))
    print(f"[merge] {len(stats)} segments")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (l, c) in enumerate(stats):
        a = min(bounds[k] * HOP, len(x))
        b = len(x) if k == len(stats) - 1 else bounds[k + 1] * HOP
        rows.append({
            "segment": k, "cut": kinds[k],
            "start_s": round(a / sr, 3), "end_s": round(b / sr, 3),
            "duration_s": round((b - a) / sr, 3),
            "file_start_s": round(a / sr + offset_s, 3),
            "file_end_s": round(b / sr + offset_s, 3),
            "start_sample": a, "end_sample": b,
            "level_p90_dbfs": round(l, 1),
            "centroid_hz": "" if np.isnan(c) else round(2 ** c),
        })
    with open(args.out / "segments.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['segment']:3d}  {r['start_s']:7.2f}-{r['end_s']:7.2f} s "
              f"({r['duration_s']:6.2f})  {r['cut']:5s}  level {r['level_p90_dbfs']:6.1f} dB  "
              f"centroid {r['centroid_hz']} Hz")
    plot(args.out / "segments.png", x, sr, level, centroid, bounds, kinds, offset_s)
    print(f"[done]  {args.out}/segments.csv, segments.png")


if __name__ == "__main__":
    main()
