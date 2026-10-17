"""Plot a probe_conformer run's training curves from its slurm logs.

    python scripts/plot_probe_conformer.py results/encoder/encoder_probe_conformer_c1 \\
        --resume-step 3420 --out docs/figures/probe_conformer_c1_training.png

Reads every ``slurm-*.log`` in the run directory, in job-id order, so a resumed run
plots as one curve (a later log wins on a repeated step). Plots train loss, val loss
and the train ESR split into real and made-up amps; ``--resume-step`` marks where a
resumed job took over.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"^\s+s(\d+)\s+train loss ([\d.]+)\s+esr real ([\d.]+) made-up ([\d.]+)"
                  r"\s+val loss ([\d.]+)")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def read_rows(run: Path) -> dict[int, tuple[float, ...]]:
    rows = {}
    for log in sorted(run.glob("slurm-*.log"), key=lambda p: int(p.stem.split("-")[1])):
        for line in log.read_text().splitlines():
            m = LINE.match(line)
            if m:
                rows[int(m[1])] = tuple(float(x) for x in m.groups()[1:])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/training_progress.png")
    ap.add_argument("--resume-step", type=int, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    rows = read_rows(args.run)
    steps = sorted(rows)
    train, real, made_up, val = zip(*(rows[s] for s in steps))
    best = min(range(len(steps)), key=lambda i: val[i])

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for label, y, color, ls in (("train loss", train, "#2a78d6", "-"),
                                ("val loss", val, "#eb6834", "-"),
                                ("train ESR, real amps", real, "#1baf7a", "--"),
                                ("train ESR, made-up amps", made_up, "#6250d6", "--")):
        ax.plot(steps, y, color=color, lw=2, ls=ls, label=label)
        ax.annotate(f"{y[-1]:.3f}", (steps[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.scatter([steps[best]], [val[best]], s=40, color="#eb6834", edgecolor=SURFACE, lw=2,
               zorder=5)
    ax.annotate(f"best val {val[best]:.4f} @ s{steps[best]}", (steps[best], val[best]),
                xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color=INK2)
    if args.resume_step is not None:
        ax.axvline(args.resume_step, color=INK2, lw=1, ls=":")
        ax.text(args.resume_step, ax.get_ylim()[1], "  resumed", va="top", fontsize=9, color=INK2)

    ax.set_xlabel("step", color=INK2)
    ax.set_ylabel("loss", color=INK2)
    ax.set_title(args.title or f"{args.run.name} — training progress", loc="left", color=INK,
                 fontsize=13)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.set_xlim(0, steps[-1] * 1.06)
    ax.legend(frameon=False, labelcolor=INK, loc="upper right")
    fig.tight_layout()

    out = args.out or args.run / "training_progress.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"{out}: {len(steps)} evals, s{steps[0]}..s{steps[-1]}")


if __name__ == "__main__":
    main()
