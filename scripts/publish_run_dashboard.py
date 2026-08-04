"""Publish a static training dashboard readable from anywhere -- no SSH, no tunnel.

Reads every ``results/emulate/*/train_log.csv`` and renders one self-contained HTML
file (inline SVG curves, no external assets) into a web-served directory. On Aalto
that is ``~/public_html``, served at ``https://users.aalto.fi/~<user>/``.

Each card also carries the tail of that run's stdout (``<run>/train_stdout.log``, or
the older ``results/emulate/<name>.out``) in two panes -- epoch summaries only, or
every line including per-step progress. ``--log-lines 0`` leaves it out.

    conda run -n open-amp3000 python scripts/publish_run_dashboard.py \
        --out ~/public_html/<unguessable-token>

Add ``--watch 300`` to keep republishing every 5 minutes, and run it under nohup so
it survives logout:

    nohup python scripts/publish_run_dashboard.py --watch 300 \
        --out ~/public_html/<token> > /dev/null 2>&1 &

The web root is a *different* volume from the shared home and the server will not
follow a symlink back into ``/m/home`` (it answers 403), so the page is written as a
real file. Nothing here imports torch or touches a run directory for writing -- it is
read-only against the results tree and safe to run beside live training.

Note that anything placed under ``public_html`` is world-readable with no
authentication, which is why ``--out`` has no default: pick a path with a random
segment so the URL is unguessable.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Design tokens -------------------------------------------------------------
# Categorical slots 1 and 2 from the validated palette (all six checks pass in both
# modes, all-pairs: worst CVD dE 24.7 light / 26.8 dark). Train is slot 1, val is
# slot 2; the pairing is fixed so a run's colours never shift between cards.
SERIES = [
    ("train_esr", "Train ESR", "var(--series-1)"),
    ("val_esr", "Val ESR", "var(--series-2)"),
]

# Chart geometry, in viewBox units. The box includes the x-axis band so a card never
# grows a nested scrollbar.
CW, CH = 440, 236
ML, MR, MT, MB = 46, 56, 14, 30      # right margin holds the end labels


# A stdout line is either a per-step progress line (thousands of them, one per 50
# steps) or something worth reading. The split drives the log view's two panes.
STEP_LINE = re.compile(r"^\s*e\d+\s+step\s")


def classify_line(line: str) -> str:
    if STEP_LINE.match(line):
        return "step"
    s = line.lstrip()
    if s.startswith("[epoch"):
        return "epoch"
    if s.startswith(("[nan-guard]", "[stop]")):
        return "warn"
    if s.startswith("[dataset]"):
        return "notice"
    return "other"


def tail_lines(path: Path, n: int, max_bytes: int = 512_000) -> list[str]:
    """Last ``n`` lines of a text file, reading at most ``max_bytes`` from the end."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()               # drop the partial line the seek landed in
            data = fh.read()
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-n:] if n else []


# --- Reading the results tree --------------------------------------------------
def _finite(text: str) -> float | None:
    """CSV float, or None for the nan/inf the nan-guard writes on a dead epoch."""
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def find_stdout(run_dir: Path, results_dir: Path) -> Path | None:
    """The run's stdout capture: in-run log first, then the nohup file beside it.

    Newer runs write ``<run>/train_stdout.log``; the older ones were started with
    ``nohup ... > results/emulate/<name>.out``, so both layouts are checked.
    """
    for cand in (run_dir / "train_stdout.log", results_dir / f"{run_dir.name}.out"):
        if cand.is_file():
            return cand
    return None


def read_run(run_dir: Path, stale_after: float, results_dir: Path,
             log_lines: int) -> dict | None:
    """One run's curves, config and status. None if it has no readable log yet."""
    log = run_dir / "train_log.csv"
    try:
        with log.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, UnicodeDecodeError):
        return None
    if not rows:
        return None

    # A resumed run can rewrite epochs it already logged; keep the last of each so
    # the line never doubles back.
    by_epoch: dict[int, dict] = {}
    for r in rows:
        try:
            by_epoch[int(r["epoch"])] = r
        except (KeyError, TypeError, ValueError):
            continue
    points = []
    for epoch in sorted(by_epoch):
        r = by_epoch[epoch]
        points.append({
            "epoch": epoch,
            "train_esr": _finite(r.get("train_esr", "")),
            "val_esr": _finite(r.get("val_esr", "")),
            "train_loss": _finite(r.get("train_loss", "")),
            "lr": (r.get("lr") or "").strip(),
            "elapsed_s": _finite(r.get("elapsed_s", "")),
        })
    if not points:
        return None

    cfg = {}
    hist = run_dir / "config_history.jsonl"
    if hist.is_file():                              # last entry wins: a resume rewrites it
        try:
            for line in hist.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    cfg = json.loads(line)
        except (OSError, ValueError):
            cfg = {}

    metrics = {}
    mfile = run_dir / "metrics.json"
    if mfile.is_file():
        try:
            metrics = json.loads(mfile.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metrics = {}

    age = time.time() - log.stat().st_mtime
    # metrics.json is only written after the epoch loop exits, so it is the one
    # unambiguous "this run is over" signal -- epoch count alone misses early stop.
    status = "finished" if metrics else ("live" if age < stale_after else "stalled")
    vals = [p["val_esr"] for p in points if p["val_esr"] is not None]

    # Step lines outnumber everything else roughly 30:1, so a plain tail is all
    # step lines and the epoch summaries never appear. Keep two views: the raw
    # tail, and the same window with the step lines dropped.
    stdout_path = find_stdout(run_dir, results_dir) if log_lines else None
    window = tail_lines(stdout_path, log_lines * 30) if stdout_path else []
    tail = window[-log_lines:]
    key_tail = [ln for ln in window if classify_line(ln) != "step"][-log_lines:]

    return {
        "name": run_dir.name,
        "points": points,
        "cfg": cfg,
        "metrics": metrics,
        "status": status,
        "age_s": age,
        "stdout_path": stdout_path,
        "stdout_age_s": (time.time() - stdout_path.stat().st_mtime) if stdout_path else None,
        "tail": tail,
        "key_tail": key_tail,
        "best_val": min(vals) if vals else None,
        "best_epoch": min(((p["val_esr"], p["epoch"]) for p in points
                           if p["val_esr"] is not None), default=(None, None))[1],
        "max_epochs": cfg.get("max_epochs") or (cfg.get("emulate_cfg") or {}).get("epochs"),
        "arch": metrics.get("arch") or (cfg.get("emulate_cfg") or {}).get("arch"),
        "params": metrics.get("params"),
    }


def collect(results_dir: Path, stale_after: float, log_lines: int) -> list[dict]:
    """Every run with a log, live ones first, then stalled, then finished."""
    runs = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and (d / "train_log.csv").is_file():
            run = read_run(d, stale_after, results_dir, log_lines)
            if run:
                runs.append(run)
    rank = {"live": 0, "stalled": 1, "finished": 2}
    runs.sort(key=lambda r: (rank[r["status"]], r["age_s"]))
    return runs


# --- Formatting ----------------------------------------------------------------
def fmt_age(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds / 60)}m ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h ago"
    return f"{int(seconds / 86400)}d ago"


def fmt_dur(seconds: float) -> str:
    h, m = divmod(int(seconds / 60), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def fmt_params(n: int | None) -> str:
    if not n:
        return "--"
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """Round tick values spanning [lo, hi] -- 1/2/5 x a power of ten."""
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(count, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), 10 * mag)
    first = math.ceil(lo / step) * step
    ticks, t = [], first
    while t <= hi + step * 0.01:
        ticks.append(round(t, 10))
        t += step
    return ticks or [lo]


def tick_label(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:g}"


# --- The chart -----------------------------------------------------------------
def quantile(sorted_vals: list[float], p: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    i = p * (len(sorted_vals) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (i - lo)


def render_chart(run: dict, uid: int) -> str:
    """A small-multiple line chart: train + val ESR against epoch, on one axis.

    Rendered server-side and complete without JavaScript -- the hover layer added in
    the page script only surfaces values the table view already carries.
    """
    pts = run["points"]
    xs = [p["epoch"] for p in pts]
    ys = [p[k] for k, _, _ in SERIES for p in pts if p[k] is not None]
    if not ys:
        return '<p class="empty">No finite ESR logged yet.</p>'

    x0, x1 = min(xs), max(xs)
    # A single overflow epoch (some runs spike to 1e13 before the nan-guard catches
    # it) would flatten every real curve against the baseline, so the axis top is a
    # robust bound rather than the max. Anything above it is clipped by the plot
    # frame -- the line visibly leaves the top -- and the count is stated under the
    # chart with the true maximum, which the table view carries exactly.
    ordered = sorted(ys)
    lo_v, hi_v = ordered[0], ordered[-1]
    bound = max(quantile(ordered, 0.5) * 8, quantile(ordered, 0.90) * 1.2, lo_v * 1.05)
    top = min(hi_v, bound) if bound > lo_v else hi_v
    n_clipped = sum(1 for v in ys if v > top)

    y0, y1 = lo_v, top
    pad = (y1 - y0) * 0.08 or (abs(y1) * 0.1 or 0.01)
    y0, y1 = y0 - pad, y1 + pad
    span_x = (x1 - x0) or 1

    def px(e: float) -> float:
        return ML + (e - x0) / span_x * (CW - ML - MR)

    def py(v: float) -> float:
        return MT + (y1 - v) / (y1 - y0) * (CH - MT - MB)

    plot_l, plot_r, plot_t, plot_b = ML, CW - MR, MT, CH - MB
    out = [f'<svg class="chart" viewBox="0 0 {CW} {CH}" role="img" '
           f'aria-label="Train and validation ESR per epoch for {html.escape(run["name"])}">',
           f'<clipPath id="plot{uid}"><rect x="{plot_l}" y="{plot_t - 2}" '
           f'width="{plot_r - plot_l}" height="{plot_b - plot_t + 2}"/></clipPath>']

    # Gridlines: solid hairlines one step off the surface, never dashed.
    for t in nice_ticks(y0, y1):
        y = py(t)
        if not plot_t - 1 <= y <= plot_b + 1:
            continue
        out.append(f'<line class="grid" x1="{plot_l}" y1="{y:.1f}" x2="{plot_r}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick ty" x="{plot_l - 7}" y="{y + 3.2:.1f}">{tick_label(t)}</text>')
    out.append(f'<line class="axis" x1="{plot_l}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}"/>')

    for e in (x0, x1) if x1 - x0 < 3 else nice_ticks(x0, x1, 4):
        e = round(e)
        if x0 <= e <= x1:
            out.append(f'<text class="tick tx" x="{px(e):.1f}" y="{plot_b + 15}">{e}</text>')
    out.append(f'<text class="axis-name" x="{(plot_l + plot_r) / 2:.0f}" y="{CH - 2}">epoch</text>')

    ends = []
    for key, label, colour in SERIES:
        vals = [(p["epoch"], p[key]) for p in pts if p[key] is not None]
        if not vals:
            continue
        seq = [(px(e), py(v)) for e, v in vals]
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in seq)
        out.append(f'<path class="line" d="{d}" stroke="{colour}" clip-path="url(#plot{uid})"/>')
        ex, ey = seq[-1]
        last = vals[-1][1]
        if last <= y1:                              # a run ending on a spike has no
            # 2px surface ring keeps the end dot legible where the two lines cross.
            out.append(f'<circle class="dot" cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{colour}"/>')
            ends.append((key, ex, ey, last))        # in-frame endpoint to label

    # Direct-label the endpoints, but only where they separate -- nudging converged
    # labels apart detaches them from their lines. On collision the val number stays
    # (it is the one being watched) and train falls back to legend + tooltip + table.
    if len(ends) == 2 and abs(ends[0][2] - ends[1][2]) < 11:
        ends = [e for e in ends if e[0] == "val_esr"]
    for _, ex, ey, val in ends:
        out.append(f'<text class="end-label" x="{ex + 7:.1f}" y="{ey + 3.4:.1f}">{val:.4f}</text>')

    out.append(f'<g class="hover" data-l="{plot_l}" data-r="{plot_r}" '
               f'data-t="{plot_t}" data-b="{plot_b}">'
               f'<line class="crosshair" y1="{plot_t}" y2="{plot_b}"/></g>')
    out.append("</svg>")
    if n_clipped:
        out.append(f'<p class="clip-note">{n_clipped} point{"s" if n_clipped > 1 else ""} '
                   f'above the axis (max {hi_v:.3g}) &middot; exact values in the table view</p>')
    return "".join(out)


# --- The page ------------------------------------------------------------------
def render_log(run: dict, uid: int) -> str:
    """The run's stdout tail, as two panes: epoch summaries only, or every line.

    Both are rendered server-side, so the pane switch is a CSS class flip and the
    log stays readable with JavaScript off (the default pane is the useful one).
    """
    if not run.get("stdout_path"):
        return ""

    def pane(lines: list[str], cls: str) -> str:
        if not lines:
            return f'<pre class="log {cls}"><span class="ln other">(nothing yet)</span></pre>'
        body = "".join(f'<span class="ln {classify_line(ln)}">{html.escape(ln)}</span>\n'
                       for ln in lines)
        return f'<pre class="log {cls}">{body}</pre>'

    name = html.escape(run["stdout_path"].name)
    age = fmt_age(run["stdout_age_s"]) if run["stdout_age_s"] is not None else "unknown"
    return f"""<details class="log-view" data-uid="{uid}">
    <summary>Log &mdash; {name} <span class="log-age">written {age}</span></summary>
    <div class="log-tabs" role="group" aria-label="Log detail">
      <button type="button" data-pane="key" aria-pressed="true">Epochs</button>
      <button type="button" data-pane="all" aria-pressed="false">All lines</button>
    </div>
    {pane(run['key_tail'], 'pane-key')}
    {pane(run['tail'], 'pane-all')}
  </details>"""


def render_card(run: dict, uid: int) -> str:
    pts = run["points"]
    last = pts[-1]
    done, target = len(pts), run["max_epochs"]
    progress = f"{done}/{target}" if target else str(done)

    eta = ""
    if run["status"] == "live" and target and last["elapsed_s"] and done < target:
        per_epoch = last["elapsed_s"] / done
        eta = f"~{fmt_dur(per_epoch * (target - done))} left"

    label = {"live": "Live", "stalled": "Stalled", "finished": "Finished"}[run["status"]]
    icon = {"live": "&#9679;", "stalled": "&#9650;", "finished": "&#10003;"}[run["status"]]

    stats = [
        ("Epoch", progress),
        ("Best val ESR", f"{run['best_val']:.4f}" if run["best_val"] is not None else "--"),
        ("Latest val ESR", f"{last['val_esr']:.4f}" if last["val_esr"] is not None else "--"),
        ("LR", last["lr"] or "--"),
    ]
    tiles = "".join(
        f'<div class="stat"><span class="stat-label">{k}</span>'
        f'<span class="stat-value">{html.escape(v)}</span></div>' for k, v in stats)

    meta = " &middot; ".join(x for x in [
        html.escape(run["arch"] or "unknown arch"),
        fmt_params(run["params"]) + " params" if run["params"] else "",
        eta or (f"{fmt_dur(last['elapsed_s'])} elapsed" if last["elapsed_s"] else ""),
    ] if x)

    def cells(p: dict) -> str:
        esr = lambda v: "--" if v is None else f"{v:.5f}"
        return (f"<td>{p['epoch']}</td><td>{esr(p['train_esr'])}</td>"
                f"<td>{esr(p['val_esr'])}</td><td>{html.escape(p['lr'])}</td>"
                f"<td>{'--' if p['elapsed_s'] is None else fmt_dur(p['elapsed_s'])}</td>")

    rows = "".join(f"<tr>{cells(p)}</tr>" for p in reversed(pts))
    log = render_log(run, uid)

    return f"""<article class="card" data-status="{run['status']}" data-run="{html.escape(run['name'])}">
  <header class="card-head">
    <h2>{html.escape(run['name'])}</h2>
    <span class="pill pill-{run['status']}"><span aria-hidden="true">{icon}</span>{label}</span>
  </header>
  <p class="meta">{meta}</p>
  <p class="meta">Last epoch logged {fmt_age(run['age_s'])}</p>
  <div class="stats">{tiles}</div>
  {render_chart(run, uid)}
  <details class="table-view">
    <summary>Table view</summary>
    <div class="table-scroll"><table>
      <thead><tr><th>Epoch</th><th>Train ESR</th><th>Val ESR</th><th>LR</th><th>Elapsed</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
  </details>
  {log}
</article>"""


def render_page(runs: list[dict], stale_after: float, refresh: int) -> str:
    now = datetime.now(timezone.utc).astimezone()
    n_live = sum(r["status"] == "live" for r in runs)
    n_stalled = sum(r["status"] == "stalled" for r in runs)
    best = min((r["best_val"] for r in runs if r["best_val"] is not None), default=None)
    best_run = next((r["name"] for r in runs
                     if r["best_val"] is not None and r["best_val"] == best), "")

    summary = [("Runs tracked", str(len(runs))), ("Live", str(n_live))]
    if n_stalled:
        summary.append(("Stalled", str(n_stalled)))
    summary.append(("Best val ESR", f"{best:.4f}" if best is not None else "--"))
    summary_tiles = "".join(
        f'<div class="stat"><span class="stat-label">{k}</span>'
        f'<span class="stat-value">{html.escape(v)}</span></div>' for k, v in summary)

    payload = {r["name"]: {"epochs": [p["epoch"] for p in r["points"]],
                           "train_esr": [p["train_esr"] for p in r["points"]],
                           "val_esr": [p["val_esr"] for p in r["points"]]} for r in runs}
    cards = "\n".join(render_card(r, i) for i, r in enumerate(runs)) \
        or '<p class="empty">No runs found.</p>'
    legend = "".join(
        f'<span class="key"><svg width="16" height="8" aria-hidden="true">'
        f'<line x1="1" y1="4" x2="15" y2="4" stroke="{c}" stroke-width="2" '
        f'stroke-linecap="round"/></svg>{lab}</span>' for _, lab, c in SERIES)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<meta http-equiv="refresh" content="{refresh}">
<title>openamp training runs</title>
<style>
:root {{
  color-scheme: light;
  --plane: #f9f9f7; --surface: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,.10);
  --series-1: #2a78d6; --series-2: #eb6834;
  --good: #0ca30c; --warning: #fab219; --neutral: #898781;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --plane: #0d0d0d; --surface: #1a1a19;
    --text-primary: #fff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
    --series-1: #3987e5; --series-2: #d95926;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --plane: #0d0d0d; --surface: #1a1a19;
  --text-primary: #fff; --text-secondary: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
  --series-1: #3987e5; --series-2: #d95926;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 28px 24px 64px;
  background: var(--plane); color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}}
h1 {{ font-size: 21px; margin: 0 0 2px; font-weight: 600; }}
h2 {{ font-size: 15px; margin: 0; font-weight: 600; overflow-wrap: anywhere; }}
.wrap {{ max-width: 1360px; margin: 0 auto; }}
.sub {{ color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 20px 30px; }}
.stat {{ display: flex; flex-direction: column; gap: 1px; }}
.stat-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
.stat-value {{ font-size: 17px; font-weight: 600; }}
.summary {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 20px; margin-bottom: 18px;
}}
.controls {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  margin-bottom: 18px; font-size: 13px;
}}
.controls button {{
  font: inherit; color: var(--text-secondary); background: var(--surface);
  border: 1px solid var(--border); border-radius: 999px; padding: 5px 13px; cursor: pointer;
}}
.controls button[aria-pressed="true"] {{ color: var(--text-primary); border-color: var(--axis); font-weight: 600; }}
.controls .spacer {{ flex: 1; }}
.legend {{ display: flex; gap: 16px; color: var(--text-secondary); }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }}
.grid-cards {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(370px, 1fr)); }}
.card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 18px 12px; min-width: 0;
}}
.card-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }}
.meta {{ color: var(--text-secondary); font-size: 12.5px; margin: 4px 0 0; }}
.card .stats {{ margin: 14px 0 6px; gap: 14px 22px; }}
.card .stat-value {{ font-size: 15px; }}
.pill {{
  display: inline-flex; align-items: center; gap: 5px; white-space: nowrap;
  font-size: 11.5px; font-weight: 600; padding: 2px 9px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--text-secondary);
}}
.pill-live {{ color: var(--good); }}
.pill-stalled {{ color: var(--warning); }}
.chart {{ width: 100%; height: auto; display: block; margin-top: 4px; touch-action: pan-y; }}
.line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.dot {{ stroke: var(--surface); stroke-width: 2; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.axis {{ stroke: var(--axis); stroke-width: 1; }}
.tick {{ fill: var(--muted); font-size: 10px; font-variant-numeric: tabular-nums; }}
.ty {{ text-anchor: end; }}
.tx {{ text-anchor: middle; }}
.axis-name {{ fill: var(--muted); font-size: 10px; text-anchor: middle; }}
.end-label {{ fill: var(--text-secondary); font-size: 10.5px; font-variant-numeric: tabular-nums; }}
.crosshair {{ stroke: var(--axis); stroke-width: 1; visibility: hidden; }}
.clip-note {{ color: var(--muted); font-size: 11px; margin: 2px 0 0; }}
.table-view, .log-view {{ margin-top: 6px; border-top: 1px solid var(--border); padding-top: 8px; }}
.log-age {{ color: var(--muted); font-size: 11px; }}
.log-tabs {{ display: flex; gap: 6px; margin: 8px 0 6px; }}
.log-tabs button {{
  font: inherit; font-size: 11.5px; color: var(--text-secondary); background: var(--plane);
  border: 1px solid var(--border); border-radius: 999px; padding: 2px 10px; cursor: pointer;
}}
.log-tabs button[aria-pressed="true"] {{ color: var(--text-primary); border-color: var(--axis); font-weight: 600; }}
.log {{
  margin: 0; max-height: 260px; overflow: auto; background: var(--plane);
  border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px;
  font: 11.5px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre; tab-size: 2;
}}
.log .ln {{ display: block; color: var(--text-secondary); }}
.log .ln.epoch {{ color: var(--text-primary); font-weight: 600; }}
.log .ln.warn {{ color: var(--warning); }}
.log .ln.notice {{ color: var(--muted); }}
.log .ln.step {{ color: var(--muted); }}
.pane-all {{ display: none; }}                 /* Epochs is the default pane */
.log-view.show-all .pane-key {{ display: none; }}
.log-view.show-all .pane-all {{ display: block; }}
summary {{ cursor: pointer; font-size: 12.5px; color: var(--text-secondary); }}
.table-scroll {{ max-height: 260px; overflow: auto; margin-top: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 3px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
th {{ position: sticky; top: 0; background: var(--surface); color: var(--muted); font-weight: 500; }}
td:first-child, th:first-child {{ text-align: left; }}
.empty {{ color: var(--muted); font-size: 13px; }}
.tooltip {{
  position: fixed; z-index: 10; pointer-events: none; opacity: 0;
  transition: opacity .1s; background: var(--surface); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 7px 10px;
  font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.14); min-width: 128px;
}}
.tt-head {{ color: var(--muted); font-size: 11px; margin-bottom: 4px; }}
.tt-row {{ display: flex; align-items: center; gap: 7px; }}
.tt-value {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.tt-name {{ color: var(--text-secondary); }}
@media (max-width: 560px) {{
  body {{ padding: 20px 14px 48px; }}
  .grid-cards {{ grid-template-columns: 1fr; }}
  /* Wide log lines wrap rather than hide behind a sideways scroll on a phone;
     desktop keeps them column-aligned. */
  .log {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <h1>openamp training runs</h1>
  <p class="sub">Generated {now.strftime('%Y-%m-%d %H:%M %Z')} &middot;
     page refreshes every {refresh}s &middot; a run counts as stalled after
     {int(stale_after / 60)}m without a new epoch{f" &middot; best so far: {html.escape(best_run)}" if best_run else ""}</p>

  <section class="summary"><div class="stats">{summary_tiles}</div></section>

  <div class="controls">
    <button type="button" data-filter="all" aria-pressed="true">All</button>
    <button type="button" data-filter="live" aria-pressed="false">Live</button>
    <button type="button" data-filter="finished" aria-pressed="false">Finished</button>
    <span class="spacer"></span>
    <span class="legend">{legend}</span>
    <button type="button" id="theme" aria-pressed="false">Dark</button>
  </div>

  <div class="grid-cards">
{cards}
  </div>
</div>

<div class="tooltip" id="tt" role="status" aria-live="polite"></div>
<script id="run-data" type="application/json">{json.dumps(payload)}</script>
<script>
(function () {{
  "use strict";
  var DATA = JSON.parse(document.getElementById("run-data").textContent);
  var SERIES = [["train_esr", "Train ESR", "--series-1"], ["val_esr", "Val ESR", "--series-2"]];
  var tt = document.getElementById("tt");

  // Filter row scopes every card below it.
  document.querySelectorAll("[data-filter]").forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      var want = btn.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach(function (b) {{
        b.setAttribute("aria-pressed", String(b === btn));
      }});
      document.querySelectorAll(".card").forEach(function (card) {{
        card.hidden = want !== "all" && card.dataset.status !== want;
      }});
    }});
  }});

  // Log panes: switch which pre is shown, and keep the newest line in view.
  var toBottom = function (view) {{
    view.querySelectorAll(".log").forEach(function (p) {{ p.scrollTop = p.scrollHeight; }});
  }};
  document.querySelectorAll(".log-view").forEach(function (view) {{
    view.addEventListener("toggle", function () {{ if (view.open) toBottom(view); }});
    view.querySelectorAll(".log-tabs button").forEach(function (btn) {{
      btn.addEventListener("click", function () {{
        view.classList.toggle("show-all", btn.dataset.pane === "all");
        view.querySelectorAll(".log-tabs button").forEach(function (b) {{
          b.setAttribute("aria-pressed", String(b === btn));
        }});
        toBottom(view);
      }});
    }});
  }});

  var themeBtn = document.getElementById("theme");
  var dark = matchMedia("(prefers-color-scheme: dark)").matches;
  var sync = function () {{
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    themeBtn.textContent = dark ? "Light" : "Dark";
    themeBtn.setAttribute("aria-pressed", String(dark));
  }};
  sync();
  themeBtn.addEventListener("click", function () {{ dark = !dark; sync(); }});

  // Crosshair + one tooltip listing every series at that epoch. Values shown here
  // are all present in each card's table view, so hover only ever enhances.
  document.querySelectorAll(".card").forEach(function (card) {{
    var svg = card.querySelector(".chart");
    var g = svg && svg.querySelector(".hover");
    var run = DATA[card.dataset.run];
    if (!g || !run || !run.epochs.length) return;
    var line = g.querySelector(".crosshair");
    var L = +g.dataset.l, R = +g.dataset.r;
    var lo = run.epochs[0], hi = run.epochs[run.epochs.length - 1];

    var hide = function () {{ line.style.visibility = "hidden"; tt.style.opacity = 0; }};

    var show = function (ev) {{
      var box = svg.getBoundingClientRect();
      var vx = (ev.clientX - box.left) / box.width * {CW};
      var frac = (vx - L) / (R - L || 1);
      var target = lo + frac * ((hi - lo) || 1);
      var i = 0, bestD = Infinity;                 // snap to the nearest epoch
      run.epochs.forEach(function (e, k) {{
        var d = Math.abs(e - target);
        if (d < bestD) {{ bestD = d; i = k; }}
      }});
      var x = L + (run.epochs[i] - lo) / ((hi - lo) || 1) * (R - L);
      line.setAttribute("x1", x); line.setAttribute("x2", x);
      line.style.visibility = "visible";

      tt.textContent = "";
      var head = document.createElement("div");
      head.className = "tt-head";
      head.textContent = card.dataset.run + " \\u00b7 epoch " + run.epochs[i];
      tt.appendChild(head);                        // textContent: run names are data
      SERIES.forEach(function (s) {{
        var v = run[s[0]][i];
        var row = document.createElement("div");
        row.className = "tt-row";
        var key = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        key.setAttribute("width", "14"); key.setAttribute("height", "8");
        key.setAttribute("aria-hidden", "true");
        var seg = document.createElementNS("http://www.w3.org/2000/svg", "line");
        seg.setAttribute("x1", 1); seg.setAttribute("y1", 4);
        seg.setAttribute("x2", 13); seg.setAttribute("y2", 4);
        seg.setAttribute("stroke", "var(" + s[2] + ")");
        seg.setAttribute("stroke-width", "2"); seg.setAttribute("stroke-linecap", "round");
        key.appendChild(seg);
        var val = document.createElement("span");
        val.className = "tt-value";
        val.textContent = v == null ? "--" : v.toFixed(5);
        var name = document.createElement("span");
        name.className = "tt-name";
        name.textContent = s[1];
        row.append(key, val, name);
        tt.appendChild(row);
      }});

      var w = tt.offsetWidth, h = tt.offsetHeight;
      tt.style.left = Math.min(ev.clientX + 14, innerWidth - w - 8) + "px";
      tt.style.top = Math.max(8, ev.clientY - h - 12) + "px";
      tt.style.opacity = 1;
    }};

    svg.addEventListener("pointermove", show);
    svg.addEventListener("pointerdown", show);
    svg.addEventListener("pointerleave", hide);
    svg.addEventListener("blur", hide);
  }});
}})();
</script>
</body>
</html>"""


# --- Publishing ----------------------------------------------------------------
def publish(page: str, out_dir: Path) -> Path:
    """Write index.html into the web root, replacing it atomically.

    The rename means a reader never fetches a half-written page, and the 0o644 mode
    matters: the web server reads as `other`, and the default umask on a fresh temp
    file can leave it unreadable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o755)                        # the server traverses as `other`
    target = out_dir / "index.html"
    tmp = out_dir / ".index.html.tmp"
    tmp.write_text(page, encoding="utf-8")
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True,
                    help="web-served output directory, e.g. ~/public_html/<random-token>; "
                         "anything under public_html is world-readable, so use an "
                         "unguessable path")
    ap.add_argument("--results", type=Path, default=Path("results/emulate"),
                    help="directory of run directories (default: results/emulate)")
    ap.add_argument("--stale-min", type=float, default=20.0,
                    help="minutes without a new epoch before a run reads as stalled")
    ap.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                    help="republish every SECONDS instead of once and exiting")
    ap.add_argument("--log-lines", type=int, default=80, metavar="N",
                    help="lines of train_stdout.log (or <name>.out) to show per run; "
                         "each line costs page weight on every refresh, so keep it "
                         "modest; 0 disables the log view (default: 80)")
    ap.add_argument("--refresh", type=int, default=60,
                    help="seconds between browser auto-refreshes (default: 60)")
    args = ap.parse_args()

    out_dir = args.out.expanduser()
    results = args.results.expanduser()
    if not results.is_dir():
        raise SystemExit(f"no such results directory: {results}")
    stale_after = args.stale_min * 60

    while True:
        runs = collect(results, stale_after, max(args.log_lines, 0))
        target = publish(render_page(runs, stale_after, args.refresh), out_dir)
        live = sum(r["status"] == "live" for r in runs)
        print(f"{datetime.now().strftime('%H:%M:%S')}  wrote {target} "
              f"({len(runs)} runs, {live} live)", flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
