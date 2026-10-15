"""What a given embedding cosine is actually worth, in ESR.

    OPENAMP_DATA_DIR=data python scripts/cos_esr_curve.py

:mod:`openamp.probes.probe_attn` optimizes cosine to a target embedding row and
renders ESR only once, at the very end, so nothing converts between the two: an
encoder sitting at cos 0.42 could be nearly right or nowhere near, and no
report currently says which. This measures the conversion directly -- take the
*enrolled* vectors (the ones that do reach the ceiling), corrupt them to a
known cosine, and render through the same frozen generator that probe_attn's
``[GATE]`` uses.

Two corruption shapes, because they are not interchangeable:

- ``iso`` -- the error is spread isotropically over all 512 directions: a
  random unit vector orthogonal to the target, mixed in to land on cosine *c*
  exactly. The neutral case, and the one a cosine number implicitly assumes.
- ``trunc-k`` -- the vector is projected onto the *table's* top-k principal
  components and the tail is dropped. This is the shape a pooler over a codec
  actually fails in (big directions right, tail missing), so its cosine is
  measured rather than chosen.

Both keep ``|e|`` at the enrolled radius, so only direction is under test --
v2 is an ``embed_normalize`` run, i.e. every row already shares one radius, and
letting the norm move would confound direction error with magnitude error.

The ``exact`` row is the control: it must reproduce the run's own enrollment
ceiling (mean 0.0231 / median 0.0089 for delta_a2_tanh_r32_512_v2). If it does
not, nothing else in the table is trustworthy.

Conditions run in order of what they settle first, and each is appended to
``report.csv`` as it finishes, so an interrupted run still leaves the rows it
did get.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from openamp.core.config import load_config
from openamp.probes.probe import holdout_esr

# (kind, value) -- ordered so the most decisive rows land first.
CONDITIONS: list[tuple[str, float | int | None]] = [
    ("exact", None),
    ("iso", 0.42),   # the cosine probe_attn actually reaches on held-out amps
    ("iso", 0.50),
    ("trunc", 200),
    ("trunc", 100),
    ("iso", 0.70),
    ("iso", 0.90),
    ("trunc", 30),
    ("iso", 0.95),
    ("iso", 0.99),
    ("trunc", 10),
]


def isotropic(V: np.ndarray, c: float, rng: np.random.Generator) -> np.ndarray:
    """Every row corrupted to cosine exactly ``c``, error spread over all directions."""
    out = np.empty_like(V)
    for i, v in enumerate(V):
        norm = float(np.linalg.norm(v))
        u = v / norm
        g = rng.standard_normal(len(v))
        g -= (g @ u) * u
        out[i] = norm * (c * u + np.sqrt(max(0.0, 1.0 - c * c)) * (g / np.linalg.norm(g)))
    return out


def truncate(V: np.ndarray, table: np.ndarray, k: int) -> np.ndarray:
    """Project onto the table's top-``k`` PCs, drop the tail, restore the radius."""
    mu = table.mean(0)
    _, _, Vt = np.linalg.svd(table - mu, full_matrices=False)
    basis = Vt[:k]
    rec = ((V - mu) @ basis.T) @ basis + mu
    return rec * (np.linalg.norm(V, axis=1, keepdims=True)
                  / np.linalg.norm(rec, axis=1, keepdims=True))


def mean_cosine(A: np.ndarray, B: np.ndarray) -> float:
    a = A / np.linalg.norm(A, axis=1, keepdims=True)
    b = B / np.linalg.norm(B, axis=1, keepdims=True)
    return float(np.sum(a * b, axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2"))
    ap.add_argument("--out", type=Path, default=Path("results/encoder/cos_esr_curve"))
    ap.add_argument("--test-pairs", type=int, default=150)
    ap.add_argument("--limit-devices", type=int, default=0,
                    help="0 = every enrolled amp; otherwise the first N (cheaper).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    enr = torch.load(args.run / "enroll" / "enrolled_embeddings.pt",
                     map_location="cpu", weights_only=False)
    tbl = torch.load(args.run / "embedding.pt", map_location="cpu", weights_only=False)
    table = tbl["embedding"].float().numpy()

    order = np.argsort([int(d) for d in enr["device_ids"]])
    ids = [int(enr["device_ids"][i]) for i in order]
    V = enr["embedding"].float().numpy()[order]
    if args.limit_devices:
        ids, V = ids[:args.limit_devices], V[:args.limit_devices]

    print(f"[curve] run={args.run.name}  {len(ids)} enrolled amps x {V.shape[1]}d  "
          f"test_pairs={args.test_pairs}  seed={args.seed}")
    print(f"[curve] enrolled radius mean {np.linalg.norm(V, axis=1).mean():.4f}  "
          f"table radius mean {np.linalg.norm(table, axis=1).mean():.4f}")

    csv_path = args.out / "report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["condition", "nominal", "cos_achieved",
                                 "esr_mean", "esr_median", "minutes"])

    rows = []
    for kind, val in CONDITIONS:
        t0 = time.time()
        if kind == "exact":
            P, label = V.copy(), "exact"
        elif kind == "iso":
            P, label = isotropic(V, float(val), rng), f"iso c={val:g}"
        else:
            P, label = truncate(V, table, int(val)), f"trunc k={val}"

        cos = mean_cosine(P, V)
        esr = holdout_esr(cfg, args.run, {d: P[i] for i, d in enumerate(ids)},
                          device=args.device, test_pairs=args.test_pairs)
        mins = (time.time() - t0) / 60.0
        row = {"condition": label, "nominal": val, "cos_achieved": cos,
               "esr_mean": esr["mean"], "esr_median": esr["median"],
               "minutes": mins, "per_device": esr["per_device"]}
        rows.append(row)
        print(f"  {label:14} cos {cos:+.3f}   ESR mean {esr['mean']:8.4f}  "
              f"median {esr['median']:8.4f}   ({mins:.1f} min)", flush=True)

        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([label, val, f"{cos:.4f}", f"{esr['mean']:.6f}",
                                     f"{esr['median']:.6f}", f"{mins:.2f}"])
        (args.out / "report.json").write_text(
            json.dumps({"run": str(args.run), "n_devices": len(ids),
                        "test_pairs": args.test_pairs, "seed": args.seed,
                        "rows": rows}, indent=2, default=float), encoding="utf-8")

    print(f"\n[curve] wrote {csv_path}")


if __name__ == "__main__":
    main()
