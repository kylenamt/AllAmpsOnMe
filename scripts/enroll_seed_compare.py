"""Is an amp's enrolled embedding a point, or a set?

    python scripts/enroll_seed_compare.py --a <run_a> --b <run_b>

Enrollment fits one embedding row per held-out amp against a frozen generator.
Nothing guarantees that fit is unique: if the map ``e -> audio`` has flat or
degenerate directions, several quite different vectors reach the same ESR, and
then "the correct ``e``" is a region rather than a point. An encoder regressed
onto *one* of those solutions can never score better than the spread between
them, no matter how good its input is -- which would make
:mod:`openamp.probes.probe_attn`'s cosine the wrong yardstick rather than a
disappointing one.

This compares two enrollment runs of the **same amps** that differ only in
seed, and reports, per amp:

- the cosine between the two fitted vectors (how much the answer moved), and
- each fit's own held-out ESR (whether both are genuinely good answers).

Read it against two baselines this script also prints: the cosine between
*different* amps (the floor -- unrelated vectors), and what probe_attn achieved
(+0.423 on delta_a2_tanh_r32_512_v2). Two fits both at ESR ~0.023 but only
cos 0.6 apart would mean 0.42 is already near the metric's own ceiling.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

PROBE_ATTN_COS = 0.423  # probe_attn r0p75 on this run's held-out amps, for context


def load_side(run: Path) -> tuple[dict[int, np.ndarray], dict[int, float], int | None]:
    blob = torch.load(run / "enroll" / "enrolled_embeddings.pt",
                      map_location="cpu", weights_only=False)
    vecs = {int(d): blob["embedding"][i].float().numpy()
            for i, d in enumerate(blob["device_ids"])}
    esr: dict[int, float] = {}
    csv_path = run / "enroll" / "enrollment.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("test_esr") not in (None, ""):
                    esr[int(r["device_id"])] = float(r["test_esr"])
    return vecs, esr, blob.get("seed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2"))
    ap.add_argument("--b", type=Path, default=Path("results/emulate/v2__seed999"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/encoder/enroll_seed_compare.json"))
    args = ap.parse_args()

    A, esr_a, seed_a = load_side(args.a)
    B, esr_b, seed_b = load_side(args.b)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit("No device ids in common between the two enrollments.")
    print(f"[cmp] A {args.a.name} (seed {seed_a}, {len(A)} amps)")
    print(f"[cmp] B {args.b.name} (seed {seed_b}, {len(B)} amps)")
    print(f"[cmp] {len(shared)} amps in common\n")
    if seed_a is not None and seed_a == seed_b:
        print("[cmp] WARNING: both sides record the same seed — this is not a "
              "seed comparison.\n")

    unit = lambda v: v / np.linalg.norm(v)
    rows = []
    print(f"  {'dev':>5}  {'cos(A,B)':>9}  {'ESR A':>8}  {'ESR B':>8}")
    for d in shared:
        c = float(unit(A[d]) @ unit(B[d]))
        ra, rb = esr_a.get(d, float("nan")), esr_b.get(d, float("nan"))
        rows.append({"device_id": d, "cos": c, "esr_a": ra, "esr_b": rb})
        print(f"  {d:>5}  {c:>+9.3f}  {ra:>8.4f}  {rb:>8.4f}")

    cos = np.array([r["cos"] for r in rows])
    # The floor: how close are two *different* amps in side A, same subset.
    M = np.stack([unit(A[d]) for d in shared])
    off = (M @ M.T)[~np.eye(len(shared), dtype=bool)]
    ea = np.array([r["esr_a"] for r in rows], dtype=float)
    eb = np.array([r["esr_b"] for r in rows], dtype=float)

    print(f"\n[cmp] same amp, two seeds : cos mean {cos.mean():+.3f}  "
          f"median {np.median(cos):+.3f}  min {cos.min():+.3f}  max {cos.max():+.3f}")
    print(f"[cmp] different amps      : cos mean {off.mean():+.3f}  <- the floor")
    print(f"[cmp] probe_attn encoder   : cos {PROBE_ATTN_COS:+.3f}  <- what we are trying to beat")
    print(f"[cmp] ESR  A mean {np.nanmean(ea):.4f}  B mean {np.nanmean(eb):.4f}  "
          f"(both should sit near the enrollment ceiling)")

    verdict = ("target is a well-defined point — probe_attn's gap is a real "
               "encoder/input failure"
               if cos.mean() >= 0.95 else
               "enrollment itself is not converging — fix that before reading anything else"
               if np.nanmean(eb) > 0.10 else
               "the target is a SET, not a point — cosine to one row is the wrong "
               "objective; train through the frozen decoder instead")
    print(f"\n[cmp] {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"a": str(args.a), "b": str(args.b), "seed_a": seed_a, "seed_b": seed_b,
         "n_shared": len(shared), "within_amp_cos_mean": float(cos.mean()),
         "within_amp_cos_median": float(np.median(cos)),
         "between_amp_cos_mean": float(off.mean()),
         "probe_attn_cos": PROBE_ATTN_COS,
         "esr_a_mean": float(np.nanmean(ea)), "esr_b_mean": float(np.nanmean(eb)),
         "verdict": verdict, "per_device": rows}, indent=2), encoding="utf-8")
    print(f"[cmp] wrote {args.out}")


if __name__ == "__main__":
    main()
