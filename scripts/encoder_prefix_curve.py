"""Does an encoder's *predicted* prefix already sound right?

    OPENAMP_DATA_DIR=data python scripts/encoder_prefix_curve.py \
        --probe results/encoder/<A>/probe_attn.pt --probe results/encoder/<B>/probe_attn.pt

For a generator trained with nested dropout (``embed_prefix_sizes``), cut each
encoder prediction to its first k dims (tail zeroed, no renormalize -- the
training-time cut, rendered through ``render_esr`` so it is not stretched back
to full length) and measure test ESR through the frozen generator. The amps are
the encoders' own held-out table amps (``enc-val``, rebuilt from the saved seed
and val_frac exactly as :mod:`openamp.probes.probe_esr` does): the generator
trained on them, so their table row is an exact oracle, and no encoder saw them.

The oracle -- the same amps' table rows, cut the same way -- is rendered once as
the reference curve. Every probe must share one split (same ids, seed,
val_frac), which is checked. Per row, besides ESR:

- ``cos_prefix`` -- cosine of the cut prediction to the cut oracle, ``[:k]`` vs ``[:k]``.
- ``rel_err``    -- ``|pred[:k] - oracle[:k]| / |oracle[:k]|``: the size of the
  error relative to what the prefix holds (length errors count, unlike cosine).
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
from openamp.encoders import FrameStore
from openamp.evaluation import resolve_device
from openamp.probes.probe_esr import load_encoder, predict_embeddings

from prefix_embedding_analysis import KS, cut, render_esr


def enc_val_ids(d0: dict, store: FrameStore) -> list[int]:
    """probe_attn's held-out table amps, rebuilt from the checkpoint's saved args."""
    saved = [int(d) for d in d0["device_ids"]]
    a = d0["args"]
    perm = np.random.default_rng(int(a["seed"])).permutation(len(saved))
    n_val = max(1, int(round(float(a["val_frac"]) * len(saved))))
    return sorted(saved[i] for i in perm[:n_val] if saved[i] in store)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=Path, action="append", required=True,
                    help="probe_attn.pt; repeat for several encoders")
    ap.add_argument("--run", type=Path,
                    default=Path("results/emulate/delta_a2_tanh_r32_512_v2_prefix"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/encoder/encoder_prefix_curve_v2prefix"))
    ap.add_argument("--test-pairs", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    dev = torch.device(resolve_device(args.device))
    args.out.mkdir(parents=True, exist_ok=True)

    tbl = torch.load(args.run / "embedding.pt", map_location="cpu", weights_only=False)
    T = tbl["embedding"].float().numpy().astype(np.float64)
    row = {int(d): i for i, d in enumerate(tbl["device_ids"])}
    scale = float(np.linalg.norm(T, axis=1).mean())

    # --- Predictions, one per encoder, on the shared enc-val amps -------------------
    ids, preds = None, {}
    for p in args.probe:
        d0 = torch.load(p, map_location="cpu", weights_only=False)
        store = FrameStore(Path(d0["args"]["frames"]))
        these = enc_val_ids(d0, store)
        if ids is not None and these != ids:
            raise SystemExit(f"{p}: different enc-val split from {args.probe[0]}")
        ids = these
        enc, _ = load_encoder(p, store, T.shape[1], dev)
        with torch.no_grad():
            e = predict_embeddings(enc, ids, scale, dev, batch=32)
        preds[p.parent.name] = np.stack([e[d] for d in ids]).astype(np.float64)
        del enc
        torch.cuda.empty_cache()
    oracle = T[[row[d] for d in ids]]
    print(f"[curve] run={args.run.name}  {len(ids)} enc-val amps x {args.test_pairs} "
          f"test pairs  |e|={scale:.4f}  sources: oracle, {', '.join(preds)}")

    csv_path = args.out / "report.csv"
    with csv_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(["source", "k", "cos_prefix", "rel_err", "esr_mean",
                                 "esr_median", "minutes"])
    rows = []
    for source, V in [("oracle", oracle)] + list(preds.items()):
        for k in KS:
            t0 = time.time()
            P = cut(V, k)
            a, b = P[:, :k], oracle[:, :k]
            cosp = float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1)
                                                     * np.linalg.norm(b, axis=1))))
            rel = float(np.mean(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)))
            esr = render_esr(cfg, args.run, {d: P[i].astype(np.float32)
                                             for i, d in enumerate(ids)},
                             device=args.device, test_pairs=args.test_pairs)
            mins = (time.time() - t0) / 60
            rows.append({"source": source, "k": k, "cos_prefix": cosp, "rel_err": rel,
                         "esr_mean": esr["mean"], "esr_median": esr["median"],
                         "minutes": mins, "per_device": esr["per_device"]})
            print(f"  {source[:48]:48} k={k:4d}  cos[:k] {cosp:.3f}  rel err {rel:.3f}  "
                  f"ESR mean {esr['mean']:.4f}  median {esr['median']:.4f}  "
                  f"({mins:.1f} min)", flush=True)
            with csv_path.open("a", newline="") as fh:
                csv.writer(fh).writerow([source, k, f"{cosp:.4f}", f"{rel:.4f}",
                                         f"{esr['mean']:.6f}", f"{esr['median']:.6f}",
                                         f"{mins:.2f}"])
            (args.out / "report.json").write_text(json.dumps(
                {"run": str(args.run), "probes": [str(p) for p in args.probe],
                 "ids": ids, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
