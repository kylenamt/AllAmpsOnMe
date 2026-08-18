"""Does the embedding mean anything? — the diagnostics ESR cannot give you.

Reconstruction ESR is not evidence that the joint model works. An encoder that
emits the same vector for every amp, paired with a generator that learned one
average amp, still posts a respectable ESR: the corpus mean is not a terrible
model of any single amp. Every check here exists to separate that outcome from a
real one, and they are the gate on the whole approach (guidelines §8).

Three questions, in the order they should be asked:

1. **Does the generator use ``e``?** Re-render with the embedding zeroed, and with
   the batch's embeddings permuted. Both must be clearly worse. If they are not,
   the conditioning pathway is dead (collapse mode A) and the ESR number is
   measuring an unconditional model.
2. **Does ``e`` leak content?** Re-render one window with a *different reference
   segment of the same amp*: the output should barely move, because the two
   references differ only in content. Then with a *different amp's* reference: it
   should move a lot. A same-amp swap that changes the output means the embedding
   is carrying audio, not tone (collapse mode B).
3. **Does it generalize to amps it never saw?** The point of the encoder is that
   an unseen amp costs one forward pass instead of an enrollment fit, so
   held-out-device ESR conditioned on a correct reference is the headline number.

Geometry (t-SNE, k-NN retrieval) comes last: it is the readable picture of what
the first three measured.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from openamp.core.config import Config
from openamp.emulate.dataset import build_device_index, device_names
from openamp.emulate.train import _make_loader
from openamp.joint_model.dataset import JointDataset
from openamp.joint_model.model import load_joint
from openamp.joint_model.train import EVAL_MODES, evaluate_esr

__all__ = ["diagnose", "conditioning_check", "leakage_check", "capture_embeddings"]

DEFAULT_PAIRS = 2000
DEFAULT_WINDOWS = 8          # reference segments per device for the geometry pass


def ref_mode_for_ck(ck: dict) -> str:
    """Reference kind a checkpoint's conditioning source needs."""
    return ("device" if ck.get("joint_cfg", {}).get("enc_kind", "wavenet") == "fingerprint"
            else "audio")


def _dataset(cfg: Config, ck: dict, device_ids, *, split: str, pairs: int, seed: int):
    """A :class:`JointDataset` over an explicit device set, sized like the run's."""
    jcfg_ref = int(ck["ref_samples"])
    return JointDataset(cfg, split, receptive_field=int(ck["receptive_field"]),
                        id_to_idx={int(d): i for i, d in enumerate(device_ids)},
                        ref_samples=jcfg_ref, clip_samples=int(ck["clip_samples"]),
                        ref_different_file=bool(ck["joint_cfg"]["ref_different_file"]),
                        ref_min_gap=int(round(ck["joint_cfg"]["ref_min_gap_seconds"]
                                              * ck["sample_rate"])),
                        pairs_per_epoch=pairs, seed=seed,
                        sources=tuple(ck["joint_cfg"]["sources"]),
                        # .get keeps checkpoints written before the fingerprint arm
                        # existed loading exactly as they did.
                        ref_mode=ref_mode_for_ck(ck))


# --- 1. Does the generator use e? ------------------------------------------------
def conditioning_check(model, loader, device, coeff: float, seed: int) -> dict:
    """ESR with the true, permuted and zeroed embedding.

    ``shuffled_ratio`` / ``zero_ratio`` are the numbers that matter: how many
    times worse a wrong or absent embedding makes the output. The table-conditioned
    baseline sits near x36 (0.052 -> 1.88), which is what "the model plainly uses
    its conditioning" looks like.
    """
    out = {m: evaluate_esr(model, loader, device, coeff, mode=m, seed=seed)
           for m in EVAL_MODES}
    true = out["true"]["esr"]
    return {
        "esr": true, "esr_shuffled": out["shuffled"]["esr"], "esr_zero": out["zero"]["esr"],
        "shuffled_ratio": out["shuffled"]["esr"] / true if true > 0 else float("nan"),
        "zero_ratio": out["zero"]["esr"] / true if true > 0 else float("nan"),
        "emb_spread": out["true"]["emb_spread"],
    }


# --- 2. Does e leak content? -----------------------------------------------------
def _draw_references(ds: JointDataset, device_ids, *, seed: int) -> torch.Tensor:
    """One reference per device, drawn independently of any window A.

    Under ``ref_mode="device"`` a "reference" is the device id itself, so every draw
    for a given amp is the same value by construction — which is what makes the
    same-amp arm of :func:`leakage_check` a no-op there rather than a measurement.
    """
    if ds.ref_mode == "device":
        return torch.tensor([int(d) for d in device_ids], dtype=torch.long)
    rng = np.random.default_rng(seed)
    refs = []
    for d in device_ids:
        f = ds.ref_files[int(rng.integers(0, len(ds.ref_files)))]
        c = int(rng.integers(0, f["n"] - ds.ref_samples + 1))
        refs.append(ds._read_reference(int(d), f, c))
    return torch.from_numpy(np.stack(refs, axis=0))


@torch.no_grad()
def leakage_check(model, ds: JointDataset, device, *, n_devices: int = 32,
                  seed: int = 0) -> dict:
    """Re-render one window under three different references.

    Holding window A fixed and changing only the reference isolates the
    embedding's contribution exactly. ``same_amp_delta`` should be far below
    ``cross_amp_delta``; if they are comparable, the reference's *content* is
    reaching the output.
    """
    ids = [int(d) for d in ds.device_ids[:n_devices]]
    if len(ids) < 2:
        return {"n_devices": len(ids), "note": "need >= 2 devices"}
    loader = _make_loader(ds, len(ids), 0, shuffle=False, seed=seed, drop_last=True)
    batch = next(iter(loader))
    inp = batch["input"][:len(ids)].to(device)
    R = model.receptive_field

    # Window A is fixed; only the reference changes between the three renders.
    e1 = model.embed(_draw_references(ds, ids, seed=seed + 1).to(device))
    e2 = model.embed(_draw_references(ds, ids, seed=seed + 2).to(device))
    e_other = e1.roll(1, dims=0)                       # each row now a *different* amp
    y1 = model.render(inp, e1)[..., R:]
    same = model.render(inp, e2)[..., R:]
    cross = model.render(inp, e_other)[..., R:]

    def rel(a, b):
        return float((torch.sum((a - b) ** 2) / (torch.sum(a ** 2) + 1e-12)).double())

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())

    same_d, cross_d = rel(y1, same), rel(y1, cross)
    if ds.ref_mode == "device":
        # One fixed vector per amp: a same-amp reference swap cannot change anything,
        # so report the identity rather than a measured ~0 that invites over-reading.
        # cross_amp_delta is still the real content of this check here.
        return {
            "n_devices": len(ids), "same_amp_delta": 0.0, "cross_amp_delta": cross_d,
            "cross_over_same": float("inf"),
            "same_amp_cos": 1.0, "cross_amp_cos": cos(e1, e_other),
            "note": "conditioning is one fixed vector per amp; the same-amp arm is a "
                    "no-op by construction, not a measurement",
        }
    return {
        "n_devices": len(ids),
        "same_amp_delta": same_d, "cross_amp_delta": cross_d,
        # > 1 means a different amp moves the output more than a different segment
        # of the same amp does, which is the whole point.
        "cross_over_same": cross_d / same_d if same_d > 0 else float("inf"),
        "same_amp_cos": cos(e1, e2), "cross_amp_cos": cos(e1, e_other),
    }


# --- 3 + geometry: per-capture embeddings ---------------------------------------
@torch.no_grad()
def capture_embeddings(model, ds: JointDataset, device, *, n_windows: int = DEFAULT_WINDOWS,
                       seed: int = 0):
    """``(per_segment, segment_device, per_capture, capture_device)`` embeddings.

    A capture vector is the mean over ``n_windows`` reference segments — the
    multi-reference averaging of guidelines §11, and what inference would do.
    """
    ids = [int(d) for d in ds.device_ids]
    seg, seg_dev = [], []
    for w in range(n_windows):
        refs = _draw_references(ds, ids, seed=seed + 1000 * w).to(device)
        seg.append(model.embed(refs).float().cpu().numpy())
        seg_dev.append(np.asarray(ids, dtype=np.int64))
    seg_arr = np.concatenate(seg, axis=0)
    cap = np.stack(seg, axis=0).mean(axis=0)           # [n_devices, De]
    return seg_arr, np.concatenate(seg_dev), cap, np.asarray(ids, dtype=np.int64)


# --- The gate -------------------------------------------------------------------
def diagnose(cfg: Config, run_dir: Path, *, device: str = "cuda",
             pairs: int = DEFAULT_PAIRS, n_windows: int = DEFAULT_WINDOWS,
             seed: int | None = None) -> dict:
    """Run every check against a finished run; writes ``diagnostics.json``."""
    from openamp.joint_model.geometry import _tsne, _write_tsne, knn_retrieval

    run_dir = Path(run_dir)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, ck = load_joint(run_dir, str(dev))
    seed = int(cfg.seed if seed is None else seed)
    coeff = float(ck["emulate_cfg"]["preemph"])
    batch = int(ck["emulate_cfg"]["batch_size"])
    seen = [int(d) for d in ck["device_ids"]]
    holdout = [int(d) for d in ck.get("holdout_ids", [])]
    print(f"[diag]  run={run_dir.name} seen={len(seen)} holdout={len(holdout)} "
          f"embed={model.embedding_dim}")

    out: dict = {"name": ck.get("name", run_dir.name), "n_seen": len(seen),
                 "n_holdout": len(holdout), "embedding_dim": model.embedding_dim,
                 "best_val_esr": float(ck.get("best_val_esr", float("nan")))}

    # 1. Conditioning, on the devices the run trained on.
    seen_ds = _dataset(cfg, ck, seen, split="val", pairs=pairs, seed=seed + 999)
    seen_loader = _make_loader(seen_ds, batch, 0, shuffle=False, seed=seed, drop_last=False)
    out["seen"] = conditioning_check(model, seen_loader, dev, coeff, seed)
    print(f"[diag]  seen    esr={out['seen']['esr']:.5f} "
          f"shuffled x{out['seen']['shuffled_ratio']:.1f} "
          f"zero x{out['seen']['zero_ratio']:.1f} spread={out['seen']['emb_spread']:.4f}")

    # 2. Leakage, same devices.
    out["leakage"] = leakage_check(model, seen_ds, dev, seed=seed)
    lk = out["leakage"]
    if "cross_over_same" in lk:
        print(f"[diag]  leak    same_amp={lk['same_amp_delta']:.4f} "
              f"cross_amp={lk['cross_amp_delta']:.4f} "
              f"(cross/same x{lk['cross_over_same']:.1f}) "
              f"cos same={lk['same_amp_cos']:.3f} cross={lk['cross_amp_cos']:.3f}")

    # 3. Zero-shot: devices the run never saw. This is the headline.
    if holdout:
        hold_ds = _dataset(cfg, ck, holdout, split="val", pairs=pairs, seed=seed + 555)
        hold_loader = _make_loader(hold_ds, batch, 0, shuffle=False, seed=seed,
                                   drop_last=False)
        out["holdout"] = conditioning_check(model, hold_loader, dev, coeff, seed)
        print(f"[diag]  holdout esr={out['holdout']['esr']:.5f} "
              f"shuffled x{out['holdout']['shuffled_ratio']:.1f} "
              f"zero x{out['holdout']['zero_ratio']:.1f}")

        # Geometry on the held-out captures, matching what encode-eval reports.
        seg, seg_dev, cap, cap_dev = capture_embeddings(model, hold_ds, dev,
                                                        n_windows=n_windows, seed=seed)
        if hold_ds.ref_mode == "device":
            # Every "segment" of a device is the identical vector, so query == gallery
            # and top-1 would be a trivially perfect 1.000. Reporting that would be
            # worse than reporting nothing.
            out["retrieval"] = {"note": "not applicable: conditioning is one fixed "
                                        "vector per device, so retrieval is trivial"}
            print("[diag]  knn     n/a (one fixed vector per device)")
        else:
            out["retrieval"] = knn_retrieval(seg, seg_dev, cap, cap_dev)
            print(f"[diag]  knn     top1={out['retrieval'].get('top1', float('nan')):.3f} "
                  f"top5={out['retrieval'].get('top5', float('nan')):.3f} "
                  f"chance={out['retrieval'].get('chance', float('nan')):.4f}")
        np.savez(run_dir / "capture_embeddings.npz", segment=seg, segment_device=seg_dev,
                 capture=cap, capture_device=cap_dev)
        coords = _tsne(cap, seed)
        if coords is not None:
            _write_tsne(run_dir, coords, cap_dev, device_names(cfg))

    verdict = _verdict(out)
    out["verdict"] = verdict
    print(f"[diag]  verdict: {verdict}")
    (run_dir / "diagnostics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _verdict(d: dict) -> str:
    """One line on whether the embedding is doing anything, in plain terms."""
    seen, lk = d.get("seen", {}), d.get("leakage", {})
    if not np.isfinite(seen.get("emb_spread", float("nan"))) or seen["emb_spread"] < 1e-3:
        return "COLLAPSED — the encoder emits a near-constant vector; ESR is unconditional"
    if seen.get("shuffled_ratio", 0) < 2.0:
        return ("WEAK — a wrong embedding barely hurts, so the generator is mostly "
                "ignoring the encoder")
    if lk.get("cross_over_same", 0) < 2.0:
        return ("LEAKING — a different segment of the same amp moves the output about as "
                "much as a different amp does; the embedding carries content")
    return "OK — the embedding steers the generator and survives a same-amp segment swap"
