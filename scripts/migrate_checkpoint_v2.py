#!/usr/bin/env python3
"""Convert a run directory's checkpoints from the v1 config payload to the v2 spec.

    python scripts/migrate_checkpoint_v2.py --dry-run results/emulate/*
    python scripts/migrate_checkpoint_v2.py results/emulate/nam_a2_256_mixed_dlw1

v1 checkpoints store ``emulate_cfg`` (and ``joint_cfg``) -- flat dicts of the
retired config dataclasses. v2 stores one ``spec``, the same information
regrouped, plus the corpus and run name the old payload never recorded. Nothing
reads the old keys any more, so a run directory must be converted once before
``emulate-compare``, ``emulate-export-bundle``, ``emulate-enroll`` or
``joint-eval`` will touch it again.

**Weights are never touched.** The only change is the config payload: the old
keys are replaced by ``spec``. The original file is kept as ``<name>.pt.v1``
unless ``--no-backup``, so this is reversible by moving that back.

``data_dir`` cannot be recovered from a v1 checkpoint -- it lived in
OPENAMP_DATA_DIR at the time -- so pass ``--data-dir`` if the run used anything
other than ``./data``. It only affects what the spec records; it does not move
any file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# v1 field -> v2 home, mirroring scripts/migrate_config_v2.py's tables.
MODEL = {"arch": "arch", "wn_channels": "channels", "wn_activation": "activation",
         "cond_hidden": "cond_hidden", "cond_activation": "cond_activation",
         "delta_rank": "delta_rank", "embedding_dim": "embedding_dim",
         "embed_normalize": "embed_normalize"}
DATA = {"clip_seconds": "clip_seconds", "pairs_per_epoch": "pairs_per_epoch",
        "holdout_frac": "holdout_frac"}
LOSS = {"stft_weight": "stft_weight", "preemph": "preemph_coeff"}
TRAINING = {"lr": "lr", "weight_decay": "weight_decay",
            "plateau_patience": "plateau_patience", "plateau_factor": "plateau_factor",
            "early_stop_patience": "early_stop_patience", "epochs": "epochs",
            "batch_size": "batch_size", "num_workers": "num_workers", "amp": "amp",
            "amp_dtype": "amp_dtype", "val_pairs": "val_pairs",
            "log_every": "log_every", "embed_noise_frac": "embed_noise_frac"}
ENCODER = {"enc_kind": "kind", "fp_path": "path", "fp_preprocess": "preprocess",
           "fp_layernorm": "layernorm", "fp_frames_path": "frames_path",
           "enc_pool_queries": "n_queries", "enc_pool_dim": "d_model",
           "enc_frame_window": "frame_window", "enc_channels": "channels",
           "enc_activation": "activation", "enc_attn_hidden": "attn_hidden",
           "enc_proj_hidden": "proj_hidden", "enc_normalize": "normalize",
           "enc_head": "head", "enc_taps": "taps", "enc_tap_stride": "tap_stride",
           "enc_expand_dim": "expand_dim", "enc_n_heads": "n_heads",
           "enc_expand_norm": "expand_norm", "enc_norm_groups": "norm_groups"}
REF = {"ref_seconds": "seconds", "ref_different_file": "different_file",
       "ref_min_gap_seconds": "min_gap_seconds"}

# An encoder takes only its own kwargs, so the inert keys of the *other* arms are
# dropped rather than carried into a constructor that would reject them.
BY_KIND = {"wavenet": ("channels", "activation", "attn_hidden", "proj_hidden",
                       "normalize", "head", "taps", "tap_stride", "expand_dim",
                       "n_heads", "expand_norm", "norm_groups"),
           "fingerprint": ("path", "preprocess", "layernorm", "proj_hidden",
                           "activation", "normalize"),
           "dac_frames": ("frames_path", "n_queries", "d_model", "proj_hidden",
                          "activation", "normalize", "frame_window")}


def _pick(src: dict, table: dict) -> dict:
    return {new: src[old] for old, new in table.items() if old in src}


def to_spec(ck: dict, data_dir: str, results_dir: str) -> dict:
    """Build the v2 spec dict from a v1 checkpoint's payload."""
    e = ck.get("emulate_cfg", {})
    j = ck.get("joint_cfg")
    spec = {
        "name": ck.get("name", ""),
        "data_dir": data_dir,
        "results_dir": results_dir,
        "seed": int(ck.get("seed", 1234)),
        "sample_rate": int(ck.get("sample_rate", 48000)),
        "output_format": "flac",
        "model": _pick(e, MODEL),
        "data": {"dataset": "joint" if j is not None else "emulation",
                 **_pick(e, DATA),
                 "sources": list((j or {}).get("sources", []) or [])},
        "loss": {"name": "esr_mrstft",
                 "device_loss_weighting": e.get("device_loss_weighting", 0.0),
                 **_pick(e, LOSS)},
        "training": _pick(e, TRAINING),
    }
    if j is not None:
        enc = _pick(j, ENCODER)
        kind = enc.get("kind", "wavenet")
        spec["joint"] = {
            "encoder": {"kind": kind,
                        **{k: v for k, v in enc.items() if k in BY_KIND[kind]}},
            "ref": _pick(j, REF),
            "enc_lr": j.get("enc_lr", 4.0e-3),
            "enc_weight_decay": j.get("enc_weight_decay", -1.0),
            "grad_checkpoint_encoder": j.get("grad_checkpoint_encoder", False),
        }
    return spec


def migrate_file(path: Path, *, data_dir: str, results_dir: str, dry_run: bool,
                 backup: bool) -> str:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "spec" in ck:
        return "already v2"
    if "emulate_cfg" not in ck:
        return "no config payload — not a training checkpoint"
    spec = to_spec(ck, data_dir, results_dir)
    if dry_run:
        import json
        return "would write spec:\n" + json.dumps(spec, indent=6)
    ck["spec"] = spec
    for key in ("emulate_cfg", "joint_cfg"):
        ck.pop(key, None)
    if backup:
        path.replace(path.with_suffix(".pt.v1"))
    tmp = path.with_suffix(".tmp")
    torch.save(ck, tmp)
    tmp.replace(path)
    return "migrated" + (f" (backup at {path.name}.v1)" if backup else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--data-dir", default="./data",
                    help="corpus the run trained on; v1 checkpoints did not record it")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--dry-run", action="store_true", help="print, change nothing")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    for run_dir in args.run_dirs:
        for name in ("checkpoint.pt", "last.pt"):
            path = Path(run_dir) / name
            if not path.is_file():
                continue
            status = migrate_file(path, data_dir=args.data_dir,
                                  results_dir=args.results_dir,
                                  dry_run=args.dry_run, backup=not args.no_backup)
            print(f"  {path}: {status}")


if __name__ == "__main__":
    main()
