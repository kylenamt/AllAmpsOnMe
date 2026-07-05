"""SimCLR contrastive training loop (Phase 3 spec §3).

Trains :class:`models.encoder.ConvEncoder` on Phase 2's
:class:`data.datasets.ContrastivePairDataset`: two 2 s crops of the *same*
device are a positive pair, other devices in the batch are negatives. A plain
single-GPU PyTorch loop — no Lightning. Best checkpoint is chosen by in-batch
retrieval on the ``val`` split; a self-contained checkpoint (weights + model
config) is written so ``eval/embed.py`` can reconstruct the encoder.

Run::

    python -m train.contrastive                       # full run (configs/phase3.yaml)
    python -m train.contrastive --iterations 2000     # sanity milestone
    python -m train.contrastive --resume              # continue from last.pt

Fresh random crops each epoch: the Phase 2 dataset is deterministic per index,
so we bump its ``seed`` per epoch and rebuild the loader — reproducible, yet the
encoder sees new crop positions every pass (spec §3 "two random 2 s crops").
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from .config import Phase3Config, load_config


# --- LR schedule ---------------------------------------------------------------
def lr_scale(step: int, warmup: int, total: int, schedule: str) -> float:
    """Multiplicative LR factor in [0, 1]: linear warmup then cosine/step/none."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if schedule == "none":
        return 1.0
    if schedule == "step":
        return 0.1 ** (3 * max(step - warmup, 0) // max(total - warmup, 1))
    # cosine (default): decay to 0 over the post-warmup span
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# --- Retrieval on a fixed val loader -------------------------------------------
def evaluate_retrieval(encoder, loader, device, n_batches: int) -> float:
    """Mean in-batch top-1 retrieval over ``n_batches`` of the val loader."""
    import torch

    from .objectives import retrieval_top1

    encoder.eval()
    accs = []
    with torch.no_grad():
        for k, batch in enumerate(loader):
            if k >= n_batches:
                break
            v1 = batch["view1"].to(device, non_blocking=True)
            v2 = batch["view2"].to(device, non_blocking=True)
            h1, h2 = encoder(v1), encoder(v2)
            accs.append(retrieval_top1(h1, h2))
    encoder.train()
    return float(sum(accs) / max(len(accs), 1))


def _make_loader(dataset, batch_size, workers, *, shuffle, seed, drop_last):
    import torch
    from torch.utils.data import DataLoader

    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, drop_last=drop_last, pin_memory=True,
                      generator=gen, persistent_workers=False)


def train(cfg: Phase3Config, *, data_config_path: Path | None = None,
          device: str = "cuda", resume: bool = False,
          iterations: int | None = None) -> dict:
    """Run the contrastive training loop; returns a small summary dict."""
    import numpy as np
    import torch

    from data.config import load_config as load_phase2_config
    from data.datasets import ContrastivePairDataset
    from models.encoder import (build_encoder, build_projection_head,
                                count_parameters)

    from .objectives import nt_xent_loss, retrieval_top1

    tcfg = cfg.train
    total_iters = int(iterations if iterations is not None else tcfg.iterations)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True   # fixed shapes -> autotune (speed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # --- Data (Phase 2 renders) -------------------------------------------------
    p2 = load_phase2_config(data_config_path)
    val_spd_min = 8
    if tcfg.use_crop_pool:
        from .cropcache import (CropPoolDataset, build_crop_pool,
                                default_cache_dir)
        cache_dir = Path(tcfg.cache_dir) if tcfg.cache_dir else default_cache_dir()
        crop_n = int(round(tcfg.crop_seconds * p2.sample_rate))       # output 2 s
        store_n = int(round(tcfg.pool_store_seconds * p2.sample_rate))  # stored 3 s
        print(f"[data]  crop pool dir={cache_dir} store={store_n} out={crop_n} (building if missing)")
        build_crop_pool(p2, "train", crops_per_device=tcfg.pool_crops_train,
                        crop_samples=store_n, out_dir=cache_dir, seed=cfg.seed,
                        workers=max(tcfg.num_workers, 8))
        build_crop_pool(p2, "val", crops_per_device=tcfg.pool_crops_val,
                        crop_samples=store_n, out_dir=cache_dir, seed=cfg.seed,
                        workers=max(tcfg.num_workers, 8))
        train_ds = CropPoolDataset(cache_dir, "train",
                                   samples_per_device=tcfg.samples_per_device,
                                   augment_gain=tcfg.augment_gain, seed=cfg.seed,
                                   out_samples=crop_n)
        n_devices = len(train_ds.devices)
        val_spd = max(val_spd_min, (tcfg.val_batches * tcfg.batch_size) // max(n_devices, 1) + 2)
        val_ds = CropPoolDataset(cache_dir, "val", samples_per_device=val_spd,
                                 augment_gain=False, seed=cfg.seed, out_samples=crop_n)
    else:
        train_ds = ContrastivePairDataset(
            "train", p2, samples_per_device=tcfg.samples_per_device,
            augment_gain=tcfg.augment_gain, seed=cfg.seed)
        n_devices = len(train_ds._devices)
        val_spd = max(val_spd_min, (tcfg.val_batches * tcfg.batch_size) // max(n_devices, 1) + 2)
        val_ds = ContrastivePairDataset(
            "val", p2, samples_per_device=val_spd, augment_gain=False, seed=cfg.seed)
    val_loader = _make_loader(val_ds, tcfg.batch_size, tcfg.num_workers,
                              shuffle=True, seed=cfg.seed + 777, drop_last=True)

    # --- Model ------------------------------------------------------------------
    encoder = build_encoder(cfg.model).to(dev)
    head = build_projection_head(cfg.model).to(dev)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    use_amp = bool(tcfg.amp) and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.train_log_path
    start_step, best_val = 0, -1.0
    if resume and cfg.last_ckpt_path.is_file():
        ck = torch.load(cfg.last_ckpt_path, map_location=dev)
        encoder.load_state_dict(ck["encoder"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        start_step = int(ck["step"])
        best_val = float(ck.get("best_val", -1.0))
        print(f"[resume] from step {start_step}, best_val={best_val:.4f}")
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    n_enc = count_parameters(encoder)
    print(f"[model] encoder params={n_enc:,}  proj params={count_parameters(head):,}")
    print(f"[data]  devices={n_devices}  train_len={len(train_ds)}  "
          f"val_len={len(val_ds)}  chance_top1={1.0 / (2 * tcfg.batch_size - 1):.4f}")
    print(f"[run]   device={dev}  amp={use_amp}  iters={total_iters}")

    def save_ckpt(path: Path, step: int, val_acc: float) -> None:
        torch.save({
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "optim": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val": best_val,
            "val_acc": val_acc,
            "model_cfg": asdict(cfg.model),
            "seed": cfg.seed,
            "encoder_params": n_enc,
        }, path)

    def log(rec: dict) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    # --- Loop -------------------------------------------------------------------
    step = start_step
    epoch = step // max(len(train_ds) // tcfg.batch_size, 1)
    t0 = time.time()
    ema_loss, ema_acc = None, None
    sanity_acc = None
    encoder.train()
    head.train()
    while step < total_iters:
        train_ds.seed = cfg.seed + epoch      # fresh crops each epoch
        loader = _make_loader(train_ds, tcfg.batch_size, tcfg.num_workers,
                              shuffle=True, seed=cfg.seed + epoch, drop_last=True)
        for batch in loader:
            if step >= total_iters:
                break
            v1 = batch["view1"].to(dev, non_blocking=True)
            v2 = batch["view2"].to(dev, non_blocking=True)

            for g in opt.param_groups:
                g["lr"] = tcfg.lr * lr_scale(step, tcfg.warmup_iters, total_iters,
                                             tcfg.lr_schedule)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                h1, h2 = encoder(v1), encoder(v2)
                z1, z2 = head(h1), head(h2)
                loss = nt_xent_loss(z1, z2, tcfg.temperature)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                acc = retrieval_top1(h1.detach().float(), h2.detach().float())
            lval = loss.item()
            ema_loss = lval if ema_loss is None else 0.98 * ema_loss + 0.02 * lval
            ema_acc = acc if ema_acc is None else 0.98 * ema_acc + 0.02 * acc

            if step % tcfg.log_every == 0:
                ips = (step - start_step + 1) * tcfg.batch_size / (time.time() - t0)
                cur_lr = opt.param_groups[0]["lr"]
                print(f"step {step:>7d}/{total_iters} loss {ema_loss:6.4f} "
                      f"retr {ema_acc:5.3f} lr {cur_lr:.2e} {ips:6.0f} smp/s")
                log({"step": step, "loss": lval, "loss_ema": ema_loss,
                     "train_retr": acc, "train_retr_ema": ema_acc, "lr": cur_lr})

            if step == tcfg.sanity_iters:
                sanity_acc = ema_acc
                chance = 1.0 / (2 * tcfg.batch_size - 1)
                flag = "OK" if sanity_acc > 5 * chance else "LOW — check pairing/collapse"
                print(f"[sanity] step {step}: retr_ema={sanity_acc:.3f} "
                      f"(chance={chance:.4f}) -> {flag}")

            if step > 0 and step % tcfg.val_every == 0:
                val_acc = evaluate_retrieval(encoder, val_loader, dev, tcfg.val_batches)
                improved = val_acc > best_val
                if improved:
                    best_val = val_acc
                    save_ckpt(cfg.best_ckpt_path, step, val_acc)
                print(f"[val]   step {step}: retr={val_acc:.4f} "
                      f"best={best_val:.4f}{'  *saved*' if improved else ''}")
                log({"step": step, "val_retr": val_acc, "best_val": best_val})

            if step > 0 and step % tcfg.ckpt_every == 0:
                save_ckpt(cfg.last_ckpt_path, step, ema_acc)

            step += 1
        epoch += 1

    # Final val + checkpoints.
    val_acc = evaluate_retrieval(encoder, val_loader, dev, tcfg.val_batches)
    if val_acc > best_val:
        best_val = val_acc
        save_ckpt(cfg.best_ckpt_path, step, val_acc)
    save_ckpt(cfg.last_ckpt_path, step, val_acc)
    summary = {"steps": step, "final_val_retr": val_acc, "best_val_retr": best_val,
               "sanity_retr": sanity_acc, "encoder_params": n_enc,
               "elapsed_s": time.time() - t0, "n_devices": n_devices}
    log({"summary": summary})
    print(f"[done]  steps={step} best_val_retr={best_val:.4f} "
          f"final_val_retr={val_acc:.4f} elapsed={summary['elapsed_s'] / 60:.1f} min")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase 3 contrastive training")
    ap.add_argument("--config", type=Path, default=None, help="configs/phase3.yaml")
    ap.add_argument("--data-config", type=Path, default=None, help="configs/phase2.yaml")
    ap.add_argument("--iterations", type=int, default=None, help="override iteration count")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    train(cfg, data_config_path=args.data_config, device=args.device,
          resume=args.resume, iterations=args.iterations)


if __name__ == "__main__":
    main()
