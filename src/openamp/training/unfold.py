"""Training the unfolder: K learned enrollment steps between two frozen networks.

The third loop, beside emulate and joint, and the only one that trains neither
network: the generator (an emulate run) and the first guess (a joint run's
encoder) are both finished and frozen, and what trains is
:class:`~openamp.enroll.unfold.Unfolder` -- the per-step ``V`` and step sizes.

**One training example is one amp.** Its encoder guess, K unfolded steps on
``unfold.clips`` of its clips, then the loss: its pre-emphasized ESR on as many
*different* clips. By default all from the amp's train files in the run's own
corpus; ``unfold.clips_data_dir`` / ``fit_split`` / ``loss_split`` point them
elsewhere -- e.g. fit on a NAM capture file's train region, loss on its val
region -- with each amp's renders level-matched to the run's corpus and checked
to be the same capture (``nam_sha256``). The held-out amps' fit clips follow
``fit_split`` too. ``training.batch_size`` is amps per update. The objective is ESR only -- no STFT term, whatever the
generator's own loss carried -- because ESR is the number enrollment is judged by.

**Which amps do what.** Training uses the generator's own training amps. The
generator's holdout is split by sorted id, so no amp both picks the checkpoint
and scores it:

- even positions **select**: unfolded from fixed train-file clips, scored on
  their *val* files every epoch. Best select median -> ``checkpoint.pt``; early
  stopping watches it.
- odd positions **report**: the same, scored on their *test* files with
  :data:`~openamp.enroll.fit.TEST_PAIRS_PER_DEVICE` clips each, at the start
  (encoder alone, and the untrained unfolder) and once at the end on the best
  checkpoint. These are the numbers ``metrics.json`` carries.

Scores use the test semantics (raw per-window ESR, silence-gated, mean per amp,
then median over amps) -- the definition ``enroll`` reports, so an unfolded
amp reads directly against a fully enrolled one.

**Training knobs read:** ``lr`` (``V``), ``batch_size``, ``epochs``,
``early_stop_patience``, ``num_workers``, ``amp``/``amp_dtype``, ``val_pairs``
(select clips, split over the select amps) and ``log_every`` (updates). No LR
schedule and no weight decay: decay would pull ``V`` toward zero, not toward the
``W`` it started from, and nothing here has needed a schedule yet.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from openamp.data import (AmpClipsDataset, CorpusPaths, EmulationDataset, make_loader,
                          manifest_signature)
from openamp.enroll.fit import TEST_PAIRS_PER_DEVICE
from openamp.enroll.unfold import Unfolder, per_amp_esr, zero_shot_vectors
from openamp.evaluation import SILENCE_DBFS, load_model, resolve_device
from openamp.evaluation.metrics import silence_floor, windowed_esr
from openamp.losses import build_loss
from openamp.spec import RunSpec, spec_from_checkpoint
from openamp.training import (GRAD_CLIP_NORM, RunDir, amp_setup, check_resume_compatible,
                              param_group_lrs)

__all__ = ["train", "score", "split_holdout", "LOG_COLUMNS"]

LOG_COLUMNS = ["epoch", "train_esr", "train_esr_start", "select_median", "select_mean",
               "step_len", "v_drift", "lr_v", "lr_step", "elapsed_s"]


def split_holdout(holdout_ids) -> tuple[list[int], list[int]]:
    """``(select, report)``: even and odd positions of the sorted holdout."""
    ids = sorted(int(d) for d in holdout_ids)
    return ids[0::2], ids[1::2]


@torch.no_grad()
def score(gen, vecs: torch.Tensor, ids, paths, split: str, pairs: int, *, seed: int,
          clip: int, batch_size: int, num_workers: int, dev) -> np.ndarray:
    """Per-amp test-semantics ESR of ``vecs`` ``[N, D]`` on ``split``, fp32.

    The same definition as :func:`openamp.evaluation.metrics.per_device_esr`'s
    test path; written out here because the vectors are not a table.
    """
    R = gen.receptive_field
    ids = [int(d) for d in ids]
    ds = EmulationDataset(paths, split, receptive_field=R,
                          id_to_idx={d: i for i, d in enumerate(ids)}, clip_samples=clip,
                          pairs_per_epoch=int(pairs) * len(ids), seed=seed)
    loader = make_loader(ds, batch_size, num_workers, shuffle=False, seed=seed,
                         drop_last=False)
    floor = silence_floor(clip, SILENCE_DBFS)
    num = torch.zeros(len(ids), dtype=torch.float64, device=dev)
    den = torch.zeros_like(num)
    for b in loader:
        di = b["device_idx"].to(dev, non_blocking=True)
        out = gen.forward_emb(b["input"].to(dev, non_blocking=True), vecs[di])[..., R:]
        ratios, keep = windowed_esr(out.squeeze(1), b["target"].to(dev), floor)
        if ratios.numel():
            num.index_add_(0, di[keep], ratios.double())
            den.index_add_(0, di[keep], torch.ones_like(ratios).double())
    arr = (num / den).cpu().numpy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _clip_gains(paths, clip_paths, ids) -> dict[int, float]:
    """Per-amp target gain putting ``clip_paths``' renders on ``paths``' level.

    Each corpus's renders are peak-normalized per amp (``output_scale``), so the
    same capture can sit at a different level in two corpora; the generator
    learned the run's own. Also the guard that the two corpora rendered the same
    captures at all: ``nam_sha256`` must agree for every amp, because a manifest
    path is not evidence of which capture a render came from.
    """
    if clip_paths == paths:
        return {}
    run = pd.read_parquet(paths.renders_manifest).groupby("device_id").first()
    other = pd.read_parquet(clip_paths.renders_manifest).groupby("device_id").first()
    missing = sorted(set(ids) - set(other.index))
    if missing:
        raise RuntimeError(f"{clip_paths.renders_manifest} has no renders for amps "
                           f"{missing[:10]}")
    bad = [d for d in ids if run.loc[d, "nam_sha256"] != other.loc[d, "nam_sha256"]]
    if bad:
        raise RuntimeError(f"{len(bad)} amps were rendered from a different capture in "
                           f"{clip_paths.renders_manifest} than in the run's corpus: "
                           f"{bad[:10]}")
    return {int(d): float(run.loc[d, "output_scale"] / other.loc[d, "output_scale"])
            for d in ids}


def _fit_clips(ds: AmpClipsDataset, num_workers: int):
    """Every amp's fit clips, drawn once and kept on the CPU: ``([N, n, R+T], [N, n, T])``."""
    loader = make_loader(ds, 1, num_workers, shuffle=False, seed=ds.seed, drop_last=False)
    xs, ts = zip(*[(b["xf"][0], b["tf"][0]) for b in loader])
    return torch.stack(xs), torch.stack(ts)


def _unfold_all(unf, gen, e0, x, t, coeff, amp_dtype, chunk: int) -> torch.Tensor:
    """Unfold every held-out amp from its fixed clips, ``chunk`` amps at a time."""
    out = []
    with torch.no_grad():
        for i in range(0, len(e0), chunk):
            e, _, _ = unf(gen, e0[i:i + chunk], x[i:i + chunk].to(e0.device),
                          t[i:i + chunk].to(e0.device), coeff, amp_dtype=amp_dtype)
            out.append(e)
    return torch.cat(out)


def _train_step(unf, gen, opt, b, e0, coeff, amp_dtype, dev) -> tuple[float, float, float]:
    """One update on one batch of amps. A function so its ~GiB of tensors die on return."""
    xf, tf = b["xf"].to(dev, non_blocking=True), b["tf"].to(dev, non_blocking=True)
    xs, ts = b["xs"].to(dev, non_blocking=True), b["ts"].to(dev, non_blocking=True)
    eK, esr0, step_len = unf(gen, e0, xf, tf, coeff, amp_dtype=amp_dtype)
    A, n = xs.shape[:2]
    with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        y = gen.forward_emb(xs.reshape(A * n, -1), eK.repeat_interleave(n, 0))
    loss = per_amp_esr(y[..., gen.receptive_field:].float().view(A, n, -1), ts,
                       coeff).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(unf.parameters(), GRAD_CLIP_NORM)
    opt.step()
    return loss.item(), float(esr0.mean()), step_len


def train(spec: RunSpec, *, device: str = "cuda", epochs: int | None = None,
          resume: bool = False) -> dict:
    if not spec.is_unfold:
        raise ValueError("training.unfold.train needs a run spec with an unfold: section")
    u, tcfg = spec.unfold, spec.training
    dev = torch.device(resolve_device(device))
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    torch.backends.cudnn.benchmark = True
    max_epochs = int(epochs if epochs is not None else tcfg.epochs)

    # --- The frozen halves ------------------------------------------------------
    gen, gck = load_model(Path(u.generator_run), str(dev))
    gen.requires_grad_(False)
    gen.eval()
    gspec = spec_from_checkpoint(gck)
    # The generator *is* the model; its corpus and loss come with it.
    spec = spec.replace(model=gspec.model, data=gspec.data, loss=gspec.loss,
                        sample_rate=gspec.sample_rate)
    unf = Unfolder(gen, u.steps).to(dev)
    R, D = gen.receptive_field, gen.embedding_dim
    coeff = float(build_loss(spec.loss).coeff)
    clip = spec.clip_samples
    use_amp, amp_dtype, _ = amp_setup(tcfg, dev)
    adt = amp_dtype if use_amp else None

    train_ids = sorted(int(d) for d in gck["device_ids"])
    holdout = sorted(int(d) for d in gck["holdout_ids"])
    sel_ids, rep_ids = split_holdout(holdout)
    paths = spec.paths
    sig = manifest_signature(paths)
    if gck.get("manifest_sha256") and sig != gck["manifest_sha256"]:
        print("[unfold] WARNING: manifest changed since the generator was trained — "
              "renders may not match what it saw")

    run = RunDir(spec.run_dir)
    cache = run.path / "encoder_vectors.pt"
    got = torch.load(cache, weights_only=False) if cache.is_file() else {}
    if (got.get("encoder_run") != u.encoder_run
            or not set(train_ids + holdout) <= set(got.get("vectors", {}))):
        t0 = time.time()
        got = {"encoder_run": u.encoder_run, "frames_path": u.frames_path,
               "vectors": zero_shot_vectors(Path(u.encoder_run), train_ids + holdout, dev,
                                            frames_path=u.frames_path)}
        torch.save(got, cache)
        print(f"[unfold] encoder guesses for {len(got['vectors'])} amps "
              f"({(time.time() - t0) / 60:.1f} min) -> {cache}")
    raw = got["vectors"]

    def e0_of(ids) -> torch.Tensor:
        v = torch.from_numpy(np.stack([raw[int(d)] for d in ids])).to(dev)
        return gen.project_embedding(v).detach()

    # --- Data -------------------------------------------------------------------
    clip_paths = (CorpusPaths.from_data_dir(Path(u.clips_data_dir), spec.output_format)
                  if u.clips_data_dir else paths)
    gains = _clip_gains(paths, clip_paths, train_ids + holdout)

    def clips_base(split: str) -> EmulationDataset:
        return EmulationDataset(clip_paths, split, receptive_field=R,
                                id_to_idx={d: i for i, d in enumerate(train_ids + holdout)},
                                clip_samples=clip, pairs_per_epoch=1, seed=spec.seed)

    fit_base = clips_base(u.fit_split)
    loss_base = fit_base if u.loss_split == u.fit_split else clips_base(u.loss_split)
    train_ds = AmpClipsDataset(fit_base, train_ids, u.clips, seed=spec.seed,
                               score_base=loss_base, gains=gains)
    sel_x, sel_t = _fit_clips(AmpClipsDataset(fit_base, sel_ids, u.clips, seed=spec.seed + 1,
                                              gains=gains), tcfg.num_workers)
    rep_x, rep_t = _fit_clips(AmpClipsDataset(fit_base, rep_ids, u.clips, seed=spec.seed + 2,
                                              gains=gains), tcfg.num_workers)
    if u.clips_data_dir:
        print(f"[unfold] clips from {u.clips_data_dir}: "
              f"fit on {u.fit_split} ({len(fit_base.files)} file(s)), loss on "
              f"{u.loss_split} ({len(loss_base.files)} file(s)); "
              f"{sum(abs(g - 1) > 1e-6 for g in gains.values())} amps level-matched")
    sel_e0, rep_e0 = e0_of(sel_ids), e0_of(rep_ids)
    A = int(tcfg.batch_size)
    sel_pairs = max(1, int(tcfg.val_pairs) // max(len(sel_ids), 1))

    def score_on(vecs, ids, split, pairs) -> np.ndarray:
        return score(gen, vecs, ids, paths, split, pairs, seed=spec.seed + 7, clip=clip,
                     batch_size=u.clips, num_workers=tcfg.num_workers, dev=dev)

    def select() -> np.ndarray:
        return score_on(_unfold_all(unf, gen, sel_e0, sel_x, sel_t, coeff, adt, A),
                        sel_ids, "val", sel_pairs)

    def report() -> np.ndarray:
        return score_on(_unfold_all(unf, gen, rep_e0, rep_x, rep_t, coeff, adt, A),
                        rep_ids, "test", TEST_PAIRS_PER_DEVICE)

    # --- Optim / resume ---------------------------------------------------------
    opt = torch.optim.Adam([{"params": [unf.V], "lr": tcfg.lr, "name": "V"},
                            {"params": [unf.log_s], "lr": u.step_lr, "name": "step"}])
    n_params = sum(p.numel() for p in unf.parameters())

    def save_ckpt(path: Path, epoch: int, sel_median: float) -> None:
        run.save_checkpoint(path, {
            "unfolder": unf.state_dict(), "optim": opt.state_dict(),
            "spec": spec.to_dict(), "device_ids": train_ids, "holdout_ids": holdout,
            "select_ids": sel_ids, "report_ids": rep_ids, "receptive_field": R,
            "manifest_sha256": sig, "params": n_params, "name": spec.name,
            "epoch": epoch, "select_median": sel_median, "best_select_median": best,
            "since_improved": since, "init": init})

    start_epoch, best, since, resumed_from, init = 0, float("inf"), 0, None, {}
    if resume and run.last.is_file():
        ck = torch.load(run.last, map_location=dev, weights_only=False)
        saved = ck["spec"].get("unfold", {})
        structural = ("generator_run", "encoder_run", "steps")
        check_resume_compatible(
            ck, [({k: saved.get(k) for k in structural},
                  {k: u.to_dict()[k] for k in structural}, "unfold.")],
            receptive_field=R, n_devices=len(train_ids),
            dynamics="lr, step_lr, clips, init_step_frac, batch_size, epochs")
        unf.load_state_dict(ck["unfolder"])
        opt.load_state_dict(ck["optim"])
        for g in opt.param_groups:             # the config is authoritative on resume
            g["lr"] = tcfg.lr if g["name"] == "V" else u.step_lr
        resumed_from, init = int(ck["epoch"]), dict(ck["init"])
        start_epoch = resumed_from + 1
        best, since = float(ck["best_select_median"]), int(ck["since_improved"])
        print(f"[resume] from epoch {start_epoch} (best select median {best:.5f}); "
              f"lr_v={tcfg.lr:.1e} lr_step={u.step_lr:.1e} (from config)")
    else:
        run.start_log(LOG_COLUMNS)
        train_ds.epoch = 0
        b = [train_ds[i] for i in range(min(A, len(train_ids)))]
        step_len = unf.calibrate(gen, e0_of(train_ids[:len(b)]),
                                 torch.stack([q["xf"] for q in b]).to(dev),
                                 torch.stack([q["tf"] for q in b]).to(dev),
                                 coeff, u.init_step_frac, amp_dtype=adt)
        del b
        print(f"[unfold] K={unf.steps} L={unf.V.shape[1]} r={unf.V.shape[2]} D={D} "
              f"trainable={n_params:,} | step length {step_len:.4f} | amps: train "
              f"{len(train_ids)}, select {len(sel_ids)}, report {len(rep_ids)} | "
              f"{A} amps x {u.clips} clips per update | amp={use_amp}", flush=True)
        t0 = time.time()
        enc = score_on(rep_e0, rep_ids, "test", TEST_PAIRS_PER_DEVICE)
        rep0, sel0 = report(), select()
        init = {"report_encoder_median": float(np.nanmedian(enc)),
                "report_encoder_mean": float(np.nanmean(enc)),
                "report_untrained_median": float(np.nanmedian(rep0)),
                "report_untrained_mean": float(np.nanmean(rep0)),
                "select_untrained_median": float(np.nanmedian(sel0))}
        best = init["select_untrained_median"]
        print(f"[init] report: encoder alone median {init['report_encoder_median']:.4f} "
              f"mean {init['report_encoder_mean']:.4f} | untrained median "
              f"{init['report_untrained_median']:.4f} mean "
              f"{init['report_untrained_mean']:.4f} | select median {best:.4f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        run.append_log([-1, "", "", f"{best:.6f}", f"{np.nanmean(sel0):.6f}", "",
                        f"{unf.drift():.6f}", f"{tcfg.lr:.2e}", f"{u.step_lr:.2e}", "0.0"])
        save_ckpt(run.checkpoint, -1, best)  # never report worse than the untrained steps
    run.save_spec(spec)
    run.append_config_history(mode="resume" if resumed_from is not None else "start",
                              start_epoch=start_epoch, resumed_from=resumed_from,
                              best_val=best, max_epochs=max_epochs, spec=spec.to_dict())

    # --- Loop -------------------------------------------------------------------
    t0 = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs):
        ep_t0 = time.time()
        train_ds.epoch = epoch
        loader = make_loader(train_ds, A, tcfg.num_workers, shuffle=True,
                             seed=spec.seed + epoch, drop_last=True)
        losses, starts, lens = [], [], []
        for it, b in enumerate(loader):
            ids = [train_ids[int(j)] for j in b["amp"]]
            loss, esr0, step_len = _train_step(unf, gen, opt, b, e0_of(ids), coeff, adt, dev)
            losses.append(loss); starts.append(esr0); lens.append(step_len)
            if it % tcfg.log_every == 0:
                print(f"  e{epoch:02d} update {it:4d}/{len(loader)} ESR {loss:.4f} "
                      f"(from {esr0:.4f}) step {step_len:.4f} "
                      f"{(time.time() - ep_t0) / 60:.1f} min", flush=True)
            del b
        if dev.type == "cuda":
            torch.cuda.empty_cache()           # the select pass needs the same memory again

        sel = select()
        med = float(np.nanmedian(sel))
        improved = med < best
        if improved:
            best, since = med, 0
            save_ckpt(run.checkpoint, epoch, med)
        else:
            since += 1
        save_ckpt(run.last, epoch, med)
        lrs = param_group_lrs(opt)
        run.append_log([epoch, f"{np.mean(losses):.6f}", f"{np.mean(starts):.6f}",
                        f"{med:.6f}", f"{np.nanmean(sel):.6f}", f"{np.mean(lens):.5f}",
                        f"{unf.drift():.6f}", f"{lrs['V']:.2e}", f"{lrs['step']:.2e}",
                        f"{time.time() - t0:.1f}"])
        print(f"[epoch {epoch:02d}] train ESR {np.mean(losses):.4f} | select median "
              f"{med:.4f} mean {np.nanmean(sel):.4f} best {best:.4f}"
              f"{'  *saved*' if improved else ''} | V drift {unf.drift():.4f} | "
              f"{(time.time() - ep_t0) / 60:.1f} min", flush=True)
        if since >= tcfg.early_stop_patience:
            print(f"[stop] no select improvement for {since} epochs")
            break

    # --- Report: the best checkpoint on amps nothing here chose by ---------------
    ck = torch.load(run.checkpoint, map_location=dev, weights_only=False)
    unf.load_state_dict(ck["unfolder"])
    rep = report()
    pd.DataFrame({"device_id": rep_ids, "test_esr": rep}).to_csv(
        run.path / "report_per_device.csv", index=False)
    metrics = {
        "name": spec.name, "steps": unf.steps, "params": n_params,
        "generator_run": u.generator_run, "encoder_run": u.encoder_run,
        "n_train": len(train_ids), "n_select": len(sel_ids), "n_report": len(rep_ids),
        **init, "best_epoch": int(ck["epoch"]), "best_select_median": float(best),
        "report_median": float(np.nanmedian(rep)), "report_mean": float(np.nanmean(rep)),
        "epochs_run": epoch + 1, "train_hours": round((time.time() - t0) / 3600.0, 4),
        "manifest_sha256": sig,
    }
    run.save_metrics(metrics)
    print(f"[report] best epoch {metrics['best_epoch']}: report median "
          f"{metrics['report_median']:.4f} mean {metrics['report_mean']:.4f} (untrained "
          f"{init.get('report_untrained_median', float('nan')):.4f}, encoder alone "
          f"{init.get('report_encoder_median', float('nan')):.4f}) -> {run.path}")
    return metrics
