"""The optimization-step plumbing both trainers set up identically.

Not the optimizer itself: emulate optimizes one flat parameter list, joint two
named groups at different learning rates, and a wrapper over those two shapes
would be longer than either. What *is* shared is everything around the step --
the LR schedule, the AMP dtype/scaler pair, gradient clipping, and reading
the current learning rates back out for a log row.
"""

from __future__ import annotations

import math

import torch

__all__ = ["GRAD_CLIP_NORM", "build_scheduler", "step_scheduler", "amp_setup",
           "param_group_lrs"]

# Safety net for loss spikes, not a training knob: healthy grad norms on this
# model sit at p50~4, p90~7, so 20 leaves normal steps untouched.
GRAD_CLIP_NORM = 20.0


def build_scheduler(opt, ecfg, *, max_epochs: int, start_epoch: int = 0):
    """The LR schedule named by ``training.scheduler``.

    ``plateau`` (the default) halves on a val-ESR plateau and so depends on how
    the run is going. ``cosine`` ignores val ESR entirely and anneals on epoch
    count alone, from each group's configured lr down to
    ``training.cosine_min_lr_frac`` of it over ``max_epochs``.

    Cosine is a ``LambdaLR``, not ``CosineAnnealingLR``, because that class's
    ``eta_min`` is one scalar shared by every parameter group: the joint
    trainer's generator and encoder run 8x apart, so a shared floor would be a
    different fraction of each. One multiplicative curve gives both groups a
    floor proportional to their own base rate.

    ``start_epoch`` replays a resumed cosine run forward to where it left off.
    Scheduler state is not checkpointed -- the config is authoritative on resume
    -- and unlike plateau, a cosine schedule is a pure function of the epoch
    number, so replaying it is exact rather than a reconstruction.
    """
    kind = ecfg.scheduler
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=ecfg.plateau_factor, patience=ecfg.plateau_patience)
    if kind != "cosine":
        raise ValueError(
            f"training.scheduler must be 'plateau' or 'cosine', got {kind!r}")

    frac = float(ecfg.cosine_min_lr_frac)
    if not 0.0 <= frac <= 1.0:
        raise ValueError(
            f"training.cosine_min_lr_frac must be in [0, 1], got {frac}")
    total = max(1, int(max_epochs))

    def curve(epoch: int) -> float:
        # Clamped, so an --epochs override past max_epochs holds the floor
        # rather than turning the cosine back up.
        t = min(int(epoch), total) / total
        return frac + (1.0 - frac) * 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, curve)
    for _ in range(int(start_epoch)):
        sched.step()
    return sched


def step_scheduler(sched, val_esr: float) -> None:
    """One epoch-end step. Plateau reads the metric; a cosine curve does not."""
    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
        sched.step(val_esr)
    else:
        sched.step()


def amp_setup(ecfg, dev) -> tuple[bool, torch.dtype, torch.amp.GradScaler]:
    """``(use_amp, amp_dtype, scaler)`` for a device.

    AMP is CUDA-only here, so a CPU run silently trains in fp32 whatever the
    config says -- which is what makes a CPU run bit-reproducible. bf16 shares
    fp32's exponent range, so it needs no loss scaling and gets a disabled
    scaler.
    """
    use_amp = bool(ecfg.amp) and dev.type == "cuda"
    if ecfg.amp_dtype not in ("fp16", "bf16"):
        raise ValueError(f"emulate.amp_dtype must be 'fp16' or 'bf16', got {ecfg.amp_dtype!r}")
    amp_dtype = torch.bfloat16 if ecfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    return use_amp, amp_dtype, scaler


def param_group_lrs(opt) -> dict:
    """``{group name: lr}``, for the CSV. Unnamed groups key on their index."""
    return {g.get("name", str(i)): g["lr"] for i, g in enumerate(opt.param_groups)}
