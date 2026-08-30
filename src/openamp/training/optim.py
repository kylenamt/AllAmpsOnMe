"""The optimization-step plumbing both trainers set up identically.

Not the optimizer itself: emulate optimizes one flat parameter list, joint two
named groups at different learning rates, and a wrapper over those two shapes
would be longer than either. What *is* shared is everything around the step --
the plateau scheduler, the AMP dtype/scaler pair, gradient clipping, and reading
the current learning rates back out for a log row.
"""

from __future__ import annotations

import torch

__all__ = ["GRAD_CLIP_NORM", "build_scheduler", "amp_setup", "param_group_lrs"]

# Safety net for loss spikes, not a training knob: healthy grad norms on this
# model sit at p50~4, p90~7, so 20 leaves normal steps untouched.
GRAD_CLIP_NORM = 20.0


def build_scheduler(opt, ecfg):
    """Reduce-on-plateau on val ESR -- the only schedule either trainer uses."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=ecfg.plateau_factor, patience=ecfg.plateau_patience)


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
