"""The parts of a training run that are not the loop.

Two loops live on, in :mod:`openamp.training.emulate` and
:mod:`openamp.training.joint`: one conditions the generator on a table row,
the other on an encoder's output, and they disagree about what validation means
(the joint loop measures shuffled/zero/holdout probes every epoch). Folding them
into one loop with flags would hide that difference rather than remove it.

What they genuinely shared was the bookkeeping around the loop, and that is
here:

- :mod:`~openamp.training.artifacts` -- :class:`RunDir`, everything a run
  writes, plus the resume compatibility check.
- :mod:`~openamp.training.optim` -- the plateau scheduler, the AMP dtype/scaler
  pair, the gradient-clip norm, and reading learning rates back for a log row.
"""

from __future__ import annotations

from openamp.training.artifacts import (RunDir, check_resume_compatible,
                                        same_config_value)
from openamp.training.optim import (GRAD_CLIP_NORM, amp_setup, build_scheduler,
                                    param_group_lrs)

__all__ = ["RunDir", "check_resume_compatible", "same_config_value",
           "GRAD_CLIP_NORM", "amp_setup", "build_scheduler", "param_group_lrs"]
