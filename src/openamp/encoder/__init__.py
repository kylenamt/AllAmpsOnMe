"""Cross-attention pooling over frozen DAC frames, regressed onto a trained table.

A feasibility probe, deliberately standalone: nothing here is wired into
``openamp.joint_model``, ``openamp.cli`` or ``openamp.core.config``, and nothing
imports from here. Two argparse entry points and a printed report.

**The question.** Every conditioning front-end tried in this project recognizes
amps and fails to *place* them -- ``joint/proto`` reaches holdout retrieval top-1
0.978 while its holdout ESR is 0.762, against 0.254 for a vector fitted directly
on the same amps. Rather than train an encoder against reconstruction loss and
hope placement follows, this probe supervises placement directly: the emulate
run's already-trained embedding table *is* the regression target, and only a
pooler is trained. The generator never appears in the loop -- it is loaded once,
frozen, to turn a predicted embedding into a held-out ESR.

**Why cross-attention rather than the mean+std of**
:mod:`openamp.encoders.fingerprint`. Measured over the 405 devices
those two artifacts share: the trained table's cloud has a participation ratio
of 52.5, the mean+std fingerprint cloud 2.84. A 52-dimensional target cannot be
predicted from a 2.8-dimensional input, and that 2.84 is a property of the
*pooling*, not of DAC's frames -- the frames are 14,728 x 1024 before anything
collapses them. Whether a learned pooler recovers more of that is the whole
experiment, and it has a number to beat before any ESR is measured.

- :mod:`~openamp.encoder.extract` -- frozen DAC over each render, frames kept.
- :mod:`~openamp.encoder.probe`   -- the pooler, the regression, the report.

``python -m openamp.encoder.extract`` then ``python -m openamp.encoder.probe``.
"""

from __future__ import annotations

__all__ = ["extract", "probe"]
