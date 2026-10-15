"""Fitting an embedding for an amp the network never trained on.

The generator is frozen; only the embedding moves. Two entry points, differing
in what the amp is available as:

- :func:`~openamp.enroll.fit.enroll` — a held-out device that *is* in the
  corpus, with rendered train/val/test splits. The Phase 5 measurement: how
  close does a fitted vector get to a trained row?
- :func:`~openamp.enroll.fit.enroll_pair` — a raw ``(dry, wet)`` capture pair
  from outside the corpus entirely, which is what a user actually has.

The pair path needs the two signals on one sample grid first, which is
:mod:`~openamp.enroll.alignment`'s job — a reamp's wet file trails its dry by an
amount nothing in the file records.

The amortized counterpart — mapping an amp straight to a vector instead of
fitting one — is :mod:`openamp.encoders`, trained by
:mod:`openamp.training.joint`. Between the two, :mod:`~openamp.enroll.unfold`
takes an encoder's guess and a few *learned* fitting steps, trained by
:mod:`openamp.training.unfold`.
"""

from __future__ import annotations

from openamp.enroll.alignment import (blip_lag, estimate_lag, nam_signal_regions,
                                      render_dry)
from openamp.enroll.fit import enroll, enroll_pair, load_pair_model
from openamp.enroll.unfold import Unfolder, load_unfold, zero_shot_vectors

__all__ = ["enroll", "enroll_pair", "load_pair_model",
           "estimate_lag", "blip_lag", "nam_signal_regions", "render_dry",
           "Unfolder", "load_unfold", "zero_shot_vectors"]
