"""Getting a (dry, wet) capture pair onto the same sample grid, and its regions.

A TONE3000-style capture is a reamp: the wet file trails the dry by the
interface's round-trip latency plus the device's group delay, and nothing in the
file says by how much. Enrollment fits an embedding by comparing the two sample
for sample, so an unaligned pair fits nothing -- alignment is the precondition,
not a nicety.

Two estimators, in order of preference:

- :func:`blip_lag` -- locates NAM's calibration blip in both signals and takes
  the difference of the leading edges. Exact when the capture has one.
- :func:`estimate_lag` -- pre-emphasized FFT cross-correlation, for material
  with no blip. Carries a residual bias equal to the device's group delay.

:func:`nam_signal_regions` then says which spans of the standard capture signal
are what, so a fit can train on one span and validate on a disjoint one.
:func:`render_dry` is the chunked forward pass that turns a fitted embedding
back into audio.
"""

from __future__ import annotations

import numpy as np
import torch

from openamp.evaluation import resolve_device

__all__ = ["estimate_lag", "blip_lag", "nam_signal_regions", "render_dry"]


def estimate_lag(dry: np.ndarray, wet: np.ndarray, *, max_lag: int = 4800,
                 probe_samples: int = 48_000 * 30, preemph: float = 0.95) -> int:
    """Coarse estimate of the samples wet trails dry (negative: early).

    Pre-emphasized FFT cross-correlation, peak taken on |xcorr| (polarity-
    safe). Carries a residual bias equal to the device's group delay --
    fallback for material with no calibration blip; prefer blip_lag when one
    exists. Align with wet[lag:] / dry[-lag:] and trim to the common length.
    """
    n = int(min(len(dry), len(wet), probe_samples))
    a = np.asarray(dry[:n], dtype=np.float64)
    b = np.asarray(wet[:n], dtype=np.float64)
    a = a[1:] - preemph * a[:-1]                     # whiten: sharpen the peak
    b = b[1:] - preemph * b[:-1]
    a -= a.mean()
    b -= b.mean()
    size = 1 << int(np.ceil(np.log2(2 * len(a) - 1)))
    # correlation theorem: irfft(FB * conj(FA))[k] = sum_t wet[t+k] * dry[t]
    xc = np.abs(np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size))
    lags = np.concatenate([np.arange(max_lag + 1), np.arange(-max_lag, 0)])
    vals = np.concatenate([xc[:max_lag + 1], xc[-max_lag:]])
    return int(lags[int(np.argmax(vals))])


def _leading_edge(x: np.ndarray, thresh_frac: float, *, noise_mult: float = 8.0,
                  max_ramp: int = 256) -> int:
    """First arrival of the loudest transient in x.

    Coarse-locates the transient at half-peak, estimates the noise floor ahead
    of it, then takes the first crossing of max(noise_mult*noise, thresh_frac*peak)
    within max_ramp samples before that point -- avoids reporting an amp's
    ramped-up response instead of the true onset.
    """
    seg = np.abs(np.asarray(x, dtype=np.float64) - float(np.mean(x)))
    peak = float(seg.max())
    if peak <= 0.0:
        raise ValueError("no signal in the blip search window")
    coarse = int(np.argmax(seg >= 0.5 * peak))
    lo = max(coarse - int(max_ramp), 0)
    noise = float(np.sqrt(np.mean(seg[:lo] ** 2))) if lo > 32 else 0.0
    thresh = max(noise_mult * noise, thresh_frac * peak)
    window = seg[lo:coarse + 1]
    hit = window >= thresh
    return lo + (int(np.argmax(hit)) if hit.any() else int(len(window) - 1))


def blip_lag(dry: np.ndarray, wet: np.ndarray, *, sample_rate: int = 48_000,
             search_seconds: float = 2.0, thresh_frac: float = 0.02) -> int:
    """Latency (samples wet trails dry) from the leading NAM/TONE3000 capture blip.

    First-arrival, not cross-correlation: the interface round-trip is a pure
    delay L, while the amp's impulse response starts immediately but has its
    energy centroid (group delay) a few samples in. Leading-edge recovers L
    alone and leaves the group delay in the target; xcorr conflates the two
    (estimate_lag's bias). Requires the real blip at the very start of the
    signal; ValueError if a window is silent.
    """
    ns = int(min(len(dry), len(wet), round(search_seconds * sample_rate)))
    if ns < 2:
        raise ValueError("signals too short for a blip search")
    return int(_leading_edge(wet[:ns], thresh_frac) - _leading_edge(dry[:ns], thresh_frac))


# NAM standardized reamp-signal layout (sdatkinson/neural-amp-modeler,
# nam/train/core.py): train = signal[train_start:validation_start] (blips
# kept, lead-in dropped), validation = signal[validation_start:]. Only v3.0.0
# (current TONE3000 sweep) is listed; add older signals here as needed.
_NAM_SIGNALS = {
    "v3_0_0": {"length": 9_120_000, "train_start": 480_000,      # 3:10 @ 48 kHz
               "validation_start": -432_000, "blips": (504_000, 552_000)},
}


def nam_signal_regions(n_samples: int, *, sample_rate: int = 48_000,
                       version: str | None = None, tol: float = 0.005) -> dict | None:
    """NAM's own train/val regions for a standardized reamp signal, or None.

    Mirrors NAM's own split so a pair fit trains/validates on the same regions
    NAM reports, not a blind last-val_frac slice. Signal identified by length
    (within tol) unless version forces a layout. None when n_samples matches no
    known signal or the rate isn't 48kHz -- callers fall back to val_frac.
    """
    if sample_rate != 48_000:
        return None
    if version is not None and version not in _NAM_SIGNALS:
        raise ValueError(f"unknown NAM signal version {version!r}; "
                         f"known: {sorted(_NAM_SIGNALS)}")
    items = ([(version, _NAM_SIGNALS[version])] if version is not None
             else _NAM_SIGNALS.items())
    for name, spec in items:
        if abs(n_samples - spec["length"]) <= tol * spec["length"]:
            ts = int(spec["train_start"])
            vs = n_samples + int(spec["validation_start"])   # negative -> from end
            return {"version": name, "train": (ts, vs), "val": (vs, n_samples),
                    "lead_in": (0, ts), "blips": tuple(spec["blips"])}
    return None


def _norm_region(region: tuple[int, int], n: int) -> tuple[int, int]:
    """Validate a (start, stop) sample region, resolving negatives from n."""
    a, b = region
    a = int(a) if a >= 0 else n + int(a)
    b = int(b) if b >= 0 else n + int(b)
    if not 0 <= a < b <= n:
        raise ValueError(f"region {region} out of bounds for length {n}")
    return a, b


@torch.no_grad()
def render_dry(model, dry: np.ndarray, *, row: int = 0, device: str = "cpu",
               chunk_samples: int = 480_000) -> np.ndarray:
    """Stream dry through the model with real left-context; returns [len(dry)].

    Chunked so multi-minute captures fit in memory; each chunk re-reads its own
    receptive-field context, so output matches a single full-length forward.
    """
    dev = torch.device(resolve_device(device))
    R = model.receptive_field
    di = torch.tensor([int(row)], dtype=torch.long, device=dev)
    out = np.empty(len(dry), dtype=np.float32)
    for s in range(0, len(dry), int(chunk_samples)):
        e = min(s + int(chunk_samples), len(dry))
        a, pad = s - R, 0
        if a < 0:
            pad, a = -a, 0
        x = dry[a:e].astype(np.float32, copy=False)
        if pad:
            x = np.concatenate([np.zeros(pad, dtype=np.float32), x])
        xt = torch.from_numpy(np.ascontiguousarray(x))[None].to(dev)
        out[s:e] = model(xt, di)[..., R:].squeeze(0).squeeze(0).cpu().numpy()
    return out


