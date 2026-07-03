"""Probe signal + audio metrics shared by `validate` and `dedup` (spec §5.5, §5.6).

The probe is a fixed, deterministic 5 s test signal:
    2 s log sine sweep 20 Hz–20 kHz @ -12 dBFS  +  1 s silence  +  2 s DI snippet

If ``assets/probe.wav`` is bundled it is used verbatim; otherwise the signal is
synthesized deterministically (same array every run) so validation and ESR-based
dedup are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SWEEP_SECONDS = 2.0
SILENCE_SECONDS = 1.0
DI_SECONDS = 2.0
SWEEP_DBFS = -12.0

# Validation thresholds (spec §5.5.4).
NOT_SILENT_MIN_DBFS = -80.0   # non-silent segments must exceed this
EXPLODE_MAX_DBFS = 6.0        # peak must stay below +6 dBFS
SILENCE_OUT_MAX_DBFS = -40.0  # silence in -> near-silence out

_ASSET_PROBE = Path(__file__).with_name("assets") / "probe.wav"


@dataclass
class Probe:
    signal: np.ndarray            # float32, mono, in [-1, 1]
    sample_rate: int
    sweep: slice
    silence: slice
    di: slice

    @property
    def non_silent(self) -> np.ndarray:
        """Indices expected to produce audible output (sweep + DI)."""
        idx = np.r_[np.arange(self.sweep.start, self.sweep.stop),
                    np.arange(self.di.start, self.di.stop)]
        return idx


def dbfs(amplitude: float) -> float:
    amplitude = float(amplitude)
    if amplitude <= 0:
        return -np.inf
    return 20.0 * np.log10(amplitude)


def peak_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -np.inf
    return dbfs(float(np.max(np.abs(x))))


def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return dbfs(rms)


def esr(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """Error-to-signal ratio (spec §5.6): ||est - tgt||^2 / ||tgt||^2."""
    n = min(len(estimate), len(target))
    if n == 0:
        return np.inf
    est = np.asarray(estimate[:n], dtype=np.float64)
    tgt = np.asarray(target[:n], dtype=np.float64)
    denom = float(np.sum(tgt ** 2))
    if denom < eps:
        # Both silent -> identical; else fully erroneous.
        return 0.0 if float(np.sum(est ** 2)) < eps else np.inf
    return float(np.sum((est - tgt) ** 2) / denom)


def _log_sweep(sr: int, seconds: float, f0: float = 20.0, f1: float = 20000.0,
               amp_dbfs: float = SWEEP_DBFS) -> np.ndarray:
    n = int(round(seconds * sr))
    t = np.arange(n) / sr
    T = max(seconds, 1e-9)
    k = np.log(f1 / f0)
    phase = 2 * np.pi * f0 * T / k * (np.exp(t / T * k) - 1.0)
    amp = 10 ** (amp_dbfs / 20.0)
    return (amp * np.sin(phase)).astype(np.float32)


def _synth_di(sr: int, seconds: float, seed: int = 0) -> np.ndarray:
    """Deterministic pseudo guitar-DI: a few decaying plucked notes.

    Placeholder for a bundled real DI snippet; deterministic so dedup is stable.
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sr))
    out = np.zeros(n, dtype=np.float32)
    notes_hz = [82.41, 110.0, 146.83, 196.0]  # E2 A2 D3 G3
    seg = n // len(notes_hz)
    for i, f in enumerate(notes_hz):
        start = i * seg
        length = seg
        t = np.arange(length) / sr
        env = np.exp(-t * 6.0)
        # fundamental + a couple harmonics with light detune
        wave = (np.sin(2 * np.pi * f * t)
                + 0.5 * np.sin(2 * np.pi * 2 * f * t)
                + 0.25 * np.sin(2 * np.pi * 3 * f * t))
        wave += 0.02 * rng.standard_normal(length)  # pick noise
        out[start:start + length] = (0.3 * env * wave).astype(np.float32)
    return out


def generate_probe(sample_rate: int = 48_000) -> Probe:
    sweep = _log_sweep(sample_rate, SWEEP_SECONDS)
    silence = np.zeros(int(round(SILENCE_SECONDS * sample_rate)), dtype=np.float32)
    if _ASSET_PROBE.is_file():
        di = _load_wav_tail(_ASSET_PROBE, sample_rate, DI_SECONDS)
    else:
        di = _synth_di(sample_rate, DI_SECONDS)
    signal = np.concatenate([sweep, silence, di]).astype(np.float32)
    a = len(sweep)
    b = a + len(silence)
    c = b + len(di)
    return Probe(signal=signal, sample_rate=sample_rate,
                 sweep=slice(0, a), silence=slice(a, b), di=slice(b, c))


def _load_wav_tail(path: Path, sample_rate: int, seconds: float) -> np.ndarray:
    import soundfile as sf  # local import; only when a bundled probe exists

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sample_rate:  # simple linear resample; good enough for a probe
        n_out = int(round(len(data) * sample_rate / sr))
        data = np.interp(np.linspace(0, len(data), n_out, endpoint=False),
                         np.arange(len(data)), data).astype(np.float32)
    n = int(round(seconds * sample_rate))
    if len(data) >= n:
        return data[:n]
    return np.pad(data, (0, n - len(data)))
