"""Audio I/O, level/error metrics, and the deterministic validation probe.

The one place that depends on ``soundfile`` and ``torchaudio``. Every heavy
import is local so the rest of the package (and its unit tests) load on a bare
numpy stack. If these deps are missing, a clear install hint is raised — the
same pattern ``openamp.nam`` uses for torch/NAM.

The probe is a fixed, deterministic 5 s test signal used by the acquisition
``validate`` and ``dedup`` stages:

    2 s log sine sweep 20 Hz–20 kHz @ -12 dBFS  +  1 s silence  +  2 s DI snippet

It is synthesized deterministically (same array every run) so validation and
ESR-based dedup are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_INSTALL_HINT = (
    "Audio I/O needs soundfile + torchaudio. Install the package: "
    "`pip install -e .`"
)


class AudioIOError(RuntimeError):
    """Raised when an audio file cannot be read/written or resampled."""


def _soundfile():
    try:
        import soundfile as sf  # local heavy import
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise AudioIOError(_INSTALL_HINT) from exc
    return sf


def audio_info(path: str | Path) -> tuple[int, int]:
    """Return ``(num_frames, sample_rate)`` without decoding the whole file."""
    sf = _soundfile()
    info = sf.info(str(path))
    return int(info.frames), int(info.samplerate)


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a file to mono float32 in [-1, 1]; return ``(signal, sample_rate)``."""
    sf = _soundfile()
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    data = _to_mono(data)
    return data, int(sr)


def seek_read(path: str | Path, start_frame: int, num_frames: int) -> np.ndarray:
    """Read ``num_frames`` mono float32 samples starting at ``start_frame``.

    Uses soundfile's seek so DataLoader workers stream clips without loading the
    whole file (spec §6). Short reads (EOF) are zero-padded to ``num_frames``.
    """
    sf = _soundfile()
    with sf.SoundFile(str(path)) as f:
        f.seek(int(start_frame))
        data = f.read(int(num_frames), dtype="float32", always_2d=False)
    data = _to_mono(data)
    if len(data) < num_frames:
        data = np.pad(data, (0, num_frames - len(data)))
    return data.astype(np.float32, copy=False)


def write_flac(path: str | Path, signal: np.ndarray, sample_rate: int,
               subtype: str = "PCM_24") -> None:
    """Write mono float32 ``signal`` as 24-bit FLAC (spec §3.2.6, §4.2)."""
    _write(path, signal, sample_rate, fmt="FLAC", subtype=subtype)


def write_wav(path: str | Path, signal: np.ndarray, sample_rate: int,
              subtype: str = "PCM_24") -> None:
    """Write mono float32 ``signal`` as WAV (used for QA exports, spec §5.4)."""
    _write(path, signal, sample_rate, fmt="WAV", subtype=subtype)


def resample(signal: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality resample via torchaudio kaiser-windowed sinc (spec §3.2.2).

    No-op when ``sr_in == sr_out``. Returns mono float32.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if sr_in == sr_out:
        return signal
    try:
        import torch  # local heavy import
        import torchaudio.functional as AF
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise AudioIOError(_INSTALL_HINT) from exc
    x = torch.from_numpy(signal).unsqueeze(0)
    y = AF.resample(
        x, sr_in, sr_out,
        resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=64,
    )
    return y.squeeze(0).numpy().astype(np.float32)


def _to_mono(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.reshape(-1).astype(np.float32, copy=False)


def _write(path: str | Path, signal: np.ndarray, sample_rate: int, *,
           fmt: str, subtype: str) -> None:
    sf = _soundfile()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    # Atomic-ish: write to a temp sibling then replace, so a crash can't leave a
    # half-written render that the resume logic would mistake for complete.
    tmp = path.with_suffix(path.suffix + ".tmp")
    sf.write(str(tmp), x, int(sample_rate), format=fmt, subtype=subtype)
    tmp.replace(path)


# --------------------------------------------------------------------------
# Level / error metrics
# --------------------------------------------------------------------------

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
    """Error-to-signal ratio: ||est - tgt||^2 / ||tgt||^2."""
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


# --------------------------------------------------------------------------
# Validation probe signal
# --------------------------------------------------------------------------

SWEEP_SECONDS = 2.0
SILENCE_SECONDS = 1.0
DI_SECONDS = 2.0
SWEEP_DBFS = -12.0

# Probe-response validation thresholds.
PROBE_NOT_SILENT_MIN_DBFS = -80.0  # non-silent segments must exceed this
EXPLODE_MAX_DBFS = 6.0             # peak must stay below +6 dBFS
SILENCE_OUT_MAX_DBFS = -40.0       # silence in -> near-silence out


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
    """Deterministic pseudo guitar-DI: a few decaying plucked notes."""
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
    di = _synth_di(sample_rate, DI_SECONDS)
    signal = np.concatenate([sweep, silence, di]).astype(np.float32)
    a = len(sweep)
    b = a + len(silence)
    c = b + len(di)
    return Probe(signal=signal, sample_rate=sample_rate,
                 sweep=slice(0, a), silence=slice(a, b), di=slice(b, c))
