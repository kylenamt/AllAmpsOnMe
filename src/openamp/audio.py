"""Audio I/O + resampling adapters (spec §3.2).

The one place that depends on ``soundfile`` and ``torchaudio``. Every import is
local so the rest of the package (and its unit tests) load on a bare numpy stack.
If these optional deps are missing, a clear install hint is raised — the same
pattern Phase 1 uses for its NAM/torch backend (``t3k.nam_backend``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_INSTALL_HINT = (
    "Phase 2 audio I/O needs soundfile + torchaudio. Install the render extras: "
    "`pip install -e .[phase2]` (or `pip install -r requirements-phase2.txt`)."
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
