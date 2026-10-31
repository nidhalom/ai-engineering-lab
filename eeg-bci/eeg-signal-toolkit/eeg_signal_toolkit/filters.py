"""Zero-phase IIR filters for EEG arrays shaped (..., n_samples)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass(x: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Butterworth band-pass, applied forward and backward (zero phase)."""
    nyq = fs / 2.0
    if not 0 < low < high < nyq:
        raise ValueError(f"need 0 < low < high < fs/2, got low={low}, high={high}, fs={fs}")
    sos = signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def notch(x: np.ndarray, freq: float, fs: float, q: float = 30.0) -> np.ndarray:
    """Remove mains interference (50 Hz in Algeria/Europe, 60 Hz in the US)."""
    if not 0 < freq < fs / 2:
        raise ValueError("notch frequency must be below the Nyquist frequency")
    b, a = signal.iirnotch(freq, q, fs=fs)
    return signal.filtfilt(b, a, x, axis=-1)
