"""Welch band-power features."""

from __future__ import annotations

import numpy as np
from scipy import signal

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def band_power(
    x: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    log: bool = True,
    nperseg: int | None = None,
) -> np.ndarray:
    """Band power per channel and band.

    ``x`` has shape (..., n_channels, n_samples). Returns (..., n_channels * n_bands), ordered
    channel-major then band, in the order of ``bands``. Power is the integral of the Welch PSD
    over the band (trapezoid rule); ``log=True`` returns log10 power.
    """
    bands = bands or DEFAULT_BANDS
    nperseg = nperseg or min(x.shape[-1], int(fs * 2))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, axis=-1)
    out = []
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() < 2:
            raise ValueError(f"fewer than 2 PSD bins in band {lo}-{hi} Hz; use a longer window")
        out.append(np.trapezoid(psd[..., mask], freqs[mask], axis=-1))
    power = np.stack(out, axis=-1)  # (..., ch, bands)
    if log:
        power = np.log10(power + 1e-20)
    return power.reshape(*power.shape[:-2], -1)
