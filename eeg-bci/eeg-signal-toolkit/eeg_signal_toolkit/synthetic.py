"""Synthetic motor-imagery-like data (NOT real EEG).

Two latent oscillatory sources (mu ~10 Hz and beta ~22 Hz) are mixed into the channels with random
spatial patterns: source 0 loads mostly on the first half of the channels, source 1 on the second
half. In class k, source k is strong and the other is weak, so the class is encoded in the spatial
distribution of band power (like lateralised ERD/ERS). White noise is added on top.
"""

from __future__ import annotations

import numpy as np


def make_synthetic_mi(
    n_trials: int = 80,
    n_channels: int = 8,
    fs: float = 128.0,
    duration: float = 3.0,
    noise: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X (n_trials, n_channels, n_samples) and balanced labels y in {0, 1}."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    pat = rng.normal(size=(2, n_channels))
    pat[0, : n_channels // 2] += 2.0
    pat[1, n_channels // 2 :] += 2.0
    y = np.tile([0, 1], n_trials // 2 + 1)[:n_trials]
    X = np.empty((n_trials, n_channels, n))
    for i in range(n_trials):
        sources = np.zeros((2, n))
        for s in range(2):
            amp = 2.0 if y[i] == s else 0.6
            f_mu = rng.normal(10.0, 0.5)
            f_beta = rng.normal(22.0, 1.0)
            sources[s] = amp * (
                np.sin(2 * np.pi * f_mu * t + rng.uniform(0, 2 * np.pi))
                + 0.5 * np.sin(2 * np.pi * f_beta * t + rng.uniform(0, 2 * np.pi))
            )
        X[i] = pat.T @ sources + noise * rng.normal(size=(n_channels, n))
    return X, y
