"""Sliding-window segmentation."""

from __future__ import annotations

import numpy as np


def sliding_windows(
    x: np.ndarray, size: int, step: int | None = None, overlap: float | None = None
) -> np.ndarray:
    """Cut (n_channels, n_samples) into windows (n_windows, n_channels, size).

    Give either ``step`` (samples between window starts) or ``overlap`` (fraction in [0, 1)).
    Trailing samples that do not fill a whole window are dropped.

    Careful: overlapping windows from one recording are highly correlated. Never let windows of
    the same trial/session land on both sides of a train/test split.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must have shape (n_channels, n_samples)")
    if (step is None) == (overlap is None):
        raise ValueError("pass exactly one of step / overlap")
    if step is None:
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        step = max(1, int(round(size * (1 - overlap))))
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")
    n = x.shape[1]
    if n < size:
        return np.empty((0, x.shape[0], size), dtype=x.dtype)
    starts = np.arange(0, n - size + 1, step)
    return np.stack([x[:, s : s + size] for s in starts])
