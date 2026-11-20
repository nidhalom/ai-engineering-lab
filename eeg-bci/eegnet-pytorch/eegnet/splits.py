"""Group-wise splitting: keep every window of a trial / session / subject on one side."""

from __future__ import annotations

import numpy as np


def group_train_test_split(groups, test_fraction: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) so that no group appears in both.

    ``groups`` assigns each window to its origin (trial id, session id or subject id). Use subject
    ids for cross-subject evaluation, session ids for cross-session evaluation.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("need at least two groups")
    rng = np.random.default_rng(seed)
    n_test = min(len(unique) - 1, max(1, round(len(unique) * test_fraction)))
    test_groups = rng.choice(unique, size=n_test, replace=False)
    is_test = np.isin(groups, test_groups)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)
