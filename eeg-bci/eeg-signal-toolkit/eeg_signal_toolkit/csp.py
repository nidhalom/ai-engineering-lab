"""Common Spatial Patterns for two classes, via a generalized eigenproblem."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def _mean_cov(trials: np.ndarray) -> np.ndarray:
    covs = []
    for t in trials:
        c = t @ t.T
        covs.append(c / np.trace(c))
    return np.mean(covs, axis=0)


class CSP:
    """Two-class CSP with an sklearn-style fit/transform API.

    Solves  C1 w = lambda (C1 + C2) w.  Eigenvalues near 1 give filters that maximise the variance
    of class 1 relative to class 2, eigenvalues near 0 the opposite. ``n_components`` filters are
    kept, half from each end of the spectrum. ``transform`` returns normalised log-variance.
    """

    def __init__(self, n_components: int = 4, reg: float = 1e-6):
        if n_components % 2:
            raise ValueError("n_components must be even")
        self.n_components = n_components
        self.reg = reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP":
        X, y = np.asarray(X, float), np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP needs exactly two classes")
        if X.ndim != 3:
            raise ValueError("X must have shape (n_trials, n_channels, n_samples)")
        self.classes_ = classes
        c1 = _mean_cov(X[y == classes[0]])
        c2 = _mean_cov(X[y == classes[1]])
        n_ch = c1.shape[0]
        if self.n_components > n_ch:
            raise ValueError("n_components cannot exceed the number of channels")
        eigvals, eigvecs = linalg.eigh(c1, c1 + c2 + self.reg * np.eye(n_ch))  # ascending
        k = self.n_components // 2
        order = np.r_[np.arange(k), np.arange(n_ch - k, n_ch)]
        self.eigenvalues_ = eigvals
        self.filters_ = eigvecs[:, order].T  # (n_components, n_channels)
        # spatial patterns: columns of the inverse of the full filter matrix
        self.patterns_ = linalg.pinv(eigvecs.T)[:, order].T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        proj = np.einsum("fc,nct->nft", self.filters_, X)
        var = proj.var(axis=-1)
        return np.log(var / var.sum(axis=1, keepdims=True))

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
