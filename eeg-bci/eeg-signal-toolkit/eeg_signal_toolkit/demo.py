"""Demo: band-power vs CSP features on synthetic data.  Run: python -m eeg_signal_toolkit.demo"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

from . import CSP, band_power, bandpass, make_synthetic_mi, notch, sliding_windows


def main() -> None:
    fs = 128.0
    X, y = make_synthetic_mi(n_trials=60, n_channels=8, fs=fs, noise=12.0, seed=1)
    print("SYNTHETIC data (sinusoids in mu/beta + noise) - not real EEG")
    print(f"trials={X.shape[0]} channels={X.shape[1]} samples={X.shape[2]} fs={fs:.0f} Hz")
    X = notch(X, 50.0, fs)
    X = bandpass(X, 8.0, 30.0, fs)
    w = sliding_windows(X[0], size=128, overlap=0.5)
    print(f"one trial -> {w.shape[0]} windows of {w.shape[2]} samples (50% overlap)")

    bp = band_power(X, fs, {"mu": (8, 13), "beta": (13, 30)})
    scores = {"band power + LDA": [], "CSP + LDA": []}
    # Splits are by trial, and CSP is fitted on the training fold only.
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        scores["band power + LDA"].append(LDA().fit(bp[tr], y[tr]).score(bp[te], y[te]))
        csp = CSP(4).fit(X[tr], y[tr])
        scores["CSP + LDA"].append(LDA().fit(csp.transform(X[tr]), y[tr]).score(csp.transform(X[te]), y[te]))
    for name, s in scores.items():
        print(f"{name:18s} 5-fold accuracy: {np.mean(s):.3f} +/- {np.std(s):.3f}")


if __name__ == "__main__":
    main()
