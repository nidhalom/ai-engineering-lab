# eeg-signal-toolkit

A small, dependency-light toolkit for the first steps of an EEG / brain-computer-interface pipeline: zero-phase band-pass and notch filters, sliding windows with overlap, Welch band-power features, and a two-class **Common Spatial Patterns (CSP)** implementation written from scratch on top of numpy/scipy (generalized eigenproblem). It grew out of my motor-imagery work for my Master's in AI at USTHB, where CSP + classical classifiers were one of the baselines next to EEGNet.

## Features

- `bandpass` (Butterworth, `sosfiltfilt`) and `notch` (IIR notch) filters for arrays shaped `(..., n_samples)`
- `sliding_windows` with `step` or `overlap`, returning `(n_windows, n_channels, size)`
- `band_power`: log band power per channel from the Welch PSD (delta/theta/mu/beta by default, custom bands supported)
- `CSP`: scikit-learn style `fit` / `transform`, solves `C1 w = lambda (C1 + C2) w` with `scipy.linalg.eigh`, exposes `filters_`, `patterns_` and `eigenvalues_`
- `make_synthetic_mi`: synthetic motor-imagery-like data generator (see below)
- Optional `load_mne_sample()` if `mne` is installed

> **The demo and the tests run on SYNTHETIC signals** (sinusoids in the mu and beta bands mixed through random spatial patterns, plus white noise). They check that the code is correct; the accuracies printed below say nothing about real EEG performance.

## Quick start

```bash
pip install -e ".[test]"
pytest -q
python -m eeg_signal_toolkit.demo
```

```python
from eeg_signal_toolkit import CSP, bandpass, band_power, make_synthetic_mi, sliding_windows

X, y = make_synthetic_mi(n_trials=60, fs=128)       # (60, 8, 384)
X = bandpass(X, 8, 30, fs=128)
features = CSP(n_components=4).fit(X, y).transform(X)   # (60, 4) log-variance features
```

## Example output

Real output of `python -m eeg_signal_toolkit.demo` (60 synthetic trials, 8 channels, heavy noise, LDA from scikit-learn, 5-fold stratified CV by trial, CSP fitted on the training fold only):

```
SYNTHETIC data (sinusoids in mu/beta + noise) - not real EEG
trials=60 channels=8 samples=384 fs=128 Hz
one trial -> 5 windows of 128 samples (50% overlap)
band power + LDA   5-fold accuracy: 0.933 +/- 0.062
CSP + LDA          5-fold accuracy: 1.000 +/- 0.000
```

## Project structure

```
eeg_signal_toolkit/
  filters.py      bandpass, notch
  windows.py      sliding_windows
  features.py     band_power
  csp.py          CSP
  synthetic.py    make_synthetic_mi
  mne_loader.py   optional MNE sample loader
  demo.py         synthetic demo (needs scikit-learn)
tests/test_toolkit.py
```

## How it works

CSP estimates the trace-normalised covariance matrix of each class, then solves the generalized eigenproblem `C1 w = lambda (C1 + C2) w`. Filters at the two ends of the eigenvalue spectrum maximise the variance ratio between the classes; log-variance of the filtered trial is the feature. The tests verify that the returned filters actually satisfy the eigen-equation, not just that accuracy is high.

## Limitations

- Two-class CSP only; no multi-class (one-vs-rest) extension, no regularised/Riemannian variants.
- Synthetic data is far easier than real motor imagery (noise, artefacts, non-stationarity, subject variability).
- When windowing a recording, keep all windows of a trial/session on the same side of a train/test split; overlapping windows are near-duplicates and leak otherwise. CSP must also be fitted on training data only.
- `load_mne_sample` loads MNE's auditory sample data (not motor imagery) and was not exercised in CI.

## License

MIT, see [LICENSE](LICENSE).
