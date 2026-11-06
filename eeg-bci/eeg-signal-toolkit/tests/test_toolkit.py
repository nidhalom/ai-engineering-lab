import numpy as np
import pytest

from eeg_signal_toolkit import CSP, band_power, bandpass, make_synthetic_mi, notch, sliding_windows

FS = 128.0


def tone(f, n=1024):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * f * t)


def rms(x):
    return np.sqrt(np.mean(x[100:-100] ** 2))


def test_bandpass_keeps_inband_removes_outband():
    y = bandpass(tone(10) + tone(40), 8, 13, FS)
    assert rms(y) == pytest.approx(rms(tone(10)), rel=0.1)
    assert rms(y - tone(10)) < 0.1


def test_bandpass_validates_edges():
    with pytest.raises(ValueError):
        bandpass(tone(10), 10, 70, FS)


def test_notch_removes_mains():
    y = notch(tone(10) + tone(50), 50, FS)
    assert rms(y - tone(10)) < 0.1


def test_filters_work_on_multichannel():
    x = np.stack([tone(10), tone(12)])
    assert bandpass(x, 8, 13, FS).shape == x.shape


def test_sliding_windows_shape_and_content():
    x = np.arange(2 * 100).reshape(2, 100)
    w = sliding_windows(x, size=40, overlap=0.5)
    assert w.shape == (4, 2, 40)  # starts 0, 20, 40, 60
    assert np.array_equal(w[1], x[:, 20:60])


def test_sliding_windows_short_and_bad_args():
    assert sliding_windows(np.zeros((2, 10)), 20, step=5).shape == (0, 2, 20)
    with pytest.raises(ValueError):
        sliding_windows(np.zeros((2, 100)), 10)
    with pytest.raises(ValueError):
        sliding_windows(np.zeros((2, 100)), 10, step=5, overlap=0.5)


def test_band_power_peaks_in_right_band():
    x = np.stack([tone(10, 512), tone(22, 512)])
    p = band_power(x, FS, {"mu": (8, 13), "beta": (13, 30)}, log=False)
    assert p.shape == (4,)  # ch0-mu, ch0-beta, ch1-mu, ch1-beta
    assert p[0] > 10 * p[1]
    assert p[3] > 10 * p[2]


def test_band_power_batch_shape():
    X, _ = make_synthetic_mi(n_trials=6, n_channels=4)
    assert band_power(X, FS).shape == (6, 4 * 4)


def test_csp_separates_synthetic_classes():
    X, y = make_synthetic_mi(n_trials=60, n_channels=8, noise=1.0, seed=3)
    csp = CSP(4).fit(X[:40], y[:40])
    ftr, fte = csp.transform(X[:40]), csp.transform(X[40:])
    assert ftr.shape == (40, 4)
    d = lambda f: f[:, 0] - f[:, -1]
    thr = (d(ftr)[y[:40] == 0].mean() + d(ftr)[y[:40] == 1].mean()) / 2
    sign = 1 if d(ftr)[y[:40] == 1].mean() > thr else -1
    pred = ((d(fte) - thr) * sign > 0).astype(int)
    assert (pred == y[40:]).mean() > 0.9


def test_csp_filters_solve_generalized_eigenproblem():
    X, y = make_synthetic_mi(n_trials=40, seed=5)
    csp = CSP(2, reg=0).fit(X, y)

    def cov(k):
        return np.mean([t @ t.T / np.trace(t @ t.T) for t in X[y == k]], axis=0)

    w, lam = csp.filters_[0], csp.eigenvalues_[0]
    assert np.allclose(cov(0) @ w, lam * (cov(0) + cov(1)) @ w, atol=1e-8)


def test_csp_requires_two_classes():
    with pytest.raises(ValueError):
        CSP(2).fit(np.ones((3, 4, 50)), np.array([0, 1, 2]))


def test_synthetic_deterministic_and_balanced():
    a, ya = make_synthetic_mi(seed=7)
    b, _ = make_synthetic_mi(seed=7)
    assert np.array_equal(a, b) and ya.sum() == len(ya) // 2
