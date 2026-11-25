"""Window-level vs group-level splits: how overlapping windows leak.

Every "trial" below is a recording with its own random oscillatory signature and a RANDOM label,
so there is nothing to learn that generalises to a new trial. Cut into overlapping windows and
split at the window level, the network still scores high because it recognises the trial it has
already seen. Split by trial and the accuracy falls to chance.

Run:  python examples/leakage_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegnet import EEGNet, evaluate, fit, group_train_test_split  # noqa: E402

FS, N_CH, WIN, STEP = 64, 8, 128, 16  # 87.5 % overlap


def make_trials(n_trials: int, seed: int):
    rng = np.random.default_rng(seed)
    t = np.arange(FS * 6) / FS
    X, y, groups = [], [], []
    for trial in range(n_trials):
        label = trial % 2  # random with respect to the signal below
        freqs = rng.uniform(4, 30, size=(N_CH, 2))
        phase = rng.uniform(0, 2 * np.pi, size=(N_CH, 2))
        rec = sum(np.sin(2 * np.pi * freqs[:, [k]] * t + phase[:, [k]]) for k in range(2))
        rec = rec + 0.3 * rng.normal(size=rec.shape)
        for s in range(0, rec.shape[1] - WIN + 1, STEP):
            X.append(rec[:, s : s + WIN])
            y.append(label)
            groups.append(trial)
    return np.array(X, dtype=np.float32), np.array(y), np.array(groups)


def run(train_idx, test_idx, X, y) -> float:
    torch.manual_seed(0)
    model = EEGNet(N_CH, WIN, 2, kernel_length=32, dropout=0.25)
    fit(model, X[train_idx], y[train_idx], epochs=25, lr=5e-3)
    return evaluate(model, X[test_idx], y[test_idx])


def main() -> None:
    X, y, groups = make_trials(n_trials=24, seed=0)
    print(f"{len(X)} windows from {len(np.unique(groups))} trials, labels carry no real signal (chance = 0.50)")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    cut = int(0.75 * len(X))
    acc_window = run(perm[:cut], perm[cut:], X, y)
    tr, te = group_train_test_split(groups, 0.25, seed=0)
    assert not set(groups[tr]) & set(groups[te])
    acc_group = run(tr, te, X, y)
    print(f"random window split : test accuracy {acc_window:.3f}  (leaky)")
    print(f"split by trial      : test accuracy {acc_group:.3f}  (honest)")


if __name__ == "__main__":
    main()
