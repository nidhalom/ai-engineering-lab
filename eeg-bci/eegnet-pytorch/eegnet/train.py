"""Minimal training / evaluation helpers (CPU friendly)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .model import EEGNet, apply_max_norm


def _tensors(X, y):
    return torch.as_tensor(np.asarray(X), dtype=torch.float32), torch.as_tensor(np.asarray(y), dtype=torch.long)


def fit(
    model: EEGNet,
    X,
    y,
    epochs: int = 50,
    lr: float = 1e-2,
    batch_size: int = 32,
    seed: int = 0,
    verbose: bool = False,
) -> list[float]:
    """Train with Adam + cross-entropy and max-norm constraints. Returns the per-epoch mean loss."""
    torch.manual_seed(seed)
    Xt, yt = _tensors(X, y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:  # BatchNorm needs more than one sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            apply_max_norm(model)
            total += loss.item() * len(idx)
        history.append(total / len(Xt))
        if verbose:
            print(f"epoch {epoch + 1:3d}  loss {history[-1]:.4f}")
    return history


@torch.no_grad()
def predict(model: EEGNet, X, batch_size: int = 128) -> np.ndarray:
    model.eval()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    out = [model(Xt[i : i + batch_size]).argmax(1) for i in range(0, len(Xt), batch_size)]
    return torch.cat(out).numpy()


def evaluate(model: EEGNet, X, y) -> float:
    """Accuracy."""
    return float((predict(model, X) == np.asarray(y)).mean())
