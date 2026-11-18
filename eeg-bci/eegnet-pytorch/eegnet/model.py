"""EEGNet-v2 (Lawhern et al., 2018).

Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
Journal of Neural Engineering, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c

Input  (batch, 1, n_channels, n_samples)  or  (batch, n_channels, n_samples).

Block 1: temporal conv (1 x kernel_length) -> BN -> depthwise spatial conv (n_channels x 1,
         depth multiplier D) -> BN -> ELU -> AvgPool(1x4) -> Dropout
Block 2: separable conv (depthwise 1x16 + pointwise 1x1) -> BN -> ELU -> AvgPool(1x8) -> Dropout
Output : Linear on the flattened features (returns logits).
"""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        f1: int = 8,
        depth: int = 2,
        f2: int | None = None,
        kernel_length: int = 64,
        separable_kernel: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        f2 = f2 or f1 * depth
        if n_samples < 32:
            raise ValueError("n_samples must be at least 32 (two pooling stages: 4 * 8)")
        self.n_channels, self.n_samples, self.n_classes = n_channels, n_samples, n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, (n_channels, 1), groups=f1, bias=False),  # depthwise spatial filters
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * depth, f1 * depth, (1, separable_kernel), padding="same", groups=f1 * depth, bias=False),
            nn.Conv2d(f1 * depth, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(f2 * (n_samples // 32), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.block2(self.block1(x))
        return self.classifier(x.flatten(1))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def apply_max_norm(model: EEGNet) -> None:
    """Max-norm constraints from the paper: spatial depthwise filters <= 1, classifier rows <= 0.25.

    Call after every optimiser step.
    """
    spatial = model.block1[2].weight  # (f1*D, 1, C, 1)
    norms = spatial.flatten(1).norm(dim=1).clamp(min=1e-8)
    spatial.mul_((1.0 / norms).clamp(max=1.0).view(-1, 1, 1, 1))
    w = model.classifier.weight  # (classes, features)
    norms = w.norm(dim=1).clamp(min=1e-8)
    w.mul_((0.25 / norms).clamp(max=1.0).view(-1, 1))
