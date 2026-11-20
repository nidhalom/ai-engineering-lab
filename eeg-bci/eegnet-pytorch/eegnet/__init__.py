"""EEGNet in PyTorch."""

from .model import EEGNet, apply_max_norm
from .splits import group_train_test_split
from .train import evaluate, fit

__all__ = ["EEGNet", "apply_max_norm", "fit", "evaluate", "group_train_test_split"]
__version__ = "0.1.0"
