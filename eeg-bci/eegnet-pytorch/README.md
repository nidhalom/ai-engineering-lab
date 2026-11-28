# eegnet-pytorch

A clean, readable PyTorch reimplementation of **EEGNet** (Lawhern et al., 2018), the compact convolutional network for EEG-based brain-computer interfaces. The number of channels, samples and classes is configurable, the paper's max-norm constraints are included, and a small CPU-friendly training loop plus group-wise splitting helpers come with it. I used EEGNet as one of the models in my Master's project (USTHB) on motor-imagery EEG, and the biggest lesson was about evaluation, not architecture: see "Leakage" below.

> Reference: Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). *EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.* Journal of Neural Engineering, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c

## Features

- `EEGNet(n_channels, n_samples, n_classes, f1=8, depth=2, f2=None, kernel_length=64, dropout=0.5)`; accepts `(B, C, T)` or `(B, 1, C, T)` input and returns logits
- `apply_max_norm(model)`: spatial-filter norm <= 1 and classifier row norm <= 0.25, as in the paper
- `fit` / `evaluate`: Adam + cross-entropy loop, CPU friendly
- `group_train_test_split(groups)`: split by trial / session / subject so no group appears on both sides
- `examples/leakage_demo.py`: shows window-level leakage on synthetic data
- Tests (CPU only, about a minute): output shapes, an exact parameter-count check, max-norm, and overfitting of a tiny batch

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
pytest -q
python examples/leakage_demo.py
```

```python
import torch
from eegnet import EEGNet, fit, evaluate

model = EEGNet(n_channels=22, n_samples=128, n_classes=4)   # 1,716 parameters
fit(model, X_train, y_train, epochs=100)                     # X: (N, 22, 128) float32
print(evaluate(model, X_test, y_test))
```

## Leakage: why window-wise splits lie

EEG recordings are usually cut into short, heavily overlapping windows (for example 2 s windows every 0.25 s). Neighbouring windows are almost the same signal. If you shuffle windows and split them randomly, windows of the same trial end up in both train and test, and a network can reach a high test score by recognising the *recording* (its noise, drift, electrode impedance, subject-specific rhythm) instead of the mental task. The same applies to sessions and subjects: the right split depends on the claim you want to make (new trial, new session, new subject). In my thesis work, scores dropped substantially once splits were done by subject/session instead of by window.

`examples/leakage_demo.py` makes this concrete. It builds 24 synthetic trials, each with its own random oscillatory signature and a label that has **no relation to the signal**, cuts them into 87.5 %-overlapping windows, and trains EEGNet two ways. Real output:

```
408 windows from 24 trials, labels carry no real signal (chance = 0.50)
random window split : test accuracy 1.000  (leaky)
split by trial      : test accuracy 0.667  (honest)
```

The random-window split scores a perfect 1.000 on data with nothing to learn. The trial-wise split is the honest number; with only 6 test trials it is very noisy (0.667 is 4 of 6 trials' worth of luck above 0.5, not skill), so treat it as "around chance". Use `group_train_test_split(groups)` and pass trial/session/subject ids.

## Project structure

```
eegnet/
  model.py    EEGNet, apply_max_norm
  train.py    fit, predict, evaluate
  splits.py   group_train_test_split
examples/leakage_demo.py
tests/test_eegnet.py
```

## How it works

Block 1 learns temporal filters (1 x 64 convolution), then a depthwise convolution over all channels learns `depth` spatial filters per temporal filter; batch norm, ELU, average pooling (4) and dropout follow. Block 2 is a separable convolution (depthwise 1 x 16 + pointwise 1 x 1), ELU, average pooling (8) and dropout. A linear layer produces the logits. With 22 channels, 128 samples and 4 classes the model has 1,716 parameters; the test suite checks that number against a hand calculation.

## Limitations

- Not benchmarked on BCI Competition IV 2a or any real dataset here; the demo uses synthetic data only. No claim about accuracy on real EEG.
- Implements the 2018 "EEGNet-8,2"-style architecture (v2 layout with padding "same"); other variants (EEGNet-fusion, different pooling) are not included.
- The training loop is deliberately minimal: no scheduler, early stopping or validation split.

## License

MIT, see [LICENSE](LICENSE).
