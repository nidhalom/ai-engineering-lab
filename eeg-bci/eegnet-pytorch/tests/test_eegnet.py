import numpy as np
import pytest
import torch

from eegnet import EEGNet, apply_max_norm, evaluate, fit, group_train_test_split


@pytest.mark.parametrize("c,t,k", [(22, 128, 4), (3, 64, 2), (8, 250, 3)])
def test_output_shape(c, t, k):
    model = EEGNet(c, t, k)
    assert model(torch.randn(5, 1, c, t)).shape == (5, k)
    assert model(torch.randn(5, c, t)).shape == (5, k)  # 3-D input accepted


def test_parameter_count_matches_hand_calculation():
    # C=22, T=128, 4 classes, F1=8, D=2, F2=16, kernel 64:
    # temporal 8*64=512 | BN 16 | spatial 16*22=352 | BN 32 | sep-depthwise 16*16=256
    # | pointwise 16*16=256 | BN 32 | linear 16*(128//32)*4 + 4 = 260   -> 1716
    assert EEGNet(22, 128, 4).n_parameters == 1716


def test_model_is_compact():
    assert EEGNet(22, 640, 4).n_parameters < 5_000


def test_rejects_too_short_input():
    with pytest.raises(ValueError):
        EEGNet(4, 16, 2)


def test_max_norm_constraints():
    model = EEGNet(6, 64, 3)
    with torch.no_grad():
        model.block1[2].weight.mul_(50)
        model.classifier.weight.mul_(50)
    apply_max_norm(model)
    assert model.block1[2].weight.flatten(1).norm(dim=1).max() <= 1 + 1e-5
    assert model.classifier.weight.norm(dim=1).max() <= 0.25 + 1e-5


def test_overfits_tiny_batch():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(16, 4, 64, generator=g)
    y = torch.arange(16) % 2
    torch.manual_seed(0)
    model = EEGNet(4, 64, 2, kernel_length=16, dropout=0.0)
    history = fit(model, X, y, epochs=150, lr=1e-2, batch_size=16)
    assert history[-1] < 0.5 * history[0]
    assert evaluate(model, X, y) == 1.0


def test_group_split_has_no_overlap():
    groups = np.repeat(np.arange(10), 7)
    tr, te = group_train_test_split(groups, 0.3, seed=1)
    assert not set(groups[tr]) & set(groups[te])
    assert len(tr) + len(te) == len(groups)
    assert len(np.unique(groups[te])) == 3


def test_group_split_needs_two_groups():
    with pytest.raises(ValueError):
        group_train_test_split([1, 1, 1])
