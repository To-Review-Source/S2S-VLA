import numpy as np
import torch
from torch.nn import functional as F
from s2s_vla.calibration import binary_nll, fit_temperature
from s2s_vla.losses import dual_objective_loss
from s2s_vla.model import DualHeadEvaluator


def test_soft_bce_equals_average_binary_bce():
    logit = torch.tensor(0.7)
    outcomes = torch.tensor([1.0, 0.0, 1.0, 1.0])
    soft = F.binary_cross_entropy_with_logits(logit, outcomes.mean())
    expanded = F.binary_cross_entropy_with_logits(logit.expand(4), outcomes)
    torch.testing.assert_close(soft, expanded)


def test_probability_and_preference_targets_are_compatible(method):
    targets = torch.tensor([[0.2, 0.1], [0.6, 0.4], [0.8, 0.2]])
    logits = torch.logit(targets).requires_grad_()
    terms = dual_objective_loss(logits, targets, torch.tensor([0.2, 0.1, 0.3]), [0, 3], method, [1, 1, 0.2])
    terms["loss"].backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-6, rtol=0)


def test_preference_comparisons_stay_inside_context(method):
    logits = torch.tensor([[0.4, -0.1], [0.7, 0.2], [-0.1, 0.3], [0.9, -0.2]])
    targets = torch.tensor([[0.3, 0.1], [0.6, 0.4], [0.8, 0.2], [0.2, 0.3]])
    roughness = torch.tensor([0.1, 0.2, 0.1, 0.3])
    together = dual_objective_loss(logits, targets, roughness, [0, 2, 4], method, [1, 1, 1])
    separate = [dual_objective_loss(logits[i:i + 2], targets[i:i + 2], roughness[i:i + 2], [0, 2], method, [1, 1, 1]) for i in (0, 2)]
    for key in together:
        torch.testing.assert_close(together[key], (separate[0][key] + separate[1][key]) / 2)


def test_padding_mask_and_frozen_features(method):
    model = DualHeadEvaluator(method)
    features = torch.randn(2, method.feature_dim, requires_grad=True)
    prefixes = torch.randn(2, method.prefix_length, method.action_dim)
    lengths = torch.ones(2, dtype=torch.long)
    previous = torch.zeros(2, method.action_dim)
    remaining = torch.ones(2, dtype=torch.long)
    first = model(features, prefixes, lengths, previous, remaining)
    changed = prefixes.clone()
    changed[:, 1:] = 1000
    second = model(features, changed, lengths, previous, remaining)
    torch.testing.assert_close(first, second)
    first.sum().backward()
    assert features.grad is None
    assert model.success_head.weight.grad is not None


def test_temperature_fitting_reduces_nll():
    logits = np.array([-8, -4, 4, 8], dtype=float)
    targets = np.array([0.3, 0.4, 0.6, 0.7])
    temperature = fit_temperature(logits, targets)
    assert temperature > 1
    assert binary_nll(logits / temperature, targets) < binary_nll(logits, targets)
