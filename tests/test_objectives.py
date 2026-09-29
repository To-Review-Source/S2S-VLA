import numpy as np
import pytest
from s2s_vla.config import MethodConfig
from s2s_vla.objectives import augmented_tchebycheff, command_roughness, training_roughness_scale


def test_exact_minimum_is_nondominated():
    rng = np.random.default_rng(42)
    for _ in range(30):
        costs = rng.uniform(0, 2, size=(100, 3))
        weights = rng.dirichlet(np.ones(3))
        scores = augmented_tchebycheff(costs, weights, [1, 1, 0.6], 0.05)
        best = costs[np.argmin(scores)]
        dominates = np.all(costs <= best, axis=-1) & np.any(costs < best, axis=-1)
        assert not dominates.any()


def test_augmentation_breaks_dominated_max_tie():
    costs = np.array([[0.6, 0.1, 0.1], [0.6, 0.2, 0.1]])
    scores = augmented_tchebycheff(costs, [1 / 3] * 3, [1] * 3, 0.1)
    assert scores[0] < scores[1]


def test_roughness_has_boundary_and_excludes_discrete_command():
    method = MethodConfig(action_dim=2, feature_dim=2, continuous_indices=[0], roughness_operator=[[2]], action_scale=[1, 1])
    prefixes = np.array([[[1, 0], [1, 1]], [[0, 1], [0, 0]]], dtype=float)
    np.testing.assert_allclose(command_roughness(prefixes, [0, 0], method), [2, 0])


@pytest.mark.parametrize("weights", [[1, 0, 0], [-1, 1, 1], [1, 1, 1], [float("nan"), 0.5, 0.5]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        augmented_tchebycheff([[0, 0, 0]], weights, [1] * 3, 0.1)


def test_zero_roughness_training_scale_stays_positive():
    assert training_roughness_scale(np.zeros(5)) > 0
