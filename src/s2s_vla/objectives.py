import numpy as np
from s2s_vla.config import MethodConfig


def command_roughness(prefixes, previous_action, method: MethodConfig):
    prefixes = np.asarray(prefixes, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    if prefixes.ndim != 3 or prefixes.shape[1] < 1 or prefixes.shape[2] != method.action_dim:
        raise ValueError("prefixes must have shape [K, h, action_dim]")
    if previous_action.shape != (method.action_dim,):
        raise ValueError("Invalid preceding command")
    if not np.isfinite(prefixes).all() or not np.isfinite(previous_action).all():
        raise ValueError("Commands must be finite")
    predecessors = np.concatenate([np.broadcast_to(previous_action, (len(prefixes), 1, method.action_dim)), prefixes[:, :-1]], axis=1)
    differences = (prefixes - predecessors)[..., method.continuous_indices]
    transformed = differences @ np.asarray(method.roughness_operator, dtype=float).T
    return np.square(transformed).sum(axis=-1).mean(axis=-1)


def augmented_tchebycheff(costs, weights, scales, rho):
    costs = np.asarray(costs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if costs.ndim < 1 or costs.shape[-1] != 3 or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("Costs must be finite nonnegative triples")
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must be positive and sum to one")
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(rho) or rho <= 0:
        raise ValueError("Scales and rho must be finite and strictly positive")
    deviations = costs / scales * weights
    return deviations.max(axis=-1) + rho * deviations.sum(axis=-1)


def cost_vectors(probabilities, roughness):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    roughness = np.asarray(roughness, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2 or roughness.shape != (len(probabilities),):
        raise ValueError("Invalid objective array shapes")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    return np.column_stack((1 - probabilities[:, 0], probabilities[:, 1], roughness))


def training_roughness_scale(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid training roughness")
    return float(max(np.quantile(values, 0.95), 1e-8))
