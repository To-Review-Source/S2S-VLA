import torch
from torch.nn import functional as F
from s2s_vla.config import MethodConfig


def torch_scores(costs, weights, scales, rho):
    deviations = costs[:, None, :] / scales * weights[None, :, :]
    return deviations.amax(-1) + rho * deviations.sum(-1)


def dual_objective_loss(logits, targets, roughness, offsets, method: MethodConfig, scales):
    targets = targets.detach()
    roughness = roughness.detach()
    weights = logits.new_tensor(method.preferences)
    scales = logits.new_tensor(scales).detach()
    probabilities = logits.sigmoid()
    predicted_costs = torch.stack([1 - probabilities[:, 0], probabilities[:, 1], roughness], dim=-1)
    empirical_costs = torch.stack([1 - targets[:, 0], targets[:, 1], roughness], dim=-1)
    predicted_scores = torch_scores(predicted_costs, weights, scales, method.rho)
    target_scores = torch_scores(empirical_costs, weights, scales, method.rho).detach()
    event_losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(-1)
    probability_terms = []
    preference_terms = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        probability_terms.append(event_losses[start:end].mean())
        if end - start < 2:
            raise ValueError("Training contexts require at least two candidates")
        i, j = torch.triu_indices(end - start, end - start, offset=1, device=logits.device)
        i, j = i + start, j + start
        predicted_pair_logits = (predicted_scores[j] - predicted_scores[i]) / method.pair_temperature
        target_pairs = ((target_scores[j] - target_scores[i]) / method.pair_temperature).sigmoid()
        preference_terms.append(F.binary_cross_entropy_with_logits(predicted_pair_logits, target_pairs))
    probability_loss = torch.stack(probability_terms).mean()
    preference_loss = torch.stack(preference_terms).mean()
    loss = probability_loss + method.preference_loss_weight * preference_loss
    return {"loss": loss, "probability_loss": probability_loss, "preference_loss": preference_loss}
