import torch
from torch import nn
from s2s_vla.config import MethodConfig


class DualHeadEvaluator(nn.Module):
    def __init__(self, method: MethodConfig):
        super().__init__()
        self.method = method
        hidden = method.hidden_dim
        width = method.prefix_length * (method.action_dim + 1)
        self.action_encoder = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.context_projection = nn.Sequential(nn.Linear(method.feature_dim, hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(2 * hidden + method.action_dim + 1, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.success_head = nn.Linear(hidden, 1)
        self.anomaly_head = nn.Linear(hidden, 1)
        self.register_buffer("feature_mean", torch.zeros(method.feature_dim))
        self.register_buffer("feature_std", torch.ones(method.feature_dim))
        self.register_buffer("action_scale", torch.tensor(method.action_scale, dtype=torch.float32))

    def forward(self, features, prefixes, lengths, previous_actions, remaining):
        batch = prefixes.shape[0]
        if prefixes.shape != (batch, self.method.prefix_length, self.method.action_dim):
            raise ValueError("Unexpected padded prefix shape")
        if lengths.shape != (batch,) or bool(((lengths < 1) | (lengths > self.method.prefix_length)).any()):
            raise ValueError("Invalid prefix lengths")
        if bool((lengths > remaining).any()) or bool((remaining < 1).any()):
            raise ValueError("Prefix exceeds remaining budget")
        mask = torch.arange(self.method.prefix_length, device=prefixes.device)[None] < lengths[:, None]
        commands = prefixes / self.action_scale
        commands = torch.where(mask[..., None], commands, torch.zeros_like(commands))
        encoded_actions = self.action_encoder(torch.cat([commands.flatten(1), mask.to(commands.dtype)], dim=-1))
        encoded_context = self.context_projection((features.detach() - self.feature_mean) / self.feature_std)
        budget = remaining.to(features.dtype).unsqueeze(-1) / self.method.max_steps
        hidden = self.fusion(torch.cat([encoded_context, encoded_actions, previous_actions / self.action_scale, budget], dim=-1))
        return torch.cat([self.success_head(hidden), self.anomaly_head(hidden)], dim=-1)
