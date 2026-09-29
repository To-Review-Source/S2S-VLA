from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import numpy as np


@dataclass
class MethodConfig:
    action_dim: int = 7
    feature_dim: int = 4096
    horizon: int = 16
    prefix_length: int = 4
    candidates: int = 8
    max_steps: int = 300
    continuous_indices: list[int] = field(default_factory=lambda: list(range(6)))
    roughness_operator: list[list[float]] = field(default_factory=lambda: np.eye(6).tolist())
    action_scale: list[float] = field(default_factory=lambda: [1.0] * 7)
    preferences: list[list[float]] = field(default_factory=lambda: [[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.6, 0.1, 0.3], [1 / 3] * 3])
    rho: float = 0.05
    pair_temperature: float = 0.1
    preference_loss_weight: float = 0.2
    hidden_dim: int = 128

    def __post_init__(self):
        for key in ("action_dim", "feature_dim", "horizon", "prefix_length", "candidates", "max_steps", "hidden_dim"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.candidates < 2 or self.prefix_length > self.horizon:
            raise ValueError("Require candidates >= 2 and prefix_length <= horizon")
        indices = self.continuous_indices
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= self.action_dim:
            raise ValueError("Invalid continuous_indices")
        operator = np.asarray(self.roughness_operator, dtype=float)
        if operator.ndim != 2 or operator.shape[1] != len(indices) or operator.shape[0] < 1 or not np.isfinite(operator).all():
            raise ValueError("roughness_operator must be finite with one column per continuous coordinate")
        scales = np.asarray(self.action_scale, dtype=float)
        if scales.shape != (self.action_dim,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("action_scale must contain positive finite values")
        weights = np.asarray(self.preferences, dtype=float)
        if weights.ndim != 2 or weights.shape[0] < 1 or weights.shape[1] != 3 or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("preferences must be a nonempty array of strictly positive triples")
        if not np.allclose(weights.sum(-1), 1.0):
            raise ValueError("Each preference must sum to one")
        values = [self.rho, self.pair_temperature, self.preference_loss_weight]
        if not np.isfinite(values).all() or self.rho <= 0 or self.pair_temperature <= 0 or self.preference_loss_weight < 0:
            raise ValueError("Invalid scalarization or training parameters")

    def to_dict(self):
        return asdict(self)


@dataclass
class ExperimentConfig:
    method: MethodConfig
    runtime_factory: str
    runtime_kwargs: dict = field(default_factory=dict)
    seed: int = 42
    episodes: int = 64
    branches: int = 4
    context_stride: int = 3
    split_fractions: list[float] = field(default_factory=lambda: [0.6, 0.2, 0.1, 0.1])
    epochs: int = 10
    batch_contexts: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    def __post_init__(self):
        if self.episodes < 4 or min(self.branches, self.context_stride, self.epochs, self.batch_contexts) < 1:
            raise ValueError("Invalid collection or training sizes")
        fractions = np.asarray(self.split_fractions)
        if fractions.shape != (4,) or not np.isfinite(fractions).all() or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
            raise ValueError("Four positive split fractions summing to one are required")
        if not np.isfinite([self.learning_rate, self.weight_decay]).all() or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        values["method"] = MethodConfig(**values["method"])
        return cls(**values)
