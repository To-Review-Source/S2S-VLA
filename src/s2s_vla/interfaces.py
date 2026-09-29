from dataclasses import dataclass
from typing import Any, Protocol
import importlib
import numpy as np
from s2s_vla.config import ExperimentConfig, MethodConfig


@dataclass
class Context:
    observation: Any
    instruction: str
    previous_action: np.ndarray
    remaining: int


@dataclass
class Proposal:
    features: np.ndarray
    chunks: np.ndarray

    def validate(self, method: MethodConfig, count: int):
        self.features = np.asarray(self.features, dtype=np.float32)
        self.chunks = np.asarray(self.chunks, dtype=np.float32)
        if self.features.shape != (method.feature_dim,) or not np.isfinite(self.features).all():
            raise ValueError("Invalid frozen context features")
        if self.chunks.shape != (count, method.horizon, method.action_dim):
            raise ValueError("Policy returned an unexpected chunk shape")


@dataclass
class Transition:
    observation: Any
    success: bool = False
    anomaly: bool = False
    terminated: bool = False
    truncated: bool = False

    @property
    def done(self):
        return self.success or self.terminated or self.truncated


class Policy(Protocol):
    def propose(self, context: Context, count: int, rng: np.random.Generator) -> Proposal: ...


class Environment(Protocol):
    instruction: str
    initial_action: np.ndarray

    def reset(self, seed: int) -> Any: ...
    def step(self, action: np.ndarray) -> Transition: ...
    def valid_prefix(self, prefix: np.ndarray) -> bool: ...
    def stop(self) -> None: ...


class ResettableEnvironment(Environment, Protocol):
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def reseed(self, seed: int) -> None: ...


@dataclass
class Runtime:
    environment: Environment
    policy: Policy


def load_runtime(config: ExperimentConfig) -> Runtime:
    module, name = config.runtime_factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    runtime = factory(config.method, **config.runtime_kwargs)
    if not isinstance(runtime, Runtime):
        raise TypeError("Runtime factory must return s2s_vla.interfaces.Runtime")
    return runtime


def valid_indices(environment: Environment, prefixes: np.ndarray) -> np.ndarray:
    return np.asarray([i for i, prefix in enumerate(prefixes) if np.isfinite(prefix).all() and environment.valid_prefix(prefix)], dtype=np.int64)
