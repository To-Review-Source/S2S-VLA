import pytest
import torch
from s2s_vla.config import ExperimentConfig, MethodConfig


@pytest.fixture
def method():
    return MethodConfig(action_dim=2, feature_dim=7, horizon=4, prefix_length=2, candidates=4, max_steps=12, continuous_indices=[0, 1], roughness_operator=[[1, 0], [0, 1]], action_scale=[0.22, 0.22], hidden_dim=16, preferences=[[0.8, 0.1, 0.1], [0.4, 0.4, 0.2]])


@pytest.fixture
def config(method):
    return ExperimentConfig(method=method, runtime_factory="s2s_vla.adapters.example:create_runtime", seed=19, episodes=12, branches=3, context_stride=2, epochs=2, batch_contexts=4)


@pytest.fixture(scope="session", autouse=True)
def torch_threads():
    torch.set_num_threads(1)
