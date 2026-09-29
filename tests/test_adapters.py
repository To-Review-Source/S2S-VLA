import copy
import numpy as np
import torch
from s2s_vla.adapters.gymnasium import GymnasiumAdapter
from s2s_vla.adapters.torch_policy import FrozenTorchPolicy
from s2s_vla.interfaces import Context


class GymStub:
    def __init__(self):
        self.position = np.zeros(2)
        self.rng = np.random.default_rng()
        self.stopped = False

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros(2)
        self.stopped = False
        return self.position.copy(), {}

    def step(self, action):
        self.position += action + self.rng.normal(0, 0.1, 2)
        return self.position.copy(), 0.0, True, False, {"success": False, "harm": True}


def make_adapter():
    def snapshot(env):
        return copy.deepcopy((env.position, env.rng.bit_generator.state))
    def restore(env, state):
        env.position = state[0].copy()
        env.rng.bit_generator.state = copy.deepcopy(state[1])
    return GymnasiumAdapter(GymStub(), [0, 0], lambda env, obs, info: "test", lambda env, obs, info: obs, lambda env, obs, reward, done, trunc, info: info["success"], lambda env, obs, reward, done, trunc, info: info["harm"], lambda env, prefix: True, lambda env: setattr(env, "stopped", True), snapshot_fn=snapshot, restore_fn=restore, reseed_fn=lambda env, seed: setattr(env, "rng", np.random.default_rng(seed)))


def test_gym_termination_is_not_automatically_success():
    adapter = make_adapter()
    adapter.reset(2)
    transition = adapter.step(np.zeros(2))
    assert transition.terminated
    assert transition.anomaly
    assert not transition.success


def test_snapshot_restores_physics_rng_and_observation():
    adapter = make_adapter()
    adapter.reset(2)
    snapshot = adapter.snapshot()
    first = adapter.step(np.zeros(2)).observation
    adapter.restore(snapshot)
    np.testing.assert_array_equal(adapter.last_observation, np.zeros(2))
    adapter.reseed(9)
    np.testing.assert_array_equal(adapter.env.position, np.zeros(2))
    other = adapter.step(np.zeros(2)).observation
    assert not np.allclose(first, other)
    adapter.restore(snapshot)
    repeated = adapter.step(np.zeros(2)).observation
    np.testing.assert_array_equal(first, repeated)


def test_frozen_policy_reuses_features_and_respects_rng(method):
    calls = []
    model = torch.nn.Linear(3, method.feature_dim)
    def encode(model, context):
        calls.append("encode")
        features = model(torch.ones(3))
        return features, features
    def sample(model, context, cache, count, horizon, generator):
        assert cache.shape == (method.feature_dim,)
        return torch.randn(count, horizon, method.action_dim, generator=generator)
    policy = FrozenTorchPolicy(model, method, encode, sample)
    context = Context({}, "test", np.zeros(2), 4)
    first = policy.propose(context, 3, np.random.default_rng(4))
    second = policy.propose(context, 3, np.random.default_rng(4))
    assert calls == ["encode", "encode"]
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    np.testing.assert_array_equal(first.chunks, second.chunks)
