import numpy as np
from s2s_vla.collection import branch_outcome
from s2s_vla.controller import run_episode
from s2s_vla.inference import Selector
from s2s_vla.interfaces import Proposal, Runtime, Transition, Context


class ScriptedEnvironment:
    instruction = "complete"
    initial_action = np.zeros(2, dtype=np.float32)

    def __init__(self, success_step=3, invalid=False):
        self.success_step = success_step
        self.invalid = invalid
        self.elapsed = 0
        self.stopped = False

    def reset(self, seed):
        self.elapsed = 0
        self.stopped = False
        return np.zeros(7)

    def valid_prefix(self, prefix):
        return not self.invalid

    def step(self, action):
        self.elapsed += 1
        success = self.elapsed >= self.success_step
        return Transition(np.zeros(7), success=success, anomaly=self.elapsed == 2, terminated=success)

    def stop(self):
        self.stopped = True

    def restore(self, snapshot):
        self.elapsed = snapshot

    def reseed(self, seed):
        pass

    def snapshot(self):
        raise AssertionError("Online controller must never request a simulator snapshot")


class ConstantPolicy:
    def __init__(self, method):
        self.method = method

    def propose(self, context, count, rng):
        return Proposal(np.zeros(self.method.feature_dim), np.zeros((count, self.method.horizon, self.method.action_dim)))


def test_prefix_anomaly_does_not_include_continuation(method):
    environment = ScriptedEnvironment()
    policy = ConstantPolicy(method)
    context = Context(np.zeros(7), "complete", np.zeros(2), 4)
    success, anomaly = branch_outcome(environment, policy, 0, context, np.zeros((1, 2)), method, 42)
    assert success == 1
    assert anomaly == 0
    success, anomaly = branch_outcome(environment, policy, 0, context, np.zeros((2, 2)), method, 42)
    assert success == 1
    assert anomaly == 1


def test_controller_respects_budget_without_online_rollouts(method):
    method.max_steps = 5
    environment = ScriptedEnvironment(success_step=100)
    result = run_episode(Runtime(environment, ConstantPolicy(method)), method, Selector(method, strategy="first"), 1, 2)
    assert result["steps"] == 5
    assert [x["executed_steps"] for x in result["decisions"]] == [2, 2, 1]
    assert not result["success"]


def test_terminal_success_interrupts_prefix(method):
    environment = ScriptedEnvironment(success_step=1)
    result = run_episode(Runtime(environment, ConstantPolicy(method)), method, Selector(method, strategy="first"), 1, 2)
    assert result["steps"] == 1
    assert result["success"]


def test_empty_valid_set_stops_and_fails(method):
    environment = ScriptedEnvironment(invalid=True)
    result = run_episode(Runtime(environment, ConstantPolicy(method)), method, Selector(method, strategy="first"), 1, 2)
    assert environment.stopped
    assert result["stopped_invalid"]
    assert not result["success"]
    assert result["steps"] == 0


def test_ties_use_original_candidate_order(method):
    class Predictor:
        def predict(self, features, prefixes, previous_action, remaining):
            return np.full((len(prefixes), 2), 0.5)
    selector = Selector(method, Predictor())
    decision = selector.select(np.zeros(7), np.zeros((3, 2, 2)), np.zeros(2), 4)
    assert decision.index == 0
