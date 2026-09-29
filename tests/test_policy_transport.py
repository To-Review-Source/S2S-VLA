import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
import numpy as np
from s2s_vla.adapters.http import HTTPPolicy, proposal_from_payload, proposal_payload
from s2s_vla.config import MethodConfig
from s2s_vla.controller import run_episode
from s2s_vla.interfaces import Proposal, Runtime, Transition
from s2s_vla.objectives import command_roughness


class RecordingEnvironment:
    instruction = "complete"
    initial_action = np.zeros(2, dtype=np.float32)

    def __init__(self):
        self.stop_calls = 0
        self.commands = []

    def reset(self, seed):
        self.stop_calls = 0
        self.commands = []
        return np.zeros(2)

    def valid_prefix(self, prefix):
        return bool(np.all(np.abs(prefix) <= 1))

    def step(self, command):
        self.commands.append(command.copy())
        return Transition(np.zeros(2))

    def stop(self):
        self.stop_calls += 1


class FixedPolicy:
    def __init__(self, chunks):
        self.chunks = chunks

    def propose(self, context, count, rng):
        return Proposal(np.zeros(2), self.chunks.copy())


class FirstSelector:
    def __init__(self, method):
        self.method = method

    def select(self, features, prefixes, previous_action, remaining):
        return SimpleNamespace(index=0, probabilities=None, scores=np.zeros(len(prefixes)), roughness=command_roughness(prefixes, previous_action, self.method))


class PolicyTransportTests(unittest.TestCase):
    def setUp(self):
        self.method = MethodConfig(action_dim=2, feature_dim=2, horizon=4, prefix_length=2, candidates=3, max_steps=2, continuous_indices=[0, 1], roughness_operator=[[1, 0], [0, 1]], action_scale=[1, 1])
        self.chunks = np.zeros((3, 4, 2), dtype=np.float32)
        self.environment = RecordingEnvironment()
        self.selector = FirstSelector(self.method)

    def payload(self):
        proposal = Proposal(np.zeros(2), self.chunks.copy())
        proposal.validate(self.method, self.method.candidates)
        return json.loads(json.dumps(proposal_payload(proposal), allow_nan=False))

    def run_remote(self):
        runtime = Runtime(self.environment, HTTPPolicy("http://policy"))
        with patch("s2s_vla.adapters.http.request_json", return_value=self.payload()):
            return run_episode(runtime, self.method, self.selector, 1, 2)

    def test_nonfinite_commands_round_trip_through_strict_json(self):
        self.chunks[0, 0] = [np.nan, np.inf]
        self.chunks[2, 3, 0] = -np.inf
        payload = self.payload()
        self.assertEqual(payload["chunks"][0][0], [None, None])
        self.assertFalse(payload["finite_steps"][0][0])
        self.assertFalse(payload["finite_steps"][2][3])
        proposal = proposal_from_payload(payload)
        np.testing.assert_array_equal(np.isfinite(proposal.chunks), np.isfinite(self.chunks))
        np.testing.assert_array_equal(proposal.chunks[np.isfinite(self.chunks)], self.chunks[np.isfinite(self.chunks)])

    def test_legacy_finite_payload_remains_compatible(self):
        payload = self.payload()
        del payload["finite_steps"]
        proposal = proposal_from_payload(payload)
        np.testing.assert_array_equal(proposal.chunks, self.chunks)

    def test_inconsistent_finiteness_mask_is_rejected(self):
        self.chunks[0, 0, 0] = np.nan
        payload = self.payload()
        payload["finite_steps"][0][0] = True
        with self.assertRaises(ValueError):
            proposal_from_payload(payload)

    def test_malformed_finiteness_masks_are_rejected(self):
        for mask in ([True], [[1] * 4] * 3, [[True] * 3] * 3):
            with self.subTest(mask=mask):
                payload = self.payload()
                payload["finite_steps"] = mask
                with self.assertRaises(ValueError):
                    proposal_from_payload(payload)

    def test_mixed_candidates_preserve_valid_candidate_order(self):
        self.chunks[0, 0, 0] = np.nan
        self.chunks[1] = 0.25
        self.chunks[2] = 0.5
        report = self.run_remote()
        self.assertEqual(report["decisions"][0]["valid_indices"], [1, 2])
        self.assertEqual(report["decisions"][0]["selected_index"], 1)
        np.testing.assert_array_equal(self.environment.commands, self.chunks[1, :2])

    def test_all_invalid_candidates_stop_and_return_failure(self):
        self.chunks[:, 0, 0] = np.nan
        report = self.run_remote()
        self.assertTrue(report["stopped_invalid"])
        self.assertFalse(report["success"])
        self.assertEqual(report["steps"], 0)
        self.assertEqual(self.environment.stop_calls, 1)
        self.assertEqual(self.environment.commands, [])

    def test_unused_chunk_suffix_does_not_invalidate_prefix(self):
        self.chunks[:, 2:] = np.nan
        report = self.run_remote()
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["decisions"][0]["valid_indices"], [0, 1, 2])
        self.assertFalse(report["stopped_invalid"])

    def test_remaining_budget_caps_finiteness_check(self):
        self.method.max_steps = 1
        self.chunks[:, 1:] = np.nan
        report = self.run_remote()
        self.assertEqual(report["steps"], 1)
        self.assertEqual(report["decisions"][0]["valid_indices"], [0, 1, 2])

    def test_platform_validity_still_filters_finite_commands(self):
        self.chunks[0] = 2
        self.chunks[2] = -2
        report = self.run_remote()
        self.assertEqual(report["decisions"][0]["valid_indices"], [1])
        self.assertEqual(report["decisions"][0]["selected_index"], 1)

    def test_http_failure_stops_before_propagating_error(self):
        error = HTTPError("http://policy/propose", 422, "invalid response", {}, None)
        runtime = Runtime(self.environment, HTTPPolicy("http://policy"))
        with patch("s2s_vla.adapters.http.request_json", side_effect=error):
            with self.assertRaises(HTTPError) as raised:
                run_episode(runtime, self.method, self.selector, 1, 2)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.environment.stop_calls, 1)
        self.assertEqual(self.environment.commands, [])

    def test_evaluator_failure_stops_before_propagating_error(self):
        error = RuntimeError("evaluator unavailable")
        runtime = Runtime(self.environment, FixedPolicy(self.chunks))
        with patch.object(self.selector, "select", side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                run_episode(runtime, self.method, self.selector, 1, 2)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.environment.stop_calls, 1)

    def test_step_failure_stops_before_propagating_error(self):
        error = RuntimeError("execution interrupted")
        runtime = Runtime(self.environment, FixedPolicy(self.chunks))
        with patch.object(self.environment, "step", side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                run_episode(runtime, self.method, self.selector, 1, 2)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.environment.stop_calls, 1)

    def test_local_and_remote_candidates_have_same_decision(self):
        self.chunks[0, 0, 0] = np.inf
        self.chunks[1, 2:] = np.nan
        self.chunks[2] = 0.5
        remote = self.run_remote()
        local = run_episode(Runtime(self.environment, FixedPolicy(self.chunks)), self.method, self.selector, 1, 2)
        for key in ("success", "stopped_invalid", "steps", "mean_command_roughness"):
            self.assertEqual(remote[key], local[key])
        for key in ("selected_index", "valid_indices", "scores", "roughness"):
            self.assertEqual(remote["decisions"][0][key], local["decisions"][0][key])


if __name__ == "__main__":
    unittest.main()
