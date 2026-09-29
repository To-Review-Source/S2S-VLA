import copy
import numpy as np
from s2s_vla.interfaces import Transition


class GymnasiumAdapter:
    def __init__(self, env, initial_action, instruction_fn, observation_fn, success_fn, anomaly_fn, validity_fn, stop_fn, action_transform=None, snapshot_fn=None, restore_fn=None, reseed_fn=None):
        self.env = env
        self.initial_action = np.asarray(initial_action, dtype=np.float32)
        self.instruction_fn = instruction_fn
        self.observation_fn = observation_fn
        self.success_fn = success_fn
        self.anomaly_fn = anomaly_fn
        self.validity_fn = validity_fn
        self.stop_fn = stop_fn
        self.action_transform = action_transform or (lambda command: command)
        self.snapshot_fn = snapshot_fn
        self.restore_fn = restore_fn
        self.reseed_fn = reseed_fn
        self.instruction = ""
        self.last_observation = None

    def reset(self, seed):
        observation, info = self.env.reset(seed=seed)
        self.instruction = str(self.instruction_fn(self.env, observation, info))
        self.last_observation = self.observation_fn(self.env, observation, info)
        return copy.deepcopy(self.last_observation)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(self.action_transform(np.asarray(action).copy()))
        success = bool(self.success_fn(self.env, observation, reward, terminated, truncated, info))
        anomaly = bool(self.anomaly_fn(self.env, observation, reward, terminated, truncated, info))
        self.last_observation = self.observation_fn(self.env, observation, info)
        return Transition(copy.deepcopy(self.last_observation), success, anomaly, bool(terminated), bool(truncated))

    def valid_prefix(self, prefix):
        return bool(self.validity_fn(self.env, prefix))

    def stop(self):
        self.stop_fn(self.env)

    def snapshot(self):
        if self.snapshot_fn is None:
            raise NotImplementedError("A complete simulator snapshot callback is required for offline collection")
        return copy.deepcopy({"simulator": self.snapshot_fn(self.env), "observation": self.last_observation, "instruction": self.instruction})

    def restore(self, snapshot):
        if self.restore_fn is None:
            raise NotImplementedError("A simulator restoration callback is required for offline collection")
        snapshot = copy.deepcopy(snapshot)
        self.restore_fn(self.env, snapshot["simulator"])
        self.last_observation = snapshot["observation"]
        self.instruction = snapshot["instruction"]

    def reseed(self, seed):
        if self.reseed_fn is None:
            raise NotImplementedError("Reseeding must preserve the restored physical state")
        self.reseed_fn(self.env, seed)
