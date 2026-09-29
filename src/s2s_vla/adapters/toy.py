import copy
import numpy as np
from s2s_vla.config import MethodConfig
from s2s_vla.interfaces import Context, Proposal, Runtime, Transition


class ToyEnvironment:
    instruction = "Move the point to the target."

    def __init__(self, max_steps=24, noise=0.015):
        self.max_steps = max_steps
        self.noise = noise
        self.initial_action = np.zeros(2, dtype=np.float32)
        self.rng = np.random.default_rng()
        self.position = np.zeros(2)
        self.goal = np.zeros(2)
        self.obstacle = np.zeros(2)
        self.radius = 0.2
        self.elapsed = 0
        self.stopped = False

    def observation(self):
        return {"position": self.position.copy(), "goal": self.goal.copy(), "obstacle": self.obstacle.copy(), "radius": self.radius}

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.position = np.array([-0.8, self.rng.uniform(-0.45, 0.45)])
        self.goal = np.array([0.8, self.rng.uniform(-0.45, 0.45)])
        self.obstacle = np.array([self.rng.uniform(-0.1, 0.1), self.rng.uniform(-0.15, 0.15)])
        self.radius = float(self.rng.uniform(0.16, 0.25))
        self.elapsed = 0
        self.stopped = False
        return self.observation()

    def valid_prefix(self, prefix):
        return prefix.ndim == 2 and prefix.shape[1] == 2 and bool(np.all(np.abs(prefix) <= 0.22 + 1e-6))

    def step(self, action):
        if self.stopped or self.elapsed >= self.max_steps:
            raise RuntimeError("Episode has already stopped")
        action = np.asarray(action, dtype=float)
        if not self.valid_prefix(action[None]):
            raise ValueError("Invalid toy action")
        before = self.position.copy()
        proposed = before + action + self.rng.normal(0, self.noise, 2)
        segment = proposed - before
        projection = np.clip(np.dot(self.obstacle - before, segment) / max(np.dot(segment, segment), 1e-12), 0, 1)
        anomaly = bool(np.linalg.norm(before + projection * segment - self.obstacle) < self.radius or np.any(np.abs(proposed) > 1.1))
        self.position = np.clip(proposed, -1.1, 1.1)
        self.elapsed += 1
        success = bool(np.linalg.norm(self.position - self.goal) < 0.12)
        truncated = self.elapsed >= self.max_steps and not success
        self.stopped = success or truncated
        return Transition(self.observation(), success, anomaly, success, truncated)

    def stop(self):
        self.stopped = True

    def snapshot(self):
        return copy.deepcopy({"position": self.position, "goal": self.goal, "obstacle": self.obstacle, "radius": self.radius, "elapsed": self.elapsed, "stopped": self.stopped, "rng": self.rng.bit_generator.state})

    def restore(self, snapshot):
        snapshot = copy.deepcopy(snapshot)
        for key in ("position", "goal", "obstacle", "radius", "elapsed", "stopped"):
            setattr(self, key, snapshot[key])
        self.rng.bit_generator.state = snapshot["rng"]

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)


class ToyPolicy:
    def __init__(self, method: MethodConfig):
        if method.action_dim != 2 or method.feature_dim != 7:
            raise ValueError("Toy runtime requires action_dim=2 and feature_dim=7")
        self.method = method

    def propose(self, context: Context, count, rng):
        obs = context.observation
        position = np.asarray(obs["position"])
        goal = np.asarray(obs["goal"])
        obstacle = np.asarray(obs["obstacle"])
        features = np.concatenate([position, goal, obstacle, [obs["radius"]]]).astype(np.float32)
        chunks = np.empty((count, self.method.horizon, 2), dtype=np.float32)
        for k in range(count):
            mode = int(rng.choice([-1, 0, 1], p=[0.25, 0.5, 0.25]))
            predicted_position = position.copy()
            speed = float(rng.uniform(0.10, 0.23))
            for t in range(self.method.horizon):
                waypoint = goal
                if mode and predicted_position[0] < obstacle[0] + obs["radius"]:
                    waypoint = obstacle + np.array([0.25, mode * (obs["radius"] + 0.23)])
                direction = waypoint - predicted_position
                command = direction / max(np.linalg.norm(direction), 1e-8) * min(speed, np.linalg.norm(direction))
                command = np.clip(command + rng.normal(0, 0.035, 2), -0.22, 0.22)
                chunks[k, t] = command
                predicted_position += command
        return Proposal(features, chunks)


def create_runtime(method, noise=0.015):
    return Runtime(ToyEnvironment(method.max_steps, noise), ToyPolicy(method))
