from dataclasses import dataclass
import numpy as np
import torch
from s2s_vla.objectives import augmented_tchebycheff, command_roughness, cost_vectors
from s2s_vla.storage import read_json, sha256
from s2s_vla.training import load_checkpoint


class TorchPredictor:
    def __init__(self, checkpoint, calibration, device="cpu", batch_size=128):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model, self.payload = load_checkpoint(checkpoint, device)
        self.method = self.model.method
        self.scales = self.payload["scales"]
        self.checkpoint_sha256 = sha256(checkpoint)
        self.calibration_sha256 = sha256(calibration)
        calibration_data = read_json(calibration)
        if calibration_data["checkpoint_sha256"] != self.checkpoint_sha256 or calibration_data["dataset_sha256"] != self.payload["dataset_sha256"]:
            raise ValueError("Calibration artifact does not match checkpoint")
        self.temperatures = np.asarray(calibration_data["temperatures"], dtype=np.float32)
        if self.temperatures.shape != (2,) or not np.isfinite(self.temperatures).all() or np.any(self.temperatures <= 0):
            raise ValueError("Invalid calibration temperatures")
        self.device = device
        self.batch_size = batch_size

    def predict(self, features, prefixes, previous_action, remaining):
        features = np.asarray(features, dtype=np.float32)
        prefixes = np.asarray(prefixes, dtype=np.float32)
        previous_action = np.asarray(previous_action, dtype=np.float32)
        if features.shape != (self.method.feature_dim,) or previous_action.shape != (self.method.action_dim,):
            raise ValueError("Invalid context dimensions")
        if prefixes.ndim != 3 or prefixes.shape[0] < 1 or prefixes.shape[2] != self.method.action_dim:
            raise ValueError("Invalid prefix dimensions")
        if not 1 <= remaining <= self.method.max_steps or prefixes.shape[1] != min(self.method.prefix_length, remaining):
            raise ValueError("Prefix length must equal the capped execution horizon")
        if not all(np.isfinite(value).all() for value in (features, prefixes, previous_action)):
            raise ValueError("Non-finite evaluator input")
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(prefixes), self.batch_size):
                values = prefixes[start:start + self.batch_size]
                count, length, _ = values.shape
                padded = np.zeros((count, self.method.prefix_length, self.method.action_dim), dtype=np.float32)
                padded[:, :length] = values
                logits = self.model(torch.as_tensor(np.repeat(features[None], count, 0), device=self.device), torch.as_tensor(padded, device=self.device), torch.full((count,), length, device=self.device), torch.as_tensor(np.repeat(previous_action[None], count, 0), device=self.device), torch.full((count,), remaining, device=self.device))
                probabilities = torch.sigmoid(logits / logits.new_tensor(self.temperatures))
                predictions.append(probabilities.cpu().numpy())
        return np.concatenate(predictions)


@dataclass
class Decision:
    index: int
    probabilities: np.ndarray | None
    roughness: np.ndarray
    costs: np.ndarray | None
    scores: np.ndarray


class Selector:
    def __init__(self, method, predictor=None, weights=None, scales=None, strategy="s2s"):
        if strategy not in ("s2s", "first", "success", "weighted_sum"):
            raise ValueError("Unknown selection strategy")
        self.method = method
        self.predictor = predictor
        self.weights = np.asarray(weights if weights is not None else method.preferences[0], dtype=float)
        self.scales = np.asarray(scales if scales is not None else [1, 1, 1], dtype=float)
        augmented_tchebycheff(np.zeros((1, 3)), self.weights, self.scales, method.rho)
        self.strategy = strategy
        if strategy != "first" and predictor is None:
            raise ValueError("This strategy requires a calibrated predictor")

    def select(self, features, prefixes, previous_action, remaining):
        roughness = command_roughness(prefixes, previous_action, self.method)
        if len(prefixes) == 0:
            raise ValueError("Cannot select from an empty candidate set")
        if self.strategy == "first":
            return Decision(0, None, roughness, None, np.zeros(len(prefixes)))
        probabilities = self.predictor.predict(features, prefixes, previous_action, remaining)
        costs = cost_vectors(probabilities, roughness)
        if self.strategy == "success":
            scores = costs[:, 0]
        elif self.strategy == "weighted_sum":
            scores = (costs / self.scales * self.weights).sum(-1)
        else:
            scores = augmented_tchebycheff(costs, self.weights, self.scales, self.method.rho)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite selection score")
        return Decision(int(np.argmin(scores)), probabilities, roughness, costs, scores)


def load_selection(path, predictor):
    value = read_json(path)
    if value["checkpoint_sha256"] != predictor.checkpoint_sha256 or value["calibration_sha256"] != predictor.calibration_sha256:
        raise ValueError("Selected preferences do not match calibrated evaluator")
    if value["dataset_sha256"] != predictor.payload["dataset_sha256"]:
        raise ValueError("Preference validation used a different dataset")
    return value
