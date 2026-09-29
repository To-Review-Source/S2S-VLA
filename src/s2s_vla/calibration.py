from functools import partial
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from torch.utils.data import DataLoader
from s2s_vla.data import BranchDataset, collate_contexts, forward_batch
from s2s_vla.storage import sha256, write_json
from s2s_vla.training import load_checkpoint


def binary_nll(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if logits.shape != targets.shape or logits.size == 0 or not np.isfinite(logits).all() or not np.isfinite(targets).all() or np.any((targets < 0) | (targets > 1)):
        raise ValueError("Invalid calibration arrays")
    return float(np.mean(np.logaddexp(0, logits) - targets * logits))


def fit_temperature(logits, targets):
    result = minimize_scalar(lambda log_temperature: binary_nll(logits / np.exp(log_temperature), targets), bounds=(-6.0, 6.0), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature optimization failed")
    temperature = float(np.exp(result.x))
    if binary_nll(logits / temperature, targets) > binary_nll(logits, targets):
        return 1.0
    return temperature


def probability_metrics(logits, targets, temperatures):
    result = {}
    for i, name in enumerate(("success", "anomaly")):
        scaled = logits[:, i] / temperatures[i]
        probabilities = expit(scaled)
        result[name] = {"nll": binary_nll(scaled, targets[:, i]), "frequency_mse": float(np.mean((probabilities - targets[:, i]) ** 2)), "mean_prediction": float(probabilities.mean()), "mean_target": float(targets[:, i].mean())}
    return result


def calibrate(checkpoint, dataset_root, output, device="cpu"):
    model, payload = load_checkpoint(checkpoint, device)
    dataset = BranchDataset(dataset_root, "calibration")
    if sha256(Path(dataset_root) / "manifest.json") != payload["dataset_sha256"]:
        raise ValueError("Calibration must use the checkpoint's dataset partition manifest")
    groups = sorted({row["group_id"] for row in dataset.rows})
    if set(groups) & set(payload["train_groups"]):
        raise ValueError("Calibration and training groups overlap")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=partial(collate_contexts, method=model.method))
    logits, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            logits.append(forward_batch(model, batch, device).cpu().numpy())
            targets.append(batch["targets"].numpy())
    logits, targets = np.concatenate(logits), np.concatenate(targets)
    temperatures = [fit_temperature(logits[:, i], targets[:, i]) for i in range(2)]
    result = {"schema_version": 1, "checkpoint_sha256": sha256(checkpoint), "dataset_sha256": payload["dataset_sha256"], "groups": groups, "temperatures": temperatures, "temperature_log_bounds": [-6.0, 6.0], "before": probability_metrics(logits, targets, [1.0, 1.0]), "after": probability_metrics(logits, targets, temperatures)}
    write_json(output, result)
    return result
