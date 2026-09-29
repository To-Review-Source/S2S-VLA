from pathlib import Path
import numpy as np
from s2s_vla.controller import aggregate_episodes, run_episode
from s2s_vla.data import load_manifest
from s2s_vla.inference import Selector, TorchPredictor, load_selection
from s2s_vla.interfaces import load_runtime
from s2s_vla.storage import sha256, write_json


def check_experiment(config, dataset_root, predictor):
    manifest = load_manifest(dataset_root)
    if config.method.to_dict() != manifest["config"]["method"]:
        raise ValueError("Evaluation configuration differs from collection method")
    if predictor is not None:
        if config.method.to_dict() != predictor.method.to_dict():
            raise ValueError("Evaluation configuration differs from checkpoint")
        if sha256(Path(dataset_root) / "manifest.json") != predictor.payload["dataset_sha256"]:
            raise ValueError("Evaluation dataset does not match checkpoint")
    return manifest


def evaluate_groups(config, groups, selector):
    runtime = load_runtime(config)
    episodes = []
    for group in groups:
        policy_seed = int(np.random.SeedSequence([config.seed, group["seed"], 999]).generate_state(1)[0])
        episode = run_episode(runtime, config.method, selector, group["seed"], policy_seed)
        episode["group_id"] = group["id"]
        episodes.append(episode)
    return {"summary": aggregate_episodes(episodes), "episodes": episodes}


def select_preferences(config, dataset_root, checkpoint, calibration, output, device="cpu"):
    predictor = TorchPredictor(checkpoint, calibration, device)
    manifest = check_experiment(config, dataset_root, predictor)
    groups = [group for group in manifest["groups"] if group["partition"] == "validation"]
    candidates = []
    for weights in config.method.preferences:
        selector = Selector(config.method, predictor, weights, predictor.scales)
        report = evaluate_groups(config, groups, selector)
        candidates.append({"weights": weights, **report})
    best = max(range(len(candidates)), key=lambda i: candidates[i]["summary"]["success_rate"])
    result = {"schema_version": 1, "checkpoint_sha256": predictor.checkpoint_sha256, "calibration_sha256": predictor.calibration_sha256, "dataset_sha256": predictor.payload["dataset_sha256"], "weights": candidates[best]["weights"], "selected_grid_index": best, "selection_metric": "validation_closed_loop_success_rate", "tie_break": "first_grid_entry", "groups": [group["id"] for group in groups], "candidate_budget": config.method.candidates, "validation_results": candidates}
    write_json(output, result)
    return result


def evaluate(config, dataset_root, checkpoint, calibration, selection, output, device="cpu", strategy="s2s", predictor_url=None):
    if strategy == "first":
        predictor = None
        selection_data = None
        selector = Selector(config.method, strategy="first")
    else:
        if predictor_url:
            from s2s_vla.adapters.http import HTTPPredictor
            predictor = HTTPPredictor(predictor_url)
        else:
            predictor = TorchPredictor(checkpoint, calibration, device)
        selection_data = load_selection(selection, predictor)
        selector = Selector(config.method, predictor, selection_data["weights"], predictor.scales, strategy)
    manifest = check_experiment(config, dataset_root, predictor)
    groups = [group for group in manifest["groups"] if group["partition"] == "test"]
    if selection_data and set(selection_data["groups"]) & {group["id"] for group in groups}:
        raise ValueError("Preference validation overlaps the test partition")
    result = evaluate_groups(config, groups, selector)
    result.update({"strategy": strategy, "candidate_budget": config.method.candidates, "prefix_length": config.method.prefix_length, "dataset_sha256": sha256(Path(dataset_root) / "manifest.json"), "weights": selector.weights.tolist() if predictor else None, "checkpoint_sha256": predictor.checkpoint_sha256 if predictor else None, "calibration_sha256": predictor.calibration_sha256 if predictor else None, "ablation_weight_policy": "reuse_s2s_validation_weights" if strategy == "weighted_sum" else None})
    write_json(output, result)
    return result
