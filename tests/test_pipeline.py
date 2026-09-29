import copy
import numpy as np
import pytest
from fastapi.testclient import TestClient
from s2s_vla.calibration import calibrate
from s2s_vla.collection import collect
from s2s_vla.data import BranchDataset, load_manifest
from s2s_vla.experiments import evaluate, select_preferences
from s2s_vla.inference import TorchPredictor
from s2s_vla.interfaces import Context, load_runtime
from s2s_vla.serving import create_evaluator_app, create_policy_app
from s2s_vla.storage import read_json, write_json
from s2s_vla.training import train


@pytest.fixture
def artifacts(tmp_path, config):
    data = tmp_path / "data"
    manifest = collect(config, data)
    checkpoint = train(config, data, tmp_path / "training")
    calibration = tmp_path / "calibration.json"
    calibrate(checkpoint, data, calibration)
    return data, manifest, checkpoint, calibration


def test_complete_pipeline_and_artifact_integrity(artifacts, config, tmp_path):
    data, manifest, checkpoint, calibration = artifacts
    partitions = {}
    for group in manifest["groups"]:
        partitions.setdefault(group["partition"], set()).add(group["id"])
    for name, values in partitions.items():
        for other, other_values in partitions.items():
            if name != other:
                assert not values & other_values
    selection = tmp_path / "selection.json"
    selected = select_preferences(config, data, checkpoint, calibration, selection)
    assert selected["weights"] in config.method.preferences
    report = evaluate(config, data, checkpoint, calibration, selection, tmp_path / "test.json")
    assert 0 <= report["summary"]["success_rate"] <= 1
    assert {x["group_id"] for x in report["episodes"]} == partitions["test"]
    predictor = TorchPredictor(checkpoint, calibration)
    item = BranchDataset(data, "calibration")[0]
    probabilities = predictor.predict(item["features"], item["prefixes"], item["previous_action"], int(item["remaining"]))
    reverse = predictor.predict(item["features"], item["prefixes"][::-1].copy(), item["previous_action"], int(item["remaining"]))
    np.testing.assert_allclose(probabilities, reverse[::-1], atol=1e-6)
    wrong = read_json(calibration)
    wrong["checkpoint_sha256"] = "wrong"
    write_json(tmp_path / "wrong.json", wrong)
    with pytest.raises(ValueError):
        TorchPredictor(checkpoint, tmp_path / "wrong.json")


def test_evaluator_http_matches_local(artifacts):
    data, manifest, checkpoint, calibration = artifacts
    item = BranchDataset(data, "calibration")[0]
    payload = {"features": item["features"].tolist(), "prefixes": item["prefixes"].tolist(), "previous_action": item["previous_action"].tolist(), "remaining": int(item["remaining"])}
    client = TestClient(create_evaluator_app(checkpoint, calibration))
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    local = TorchPredictor(checkpoint, calibration).predict(**payload)
    np.testing.assert_allclose(response.json()["probabilities"], local, atol=1e-6)
    payload["remaining"] = 0
    assert client.post("/predict", json=payload).status_code == 422


def test_policy_http_contract(config):
    runtime = load_runtime(config)
    observation = runtime.environment.reset(4)
    payload = {"observation": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in observation.items()}, "instruction": runtime.environment.instruction, "previous_action": [0, 0], "remaining": 10, "count": 3, "seed": 19}
    client = TestClient(create_policy_app(config))
    response = client.post("/propose", json=payload)
    assert response.status_code == 200
    assert np.asarray(response.json()["chunks"]).shape == (3, config.method.horizon, 2)
    context = Context(observation, runtime.environment.instruction, np.zeros(2), 10)
    local = runtime.policy.propose(context, 3, np.random.default_rng(19))
    np.testing.assert_allclose(response.json()["chunks"], local.chunks)


def test_manifest_rejects_group_leakage(tmp_path, config):
    data = tmp_path / "data"
    collect(config, data)
    manifest = read_json(data / "manifest.json")
    manifest["groups"].append(copy.deepcopy(manifest["groups"][0]))
    write_json(data / "manifest.json", manifest)
    with pytest.raises(ValueError):
        load_manifest(data)


def test_dataset_detects_corrupted_branch_record(tmp_path, config):
    data = tmp_path / "data"
    manifest = collect(config, data)
    row = next(row for row in manifest["records"] if row["partition"] == "train")
    with (data / row["path"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        BranchDataset(data, "train")
