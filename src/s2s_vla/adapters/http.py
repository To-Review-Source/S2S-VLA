import json
from urllib.request import Request, urlopen
import numpy as np
from s2s_vla.config import MethodConfig
from s2s_vla.interfaces import Proposal
from s2s_vla.storage import jsonable


def request_json(url, payload=None, timeout=120):
    data = None if payload is None else json.dumps(jsonable(payload), allow_nan=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def proposal_payload(proposal):
    finite = np.isfinite(proposal.chunks)
    chunks = proposal.chunks.astype(object)
    chunks[~finite] = None
    return {"features": proposal.features.tolist(), "chunks": chunks.tolist(), "finite_steps": finite.all(axis=-1).tolist()}


def proposal_from_payload(response):
    features = np.asarray(response["features"], dtype=np.float32)
    chunks = np.asarray(response["chunks"], dtype=np.float32)
    if "finite_steps" in response:
        finite_steps = np.asarray(response["finite_steps"])
        if chunks.ndim != 3 or finite_steps.dtype.kind != "b" or finite_steps.shape != chunks.shape[:2]:
            raise ValueError("Invalid remote action finiteness mask")
        if not np.array_equal(finite_steps, np.isfinite(chunks).all(axis=-1)):
            raise ValueError("Remote action finiteness mask disagrees with commands")
    return Proposal(features, chunks)


class HTTPPolicy:
    def __init__(self, base_url, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def propose(self, context, count, rng):
        response = request_json(self.base_url + "/propose", {"observation": context.observation, "instruction": context.instruction, "previous_action": context.previous_action, "remaining": context.remaining, "count": count, "seed": int(rng.integers(0, 2**63 - 1))}, self.timeout)
        return proposal_from_payload(response)


class HTTPPredictor:
    def __init__(self, base_url, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        metadata = request_json(self.base_url + "/metadata", timeout=timeout)
        self.method = MethodConfig(**metadata["method"])
        self.scales = metadata["scales"]
        self.checkpoint_sha256 = metadata["checkpoint_sha256"]
        self.calibration_sha256 = metadata["calibration_sha256"]
        self.payload = {"dataset_sha256": metadata["dataset_sha256"]}

    def predict(self, features, prefixes, previous_action, remaining):
        response = request_json(self.base_url + "/predict", {"features": features, "prefixes": prefixes, "previous_action": previous_action, "remaining": remaining}, self.timeout)
        if response["checkpoint_sha256"] != self.checkpoint_sha256 or response["calibration_sha256"] != self.calibration_sha256:
            raise RuntimeError("Remote evaluator changed during evaluation")
        return np.asarray(response["probabilities"], dtype=np.float32)
