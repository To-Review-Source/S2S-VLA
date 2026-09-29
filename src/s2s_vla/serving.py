from threading import Lock
from typing import Any
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from s2s_vla.adapters.http import proposal_payload
from s2s_vla.inference import TorchPredictor
from s2s_vla.interfaces import Context, load_runtime


class ProposeRequest(BaseModel):
    observation: Any
    instruction: str
    previous_action: list[float]
    remaining: int = Field(gt=0)
    count: int = Field(ge=1, le=4096)
    seed: int = Field(ge=0, lt=2**63)


class PredictRequest(BaseModel):
    features: list[float]
    prefixes: list[list[list[float]]]
    previous_action: list[float]
    remaining: int = Field(gt=0)


def create_policy_app(config):
    policy = load_runtime(config).policy
    lock = Lock()
    app = FastAPI(title="S2S-VLA policy")

    @app.get("/health")
    def health():
        return {"status": "ready", "service": "policy"}

    @app.post("/propose")
    def propose(request: ProposeRequest):
        previous = np.asarray(request.previous_action, dtype=np.float32)
        if previous.shape != (config.method.action_dim,) or not np.isfinite(previous).all():
            raise HTTPException(422, "Invalid previous_action")
        if request.remaining > config.method.max_steps:
            raise HTTPException(422, "Remaining budget exceeds configured maximum")
        context = Context(request.observation, request.instruction, previous, request.remaining)
        try:
            with lock:
                proposal = policy.propose(context, request.count, np.random.default_rng(request.seed))
                proposal.validate(config.method, request.count)
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from error
        return proposal_payload(proposal)

    return app


def create_evaluator_app(checkpoint, calibration, device="cpu"):
    predictor = TorchPredictor(checkpoint, calibration, device)
    app = FastAPI(title="S2S-VLA evaluator")
    lock = Lock()

    @app.get("/health")
    def health():
        return {"status": "ready", "service": "evaluator"}

    @app.get("/metadata")
    def metadata():
        return {"method": predictor.method.to_dict(), "scales": predictor.scales, "checkpoint_sha256": predictor.checkpoint_sha256, "calibration_sha256": predictor.calibration_sha256, "dataset_sha256": predictor.payload["dataset_sha256"]}

    @app.post("/predict")
    def predict(request: PredictRequest):
        try:
            with lock:
                probabilities = predictor.predict(request.features, request.prefixes, request.previous_action, request.remaining)
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from error
        return {"probabilities": probabilities.tolist(), "checkpoint_sha256": predictor.checkpoint_sha256, "calibration_sha256": predictor.calibration_sha256}

    return app
