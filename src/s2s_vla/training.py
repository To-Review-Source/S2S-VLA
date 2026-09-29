from functools import partial
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from s2s_vla.config import MethodConfig
from s2s_vla.data import BranchDataset, collate_contexts, forward_batch
from s2s_vla.losses import dual_objective_loss
from s2s_vla.model import DualHeadEvaluator
from s2s_vla.objectives import training_roughness_scale
from s2s_vla.storage import sha256, write_json


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint schema")
    method = MethodConfig(**payload["method"])
    model = DualHeadEvaluator(method)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def train(config, dataset_root, output, device="cpu"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    dataset = BranchDataset(dataset_root, "train")
    if dataset.method.to_dict() != config.method.to_dict():
        raise ValueError("Dataset and training method configurations differ")
    model = DualHeadEvaluator(config.method).to(device)
    features = np.stack([item["features"] for item in dataset.items])
    with torch.no_grad():
        model.feature_mean.copy_(torch.as_tensor(features.mean(0), device=device))
        model.feature_std.copy_(torch.as_tensor(np.maximum(features.std(0), 1e-4), device=device))
    roughness = np.concatenate([item["roughness"] for item in dataset.items])
    scales = [1.0, 1.0, training_roughness_scale(roughness)]
    loader = DataLoader(dataset, batch_size=config.batch_contexts, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(config.seed), collate_fn=partial(collate_contexts, method=config.method))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    for epoch in range(config.epochs):
        model.train()
        totals = {key: 0.0 for key in ("loss", "probability_loss", "preference_loss")}
        context_count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, batch, device)
            losses = dual_objective_loss(logits, batch["targets"].to(device), batch["roughness"].to(device), batch["offsets"], config.method, scales)
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError("Non-finite training loss")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            count = len(batch["offsets"]) - 1
            context_count += count
            for key in totals:
                totals[key] += float(losses[key].detach()) * count
        row = {"epoch": epoch + 1, **{key: value / context_count for key, value in totals.items()}}
        history.append(row)
        print(f"epoch={epoch + 1} loss={row['loss']:.6f}", flush=True)
    payload = {"schema_version": 1, "method": config.method.to_dict(), "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "scales": scales, "train_groups": sorted({row["group_id"] for row in dataset.rows}), "dataset_sha256": sha256(Path(dataset_root) / "manifest.json"), "epochs": config.epochs, "seed": config.seed, "feature_source": "frozen_policy", "probability_semantics": ["prefix_then_reference_success", "prefix_anomaly"]}
    torch.save(payload, output / "evaluator.pt")
    write_json(output / "training.json", {"history": history, "scales": scales, "config": config.to_dict(), "contexts": len(dataset), "candidates": int(sum(len(item["prefixes"]) for item in dataset.items))})
    return output / "evaluator.pt"
