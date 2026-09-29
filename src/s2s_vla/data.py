from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from s2s_vla.config import MethodConfig
from s2s_vla.objectives import command_roughness
from s2s_vla.storage import read_json, sha256


def load_manifest(root):
    manifest = read_json(Path(root) / "manifest.json")
    if manifest["schema_version"] != 1 or manifest["status"] != "complete":
        raise ValueError("Dataset is incomplete or uses an unsupported schema")
    groups = manifest["groups"]
    if len({x["id"] for x in groups}) != len(groups) or len({x["seed"] for x in groups}) != len(groups):
        raise ValueError("Episode groups and initial-condition seeds must be disjoint")
    membership = {x["id"]: x["partition"] for x in groups}
    if set(membership.values()) != {"train", "calibration", "validation", "test"}:
        raise ValueError("Dataset requires four disjoint partitions")
    paths, contexts = set(), set()
    for record in manifest["records"]:
        if record["group_id"] not in membership or membership[record["group_id"]] != record["partition"]:
            raise ValueError("Related contexts cross data partitions")
        if record["path"] in paths or record["context_id"] in contexts:
            raise ValueError("Duplicate dataset records")
        paths.add(record["path"])
        contexts.add(record["context_id"])
    return manifest


class BranchDataset(Dataset):
    def __init__(self, root, partition, verify=True):
        if partition not in ("train", "calibration"):
            raise ValueError("Branch labels are reserved for train and calibration")
        self.root = Path(root).resolve()
        self.manifest = load_manifest(root)
        self.method = MethodConfig(**self.manifest["config"]["method"])
        self.rows = [row for row in self.manifest["records"] if row["partition"] == partition]
        if not self.rows:
            raise ValueError(f"No records for {partition}")
        self.partition = partition
        self.items = []
        for row in self.rows:
            path = (self.root / row["path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Record path escapes dataset root")
            if verify and sha256(path) != row["sha256"]:
                raise ValueError(f"Record checksum mismatch: {row['context_id']}")
            with np.load(path, allow_pickle=False) as archive:
                item = {key: archive[key].copy() for key in archive.files}
            self.validate(item)
            self.items.append(item)

    def validate(self, item):
        method = self.method
        prefixes = item["prefixes"]
        if prefixes.ndim != 3 or prefixes.shape[0] < 2 or prefixes.shape[2] != method.action_dim:
            raise ValueError("Invalid candidate prefixes")
        count, length, _ = prefixes.shape
        remaining = int(item["remaining"])
        if length != min(method.prefix_length, remaining) or not 1 <= remaining <= method.max_steps:
            raise ValueError("Incorrect prefix horizon")
        if item["features"].shape != (method.feature_dim,) or item["previous_action"].shape != (method.action_dim,):
            raise ValueError("Invalid feature or previous-command shape")
        for key in ("success", "anomaly"):
            values = item[key]
            if values.ndim != 2 or values.shape[0] != count or values.shape[1] < 1 or not np.isin(values, [0, 1]).all():
                raise ValueError("Branch outcomes must be binary")
        if item["success"].shape != item["anomaly"].shape or item["branch_seeds"].shape != item["success"].shape:
            raise ValueError("Branch arrays must have matching shapes")
        for key in ("features", "prefixes", "previous_action", "roughness"):
            if not np.isfinite(item[key]).all():
                raise ValueError("Non-finite data")
        expected = command_roughness(prefixes, item["previous_action"], method)
        if not np.allclose(item["roughness"], expected, rtol=1e-6, atol=1e-8):
            raise ValueError("Stored roughness disagrees with commands")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_contexts(items, method):
    arrays = {name: [] for name in ("features", "prefixes", "lengths", "previous_actions", "remaining", "targets", "roughness")}
    offsets = [0]
    for item in items:
        count, length, _ = item["prefixes"].shape
        padded = np.zeros((count, method.prefix_length, method.action_dim), dtype=np.float32)
        padded[:, :length] = item["prefixes"]
        arrays["features"].append(np.repeat(item["features"][None], count, axis=0))
        arrays["prefixes"].append(padded)
        arrays["lengths"].append(np.full(count, length, dtype=np.int64))
        arrays["previous_actions"].append(np.repeat(item["previous_action"][None], count, axis=0))
        arrays["remaining"].append(np.full(count, int(item["remaining"]), dtype=np.int64))
        arrays["targets"].append(np.stack([item["success"].mean(-1), item["anomaly"].mean(-1)], axis=-1))
        arrays["roughness"].append(item["roughness"])
        offsets.append(offsets[-1] + count)
    batch = {key: torch.as_tensor(np.concatenate(value), dtype=torch.long if key in ("lengths", "remaining") else torch.float32) for key, value in arrays.items()}
    batch["offsets"] = offsets
    return batch


def forward_batch(model, batch, device):
    names = ("features", "prefixes", "lengths", "previous_actions", "remaining")
    return model(**{name: batch[name].to(device) for name in names})
