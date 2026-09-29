from pathlib import Path
import copy
import numpy as np
from s2s_vla.config import ExperimentConfig
from s2s_vla.interfaces import Context, load_runtime, valid_indices
from s2s_vla.objectives import command_roughness
from s2s_vla.storage import sha256, write_json


def execute_prefix(environment, prefix, remaining, previous_action):
    if len(prefix) < 1 or len(prefix) > remaining:
        raise ValueError("Invalid execution horizon")
    anomaly = False
    consumed = 0
    transition = None
    previous_action = np.asarray(previous_action).copy()
    for command in prefix:
        transition = environment.step(command)
        consumed += 1
        previous_action = np.asarray(command).copy()
        anomaly = anomaly or transition.anomaly
        if transition.done:
            break
    return transition, consumed, previous_action, bool(anomaly)


def branch_outcome(environment, policy, snapshot, context, prefix, method, seed):
    children = np.random.SeedSequence(seed).spawn(2)
    environment.restore(copy.deepcopy(snapshot))
    environment.reseed(int(children[0].generate_state(1)[0]))
    rng = np.random.default_rng(children[1])
    transition, consumed, previous, anomaly = execute_prefix(environment, prefix, context.remaining, context.previous_action)
    remaining = context.remaining - consumed
    while not transition.done and remaining > 0:
        continuation_context = Context(transition.observation, context.instruction, previous, remaining)
        proposal = policy.propose(continuation_context, 1, rng)
        proposal.validate(method, 1)
        next_prefix = proposal.chunks[0, :min(method.prefix_length, remaining)]
        if len(valid_indices(environment, next_prefix[None])) == 0:
            environment.stop()
            break
        transition, consumed, previous, _ = execute_prefix(environment, next_prefix, remaining, previous)
        remaining -= consumed
    return int(transition.success), int(anomaly)


def make_groups(config):
    indices = np.random.default_rng(config.seed).permutation(config.episodes)
    counts = np.ones(4, dtype=int)
    remainder = config.episodes - 4
    allocations = np.asarray(config.split_fractions) * remainder
    counts += np.floor(allocations).astype(int)
    for i in np.argsort(-(allocations - np.floor(allocations)), kind="stable")[:config.episodes - counts.sum()]:
        counts[i] += 1
    groups = []
    position = 0
    for partition, size in zip(("train", "calibration", "validation", "test"), counts):
        for index in indices[position:position + size]:
            groups.append({"id": f"episode-{int(index):06d}", "seed": config.seed + 10000 + int(index), "partition": partition})
        position += size
    return groups


def collect(config: ExperimentConfig, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Collection output must be empty")
    (output / "records").mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(config)
    environment, policy = runtime.environment, runtime.policy
    for name in ("snapshot", "restore", "reseed"):
        if not callable(getattr(environment, name, None)):
            raise TypeError(f"Offline collection requires environment.{name}")
    groups = make_groups(config)
    manifest = {"schema_version": 1, "status": "collecting", "config": config.to_dict(), "groups": groups, "records": [], "continuation": "base_policy", "labels": ["success_after_prefix_and_reference", "anomaly_during_prefix"]}
    write_json(output / "manifest.json", manifest)
    for group in groups:
        if group["partition"] not in ("train", "calibration"):
            continue
        observation = environment.reset(group["seed"])
        previous = np.asarray(environment.initial_action, dtype=np.float32).copy()
        remaining = config.method.max_steps
        decision = 0
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, group["seed"], 1]))
        label_rng = np.random.default_rng(np.random.SeedSequence([config.seed, group["seed"], 2]))
        while remaining > 0:
            context = Context(observation, environment.instruction, previous, remaining)
            h = min(config.method.prefix_length, remaining)
            if decision % config.context_stride == 0:
                proposal = policy.propose(context, config.method.candidates, rng)
                proposal.validate(config.method, config.method.candidates)
                prefixes = proposal.chunks[:, :h].copy()
                indices = valid_indices(environment, prefixes)
                prefixes = prefixes[indices]
                if len(prefixes) >= 2:
                    snapshot = environment.snapshot()
                    success = np.zeros((len(prefixes), config.branches), dtype=np.uint8)
                    anomaly = np.zeros_like(success)
                    seeds = label_rng.integers(0, 2**62, size=success.shape, dtype=np.int64)
                    try:
                        for k, prefix in enumerate(prefixes):
                            for r in range(config.branches):
                                success[k, r], anomaly[k, r] = branch_outcome(environment, policy, snapshot, context, prefix, config.method, int(seeds[k, r]))
                    finally:
                        environment.restore(snapshot)
                    context_id = f"{group['id']}-step-{config.method.max_steps - remaining:05d}"
                    relative = f"records/{context_id}.npz"
                    np.savez_compressed(output / relative, features=proposal.features, prefixes=prefixes, previous_action=previous, remaining=np.int64(remaining), success=success, anomaly=anomaly, branch_seeds=seeds, roughness=command_roughness(prefixes, previous, config.method), instruction=np.asarray(context.instruction))
                    manifest["records"].append({"path": relative, "group_id": group["id"], "context_id": context_id, "partition": group["partition"], "sha256": sha256(output / relative)})
            proposal = policy.propose(context, 1, rng)
            proposal.validate(config.method, 1)
            prefix = proposal.chunks[0, :h]
            if len(valid_indices(environment, prefix[None])) == 0:
                environment.stop()
                break
            transition, consumed, previous, _ = execute_prefix(environment, prefix, remaining, previous)
            observation = transition.observation
            remaining -= consumed
            decision += 1
            if transition.done:
                break
        write_json(output / "manifest.json", manifest)
    partitions = {row["partition"] for row in manifest["records"]}
    if not {"train", "calibration"}.issubset(partitions):
        raise RuntimeError("Insufficient valid branch data in train or calibration partition")
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    return manifest
