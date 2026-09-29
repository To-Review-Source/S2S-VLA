import time
import numpy as np
from s2s_vla.collection import execute_prefix
from s2s_vla.interfaces import Context, valid_indices
from s2s_vla.objectives import command_roughness
from s2s_vla.storage import jsonable


def run_episode(runtime, method, selector, environment_seed, policy_seed):
    try:
        return _run_episode(runtime, method, selector, environment_seed, policy_seed)
    except Exception:
        runtime.environment.stop()
        raise


def _run_episode(runtime, method, selector, environment_seed, policy_seed):
    environment, policy = runtime.environment, runtime.policy
    observation = environment.reset(int(environment_seed))
    previous = np.asarray(environment.initial_action, dtype=np.float32).copy()
    if previous.shape != (method.action_dim,):
        raise ValueError("Platform initial command has incorrect dimensions")
    remaining = method.max_steps
    rng = np.random.default_rng(policy_seed)
    records = []
    success = False
    anomaly = False
    stopped_invalid = False
    roughness_sum = 0.0
    while remaining > 0:
        context = Context(observation, environment.instruction, previous, remaining)
        start = time.perf_counter()
        proposal = policy.propose(context, method.candidates, rng)
        proposal.validate(method, method.candidates)
        prefixes = proposal.chunks[:, :min(method.prefix_length, remaining)]
        indices = valid_indices(environment, prefixes)
        if len(indices) == 0:
            environment.stop()
            stopped_invalid = True
            break
        decision = selector.select(proposal.features, prefixes[indices], previous, remaining)
        latency = time.perf_counter() - start
        selected = int(indices[decision.index])
        prefix = prefixes[selected]
        transition, consumed, new_previous, prefix_anomaly = execute_prefix(environment, prefix, remaining, previous)
        realized_roughness = float(command_roughness(prefix[None, :consumed], previous, method)[0])
        roughness_sum += realized_roughness * consumed
        records.append({"remaining_before": remaining, "executed_steps": consumed, "selected_index": selected, "valid_indices": indices.tolist(), "probabilities": jsonable(decision.probabilities), "scores": decision.scores.tolist(), "roughness": decision.roughness.tolist(), "anomaly": prefix_anomaly, "decision_latency_seconds": latency})
        remaining -= consumed
        previous = new_previous
        observation = transition.observation
        anomaly = anomaly or prefix_anomaly
        success = transition.success
        if transition.done:
            break
    steps = method.max_steps - remaining
    return {"environment_seed": int(environment_seed), "policy_seed": int(policy_seed), "success": bool(success), "anomaly": bool(anomaly), "stopped_invalid": stopped_invalid, "steps": steps, "mean_command_roughness": roughness_sum / max(steps, 1), "decisions": records}


def aggregate_episodes(episodes):
    if not episodes:
        raise ValueError("At least one evaluation episode is required")
    latencies = [decision["decision_latency_seconds"] for episode in episodes for decision in episode["decisions"]]
    return {"episodes": len(episodes), "success_rate": float(np.mean([x["success"] for x in episodes])), "anomaly_rate": float(np.mean([x["anomaly"] for x in episodes])), "mean_steps": float(np.mean([x["steps"] for x in episodes])), "mean_command_roughness": float(np.mean([x["mean_command_roughness"] for x in episodes])), "invalid_stop_rate": float(np.mean([x["stopped_invalid"] for x in episodes])), "mean_decision_latency_seconds": float(np.mean(latencies)) if latencies else 0.0}
