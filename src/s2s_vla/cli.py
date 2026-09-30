import argparse
from pathlib import Path
import json
import torch
from s2s_vla.config import ExperimentConfig


def parser():
    root = argparse.ArgumentParser(prog="s2s-vla")
    root.add_argument("--threads", type=int, default=1)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("collect", "train", "select-preferences", "evaluate", "demo", "serve-policy"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name != "serve-policy":
            command.add_argument("--output", required=True)
        if name in ("train", "select-preferences", "evaluate"):
            command.add_argument("--data", required=True)
        if name in ("train", "select-preferences", "evaluate", "demo"):
            command.add_argument("--device", default="cpu")
        if name in ("select-preferences", "evaluate"):
            command.add_argument("--checkpoint", required=name != "evaluate")
            command.add_argument("--calibration", required=name != "evaluate")
        if name == "evaluate":
            command.add_argument("--selection")
            command.add_argument("--strategy", choices=["s2s", "first", "success", "weighted_sum"], default="s2s")
            command.add_argument("--predictor-url")
        if name == "serve-policy":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=3200)
    command = commands.add_parser("calibrate")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--data", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--device", default="cpu")
    command = commands.add_parser("serve-evaluator")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--calibration", required=True)
    command.add_argument("--device", default="cpu")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=3100)
    return root


def run_demo(config, output, device):
    from s2s_vla.calibration import calibrate
    from s2s_vla.collection import collect
    from s2s_vla.experiments import evaluate, select_preferences
    from s2s_vla.storage import write_json
    from s2s_vla.training import train
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Demo output must be empty")
    collect(config, output / "data")
    checkpoint = train(config, output / "data", output / "training", device)
    calibration = output / "calibration.json"
    selection = output / "selection.json"
    calibrate(checkpoint, output / "data", calibration, device)
    select_preferences(config, output / "data", checkpoint, calibration, selection, device)
    results = {}
    for strategy in ("first", "success", "weighted_sum", "s2s"):
        report = evaluate(config, output / "data", checkpoint, calibration, selection, output / f"test-{strategy}.json", device, strategy)
        results[strategy] = report["summary"]
    write_json(output / "summary.json", {"runtime": config.runtime_factory, "results": results, "scope": "example_pipeline_validation" if "adapters.example" in config.runtime_factory else "adapter_evaluation"})
    return results


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(arguments.threads)
    config = ExperimentConfig.load(arguments.config) if hasattr(arguments, "config") else None
    command = arguments.command
    if command == "collect":
        from s2s_vla.collection import collect
        result = collect(config, arguments.output)
        print(json.dumps({"records": len(result["records"]), "groups": len(result["groups"])}))
    elif command == "train":
        from s2s_vla.training import train
        print(train(config, arguments.data, arguments.output, arguments.device))
    elif command == "calibrate":
        from s2s_vla.calibration import calibrate
        result = calibrate(arguments.checkpoint, arguments.data, arguments.output, arguments.device)
        print(json.dumps({"temperatures": result["temperatures"]}))
    elif command == "select-preferences":
        from s2s_vla.experiments import select_preferences
        result = select_preferences(config, arguments.data, arguments.checkpoint, arguments.calibration, arguments.output, arguments.device)
        print(json.dumps({"weights": result["weights"]}))
    elif command == "evaluate":
        from s2s_vla.experiments import evaluate
        if arguments.strategy != "first" and (not arguments.selection or not arguments.predictor_url and not (arguments.checkpoint and arguments.calibration)):
            raise ValueError("Selection artifact and local checkpoint/calibration or predictor URL are required")
        result = evaluate(config, arguments.data, arguments.checkpoint, arguments.calibration, arguments.selection, arguments.output, arguments.device, arguments.strategy, arguments.predictor_url)
        print(json.dumps(result["summary"]))
    elif command == "demo":
        print(json.dumps(run_demo(config, arguments.output, arguments.device), indent=2))
    elif command.startswith("serve-"):
        import uvicorn
        from s2s_vla.serving import create_evaluator_app, create_policy_app
        app = create_policy_app(config) if command == "serve-policy" else create_evaluator_app(arguments.checkpoint, arguments.calibration, arguments.device)
        uvicorn.run(app, host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
