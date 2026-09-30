# Select to Succeed: Pareto-Consistent Action Selection for VLAs via Augmented Tchebycheff Scalarization

Dataset-specific code will be released as soon as possible.

## Running the Code

### Installation

Use Python 3.10 or later and run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[serve,test]"
```

### Run the Example

```bash
s2s-vla demo --config configs/example.json --output outputs/demo --device cpu
```

This runs data collection, evaluator training, probability calibration, preference selection, and evaluation in the included example environment. Use a new or empty output directory.

### Run Individual Stages

Collect data:

```bash
s2s-vla collect --config configs/example.json --output outputs/data
```

Train the evaluator:

```bash
s2s-vla train --config configs/example.json --data outputs/data --output outputs/training --device cpu
```

Calibrate the evaluator:

```bash
s2s-vla calibrate --checkpoint outputs/training/evaluator.pt --data outputs/data --output outputs/calibration.json --device cpu
```

Select preferences:

```bash
s2s-vla select-preferences --config configs/example.json --data outputs/data --checkpoint outputs/training/evaluator.pt --calibration outputs/calibration.json --output outputs/selection.json --device cpu
```

Evaluate:

```bash
s2s-vla evaluate --config configs/example.json --data outputs/data --checkpoint outputs/training/evaluator.pt --calibration outputs/calibration.json --selection outputs/selection.json --output outputs/test.json --device cpu
```

Use `--device cuda:0` with a compatible GPU and CUDA-enabled PyTorch installation.

### Run Tests

```bash
python -m pytest -q
```

## Repository Structure

| Path | Description |
| --- | --- |
| `configs/example.json` | Configuration for the runnable example. |
| `configs/robot.template.json` | Robot integration configuration template. |
| `src/s2s_vla/config.py` | Method and experiment settings. |
| `src/s2s_vla/interfaces.py` | Policy, environment, and runtime interfaces. |
| `src/s2s_vla/collection.py` | Offline branch data collection. |
| `src/s2s_vla/data.py` | Dataset loading and batching. |
| `src/s2s_vla/model.py` | Dual-head evaluator. |
| `src/s2s_vla/losses.py` | Training objectives. |
| `src/s2s_vla/training.py` | Evaluator training. |
| `src/s2s_vla/calibration.py` | Probability calibration. |
| `src/s2s_vla/objectives.py` | Command roughness and candidate scoring. |
| `src/s2s_vla/inference.py` | Candidate prediction and selection. |
| `src/s2s_vla/controller.py` | Receding-horizon execution. |
| `src/s2s_vla/experiments.py` | Preference selection and evaluation. |
| `src/s2s_vla/serving.py` | Policy and evaluator HTTP services. |
| `src/s2s_vla/adapters/` | Model and environment adapters. |
| `src/s2s_vla/cli.py` | Command-line interface. |
| `examples/policy_example.py` | Example connecting a local environment to an HTTP policy service. |
| `tests/` | Unit and integration tests. |
