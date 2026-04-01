# Copilot Instructions

## Project Overview

Udacity course exercises for building reproducible ML workflows. Five lessons (16 exercises total) progressively build a full MLOps pipeline using MLflow, Hydra, Weights & Biases (wandb), scikit-learn, and PyTorch. Each exercise has a `starter/` (incomplete) and `solution/` directory.

## Environment Setup

```bash
# For IDE/exploration support
python3.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# For running exercises (required — uses conda isolation)
mlflow run .

# Docker environment (Jupyter :8888, MLflow tracking server :5000)
docker compose up
```

## Running Exercises

```bash
# Run an exercise
cd lesson-N-*/exercises/exercise_X/starter/
mlflow run .

# Override Hydra config values
mlflow run . -P hydra_options="main.project_name=my_project random_forest.max_depth=5"

# Run data validation tests (Lesson 3+)
pytest test_data.py -v

# Run a single test
pytest test_data.py::test_column_names -v
```

## Architecture

Each exercise follows a **modular multi-step pipeline** pattern:

- `main.py` — Hydra-decorated orchestrator that chains MLflow component runs
- `config.yaml` — Hydra config defining steps to execute and hyperparameters
- `conda.yml` — Isolated conda environment for the exercise
- `MLproject` — MLflow entry point; all components accept `hydra_options` parameter

Sub-components (e.g., `download/`, `preprocess/`, `random_forest/`, `evaluate/`) each have their own `MLproject` + `conda.yml` and are invoked by the top-level `main.py` via `mlflow.run()`.

**Full pipeline stages** (Lesson 5): download → preprocess → check_data → segregate → random_forest → evaluate

## Key Conventions

### `main.py` Orchestrator Pattern

```python
import mlflow, hydra, os
from omegaconf import DictConfig

@hydra.main(config_name='config')
def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
    root_path = hydra.utils.get_original_cwd()

    if "step_name" in config["main"]["execute_steps"]:
        mlflow.run(
            os.path.join(root_path, "component_dir"),
            "main",
            parameters={"param": config["section"]["param"]}
        )
```

### `config.yaml` Structure

```yaml
main:
  project_name: exercise_N
  experiment_name: dev
  execute_steps:
    - download
    - preprocess
    - check_data
    - segregate
    - random_forest
    - evaluate
random_forest:
  n_estimators: 100
  criterion: gini
  max_depth: null
```

### `conda.yml` Pattern

All conda environments use Python 3.13, `conda-forge` channel, and install `wandb` and `omegaconf` via pip (not conda):

```yaml
name: ex_N
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.13
  - scikit-learn=1.7.2
  - mlflow=3.2.0
  - hydra-core=1.3.2
  - pip=24.3.1
  - pip:
    - wandb==0.24.0
    - omegaconf==2.3.0
```

### Data Validation Tests (Lesson 3+)

Test files use pytest with a `data` fixture defined in `conftest.py` that loads the wandb artifact:

```python
# conftest.py provides: data (pd.DataFrame), ref_data, kl_threshold
def test_column_names(data):
    expected_columns = [...]
    assert list(data.columns) == expected_columns
```

### wandb Artifact References

Artifacts are referenced as `"artifact_name:version"` strings (e.g., `"preprocessed_data.csv:latest"`) in `config.yaml` and passed as MLflow parameters.

## Tech Stack Versions

| Package | Version |
|---|---|
| Python | 3.13 |
| mlflow | 3.2.0 (exercises) / 3.8.1 (Docker) |
| wandb | 0.24.0 |
| hydra-core | 1.3.2 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.2 |
| pytest | 8.4.2 |
