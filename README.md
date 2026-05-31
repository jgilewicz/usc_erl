# Evolutionary + Reinforcement Learning (SC-ERL)

This repository contains implementations of hybrid evolutionary and reinforcement-learning algorithms, along with tools to run experiments and perform hyperparameter optimization using Optuna. The project originates from master's research and focuses on comparative experiments (ERL, SC-ERL, DDPG, TD3, PPO) and the study of predictive uncertainty methods (dropout, ensemble, evidential).

Repository layout
- `entry_point.py` — main Hydra-based training launcher.
- `src/` — algorithm implementations, shared modules, surrogate controller, and utilities.
- `configs/` — Hydra configuration files. Per-algorithm and per-environment YAMLs are stored under `configs/algorithm/`.
- `src/optimization/optuna_tune.py` — runs a single Optuna study for one algorithm/environment.
- `src/optimization/run_optuna_grid.py` — helper to run Optuna studies across algorithm/environment combinations.
- `outputs/` — experiment outputs, best-params files and saved configs.

Requirements
- Python 3.10+ (recommended).
- Project dependencies are declared in `pyproject.toml`. Create and activate a virtual environment before installing dependencies.

Quick start
1. Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2. Run a single training run using the Taskfile:

```bash
task run ALGO=sc_erl CLI_ARGS="env.id=HalfCheetah-v5"
```

3. Run an Optuna HPO study for a single algorithm/environment using the Taskfile (example for the SC-ERL evidential variant):

```bash
task tune ALGO=sc_erl CLI_ARGS="name=sc_erl_evidential env.id=FetchPush-v4 eval_env.id=FetchPush-v4"
```

4. Run the Optuna grid runner to execute multiple studies using the Taskfile:

```bash
task tune-all
```

Configuration and tuning
- All configuration files are in `configs/`. When an Optuna study finishes, a per-environment algorithm config is saved into `configs/algorithm/<algo>/<algo>_<env>.yaml`.
- Optuna settings (study name, `n_trials`, `storage`) are defined in `configs/tune.yaml` (for example `storage: "sqlite:///optuna_hpo.db"`).
- The SC-ERL family supports variants: `sc_erl`, `sc_erl_dropout`, `sc_erl_ensemble`, and `sc_erl_evidential`. During HPO the implementation uses parameter-freezing rules so that only variant-specific parameters are searched.

Outputs
- Experiment artifacts and best-parameters are written to `outputs/` (e.g. `optuna_best_<algo>_<env>.yaml`).

Practical tips
- Use short Optuna runs for validation (`n_trials=3`) before launching full studies.
- Ensure adequate memory and GPU resources for long experiments.

Support and next steps
- This repository is part of private research. I can help with experiment execution, tuning spaces, or preparing reproducibility scripts on request.

---
If you want, I can also provide an English README with example cluster job scripts or generate a `requirements.txt` from the current environment.
