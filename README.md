# Evolutionary + Reinforcement Learning (SC-ERL)

This repository contains implementations of hybrid evolutionary and reinforcement-learning algorithms, along with tools to run experiments. The project originates from master's research and focuses on comparative experiments (ERL, SC-ERL, DDPG, TD3, PPO) and the study of predictive uncertainty methods (dropout, ensemble, evidential). The ERL baseline is configured with distilled crossover by default so the comparison reflects the newer distillation-based evolutionary variants rather than the older parameter-crossover setup.

Repository layout
- `entry_point.py` — main Hydra-based training launcher.
- `src/` — algorithm implementations, shared modules, surrogate controller, and utilities.
- `configs/` — Hydra configuration files. Per-algorithm and per-environment YAMLs are stored under `configs/algorithm/`.
- `outputs/` — experiment outputs and saved configs.

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



Configuration
- All configuration files are in `configs/`. Per-algorithm and per-environment configuration files are loaded from `configs/algorithm/<algo>/`.
- The ERL baseline uses distilled crossover by default, matching the newer evolutionary setup built around distilled crossover, proximal mutation, and individual-based control.
- The SC-ERL family supports variants: `sc_erl`, `sc_erl_dropout`, `sc_erl_ensemble`, and `sc_erl_evidential`.

Outputs
- Experiment artifacts are written to `outputs/`.

Practical tips
- Ensure adequate memory and GPU resources for long experiments.

Support and next steps
- This repository is part of private research. I can help with experiment execution, tuning spaces, or preparing reproducibility scripts on request.
