# SC-ERL: Uncertainty-Gated Surrogate-Assisted Evolutionary Reinforcement Learning

Master's research repository implementing a modular hybrid framework combining deep reinforcement learning (RL), evolutionary algorithms (EA), and uncertainty-guided surrogate optimization on continuous control tasks (MuJoCo).

## Overview

The central contribution is **SC-ERL** — a novel algorithm that gates genetic algorithm fitness evaluations using a learned critic as a surrogate. Instead of running every candidate policy through slow environment rollouts, the surrogate estimates fitness at near-zero cost. Epistemic uncertainty determines when the surrogate is trusted versus when a real rollout is triggered.

Baselines included: DDPG, TD3, PPO, SAC, CrossQ (all via Stable-Baselines3 / PyTorch), and canonical ERL (configured with distilled crossover).

---

## Repository Layout

```
ue_sc_erl/
├── entry_point.py                  # Hydra experiment launcher & auto device selection
├── pyproject.toml                  # Python 3.12 dependencies (uv)
├── Taskfile.yml                    # CLI task orchestrator
├── configs/
│   ├── config.yaml                 # Global defaults (seed, device, wandb, env)
│   └── algorithm/                  # Per-algorithm Hydra configs
│       ├── ddpg.yaml, ppo.yaml, td3.yaml, erl.yaml
│       ├── sc_erl.yaml             # Surrogate parameters (beta, dropout_p, omega, k)
│       ├── erl/<env>.yaml          # Environment-specific ERL overrides
│       └── sc_erl/<env>.yaml       # Environment-specific SC-ERL overrides
├── src/
│   ├── algorithms/
│   │   ├── DDPG/, PPO/, TD3/       # Classical continuous control baselines (PyTorch)
│   │   ├── SAC/                    # SAC via Stable-Baselines3 (PyTorch)
│   │   ├── CrossQ/                 # CrossQ via sb3-contrib — batch-norm critic (PyTorch)
│   │   ├── ERL/                    # Canonical ERL (DDPG + GA with shared replay buffer)
│   │   └── SC_ERL/                 # Novel uncertainty-gated surrogate-assisted ERL
│   ├── common/
│   │   ├── surrogate_controller.py # Epistemic uncertainty gating & Q-value normalization
│   │   ├── utils.py                # Huber loss, soft-updates, parameter flattening
│   │   ├── reply_buffer.py         # Experience replay (Transition & Buffer)
│   │   └── wandb_logger.py         # WandB telemetry interface
│   └── modules/
│       ├── deep_modules.py         # Actor, Critic, StochasticActor, EvidentialCritic
│       ├── ensemble_module.py      # Multi-critic ensemble with prediction std
│       ├── evolution_module.py     # Elite preservation, selection, sparse mutation
│       └── mc_dropout_module.py    # MC Dropout runner for epistemic variance
└── plots_and_tests/
    ├── generate_results.py         # Full reporting pipeline (plots + stats + LaTeX)
    └── download_results.py         # Download metrics from WandB
```

---

## Setup

Requires Python 3.12 and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Running Experiments

### Single run

```bash
task run ALGO=sc_erl CLI_ARGS="env.id=HalfCheetah-v5 surrogate.mode=dropout"
```

Supported `ALGO` values: `sc_erl`, `erl`, `td3`, `ddpg`, `ppo`, `sac`, `crossq`.

SC-ERL `surrogate.mode` options: `dropout`, `ensemble`, `evidential`, `random`.

### RL backbone (`ddpg` / `td3` / `crossq`)

`sc_erl` and `erl` pick their RL learner via `backbone`. The `crossq` backbone is a **native** CrossQ learner (BatchRenorm critic, no target networks — not the standalone SB3 `crossq` baseline). It works with all four surrogate modes.

**MuJoCo:**

```bash
# ERL with CrossQ backbone
task run ALGO=erl CLI_ARGS="backbone=crossq env.id=HalfCheetah-v5 eval_env.id=HalfCheetah-v5"

# SC-ERL with CrossQ backbone (choose surrogate.mode: random | dropout | ensemble | evidential)
task run ALGO=sc_erl CLI_ARGS="backbone=crossq surrogate.mode=ensemble env.id=HalfCheetah-v5 eval_env.id=HalfCheetah-v5"
```

**DMC dog** (backend auto-detected from the `dm_control/` prefix; set explicitly to be safe):

```bash
# ERL with CrossQ backbone on DMC
task run ALGO=erl CLI_ARGS="backbone=crossq env.id=dm_control/dog-stand-v0 eval_env.id=dm_control/dog-stand-v0 env.backend=fancy_gym eval_env.backend=fancy_gym"

# SC-ERL with CrossQ backbone on DMC
task run ALGO=sc_erl CLI_ARGS="backbone=crossq surrogate.mode=ensemble env.id=dm_control/dog-stand-v0 eval_env.id=dm_control/dog-stand-v0 env.backend=fancy_gym eval_env.backend=fancy_gym"
```

Tune BatchRenorm momentum with `rl.bn_momentum` (default `0.01`).

### Full experiment matrix (5 envs × 8 algo variants × 5 seeds, parallelized)

```bash
task run-parallel
```

Control parallelism with `PARALLEL=N` (default: 4).

### Single DMC dog run

```bash
task run-dmc ENV=dm_control/dog-stand-v0 MODE=ensemble SEED=0
```

### Full DMC matrix (5 dog tasks × 4 surrogate modes × 5 seeds, parallelized)

```bash
task run-parallel-dmc
```

### SLURM (cluster)

**MuJoCo** — 50 tasks per env (10 algos × 5 seeds):
```bash
TARGET_ENV=HalfCheetah-v5 sbatch --array=0-49 slurm_run_array.sh
```

**DMC dog** — 20 tasks per env (4 SC-ERL modes × 5 seeds). Same script, backend auto-detected from the `dm_control/` prefix:
```bash
TARGET_ENV=dm_control/dog-stand-v0 sbatch --array=0-19 slurm_run_array.sh
TARGET_ENV=dm_control/dog-walk-v0  sbatch --array=0-19 slurm_run_array.sh
TARGET_ENV=dm_control/dog-trot-v0  sbatch --array=0-19 slurm_run_array.sh
TARGET_ENV=dm_control/dog-run-v0   sbatch --array=0-19 slurm_run_array.sh
TARGET_ENV=dm_control/dog-fetch-v0 sbatch --array=0-19 slurm_run_array.sh
```

### Reports

```bash
task report
```

Runs `plots_and_tests/generate_results.py` — compiles WandB metrics, statistical tests, and PDF report.

### Clean

```bash
task clean
```

---

## Surrogate Gating Logic

SC-ERL evaluates whether each candidate policy needs a real rollout or can be scored cheaply via the critic surrogate:

1. Compute epistemic uncertainty `σ_Q(πᵢ)` for every individual.
2. If `σ_Q(πᵢ)` exceeds the population's 75th-percentile threshold, **or** a random ε-coin flip fires (ε=0.10) → real environment rollout.
3. Otherwise → surrogate fitness via Lower Confidence Bound: `f_LCB = μ_Q − β·σ_Q`.

### Uncertainty methods

| Mode | Mechanism |
|------|-----------|
| `dropout` | T MC Dropout forward passes; empirical variance across passes |
| `ensemble` | N independent critic heads; std across predictions |
| `evidential` | Single forward pass; analytic NIG epistemic variance `β/(v(α−1))` |
| `random` | Probabilistic coin-flip baseline (no uncertainty estimation) |

---

## Configuration Reference

Key parameters in `configs/algorithm/sc_erl.yaml`:

| Parameter | Description |
|-----------|-------------|
| `backbone` | RL learner: `ddpg` (default), `td3`, or `crossq` (native BatchRenorm CrossQ) |
| `rl.bn_momentum` | BatchRenorm momentum for the `crossq` backbone (default: `0.01`) |
| `surrogate.mode` | Uncertainty method: `dropout`, `ensemble`, `evidential`, `random` |
| `surrogate.beta` | LCB penalty weight (higher → more real rollouts) |
| `surrogate.omega` | Percentile threshold for gating (default: 75) |
| `surrogate.k` | Replay buffer slice size for surrogate evaluation |
| `surrogate.dropout_p` | Dropout probability for MC Dropout mode |
| `surrogate.mc_samples` | Number of MC forward passes (T) |
| `surrogate.k_ensembles` | Number of critic heads for ensemble mode |
| `evolution.mutation_std` | Gaussian mutation standard deviation |
| `evolution.mutation_prob` | Fraction of parameters mutated per individual |
| `evolution.elite_ratio` | Fraction of top individuals preserved each generation |
| `evolution.rl_injection_interval` | Steps between RL actor → GA population injections |

Global config (`configs/config.yaml`): `seed`, `device` (`auto`/`cuda`/`mps`/`cpu`), `n_steps`, `wandb.*`.

---

## Additional Baselines: SAC and CrossQ

### SAC (Stable-Baselines3 / PyTorch)

A thin wrapper around [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) SAC. Uses the same WandB callback and eval loop as the rest of the framework. No extra setup beyond `uv sync`.

```bash
task run ALGO=sac CLI_ARGS="env.id=HalfCheetah-v5"
```

Key config knobs (`configs/algorithm/sac.yaml`): `rl.learning_rate`, `rl.ent_coef` (`auto` or float), `warmup.warmup_steps`.

### CrossQ (sb3-contrib / PyTorch)

A thin wrapper around [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) CrossQ — a batch-normalised critic algorithm that is highly sample-efficient. Runs on PyTorch, uses the same `device` as all other algorithms.

```bash
task run ALGO=crossq CLI_ARGS="env.id=HalfCheetah-v5"
```

Key config knobs (`configs/algorithm/crossq.yaml`): `rl.learning_rate`, `rl.gradient_steps`, `rl.policy_delay`.

---

## Environments

### MuJoCo v5 (default)

`HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5`.

### DeepMind Control Suite (via shimmy)

Five DMC dog locomotion tasks are supported through [shimmy](https://github.com/Farama-Foundation/Shimmy), which registers `dm_control` environments into the Gymnasium registry:

| Task | Env ID |
|------|--------|
| Stand | `dm_control/dog-stand-v0` |
| Walk | `dm_control/dog-walk-v0` |
| Trot | `dm_control/dog-trot-v0` |
| Run | `dm_control/dog-run-v0` |
| Fetch | `dm_control/dog-fetch-v0` |

The environment backend is selected via `env.backend`:
- `auto` (default) — detects `dm_control/`, `fancy/`, or `metaworld/` prefixes automatically.
- `mujoco` — force MuJoCo/Gymnasium without shimmy import.
- `fancy_gym` — force shimmy import regardless of env ID.

Environment-specific configs under `configs/algorithm/sc_erl/` are auto-loaded and already set `backend: fancy_gym` for all dog tasks.

