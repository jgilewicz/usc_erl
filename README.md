# SC-ERL: Uncertainty-Gated Surrogate-Assisted Evolutionary Reinforcement Learning

Master's research repository implementing a modular hybrid framework combining deep reinforcement learning (RL), evolutionary algorithms (EA), and uncertainty-guided surrogate optimization on continuous control tasks (MuJoCo).

## Overview

The central contribution is **SC-ERL** — a novel algorithm that gates genetic algorithm fitness evaluations using a learned critic as a surrogate. Instead of running every candidate policy through slow environment rollouts, the surrogate estimates fitness at near-zero cost. Epistemic uncertainty determines when the surrogate is trusted versus when a real rollout is triggered.

Baselines included: DDPG, TD3, PPO, and canonical ERL (configured with distilled crossover).

---

## Repository Layout

```
magisterka_evo_rl/
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
│   │   ├── DDPG/, PPO/, TD3/       # Classical continuous control baselines
│   │   ├── ERL/                    # Canonical ERL (DDPG + GA with shared replay buffer)
│   │   └── SC_ERL/                 # Novel uncertainty-gated surrogate-assisted ERL
│   ├── common/
│   │   ├── surrogate_controller.py # Epistemic uncertainty gating & Q-value normalization
│   │   ├── evolution_module.py     # Elite preservation, selection, sparse mutation
│   │   ├── utils.py                # Huber loss, soft-updates, parameter flattening
│   │   ├── reply_buffer.py         # Experience replay (Transition & Buffer)
│   │   └── wandb_logger.py         # WandB telemetry interface
│   └── modules/
│       ├── deep_modules.py         # Actor, Critic, StochasticActor, EvidentialCritic
│       ├── ensemble_module.py      # Multi-critic ensemble with prediction std
│       └── mc_dropout_module.py    # MC Dropout runner for epistemic variance
└── plots_and_tests/
    ├── generate_results.py         # Full reporting pipeline (plots + stats + PDF)
    ├── process_results.py          # Raw data compiler and curve generator
    ├── statistical_tests.py        # Welch's t-test, Shapiro-Wilk, Mann-Whitney U
    ├── generate_report.py          # Markdown summary
    └── generate_pdf.py             # FPDF2 PDF builder
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

Supported `ALGO` values: `sc_erl`, `erl`, `td3`, `ddpg`, `ppo`.

SC-ERL `surrogate.mode` options: `dropout`, `ensemble`, `evidential`, `random`.

### Full experiment matrix (5 envs × 8 algo variants × 5 seeds, parallelized)

```bash
task run-parallel
```

Control parallelism with `PARALLEL=N` (default: 4).

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

## Environments

All experiments target MuJoCo v5 continuous control:
`HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5`.

MetaWorld `-v2` environments are supported via automatic remapping to `-v3-goal-observable` (requires `metaworld` installed).
