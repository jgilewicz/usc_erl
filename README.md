# USC-ERL: Uncertainty-Gated Surrogate-Assisted Evolutionary Reinforcement Learning

ECML PKDD 2026 LFSM Wokshop research repository implementing a modular hybrid framework combining deep reinforcement learning (RL), evolutionary algorithms (EA), and uncertainty-guided surrogate optimization on continuous control tasks.

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
│   │   ├── DDPG/, TD3/, PPO/       # Thin Stable-Baselines3 wrappers (PyTorch)
│   │   ├── SAC/                    # SAC via Stable-Baselines3 (PyTorch)
│   │   ├── CrossQ/                 # CrossQ via sb3-contrib — batch-norm critic (PyTorch)
│   │   ├── ERL/                    # Canonical ERL (TD3 + GA with shared replay buffer)
│   │   └── SC_ERL/                 # Novel uncertainty-gated surrogate-assisted ERL
│   ├── common/
│   │   ├── surrogate_controller.py # Epistemic uncertainty gating & Q-value normalization
│   │   ├── sb3_callback.py         # Shared eval + WandB logging callback (all SB3 baselines)
│   │   ├── utils.py                # Huber loss, soft-updates, parameter flattening
│   │   ├── reply_buffer.py         # Experience replay (Transition & Buffer)
│   │   └── wandb_logger.py         # WandB telemetry interface
│   └── modules/
│       ├── deep_modules.py         # Actor, Critic, EvidentialCritic
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

MuJoCo v5 / DMC and MyoSuite pin incompatible `mujoco`/`gymnasium` versions, so MyoSuite lives in
its own venv directory (`.venv-myosuite`) rather than `.venv` — `task run` / `task run-myo` handle
syncing the right one automatically:

```bash
uv sync --extra mujoco-envs                                    # MuJoCo v5 + DMC → .venv
UV_PROJECT_ENVIRONMENT=.venv-myosuite uv sync --extra myosuite  # MyoSuite → .venv-myosuite
```

Or with pip (pick the matching dependency set from `pyproject.toml`'s `[project.optional-dependencies]`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mujoco-envs]"
```

---

## Running Experiments

### Single run

```bash
task run ALGO=sc_erl CLI_ARGS="env.id=HalfCheetah-v5 surrogate.mode=dropout"
```

Supported `ALGO` values: `sc_erl`, `erl`, `td3`, `ddpg`, `ppo`, `sac`, `crossq`.

SC-ERL `surrogate.mode` options: `dropout`, `ensemble`, `evidential`, `random`.

`sc_erl` and `erl` always update their RL actor/critic via TD3 (twin critics, delayed policy updates, target-action smoothing) — this was the best-performing gradient update in experiments, so there is no backbone selection.

### Single DMC dog run

```bash
task run ALGO=sc_erl CLI_ARGS="env.id=dm_control/dog-stand-v0 surrogate.mode=ensemble"
```

### Single MyoSuite run

MyoSuite pins an older `mujoco`/`gymnasium` than the MuJoCo v5/DMC stack, so it runs from a
separate venv (`.venv-myosuite`) — `task run-myo` syncs and activates it automatically:

```bash
task run-myo ALGO=sc_erl ENV=myoHandReachRandom-v0
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

**MyoSuite** — 50 tasks per env (10 algos × 5 seeds), backend auto-detected from the `myo` prefix.
Runs from a separate venv (`.venv-myosuite`) — build it once per cluster checkout:
```bash
UV_PROJECT_ENVIRONMENT=.venv-myosuite uv sync --extra myosuite
TARGET_ENV=myoElbowPose1D6MRandom-v0 sbatch --array=0-49 slurm_run_array.sh
TARGET_ENV=myoHandReachRandom-v0     sbatch --array=0-49 slurm_run_array.sh
TARGET_ENV=myoHandPenTwirlRandom-v0  sbatch --array=0-49 slurm_run_array.sh
TARGET_ENV=myoHandObjHoldRandom-v0   sbatch --array=0-49 slurm_run_array.sh
TARGET_ENV=myoLegWalk-v0             sbatch --array=0-49 slurm_run_array.sh
```

### Reports

```bash
task report
```

Runs `plots_and_tests/generate_results.py` — compiles WandB metrics, statistical tests, and PDF report.

#### Statistical methodology

Because each environment is compared across many algorithms, the per-environment significance tables run a **family** of up to `|proposed| × |baselines|` pairwise tests (3 SC-ERL variants × 7 baselines = 21). Each pair uses a Welch *t*-test when both groups pass a Shapiro–Wilk normality check, otherwise the Mann–Whitney *U* test.

To control the family-wise error rate under this many comparisons, all raw *p*-values within one environment's table are corrected with the **Holm–Bonferroni step-down procedure** (`holm_bonferroni` in `generate_results.py`). Holm–Bonferroni is uniformly more powerful than plain Bonferroni while still bounding the probability of *any* false positive at `α = 0.05`. The tables report both the raw *p* and the adjusted `p_adj`, and significance stars (`*/**/***`) are decided on `p_adj`. Cross-environment comparisons are handled separately by the Friedman omnibus test + Nemenyi critical-difference diagram.

### Clean

```bash
task clean
```

---

## Surrogate Gating Logic

SC-ERL evaluates whether each candidate policy needs a real rollout or can be scored cheaply via the critic surrogate. `surrogate.gate_mode` selects how the deterministic part of that decision is made:

- `topk` (default) — real-evaluate the `surrogate.rho` fraction of the population with the highest predicted uncertainty `σ_Q(πᵢ)`, plus a random ε-coin flip (ε=0.10) on the rest so the surrogate error can still be measured on an unbiased sample. `rho` is adapted every `surrogate.e_hat_window` generations toward a target normalized surrogate error `surrogate.e_star`.
- `relative` (ablation) — real-evaluate if `σ_Q(πᵢ)` exceeds a MAD-based threshold (`surrogate.mad_k`), **or** the ε-coin flip fires.

Otherwise → surrogate fitness via Lower Confidence Bound: `f_LCB = μ_Q − β·σ_Q`, squashed via `surrogate.fitness_norm` (`tanh`, default, or `clip`).

### Uncertainty methods

| Mode | Mechanism |
|------|-----------|
| `dropout` | T MC Dropout forward passes; empirical variance across passes |
| `ensemble` | N independent critic heads; std across predictions |
| `evidential` | Single forward pass; analytic NIG epistemic variance `β/(v(α−1))` |
| `random` | Probabilistic coin-flip baseline (no uncertainty estimation) |

### Known limitation: ensemble disagreement is not a validated epistemic signal

`surrogate.mode=ensemble` **inverts rank** across task classes — it beats `random` on DMC dog locomotion (Nemenyi rank 3.40 vs 5.60) but loses to it on classic MuJoCo (5.40 vs 4.00). We pre-registered two candidate explanations and tested both on a matched budget (`dog-walk` vs `Swimmer-v5`, 3 seeds, 500k steps, checkpointed at 20/60/100% of training) before looking at results — both were **refuted, not marginally**:

| criterion | threshold to confirm | threshold to refute | result |
|---|---|---|---|
| action-sensitivity ratio (dog / Swimmer) | > 3× | < 2× | 0.85× |
| `sigma_cv` (dog vs Swimmer) | dog > 5%, Swimmer < 2% | comparable | 2.00% vs 2.18% |

Two further observations from the same runs, independent of the gate itself:

- **`sigma_ratio_ood < 1` in every environment, every seed, every training phase.** The ensemble assigns *lower* disagreement to out-of-distribution actions (uniform random) than to the actor's own actions — the opposite of what an epistemic estimator should do. This isn't a property of one environment or one gate config; it looks like a property of critic-disagreement-on-(s,a) as an uncertainty signal in general, relevant to any disagreement-based method in RL, not just this gate.
- **`gate_auc_d`** (behavioral-distance-augmented gate score, swept over `λ ∈ {0, 0.25, 0.5, 1, 2}`) never beat plain uncertainty-based gating on AUC — distance carries no signal the ensemble doesn't already have. Removed from the codebase (`_gate_score`, `surrogate.lam`); the analysis script (`scripts/analyze_ensemble_checkpoints.py`) still reports `corr_sigma_d` if this needs revisiting.

Reproduction script and pre-registered pass/fail criteria: `scripts/analyze_ensemble_checkpoints.py`. Up to three attempts at a mechanistic explanation were budgeted; the first was unambiguous enough (not a marginal miss on either criterion) that the remaining two weren't spent. The mechanism is treated as an open problem for now — the paper reports the ranking gap without a causal explanation.

---

## Configuration Reference

Key parameters in `configs/algorithm/sc_erl.yaml`:

| Parameter | Description |
|-----------|-------------|
| `rl.policy_noise` / `rl.noise_clip` / `rl.policy_delay` | TD3 target-smoothing noise, clip, and actor update delay |
| `surrogate.mode` | Uncertainty method: `dropout`, `ensemble`, `evidential`, `random` |
| `surrogate.beta` | LCB penalty weight (higher → more real rollouts) |
| `surrogate.omega` | Real-vs-surrogate coin-flip threshold, `random` mode only |
| `surrogate.gate_mode` | Gating strategy: `topk` (rank-based, default) or `relative` (MAD-threshold ablation) |
| `surrogate.rho` / `rho_min` / `rho_max` / `rho_eta` | `topk` real-eval budget fraction and its adaptation bounds/rate |
| `surrogate.e_star` / `e_hat_window` | Target normalized surrogate error and the generation window `rho` is adapted over |
| `surrogate.fitness_norm` | Surrogate fitness squashing: `tanh` (default) or `clip` |
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

## Additional Baselines: DDPG, TD3, PPO, SAC, CrossQ

### DDPG, TD3, PPO (Stable-Baselines3 / PyTorch)

Thin wrappers around Stable-Baselines3's `DDPG`, `TD3`, and `PPO`. All three share the same
`EvalAndLogCallback` (`src/common/sb3_callback.py`) for periodic evaluation and WandB logging.

```bash
task run ALGO=ddpg CLI_ARGS="env.id=HalfCheetah-v5"
task run ALGO=td3 CLI_ARGS="env.id=HalfCheetah-v5"
task run ALGO=ppo CLI_ARGS="env.id=HalfCheetah-v5"
```

Key config knobs: `rl.learning_rate`, `rl.tau` (DDPG/TD3), `rl.exploration_noise_std` (DDPG/TD3),
`rl.policy_noise`/`rl.noise_clip`/`rl.policy_delay` (TD3), `rl.gae_lambda`/`rl.clip_param`/`rl.ppo_epochs` (PPO).

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
- `auto` (default) — detects `dm_control/`, `fancy/`, `metaworld/`, or `myo` prefixes automatically.
- `mujoco` — force MuJoCo/Gymnasium without shimmy import.
- `fancy_gym` — force shimmy import regardless of env ID.
- `myosuite` — force MyoSuite import regardless of env ID.

Environment-specific configs under `configs/algorithm/sc_erl/` are auto-loaded and already set `backend: fancy_gym` for all dog tasks.

### MyoSuite (musculoskeletal control)

Five [MyoSuite](https://myosuite.readthedocs.io/en/latest/suite.html) tasks are supported. MyoSuite
pins `mujoco<3.7`/`gymnasium<1.3`, incompatible with the `mujoco>=3.10`/`gymnasium>=1.3` stack used
for MuJoCo v5/DMC — it lives in its own venv directory, `.venv-myosuite` (`task run-myo` syncs and
uses it automatically). See [Setup](#setup).

| Task | Env ID |
|------|--------|
| Elbow pose (1D, 6 muscles) | `myoElbowPose1D6MRandom-v0` |
| Hand reach | `myoHandReachRandom-v0` |
| Hand pen twirl | `myoHandPenTwirlRandom-v0` |
| Hand object hold | `myoHandObjHoldRandom-v0` |
| Leg walk | `myoLegWalk-v0` |

```bash
task run-myo ALGO=sc_erl ENV=myoHandReachRandom-v0
```

Environment-specific configs under `configs/algorithm/sc_erl/` (e.g. `sc_erl_myoHandReachRandom-v0.yaml`) set `env.id`/`env.backend`; no surrogate/evolution hyperparameters have been tuned for these envs yet, so they inherit the global `sc_erl.yaml` defaults.

