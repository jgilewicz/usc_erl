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

`surrogate.gate_mode` selects how much environment interaction each candidate policy gets:

- `h_bootstrap` (default) — uncertainty picks **how long** to evaluate a candidate, not **whether** to evaluate it. Every individual is rolled out for a per-individual number of steps `H`; the fitness is the partial return plus a calibrated critic tail (below). See [Fitness by H-bootstrap](#fitness-by-h-bootstrap).
- `topk` (baseline) — real-evaluate the `surrogate.rho` fraction of the population with the highest predicted uncertainty `σ_Q(πᵢ)`, plus a random ε-coin flip (ε=0.10) on the rest so the surrogate error can still be measured on an unbiased sample. `rho` is adapted every `surrogate.e_hat_window` generations toward a target normalized surrogate error `surrogate.e_star`.
- `relative` (ablation) — real-evaluate if `σ_Q(πᵢ)` exceeds a MAD-based threshold (`surrogate.mad_k`), **or** the ε-coin flip fires.

For `topk` / `relative`, a gated-out individual gets surrogate fitness via Lower Confidence Bound: `f_LCB = μ_Q − β·σ_Q`, squashed via `surrogate.fitness_norm` (`tanh`, default, or `clip`).

### Fitness by H-bootstrap

The binary gate measures `σ_Q` on **buffer states**, but the error it needs to predict is dominated by state-distribution mismatch — the critic does not know where a mutant will go, so `gate_auc_u` sits at 0.3–0.5. Rolling the candidate for `H` steps puts every state used in the estimate on the candidate's own trajectory, and leaves `σ` measuring exactly what remains: critic error at `s_H`.

```
alive = 1 if the episode did not end before H

F̂ = Σ_{t<H} r_t  +  a·(T − H)(1 − γ)·Q̄(s_H, π(s_H))  +  b·alive     # return scale
σ  = |a|·(T − H)(1 − γ)·std_i Q_i(s_H, π(s_H))·alive                  # return scale
fitness = F̂ − β·σ                                                     # β adaptive, as before
```

`(a, b)` is a least-squares fit of `remaining return ≈ a·feature + b` on individuals rolled out in full (the ε-subset the gate already pays for), harvested at intermediate horizons along their trajectories. Without it the raw `(1 − γ)·Q` tail relies on the critic's absolute scale, which is off by up to an order of magnitude, and the ranking collapses onto `Q(s_H)`. `H = end of episode` gives `σ = 0` and `F̂` = the real return, so a full evaluation is a special case rather than a separate branch.

`H` is chosen per individual by a stopping rule: after each `surrogate.h_chunk` slice, roll on while the elite/non-elite decision is still in doubt —

```
p = Φ( −|F̂ − F_cut| / σ )        stop when p < surrogate.p_stop
```

`F_cut` is the elite threshold (previous generation's, replaced by the current one's quantile once half the population is evaluated). The per-generation budget is `ρ · N · L_ema` (`L_ema` = EMA of observed full-episode length) — the same environment-step cost the binary gate pays at the same `ρ`, and an upper bound rather than a target: each individual is capped at an equal share of what is left, so steps given back by early stops accumulate for the individuals near the threshold instead of being spent for their own sake. `surrogate.h_alloc=uniform` spends the same budget evenly (`H = B/N`) instead — the built-in "uncertainty or just budget?" ablation, and what `surrogate.mode=random` falls back to since it has no `σ`.

Design, offline simulation results, and the differences between the online implementation and that simulation: `H_bootstrap.md`.

### Uncertainty methods

| Mode | Mechanism |
|------|-----------|
| `dropout` | T MC Dropout forward passes; empirical variance across passes |
| `ensemble` | N independent critic heads; std across predictions |
| `evidential` | Single forward pass; analytic NIG epistemic variance `β/(v(α−1))` |
| `random` | Probabilistic coin-flip baseline (no uncertainty estimation) |

### Known limitation: the ensemble rank inversion has no mechanistic explanation yet

`surrogate.mode=ensemble` **inverts rank** across task classes — it beats `random` on DMC dog locomotion (Nemenyi rank 3.40 vs 5.60) but loses to it on classic MuJoCo (5.40 vs 4.00). We pre-registered two candidate explanations and tested both on a matched budget (`dog-walk` vs `Swimmer-v5`, 3 seeds, 500k steps, checkpointed at 20/60/100% of training) before looking at results — both were **refuted, not marginally**:

| criterion | threshold to confirm | threshold to refute | result |
|---|---|---|---|
| action-sensitivity ratio (dog / Swimmer) | > 3× | < 2× | 0.85× |
| `sigma_cv` (dog vs Swimmer) | dog > 5%, Swimmer < 2% | comparable | 2.00% vs 2.18% |

One further observation from the same runs, independent of the gate itself: **`gate_auc_d`** (behavioral-distance-augmented gate score, swept over `λ ∈ {0, 0.25, 0.5, 1, 2}`) never beat plain uncertainty-based gating on AUC — distance carries no signal the ensemble doesn't already have. Removed from the codebase (`_gate_score`, `surrogate.lam`); the analysis script (`scripts/analyze_ensemble_checkpoints.py`) still reports `corr_sigma_d` if this needs revisiting.

Reproduction script and pre-registered pass/fail criteria: `scripts/analyze_ensemble_checkpoints.py`. Up to three attempts at a mechanistic explanation were budgeted; the first was unambiguous enough (not a marginal miss on either criterion) that the remaining two weren't spent. The mechanism is treated as an open problem for now — the paper reports the ranking gap without a causal explanation.

The `h_bootstrap` gate mode is the response to the diagnosis behind this limitation: if `σ` on buffer states cannot predict which mutant needs a rollout, stop asking it that question and let it set the rollout length instead. It does not explain the ensemble inversion; it removes the binary decision the inversion was measured on.

### Retracted: "uniform-action disagreement is lower than actor-action disagreement" is not evidence against the ensemble

An earlier version of this section reported `sigma_ratio_ood < 1` — lower ensemble disagreement at uniform random actions than at the TD3 actor's actions — and read it as the ensemble failing to be epistemic. The measurement was real, the reading was wrong: neither probe point is an out-of-distribution reference for the critic, so the ratio says nothing about epistemic validity in either direction. `scripts/probe_ensemble_uncertainty.py` replaces it with probes whose relation to the training data is known. Disagreement relative to the buffer's own `(s, a)` pairs, seed 0, 100k-step runs:

| probe | Swimmer 20k | Swimmer 60k | Swimmer 100k | Hopper 60k | Hopper 100k |
|---|---|---|---|---|---|
| buffer `(s, a_buf)` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| uniform action | 1.12 | 1.09 | 1.08 | 1.15 | 0.93 |
| actor action | 1.40 | 1.32 | 1.30 | 0.90 | 0.69 |
| action ×2 (out of bounds) | 1.29 | 1.19 | 1.16 | 1.08 | 0.96 |
| action ×3 (out of bounds) | 1.56 | 1.32 | 1.32 | 1.15 | 1.07 |
| state ×2 | 2.04 | 1.52 | 2.44 | 1.27 | 1.26 |
| state ×5 | 3.48 | 2.11 | 3.93 | 1.73 | 1.72 |
| actor action after gradient *descent* on μ_Q | 1.09 | 1.06 | 1.06 | 0.90 | 1.39 |
| old `sigma_ratio_ood` (uniform / actor) | 0.80 | 0.83 | 0.83 | 1.27 | 1.36 |

- **The ensemble is a working deep ensemble.** Disagreement grows monotonically with distance from the data on inputs that are genuinely OOD — scaled states in every checkpoint, out-of-bound actions on Swimmer — and the implementation shows the same monotone ordering on a synthetic supervised regression (train points < uniform < argmax of the mean < out-of-bounds < scaled states).
- **Uniform actions are not OOD for this critic.** Warmup writes uniform actions into the buffer (25k on Hopper, 100k on dog-walk; a 1M buffer never evicts them within 500k steps), GA mutants and exploration noise widen the action marginal further, and `|a| ≈ 0.5` is *closer* to the buffer's action marginal than the actor's saturated `|a| ≈ 0.9–0.96`.
- **The actor's action is not an in-distribution reference either.** It is the argmax of the ensemble mean — an adversarially chosen point where disagreement is inflated, an effect that reproduces in pure supervised regression with no RL — and simultaneously the point where TD-target supervision and near-actor rollouts concentrate, which deflates it. Which effect wins depends on environment and phase: on Swimmer the actor sits at out-of-bounds-level disagreement and gradient descent on μ_Q returns it to the uniform level; on Hopper it sits below the buffer floor and descent *raises* it. The old ratio flips sign accordingly.
- **Caveat for absolute thresholds:** `LayerNorm` in the critic bounds extrapolation, so the OOD signal is compressed relative to a norm-free critic (on the synthetic task, ×5 states raise disagreement to about 3× the training floor with `LayerNorm` versus about 30× without it). The rank-based `topk` gate is unaffected; the MAD-threshold `relative` gate is not.

The validity test that matters for the gate is whether disagreement at *population* actions predicts surrogate error. That is logged live as `gate_auc_u` / `gate_spearman` (`SurrogateController._update_gate_quality`) and should be reported per environment instead of any single-point probe.

---

## Configuration Reference

Key parameters in `configs/algorithm/sc_erl.yaml`:

| Parameter | Description |
|-----------|-------------|
| `rl.policy_noise` / `rl.noise_clip` / `rl.policy_delay` | TD3 target-smoothing noise, clip, and actor update delay |
| `surrogate.mode` | Uncertainty method: `dropout`, `ensemble`, `evidential`, `random` |
| `surrogate.beta` | LCB penalty weight (higher → more real rollouts) |
| `surrogate.omega` | Real-vs-surrogate coin-flip threshold, `random` mode only |
| `surrogate.gate_mode` | Evaluation strategy: `h_bootstrap` (default), `topk` (rank-based binary gate), `relative` (MAD-threshold ablation) |
| `surrogate.h_chunk` | Rollout slice Δ between stopping decisions; also the minimum `H` per individual |
| `surrogate.h_alloc` | Budget allocation: `adaptive` (uncertainty-driven stopping) or `uniform` (equal `H`) |
| `surrogate.p_stop` | Stop once `P(wrong side of the elite cut) < p_stop` |
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

