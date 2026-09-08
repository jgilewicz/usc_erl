# CLAUDE.md — SC-ERL Developer Guide

This file is the authoritative reference for AI-assisted development on this codebase. Read it before making any changes.

---

## Project Summary

**SC-ERL** is a hybrid evolutionary + deep RL framework. A population of GA actors evolves alongside a TD3 RL agent sharing a replay buffer. The key novelty is a **surrogate controller** that uses epistemic uncertainty to gate whether each candidate policy needs a real environment rollout or can be scored cheaply via the critic.

Algorithms: `sc_erl` (4 surrogate modes), `erl`, `td3`, `ddpg`, `ppo`, `sac`, `crossq` (all SB3/PyTorch).

Both `sc_erl` and `erl` are updated exclusively via TD3 gradient steps (twin critics, delayed policy updates, target-action smoothing) — experiments showed TD3 outperforms the DDPG and native-CrossQ backbones that used to be selectable, so those were removed; there is no `backbone` parameter anymore.
Environments:
- **MuJoCo v5**: `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5`.
- **DMC via fancy_gym**: `dm_control/dog-{stand,walk,trot,run,fetch}-v0`.
- **MyoSuite**: `myoElbowPose1D6MRandom-v0`, `myoHandReachRandom-v0`, `myoHandPenTwirlRandom-v0`, `myoHandObjHoldRandom-v0`, `myoLegWalk-v0`. Runs from a separate `.venv-myosuite` directory (`task run-myo`) — see [Dependency Extras](#dependency-extras-mujoco-envs-vs-myosuite) below.

---

## Key Source Files

| File | Role |
|------|------|
| `entry_point.py` | Hydra launcher; routes config → algorithm constructor |
| `src/algorithms/SC_ERL/sc_erl.py` | Main SC-ERL training loop |
| `src/algorithms/ERL/erl.py` | Canonical ERL baseline |
| `src/common/surrogate_controller.py` | Uncertainty gating, LCB scoring, EMA normalization, H-bootstrap evaluation |
| `src/common/h_bootstrap.py` | `EpisodeRunner` (chunked rollout), `TailCalibrator` (affine tail fit), `stop_probability` |
| `src/modules/evolution_module.py` | Elite preservation, tournament selection, sparse mutation |
| `src/modules/deep_modules.py` | Actor, Critic, EvidentialCritic (NIG) |
| `src/modules/ensemble_module.py` | Multi-critic ensemble with prediction std |
| `src/modules/mc_dropout_module.py` | MC Dropout runner (`_enable_only_dropout` toggles only Dropout) |
| `src/common/utils.py` | Huber loss, soft-update, weight flattening, `td3_train_critics` / `td3_update_actor` |
| `src/common/reply_buffer.py` | Replay buffer (Transition namedtuple + circular buffer) |
| `src/algorithms/SAC/sac.py` | SAC wrapper — SB3 model + shared eval/WandB callback |
| `src/algorithms/CrossQ/crossq.py` | CrossQ wrapper — sb3-contrib PyTorch model + shared eval/WandB callback |
| `src/algorithms/DDPG/ddpg.py`, `TD3/td3.py`, `PPO/ppo.py` | Thin SB3 wrappers (`stable_baselines3.DDPG/TD3/PPO`) + shared eval/WandB callback |
| `src/common/sb3_callback.py` | `EvalAndLogCallback` — periodic eval + WandB logging shared by all 5 SB3 baselines |
| `configs/algorithm/sac.yaml` | SAC hyperparameters (`learning_rate`, `ent_coef`) |
| `configs/algorithm/crossq.yaml` | CrossQ hyperparameters (`learning_rate`, `gradient_steps`) |
| `configs/algorithm/sc_erl.yaml` | SC-ERL hyperparameters (surrogate, evolution, rl, network) |
| `configs/config.yaml` | Global defaults (seed, device, wandb, env) |

---

## Critical Constraints

### 1. Exclude normalization layers from mutation
Never mutate `nn.LayerNorm` or `nn.BatchNorm` parameters. The evolution module explicitly excludes them during flattening. If you add weight-manipulation code, follow the same pattern:

```python
excluded_params = set()
for m in module.modules():
    if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        for p in m.parameters():
            excluded_params.add(p)
```

Norm layers live only in **critics**, which are never mutated (evolution mutates actors, whose only norm is `LayerNorm`, already excluded).

### 2. Never hardcode device
Always use `cfg.device` or the `device` parameter passed into modules. Auto-detection priority: `CUDA → MPS → CPU`. All `torch.Tensor` and `nn.Module` instances must be created on the correct device.

### 3. Clamp actions before env.step
Continuous action spaces require boundary enforcement:

```python
action = np.clip(action, env.action_space.low, env.action_space.high)
```

### 4. RL injection overwrites the worst individual
When syncing the RL agent into the GA population (`rl_injection_interval`), the injection target is `np.argmin(fitnesses)` — never the elite. This is intentional; do not change it.

### 5. WandB logging — pass dicts directly
```python
if self.logger is not None:
    self.logger.log({"train/critic_loss": loss, "surrogate/uncertainty_mean": sigma})
```
Do not format scalars to strings before passing to the logger.

---

## Mutation Operator (Sparse Multi-Strength)

Mutation applies to ~10% of parameters (`mutation_prob`). For each selected parameter:
- 90%: normal mutation — `w * (1 + N(0, σ²))`
- 5%: super mutation — `w * (1 + N(0, 100·σ²))`
- 5%: reset mutation — `N(0, 1)`

After mutation, weights are clamped to `[-1e6, 1e6]`.

---

## H-bootstrap (default: `surrogate.gate_mode=h_bootstrap`)

Uncertainty picks the rollout **length** per individual instead of a binary
real/surrogate decision. `SurrogateController._h_bootstrap_evaluation` +
`src/common/h_bootstrap.py`.

```
F̂ = Σ_{t<H} r_t + a·(T − H)(1 − γ)·Q̄(s_H, π(s_H)) + b·alive
σ  = |a|·(T − H)(1 − γ)·std_i Q_i(s_H, π(s_H))·alive
fitness = F̂ − β·σ            stop when Φ(−|F̂ − F_cut| / σ) < p_stop
```

- `(a, b)`: least squares on `(feature, remaining return)` pairs harvested at
  `h_chunk`-spaced horizons from ε-individuals rolled out in full. Never skip
  the calibration — the raw `(1 − γ)·Q` tail rides on the critic's absolute
  scale, which is off by ~10×, and the ranking degenerates to `Q(s_H)`.
- Uncalibrated (fewer than 30 pairs): `a = b = 0`, so fitness is the partial
  return and `σ = 0`. Half the budget goes to full rollouts until it fits.
- Budget per generation: `ρ · N · L_ema`; `Δ = clip((B − n_ε·L_ema)/N, 1, h_chunk)`.
  `ρ` and its `e_star`/`e_hat_window` adaptation keep their existing meaning,
  with `ê = |F̂(h) − R|` measured on ε-individuals.
- `T` from `env.spec.max_episode_steps`; DMC via shimmy reports `None`, so it
  grows to the longest observed episode instead.
- `surrogate.mode=random` has no σ → forced to `h_alloc=uniform`. That, plus
  `h_alloc=uniform` on the uncertainty modes, is the "uncertainty or budget?"
  ablation at identical cost.
- Only one `EpisodeRunner` may be live per env — it steps the env in place, so
  a second one resets the episode the first is holding.

## Surrogate Gating (LCB) — baseline gate modes

`f_LCB(πᵢ) = μ_Q(πᵢ) − β · σ_Q(πᵢ)`

Gating decision is controlled by `surrogate.gate_mode`:
- `topk` — real-evaluate the `rho` fraction of the population with the highest predicted uncertainty (`SurrogateController._gate_mask`), plus a random ε-coin-flip (ε=0.10) on the rest for unbiased exploration. `rho` adapts via `_update_rho` toward a target normalized surrogate error `e_star`, using `e_hat` pooled over `e_hat_window` generations (`_surrogate_error`, computed only from the ε-selected individuals, since they're the only unbiased sample of surrogate-vs-real error).
- `relative` — the original per-individual gate: if `σ_Q > percentile_threshold(mad_k)` OR the ε-coin-flip fires → real rollout, otherwise accept the surrogate fitness. Kept as an ablation baseline against `topk`.

Q-values are normalized via EMA running bounds before LCB (EMA factor α=0.05), then squashed to `[-1, 1]` via `surrogate.fitness_norm`: `tanh` (default, squeezes both tails toward the center) or `clip` (hard clip, no compression).

`SurrogateController` also records per-state uncertainty each generation (`last_per_state_mu`/`last_per_state_sigma`, shape `(n_pop, k)`, and `last_obs_batch`, shape `(k, state_dim)`) alongside the scalar `last_uncertainty`; under `debug=true` it asserts the two are consistent.

---

## Numerical Stability

- **Huber Loss** (Smooth L1) for critic updates — robust to OOD reward spikes.
- **Critic weight decay** (`1e-4`) — keeps Q-values bounded.
- **EMA Q-normalization** — prevents LCB from being destabilized by transient Q-spikes.
- **Elite protection** — best-evolved actors are never overwritten by RL injection.

---

## ERL / SC-ERL RL Update (TD3-only)

Both `erl` and `sc_erl` train their RL actor/critic exclusively via TD3 (`td3_train_critics` / `td3_update_actor` in `src/common/utils.py`): twin `Critic` networks with target networks, delayed policy updates (`rl.policy_delay`), and target-action smoothing noise (`rl.policy_noise`, `rl.noise_clip`). There is no `backbone` config option — DDPG-style single-critic updates and the native BatchRenorm CrossQ critic path were both removed after experiments showed TD3 was consistently the better gradient update for both algorithms.

Critic type is still chosen by `surrogate.mode` for SC-ERL (unaffected by the TD3-only change):
- `random` / `dropout` → twin scalar `Critic` (min-double-Q).
- `ensemble` → `EnsembleModule` of `Critic`.
- `evidential` → `EvidentialCritic` with NIG loss.

---

## Task Runner (Taskfile.yml)

```bash
task run ALGO=sc_erl CLI_ARGS="env.id=HalfCheetah-v5 surrogate.mode=ensemble"
task run ALGO=sac CLI_ARGS="env.id=HalfCheetah-v5"
task run ALGO=crossq CLI_ARGS="env.id=HalfCheetah-v5"
task run-myo ALGO=sc_erl ENV=myoHandReachRandom-v0   # MyoSuite env, separate extra
task report                          # Compile metrics, stats, PDF
task clean                           # Wipe outputs/, results/, wandb/
```

`task run` uses `uv run --extra mujoco-envs python entry_point.py` (MuJoCo v5 + DMC/fancy_gym),
against the normal `.venv`. `task run-myo` runs against a **separate** `.venv-myosuite` — see
[Dependency Extras](#dependency-extras-mujoco-envs-vs-myosuite).
Do not invoke `entry_point.py` directly without `uv run` unless the correct venv is already active.

## Dependency Extras: `mujoco-envs` vs `myosuite`

MyoSuite (`myosuite<=2.12.2`) pins `mujoco<3.7` and `gymnasium<1.3`, which is incompatible
with the `mujoco>=3.10.0` / `gymnasium>=1.3.0` / `dm-control>=1.0.20` stack used for MuJoCo v5
and DMC envs. These live in two mutually-exclusive `uv` optional-dependency groups declared
in `pyproject.toml` (`[tool.uv] conflicts`), so they cannot resolve into one environment:

- `mujoco-envs` — `dm-control`, `gymnasium[mujoco]>=1.3.0`, `mujoco>=3.10.0`. Installed in `.venv`.
- `myosuite` — `myosuite`, `gymnasium[mujoco]<1.3`, `mujoco>=3.6,<3.7`. Installed in `.venv-myosuite`.

MyoSuite therefore lives in a **separate on-disk venv directory**, not a swap of `.venv` — this
matters on SLURM where mujoco/DMC and MyoSuite array jobs can run concurrently on the same shared
filesystem; swapping one `.venv` in place would race. Build/update it via:

```bash
UV_PROJECT_ENVIRONMENT=.venv-myosuite uv sync --extra myosuite
```

`task run-myo` does this automatically before each run (`env: {UV_PROJECT_ENVIRONMENT: .venv-myosuite}`
+ `uv sync --extra myosuite` + `uv run --no-sync`). Bare `uv sync` (no extra) installs neither
`mujoco-envs` nor `myosuite` — always pass `--extra mujoco-envs` for the default `.venv`.

---

## Config System (Hydra)

- Base config: `configs/config.yaml`
- Algorithm defaults: `configs/algorithm/<algo>.yaml`
- Environment-specific overrides: `configs/algorithm/<algo>/<algo>_<env>.yaml` (loaded automatically in `entry_point.py`)
- Override at CLI: append `key=value` pairs after `algorithm=<algo>`

---

## Development Notes

- Python 3.12, managed with `uv` (`pyproject.toml`).
- No `fetch_wrappers.py` in active use — Fetch Robotics environments are not part of current experiments.
- The `plots_and_tests/` pipeline reads from WandB; ensure runs are logged before generating reports.
- `outputs/` and `results/` are gitignored; never commit experiment artifacts.
- **fancy_gym backend**: `make_env()` in `entry_point.py` auto-detects DMC/fancy envs by `dm_control/`, `fancy/`, or `metaworld/` prefix and imports `fancy_gym` lazily. Set `env.backend=fancy_gym` explicitly to force it. Dog env-specific configs already set this.
- **myosuite backend**: `make_env()` auto-detects MyoSuite envs by the `myo` prefix (all MyoSuite env IDs start with it) and lazily imports `myosuite` to register them. Set `env.backend=myosuite` explicitly to force it. Requires running from `.venv-myosuite` (`task run-myo`, or `UV_PROJECT_ENVIRONMENT=.venv-myosuite uv run ...`) — see [Dependency Extras](#dependency-extras-mujoco-envs-vs-myosuite).
- **SLURM MyoSuite jobs**: `slurm_run_array.sh` detects the `myo` prefix, sets `BACKEND=myosuite`, and activates `.venv-myosuite` instead of `.venv` — build it once per checkout with `UV_PROJECT_ENVIRONMENT=.venv-myosuite uv sync --extra myosuite` before submitting.
- **Ensemble checkpoints**: `surrogate.mode=ensemble` runs dump `checkpoints/ckpt_<env>_seed<seed>_<step>.pt` at 20/60/100% of `n_steps` — ensemble and `critic_2` state, RL actor, full population, and a 5000-sample `(obs_batch, act_batch)` buffer slice. `scripts/analyze_ensemble_checkpoints.py` runs the pre-registered rank-inversion mechanism test; `scripts/probe_ensemble_uncertainty.py` checks where disagreement sits relative to the buffer. Never use "σ at uniform actions vs σ at the actor's actions" as an OOD test — neither point is an OOD reference (see README).
- **H-bootstrap**: `H_bootstrap.md` holds the design, the offline results, and section 6 on how the online implementation differs from the simulation (online stopping instead of interleaved racing, since one `gym.Env` cannot resume an episode). `scripts/simulate_h_bootstrap.py` simulates strategies from ensemble checkpoints (trajectories cached in `outputs/h_bootstrap_cache/`); `--verify <ckpt>` cross-checks the production `TailCalibrator` against the offline reference.
- **DMC config naming**: env-specific configs for DMC follow the sanitized slug convention — `dm_control/dog-stand-v0` → `sc_erl_dm_control_dog-stand-v0.yaml`. MyoSuite env configs follow the same pattern (e.g. `sc_erl_myoHandReachRandom-v0.yaml`), currently added only under `configs/algorithm/sc_erl/` (matching the existing DMC precedent) and setting only `env.id`/`env.backend` — no tuned surrogate/evolution hyperparameters exist yet for these envs.
- **SLURM**: single `slurm_run_array.sh` handles all three backends. Backend, venv dir, and algo matrix auto-detected from `TARGET_ENV` prefix (`dm_control/`/`fancy/`/`metaworld/` → fancy_gym + 4 SC-ERL modes, 20 tasks, `.venv`; `myo` → myosuite + 10 algos, 50 tasks, `.venv-myosuite`; otherwise → MuJoCo + 10 algos, 50 tasks, `.venv`). Pass `--array=0-19` for DMC, `--array=0-49` for MuJoCo/MyoSuite.
- **SAC** wraps SB3 `SAC` (PyTorch). Shares `cfg.device` normally. No extra setup.
- **CrossQ** wraps `sb3_contrib.CrossQ` (PyTorch). Accepts `device` directly; no JAX setup needed.
- **DDPG / TD3 / PPO** are thin SB3 wrappers (`stable_baselines3.DDPG/TD3/PPO`), same pattern as SAC/CrossQ. All custom PyTorch training loops for these three were removed — DDPG/TD3 use `NormalActionNoise` for exploration; PPO uses SB3's built-in rollout buffer and GAE. All five SB3 baselines (DDPG, TD3, PPO, SAC, CrossQ) share one `EvalAndLogCallback` (`src/common/sb3_callback.py`) instead of each defining its own.
