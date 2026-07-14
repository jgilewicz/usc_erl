# CLAUDE.md — SC-ERL Developer Guide

This file is the authoritative reference for AI-assisted development on this codebase. Read it before making any changes.

---

## Project Summary

**SC-ERL** is a hybrid evolutionary + deep RL framework. A population of GA actors evolves alongside a DDPG/TD3/CrossQ RL agent sharing a replay buffer. The key novelty is a **surrogate controller** that uses epistemic uncertainty to gate whether each candidate policy needs a real environment rollout or can be scored cheaply via the critic.

Algorithms: `sc_erl` (4 surrogate modes), `erl`, `td3`, `ddpg`, `ppo`, `sac`, `crossq` (all SB3/PyTorch).

Both `sc_erl` and `erl` select their RL learner via `backbone`: `ddpg` (default), `td3`, or `crossq` (native, not the SB3 `crossq` baseline). See "CrossQ backbone" below.
Environments:
- **MuJoCo v5**: `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5`.
- **DMC via fancy_gym**: `dm_control/dog-{stand,walk,trot,run,fetch}-v0`.

---

## Key Source Files

| File | Role |
|------|------|
| `entry_point.py` | Hydra launcher; routes config → algorithm constructor |
| `src/algorithms/SC_ERL/sc_erl.py` | Main SC-ERL training loop |
| `src/algorithms/ERL/erl.py` | Canonical ERL baseline |
| `src/common/surrogate_controller.py` | Uncertainty gating, LCB scoring, EMA normalization |
| `src/modules/evolution_module.py` | Elite preservation, tournament selection, sparse mutation |
| `src/modules/deep_modules.py` | Actor, Critic, StochasticActor, EvidentialCritic (NIG), BatchNormCritic + BatchRenorm1d (CrossQ) |
| `src/modules/ensemble_module.py` | Multi-critic ensemble (+ `crossq_compute_loss` for the CrossQ ensemble) |
| `src/modules/mc_dropout_module.py` | MC Dropout runner (`_enable_only_dropout` keeps BatchNorm in eval) |
| `src/common/utils.py` | Huber loss, soft-update (`polyak_update`), weight flattening, `crossq_train_critics` / `crossq_update_actor` |
| `src/common/reply_buffer.py` | Replay buffer (Transition namedtuple + circular buffer) |
| `src/algorithms/SAC/sac.py` | SAC wrapper — SB3 model + WandB callback |
| `src/algorithms/CrossQ/crossq.py` | CrossQ wrapper — sb3-contrib PyTorch model + WandB callback |
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

Norm layers live only in **critics**, which are never mutated (evolution mutates actors, whose only norm is `LayerNorm`, already excluded). The CrossQ `BatchRenorm1d` is a custom module **not** covered by the `isinstance` check above — this is fine as long as it stays out of the actor. Never put BatchNorm/BatchRenorm in a population actor.

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

## Surrogate Gating (LCB)

`f_LCB(πᵢ) = μ_Q(πᵢ) − β · σ_Q(πᵢ)`

Gating decision: if `σ_Q > percentile_threshold(ω)` OR random ε-coin flip (ε=0.10) → real rollout. Otherwise → accept surrogate fitness.

Q-values are normalized via EMA running bounds before LCB (EMA factor α=0.05), then tanh-clipped to `[-1, 1]`.

---

## Numerical Stability

- **Huber Loss** (Smooth L1) for critic updates — robust to OOD reward spikes.
- **Critic weight decay** (`1e-4`) — keeps Q-values bounded.
- **EMA Q-normalization** — prevents LCB from being destabilized by transient Q-spikes.
- **Elite protection** — best-evolved actors are never overwritten by RL injection.

---

## CrossQ backbone (`backbone=crossq`)

A native CrossQ RL learner for `sc_erl` and `erl` — **not** the SB3-contrib `crossq` baseline (that stays a separate standalone algorithm). CrossQ ≈ "TD3/SAC minus target networks plus BatchNorm in the critic", deterministic (no entropy term, no `alpha`). The population actor stays the deterministic `Actor`; only the critic side changes.

Mechanics (`crossq_train_critics` in `src/common/utils.py`):
- **Joint forward pass** — current `(s,a)` and next `(s',a')` are concatenated along the batch dim and pushed through the critic in ONE forward, so BatchNorm normalises both with shared statistics. Split back with `[:b]` (current) / `[b:]` (next).
- **No target networks, no soft-update.** Only the next-Q slice is detached (stop-gradient); the current-Q slice keeps its gradient.
- **`crossq_update_actor`** switches the critic to `eval` for the policy-gradient pass so it does not pollute BatchNorm running stats.

Normalization is **`BatchRenorm1d`** (Ioffe 2017, as in CrossQ), not plain BatchNorm — `r`/`d` corrections ramped from the BatchNorm identity over `warmup_steps` (counts critic training forward passes, not env steps). `bn_momentum` (config `rl.bn_momentum`, default `0.01`) sets its momentum. At `eval` it uses running stats → surrogate scoring is deterministic and safe for any batch size (incl. batch=1).

Critic type is chosen by `surrogate.mode` so the `SurrogateController` contract holds:
- `random` / `dropout` → twin scalar `BatchNormCritic` (min-double-Q). Dropout mode adds `nn.Dropout`; `MCDropout._enable_only_dropout` keeps BatchRenorm in eval while toggling only Dropout, so MC variance comes solely from dropout.
- `ensemble` → `EnsembleModule` of `BatchNormCritic`, trained via `EnsembleModule.crossq_compute_loss` (no twin, no target).
- `evidential` → `EvidentialCritic(use_bn=True)` with NIG loss (no twin, no target).

**Gotcha:** the joint forward requires BatchRenorm in `train` mode; the surrogate leaves the critic in `eval` after scoring, so `crossq_train_critics` re-asserts `train()` at entry. Never run a BatchRenorm critic in `train` on a batch of size 1.

---

## Task Runner (Taskfile.yml)

```bash
task run ALGO=sc_erl CLI_ARGS="env.id=HalfCheetah-v5 surrogate.mode=ensemble"
task run ALGO=sac CLI_ARGS="env.id=HalfCheetah-v5"
task run ALGO=crossq CLI_ARGS="env.id=HalfCheetah-v5"
task run-parallel                    # Full matrix: 10 algos × 5 MuJoCo envs × 5 seeds
task run-dmc ENV=dm_control/dog-stand-v0 MODE=ensemble SEED=0  # Single DMC run
task run-parallel-dmc                # Full matrix: SC-ERL × 5 DMC dog envs × 4 modes × 5 seeds
task report                          # Compile metrics, stats, PDF
task clean                           # Wipe outputs/, results/, wandb/
```

All runs use `uv run python entry_point.py`. Do not invoke `entry_point.py` directly without `uv run` unless the venv is already activated.

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
- **DMC config naming**: env-specific configs for DMC follow the sanitized slug convention — `dm_control/dog-stand-v0` → `sc_erl_dm_control_dog-stand-v0.yaml`.
- **SLURM**: single `slurm_run_array.sh` handles both backends. Backend and algo matrix auto-detected from `TARGET_ENV` prefix (`dm_control/` → fancy_gym + 4 SC-ERL modes, 20 tasks; otherwise → MuJoCo + 10 algos, 50 tasks). Pass `--array=0-19` for DMC, `--array=0-49` for MuJoCo.
- **SAC** wraps SB3 `SAC` (PyTorch). Shares `cfg.device` normally. No extra setup.
- **CrossQ** wraps `sb3_contrib.CrossQ` (PyTorch). Accepts `device` directly; no JAX setup needed.
