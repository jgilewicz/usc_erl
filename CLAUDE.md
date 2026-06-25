# CLAUDE.md — SC-ERL Developer Guide

This file is the authoritative reference for AI-assisted development on this codebase. Read it before making any changes.

---

## Project Summary

**SC-ERL** is a hybrid evolutionary + deep RL framework. A population of GA actors evolves alongside a TD3/DDPG RL agent sharing a replay buffer. The key novelty is a **surrogate controller** that uses epistemic uncertainty to gate whether each candidate policy needs a real environment rollout or can be scored cheaply via the critic.

Algorithms: `sc_erl` (4 surrogate modes), `erl`, `td3`, `ddpg`, `ppo`, `sac`, `crossq` (all SB3/PyTorch).
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
| `src/modules/deep_modules.py` | Actor, Critic, StochasticActor, EvidentialCritic (NIG) |
| `src/modules/ensemble_module.py` | Multi-critic ensemble |
| `src/modules/mc_dropout_module.py` | MC Dropout runner |
| `src/common/utils.py` | Huber loss, soft-update (`polyak_update`), weight flattening |
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
