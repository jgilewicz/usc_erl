"""
Optuna hyperparameter tuning for SC-ERL — two-stage approach.

Stage 1  (--mode random):
  Tunes shared surrogate params + TD3 backbone params using the cheapest
  surrogate mode (random coin-flip, no uncertainty estimation overhead).

Stage 2  (--mode dropout|ensemble|evidential):
  Loads the best shared params from a Stage 1 study (--base-study) and fixes
  them, then tunes only the mode-specific params.

Usage
-----
# Stage 1 — shared + TD3 backbone params
python optim/tune_sc_erl.py --env HalfCheetah-v5 --mode random --n-trials 30

# Stage 2 — method-specific params only
python optim/tune_sc_erl.py --env HalfCheetah-v5 --mode dropout  --n-trials 20 \\
    --base-study optuna_random_HalfCheetah-v5.db
python optim/tune_sc_erl.py --env HalfCheetah-v5 --mode ensemble --n-trials 20 \\
    --base-study optuna_random_HalfCheetah-v5.db
python optim/tune_sc_erl.py --env HalfCheetah-v5 --mode evidential --n-trials 20 \\
    --base-study optuna_random_HalfCheetah-v5.db
"""

import argparse
import json
import pathlib
import subprocess
import sys

import optuna
import yaml

SC_ERL_CFG = pathlib.Path("configs/algorithm/sc_erl.yaml")

# ---------------------------------------------------------------------------
# YAML patching helpers
# ---------------------------------------------------------------------------

def _set_nested(d: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def patch_yaml(params: dict) -> str:
    """Apply *params* to sc_erl.yaml; return the original file text."""
    original = SC_ERL_CFG.read_text()
    cfg = yaml.safe_load(original)
    for dotted_key, value in params.items():
        _set_nested(cfg, dotted_key, value)
    SC_ERL_CFG.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    return original


def restore_yaml(original: str) -> None:
    SC_ERL_CFG.write_text(original)


# ---------------------------------------------------------------------------
# Search-space definitions
# ---------------------------------------------------------------------------

def suggest_shared_params(trial: optuna.Trial) -> dict:
    """Shared surrogate params — tuned in Stage 1 (random mode)."""
    return {
        "surrogate.beta": trial.suggest_float("surrogate.beta", 0.1, 10.0, log=True),
        "surrogate.omega": trial.suggest_float("surrogate.omega", 0.3, 0.9),
        "surrogate.k": trial.suggest_int("surrogate.k", 1024, 8192, step=1024),
        "surrogate.epsilon": trial.suggest_float("surrogate.epsilon", 0.01, 0.2),
        "surrogate.mad_k": trial.suggest_float("surrogate.mad_k", 0.5, 4.0),
    }


def suggest_backbone_params(trial: optuna.Trial, backbone: str) -> dict:
    """TD3 backbone params — tuned in Stage 1 alongside shared params."""
    if backbone != "td3":
        return {}
    return {
        "rl.policy_noise": trial.suggest_float("rl.policy_noise", 0.1, 0.5),
        "rl.noise_clip": trial.suggest_float("rl.noise_clip", 0.3, 0.8),
        "rl.policy_delay": trial.suggest_int("rl.policy_delay", 1, 4),
    }


def suggest_mode_params(trial: optuna.Trial, mode: str) -> dict:
    """Mode-specific params — tuned in Stage 2."""
    if mode == "dropout":
        return {
            "surrogate.dropout_p": trial.suggest_float("surrogate.dropout_p", 0.05, 0.4),
            "surrogate.mc_samples": trial.suggest_int("surrogate.mc_samples", 10, 50),
        }
    if mode == "ensemble":
        return {
            "surrogate.k_ensembles": trial.suggest_int("surrogate.k_ensembles", 3, 10),
        }
    if mode == "evidential":
        return {
            "surrogate.lam": trial.suggest_float("surrogate.lam", 0.01, 1.0, log=True),
        }
    return {}  # random mode — no mode-specific params


def load_base_params(base_study_path: str) -> dict:
    """Load best trial params from a Stage 1 SQLite study."""
    storage = f"sqlite:///{base_study_path}"
    summaries = optuna.get_all_study_summaries(storage=storage)
    if not summaries:
        raise ValueError(f"No studies found in {base_study_path}")
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    return study.best_trial.params


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(env: str, mode: str, backbone: str, n_steps: int, seed: int,
                   base_params: dict | None):
    def objective(trial: optuna.Trial) -> float:
        params: dict = {}

        if base_params is not None:
            # Stage 2: fix shared + backbone params from Stage 1, tune mode-specific only
            params.update(base_params)
            params.update(suggest_mode_params(trial, mode))
        else:
            # Stage 1 (random) or standalone: tune everything
            params.update(suggest_shared_params(trial))
            params.update(suggest_backbone_params(trial, backbone))
            params.update(suggest_mode_params(trial, mode))

        params["surrogate.mode"] = mode
        params["backbone"] = backbone

        run_name = f"optuna_{mode}_{env}_trial{trial.number}"
        result_file = pathlib.Path(f"outputs/optuna/{run_name}/result.json")

        original_yaml = patch_yaml(params)
        try:
            cmd = [
                sys.executable, "entry_point.py",
                "algorithm=sc_erl",
                f"env.id={env}",
                f"eval_env.id={env}",
                f"seed={seed}",
                f"n_steps={n_steps}",
                "wandb.enabled=false",
                f"result_file={result_file}",
                f"hydra.run.dir=outputs/optuna/{run_name}",
            ]
            proc = subprocess.run(cmd, check=False)
        finally:
            restore_yaml(original_yaml)

        if proc.returncode != 0:
            raise optuna.TrialPruned(
                f"Training subprocess exited with code {proc.returncode}"
            )
        if not result_file.exists():
            raise optuna.TrialPruned("result.json was not written by training run")

        data = json.loads(result_file.read_text())
        return float(data["eval_reward"])

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna two-stage hyperparameter tuning for SC-ERL"
    )
    p.add_argument(
        "--env",
        required=True,
        choices=["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5", "Swimmer-v5"],
        help="MuJoCo environment id",
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["random", "dropout", "ensemble", "evidential"],
        help="Stage 1: random — tunes shared+backbone params. Stage 2: dropout/ensemble/evidential",
    )
    p.add_argument(
        "--backbone",
        choices=["ddpg", "td3"],
        default="td3",
        help="RL backbone (default: td3)",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials (default: 30)",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=200_000,
        help="Training steps per trial (default: 200000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed RNG seed for all training runs (default: 0)",
    )
    p.add_argument(
        "--sampler",
        choices=["tpe", "random"],
        default="tpe",
        help="Search strategy: tpe = Bayesian TPE (default), random = random search",
    )
    p.add_argument(
        "--base-study",
        default=None,
        help="Stage 2: path to Stage 1 SQLite DB to load fixed shared params from",
    )
    p.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (default: sqlite:///optuna_<mode>_<env>.db)",
    )
    p.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name (default: sc_erl_<mode>_<env>)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    study_name = args.study_name or f"sc_erl_{args.mode}_{args.env}"
    storage = args.storage or f"sqlite:///optuna_{args.mode}_{args.env}.db"

    base_params = None
    if args.base_study:
        if args.mode == "random":
            print("WARNING: --base-study is ignored for --mode random (Stage 1)")
        else:
            base_params = load_base_params(args.base_study)
            print(f"Loaded {len(base_params)} shared params from {args.base_study}")

    sampler = (
        optuna.samplers.TPESampler(seed=args.seed)
        if args.sampler == "tpe"
        else optuna.samplers.RandomSampler(seed=args.seed)
    )

    stage = "1 (shared+backbone)" if args.mode == "random" else f"2 ({args.mode}-specific)"
    print(f"Stage      : {stage}")
    print(f"Study      : {study_name}")
    print(f"Storage    : {storage}")
    print(f"Env        : {args.env}")
    print(f"Mode       : {args.mode}")
    print(f"Backbone   : {args.backbone}")
    print(f"Sampler    : {args.sampler.upper()}")
    print(f"Trials     : {args.n_trials}")
    print(f"Steps/trial: {args.n_steps:,}")
    print(f"Train seed : {args.seed}")
    if base_params:
        print(f"Fixed params from Stage 1: {list(base_params.keys())}")
    print()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    study.optimize(
        make_objective(args.env, args.mode, args.backbone, args.n_steps, args.seed,
                       base_params),
        n_trials=args.n_trials,
    )

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best trial : #{best.number}  eval_reward = {best.value:.4f}")
    print("=" * 60)
    col_w = max(len(k) for k in best.params) + 2
    print(f"  {'Parameter':<{col_w}}  Value")
    print(f"  {'-' * col_w}  -----")
    for k, v in sorted(best.params.items()):
        print(f"  {k:<{col_w}}  {v}")
    print()


if __name__ == "__main__":
    main()
