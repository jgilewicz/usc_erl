import argparse
import json
import pathlib
import subprocess
import sys

import optuna
import yaml

SC_ERL_CFG = pathlib.Path("configs/algorithm/sc_erl.yaml")


def _set_nested(d: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def patch_yaml(params: dict) -> str:
    original = SC_ERL_CFG.read_text()
    cfg = yaml.safe_load(original)
    for dotted_key, value in params.items():
        _set_nested(cfg, dotted_key, value)
    SC_ERL_CFG.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    return original


def restore_yaml(original: str) -> None:
    SC_ERL_CFG.write_text(original)


def suggest_params(trial: optuna.Trial, mode: str) -> dict:
    params: dict = {}

    # --- shared surrogate params ---
    params["surrogate.beta"] = trial.suggest_float(
        "surrogate.beta", 0.1, 10.0, log=True
    )
    params["surrogate.omega"] = trial.suggest_float("surrogate.omega", 0.3, 0.9)
    params["surrogate.k"] = trial.suggest_int("surrogate.k", 2000, 16000, step=2000)
    params["surrogate.percentile"] = trial.suggest_int("surrogate.percentile", 50, 90)
    params["surrogate.epsilon"] = trial.suggest_float("surrogate.epsilon", 0.01, 0.2)

    if mode == "dropout":
        params["surrogate.dropout_p"] = trial.suggest_float(
            "surrogate.dropout_p", 0.05, 0.4
        )
        params["surrogate.mc_samples"] = trial.suggest_int(
            "surrogate.mc_samples", 10, 50
        )
    elif mode == "ensemble":
        params["surrogate.k_ensembles"] = trial.suggest_int(
            "surrogate.k_ensembles", 3, 10
        )
    elif mode == "evidential":
        params["surrogate.lam"] = trial.suggest_float(
            "surrogate.lam", 0.01, 1.0, log=True
        )

    # always fix the mode field so the YAML matches what we're tuning
    params["surrogate.mode"] = mode

    return params


def make_objective(env: str, mode: str, n_steps: int, seed: int):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, mode)
        run_name = f"optuna_{mode}_{env}_trial{trial.number}"
        result_file = pathlib.Path(f"outputs/optuna/{run_name}/result.json")

        original_yaml = patch_yaml(params)
        try:
            cmd = [
                sys.executable,
                "entry_point.py",
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for SC-ERL surrogate modes"
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
        choices=["dropout", "ensemble", "evidential"],
        help="SC-ERL surrogate mode to tune",
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
        help="Search strategy: tpe = Bayesian (TPE, default), random = random search",
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

    sampler = (
        optuna.samplers.TPESampler(seed=args.seed)
        if args.sampler == "tpe"
        else optuna.samplers.RandomSampler(seed=args.seed)
    )

    print(f"Study      : {study_name}")
    print(f"Storage    : {storage}")
    print(f"Env        : {args.env}")
    print(f"Mode       : {args.mode}")
    print(f"Sampler    : {args.sampler.upper()}")
    print(f"Trials     : {args.n_trials}")
    print(f"Steps/trial: {args.n_steps:,}")
    print(f"Train seed : {args.seed}")
    print()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    study.optimize(
        make_objective(args.env, args.mode, args.n_steps, args.seed),
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
