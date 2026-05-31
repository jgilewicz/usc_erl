import os
import sys
import numpy as np
import optuna
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Insert project src directory into sys.path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(project_root, "src"))

import gymnasium as gym
import gymnasium_robotics
from algorithms.ERL import ERL
from algorithms.PPO import PPO
from algorithms.SC_ERL import SC_ERL
from algorithms.TD3 import TD3
from algorithms.DDPG import DDPG
from common.surrogate_controller import SurrogateMode


def resolve_algorithm_name(name: str) -> str:
    if name.startswith("sc_erl"):
        return "sc_erl"
    return name


def sanitize_env_id(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_")


def apply_best_params_to_algorithm_cfg(
    algo_cfg: DictConfig, best_params: dict
) -> DictConfig:
    updated_cfg = OmegaConf.create(OmegaConf.to_container(algo_cfg, resolve=True))

    if "actor_lr" in best_params:
        if "rl" not in updated_cfg:
            updated_cfg["rl"] = {}
        updated_cfg["rl"]["actor_lr"] = best_params["actor_lr"]
    if "critic_lr" in best_params:
        if "rl" not in updated_cfg:
            updated_cfg["rl"] = {}
        updated_cfg["rl"]["critic_lr"] = best_params["critic_lr"]

    if "mutation_std" in best_params:
        if "evolution" not in updated_cfg:
            updated_cfg["evolution"] = {}
        updated_cfg["evolution"]["mutation_std"] = best_params["mutation_std"]
    if "mutation_prob" in best_params:
        if "evolution" not in updated_cfg:
            updated_cfg["evolution"] = {}
        updated_cfg["evolution"]["mutation_prob"] = best_params["mutation_prob"]

    if "omega" in best_params:
        if "surrogate" not in updated_cfg:
            updated_cfg["surrogate"] = {}
        updated_cfg["surrogate"]["omega"] = best_params["omega"]
    if "beta" in best_params:
        if "surrogate" not in updated_cfg:
            updated_cfg["surrogate"] = {}
        updated_cfg["surrogate"]["beta"] = best_params["beta"]
    if "dropout_p" in best_params:
        if "surrogate" not in updated_cfg:
            updated_cfg["surrogate"] = {}
        updated_cfg["surrogate"]["dropout_p"] = best_params["dropout_p"]
    if "lam" in best_params:
        if "surrogate" not in updated_cfg:
            updated_cfg["surrogate"] = {}
        updated_cfg["surrogate"]["lam"] = best_params["lam"]
    if "percentile" in best_params:
        if "surrogate" not in updated_cfg:
            updated_cfg["surrogate"] = {}
        updated_cfg["surrogate"]["percentile"] = best_params["percentile"]
    if "epsilon" in best_params:
        if "surrogate" not in updated_cfg:
            updated_cfg["surrogate"] = {}
        updated_cfg["surrogate"]["epsilon"] = best_params["epsilon"]

    if "policy_noise" in best_params:
        if "rl" not in updated_cfg:
            updated_cfg["rl"] = {}
        updated_cfg["rl"]["policy_noise"] = best_params["policy_noise"]
    if "noise_clip" in best_params:
        if "rl" not in updated_cfg:
            updated_cfg["rl"] = {}
        updated_cfg["rl"]["noise_clip"] = best_params["noise_clip"]

    if "clip_param" in best_params:
        if "rl" not in updated_cfg:
            updated_cfg["rl"] = {}
        updated_cfg["rl"]["clip_param"] = best_params["clip_param"]
    if "entropy_coef" in best_params:
        if "rl" not in updated_cfg:
            updated_cfg["rl"] = {}
        updated_cfg["rl"]["entropy_coef"] = best_params["entropy_coef"]

    if "grad_clip_norm" in best_params:
        updated_cfg["grad_clip_norm"] = best_params["grad_clip_norm"]

    return updated_cfg


def save_algorithm_config(
    algo_name: str, env_id: str, algo_cfg: DictConfig, best_params: dict
) -> str:
    env_slug = sanitize_env_id(env_id)
    out_dir = os.path.join(project_root, "configs", "algorithm", algo_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{algo_name}_{env_slug}.yaml")

    updated_cfg = apply_best_params_to_algorithm_cfg(algo_cfg, best_params)
    OmegaConf.save(updated_cfg, out_path)

    with open(out_path, "r", encoding="utf-8") as file_handle:
        yaml_content = file_handle.read()
    with open(out_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"# @package _global_\n{yaml_content}")

    return out_path


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    return env


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    gym.register_envs(gymnasium_robotics)

    tune_cfg = OmegaConf.load(os.path.join(project_root, "configs", "tune.yaml"))
    algo_name = resolve_algorithm_name(cfg.name)

    # Auto device selection priority: CUDA > MPS (Apple Silicon) > CPU
    if cfg.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    def objective(trial: optuna.Trial) -> float:
        # Clone config first — all variants override from here
        trial_cfg = OmegaConf.to_container(cfg, resolve=True)

        # -----------------------------------------------------------------------
        # sc_erl (random mode): searches the full shared base parameter space.
        # sc_erl_dropout: searches ONLY dropout_p (uses base params from sc_erl run).
        # sc_erl_ensemble: searches ONLY k_ensembles (uses base params from sc_erl run).
        # -----------------------------------------------------------------------

        if cfg.name == "sc_erl_dropout":
            # Tune only the dropout-specific structural parameter
            dropout_p = trial.suggest_float("dropout_p", 0.05, 0.5)
            trial_cfg["surrogate"]["dropout_p"] = dropout_p

        elif cfg.name == "sc_erl_ensemble":
            # Tune only the number of ensemble heads
            k_ensembles = trial.suggest_int("k_ensembles", 3, 10)
            trial_cfg["surrogate"]["k_ensembles"] = k_ensembles

        elif cfg.name == "sc_erl_evidential":
            # Evidential-specific tuning: lam (log scale) and epsilon
            lam = trial.suggest_float("lam", 0.001, 0.5, log=True)
            epsilon = trial.suggest_float("epsilon", 0.05, 0.25)
            trial_cfg["surrogate"]["lam"] = lam
            trial_cfg["surrogate"]["epsilon"] = epsilon

        else:
            # Base search space reduced: freeze standard DL params (actor_lr, critic_lr, grad_clip_norm)
            # Tune only evolution params for evolutionary methods when applicable.
            if cfg.name == "erl" or cfg.name == "sc_erl":
                mutation_std = trial.suggest_float("mutation_std", 0.01, 0.2)
                mutation_prob = trial.suggest_float("mutation_prob", 0.1, 0.9)
                trial_cfg["evolution"]["mutation_std"] = mutation_std
                trial_cfg["evolution"]["mutation_prob"] = mutation_prob

            if cfg.name == "sc_erl":
                # Base SC-ERL (random mode) tunes only the surrogate gating params
                omega = trial.suggest_float("omega", 0.1, 0.9)
                beta = trial.suggest_float("beta", 0.1, 5.0)
                trial_cfg["surrogate"]["omega"] = omega
                trial_cfg["surrogate"]["beta"] = beta

        # Run the target algorithm
        try:
            if cfg.name == "erl":
                score = ERL(
                    population_size=trial_cfg["evolution"]["population_size"],
                    buffer_size=trial_cfg["buffer_size"],
                    rng=np.random.default_rng(seed=trial_cfg["seed"]),
                    env=make_env(trial_cfg["env"]["id"]),
                    eval_env=make_env(trial_cfg["eval_env"]["id"]),
                    n_steps=trial_cfg["n_steps"],
                    batch_size=trial_cfg["batch_size"],
                    device=device,
                    actor_hidden_dim=trial_cfg["network"]["actor_hidden_dim"],
                    critic_hidden_dim=trial_cfg["network"]["critic_hidden_dim"],
                    gamma=trial_cfg["rl"]["gamma"],
                    tau=trial_cfg["rl"]["tau"],
                    mutation_std=trial_cfg["evolution"]["mutation_std"],
                    mutation_prob=trial_cfg["evolution"]["mutation_prob"],
                    elite_ratio=trial_cfg["evolution"]["elite_ratio"],
                    rl_injection_interval=trial_cfg["evolution"][
                        "rl_injection_interval"
                    ],
                    warmup_steps=trial_cfg["warmup"]["warmup_steps"],
                    actor_lr=trial_cfg["rl"]["actor_lr"],
                    critic_lr=trial_cfg["rl"]["critic_lr"],
                    evaluate_episodes=trial_cfg["evaluation"]["evaluate_episodes"],
                    exploration_noise_std=trial_cfg["rl"]["exploration_noise_std"],
                    gradient_steps=trial_cfg["rl"]["gradient_steps"],
                    logger=None,
                    debug=False,
                    grad_clip_norm=trial_cfg["grad_clip_norm"],
                )
            elif cfg.name == "td3":
                score = TD3(
                    buffer_size=trial_cfg["buffer_size"],
                    rng=np.random.default_rng(seed=trial_cfg["seed"]),
                    env=make_env(trial_cfg["env"]["id"]),
                    eval_env=make_env(trial_cfg["eval_env"]["id"]),
                    n_steps=trial_cfg["n_steps"],
                    batch_size=trial_cfg["batch_size"],
                    device=device,
                    actor_hidden_dim=trial_cfg["network"]["actor_hidden_dim"],
                    critic_hidden_dim=trial_cfg["network"]["critic_hidden_dim"],
                    gamma=trial_cfg["rl"]["gamma"],
                    tau=trial_cfg["rl"]["tau"],
                    actor_lr=trial_cfg["rl"]["actor_lr"],
                    critic_lr=trial_cfg["rl"]["critic_lr"],
                    policy_noise=trial_cfg["rl"]["policy_noise"],
                    noise_clip=trial_cfg["rl"]["noise_clip"],
                    policy_delay=trial_cfg["rl"]["policy_delay"],
                    exploration_noise_std=trial_cfg["rl"]["exploration_noise_std"],
                    warmup_steps=trial_cfg["warmup"]["warmup_steps"],
                    evaluate_episodes=trial_cfg["evaluation"]["evaluate_episodes"],
                    eval_interval=trial_cfg["evaluation"]["eval_interval"],
                    logger=None,
                    debug=False,
                    grad_clip_norm=trial_cfg["grad_clip_norm"],
                )
            elif cfg.name == "ddpg":
                score = DDPG(
                    buffer_size=trial_cfg["buffer_size"],
                    rng=np.random.default_rng(seed=trial_cfg["seed"]),
                    env=make_env(trial_cfg["env"]["id"]),
                    eval_env=make_env(trial_cfg["eval_env"]["id"]),
                    n_steps=trial_cfg["n_steps"],
                    batch_size=trial_cfg["batch_size"],
                    device=device,
                    actor_hidden_dim=trial_cfg["network"]["actor_hidden_dim"],
                    critic_hidden_dim=trial_cfg["network"]["critic_hidden_dim"],
                    gamma=trial_cfg["rl"]["gamma"],
                    tau=trial_cfg["rl"]["tau"],
                    actor_lr=trial_cfg["rl"]["actor_lr"],
                    critic_lr=trial_cfg["rl"]["critic_lr"],
                    exploration_noise_std=trial_cfg["rl"]["exploration_noise_std"],
                    warmup_steps=trial_cfg["warmup"]["warmup_steps"],
                    evaluate_episodes=trial_cfg["evaluation"]["evaluate_episodes"],
                    eval_interval=trial_cfg["evaluation"]["eval_interval"],
                    logger=None,
                    debug=False,
                    grad_clip_norm=trial_cfg["grad_clip_norm"],
                )
            # All SC-ERL variants (dropout, ensemble, random) use the same constructor.
            # The surrogate mode is read from the algorithm config YAML (already set per variant).
            elif cfg.name.startswith("sc_erl"):
                score = SC_ERL(
                    population_size=trial_cfg["evolution"]["population_size"],
                    buffer_size=trial_cfg["buffer_size"],
                    rng=np.random.default_rng(seed=trial_cfg["seed"]),
                    env=make_env(trial_cfg["env"]["id"]),
                    eval_env=make_env(trial_cfg["eval_env"]["id"]),
                    n_steps=trial_cfg["n_steps"],
                    batch_size=trial_cfg["batch_size"],
                    device=device,
                    actor_hidden_dim=trial_cfg["network"]["actor_hidden_dim"],
                    critic_hidden_dim=trial_cfg["network"]["critic_hidden_dim"],
                    gamma=trial_cfg["rl"]["gamma"],
                    tau=trial_cfg["rl"]["tau"],
                    mutation_std=trial_cfg["evolution"]["mutation_std"],
                    mutation_prob=trial_cfg["evolution"]["mutation_prob"],
                    elite_ratio=trial_cfg["evolution"]["elite_ratio"],
                    rl_injection_interval=trial_cfg["evolution"][
                        "rl_injection_interval"
                    ],
                    warmup_steps=trial_cfg["warmup"]["warmup_steps"],
                    actor_lr=trial_cfg["rl"]["actor_lr"],
                    critic_lr=trial_cfg["rl"]["critic_lr"],
                    exploration_noise_std=trial_cfg["rl"]["exploration_noise_std"],
                    gradient_steps=trial_cfg["rl"]["gradient_steps"],
                    omega=trial_cfg["surrogate"]["omega"],
                    surrogate_mode=SurrogateMode.to_mode(
                        trial_cfg["surrogate"]["mode"]
                    ),
                    k=trial_cfg["surrogate"]["k"],
                    k_ensembles=trial_cfg["surrogate"]["k_ensembles"],
                    beta=trial_cfg["surrogate"]["beta"],
                    epsilon=trial_cfg["surrogate"].get("epsilon", 0.10),
                    percentile=trial_cfg["surrogate"].get("percentile", 75),
                    lam=trial_cfg["surrogate"].get("lam", 0.1),
                    logger=None,
                    debug=False,
                    grad_clip_norm=trial_cfg["grad_clip_norm"],
                    dropout_p=trial_cfg["surrogate"]["dropout_p"],
                    mc_samples=trial_cfg["surrogate"]["mc_samples"],
                )
            elif cfg.name == "ppo":
                score = PPO(
                    env=make_env(trial_cfg["env"]["id"]),
                    eval_env=make_env(trial_cfg["eval_env"]["id"]),
                    n_steps=trial_cfg["n_steps"],
                    rollout_steps=trial_cfg["buffer_size"],
                    batch_size=trial_cfg["batch_size"],
                    ppo_epochs=trial_cfg["rl"]["ppo_epochs"],
                    device=device,
                    actor_hidden_dim=trial_cfg["network"]["actor_hidden_dim"],
                    critic_hidden_dim=trial_cfg["network"]["critic_hidden_dim"],
                    gamma=trial_cfg["rl"]["gamma"],
                    gae_lambda=trial_cfg["rl"]["gae_lambda"],
                    clip_param=trial_cfg["rl"]["clip_param"],
                    entropy_coef=trial_cfg["rl"]["entropy_coef"],
                    actor_lr=trial_cfg["rl"]["actor_lr"],
                    critic_lr=trial_cfg["rl"]["critic_lr"],
                    evaluate_episodes=trial_cfg["evaluation"]["evaluate_episodes"],
                    eval_interval=trial_cfg["evaluation"]["eval_interval"],
                    logger=None,
                    debug=False,
                    grad_clip_norm=trial_cfg["grad_clip_norm"],
                )
            else:
                score = -float("inf")
        except Exception as e:
            print(f"Trial failed with exception: {e}")
            score = -float("inf")

        return score

    print(
        f"Starting Optuna HPO Study '{tune_cfg.study_name}' for algorithm '{cfg.name}'..."
    )
    study = optuna.create_study(
        study_name=tune_cfg.study_name,
        direction="maximize",
        storage=tune_cfg.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=0, interval_steps=1
        ),
    )
    study.optimize(objective, n_trials=tune_cfg.n_trials)

    print("\nOptuna Optimization successfully completed!")
    print(f"Best Trial value: {study.best_trial.value}")
    print("Best Hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Save best parameters to yaml file
    out_dir = os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"optuna_best_{algo_name}_{cfg.env.id}.yaml")
    OmegaConf.save(OmegaConf.create(study.best_trial.params), out_path)
    print(f"Saved best parameters to {out_path}")

    try:
        algo_base_path = os.path.join(
            project_root, "configs", "algorithm", f"{algo_name}.yaml"
        )

        if os.path.exists(algo_base_path):
            algo_cfg = OmegaConf.load(algo_base_path)
            env_specific_path = save_algorithm_config(
                algo_name=algo_name,
                env_id=cfg.env.id,
                algo_cfg=algo_cfg,
                best_params=study.best_trial.params,
            )
            print(f"Saved environment-specific config to {env_specific_path}")

    except Exception as e:
        print(f"Warning: Could not save environment-specific config: {e}")


if __name__ == "__main__":
    main()
