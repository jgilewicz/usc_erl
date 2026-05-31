import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gymnasium as gym
import gymnasium_robotics
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from common.surrogate_controller import SurrogateMode
from common.wandb_logger import WandbLogger
from algorithms.ERL import ERL
from algorithms.PPO import PPO
from algorithms.SC_ERL import SC_ERL
from algorithms.TD3 import TD3
from algorithms.DDPG import DDPG


def resolve_algorithm_name(name: str) -> str:
    if name.startswith("sc_erl"):
        return "sc_erl"
    return name


def sanitize_env_id(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_")


def load_environment_specific_algorithm_cfg(cfg: DictConfig) -> DictConfig:
    algo_name = resolve_algorithm_name(cfg.name)
    env_slug = sanitize_env_id(cfg.env.id)
    env_specific_path = (
        Path(__file__).parent
        / "configs"
        / "algorithm"
        / algo_name
        / f"{algo_name}_{env_slug}.yaml"
    )

    if not env_specific_path.exists():
        return cfg

    env_specific_cfg = OmegaConf.load(env_specific_path)
    return OmegaConf.merge(cfg, env_specific_cfg)


def make_env(env_id: str) -> gym.Env:
    is_metaworld = False
    mw_env_id = env_id

    if env_id.endswith("-v2"):
        mw_env_id = env_id.replace("-v2", "-v3-goal-observable")
        try:
            import metaworld

            if mw_env_id in metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
                is_metaworld = True
        except ImportError:
            pass

    if is_metaworld:
        import metaworld

        env_cls = metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[mw_env_id]
        env = env_cls()
        env._freeze_rand_vec = False
        env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
    else:
        env = gym.make(env_id)

    if "Fetch" in env_id:
        from common.fetch_wrappers import FetchCustomRewardWrapper

        env = FetchCustomRewardWrapper(env, env_id)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    return env


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    cfg = load_environment_specific_algorithm_cfg(cfg)

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

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logger = None
    if cfg.wandb.enabled:
        run_name = cfg.wandb.name if cfg.wandb.name else cfg.name
        logger = WandbLogger(
            project=cfg.wandb.project,
            name=run_name,
            entity=cfg.wandb.entity,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else [cfg.name],
        )
        logger.init(
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    gym.register_envs(gymnasium_robotics)

    if cfg.name == "erl":
        ERL(
            population_size=cfg.evolution.population_size,
            buffer_size=cfg.buffer_size,
            rng=np.random.default_rng(seed=cfg.seed),
            env=make_env(cfg.env.id),
            eval_env=make_env(cfg.eval_env.id),
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            device=device,
            actor_hidden_dim=cfg.network.actor_hidden_dim,
            critic_hidden_dim=cfg.network.critic_hidden_dim,
            gamma=cfg.rl.gamma,
            tau=cfg.rl.tau,
            mutation_std=cfg.evolution.mutation_std,
            mutation_prob=cfg.evolution.mutation_prob,
            elite_ratio=cfg.evolution.elite_ratio,
            rl_injection_interval=cfg.evolution.rl_injection_interval,
            warmup_steps=cfg.warmup.warmup_steps,
            actor_lr=cfg.rl.actor_lr,
            critic_lr=cfg.rl.critic_lr,
            evaluate_episodes=cfg.evaluation.evaluate_episodes,
            exploration_noise_std=cfg.rl.exploration_noise_std,
            gradient_steps=cfg.rl.gradient_steps,
            logger=logger,
            debug=cfg.debug,
            grad_clip_norm=cfg.grad_clip_norm,
        )
    elif cfg.name == "td3":
        TD3(
            buffer_size=cfg.buffer_size,
            rng=np.random.default_rng(seed=cfg.seed),
            env=make_env(cfg.env.id),
            eval_env=make_env(cfg.eval_env.id),
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            device=device,
            actor_hidden_dim=cfg.network.actor_hidden_dim,
            critic_hidden_dim=cfg.network.critic_hidden_dim,
            gamma=cfg.rl.gamma,
            tau=cfg.rl.tau,
            actor_lr=cfg.rl.actor_lr,
            critic_lr=cfg.rl.critic_lr,
            policy_noise=cfg.rl.policy_noise,
            noise_clip=cfg.rl.noise_clip,
            policy_delay=cfg.rl.policy_delay,
            exploration_noise_std=cfg.rl.exploration_noise_std,
            warmup_steps=cfg.warmup.warmup_steps,
            evaluate_episodes=cfg.evaluation.evaluate_episodes,
            eval_interval=cfg.evaluation.eval_interval,
            logger=logger,
            debug=cfg.debug,
            grad_clip_norm=cfg.grad_clip_norm,
        )
    elif cfg.name == "ddpg":
        DDPG(
            buffer_size=cfg.buffer_size,
            rng=np.random.default_rng(seed=cfg.seed),
            env=make_env(cfg.env.id),
            eval_env=make_env(cfg.eval_env.id),
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            device=device,
            actor_hidden_dim=cfg.network.actor_hidden_dim,
            critic_hidden_dim=cfg.network.critic_hidden_dim,
            gamma=cfg.rl.gamma,
            tau=cfg.rl.tau,
            actor_lr=cfg.rl.actor_lr,
            critic_lr=cfg.rl.critic_lr,
            exploration_noise_std=cfg.rl.exploration_noise_std,
            warmup_steps=cfg.warmup.warmup_steps,
            evaluate_episodes=cfg.evaluation.evaluate_episodes,
            eval_interval=cfg.evaluation.eval_interval,
            logger=logger,
            debug=cfg.debug,
            grad_clip_norm=cfg.grad_clip_norm,
        )
    elif cfg.name == "sc_erl":
        SC_ERL(
            population_size=cfg.evolution.population_size,
            buffer_size=cfg.buffer_size,
            rng=np.random.default_rng(seed=cfg.seed),
            env=make_env(cfg.env.id),
            eval_env=make_env(cfg.eval_env.id),
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            device=device,
            actor_hidden_dim=cfg.network.actor_hidden_dim,
            critic_hidden_dim=cfg.network.critic_hidden_dim,
            gamma=cfg.rl.gamma,
            tau=cfg.rl.tau,
            mutation_std=cfg.evolution.mutation_std,
            mutation_prob=cfg.evolution.mutation_prob,
            elite_ratio=cfg.evolution.elite_ratio,
            rl_injection_interval=cfg.evolution.rl_injection_interval,
            warmup_steps=cfg.warmup.warmup_steps,
            actor_lr=cfg.rl.actor_lr,
            critic_lr=cfg.rl.critic_lr,
            exploration_noise_std=cfg.rl.exploration_noise_std,
            gradient_steps=cfg.rl.gradient_steps,
            omega=cfg.surrogate.omega,
            surrogate_mode=SurrogateMode.to_mode(cfg.surrogate.mode),
            k=cfg.surrogate.k,
            k_ensembles=cfg.surrogate.k_ensembles,
            beta=cfg.surrogate.beta,
            logger=logger,
            debug=cfg.debug,
            grad_clip_norm=cfg.grad_clip_norm,
            dropout_p=cfg.surrogate.dropout_p,
            mc_samples=cfg.surrogate.mc_samples,
            min_uncertainty_floor=cfg.surrogate.min_uncertainty_floor,
        )
    elif cfg.name == "ppo":
        PPO(
            env=make_env(cfg.env.id),
            eval_env=make_env(cfg.eval_env.id),
            n_steps=cfg.n_steps,
            rollout_steps=cfg.buffer_size,
            batch_size=cfg.batch_size,
            ppo_epochs=cfg.rl.ppo_epochs,
            device=device,
            actor_hidden_dim=cfg.network.actor_hidden_dim,
            critic_hidden_dim=cfg.network.critic_hidden_dim,
            gamma=cfg.rl.gamma,
            gae_lambda=cfg.rl.gae_lambda,
            clip_param=cfg.rl.clip_param,
            entropy_coef=cfg.rl.entropy_coef,
            actor_lr=cfg.rl.actor_lr,
            critic_lr=cfg.rl.critic_lr,
            evaluate_episodes=cfg.evaluation.evaluate_episodes,
            eval_interval=cfg.evaluation.eval_interval,
            logger=logger,
            debug=cfg.debug,
            grad_clip_norm=cfg.grad_clip_norm,
        )
    else:
        raise ValueError(f"Unknown algorithm: {cfg.name}")

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
