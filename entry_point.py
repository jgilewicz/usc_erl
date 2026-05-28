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
from ERL.erl import ERL
from PPO.ppo import PPO
from SC_ERL.sc_erl import SC_ERL
from TD3.td3 import TD3
from DDPG.ddpg import DDPG


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
        
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    return env


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    device = torch.device(
        "cuda"
        if cfg.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if cfg.device == "auto"
        else cfg.device
    )

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
            population_size=cfg.population_size,
            buffer_size=cfg.buffer_size,
            rng=np.random.default_rng(seed=cfg.seed),
            env=make_env(cfg.env.id),
            eval_env=make_env(cfg.eval_env.id),
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            device=device,
            hidden_dim=cfg.network.hidden_dim,
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
            hidden_dim=cfg.network.hidden_dim,
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
            hidden_dim=cfg.network.hidden_dim,
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
        )
    elif cfg.name == "sc_erl":
        SC_ERL(
            population_size=cfg.population_size,
            buffer_size=cfg.buffer_size,
            rng=np.random.default_rng(seed=cfg.seed),
            env=make_env(cfg.env.id),
            eval_env=make_env(cfg.eval_env.id),
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            device=device,
            hidden_dim=cfg.network.hidden_dim,
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
            ensemble_size=cfg.surrogate.k,
            logger=logger,
            debug=cfg.debug,
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
            hidden_dim=cfg.network.hidden_dim,
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
        )
    else:
        raise ValueError(f"Unknown algorithm: {cfg.name}")

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
