import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
import hydra

from common.wandb_logger import WandbLogger
from ERL.erl import ERL
from TD3.td3 import TD3
from SC_ERL.sc_erl import SC_ERL
from common.surrogate_controller import SurrogateMode


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

    if cfg.name == "erl":
        ERL(
            population_size=cfg.population_size,
            buffer_size=cfg.buffer_size,
            rng=np.random.default_rng(seed=cfg.seed),
            env=gym.make(cfg.env.id),
            eval_env=gym.make(cfg.eval_env.id),
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
            env=gym.make(cfg.env.id),
            eval_env=gym.make(cfg.eval_env.id),
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
    elif cfg.name == "sc_erl":
        SC_ERL(
            population_size=cfg.population_size,
            buffer_size=cfg.buffer_size,
            rng=np.random.default_rng(seed=cfg.seed),
            env=gym.make(cfg.env.id),
            eval_env=gym.make(cfg.eval_env.id),
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
            logger=logger,
            debug=cfg.debug,
        )
    else:
        raise ValueError(f"Unknown algorithm: {cfg.name}")

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
