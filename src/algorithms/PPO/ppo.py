import torch
import gymnasium as gym

from stable_baselines3 import PPO as _SB3_PPO

from common.sb3_callback import EvalAndLogCallback
from common.wandb_logger import WandbLogger


def PPO(
    env: gym.Env,
    eval_env: gym.Env,
    n_steps: int,
    rollout_steps: int = 2048,
    batch_size: int = 64,
    ppo_epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_param: float = 0.2,
    entropy_coef: float = 0.0,
    learning_rate: float = 3e-4,
    evaluate_episodes: int = 5,
    eval_interval: int = 5000,
    logger: WandbLogger | None = None,
    debug: bool = False,
) -> float:

    model = _SB3_PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=rollout_steps,
        batch_size=batch_size,
        n_epochs=ppo_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_param,
        ent_coef=entropy_coef,
        verbose=0,
        device=str(device),
    )

    callback = EvalAndLogCallback(
        algo_name="PPO",
        eval_env=eval_env,
        eval_episodes=evaluate_episodes,
        eval_interval=eval_interval,
        wandb_logger=logger,
        debug=debug,
    )

    model.learn(total_timesteps=n_steps, callback=callback, log_interval=None)

    return callback.last_eval_reward
