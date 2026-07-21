import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import DDPG as _SB3_DDPG
from stable_baselines3.common.noise import NormalActionNoise

from common.sb3_callback import EvalAndLogCallback
from common.wandb_logger import WandbLogger


def DDPG(
    env: gym.Env,
    eval_env: gym.Env,
    n_steps: int,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
    gamma: float = 0.99,
    tau: float = 0.005,
    learning_rate: float = 3e-4,
    exploration_noise_std: float = 0.1,
    warmup_steps: int = 1000,
    evaluate_episodes: int = 5,
    eval_interval: int = 5000,
    logger: WandbLogger | None = None,
    debug: bool = False,
) -> float:

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=exploration_noise_std * np.ones(n_actions)
    )

    model = _SB3_DDPG(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=1_000_000,
        learning_starts=warmup_steps,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        action_noise=action_noise,
        verbose=0,
        device=str(device),
    )

    callback = EvalAndLogCallback(
        algo_name="DDPG",
        eval_env=eval_env,
        eval_episodes=evaluate_episodes,
        eval_interval=eval_interval,
        wandb_logger=logger,
        debug=debug,
    )

    model.learn(total_timesteps=n_steps, callback=callback, log_interval=None)

    return callback.last_eval_reward
