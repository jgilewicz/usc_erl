import torch
import gymnasium as gym

from stable_baselines3 import SAC as _SB3_SAC

from common.sb3_callback import EvalAndLogCallback
from common.wandb_logger import WandbLogger


def SAC(
    env: gym.Env,
    eval_env: gym.Env,
    n_steps: int,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
    gamma: float = 0.99,
    tau: float = 0.005,
    learning_rate: float = 3e-4,
    ent_coef: str | float = "auto",
    warmup_steps: int = 1000,
    evaluate_episodes: int = 5,
    eval_interval: int = 5000,
    logger: WandbLogger | None = None,
    debug: bool = False,
) -> float:

    model = _SB3_SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=1_000_000,
        learning_starts=warmup_steps,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0,
        device=str(device),
    )

    callback = EvalAndLogCallback(
        algo_name="SAC",
        eval_env=eval_env,
        eval_episodes=evaluate_episodes,
        eval_interval=eval_interval,
        wandb_logger=logger,
        debug=debug,
    )

    model.learn(total_timesteps=n_steps, callback=callback, log_interval=None)

    return callback.last_eval_reward
