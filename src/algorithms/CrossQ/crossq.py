import torch
import gymnasium as gym

from sb3_contrib import CrossQ as _SB3_CrossQ

from common.sb3_callback import EvalAndLogCallback
from common.wandb_logger import WandbLogger


def CrossQ(
    env: gym.Env,
    eval_env: gym.Env,
    n_steps: int,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
    gamma: float = 0.99,
    learning_rate: float = 1e-4,
    ent_coef: str | float = "auto",
    gradient_steps: int = 1,
    policy_delay: int = 1,
    warmup_steps: int = 1000,
    evaluate_episodes: int = 5,
    eval_interval: int = 5000,
    logger: WandbLogger | None = None,
    debug: bool = False,
) -> float:

    model = _SB3_CrossQ(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=1_000_000,
        learning_starts=warmup_steps,
        batch_size=batch_size,
        gamma=gamma,
        gradient_steps=gradient_steps,
        policy_delay=policy_delay,
        ent_coef=ent_coef,
        verbose=0,
        device=device,
    )

    callback = EvalAndLogCallback(
        algo_name="CrossQ",
        eval_env=eval_env,
        eval_episodes=evaluate_episodes,
        eval_interval=eval_interval,
        wandb_logger=logger,
        debug=debug,
    )

    model.learn(total_timesteps=n_steps, callback=callback, log_interval=None)

    return callback.last_eval_reward
