import torch
import gymnasium as gym

from sb3_contrib import CrossQ as _SB3_CrossQ
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy as _sb3_eval

from common.wandb_logger import WandbLogger


class _EvalAndLogCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        eval_episodes: int,
        eval_interval: int,
        wandb_logger: WandbLogger | None,
        debug: bool,
    ):
        super().__init__(verbose=0)
        self._eval_env = eval_env
        self._eval_episodes = eval_episodes
        self._eval_interval = eval_interval
        self._wandb_logger = wandb_logger
        self._debug = debug
        self.last_eval_reward = 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps % self._eval_interval != 0:
            return True

        mean_reward, _ = _sb3_eval(
            self.model,
            self._eval_env,
            n_eval_episodes=self._eval_episodes,
            deterministic=True,
            warn=False,
        )
        self.last_eval_reward = float(mean_reward)

        if self._wandb_logger is not None:
            metrics: dict = {
                "eval_reward": self.last_eval_reward,
                "total_steps": self.num_timesteps,
            }
            _KEY_MAP = {
                "train/critic_loss": "critic_loss",
                "train/actor_loss": "actor_loss",
                "train/ent_coef": "ent_coef",
                "train/ent_coef_loss": "ent_coef_loss",
            }
            if hasattr(self.model, "logger"):
                for k, v in self.model.logger.name_to_value.items():
                    metrics[_KEY_MAP.get(k, k.replace("/", "_"))] = v
            self._wandb_logger.log(metrics, step=self.num_timesteps)

        if self._debug:
            print(
                f"[CrossQ] steps={self.num_timesteps:>8d}  eval_reward={self.last_eval_reward:.2f}"
            )

        return True


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

    callback = _EvalAndLogCallback(
        eval_env=eval_env,
        eval_episodes=evaluate_episodes,
        eval_interval=eval_interval,
        wandb_logger=logger,
        debug=debug,
    )

    model.learn(total_timesteps=n_steps, callback=callback, log_interval=None)

    return callback.last_eval_reward
