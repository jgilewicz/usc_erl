import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy as _sb3_eval

from common.wandb_logger import WandbLogger

# Remap SB3's canonical logger keys to the flat names the download pipeline expects.
_KEY_MAP = {
    "train/critic_loss": "critic_loss",
    "train/actor_loss": "actor_loss",
    "train/ent_coef": "ent_coef",
    "train/ent_coef_loss": "ent_coef_loss",
    "train/value_loss": "critic_loss",
    "train/policy_gradient_loss": "actor_loss",
    "train/entropy_loss": "entropy",
}


class EvalAndLogCallback(BaseCallback):
    def __init__(
        self,
        algo_name: str,
        eval_env: gym.Env,
        eval_episodes: int,
        eval_interval: int,
        wandb_logger: WandbLogger | None,
        debug: bool,
    ):
        super().__init__(verbose=0)
        self._algo_name = algo_name
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
            if hasattr(self.model, "logger"):
                for k, v in self.model.logger.name_to_value.items():
                    metrics[_KEY_MAP.get(k, k.replace("/", "_"))] = v
            self._wandb_logger.log(metrics, step=self.num_timesteps)

        if self._debug:
            print(
                f"[{self._algo_name}] steps={self.num_timesteps:>8d}  "
                f"eval_reward={self.last_eval_reward:.2f}"
            )

        return True
