from collections import deque

import gymnasium as gym
import numpy as np
import torch
from scipy.stats import norm

from common.reply_buffer import Buffer, Transition
from modules.deep_modules import Actor

CAL_POOL_SIZE = 500
CAL_MIN_PAIRS = 30
DEFAULT_HORIZON = 1000


class EpisodeRunner:
    # Steps the env in place, so only one runner may be live per env:
    # constructing a second one resets the episode the first is holding.
    def __init__(
        self,
        policy: Actor,
        env: gym.Env,
        device: torch.device,
        replay_buffer: Buffer | None,
        store_states: bool = False,
    ) -> None:
        self.policy = policy
        self.env = env
        self.device = device
        self.replay_buffer = replay_buffer
        self.store_states = store_states

        policy.eval()
        obs, _ = env.reset()

        self.last_obs = obs
        self.cum_reward = 0.0
        self.steps = 0
        self.terminated = False
        self.done = False
        self.rewards: list[float] = []
        self.states: list[np.ndarray] = [obs] if store_states else []

    def advance(self, n_steps: int | None = None) -> None:
        remaining = n_steps
        while not self.done and (remaining is None or remaining > 0):
            self._step()
            if remaining is not None:
                remaining -= 1

    def _step(self) -> None:
        obs_t = torch.tensor(
            self.last_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(obs_t).squeeze(0).cpu().numpy()
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        next_obs, reward, terminated, truncated, _ = self.env.step(action)

        if self.replay_buffer is not None:
            self.replay_buffer.add(
                Transition(
                    state=self.last_obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=terminated,
                )
            )

        self.last_obs = next_obs
        self.cum_reward += float(reward)
        self.rewards.append(float(reward))
        if self.store_states:
            self.states.append(next_obs)
        self.steps += 1
        self.terminated = bool(terminated)
        self.done = bool(terminated or truncated)


class TailCalibrator:
    def __init__(
        self,
        gamma: float,
        horizon: int | None,
        pool_size: int = CAL_POOL_SIZE,
        min_pairs: int = CAL_MIN_PAIRS,
    ) -> None:
        self.gamma = gamma
        self.horizon = float(horizon) if horizon else float(DEFAULT_HORIZON)
        self._horizon_is_fixed = horizon is not None
        self.min_pairs = min_pairs
        self._pairs: deque[tuple[float, float]] = deque(maxlen=pool_size)
        self.a = 0.0
        self.b = 0.0
        self.calibrated = False

    @property
    def n_pairs(self) -> int:
        return len(self._pairs)

    @property
    def fitted(self) -> bool:
        return self.n_pairs >= self.min_pairs

    def observe_episode_length(self, length: int) -> None:
        if not self._horizon_is_fixed:
            self.horizon = max(self.horizon, float(length))

    def feature(self, h: int, q_mean: float) -> float:
        return max(self.horizon - h, 0.0) * (1.0 - self.gamma) * q_mean

    def add_trajectory(
        self, rewards: np.ndarray, q_mean: np.ndarray, chunk: int
    ) -> None:
        total = float(rewards.sum())
        for h in range(chunk, len(rewards), chunk):
            self._pairs.append(
                (
                    self.feature(h, float(q_mean[h])),
                    total - float(rewards[:h].sum()),
                )
            )

    def fit(self) -> None:
        if not self.fitted:
            self.a, self.b = 0.0, 0.0
            return
        design = np.array([[feature, 1.0] for feature, _ in self._pairs])
        target = np.array([remaining for _, remaining in self._pairs])
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        self.a, self.b = float(coef[0]), float(coef[1])
        self.calibrated = True

    def estimate(
        self,
        partial_return: float,
        h: int,
        q_mean: float,
        q_std: float,
        alive: bool,
    ) -> tuple[float, float]:
        if not alive:
            return partial_return, 0.0
        scale = max(self.horizon - h, 0.0) * (1.0 - self.gamma)
        f_hat = partial_return + self.a * scale * q_mean + self.b
        return f_hat, abs(self.a) * scale * q_std


def stop_probability(f_hat: float, sigma: float, f_cut: float) -> float:
    if sigma <= 0.0:
        return 0.0
    return float(norm.cdf(-abs(f_hat - f_cut) / sigma))
