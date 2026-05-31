import torch
import torch.nn as nn
import numpy as np
from enum import Enum

from common.reply_buffer import Buffer
from common.utils import surrogate_fitness, rollout_policy
from modules.mc_dropout import MCDropout


class SurrogateMode(Enum):
    RANDOM = 1
    DROPOUT = 2
    ENSEMBLE = 3
    EVIDENTIAL = 4

    @staticmethod
    def to_mode(str_value: str):
        str_value = str_value.lower()
        if str_value == "random":
            return SurrogateMode.RANDOM
        elif str_value == "dropout":
            return SurrogateMode.DROPOUT
        elif str_value == "ensemble":
            return SurrogateMode.ENSEMBLE
        elif str_value == "evidential":
            return SurrogateMode.EVIDENTIAL
        else:
            raise ValueError(f"Invalid surrogate mode: {str_value}")


class SurrogateController:
    def __init__(
        self,
        evolution_module: nn.Module,
        critic: nn.Module,
        replay_buffer: Buffer,
        device: torch.device,
        surrogate_mode: SurrogateMode = SurrogateMode.RANDOM,
        omega: float = 0.5,
        rng: np.random.Generator | None = None,
        k: int = 5,
        beta: float = 1.0,
        dropout_p: float = 0.2,
        mc_samples: int = 20,
        min_uncertainty_floor: float = 0.01,
        epsilon: float = 0.10,
        percentile: int = 75,
    ):
        self.evolution_module = evolution_module
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.device = device
        self.omega = omega
        self.rng = rng if rng is not None else np.random.default_rng()
        self.k = k
        self.beta = beta

        self.dropout_p = dropout_p
        self.mc_samples = mc_samples
        self.min_uncertainty_floor = min_uncertainty_floor
        self.epsilon = epsilon
        self.percentile = percentile
        self.last_fitness = []
        self.last_uncertainty = []
        self.last_uncertainty_mean = 0.0
        self.last_uncertainty_max = 0.0
        self.last_uncertainty_threshold = 0.0
        self.mode = "real"
        self.surrogate_mode = surrogate_mode

        # EMA-based running bounds used for surrogate fitness normalisation.
        # Prevents sudden Q-value spikes from corrupting LCB selection.
        self._running_fitness_min: float | None = None
        self._running_fitness_max: float | None = None
        self._fitness_ema_alpha: float = 0.05  # slow-moving average (< 0.1)

    def generation_based_control(
        self,
        population: list[nn.Module],
        env,
        evaluate_episodes: int = 5,
        mutation_std: float = 0.05,
        mutation_prob: float = 0.1,
        elite_ratio: float = 0.2,
        total_steps: int = 0,
        warmup_steps: int = 0,
    ) -> tuple[list[nn.Module], list[float], int]:
        self.last_uncertainty = []

        if total_steps < warmup_steps:
            fitnesses, steps = self._real_evaluation(population, env, evaluate_episodes)
            self.last_fitness = fitnesses
            self.mode = "real"

            if self.surrogate_mode in (
                SurrogateMode.DROPOUT,
                SurrogateMode.ENSEMBLE,
                SurrogateMode.EVIDENTIAL,
            ):
                try:
                    if self.surrogate_mode == SurrogateMode.DROPOUT:
                        _, self.last_uncertainty = (
                            MCDropout.fitness_evaluation_mc_dropout(
                                critic=self.critic,
                                population=population,
                                replay_buffer=self.replay_buffer,
                                k=self.k,
                                device=self.device,
                                T=self.mc_samples,
                                dropout_p=self.dropout_p,
                            )
                        )
                    elif self.surrogate_mode == SurrogateMode.EVIDENTIAL:
                        k_batch = min(self.k, len(self.replay_buffer))
                        batch = self.replay_buffer.sample_latest(batch_size=k_batch)
                        obs = batch["state"].to(self.device)
                        self.critic.eval()
                        uncertainties = []
                        for policy in population:
                            policy.eval()
                            with torch.no_grad():
                                actions = policy(obs)
                                mu, v, alpha, beta = self.critic(obs, actions)
                                epistemic_var = beta / (v * (alpha - 1.0) + 1e-6)
                                epistemic_std = torch.sqrt(epistemic_var)
                            uncertainties.append(epistemic_std.mean().item())
                        self.last_uncertainty = uncertainties
                    else:
                        k_batch = min(self.k, len(self.replay_buffer))
                        batch = self.replay_buffer.sample_latest(batch_size=k_batch)
                        obs = batch["state"].to(self.device)
                        self.critic.eval()
                        uncertainties = []
                        for policy in population:
                            policy.eval()
                            with torch.no_grad():
                                actions = policy(obs)
                                _, std_q = self.critic(obs, actions)
                            uncertainties.append(std_q.mean().item())
                        self.last_uncertainty = uncertainties

                    self._update_uncertainty_metrics()
                except Exception:
                    pass

            population = self.evolution_module.evolve(
                population=population,
                fitnesses=fitnesses,
                mutation_std=mutation_std,
                mutation_prob=mutation_prob,
                elite_ratio=elite_ratio,
                surrogate_evaluation=False,
            )
            return population, fitnesses, steps

        if self.surrogate_mode == SurrogateMode.RANDOM:
            if self.rng.random() > self.omega:
                fitnesses, steps = self._real_evaluation(
                    population, env, evaluate_episodes
                )
                surrogate = False
                self.mode = "real"
            else:
                fitnesses = surrogate_fitness(
                    population, self.critic, self.replay_buffer, self.device, self.k
                )
                steps = 0
                surrogate = True
                self.mode = "surrogate"

            self.last_fitness = fitnesses

        elif self.surrogate_mode == SurrogateMode.DROPOUT:
            surrogate_fitnesses, self.last_uncertainty = (
                MCDropout.fitness_evaluation_mc_dropout(
                    critic=self.critic,
                    population=population,
                    replay_buffer=self.replay_buffer,
                    k=self.k,
                    device=self.device,
                    T=self.mc_samples,
                    dropout_p=self.dropout_p,
                )
            )
            threshold = self._update_uncertainty_metrics()

            # Normalise raw Q-values to a stable range before LCB computation
            surrogate_fitnesses = self._normalize_surrogate_fitness(surrogate_fitnesses)

            fitnesses = []
            steps = 0
            any_surrogate = False

            for i, policy in enumerate(population):
                mu_q = surrogate_fitnesses[i]
                sigma_q = self.last_uncertainty[i]

                if sigma_q > threshold or self.rng.random() < self.epsilon:
                    fit, s = self._real_evaluation([policy], env, evaluate_episodes)
                    fitnesses.append(fit[0])
                    steps += s
                else:
                    lcb_fitness = mu_q - (self.beta * sigma_q)
                    fitnesses.append(lcb_fitness)
                    any_surrogate = True

            surrogate = any_surrogate
            self.mode = (
                "mixed"
                if any_surrogate and steps > 0
                else ("surrogate" if any_surrogate else "real")
            )

            self.last_fitness = fitnesses

        elif self.surrogate_mode == SurrogateMode.ENSEMBLE:
            k = min(self.k, len(self.replay_buffer))
            batch = self.replay_buffer.sample_latest(batch_size=k)
            obs = batch["state"].to(self.device)

            self.critic.eval()
            surrogate_fitnesses = []
            uncertainties = []

            for policy in population:
                policy.eval()
                with torch.no_grad():
                    actions = policy(obs)
                    mean_q, std_q = self.critic(obs, actions)

                surrogate_fitnesses.append(mean_q.mean().item())
                uncertainties.append(std_q.mean().item())

            self.last_uncertainty = uncertainties

            threshold = self._update_uncertainty_metrics()

            # Normalise raw Q-values to a stable range before LCB computation
            surrogate_fitnesses = self._normalize_surrogate_fitness(surrogate_fitnesses)

            fitnesses = []
            steps = 0
            any_surrogate = False

            for i, policy in enumerate(population):
                mu_q = surrogate_fitnesses[i]
                sigma_q = self.last_uncertainty[i]

                if sigma_q > threshold or self.rng.random() < self.epsilon:
                    fit, s = self._real_evaluation([policy], env, evaluate_episodes)
                    fitnesses.append(fit[0])
                    steps += s
                else:
                    lcb_fitness = mu_q - (self.beta * sigma_q)
                    fitnesses.append(lcb_fitness)
                    any_surrogate = True

            surrogate = any_surrogate
            self.mode = (
                "mixed"
                if any_surrogate and steps > 0
                else ("surrogate" if any_surrogate else "real")
            )

            self.last_fitness = fitnesses

        elif self.surrogate_mode == SurrogateMode.EVIDENTIAL:
            k_batch = min(self.k, len(self.replay_buffer))
            batch = self.replay_buffer.sample_latest(batch_size=k_batch)
            obs = batch["state"].to(self.device)

            self.critic.eval()
            surrogate_fitnesses = []
            uncertainties = []

            for policy in population:
                policy.eval()
                with torch.no_grad():
                    actions = policy(obs)
                    mu, v, alpha, beta = self.critic(obs, actions)
                    epistemic_var = beta / (v * (alpha - 1.0) + 1e-6)

                    epistemic_std = torch.sqrt(epistemic_var)

                surrogate_fitnesses.append(mu.mean().item())
                uncertainties.append(epistemic_std.mean().item())

            self.last_uncertainty = uncertainties

            threshold = self._update_uncertainty_metrics()

            # Normalise raw Q-values
            surrogate_fitnesses = self._normalize_surrogate_fitness(surrogate_fitnesses)

            fitnesses = []
            steps = 0
            any_surrogate = False

            for i, policy in enumerate(population):
                mu_q = surrogate_fitnesses[i]
                sigma_q = self.last_uncertainty[i]

                if sigma_q > threshold or self.rng.random() < self.epsilon:
                    fit, s = self._real_evaluation([policy], env, evaluate_episodes)
                    fitnesses.append(fit[0])
                    steps += s
                else:
                    lcb_fitness = mu_q - (self.beta * sigma_q)
                    fitnesses.append(lcb_fitness)
                    any_surrogate = True

            surrogate = any_surrogate
            self.mode = (
                "mixed"
                if any_surrogate and steps > 0
                else ("surrogate" if any_surrogate else "real")
            )

            self.last_fitness = fitnesses

        population = self.evolution_module.evolve(
            population=population,
            fitnesses=fitnesses,
            mutation_std=mutation_std,
            mutation_prob=mutation_prob,
            elite_ratio=elite_ratio,
            surrogate_evaluation=surrogate,
        )

        return population, fitnesses, steps

    def _real_evaluation(
        self, population, env, evaluate_episodes, store_in_buffer: bool = False
    ):
        fitnesses = []
        total_steps = 0

        for individual in population:
            fitness, steps = rollout_policy(
                policy=individual,
                env=env,
                device=self.device,
                replay_buffer=self.replay_buffer,
                episodes=evaluate_episodes,
                noise_std=0.0,
                store_in_buffer=store_in_buffer,
            )
            fitnesses.append(fitness)
            total_steps += steps

        return fitnesses, total_steps

    def _update_uncertainty_metrics(self) -> float:
        if not self.last_uncertainty:
            self.last_uncertainty_mean = 0.0
            self.last_uncertainty_max = 0.0
            self.last_uncertainty_threshold = 0.0
            return 0.0

        mean_uncertainty = float(np.mean(self.last_uncertainty))
        self.last_uncertainty_mean = mean_uncertainty
        self.last_uncertainty_max = float(np.max(self.last_uncertainty))

        # Percentile threshold (configurable)
        threshold = float(np.percentile(self.last_uncertainty, self.percentile))

        self.last_uncertainty_threshold = threshold
        return threshold

    def _normalize_surrogate_fitness(self, raw_fitnesses: list[float]) -> list[float]:
        """
        Normalises raw Q-value surrogate fitnesses to [-1, 1] using EMA-tracked
        running min/max bounds, then applies a tanh soft-clip as a final safety net.

        EMA update rule:  ema = alpha * batch_min/max + (1 - alpha) * ema
        - Low alpha (0.05) → slow-moving bounds, robust to transient spikes.
        - Normalised fitness is rescaled back to the original EMA range so that
          the LCB penalty (beta * sigma_q) operates in a consistent unit space.
        """
        alpha = self._fitness_ema_alpha
        batch_min = float(np.min(raw_fitnesses))
        batch_max = float(np.max(raw_fitnesses))

        # Warm-start on first call
        if self._running_fitness_min is None:
            self._running_fitness_min = batch_min
            self._running_fitness_max = batch_max
        else:
            self._running_fitness_min = (
                alpha * batch_min + (1 - alpha) * self._running_fitness_min
            )
            self._running_fitness_max = (
                alpha * batch_max + (1 - alpha) * self._running_fitness_max
            )

        lo, hi = self._running_fitness_min, self._running_fitness_max
        span = hi - lo

        if span < 1e-6:
            # Degenerate case: all fitnesses nearly identical — return zeros
            return [0.0] * len(raw_fitnesses)

        # Map to [-1, 1], then soft-clip via tanh to bound extreme outliers
        normalised = [np.tanh((f - lo) / span * 2.0 - 1.0) for f in raw_fitnesses]
        # Rescale back to the EMA range so LCB penalty is in original units
        rescaled = [n * (span / 2.0) + (lo + hi) / 2.0 for n in normalised]
        return rescaled
