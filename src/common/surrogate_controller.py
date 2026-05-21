import torch
import torch.nn as nn
import numpy as np
from enum import Enum

from common.reply_buffer import Buffer
from common.utils import surrogate_fitness, evaluate_policy, rollout_policy
from common.mc_dropout import MCDropout


class SurrogateMode(Enum):
    RANDOM = 1
    DROPOUT = 2
    ENSEMBLE = 3

    @staticmethod
    def to_mode(str_value: str):
        str_value = str_value.lower()
        if str_value == "random":
            return SurrogateMode.RANDOM
        elif str_value == "dropout":
            return SurrogateMode.DROPOUT
        elif str_value == "ensemble":
            return SurrogateMode.ENSEMBLE
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
    ):
        self.evolution_module = evolution_module
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.device = device
        self.omega = omega
        self.rng = rng if rng is not None else np.random.default_rng()
        self.k = k

        self.last_fitness = []
        self.last_uncertainty = []
        self.uncertainty_history = []
        self.uncertainty_history_window = max(1, k)
        self.uncertainty_percentile = 70.0
        self.last_uncertainty_mean = 0.0
        self.last_uncertainty_max = 0.0
        self.last_uncertainty_threshold = 0.0
        self.mode = "real"
        self.surrogate_mode = surrogate_mode

    def _real_or_surrogate_from_uncertainty(
        self, uncertainties: list[float]
    ) -> tuple[bool, float, float, float]:
        mean_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.0
        max_uncertainty = float(np.max(uncertainties)) if uncertainties else 0.0

        recent_history = self.uncertainty_history[-self.uncertainty_history_window :]

        if len(recent_history) < self.uncertainty_history_window:
            threshold = float("inf")
            use_real = True
        else:
            moving_mean = float(np.mean(recent_history))
            percentile_threshold = float(
                np.percentile(recent_history, self.uncertainty_percentile)
            )
            threshold = 0.5 * (moving_mean + percentile_threshold)
            use_real = mean_uncertainty > threshold

        self.uncertainty_history.append(mean_uncertainty)
        self.uncertainty_history = self.uncertainty_history[
            -self.uncertainty_history_window :
        ]

        return use_real, mean_uncertainty, max_uncertainty, threshold

    def generation_based_control(
        self,
        population: list[nn.Module],
        env,
        evaluate_episodes: int = 5,
        mutation_std: float = 0.05,
        mutation_prob: float = 0.1,
        elite_ratio: float = 0.2,
    ) -> tuple[list[nn.Module], list[float], int]:
        self.last_uncertainty = []

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
                )
            )
            use_real, mean_uncertainty, max_uncertainty, threshold = (
                self._real_or_surrogate_from_uncertainty(self.last_uncertainty)
            )
            self.last_uncertainty_mean = mean_uncertainty
            self.last_uncertainty_max = max_uncertainty
            self.last_uncertainty_threshold = threshold

            if use_real:
                fitnesses, steps = self._real_evaluation(
                    population, env, evaluate_episodes
                )
                surrogate = False
                self.mode = "real"
            else:
                fitnesses = surrogate_fitnesses
                steps = 0
                surrogate = True
                self.mode = "surrogate"

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

            use_real, mean_uncertainty, max_uncertainty, threshold = (
                self._real_or_surrogate_from_uncertainty(self.last_uncertainty)
            )
            self.last_uncertainty_mean = mean_uncertainty
            self.last_uncertainty_max = max_uncertainty
            self.last_uncertainty_threshold = threshold

            if use_real:
                fitnesses, steps = self._real_evaluation(
                    population, env, evaluate_episodes
                )
                surrogate = False
                self.mode = "real"
            else:
                fitnesses = surrogate_fitnesses
                steps = 0
                surrogate = True
                self.mode = "surrogate"

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

    def _real_evaluation(self, population, env, evaluate_episodes):
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
            )
            fitnesses.append(fitness)
            total_steps += steps

        return fitnesses, total_steps
