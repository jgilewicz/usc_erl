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
        beta: float = 1.0,
    ):
        self.evolution_module = evolution_module
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.device = device
        self.omega = omega
        self.rng = rng if rng is not None else np.random.default_rng()
        self.k = k
        self.beta = beta

        self.last_fitness = []
        self.last_uncertainty = []
        self.last_uncertainty_mean = 0.0
        self.last_uncertainty_max = 0.0
        self.last_uncertainty_threshold = 0.0
        self.mode = "real"
        self.surrogate_mode = surrogate_mode


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
            mean_uncertainty = float(np.mean(self.last_uncertainty)) if self.last_uncertainty else 0.0
            std_uncertainty = float(np.std(self.last_uncertainty)) if self.last_uncertainty else 0.0
            threshold = mean_uncertainty + std_uncertainty

            self.last_uncertainty_mean = mean_uncertainty
            self.last_uncertainty_max = float(np.max(self.last_uncertainty)) if self.last_uncertainty else 0.0
            self.last_uncertainty_threshold = threshold

            fitnesses = []
            steps = 0
            any_surrogate = False

            for i, policy in enumerate(population):
                mu_q = surrogate_fitnesses[i]
                sigma_q = self.last_uncertainty[i]

                if sigma_q > threshold:
                    # High uncertainty -> Active Exploration / Safety Verification
                    fit, s = self._real_evaluation([policy], env, evaluate_episodes)
                    fitnesses.append(fit[0])
                    steps += s
                else:
                    # Low uncertainty -> Save steps, apply safe LCB
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

            mean_uncertainty = float(np.mean(self.last_uncertainty)) if self.last_uncertainty else 0.0
            std_uncertainty = float(np.std(self.last_uncertainty)) if self.last_uncertainty else 0.0
            threshold = mean_uncertainty + std_uncertainty

            self.last_uncertainty_mean = mean_uncertainty
            self.last_uncertainty_max = float(np.max(self.last_uncertainty)) if self.last_uncertainty else 0.0
            self.last_uncertainty_threshold = threshold

            fitnesses = []
            steps = 0
            any_surrogate = False

            for i, policy in enumerate(population):
                mu_q = surrogate_fitnesses[i]
                sigma_q = self.last_uncertainty[i]

                if sigma_q > threshold:
                    # High uncertainty -> Active Exploration / Safety Verification
                    fit, s = self._real_evaluation([policy], env, evaluate_episodes)
                    fitnesses.append(fit[0])
                    steps += s
                else:
                    # Low uncertainty -> Save steps, apply safe LCB
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
