import copy
from enum import Enum
import numpy as np
import torch
import torch.nn as nn

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
        crossover_prob: float = 0.0,
        crossover_mode: str = "none",
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
        self.crossover_prob = crossover_prob
        self.crossover_mode = crossover_mode
        self.last_fitness = []
        self.last_uncertainty = []
        self.last_uncertainty_mean = 0.0
        self.last_uncertainty_max = 0.0
        self.last_uncertainty_threshold = 0.0
        self.mode = "real"
        self.surrogate_mode = surrogate_mode

        self._running_fitness_min: float | None = None
        self._running_fitness_max: float | None = None
        self._fitness_ema_alpha: float = 0.05

        self._running_real_min: float | None = None
        self._running_real_max: float | None = None

        # Tracking the best real-evaluated actor to ensure elite protection
        self.best_real_actor_state = None
        self.best_real_fitness = -float("inf")

        # Updated each call to generation_based_control for use by callers
        self.last_elite_indices: list[int] = []
        self.last_unselect_indices: list[int] = []

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
        mutation_fraction: float = 0.1,
    ) -> tuple[list[nn.Module], list[float], int, bool]:
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
                                epistemic_var = torch.nan_to_num(
                                    epistemic_var, nan=0.0, posinf=1e3, neginf=0.0
                                )
                                epistemic_std = torch.sqrt(epistemic_var.clamp(min=0.0))
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

            population, new_elitists, unselect_indices = self.evolution_module.evolve(
                population=population,
                fitnesses=fitnesses,
                mutation_std=mutation_std,
                mutation_prob=mutation_prob,
                elite_ratio=elite_ratio,
                crossover_prob=self.crossover_prob,
                crossover_mode=self.crossover_mode,
                replay_buffer=self.replay_buffer,
                mutation_fraction=mutation_fraction,
            )
            self.last_elite_indices = new_elitists
            self.last_unselect_indices = unselect_indices

            # Post-evolution elite injection: restore best-ever real actor into worst slot
            if self.best_real_actor_state is not None and fitnesses:
                worst_index = int(np.argmin(fitnesses))
                if worst_index not in new_elitists:
                    target = worst_index
                elif unselect_indices:
                    target = unselect_indices[-1]
                else:
                    target = new_elitists[-1]
                population[target].load_state_dict(self.best_real_actor_state)
                fitnesses[target] = self.best_real_fitness
                if target < len(self.last_fitness):
                    self.last_fitness[target] = self.best_real_fitness

            return population, fitnesses, steps, True

        # Snapshot the critic once before surrogate fitness evaluation to ensure
        # consistent rankings across all individuals in this generation.
        surrogate_critic = copy.deepcopy(self.critic)
        surrogate_critic.eval()

        if self.surrogate_mode == SurrogateMode.RANDOM:
            scaled_fitnesses = self._normalize_surrogate_fitness(
                surrogate_fitness(
                    population, surrogate_critic, self.replay_buffer, self.device, self.k
                )
            )

            fitnesses = []
            steps = 0
            any_surrogate = False

            for i, policy in enumerate(population):
                if self.rng.random() > self.omega:
                    fit, s = self._real_evaluation([policy], env, evaluate_episodes)
                    fitnesses.append(fit[0])
                    steps += s
                else:
                    fitnesses.append(scaled_fitnesses[i])
                    any_surrogate = True

            self.last_fitness = fitnesses

        elif self.surrogate_mode == SurrogateMode.DROPOUT:
            surrogate_fitnesses, self.last_uncertainty = (
                MCDropout.fitness_evaluation_mc_dropout(
                    critic=surrogate_critic,
                    population=population,
                    replay_buffer=self.replay_buffer,
                    k=self.k,
                    device=self.device,
                    T=self.mc_samples,
                    dropout_p=self.dropout_p,
                )
            )
            threshold = self._update_uncertainty_metrics()

            lcb_q_values = []
            for i in range(len(population)):
                mu_q = surrogate_fitnesses[i]
                sigma_q = self.last_uncertainty[i]
                lcb_q = float(np.clip(mu_q - self.beta * sigma_q, a_min=-5000.0, a_max=None))
                lcb_q_values.append(lcb_q)

            scaled_fitnesses = self._normalize_surrogate_fitness(lcb_q_values)

            fitnesses = []
            steps = 0
            any_surrogate = False

            for i, policy in enumerate(population):
                sigma_q = self.last_uncertainty[i]
                if sigma_q > threshold or self.rng.random() < self.epsilon:
                    fit, s = self._real_evaluation([policy], env, evaluate_episodes)
                    fitnesses.append(fit[0])
                    steps += s
                else:
                    fitnesses.append(scaled_fitnesses[i])
                    any_surrogate = True

            self.last_fitness = fitnesses

        elif self.surrogate_mode == SurrogateMode.ENSEMBLE:
            k = min(self.k, len(self.replay_buffer))
            batch = self.replay_buffer.sample_latest(batch_size=k)
            obs = batch["state"].to(self.device)

            surrogate_fitnesses = []
            uncertainties = []
            for policy in population:
                policy.eval()
                with torch.no_grad():
                    actions = policy(obs)
                    mean_q, std_q = surrogate_critic(obs, actions)
                surrogate_fitnesses.append(mean_q.mean().item())
                uncertainties.append(std_q.mean().item())

            self.last_uncertainty = uncertainties
            threshold = self._update_uncertainty_metrics()

            lcb_q_values = []
            for i in range(len(population)):
                mu_q = surrogate_fitnesses[i]
                sigma_q = self.last_uncertainty[i]
                lcb_q = float(np.clip(mu_q - self.beta * sigma_q, a_min=-5000.0, a_max=None))
                lcb_q_values.append(lcb_q)

            scaled_fitnesses = self._normalize_surrogate_fitness(lcb_q_values)

            fitnesses = []
            steps = 0
            any_surrogate = False

            for i, policy in enumerate(population):
                sigma_q = self.last_uncertainty[i]
                if sigma_q > threshold or self.rng.random() < self.epsilon:
                    fit, s = self._real_evaluation([policy], env, evaluate_episodes)
                    fitnesses.append(fit[0])
                    steps += s
                else:
                    fitnesses.append(scaled_fitnesses[i])
                    any_surrogate = True

            self.last_fitness = fitnesses

        elif self.surrogate_mode == SurrogateMode.EVIDENTIAL:
            k_batch = min(self.k, len(self.replay_buffer))
            batch = self.replay_buffer.sample_latest(batch_size=k_batch)
            obs = batch["state"].to(self.device)

            surrogate_fitnesses = []
            uncertainties = []
            for policy in population:
                policy.eval()
                with torch.no_grad():
                    actions = policy(obs)
                    mu, v, alpha, beta = surrogate_critic(obs, actions)
                    epistemic_var = beta / (v * (alpha - 1.0) + 1e-6)
                    epistemic_var = torch.nan_to_num(epistemic_var, nan=0.0, posinf=1e3, neginf=0.0)
                    epistemic_std = torch.sqrt(epistemic_var.clamp(min=0.0))
                surrogate_fitnesses.append(mu.mean().item())
                uncertainties.append(epistemic_std.mean().item())

            self.last_uncertainty = uncertainties
            threshold = self._update_uncertainty_metrics()

            lcb_q_values = []
            for i in range(len(population)):
                mu_q = surrogate_fitnesses[i]
                sigma_q = self.last_uncertainty[i]
                lcb_q = float(np.clip(mu_q - self.beta * sigma_q, a_min=-5000.0, a_max=None))
                lcb_q_values.append(lcb_q)

            scaled_fitnesses = self._normalize_surrogate_fitness(lcb_q_values)

            fitnesses = []
            steps = 0
            any_surrogate = False

            for i, policy in enumerate(population):
                sigma_q = self.last_uncertainty[i]
                if sigma_q > threshold or self.rng.random() < self.epsilon:
                    fit, s = self._real_evaluation([policy], env, evaluate_episodes)
                    fitnesses.append(fit[0])
                    steps += s
                else:
                    fitnesses.append(scaled_fitnesses[i])
                    any_surrogate = True

            self.last_fitness = fitnesses

        used_real_eval = not any_surrogate

        population, new_elitists, unselect_indices = self.evolution_module.evolve(
            population=population,
            fitnesses=fitnesses,
            mutation_std=mutation_std,
            mutation_prob=mutation_prob,
            elite_ratio=elite_ratio,
            crossover_prob=self.crossover_prob,
            crossover_mode=self.crossover_mode,
            replay_buffer=self.replay_buffer,
            mutation_fraction=mutation_fraction,
        )
        self.last_elite_indices = new_elitists
        self.last_unselect_indices = unselect_indices

        # Post-evolution elite injection (Global Anchor): restore best-ever real actor into worst slot
        if self.best_real_actor_state is not None and fitnesses:
            worst_index = int(np.argmin(fitnesses))
            if worst_index not in new_elitists:
                target = worst_index
            elif unselect_indices:
                target = unselect_indices[-1]
            else:
                target = new_elitists[-1]
            population[target].load_state_dict(self.best_real_actor_state)
            fitnesses[target] = self.best_real_fitness
            if target < len(self.last_fitness):
                self.last_fitness[target] = self.best_real_fitness

        return population, fitnesses, steps, used_real_eval

    def _update_real_bounds(self, real_values: list[float]):
        if not real_values:
            return
        batch_min = float(np.min(real_values))
        batch_max = float(np.max(real_values))
        alpha = self._fitness_ema_alpha

        if self._running_real_min is None:
            self._running_real_min = batch_min
            self._running_real_max = batch_max
        else:
            self._running_real_min = (
                alpha * batch_min + (1 - alpha) * self._running_real_min
            )
            self._running_real_max = (
                alpha * batch_max + (1 - alpha) * self._running_real_max
            )

    def _real_evaluation(
        self, population, env, evaluate_episodes, store_in_buffer: bool = True
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

            # Store the best real actor regardless of the surrogate predictions
            if fitness > self.best_real_fitness:
                self.best_real_fitness = fitness
                self.best_real_actor_state = copy.deepcopy(individual.state_dict())

            fitnesses.append(fitness)
            total_steps += steps

        self._update_real_bounds(fitnesses)

        return fitnesses, total_steps

    def _update_uncertainty_metrics(self) -> float:
        if not self.last_uncertainty:
            self.last_uncertainty_mean = 0.0
            self.last_uncertainty_max = 0.0
            self.last_uncertainty_threshold = 0.0
            return 0.0

        uncertainties = np.nan_to_num(
            np.array(self.last_uncertainty, dtype=np.float64),
            nan=0.0,
            posinf=1e3,
            neginf=0.0,
        )

        mean_uncertainty = float(np.mean(uncertainties))
        self.last_uncertainty_mean = mean_uncertainty
        self.last_uncertainty_max = float(np.max(uncertainties))

        threshold = float(np.percentile(uncertainties, self.percentile))
        self.last_uncertainty_threshold = threshold

        return threshold

    def _normalize_surrogate_fitness(self, raw_fitnesses: list[float]) -> list[float]:
        alpha = self._fitness_ema_alpha
        batch_min = float(np.min(raw_fitnesses))
        batch_max = float(np.max(raw_fitnesses))

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
            span = 1e-6

        normalised = [np.tanh((f - lo) / span * 2.0 - 1.0) for f in raw_fitnesses]

        if self._running_real_min is None or self._running_real_max is None:
            lo_real, hi_real = lo, hi
        else:
            lo_real, hi_real = self._running_real_min, self._running_real_max

        span_real = hi_real - lo_real
        rescaled = [((n + 1.0) / 2.0) * span_real + lo_real for n in normalised]

        return rescaled
