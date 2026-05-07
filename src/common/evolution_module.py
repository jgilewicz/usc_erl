import torch
import torch.nn as nn
import numpy as np
import copy

from .utils import get_flat_params, set_flat_params


class EvolutionModule:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        net: nn.Module,
        device: torch.device = "cpu",
    ) -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.critic_net = net
        self.device = device

        self.best_actor = None
        self.best_fitness = float("-inf")

    def _crossover(
        self, p_a: nn.Module, p_b: nn.Module, crossover_rate: float = 0.5
    ) -> nn.Module:
        params_a = get_flat_params(p_a)
        params_b = get_flat_params(p_b)

        mask = torch.rand_like(params_a) < crossover_rate
        child_params = torch.where(mask, params_a, params_b)

        child = copy.deepcopy(p_a).to(self.device)
        set_flat_params(child, child_params, self.device)
        return child

    def _mutate(
        self, model: nn.Module, mutation_rate: float = 0.01, mutation_prob: float = 0.1
    ) -> nn.Module:
        with torch.no_grad():
            for param in model.parameters():
                mask = torch.rand_like(param) < mutation_prob
                noise = torch.randn_like(param) * mutation_rate
                param.add_(mask.float() * noise)
        return model

    def evolve(
        self,
        population: list[nn.Module],
        fitnesses: list[float],
        mutation_std: float = 0.05,
        mutation_prob: float = 0.5,
        elite_ratio: float = 0.5,
        surrogate_evaluation: bool = False,
    ) -> list[nn.Module]:
        elite_count = max(1, int(len(population) * elite_ratio))
        ranked_indices = np.argsort(fitnesses)[::-1]
        elites = [population[i] for i in ranked_indices[:elite_count]]

        best_fitness = max(fitnesses)

        if best_fitness > self.best_fitness and not surrogate_evaluation:
            self.best_fitness = best_fitness
            self.best_actor = copy.deepcopy(elites[0]).to(self.device)

        elite = (
            self.best_actor
            if surrogate_evaluation and self.best_actor is not None
            else elites[0]
        )
        new_population = (
            [copy.deepcopy(elite).to(self.device)] if elite is not None else []
        )

        while len(new_population) < len(population):
            parent_a, parent_b = np.random.choice(elites, size=2)
            parent_a.to(self.device)
            parent_b.to(self.device)

            child = self._crossover(parent_a, parent_b)
            child = self._mutate(
                child, mutation_rate=mutation_std, mutation_prob=mutation_prob
            )
            new_population.append(child)

        return new_population
