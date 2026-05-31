from __future__ import annotations

import copy

import numpy as np
import torch

from common.modules import Actor, Critic
from common.utils import get_flat_params, set_flat_params


class EvolutionModule:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        net: Critic,
        device: torch.device,
    ) -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.net = net
        self.device = device

    def _clone_and_mutate(
        self,
        parent: Actor,
        mutation_std: float,
        mutation_prob: float,
        rng: np.random.Generator,
    ) -> Actor:
        child = copy.deepcopy(parent).to(self.device)
        flat_params = get_flat_params(child)
        noise = torch.zeros_like(flat_params)
        mask = torch.from_numpy(rng.random(flat_params.numel()) < mutation_prob).to(
            flat_params.device
        )
        if mask.any():
            noise_values = torch.from_numpy(
                rng.normal(0.0, mutation_std, size=int(mask.sum().item()))
            ).to(flat_params.device, dtype=flat_params.dtype)
            noise[mask] = noise_values
        set_flat_params(child, flat_params + noise, device=self.device)
        return child

    def evolve(
        self,
        population: list[Actor],
        fitnesses: list[float],
        mutation_std: float,
        mutation_prob: float,
        elite_ratio: float,
        surrogate_evaluation: bool = False,
    ) -> list[Actor]:
        if not population:
            return population

        del surrogate_evaluation

        rng = np.random.default_rng()
        ranked_indices = sorted(
            range(len(population)), key=lambda index: fitnesses[index], reverse=True
        )
        elite_count = max(1, int(len(population) * elite_ratio))
        elite_indices = ranked_indices[:elite_count]
        elites = [
            copy.deepcopy(population[index]).to(self.device) for index in elite_indices
        ]

        new_population: list[Actor] = elites.copy()
        while len(new_population) < len(population):
            parent = elites[int(rng.integers(0, len(elites)))]
            new_population.append(
                self._clone_and_mutate(
                    parent=parent,
                    mutation_std=mutation_std,
                    mutation_prob=mutation_prob,
                    rng=rng,
                )
            )

        return new_population[: len(population)]

    def sync_rl_to_pop(
        self,
        actor: Actor,
        population: list[Actor],
        fitnesses: list[float],
    ) -> None:
        if not population:
            return

        best_index = int(np.argmax(fitnesses)) if fitnesses else 0
        population[best_index].load_state_dict(copy.deepcopy(actor.state_dict()))
