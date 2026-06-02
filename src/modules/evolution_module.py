from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.deep_modules import Actor, Critic
from common.reply_buffer import Buffer
from common.utils import get_flat_params, set_flat_params


class EvolutionModule:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        net: Critic | nn.Module,
        device: torch.device,
    ) -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.net = net
        self.device = device

    def selection_tournament(
        self,
        ranked_indices: list[int],
        num_offsprings: int,
        tournament_size: int = 3,
    ) -> list[int]:
        """Conduct rank-based min-tournament selection on a sorted index list."""
        total_choices = len(ranked_indices)
        offsprings = []
        rng = np.random.default_rng()
        for _ in range(num_offsprings):
            # Select random index candidates from the ranked list
            candidates = rng.integers(0, total_choices, size=tournament_size)
            # Min index yields the highest rank (best performance)
            winner = int(np.min(candidates))
            offsprings.append(ranked_indices[winner])

        # Deduplicate and ensure even number of offsprings
        offsprings = list(set(offsprings))
        if offsprings and len(offsprings) % 2 != 0:
            offsprings.append(offsprings[rng.integers(0, len(offsprings))])
        return offsprings

    def crossover_inplace(self, gene1: Actor, gene2: Actor) -> None:
        """Conduct row-level parameter crossover in place on nn.Linear weights and biases."""
        rng = np.random.default_rng()
        for param1, param2 in zip(gene1.parameters(), gene2.parameters()):
            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2:  # Weights
                num_variables = W1.shape[0]
                num_cross_overs = int(rng.integers(0, max(2, int(num_variables * 0.3))))
                for _ in range(num_cross_overs):
                    ind_cr = int(rng.integers(0, W1.shape[0]))
                    if rng.random() < 0.5:
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        W2[ind_cr, :] = W1[ind_cr, :]
            elif len(W1.shape) == 1:  # Bias
                if rng.random() < 0.8:
                    continue
                num_variables = W1.shape[0]
                ind_cr = int(rng.integers(0, W1.shape[0]))
                if rng.random() < 0.5:
                    W1[ind_cr] = W2[ind_cr]
                else:
                    W2[ind_cr] = W1[ind_cr]

    def distillation_crossover(
        self,
        child: Actor,
        parent1: Actor,
        parent2: Actor,
        replay_buffer: Buffer,
        lr: float = 1e-3,
        epochs: int = 12,
        batch_size: int = 128,
    ) -> None:
        """Conduct Q-filtered distillation crossover to train the child network."""
        if len(replay_buffer) < batch_size:
            return

        optimizer = torch.optim.Adam(child.parameters(), lr=lr)
        child.train()
        parent1.eval()
        parent2.eval()

        critic_was_training = self.net.training
        self.net.eval()

        for _ in range(epochs):
            batch = replay_buffer.sample(batch_size=batch_size)
            states = batch["state"].to(self.device)

            with torch.no_grad():
                a1 = parent1(states)
                a2 = parent2(states)

                q1 = self.net(states, a1)
                q2 = self.net(states, a2)

                if isinstance(q1, tuple):
                    q1 = q1[0]
                if isinstance(q2, tuple):
                    q2 = q2[0]

                # Q-filtering: choose action with higher Q-value
                target_actions = torch.where(q1 > q2, a1, a2)

            optimizer.zero_grad()
            child_actions = child(states)
            loss = F.mse_loss(child_actions, target_actions)
            loss.backward()
            optimizer.step()

        self.net.train(critic_was_training)

    def _clone_and_mutate(
        self,
        parent: Actor,
        mutation_std: float,
        mutation_fraction: float,
        rng: np.random.Generator,
    ) -> Actor:
        """Mutate a deep copy of the parent using 3-strength fractional mutations.

        Args:
            mutation_fraction: Fraction of *weights* to mutate (default 0.1 = 10%).
                               Separate from the per-individual mutation_prob used in evolve.
        """
        child = copy.deepcopy(parent).to(self.device)
        flat_params = get_flat_params(child)
        noise = torch.zeros_like(flat_params)

        # Sparse fractional mutation: mutate only mutation_fraction of the weights
        mask = torch.from_numpy(rng.random(flat_params.numel()) < mutation_fraction).to(
            flat_params.device
        )

        if mask.any():
            mutated_size = int(mask.sum().item())
            rands = rng.random(mutated_size)
            noise_values = torch.zeros(mutated_size, device=flat_params.device, dtype=flat_params.dtype)
            flat_params_mutated = flat_params[mask]

            # Super mutation (5% of mutated weights), Reset (5%), Normal mutation (90%)
            super_mut_mask = rands < 0.05
            reset_mask = (rands >= 0.05) & (rands < 0.10)
            normal_mut_mask = rands >= 0.10

            # 1. Normal mutation: proportional to parameter magnitude W
            if np.any(normal_mut_mask):
                normal_noise = torch.from_numpy(
                    rng.normal(0.0, mutation_std, size=int(normal_mut_mask.sum()))
                ).to(flat_params.device, dtype=flat_params.dtype)
                noise_values[normal_mut_mask] = normal_noise * flat_params_mutated[normal_mut_mask]

            # 2. Super mutation: 10x strength, proportional to parameter magnitude W
            if np.any(super_mut_mask):
                super_noise = torch.from_numpy(
                    rng.normal(0.0, 10.0 * mutation_std, size=int(super_mut_mask.sum()))
                ).to(flat_params.device, dtype=flat_params.dtype)
                noise_values[super_mut_mask] = super_noise * flat_params_mutated[super_mut_mask]

            # 3. Reset mutation: reset parameter to standard normal N(0, 1)
            if np.any(reset_mask):
                reset_noise = torch.from_numpy(
                    rng.normal(0.0, 1.0, size=int(reset_mask.sum()))
                ).to(flat_params.device, dtype=flat_params.dtype)
                noise_values[reset_mask] = reset_noise - flat_params_mutated[reset_mask]

            # Clamping parameters within hard limits to avoid explosion [-1e6, 1e6]
            new_params = torch.clamp(flat_params + noise.index_copy(0, torch.where(mask)[0], noise_values), -1e6, 1e6)
            set_flat_params(child, new_params, device=self.device)
        else:
            set_flat_params(child, flat_params, device=self.device)

        return child

    def evolve(
        self,
        population: list[Actor],
        fitnesses: list[float],
        mutation_std: float,
        mutation_prob: float,
        elite_ratio: float,
        crossover_prob: float = 0.0,
        crossover_mode: str = "none",
        replay_buffer: Buffer | None = None,
        mutation_fraction: float = 0.1,
    ) -> tuple[list[Actor], list[int], list[int]]:
        """Perform rank-based SSNE epoch matching the reference epoch() implementation.

        Returns:
            (new_population, new_elitists, unselect_indices) where new_elitists are the
            target slots that received elite copies (protected from mutation), and
            unselect_indices are the remaining non-offspring, non-elite slots after cloning.
        """
        if not population:
            return population, [], []

        rng = np.random.default_rng()
        pop_size = len(population)

        # 1. Rank indices by fitness (0 = best)
        ranked_indices = sorted(
            range(pop_size), key=lambda index: fitnesses[index], reverse=True
        )

        # Elites = original top-k positions by fitness
        elite_count = max(2 if pop_size >= 2 else 1, int(pop_size * elite_ratio))
        elitist_index = ranked_indices[:elite_count]

        # Tournament selects offsprings (n = pop_size - elite_count); may overlap with elitist_index
        num_offsprings = pop_size - elite_count
        offspring_indices = self.selection_tournament(
            ranked_indices=ranked_indices,
            num_offsprings=num_offsprings,
            tournament_size=3,
        )

        # Unselects = slots not in offspring AND not in elitist_index
        unselect_indices = [
            i for i in range(pop_size)
            if i not in offspring_indices and i not in elitist_index
        ]
        rng.shuffle(unselect_indices)

        # Deep copy population into new_population
        new_population = [copy.deepcopy(actor).to(self.device) for actor in population]

        # Elite copy step: each elite is cloned into an unselect slot (or offspring slot as fallback).
        # new_elitists = the TARGET slots that received an elite copy; these are mutation-protected.
        new_elitists: list[int] = []
        for elite_idx in elitist_index:
            if unselect_indices:
                target = unselect_indices.pop(0)
            elif offspring_indices:
                target = offspring_indices.pop(0)
            else:
                break
            new_population[target].load_state_dict(population[elite_idx].state_dict())
            new_elitists.append(target)

        # Crossover for remaining unselects: random new_elitist × random offspring
        if unselect_indices:
            if len(unselect_indices) % 2 != 0:
                unselect_indices.append(unselect_indices[rng.integers(0, len(unselect_indices))])

            for i, j in zip(unselect_indices[0::2], unselect_indices[1::2]):
                parent1_idx = int(rng.choice(new_elitists)) if new_elitists else int(rng.choice(ranked_indices))
                parent2_idx = int(rng.choice(offspring_indices)) if offspring_indices else int(rng.choice(ranked_indices))

                new_population[i].load_state_dict(new_population[parent1_idx].state_dict())
                new_population[j].load_state_dict(population[parent2_idx].state_dict())

                if crossover_mode == "parameter":
                    self.crossover_inplace(new_population[i], new_population[j])
                elif crossover_mode == "distillation" and replay_buffer is not None:
                    self.distillation_crossover(
                        child=new_population[i],
                        parent1=copy.deepcopy(new_population[parent1_idx]).to(self.device),
                        parent2=copy.deepcopy(population[parent2_idx]).to(self.device),
                        replay_buffer=replay_buffer,
                    )
                    self.distillation_crossover(
                        child=new_population[j],
                        parent1=copy.deepcopy(population[parent2_idx]).to(self.device),
                        parent2=copy.deepcopy(new_population[parent1_idx]).to(self.device),
                        replay_buffer=replay_buffer,
                    )

        # Crossover for offsprings: pairs with crossover_prob
        if offspring_indices:
            if len(offspring_indices) % 2 != 0:
                offspring_indices.append(offspring_indices[rng.integers(0, len(offspring_indices))])

            for i, j in zip(offspring_indices[0::2], offspring_indices[1::2]):
                if rng.random() < crossover_prob:
                    if crossover_mode == "parameter":
                        self.crossover_inplace(new_population[i], new_population[j])
                    elif crossover_mode == "distillation" and replay_buffer is not None:
                        parent1_copy = copy.deepcopy(new_population[i]).to(self.device)
                        parent2_copy = copy.deepcopy(new_population[j]).to(self.device)
                        self.distillation_crossover(
                            child=new_population[i],
                            parent1=parent1_copy,
                            parent2=parent2_copy,
                            replay_buffer=replay_buffer,
                        )
                        self.distillation_crossover(
                            child=new_population[j],
                            parent1=parent2_copy,
                            parent2=parent1_copy,
                            replay_buffer=replay_buffer,
                        )

        # Mutate all slots except the new elitist target slots
        for i in range(pop_size):
            if i not in new_elitists:
                if rng.random() < mutation_prob:
                    mutated = self._clone_and_mutate(
                        parent=new_population[i],
                        mutation_std=mutation_std,
                        mutation_fraction=mutation_fraction,
                        rng=rng,
                    )
                    new_population[i].load_state_dict(mutated.state_dict())

        return new_population, new_elitists, unselect_indices

    def sync_rl_to_pop(
        self,
        actor: Actor,
        population: list[Actor],
        fitnesses: list[float],
        elite_indices: list[int],
        unselect_indices: list[int],
    ) -> None:
        if not population:
            return

        worst_index = int(np.argmin(fitnesses)) if fitnesses else 0

        if worst_index not in elite_indices:
            target = worst_index
        elif unselect_indices:
            target = unselect_indices[-1]
        else:
            target = elite_indices[-1]

        population[target].load_state_dict(copy.deepcopy(actor.state_dict()))
