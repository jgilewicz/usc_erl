import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.deep_modules import Critic


class EnsembleModule(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        critic: Critic,
        rng: np.random.Generator,
    ) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.rng: np.random.Generator = rng

        self.critics: nn.ModuleList = nn.ModuleList(
            [copy.deepcopy(critic) for _ in range(ensemble_size)]
        )

        def reset_weights(m: nn.Module) -> None:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        for critic_i in self.critics:
            torch.manual_seed(int(self.rng.integers(0, 2**32 - 1)))
            critic_i.apply(reset_weights)

    def __getitem__(self, index: int) -> Critic:
        return self.critics[index]

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_predictions: list[torch.Tensor] = [
            critic(states, actions) for critic in self.critics
        ]
        q_tensor = torch.stack(q_predictions, dim=0)

        mean_q = q_tensor.mean(dim=0)
        std_q = q_tensor.std(dim=0, unbiased=True)

        return mean_q, std_q

    def compute_loss(
        self, states: torch.Tensor, actions: torch.Tensor, target_q: torch.Tensor
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=states.device)
        mask_prob = 0.5

        target_q = target_q.view(-1, 1)

        for critic in self.critics:
            current_q = critic(states, actions).view(-1, 1)

            mask = torch.bernoulli(torch.full_like(current_q, mask_prob))

            noisy_target_q = target_q * (1.0 + torch.randn_like(target_q) * 0.02)
            loss = F.smooth_l1_loss(current_q, noisy_target_q, reduction="none")

            masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            total_loss += masked_loss

        return total_loss

    def crossq_compute_loss(
        self,
        cat_states: torch.Tensor,
        cat_actions: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        gamma: float,
        b: int,
        mask_prob: float = 0.5,
    ) -> torch.Tensor:
        reward = reward.view(-1, 1)
        done = done.view(-1, 1)

        current_qs: list[torch.Tensor] = []
        next_qs: list[torch.Tensor] = []
        for critic in self.critics:
            q_all = critic(cat_states, cat_actions).view(-1, 1)
            current_qs.append(q_all[:b])
            next_qs.append(q_all[b:])

        next_q = torch.stack(next_qs, dim=0).mean(dim=0).detach()
        target_q = reward + (1.0 - done) * gamma * next_q

        total_loss = torch.tensor(0.0, device=cat_states.device)
        for current_q in current_qs:
            mask = torch.bernoulli(torch.full_like(current_q, mask_prob))
            noisy_target_q = target_q * (1.0 + torch.randn_like(target_q) * 0.02)
            loss = F.smooth_l1_loss(current_q, noisy_target_q, reduction="none")
            total_loss += (loss * mask).sum() / (mask.sum() + 1e-8)

        return total_loss
