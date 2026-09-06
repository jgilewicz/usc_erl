import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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

    def forward_per_member(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([critic(states, actions) for critic in self.critics], dim=0)

    def compute_loss(
        self, states: torch.Tensor, actions: torch.Tensor, target_qs: torch.Tensor
    ) -> torch.Tensor:
        mask_prob = 0.5
        losses = []

        for i, critic in enumerate(self.critics):
            current_q = critic(states, actions).view(-1, 1)
            target_q = target_qs[i].view(-1, 1)

            mask = torch.bernoulli(torch.full_like(current_q, mask_prob))

            noisy_target_q = target_q * (1.0 + torch.randn_like(target_q) * 0.02)
            loss = F.smooth_l1_loss(current_q, noisy_target_q, reduction="none")

            losses.append((loss * mask).sum() / (mask.sum() + 1e-8))

        return torch.stack(losses).mean()
