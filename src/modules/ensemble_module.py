import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.modules import Critic


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
        
        target_q = target_q.view(-1, 1)

        for critic in self.critics:
            current_q = critic(states, actions).view(-1, 1)
            total_loss += F.mse_loss(current_q, target_q)

        return total_loss
