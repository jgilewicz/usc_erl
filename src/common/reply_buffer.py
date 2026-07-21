from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class Buffer:
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        rng: np.random.Generator,
        state_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self.rng = rng
        self.device = device

        self.state = torch.zeros(
            (capacity, state_dim), dtype=torch.float32, device=device
        )
        self.action = torch.zeros(
            (capacity, action_dim), dtype=torch.float32, device=device
        )
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_state = torch.zeros(
            (capacity, state_dim), dtype=torch.float32, device=device
        )
        self.done = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

        self.ptr = 0
        self.size = 0

    def add(self, transition: Transition) -> None:
        self.state[self.ptr] = torch.tensor(
            transition.state, dtype=torch.float32, device=self.device
        )
        self.action[self.ptr] = torch.tensor(
            transition.action, dtype=torch.float32, device=self.device
        )
        self.reward[self.ptr] = torch.tensor(
            transition.reward, dtype=torch.float32, device=self.device
        )
        self.next_state[self.ptr] = torch.tensor(
            transition.next_state, dtype=torch.float32, device=self.device
        )
        self.done[self.ptr] = torch.tensor(
            np.asarray(transition.done), dtype=torch.float32, device=self.device
        )

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int = 32, latest: bool = False
    ) -> dict[str, torch.Tensor]:
        if latest:
            batch_size = min(batch_size, self.size)
            if batch_size == 0:
                raise ValueError("Cannot sample from an empty buffer")
            start = (self.ptr - batch_size) % self.capacity
            if start < self.ptr:
                indices = np.arange(start, self.ptr)
            else:
                indices = np.concatenate(
                    (np.arange(start, self.capacity), np.arange(0, self.ptr))
                )
        else:
            indices = self.rng.choice(self.size, batch_size, replace=False)

        return {
            "state": self.state[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "next_state": self.next_state[indices],
            "done": self.done[indices],
        }

    def sample_latest(self, batch_size: int = None) -> dict[str, torch.Tensor]:
        return self.sample(batch_size=batch_size, latest=True)

    def __len__(self) -> int:
        return self.size
