from collections import deque, namedtuple

import numpy as np
import torch

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

PPOTransition = namedtuple(
    "PPOTransition", ("state", "action", "log_prob", "reward", "value", "done")
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
            (capacity, state_dim), dtype=torch.float32, device="cpu"
        )
        self.action = torch.zeros(
            (capacity, action_dim), dtype=torch.float32, device="cpu"
        )
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32, device="cpu")
        self.next_state = torch.zeros(
            (capacity, state_dim), dtype=torch.float32, device="cpu"
        )
        self.done = torch.zeros((capacity, 1), dtype=torch.float32, device="cpu")

        self.ptr = 0
        self.size = 0

    def add(self, transition: Transition) -> None:
        self.state[self.ptr] = torch.as_tensor(transition.state, dtype=torch.float32)
        self.action[self.ptr] = torch.as_tensor(transition.action, dtype=torch.float32)
        self.reward[self.ptr] = torch.as_tensor(transition.reward, dtype=torch.float32)
        self.next_state[self.ptr] = torch.as_tensor(transition.next_state, dtype=torch.float32)
        self.done[self.ptr] = torch.as_tensor(
            np.asarray(transition.done), dtype=torch.float32
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
            "state": self.state[indices].to(self.device),
            "action": self.action[indices].to(self.device),
            "reward": self.reward[indices].to(self.device),
            "next_state": self.next_state[indices].to(self.device),
            "done": self.done[indices].to(self.device),
        }

    def sample_latest(self, batch_size: int = None) -> dict[str, torch.Tensor]:
        if batch_size is None:
            batch_size = self.batch_size
        return self.sample(batch_size=batch_size, latest=True)

    def __len__(self) -> int:
        return self.size


class RolloutBuffer:
    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, transition: PPOTransition) -> None:
        self.states.append(torch.tensor(transition.state, dtype=torch.float32))
        self.actions.append(torch.tensor(transition.action, dtype=torch.float32))
        self.log_probs.append(torch.tensor(transition.log_prob, dtype=torch.float32))
        self.rewards.append(torch.tensor(transition.reward, dtype=torch.float32))
        self.values.append(torch.tensor(transition.value, dtype=torch.float32))
        self.dones.append(
            torch.tensor(np.asarray(transition.done), dtype=torch.float32)
        )

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> None:
        self.returns = []
        self.advantages = []

        gae = 0
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_non_terminal = 1.0
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal
                - self.values[step]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae

            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + self.values[step])

        self.states_t = torch.stack(self.states).to(self.device)
        self.actions_t = torch.stack(self.actions).to(self.device)
        self.log_probs_t = torch.stack(self.log_probs).to(self.device)
        self.returns_t = torch.stack(self.returns).to(self.device)
        self.advantages_t = torch.stack(self.advantages).to(self.device)

        self.advantages_t = (self.advantages_t - self.advantages_t.mean()) / (
            self.advantages_t.std() + 1e-8
        )

    def get_generator(self, minibatch_size: int):
        num_samples = len(self.states_t)
        indices = np.random.permutation(num_samples)

        for start in range(0, num_samples, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            yield {
                "state": self.states_t[mb_indices],
                "action": self.actions_t[mb_indices],
                "log_prob": self.log_probs_t[mb_indices],
                "return": self.returns_t[mb_indices],
                "advantage": self.advantages_t[mb_indices],
            }

    def __len__(self) -> int:
        return len(self.states)
