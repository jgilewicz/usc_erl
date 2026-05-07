import torch
import numpy as np
from collections import deque
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class Buffer:
    def __init__(
        self, capacity: int, batch_size: int, rng: np.random.Generator
    ) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self.rng = rng
        self.buffer = deque(maxlen=capacity)

    def _make_transition(self, transition: Transition) -> Transition:
        return Transition(
            state=torch.tensor(transition.state, dtype=torch.float32),
            action=torch.tensor(transition.action, dtype=torch.float32),
            reward=torch.tensor(transition.reward, dtype=torch.float32),
            next_state=torch.tensor(transition.next_state, dtype=torch.float32),
            done=torch.tensor(np.asarray(transition.done), dtype=torch.float32),
        )

    def add(self, transition: Transition) -> None:
        self.buffer.append(self._make_transition(transition))

    def sample(self, batch_size: int = 32, latest: bool = False) -> list[Transition]:
        indices = np.arange(len(self.buffer))

        if latest:
            indices = np.arange(len(self.buffer) - batch_size, len(self.buffer))
        else:
            indices = self.rng.choice(len(self.buffer), batch_size, replace=False)

        batch = [self.buffer[i] for i in indices]

        return {
            "state": torch.stack([t.state for t in batch]),
            "action": torch.stack([t.action for t in batch]),
            "reward": torch.stack([t.reward for t in batch]).unsqueeze(-1),
            "next_state": torch.stack([t.next_state for t in batch]),
            "done": torch.stack([t.done for t in batch]).unsqueeze(-1),
        }

    def sample_latest(self, batch_size: int = None) -> list[Transition]:
        return self.sample(batch_size=batch_size, latest=True)

    def __len__(self) -> int:
        return len(self.buffer)
