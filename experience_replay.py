# Define memory for Experience Replay

from collections import deque
import random


class ReplayMemory:
    def __init__(self, maxlen: int, seed: int | None = None):
        # Deque that automatically drops old entries when full
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        """Store one (state, action, new_state, reward, done) tuple."""
        self.memory.append(transition)

    def sample(self, sample_size: int):
        """Randomly sample a batch of transitions."""
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
