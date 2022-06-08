import collections
import random
from typing import Deque, NamedTuple

import jax.numpy as jnp


class Transition(NamedTuple):
    state: jnp.array
    action: int
    reward: float
    next_state: jnp.array
    done: bool
    empirical_return: float


class ReplayMemory:
    _buffer: Deque[Transition]

    def __init__(self, capacity: int = 100000, seed: int = 123):
        self._buffer = collections.deque([], capacity)
        self._rng = random.Random(seed)

    def push(self, transition: Transition):
        self._buffer.append(transition)

    def sample(self, batch_size: int = 32):
        samples = self._rng.sample(self._buffer, batch_size)

        states, actions, rewards, next_states, dones, empirical_returns = zip(*samples)

        return (
            jnp.stack(states),
            jnp.stack(actions),
            jnp.stack(rewards),
            jnp.stack(next_states),
            jnp.stack(dones),
            jnp.asarray(empirical_returns),
        )

    def is_ready(self, batch_size: int):
        return len(self._buffer) >= batch_size
