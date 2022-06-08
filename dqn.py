from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp


class Params(NamedTuple):
    online: hk.Params
    target: hk.Params


def build_dqn(actions: int = 4, layers: int = 4, hidden_size: int = 128):
    def dqn(x):
        r = x

        for _ in range(layers):
            x = jax.nn.swish(hk.Linear(hidden_size)(x))
            x = jnp.concatenate((r, x), -1)

        return hk.Linear(actions)(x)

    return hk.without_apply_rng(hk.transform(dqn))
