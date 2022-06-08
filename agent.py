import functools
from typing import List, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax

from dqn import Params, build_dqn
from replay_memory import Transition


class Agent:
    def __init__(
        self,
        *,
        n: int = 10,
        learning_rate: float = 1e-3,
        target_update: float = 0.01,
        ucb_lambda: float = 2.0,
        sail_alpha: float = 0.9,
        discount_factor: float = 0.99
    ):
        self.n = n
        self.target_update = target_update
        self.ucb_lambda = ucb_lambda
        self.sail_alpha = sail_alpha
        self.discount_factor = discount_factor

        self._optimizer = optax.chain(
            optax.clip_by_global_norm(5.0), optax.adabelief(learning_rate)
        )

    def initial_params(self, num_inputs: int, *, key):
        sample_input = jnp.zeros(num_inputs)
        rng = hk.PRNGSequence(key)

        self._dqn = build_dqn()

        params = []
        for _ in range(self.n):
            initial_params = self._dqn.init(next(rng), sample_input)
            params.append(Params(initial_params, initial_params))

        return params

    def initial_opt_state(self, params):
        return [self._optimizer.init(params[i].online) for i in range(self.n)]

    @functools.partial(jax.jit, static_argnums=(0,))
    def actor_step(
        self, ensemble_params: List[Params], obs: jnp.array, key: jax.random.PRNGKey
    ) -> int:
        q_values = jnp.stack(
            [self._dqn.apply(params.online, obs) for params in ensemble_params]
        )

        # UCB exploration
        q_mean = jnp.mean(q_values, axis=0)
        q_std = jnp.std(q_values, axis=0)
        q = q_mean + self.ucb_lambda * q_std

        return rlax.greedy().sample(key, q)

    @functools.partial(jax.jit, static_argnums=(0,))
    def learner_step(
        self,
        params: List[Params],
        tgt_params: List[Params],
        opt_states: List[optax.OptState],
        transitions: Transition,
    ) -> Tuple[List[Params], List[optax.OptState], float]:
        states, actions, rewards, next_states, dones, empirical_returns = transitions

        batch_size = states.shape[0]
        dqn_apply_tgt = lambda params, states: self._dqn.apply(params.target, states)

        # Q-value estimates from the target network subset
        q_tm1 = jnp.min(
            jnp.stack([dqn_apply_tgt(params, states) for params in tgt_params]), axis=0
        )

        q_t = jnp.min(
            jnp.stack([dqn_apply_tgt(params, next_states) for params in tgt_params]),
            axis=0,
        )

        # Self-Imitation Advantage Learning reward modification
        r_sail = rewards + self.sail_alpha * (
            jnp.maximum(empirical_returns, q_tm1[jnp.arange(batch_size), actions])
            - jnp.max(q_tm1, axis=-1)
        )

        # Compute q_target
        q_target = r_sail + (1 - dones) * self.discount_factor * jnp.max(q_t, axis=-1)
        q_target = jax.lax.stop_gradient(q_target)

        chex.assert_shape(q_tm1, (batch_size, 4))
        chex.assert_shape(q_t, (batch_size, 4))
        chex.assert_shape(r_sail, (batch_size,))
        chex.assert_shape(q_target, (batch_size,))

        new_params = []
        new_opt_states = []
        loss_sum = 0.0

        for i in range(self.n):
            np, no, q_loss = self._update_step(
                params[i], opt_states[i], states, actions, q_target
            )

            new_params.append(np)
            new_opt_states.append(no)

            loss_sum += q_loss

        return new_params, new_opt_states, loss_sum / self.n

    def _update_step(
        self,
        params: Params,
        opt_state: optax.OptState,
        states: jnp.array,
        actions: jnp.array,
        q_targets: jnp.array,
    ) -> Tuple[Params, optax.OptState, float]:
        q_loss, grads = jax.value_and_grad(self._loss)(
            params.online, states, actions, q_targets
        )

        updates, new_opt_state = self._optimizer.update(grads, opt_state)
        new_online_params = optax.apply_updates(params.online, updates)
        new_target_params = rlax.incremental_update(
            new_online_params, params.target, self.target_update
        )

        return Params(new_online_params, new_target_params), new_opt_state, q_loss

    def _loss(
        self,
        params: hk.Params,
        states: jnp.array,
        actions: jnp.array,
        target: jnp.array,
    ):
        batch_size = states.shape[0]
        q = self._dqn.apply(params, states)[jnp.arange(batch_size), actions]
        return jnp.mean(rlax.huber_loss(target - q))
