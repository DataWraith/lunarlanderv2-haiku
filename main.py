import random

import gym
import haiku as hk
import jax
import jax.numpy as jnp
from tqdm import trange

from agent import Agent
from avg_meter import AvgMeter
from replay_memory import ReplayMemory, Transition


def run(
    env_name,
    agent: Agent,
    memory: ReplayMemory,
    *,
    seed: int = 123,
    batch_size: int = 32,
    max_frames: int = 1000,
    gamma: float = 0.99,
):
    trend = AvgMeter(100)
    loss_trend = AvgMeter(2000)
    total_frames = 0

    env = gym.make(env_name)

    rng = hk.PRNGSequence(seed)
    target_rng = random.Random(seed)

    params = agent.initial_params(8, key=next(rng))
    opt_state = agent.initial_opt_state(params)

    for ep in range(1, 501):
        reward_sum = 0
        frames = 0
        rollout = []

        obs = jnp.asarray(env.reset(seed=ep))

        for frame in trange(max_frames, position=0, leave=False):
            frames += 1
            total_frames += 1

            action = int(agent.actor_step(params, obs, next(rng)))
            ns, reward, done, _ = env.step(action)
            reward_sum += reward

            ns = jnp.asarray(ns)
            rollout.append((obs, action, reward, ns, done or frame == max_frames - 1))

            obs = ns

            if done:
                break

            if memory.is_ready(20 * batch_size):
                for _ in range(2):
                    params, opt_state, loss = agent.learner_step(
                        params,
                        tuple(target_rng.sample(params, 2)),
                        opt_state,
                        memory.sample(batch_size),
                    )
                    loss_trend.append(float(loss))

        # Compute empirical returns
        returns = []
        G = 0.0
        for transition in reversed(rollout):
            G += transition[2]
            returns.append(G)
            G *= gamma

        for (state, action, reward, next_state, done), empirical_return in zip(
            rollout, reversed(returns)
        ):
            memory.push(
                Transition(state, action, reward, next_state, done, empirical_return)
            )

        trend.append(reward_sum)
        trend_mean = trend.mean()

        print(
            f"E: {ep}, Frames: {frames}, R: {reward_sum}, T: {trend_mean}, L: {loss_trend.mean()}, Total frames: {total_frames}",
            flush=True,
        )

        if trend_mean > 200:
            break


a = Agent()
m = ReplayMemory(100000)
run("LunarLander-v2", a, m)
