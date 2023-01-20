# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
from dataclasses import dataclass
from math import prod
from typing import Any, Dict, Tuple

import chex
import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
from absl import app, flags

from mava import specs as mava_specs
from mava.utils.environments import debugging_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("system", "test agent", "What agent is running.")
flags.DEFINE_string(
    "base_dir", "~/mava", "Base dir to store experiment data e.g. checkpoints."
)


# TODO: jittable buffer?
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        self.obs_buf = jnp.zeros((size, obs_dim), dtype=jnp.float32)
        self.next_obs_buf = jnp.zeros((size, obs_dim), dtype=jnp.float32)
        self.act_buf = jnp.zeros((size, act_dim), dtype=jnp.float32)
        self.rew_buf = jnp.zeros(size, dtype=jnp.float32)
        self.done_buf = jnp.zeros(size, dtype=jnp.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        obs: chex.Array,
        act: chex.Array,
        rew: chex.Array,
        next_obs: chex.Array,
        done: chex.Array,
    ) -> None:
        self.obs_buf = self.obs_buf.at[self.ptr].set(obs)
        self.next_obs_buf = self.next_obs_buf.at[self.ptr].set(next_obs)
        self.act_buf = self.act_buf.at[self.ptr].set(act)
        self.rew_buf = self.rew_buf.at[self.ptr].set(rew)
        self.done_buf = self.done_buf.at[self.ptr].set(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(
        self, key: chex.PRNGKey, batch_size: int = 32
    ) -> Dict[str, chex.Array]:
        idxs = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.size - 1
        )
        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return batch


@dataclass
class EnvironmentConfig:
    env_name: str = "simple_spread"
    seed: int = 42
    type: str = "debug"
    action_space: str = "discrete"


@dataclass
class SystemConfig:
    name: str = "random"
    seed: int = 42

    total_steps: int = 50_000
    train_freq: int = 100
    batch_size: int = 32
    actor_lr: float = 0.001
    critic_lr: float = 0.001

    gamma: float = 0.99
    ac_noise_scale: float = 0.2
    tau: float = 0.005


def init(config: SystemConfig = SystemConfig()) -> SystemConfig:
    """Init system.

    This would handle thing to be done upon system once in the beginning of a run,
    e.g. set random seeds.

    Args:
        config : init config.

    Returns:
        config.
    """

    return config


def make_environment(
    config: EnvironmentConfig = EnvironmentConfig(),
) -> Tuple[gym.Env, EnvironmentConfig]:
    """Init and return environment or wrapper.

    Args:
        config : env config.

    Returns:
        (env, config).
    """

    # return gym.make("MountainCarContinuous-v0", render_mode="human"), config
    # return gym.make("MountainCarContinuous-v0"), config
    return gym.make("Pendulum-v1", render_mode="human"), config


# TODO: make DDPG network class
def make_networks(
    key: chex.PRNGKey, action_shape: Tuple[int, ...], obs_shape: Tuple[int, ...]
) -> Tuple[hk.Transformed, hk.Transformed, hk.Params, hk.Params]:
    """Inits and returns system/networks.

    Args:
        config : system config.
        environment_spec: spec for multi-agent env.

    Returns:
        system.
    """

    @hk.without_apply_rng
    @hk.transform
    def actor(obs: chex.Array) -> chex.Array:
        actor = hk.nets.MLP(
            [64, 64, *action_shape],
            activation=jax.nn.tanh,
            activate_final=True,
        )
        return actor(obs) * 5  # TODO: take an action scale

    @hk.without_apply_rng
    @hk.transform
    def critic(obs: chex.Array, action: chex.Array) -> chex.Array:
        return hk.nets.MLP([64, 64, 1])(jnp.concatenate([obs, action], axis=1))

    dummy_obs = jnp.zeros((1, *obs_shape))
    dummy_actions = jnp.zeros((1, *action_shape))
    actor_params = actor.init(key, dummy_obs)
    critic_params = critic.init(key, dummy_obs, dummy_actions)

    return actor, critic, actor_params, critic_params


def critic_loss_fn(
    td_target: chex.Array,
    critic: hk.Transformed,
    critic_params: hk.Params,
    obs: chex.Array,
    act: chex.Array,
) -> chex.Array:
    return jnp.mean(rlax.l2_loss(td_target, critic.apply(critic_params, obs, act)))


def actor_loss_fn(
    actor_params: hk.Params,
    actor: hk.Transformed,
    critic_params: hk.Params,
    critic: hk.Transformed,
    obs: chex.Array,
) -> chex.Array:
    actions = actor.apply(actor_params, obs)
    q = critic.apply(critic_params, obs, actions)
    return -jnp.mean(q)


def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """

    key = jax.random.PRNGKey(1)
    config = init()
    env, _ = make_environment()
    # env_spec = mava_specs.MAEnvironmentSpec(env)
    key, net_key = jax.random.split(key)
    # TODO: use specs to get action and obs size
    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape
    action_max = env.action_space.low
    action_min = env.action_space.high
    # print(env.action_space)

    actor, critic, actor_params, critic_params = make_networks(
        net_key, action_shape, obs_shape
    )
    target_actor_params = copy.deepcopy(actor_params)
    target_critic_params = copy.deepcopy(critic_params)

    actor_opt = optax.adam(config.actor_lr)
    critic_opt = optax.adam(config.critic_lr)
    actor_opt_params = actor_opt.init(actor_params)
    critic_opt_params = critic_opt.init(critic_params)
    # logging.info(f"Running {FLAGS.system}")
    rb = ReplayBuffer(prod(obs_shape), prod(action_shape), 10_000)

    episode_reward = 0
    episode_length = 0

    obs, _ = env.reset()
    for global_step in range(config.total_steps):
        key, noise_key = jax.random.split(key)
        # action selection, step env
        prev_obs = copy.deepcopy(obs)
        action = jax.lax.stop_gradient(actor.apply(actor_params, obs))
        noise = jax.random.normal(noise_key, action.shape) * config.ac_noise_scale
        action = (noise + action).clip(action_min, action_max)

        obs, reward, term, trunc, info = env.step(action)

        episode_reward += reward
        episode_length += 1

        # replay add stuff
        rb.store(prev_obs, action, reward, obs, term)
        if term or trunc:
            print(
                f"""
                episode reward: {episode_reward}
                episode length: {episode_length}
                ----------------------------------"""
            )
            episode_reward = 0
            episode_length = 0
            obs, _ = env.reset()

        # train:
        if global_step % config.train_freq == 0:
            #  get sample from replay buf
            key, batch_key = jax.random.split(key)
            batch = rb.sample_batch(batch_key, config.batch_size)
            # --------------- critic updates ----------------------
            # TODO: do you need to stop grads at all of these or only the final?
            next_actions = jax.lax.stop_gradient(
                actor.apply(target_actor_params, batch["next_obs"])
            )
            next_q_values = jax.lax.stop_gradient(
                critic.apply(target_critic_params, batch["next_obs"], next_actions)
            )
            td_target = jax.lax.stop_gradient(
                batch["rew"] + config.gamma * next_q_values * (1 - batch["done"])
            )

            # TODO: move critic params to first position
            critic_grads = jax.grad(critic_loss_fn, argnums=2)(
                td_target, critic, critic_params, batch["obs"], batch["act"]
            )
            critic_updates, critic_opt_params = critic_opt.update(
                critic_grads, critic_opt_params
            )
            critic_params = optax.apply_updates(critic_params, critic_updates)

            #  ---------------------- actor updates ----------------------
            # TODO: should this be once every n critic updates?
            actor_grads = jax.grad(actor_loss_fn)(
                actor_params, actor, critic_params, critic, batch["obs"]
            )
            actor_updates, actor_opt_params = actor_opt.update(
                actor_grads, actor_opt_params
            )
            actor_params = optax.apply_updates(actor_params, actor_updates)

            #  update target networks
            target_actor_params = optax.incremental_update(
                actor_params, target_actor_params, config.tau
            )
            target_critic_params = optax.incremental_update(
                critic_params, target_critic_params, config.tau
            )


if __name__ == "__main__":
    app.run(main)
