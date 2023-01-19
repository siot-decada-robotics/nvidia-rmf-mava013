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
from typing import Any, Tuple

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

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = jnp.zeros((size, obs_dim), dtype=jnp.float32)
        self.next_obs_buf = jnp.zeros((size, obs_dim), dtype=jnp.float32)
        self.act_buf = jnp.zeros((size, act_dim), dtype=jnp.float32)
        self.rew_buf = jnp.zeros(size, dtype=jnp.float32)
        self.done_buf = jnp.zeros(size, dtype=jnp.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf = self.obs_buf.at[self.ptr].set(obs)
        self.next_obs_buf = self.next_obs_buf.at[self.ptr].set(next_obs)
        self.act_buf = self.act_buf.at[self.ptr].set(act)
        self.rew_buf = self.rew_buf.at[self.ptr].set(rew)
        self.done_buf = self.done_buf.at[self.ptr].set(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, key, batch_size=32):
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

    gamma: float = 0.99


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
) -> Tuple[Any, EnvironmentConfig]:
    """Init and return environment or wrapper.

    Args:
        config : env config.

    Returns:
        (env, config).
    """

    return gym.make("MountainCarContinuous-v0"), config


# TODO: make DDPG network class
def make_networks(
    key, action_size, obs_size
) -> Tuple[hk.Transformed, hk.Transformed, jnp.ndarray, jnp.ndarray]:
    """Inits and returns system/networks.

    Args:
        config : system config.
        environment_spec: spec for multi-agent env.

    Returns:
        system.
    """

    @hk.without_apply_rng
    @hk.transform
    def actor(obs):
        return hk.nets.MLP([64, 64, action_size])(obs)

    @hk.without_apply_rng
    @hk.transform
    def critic(obs, action):
        return hk.nets.MLP([64, 64, 1])(jnp.concatenate([obs, action], axis=1))

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_actions = jnp.zeros((1, action_size))
    actor_params = actor.init(key, dummy_obs)
    critic_params = critic.init(key, dummy_obs, dummy_actions)

    return actor, critic, actor_params, critic_params


def critic_loss_fn(td_target, critic, critic_params, obs, act):
    return jnp.mean(rlax.l2_loss(td_target, critic.apply(critic_params, obs, act)))


def actor_loss_fn(actor_params, actor, critic_params, critic, obs):
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
    env, env_config = make_environment()
    # env_spec = mava_specs.MAEnvironmentSpec(env)
    key, net_key = jax.random.split(key)
    # TODO: use specs to get action and obs size
    actor, critic, actor_params, critic_params = make_networks(net_key, 1, 2)
    target_actor_params = copy.deepcopy(actor_params)
    target_critic_params = copy.deepcopy(critic_params)

    actor_opt = optax.adam(0.001)  # TODO:  make this config option
    critic_opt = optax.adam(0.001)
    actor_opt_params = actor_opt.init(actor_params)
    critic_opt_params = critic_opt.init(critic_params)
    # logging.info(f"Running {FLAGS.system}")
    rb = ReplayBuffer(2, 1, 10_000)

    obs, _ = env.reset()
    for global_step in range(config.total_steps):
        # action selection, step env
        prev_obs = copy.deepcopy(obs)
        action = actor.apply(actor_params, jnp.expand_dims(obs, axis=0))
        # TODO: copy previous obs
        obs, reward, term, trunc, info = env.step(action)
        done = term
        # replay add stuff
        rb.store(prev_obs, action.squeeze(), reward, obs.squeeze(), done)

        # train:
        if global_step % config.train_freq == 0:
            #  get sample from replay buf
            key, batch_key = jax.random.split(key)
            batch = rb.sample_batch(batch_key, config.batch_size)
            # --------------- critic updates ----------------------
            next_actions = actor.apply(target_actor_params, batch["next_obs"])
            next_q_values = critic.apply(
                target_critic_params, batch["next_obs"], next_actions
            )
            td_target = jax.lax.stop_gradient(
                batch["rew"] + config.gamma * next_q_values * (1 - batch["done"])
            )

            critic_grads = jax.grad(critic_loss_fn, argnums=2)(
                td_target, critic, critic_params, batch["obs"], batch["act"]
            )
            critic_updates, critic_opt_params = critic_opt.update(
                critic_grads, critic_opt_params
            )
            critic_params = optax.apply_updates(critic_params, critic_updates)

            #  ---------------------- actor updates ----------------------
            actor_grads = jax.grad(actor_loss_fn)(
                actor_params, actor, critic_params, critic, batch["obs"]
            )
            actor_updates, actor_opt_params = actor_opt.update(
                actor_grads, actor_opt_params
            )
            actor_params = optax.apply_updates(actor_params, actor_updates)

            #  update target networks


if __name__ == "__main__":
    app.run(main)
