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
from datetime import datetime
from functools import partial
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
from chex import dataclass

from mava import specs as mava_specs
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("system", "test agent", "What agent is running.")
flags.DEFINE_string(
    "base_dir", "~/mava", "Base dir to store experiment data e.g. checkpoints."
)
# TODO: remove - getting OOM errors on 3050ti
jax.config.update("jax_platform_name", "cpu")


"""Wraper for Cooperative Pettingzoo environments."""
from typing import Any, Dict, List, Union

import dm_env
import matplotlib.pyplot as plt
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from pettingzoo.mpe import simple_spread_v2

from mava import types
from mava.utils.wrapper_utils import parameterized_restart


class MPE:
    """Environment wrapper for PettingZoo MARL environments."""

    def __init__(self):
        """Constructor for parallel PZ wrapper."""

        self.environment = simple_spread_v2.parallel_env(continuous_actions=True)
        self.possible_agents = self.environment.possible_agents
        self.num_agents = len(self.possible_agents)
        self.observation_space = list(self.environment.observation_spaces.values())[0]
        self.action_space = list(self.environment.action_spaces.values())[0]

    def arrayify(self, d: dict, shape: tuple, dtype: str):
        arr = []
        for agent in self.possible_agents:
            if agent in d:
                arr.append(jnp.array(d[agent], dtype))
            else:
                arr.append(jnp.zeros(shape, dtype))

        return jnp.array(arr)

    def reset(self):
        """Resets the env.

        Returns:
            obs [N, obs]:
        """
        # Reset the environment
        observations = self.environment.reset()
        return self.arrayify(
            observations, self.observation_space.shape, self.observation_space.dtype
        )

    def step(self, actions):
        """Steps in env.

        Args:
            actions: [N, act]

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        actions_dict = {}
        for i, agent in enumerate(self.possible_agents):
            if agent not in self.environment.agents:
                continue
            actions_dict[agent] = actions[i]

        # Step the environment
        next_observations, rewards, dones, _ = self.environment.step(actions_dict)

        # Add zeros to missing agents
        next_observations = self.arrayify(
            next_observations,
            self.observation_space.shape,
            self.observation_space.dtype,
        )
        rewards = self.arrayify(rewards, (1,), float)
        dones = self.arrayify(dones, (1,), bool)

        infos = {}  # TODO

        return next_observations, rewards, dones, infos


# TODO: jittable buffer?
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int, num_agents: int) -> None:
        self.obs_buf = jnp.zeros((size, num_agents, obs_dim), dtype=jnp.float32)
        self.next_obs_buf = jnp.zeros((size, num_agents, obs_dim), dtype=jnp.float32)
        self.act_buf = jnp.zeros((size, num_agents, act_dim), dtype=jnp.float32)
        self.rew_buf = jnp.zeros((size, num_agents), dtype=jnp.float32)
        self.done_buf = jnp.zeros((size, num_agents), dtype=jnp.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.num_agents = num_agents

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


# TODO: use this
@dataclass
class EnvironmentConfig:
    env_name: str = "Pendulum-v1"
    seed: int = 1  # TODO: there are two seeds
    type: str = "debug"
    action_space: str = "discrete"


@dataclass
class SystemConfig:
    seed: int = 1  # TODO: there are two seeds

    total_steps: int = 100_000
    learning_starts: int = 100
    actor_train_freq: int = 2
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    gamma: float = 0.99
    ac_noise_scale: float = 0.1
    tau: float = 0.01


@chex.dataclass
class Network:
    actor: hk.Transformed
    critic: hk.Transformed
    actor_opt: optax.GradientTransformation
    critic_opt: optax.GradientTransformation


@chex.dataclass
class NetworkParams:
    actor_params: hk.Params
    critic_params: hk.Params
    target_actor_params: hk.Params
    target_critic_params: hk.Params
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState


def make_environment(
    config: EnvironmentConfig = EnvironmentConfig(),
) -> Tuple[gym.Env, EnvironmentConfig]:
    """Init and return environment or wrapper.

    Args:
        config : env config.

    Returns:
        (env, config).
    """
    return MPE(), config


# TODO: make DDPG network class
def make_networks(
    key: chex.PRNGKey,
    action_shape: Tuple[int, ...],
    obs_shape: Tuple[int, ...],
    action_scale: float,
    config,
) -> Tuple[Network, NetworkParams]:
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
        actor = hk.nets.MLP([256, 256, *action_shape])
        return jax.nn.tanh(actor(obs)) * action_scale

    @hk.without_apply_rng
    @hk.transform
    def critic(obs: chex.Array, action: chex.Array) -> chex.Array:
        return hk.nets.MLP([256, 256, 1])(jnp.concatenate([obs, action], axis=-1))

    # networks
    actor_key, critic_key = jax.random.split(key)
    dummy_obs = jnp.zeros((1, *obs_shape))
    dummy_actions = jnp.zeros((1, *action_shape))

    actor_params = actor.init(actor_key, dummy_obs)
    target_actor_params = actor.init(actor_key, dummy_obs)
    critic_params = critic.init(critic_key, dummy_obs, dummy_actions)
    target_critic_params = critic.init(critic_key, dummy_obs, dummy_actions)

    # optimisers
    actor_opt = optax.adam(config.actor_lr)
    critic_opt = optax.adam(config.critic_lr)
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)

    network = Network(
        actor=actor,
        critic=critic,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
    )
    network_params = NetworkParams(
        actor_params=actor_params,
        critic_params=critic_params,
        target_actor_params=target_actor_params,
        target_critic_params=target_critic_params,
        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,
    )

    return network, network_params


@partial(jax.jit, static_argnames=["critic"])
def critic_loss_fn(
    critic_params: hk.Params,
    td_target: chex.Array,
    critic: hk.Transformed,
    obs: chex.Array,
    act: chex.Array,
) -> chex.Array:
    q = critic.apply(critic_params, obs, act).squeeze()
    loss = jnp.mean((td_target - q) ** 2)
    return loss, {"mean q": jnp.mean(q), "critic loss": loss}


@partial(jax.jit, static_argnames=["actor", "critic"])
def actor_loss_fn(
    actor_params: hk.Params,
    actor: hk.Transformed,
    critic_params: hk.Params,
    critic: hk.Transformed,
    obs: chex.Array,
) -> chex.Array:
    actions = actor.apply(actor_params, obs)
    q = critic.apply(critic_params, obs, actions)
    loss = -jnp.mean(q)
    return loss, {"policy loss": loss}


def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """
    time_delta = 1
    # TODO: Fix loggers!
    logger = logger_utils.make_logger(
        directory="edulogs/",
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=str(datetime.now()) + "executor",
        time_delta=time_delta,
        label="ddpg_executor",
    )

    trainer_logger = logger_utils.make_logger(
        directory="edulogs/",
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=str(datetime.now()) + "trainer",
        time_delta=time_delta,
        label="ddpg_trainer",
    )

    config = SystemConfig()
    env_conf = EnvironmentConfig()

    key = jax.random.PRNGKey(config.seed)
    key, net_key = jax.random.split(key)
    env, _ = make_environment(env_conf)

    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape
    action_max = env.action_space.high
    action_min = env.action_space.low
    action_scale = (action_max - action_min) / 2
    # TODO: action bias
    network, network_params = make_networks(
        net_key, action_shape, obs_shape, action_scale, config
    )

    rb = ReplayBuffer(prod(obs_shape), prod(action_shape), int(1e6), env.num_agents)

    episode_reward = episode_reward_log = 0
    episode_length = episode_length_log = 0
    episode_num = 0
    trainer_step = 0

    @jax.jit
    def select_action(key, global_step, actor_params, obs):
        key, noise_key, ac_key = jax.random.split(key, 3)

        action = jax.lax.cond(
            global_step > config.learning_starts,
            lambda: network.actor.apply(actor_params, obs),
            # env.action_space.sample(),
            lambda: jax.random.uniform(
                ac_key,
                (env.num_agents, *env.action_space.shape),
                minval=action_min,
                maxval=action_max,
            ),
        )

        noise = jax.random.normal(noise_key) * config.ac_noise_scale
        action = (noise + action).clip(action_min, action_max)

        return action, key

    @jax.jit
    def critic_update(network_params, batch):
        next_actions = network.actor.apply(
            network_params.target_actor_params, batch["next_obs"]
        ).clip(action_min, action_max)
        next_q_values = network.critic.apply(
            network_params.target_critic_params, batch["next_obs"], next_actions
        ).squeeze()
        # TODO: stop grad?
        td_target = batch["rew"] + (
            config.gamma * next_q_values * (1 - batch["done"]).squeeze()
        )
        critic_grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(
            network_params.critic_params,
            td_target,
            network.critic,
            batch["obs"],
            batch["act"],
        )
        critic_updates, network_params.critic_opt_state = network.critic_opt.update(
            critic_grads,
            network_params.critic_opt_state,
            network_params.critic_params,
        )
        network_params.critic_params = optax.apply_updates(
            network_params.critic_params, critic_updates
        )
        network.critic, network.critic_opt = network.critic, network.critic_opt
        return network_params, critic_info

    def actor_update(network_params, batch):
        actor_grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(
            network_params.actor_params,
            network.actor,
            network_params.critic_params,
            network.critic,
            batch["obs"],
        )
        (actor_updates, network_params.actor_opt_state,) = network.actor_opt.update(
            actor_grads,
            network_params.actor_opt_state,
            network_params.actor_params,
        )
        network_params.actor_params = optax.apply_updates(
            network_params.actor_params, actor_updates
        )
        return network_params, actor_info

    @jax.jit
    def target_update(network_params, tau):
        network_params.target_actor_params = optax.incremental_update(
            network_params.actor_params,
            network_params.target_actor_params,
            tau,
        )
        network_params.target_critic_params = optax.incremental_update(
            network_params.critic_params,
            network_params.target_critic_params,
            tau,
        )
        return network_params

    obs = env.reset()
    for global_step in range(config.total_steps):
        # TODO: jit action selection
        # action selection, step env

        action, key = select_action(
            key,
            global_step,
            network_params.actor_params,
            obs,
        )

        nobs, reward, done, info = env.step(action.tolist())

        episode_reward += reward
        episode_length += 1

        # replay add stuff
        rb.store(obs.copy(), action.copy(), reward, nobs.copy(), done)
        obs = nobs.copy()

        logger.write(  # TODO: fix logging (in mava!)
            {
                "episode": episode_num,
                "episode reward": jnp.mean(episode_reward_log),
                "episode length": episode_length_log,
                "time step": global_step,
                "raw reward": reward,
            },
        )
        if done.all():
            episode_num += 1
            episode_reward_log = episode_reward
            episode_length_log = episode_length
            episode_reward = 0
            episode_length = 0
            obs = env.reset()

        # train:
        if global_step > config.learning_starts:

            trainer_step += 1
            #  get sample from replay buf
            key, batch_key = jax.random.split(key)
            batch = rb.sample_batch(batch_key, config.batch_size)
            batch = {k: v.reshape(-1, *v.shape[2:]) for k, v in batch.items()}
            # batch = jax.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), batch)
            # --------------- critic updates ----------------------
            network_params, critic_info = critic_update(network_params, batch)

            if global_step % config.actor_train_freq == 0:
                #  ---------------------- actor updates ----------------------
                # TODO: make an update method
                # TODO: should this be once every n critic updates?
                network_params, actor_info = actor_update(network_params, batch)
                # ---------------------- update target networks ----------------------
                # TODO: method
                # TODO: should this be every train step or actor train step?
                network_params = target_update(network_params, config.tau)

        else:
            actor_info = {"policy loss": None}
            critic_info = {"critic loss": None, "policy loss": None}

        trainer_logger.write(
            {"trainer step": trainer_step, **critic_info, **actor_info}
        )


if __name__ == "__main__":
    app.run(main)
