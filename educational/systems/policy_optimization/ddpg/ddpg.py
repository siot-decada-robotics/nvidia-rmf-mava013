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


# TODO: use this
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

    total_steps: int = 100_000
    train_freq: int = 1
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    gamma: float = 0.99
    ac_noise_scale: float = 0.1
    tau: float = 0.01


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
    return gym.make("Pendulum-v1"), config
    # return gym.make("LunarLanderContinuous-v2"), config


# TODO: make DDPG network class
def make_networks(
    key: chex.PRNGKey,
    action_shape: Tuple[int, ...],
    obs_shape: Tuple[int, ...],
    action_scale: float,
) -> Tuple[hk.Transformed, hk.Transformed, hk.Params, hk.Params, hk.Params, hk.Params]:
    """Inits and returns system/networks.

    Args:
        config : system config.
        environment_spec: spec for multi-agent env.

    Returns:
        system.
    """
    #To do net as var
    @hk.without_apply_rng
    @hk.transform
    def actor(obs: chex.Array) -> chex.Array:
        actor = hk.nets.MLP([256, 256, *action_shape])
        return jax.nn.tanh(actor(obs)) * action_scale

    @hk.without_apply_rng
    @hk.transform
    def critic(obs: chex.Array, action: chex.Array) -> chex.Array:
        return hk.nets.MLP([256, 256, 1])(jnp.concatenate([obs, action], axis=-1))

    actor_key, critic_key = jax.random.split(key)
    dummy_obs = jnp.zeros((1, *obs_shape))
    dummy_actions = jnp.zeros((1, *action_shape))
    actor_params = actor.init(actor_key, dummy_obs)
    target_actor_params = actor.init(actor_key, dummy_obs)
    critic_params = critic.init(critic_key, dummy_obs, dummy_actions)
    target_critic_params = critic.init(critic_key, dummy_obs, dummy_actions)
    #TODO make return 2 data classes
    return (
        actor,
        critic,
        actor_params,
        critic_params,
        target_actor_params,
        target_critic_params,
    )


@partial(jax.jit, static_argnames=["critic"])
def critic_loss_fn(
    critic_params: hk.Params,
    td_target: chex.Array,
    critic: hk.Transformed,
    obs: chex.Array,
    act: chex.Array,
) -> chex.Array:
    q = critic.apply(critic_params, obs, act).squeeze()
    # loss = jnp.mean(rlax.l2_loss(td_target, q))
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

    key = jax.random.PRNGKey(42)
    config = SystemConfig()
    env, _ = make_environment()
    # env_spec = mava_specs.MAEnvironmentSpec(env)
    key, net_key = jax.random.split(key)

    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape
    action_max = env.action_space.high
    action_min = env.action_space.low
    action_scale = (action_max - action_min) / 2
    # TODO: action bias

    # TODO: dataclass to hold params
    (
        actor,
        critic,
        actor_params,
        critic_params,
        target_actor_params,
        target_critic_params,
    ) = make_networks(net_key, action_shape, obs_shape, action_scale)

    # TODO: put opts in dataclass
    actor_opt = optax.adam(config.actor_lr)
    critic_opt = optax.adam(config.critic_lr)
    actor_opt_params = actor_opt.init(actor_params)
    critic_opt_params = critic_opt.init(critic_params)
    rb = ReplayBuffer(prod(obs_shape), prod(action_shape), int(1e6))

    episode_reward = episode_reward_log = 0
    episode_length = episode_length_log = 0
    episode_num = 0
    trainer_step = 0

    learning_starts = 1000  # TODO: config

    obs, _ = env.reset()
    for global_step in range(config.total_steps):
        key, noise_key = jax.random.split(key)
        # TODO: jit action selection
        # action selection, step env
        if global_step > learning_starts:
            action = actor.apply(actor_params, obs)
        else:
            action = env.action_space.sample()
        noise = jax.random.normal(noise_key) * config.ac_noise_scale
        action = (noise + action).clip(action_min, action_max)

        nobs, reward, term, trunc, info = env.step(action.tolist())

        episode_reward += reward
        episode_length += 1

        # replay add stuff
        rb.store(obs.copy(), action.copy(), reward, nobs.copy(), term)
        obs = nobs.copy()

        logger.write(  # TODO: fix logging (in mava!)
            {
                "episode": episode_num,
                "episode reward": episode_reward_log,
                "episode length": episode_length_log,
                "time step": global_step,
                "raw reward": reward,
            }, 
        )
        if term or trunc:
            episode_num += 1
            episode_reward_log = episode_reward
            episode_length_log = episode_length
            episode_reward = 0
            episode_length = 0
            obs, _ = env.reset()
            

        # train:
        if global_step > learning_starts:
            
            trainer_step += 1
            #  get sample from replay buf
            key, batch_key = jax.random.split(key)
            batch = rb.sample_batch(batch_key, config.batch_size)
            # --------------- critic updates ----------------------
            next_actions = actor.apply(target_actor_params, batch["next_obs"]).clip(
                action_min, action_max
            )
            next_q_values = critic.apply(
                target_critic_params, batch["next_obs"], next_actions
            ).squeeze()

            # TODO: stop grad?
            td_target = batch["rew"] + (
                config.gamma * next_q_values * (1 - batch["done"]).squeeze()
            )

            critic_grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(
                critic_params, td_target, critic, batch["obs"], batch["act"]
            )
            critic_updates, critic_opt_params = critic_opt.update(
                critic_grads, critic_opt_params, critic_params
            )
            critic_params = optax.apply_updates(critic_params, critic_updates)

            #  ---------------------- actor updates ----------------------
            # TODO: make an update method
            # TODO: should this be once every n critic updates?
            actor_grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(
                actor_params, actor, critic_params, critic, batch["obs"]
            )
            actor_updates, actor_opt_params = actor_opt.update(
                actor_grads, actor_opt_params, actor_params
            )
            actor_params = optax.apply_updates(actor_params, actor_updates)

            # ---------------------- update target networks ----------------------
            # TODO: method
            target_actor_params = optax.incremental_update(
                actor_params, target_actor_params, config.tau
            )
            target_critic_params = optax.incremental_update(
                critic_params, target_critic_params, config.tau
            )
            trainer_logger.write(
                {"trainer step": trainer_step, **critic_info, **actor_info}
            )



if __name__ == "__main__":
    app.run(main)
