from functools import partial
import copy
import logging
from dataclasses import dataclass
from datetime import datetime
from math import prod
from typing import Any, Dict, Tuple

import chex
import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
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
# jax.config.update("jax_platform_name", "cpu")

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

    total_steps: int = 100_000
    train_freq: int = 1
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    gamma: float = 0.99
    ac_noise_scale: float = 0.1
    tau: float = 0.01


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

    return (
        actor,
        critic,
        actor_params,
        critic_params,
        target_actor_params,
        target_critic_params,
    )


def critic_loss_fn(
    critic_params: hk.Params,
    td_target: chex.Array,
    critic: hk.Transformed,
    obs: chex.Array,
    act: chex.Array,
) -> chex.Array:
    q = critic.apply(critic_params, obs, act)
    # loss = jnp.mean(rlax.l2_loss(td_target, q))
    loss = jnp.mean((td_target - q) ** 2)
    return loss, {"mean q": jnp.mean(q), "critic loss": loss}


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


@partial(jax.jit, static_argnames=["actor", "critic", "critic_opt"])
def update_critic(
    actor,
    critic,
    target_actor_params,
    critic_params,
    target_critic_params,
    critic_opt,
    critic_opt_state,
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma,
):
    next_state_actions = (actor.apply(target_actor_params, next_observations)).clip(
        -1, 1
    )  # TODO: proper clip

    qf1_next_target = critic.apply(
        target_critic_params, next_observations, next_state_actions
    ).reshape(-1)
    next_q_value = (rewards + (1 - dones) * gamma * (qf1_next_target)).reshape(-1)

    def mse_loss(params):
        qf1_a_values = critic.apply(params, observations, actions).squeeze()
        return ((qf1_a_values - next_q_value) ** 2).mean(), qf1_a_values.mean()

    (qf1_loss_value, qf1_a_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(
        critic_params
    )

    updates, new_opt_state = critic_opt.update(grads, critic_opt_state, critic_params)
    new_params = optax.apply_updates(critic_params, updates)
    return new_params, new_opt_state, qf1_loss_value, qf1_a_values


@partial(jax.jit, static_argnames=["actor", "critic", "actor_opt"])
def update_actor(
    actor,
    critic,
    actor_params,
    critic_params,
    actor_opt,
    actor_opt_state,
    observations: jnp.ndarray,
):
    def actor_loss(params):
        return -critic.apply(
            critic_params, observations, actor.apply(params, observations)
        ).mean()

    actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_params)
    updates, new_opt_state = actor_opt.update(grads, actor_opt_state, actor_params)
    new_params = optax.apply_updates(actor_params, updates)

    return new_params, new_opt_state, actor_loss_value


@jax.jit
def update_target_params(
    actor_params, critic_params, actor_target_params, critic_target_params, tau
):
    actor_target_params = optax.incremental_update(
        actor_params, actor_target_params, tau
    )

    critic_target_params = optax.incremental_update(
        critic_params, critic_target_params, tau
    )

    return actor_target_params, critic_target_params


def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """
    time_delta = 1
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
    config = init()
    env, _ = make_environment()
    # env_spec = mava_specs.MAEnvironmentSpec(env)
    key, net_key = jax.random.split(key)
    # TODO: use specs to get action and obs size
    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape
    action_max = env.action_space.high
    action_min = env.action_space.low

    action_scale = (action_max - action_min) / 2

    (
        actor,
        critic,
        actor_params,
        critic_params,
        target_actor_params,
        target_critic_params,
    ) = make_networks(net_key, action_shape, obs_shape, action_scale)
    # target_actor_params = copy.deepcopy(actor_params)
    # target_critic_params = copy.deepcopy(critic_params)

    actor_opt = optax.adam(config.actor_lr)
    critic_opt = optax.adam(config.critic_lr)
    actor_opt_params = actor_opt.init(actor_params)
    critic_opt_params = critic_opt.init(critic_params)
    rb = ReplayBuffer(prod(obs_shape), prod(action_shape), int(1e6))

    episode_reward = 0
    episode_length = 0
    episode_num = 0
    trainer_step = 0

    learning_starts = 300  # 25e3

    obs, _ = env.reset()
    for global_step in range(config.total_steps):
        key, noise_key = jax.random.split(key)
        # action selection, step env
        # prev_obs = obs.copy()  # copy.deepcopy(obs)
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
        if term or trunc:
            episode_num += 1
            logger.write(
                {
                    "episode_num": episode_num,
                    "episode reward": episode_reward,
                    "episode length": episode_length,
                }
            )
            episode_reward = 0
            episode_length = 0
            obs, _ = env.reset()

        # train:
        if global_step > learning_starts:
            trainer_step += 1
            #  get sample from replay buf
            key, batch_key = jax.random.split(key)
            batch = rb.sample_batch(batch_key, config.batch_size)

            critic_params, critic_opt_params, critic_loss, q_value = update_critic(
                actor,
                critic,
                target_actor_params,
                critic_params,
                target_critic_params,
                critic_opt,
                critic_opt_params,
                batch["obs"],
                batch["act"],
                batch["next_obs"],
                batch["rew"],
                batch["done"],
                config.gamma,
            )
            actor_params, actor_opt_params, actor_loss = update_actor(
                actor,
                critic,
                actor_params,
                critic_params,
                actor_opt_params,
                batch["obs"],
            )
            trainer_logger.write(
                {
                    "trainer step": trainer_step,
                    "critic loss": critic_loss,
                    "q value": q_value,
                    "actor loss": actor_loss,
                }
            )


if __name__ == "__main__":
    app.run(main)
