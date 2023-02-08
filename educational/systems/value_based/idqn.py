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
import os
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from absl import app, flags

import haiku as hk
import optax

from mava import specs as mava_specs
from mava.utils.environments import debugging_utils, smac_utils
from mava.utils.loggers import logger_utils

from collections import deque
from typing import Iterator, NamedTuple
import random

from dataclasses import field
import numpy as np

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("system", "test agent", "What agent is running.")
flags.DEFINE_string(
    "base_dir", os.path.expanduser("~")+"/mava", "Base dir to store experiment data e.g. checkpoints."
)

# TODO: clean up alot of these stuffs we are not using

# Array = chex.Array
# ArrayNumpy = chex.ArrayNumpy
# Numeric = chex.Numeric


@dataclass
class InitConfig:
    seed: int = 42
    learning_rate: float = 5e-4
    buffer_size: int = 50000
    warm_up_steps: int = 2000
    min_epsilon: float = 0.1
    max_epsilon: float = 0.9
    total_num_steps: int = 1500000
    batch_size: int = 32
    update_frequency: int = 20
    training_frequency: int = 10
    tau: float = 1.
    logging_frequency: int = 100


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

@chex.dataclass(frozen=True, mappable_dataclass=False)
class RandomSystemState:
    rng: jnp.ndarray

class TrainingState(NamedTuple):
  params: Dict[str, hk.Params]
  target_params: Dict[str, hk.Params]
  opt_state: Dict[str, optax.OptState]

class TransitionBatch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""
    actions: Any
    observation: Any 
    next_observation: Any
    reward: Any 
    done: Any
    
def from_singles(
    action, obs, next_obs, reward, done
) -> TransitionBatch:


    agents = action.keys()
    actions = {agent: jnp.array([action[agent]]) for agent in agents}
    rewards = {agent: jnp.array([reward[agent]]) for agent in agents}
    dones =   {agent: jnp.array([done], dtype=bool) for agent in agents}
    obs_ = {agent: obs[agent].observation for agent in agents}
    next_obs_ = {agent: next_obs[agent].observation for agent in agents}

    return TransitionBatch(
        actions=actions,
        observation=obs_,
        next_observation=next_obs_,
        reward=rewards,
        done=dones
    )


def init(config: InitConfig = InitConfig()) -> InitConfig:
    """Init system.

    This would handle thing to be done upon system once in the beginning of a run,
    e.g. set random seeds.

    Args:
        config : init config.
    """
    return config


def make_environment(
    config: EnvironmentConfig = EnvironmentConfig()
) -> Tuple[Any, EnvironmentConfig]:
    """Init and return environment or wrapper.

    Args:
        config : env config.

    Returns:
        env or wrapper.
    """
    if config.type == "debug":
        env, _ = debugging_utils.make_environment(
            env_name=config.env_name,
            action_space=config.action_space,
            random_seed=config.seed,
        )
    return env#, config


def make_system(
    environment_spec: mava_specs.MAEnvironmentSpec,
) -> Tuple[Any, SystemConfig]:
    """Inits and returns system/networks.

    Args:
        config : system config.

    Returns:
        system.
    """
    prng = jax.random.PRNGKey(42)
    q_key, _ = jax.random.split(prng)
    agent_specs = environment_spec.get_agent_environment_specs()

    # add more options to this
    def make_q_net(hidden_layer_sizes): 
        def make_actor(x):
            return hk.nets.MLP(hidden_layer_sizes)(x)

        return hk.without_apply_rng(hk.transform(make_actor))

    def make_networks():
        networks = {}
        
        for net_key, spec in agent_specs.items():
            num_actions = spec.actions.num_values
            q_net = make_q_net([32, 32, num_actions])
            obs_dim = spec.observations.observation.shape
            obs_type = spec.observations.observation.dtype
            dummy_obs = jnp.ones(shape=obs_dim,dtype=obs_type)

            q_network_params = q_net.init(q_key,dummy_obs)
            networks[net_key] = {}
            networks[net_key]["actor_params"] = q_network_params
            networks[net_key]["actor_net"] = q_net
        
        # TODO: make this function jitable by removing networks and
        # using the store approach as in mava
        def sample_fn(observation, networks, q_params):
            action = {}

            for agent_key, agent_olt in observation.items():

                q_network = networks[agent_key]["actor_net"]
                params = q_params[agent_key]
                agent_observation = agent_olt.observation
                q_network_apply = jax.jit(q_network.apply)
                q_values = q_network_apply(params,agent_observation)
                action[agent_key] = q_values.argmax(axis=-1)
                
            return action

        return networks,sample_fn
    
    networks, sample_fn = make_networks()
    return networks, sample_fn

class ReplayBuffer():

    def __init__(self,
        max_len = 100, seed = 0):

        self.buffer = deque(maxlen = max_len)
        self.num_items = 0
        random.seed(seed)

    def add(self,data):
        self.buffer.extend([data])
        self.num_items = len(self.buffer)
    
    def sample(self, batch_size: int) -> TransitionBatch:
        transitions = random.sample(self.buffer, batch_size)
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *transitions)

    @property
    def size(self):
        return self.num_items
    

def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """

    # Setup TensorBoard logging
    summary_writer = tf.summary.create_file_writer(f"{FLAGS.base_dir}/tensorboard-idqn-smac")

    # Init env and system.
    config = init()
    env = make_environment()
    test_env = make_environment()
    env_spec = mava_specs.MAEnvironmentSpec(env)
    agent_specs = env_spec.get_agent_environment_specs()
    networks, sample_fn = make_system(env_spec)
    replay_buffer = ReplayBuffer(config.buffer_size)

    # Initialise optimisers states and params
    optimisers = {}
    initial_params = {}
    initial_opt_state = {}
    
    for net_key, _ in agent_specs.items():
        optimisers[net_key] = optax.chain(optax.scale_by_adam(), optax.scale(-config.learning_rate))
        initial_params[net_key] = networks[net_key]["actor_params"]
        initial_opt_state[net_key] = optimisers[net_key].init(initial_params[net_key])
    
    state = TrainingState(initial_params, initial_params, initial_opt_state)
    
    global_num_steps = 0
    total_num_steps = config.total_num_steps
    min_epsilon, max_epsilon =  config.min_epsilon, config.max_epsilon 
    episodes = total_num_steps//50 # because each epsiode is 50 steps for this environment
    test_episodes = 10
    num_agents = len(agent_specs)

    @jax.jit
    def update(state: TrainingState, batch: TransitionBatch):
    
        loss = {}
        q_preds_dict = {}
        q_params_dict = {}
        opt_state_dict = {}
        
        (actions, obs, next_obs, rewards, dones) = (
            batch.actions, batch.observation, batch.next_observation, batch.reward, batch.done
            )

        agents = list(dones.keys())

        for net_key in agents: 
            q_params = state.params[net_key]
            q_params_target = state.target_params[net_key]
            q_next_target =  networks[net_key]["actor_net"].apply(q_params_target, next_obs[net_key])
            q_next_target = jnp.max(q_next_target, axis = -1, keepdims=True)
            next_q_value  = rewards[net_key] + (1 - dones[net_key])*0.99*q_next_target

            def mse_loss(params):
                q_pred = networks[net_key]["actor_net"].apply(params, obs[net_key])
                q_pred_value = jnp.take_along_axis(q_pred, actions[net_key], axis=-1)
                return ((q_pred_value - next_q_value)**2).mean(), q_pred_value.squeeze()
            
            (loss[net_key], q_preds_dict[net_key]), grads = jax.value_and_grad(mse_loss,has_aux=True)(q_params)
            
            updates, opt_state_dict[net_key] = optimisers[net_key].update(grads, state.opt_state[net_key])
            q_params_dict[net_key] = optax.apply_updates(q_params, updates)
            
        state = TrainingState(q_params_dict, state.target_params, opt_state_dict)

        return loss, q_preds_dict, state

    def test(env, num_episodes, networks, q_params):
        score = np.zeros(num_agents)
        for _ in range(num_episodes):
            timestep = env.reset()
            while not timestep.last():
                actions = sample_fn(timestep.observation, networks, q_params)
                timestep = env.step(actions)
                score += np.array(list(timestep.reward.values()))

        return sum(score / num_episodes)
    
    rng_key = jax.random.PRNGKey(config.seed)
    score = np.zeros(num_agents)
    for episode in range(episodes):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode / (0.4 * episodes)))
        rng_key, episode_key = jax.random.split(rng_key)
        
        timestep = env.reset()

        while not timestep.last():
            
            episode_key, action_key = jax.random.split(episode_key)
            if jax.random.uniform (action_key) < epsilon:
                actions = {agent: env.action_spaces[agent].sample() for agent in agent_specs.keys()}
            else:
                actions = sample_fn(timestep.observation, networks, state.params)
            
            new_timestep = env.step(actions)

            transition_data = from_singles(actions, timestep.observation, new_timestep.observation, 
                                new_timestep.reward,timestep.last())
            
            replay_buffer.add(transition_data)
            
            timestep = new_timestep
            
            score += np.array(list(new_timestep.reward.values()))
            global_num_steps += 1
                
        # train if time to train
        if global_num_steps > config.warm_up_steps:
            
            # train for a certain number of iterations
            for _ in range(config.training_frequency):
                batch = replay_buffer.sample(config.batch_size)
                loss, q_values, state = update(state, batch)
            
            if global_num_steps % config.logging_frequency == 0:
                with summary_writer.as_default():
                    for agent in agent_specs.keys():
                        tf.summary.scalar(f"losses/td_loss-{agent}", jax.device_get(loss[agent]), step=global_num_steps)
                        tf.summary.scalar(f"losses/q_values-{agent}", jax.device_get(q_values[agent]).mean(), global_num_steps)
            
                    tf.summary.scalar(f"epsilon", epsilon, global_num_steps)
            
            # update the network
            if episode % config.update_frequency == 0 and episode!=0:
                for net_key, _ in agent_specs.items():
                    state = state._replace(target_params = optax.incremental_update(state.params, state.target_params, config.tau))
            
                test_score = test(test_env, test_episodes, networks, state.params)
                print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                    .format(episode, episodes, sum(score /config.update_frequency), test_score, replay_buffer.size, epsilon))
                
                with summary_writer.as_default():
                    tf.summary.scalar(f"rewards/avg_train-reward", sum(score /config.update_frequency), step=global_num_steps)
                    tf.summary.scalar(f"rewards/avg_test-reward", test_score, step=global_num_steps)

                score = np.zeros(num_agents)

    

if __name__ == "__main__":
    app.run(main)