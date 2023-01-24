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

from mava import specs as mava_specs
from mava.utils.environments import debugging_utils, smac_utils
from mava.utils.loggers import logger_utils

from collections import deque

FLAGS = flags.FLAGS

flags.DEFINE_string("system", "test agent", "What agent is running.")
flags.DEFINE_string(
    "base_dir", "~/mava", "Base dir to store experiment data e.g. checkpoints."
)


@dataclass
class InitConfig:
    seed: int = 42


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
    q_key,key = jax.random.split(prng)
    agent_specs = environment_spec.get_agent_environment_specs()

    def make_q_net(hidden_layer_size, num_actions):
        def make_actor(x):
            return hk.nets.MLP([hidden_layer_size, num_actions])(x)

        return hk.without_apply_rng(hk.transform(make_actor))

    def make_networks():
        networks = {}
        hidden_layer_size = 32
        for net_key, spec in agent_specs.items():
            num_actions = spec.actions.num_values
            q_net = make_q_net(hidden_layer_size, num_actions)
            obs_dim = spec.observations.observation.shape
            obs_type = spec.observations.observation.dtype
            dummy_obs = jnp.ones(shape=obs_dim,dtype=obs_type)

            q_network_params = q_net.init(q_key,dummy_obs)
            networks[net_key] = {}
            networks[net_key]["actor_params"] = q_network_params
            networks[net_key]["actor_net"] = q_net
        

        def sample_fn(observation,networks):
            action = {}

            for agent_key, agent_olt in observation.items():

                q_network = networks[agent_key]["actor_net"]
                q_params = networks[agent_key]["actor_params"]
                agent_observation = agent_olt.observation
                q_values = q_network.apply(q_params,agent_observation)
                action[agent_key] = q_values.argmax(axis=-1)
                
            return action

        #networks["sample_fn"] = sample_fn
        return networks,sample_fn
    
    networks,sample_fn = make_networks()
    return networks,sample_fn

class ReplayBuffer():

    def __init__(self,
        max_len = 100,):
        self.max_len = 100
        self.buffer = deque(maxlen = max_len)
        self.num_items = 0

    def add(self,data):
        self.buffer.append(data)
        self.num_items = len(self.buffer)

def loss(q_params,q_networks,obs,actions,next_obs , rewards, dones,specs)  :
    loss_grads = {}
    for net_key, spec in specs.items(): 
        q_next_target =  q_networks[net_key]["actor_net"].apply(q_params[net_key],next_obs[net_key])
        q_next_target = jnp.max(q_next_target, axis = -1)
        next_q_value  = rewards[net_key] + (1 - dones[net_key])*0.99*q_next_target

        def mse_loss(params):
            q_pred = q_networks[net_key]["actor_net"].apply(q_params[net_key],obs[net_key])
            return ((q_pred - next_q_value)**2).mean(), q_pred
        
        (loss_grads[net_key],q_preds), grads = jax.value_and_grad(mse_loss,has_aux=True)(q_params[net_key])


    return loss_grads
    #TODO: CHANGE TO USE TARGET NETS
    
def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """

    # Init env and system.
    _ = init()
    env = make_environment()
    env_spec = mava_specs.MAEnvironmentSpec(env)
    networks, sample_fn = make_system(env_spec)
    replay_buffer = ReplayBuffer()
    #run system on env
    episodes = 10
     
    for episode in range(episodes):
        timestep = env.reset()
        counter  =0
        while not timestep.last():
            #get action
            actions = sample_fn(timestep.observation,networks)
            new_timestep = env.step(actions)

            replay_tuple = [actions,timestep.observation,new_timestep.observation,new_timestep.reward,timestep.last()]
            replay_buffer.add(replay_tuple)
            timestep = new_timestep
            counter += 1
            #print(counter)

if __name__ == "__main__":
    app.run(main)
