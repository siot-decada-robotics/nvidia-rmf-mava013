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


from dataclasses import dataclass
from typing import Any, Tuple

from absl import app, flags

from mava import specs as mava_specs
from mava.utils.environments import debugging_utils

import rlax 

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


def init(config: InitConfig = InitConfig()) -> InitConfig:
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

    if config.type == "debug":
        env, _ = debugging_utils.make_environment(
            env_name=config.env_name,
            action_space=config.action_space,
            random_seed=config.seed,
        )
    return env, config


import haiku as hk
import jax.numpy as jnp
import jax

import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


def make_system(
    environment_spec: mava_specs.MAEnvironmentSpec,
    config: SystemConfig = SystemConfig(),
) -> Tuple[Any, SystemConfig]:
    """Inits and returns system/networks.

    Args:
        config : system config.
        environment_spec: spec for multi-agent env.

    Returns:
        system.
    """
    agent_specs = environment_spec.get_agent_environment_specs()

    prng = jax.random.PRNGKey(config.seed)

    (
        prng,
        actor_prng,
        critic_prng,
    ) = jax.random.split(prng, 3)

    hidden_layer_size = 64

    def make_net(hidden_layer_size, num_actions):
        def make_actor(x):
            return hk.nets.MLP([hidden_layer_size, num_actions])(x)

        def make_critic(x):
            return hk.nets.MLP([hidden_layer_size, 1])(x)

        return hk.without_apply_rng(hk.transform(make_actor)), hk.without_apply_rng(
            hk.transform(make_critic)
        )

    def make_networks():

        networks = {}

        for net_key, spec in agent_specs.items():
            num_actions = spec.actions.num_values

            # networks[net_key] = {"actor_net": actor_net, "critic_net": critic_net}
            actor_net, critic_net = make_net(hidden_layer_size, num_actions)
            obs_dim = spec.observations.observation.shape
            obs_type = spec.observations.observation.dtype
            dummy_obs = jnp.ones(shape=obs_dim, dtype=obs_type)

            actor_params = actor_net.init(actor_prng, dummy_obs)
            critic_params = critic_net.init(critic_prng, dummy_obs)

            networks[net_key] = {}
            networks[net_key]["actor_params"] = actor_params
            networks[net_key]["critic_params"] = critic_params
            networks[net_key]["actor_net"] = actor_net
            networks[net_key]["critic_net"] = critic_net

        def sample_fn(obserservation, networks, key):
            action = {}
            log_prob = {}
            entropy = {}
            value = {}

            for agent_key, agent_olt in obserservation.items():

                actor_net = networks[agent_key]["actor_net"]
                actor_params = networks[agent_key]["actor_params"]
                agent_obserservation = agent_olt.observation
                logits = actor_net.apply(actor_params, agent_obserservation)

                dist = tfd.Categorical(logits=logits)  # , dtype=self._dtype)
                # TODO - FIX THIS
                action[agent_key] = dist.sample(seed=key)
                log_prob[agent_key] = dist.log_prob(action[agent_key])
                entropy[agent_key] = dist.entropy()

                critic_net = networks[agent_key]["critic_net"]
                critic_params = networks[agent_key]["critic_params"]

                value[agent_key] = jnp.squeeze(critic_net.apply(critic_params, agent_obserservation))
            
            return action, log_prob, entropy, value

        networks["sample_fn"] = sample_fn
        return networks

    networks = make_networks()


    def epoch_function(buffer, key, networks):

        # sample_idxs = jax.random.permutation(key=key, x=len(buffer))

        for i, episode in buffer.items(): 
            
            advantages = {}
            new_log_probs = {}

            # TODO the 0 is just get access to agent names. 
            for agent in episode["observations"][0].keys():
                
                rewards_ = [agent_dict[agent] for agent_dict in episode["rewards"]]
                values_ = [agent_dict[agent] for agent_dict in episode["values"]]
                

                advantage = rlax.truncated_generalized_advantage_estimation(

                    r_t = jnp.array(rewards_[1:]), 
                    # TODO: Max episode horizon value here 
                    discount_t = jnp.ones(49) * 0.99, 
                    lambda_ = 0.95, 
                    values =  jnp.array(values_)
                )

                advantages[agent] = advantage

                observations = [agent_dict[agent].observation for agent_dict in episode["observations"]]

                # Next steps compute log(a|s_t) for loss ratios






    return networks, config, prng, epoch_function
    # del environment_spec

    # class System:
    #     pass

    # logging.info(config)
    # system = System()
    # return system, config


def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """

    # TODO: Set some nice way to pad episodes. 

    init_config = init()
    env, env_config = make_environment()
    env_spec = mava_specs.MAEnvironmentSpec(env)
    networks, system_config, prng, epoch_function = make_system(env_spec)
    logging.info(f"Running {FLAGS.system}")

    simple_buffer = {}

    # Run system on env
    episodes = 1
    for episode in range(episodes):
        simple_buffer[episode] = {}
        simple_buffer[episode]["observations"] = []
        simple_buffer[episode]["actions"] = []
        simple_buffer[episode]["rewards"] = []
        simple_buffer[episode]["values"] = []
        simple_buffer[episode]["log_probs"] = []
        simple_buffer[episode]["entropy"] = []

        timestep = env.reset()
        while not timestep.last():
            # get action
            prng, action_key = jax.random.split(prng)
            action, log_prob, entropy, value = networks["sample_fn"](timestep.observation, networks, action_key)
            timestep = env.step(action)
            
            # take step
            
            simple_buffer[episode]["observations"].append(timestep.observation) 
            simple_buffer[episode]["actions"].append(action)
            simple_buffer[episode]["rewards"].append(timestep.reward)
            simple_buffer[episode]["values"].append(value)
            simple_buffer[episode]["log_probs"].append(log_prob)
            simple_buffer[episode]["entropy"].append(entropy)

    epoch_function(simple_buffer, prng, networks)

if __name__ == "__main__":
    app.run(main)
