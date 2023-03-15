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

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())


from dataclasses import dataclass
from typing import Any, Tuple

from absl import app, flags

from mava import specs as mava_specs
from mava.utils.environments import debugging_utils

from mava.types import OLT

import rlax
import optax
from mava.utils.environments import smac_utils #import make_environment

FLAGS = flags.FLAGS

flags.DEFINE_string("system", "test agent", "What agent is running.")
flags.DEFINE_string(
    "base_dir", "~/mava", "Base dir to store experiment data e.g. checkpoints."
)

from matrix_game import mat_game as mat

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
        """
        env, _ = smac_utils.make_environment(
            #env_name=config.env_name,
            #action_space=config.action_space,
            #random_seed=config.seed,
        )
        """
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
    ) = jax.random.split(prng, 2)

    hidden_layer_size = 64

    def make_net(hidden_layer_size, num_actions):
        def make_actor(x):
            return hk.nets.MLP([hidden_layer_size, hidden_layer_size, num_actions])(x)

        return hk.without_apply_rng(hk.transform(make_actor))
        

    def make_networks():

        networks = {}

        for net_key, spec in agent_specs.items():
            num_actions = spec.actions.num_values

            actor_net = make_net(hidden_layer_size, num_actions)
            obs_dim = spec.observations.observation.shape
            obs_type = spec.observations.observation.dtype
            dummy_obs = jnp.ones(shape=obs_dim, dtype=obs_type)
            actor_params = actor_net.init(actor_prng, jnp.array(dummy_obs))

            networks[net_key] = {}
            networks[net_key]["actor_params"] = actor_params
            networks[net_key]["actor_net"] = actor_net

        return networks

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


        return action, log_prob, entropy, value

    networks = make_networks()

    def make_optimisers():

        optimisers = {}

        for net_key in agent_specs.keys():

            # TODO LR NEEDS TO COME FROM CONFIG.
            #actor_optim = optax.sgd(0.01)
            #critic_optim = optax.sgd(0.01)
            actor_optim = optax.adam(0.005)

            actor_state = actor_optim.init(networks[net_key]["actor_params"])

            optimisers[net_key] = {}
            optimisers[net_key]["actor_state"] = actor_state
            optimisers[net_key]["actor_optim"] = actor_optim

        return optimisers

    optimisers = make_optimisers()


    #Only run this once
    def policy_loss(
        policy_params, observations, actions, actor_apply_fn,n_step
    ):

        logits = actor_apply_fn(policy_params, observations)
        dist = tfd.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        loss = -(n_step*log_probs).mean()
        return loss, loss

    def epoch_function(buffer, networks, optimisers):

        for agent in buffer["observations"][0].keys():

            rewards = [agent_dict[agent] for agent_dict in buffer["rewards"]]
            actions = [agent_dict[agent] for agent_dict in buffer["actions"]]

            #print(rewards)
            
            #Calculate discounted rewards
            discounted_rewards = []
            for t in range(len(rewards)):
                Gt = 0 
                pw = 0
                for r in rewards[t:]:
                    Gt = Gt + 0.99**pw*r
                    pw = pw+1
                discounted_rewards.append(Gt)
            n_step = jnp.array(discounted_rewards)

            observations = [
                agent_dict[agent].observation for agent_dict in buffer["observations"]
            ]
            legal_actions = [
                agent_dict[agent].legal_actions for agent_dict in buffer["observations"]
            ]
            terminals = [
                agent_dict[agent].terminal for agent_dict in buffer["observations"]
            ]

            observations = jnp.array(observations)
            legal_actions = jnp.array(legal_actions)
            terminals = jnp.array(terminals)

            # Next steps compute log(a|s_t) for loss ratios

            actor_net = networks[agent]["actor_net"]
            actor_net_apply_fn = actor_net.apply
            actor_params = networks[agent]["actor_params"]
            observations = jnp.array(observations)

            # TODO Change names to all be actor and not policy!!
            policy_grads, p_loss = jax.grad(policy_loss, has_aux=True)(
                actor_params,
                observations,
                jnp.array(actions),
                actor_net_apply_fn,
                n_step
            )

            updates, new_policy_optimiser_state = optimisers[agent][
                "actor_optim"
            ].update(policy_grads, optimisers[agent]["actor_state"])
            print(
                f"policy_loss {p_loss} {optax.global_norm(policy_grads)} {optax.global_norm(updates)}"
            )
            new_policy_params = optax.apply_updates(actor_params, updates)

            # Update params
            networks[agent]["actor_params"] = new_policy_params
            optimisers[agent]["actor_state"] = new_policy_optimiser_state
        return networks, optimisers

    return networks, config, prng, epoch_function, sample_fn, optimisers



def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """

    # TODO: Set some nice way to pad episodes.

    init_config = init()
    env, env_config = make_environment()
    
    env_spec = mava_specs.MAEnvironmentSpec(env)
    (
        networks,
        system_config,
        prng,
        epoch_function,
        sample_function,
        optimisers,
    ) = make_system(env_spec)
    logging.info(f"Running {FLAGS.system}")

    simple_buffer = {}

    # Run system on env
    episodes = 50
    test_buffer = {}
    for episode in range(episodes):

        simple_buffer = {}
        simple_buffer["observations"] = []
        simple_buffer["actions"] = []
        simple_buffer["rewards"] = []
        simple_buffer["values"] = []
        simple_buffer["log_probs"] = []
        simple_buffer["entropy"] = []
        ep_return = 0
        timestep= env.reset()
        step = 0
        while not timestep.last():
            # get action
            prng, action_key = jax.random.split(prng)
            action, log_prob, entropy, value = sample_function(
                timestep.observation, networks, action_key
            )
            timestep = env.step(action)
            #print(action)
            #print(timestep.observation)
            #print(timestep.reward)
            #print(action)
            #print(value)
            #exit()
            simple_buffer["observations"].append(timestep.observation)
            simple_buffer["actions"].append(action)
            simple_buffer["rewards"].append(timestep.reward)
            simple_buffer["values"].append(value)
            simple_buffer["log_probs"].append(log_prob)
            simple_buffer["entropy"].append(entropy)
            team_reward = 0

            for reward in timestep.reward.values():
                team_reward += reward
            team_reward /= 2
            ep_return += team_reward
            step += 1
        
        #exit()
        print(f"{episode}: {ep_return}")
        if episode == 0:
            test_buffer = simple_buffer.copy()
        # Can update an amount of episodes!
        # print(
        #     f"b: {[optax.global_norm(networks[a]['actor_params']) for a in networks.keys()]} {[optax.global_norm(networks[a]['critic_params']) for a in networks.keys()]} "
        # )
        networks, optimisers = epoch_function(simple_buffer, networks, optimisers)
        # print(
        #     f"a: {[optax.global_norm(networks[a]['actor_params']) for a in networks.keys()]} {[optax.global_norm(networks[a]['critic_params']) for a in networks.keys()]} "
        # )

    # Next steps: Is it training?? / Nice way to log results.


if __name__ == "__main__":
    app.run(main)
