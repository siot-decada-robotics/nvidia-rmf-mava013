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
import numpy as np 

FLAGS = flags.FLAGS

flags.DEFINE_string("system", "test agent", "What agent is running.")
flags.DEFINE_string(
    "base_dir", "~/mava", "Base dir to store experiment data e.g. checkpoints."
)

from matrix_game import mat_game as mat

NUM_EPOCHS = 10

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
        
        # env, _ = smac_utils.make_environment(
        #     #env_name=config.env_name,
        #     #action_space=config.action_space,
        #     #random_seed=config.seed,
        # )
    return env, config


import haiku as hk
import jax.numpy as jnp
import jax

import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

def compute_gae(next_value: float, rewards, masks, values: list, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)
    return returns, advantages

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
            return hk.nets.MLP([hidden_layer_size, hidden_layer_size, num_actions])(x)

        def make_critic(x):
            return hk.nets.MLP(
                [hidden_layer_size, hidden_layer_size, hidden_layer_size, 1]
            )(x)

        return hk.without_apply_rng(hk.transform(make_actor)), hk.without_apply_rng(
            hk.transform(make_critic)
        )

    def make_networks():

        networks = {}

        
        # num_actions = spec.actions.num_values
        num_actions = agent_specs["agent_0"].actions.num_values 

        # networks[net_key] = {"actor_net": actor_net, "critic_net": critic_net}
        actor_net, critic_net = make_net(hidden_layer_size, num_actions)
        # obs_dim = spec.observations.observation.shape
        # obs_type = spec.observations.observation.dtype

        obs_dim = agent_specs["agent_0"].observations.observation.shape
        obs_type = agent_specs["agent_0"].observations.observation.dtype

        dummy_obs = jnp.ones(shape=obs_dim, dtype=obs_type)
        actor_params = actor_net.init(actor_prng, jnp.array(dummy_obs))
        critic_params = critic_net.init(critic_prng, dummy_obs)


        for net_key, spec in agent_specs.items():

            networks[net_key] = {}
            networks[net_key]["actor_params"] = actor_params
            networks[net_key]["critic_params"] = critic_params
            networks[net_key]["actor_net"] = actor_net
            networks[net_key]["critic_net"] = critic_net


        return networks

    def sample_fn(obserservation, networks, key):
        action = {}
        log_prob = {}
        entropy = {}
        value = {}
        
        num_agents = len(obserservation.items())
        agent_prng_keys = jax.random.split(key,num_agents)

        for index, (agent_key, agent_olt) in enumerate(obserservation.items()):

            actor_net = networks[agent_key]["actor_net"]
            actor_params = networks[agent_key]["actor_params"]
            agent_obserservation = agent_olt.observation
            logits = actor_net.apply(actor_params, agent_obserservation)

            dist = tfd.Categorical(logits=logits)  # , dtype=self._dtype)
            # TODO - FIX THIS
            action[agent_key] = dist.sample(seed=agent_prng_keys[index])
            log_prob[agent_key] = dist.log_prob(action[agent_key])
            entropy[agent_key] = dist.entropy()

            critic_net = networks[agent_key]["critic_net"]
            critic_params = networks[agent_key]["critic_params"]

            value[agent_key] = jnp.squeeze(
                critic_net.apply(critic_params, agent_obserservation)
            )

        return action, log_prob, entropy, value

    networks = make_networks()

    def make_optimisers():

        optimisers = {}

        # TODO LR NEEDS TO COME FROM CONFIG.
        #actor_optim = optax.sgd(0.01)
        #critic_optim = optax.sgd(0.01)
        actor_optim = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate = 0.005, eps = 1e-5),
        )
        critic_optim = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate = 0.005, eps = 1e-5),
        )

        actor_state = actor_optim.init(networks["agent_0"]["actor_params"])
        critic_state = critic_optim.init(networks["agent_0"]["critic_params"])

        for net_key in agent_specs.keys():

            optimisers[net_key] = {}
            optimisers[net_key]["actor_state"] = actor_state
            optimisers[net_key]["critic_state"] = critic_state
            optimisers[net_key]["actor_optim"] = actor_optim
            optimisers[net_key]["critic_optim"] = critic_optim


        return optimisers

    optimisers = make_optimisers()

    # WON't WORK BECAUSE DIFFERENT NETWORKS
    def ppo_loss(
        old_log_probs, new_log_probs, advantages, old_values, new_values, entropy
    ):

        log_ratio = new_log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)

        policy_term_1 = advantages * ratio
        policy_term_2 = advantages * jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
        policy_loss = -jnp.mean(jnp.minimum(policy_term_1, policy_term_2))

        value_loss = jnp.mean((old_values - new_values) ** 2)

        # TODO the coefficients need to come from the config.
        total_loss = policy_loss + 0.5 * value_loss - 0.001 * jnp.mean(entropy)

        return total_loss

    def policy_loss(
        policy_params, observations, actions, old_log_probs, advantages, actor_apply_fn
    ):

        logits = actor_apply_fn(policy_params, observations)

        dist = tfd.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)

        log_ratio = new_log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)

        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # policy_term_1 = -advantages * ratio
        # policy_term_2 = -advantages * jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)

        policy_loss = rlax.clipped_surrogate_pg_loss(ratio, advantages, 0.2)
        loss = -jnp.mean(policy_loss)

        # loss = -jnp.mean(new_log_probs * advantages)

        return loss, loss

    def critic_loss(critic_params, observations, returns, critic_apply_fn):

        new_values = critic_apply_fn(critic_params, observations)
        new_values = jnp.squeeze(new_values)

        loss = jnp.mean((new_values - returns) ** 2)
        # loss = jnp.mean((returns) ** 2)

        return loss, loss

    def epoch_function(buffer, networks, optimisers):

        # sample_idxs = jax.random.permutation(key=key, x=len(buffer))

        # for i, episode in buffer.items():

        advantages = {}
        new_log_probs = {}

        # TODO the 0 is just get access to agent names.
        #print(buffer)
        #exit()
        for agent in buffer["observations"][0].keys():

            rewards_ = [agent_dict[agent] for agent_dict in buffer["rewards"]]
            values_ = [agent_dict[agent] for agent_dict in buffer["values"]]
            old_log_probs = [agent_dict[agent] for agent_dict in buffer["log_probs"]]
            actions = [agent_dict[agent] for agent_dict in buffer["actions"]]
            dones = [agent_dict[agent] for agent_dict in buffer["dones"]]

            # values_ = jax.lax.stop_gradient(values_)

            #print(rewards_)
            #print(values_)
            #exit()
            # entropy = [agent_dict[agent] for agent_dict in buffer["entropy"]]

            # TODO: Investigate this. 
            rlax_advantage = rlax.truncated_generalized_advantage_estimation(
                r_t=jnp.array(rewards_[:-1]),
                # TODO: Max episode horizon value here, also won't always be 1s.
                # discount_t=jnp.ones_like(jnp.array(values_[1:])) * 0.99,
                # discounts are 1 if 
                discount_t = jnp.array(dones)[:-1] * 0.99, 
                # lambda_=0.95,
                lambda_=0.95, 
                values=jnp.array(values_),
            )

            manual_advantage, manual_return = compute_gae(
                next_value=values_[-1], 
                rewards=rewards_[:-1],
                masks=jnp.array(dones)[1:],
                values=values_[:-1], 
                tau=0.95, 
            )


            num_timesteps = len(dones)
            clean_rl_advantage = np.zeros_like(np.array(dones))
            lastgaelam = 0
            for t in reversed(range(num_timesteps-1)):
                if t == num_timesteps - 2:
                    nextnonterminal = dones[t]
                    nextvalues = values_[t+1]
                else:
                    nextnonterminal = dones[t + 1]
                    nextvalues = values_[t+1]
                delta = rewards_[t] + 0.99 * nextvalues * nextnonterminal - values_[t]
                clean_rl_advantage[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam

            values_ = jax.lax.stop_gradient(values_)
            advantage = jax.lax.stop_gradient(rlax_advantage)
            # advantage = clean_rl_advantage[:-1]
            # returns = advantage + values_[1:]

            # targets = jnp.array(rewards_[:-1]) + 0.99 * jnp.array(values_[1: ]) 
            # advantage, returns = jax.lax.stop_gradient(targets - jnp.array(values_[:-1])), jax.lax.stop_gradient(targets)

            # advantage, returns = jax.lax.stop_gradient(jnp.array(manual_advantage)), jax.lax.stop_gradient(jnp.array(manual_return))
            


            returns = advantage + jnp.array(values_)[:-1]
            
            # print(f"Ret: {optax.global_norm(returns)}")

            returns = jax.lax.stop_gradient(returns)

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

            actor_net = networks["agent_0"]["actor_net"]
            actor_net_apply_fn = actor_net.apply
            actor_params = networks["agent_0"]["actor_params"]
            observations = jnp.array(observations)

            # TODO Change names to all be actor and not policy!!
            policy_grads, p_loss = jax.grad(policy_loss, has_aux=True)(
                actor_params,
                observations[:-1],
                jnp.array(actions)[:-1],
                jnp.array(old_log_probs)[:-1],
                advantage,
                actor_net_apply_fn,
            )

            updates, new_policy_optimiser_state = optimisers["agent_0"][
                "actor_optim"
            ].update(policy_grads, optimisers["agent_0"]["actor_state"])
            print(
                f"policy_loss: {jnp.round(p_loss, 2)} grads: {jnp.round(optax.global_norm(policy_grads), 2)} norm: {jnp.round(optax.global_norm(updates), 2)}"
            )
            new_policy_params = optax.apply_updates(actor_params, updates)

            # Update params
            networks["agent_0"]["actor_params"] = new_policy_params
            optimisers["agent_0"]["actor_state"] = new_policy_optimiser_state

            critic_net = networks["agent_0"]["critic_net"]
            critic_net_apply_fn = critic_net.apply
            critic_params = networks["agent_0"]["critic_params"]

            critic_grads, c_loss = jax.grad(critic_loss, has_aux=True)(
                critic_params, observations[:-1], returns, critic_net_apply_fn
            )

            updates, new_critic_optimiser_state = optimisers["agent_0"][
                "critic_optim"
            ].update(critic_grads, optimisers["agent_0"]["critic_state"])
            print(
                f"critic_loss {c_loss} grads: {jnp.round(optax.global_norm(critic_grads), 2)} norm: {jnp.round(optax.global_norm(updates), 2)}"
            )
            new_critic_params = optax.apply_updates(critic_params, updates)

            # Update params
            networks["agent_0"]["critic_params"] = new_critic_params
            optimisers["agent_0"]["critic_state"] = new_critic_optimiser_state

            if agent != "agent_0":
                networks[agent]["critic_params"] = networks["agent_0"]
                networks[agent]["actor_params"] = networks["agent_0"]
                optimisers[agent]["critic_state"] = optimisers["agent_0"]["critic_state"]
                optimisers[agent]["actor_state"] = optimisers["agent_0"]["actor_state"]

        return networks, optimisers

    return networks, config, prng, epoch_function, sample_fn, optimisers
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
    episodes = 200
    horizon = 200
    global_step = 0 
    test_buffer = {}
    episode = 0

    simple_buffer = {}
    simple_buffer["observations"] = []
    simple_buffer["actions"] = []
    simple_buffer["rewards"] = []
    simple_buffer["values"] = []
    simple_buffer["log_probs"] = []
    simple_buffer["entropy"] = []
    simple_buffer["dones"] = []

    for step in range(horizon):


        ep_return = 0
        timestep = env.reset()
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

            if timestep.last():
                simple_buffer["dones"].append({'agent_0': 0.0, 'agent_1': 0.0, 'agent_2': 0.0})
            else: 
                simple_buffer["dones"].append(timestep.discount)
            
            team_reward = 0
            for reward in timestep.reward.values():
                team_reward += reward
            team_reward /= 3
            ep_return += team_reward
            step += 1
            global_step += 1

            if timestep.last(): 

                x = 0 

            if global_step % horizon == 0: 

                for _ in range(NUM_EPOCHS):
                    networks, optimisers = epoch_function(simple_buffer, networks, optimisers)

                simple_buffer["observations"] = []
                simple_buffer["actions"] = []
                simple_buffer["rewards"] = []
                simple_buffer["values"] = []
                simple_buffer["log_probs"] = []
                simple_buffer["entropy"] = []
                simple_buffer["dones"] = []

        episode += 1
        
        #exit()
        print("")
        print(f"EPISODE {episode} RETURN: {ep_return}")
        print("")
        if episode == 0:
            test_buffer = simple_buffer.copy()
        # Can update an amount of episodes!
        # print(
        #     f"b: {[optax.global_norm(networks[a]['actor_params']) for a in networks.keys()]} {[optax.global_norm(networks[a]['critic_params']) for a in networks.keys()]} "
        # )
        # print(
        #     f"a: {[optax.global_norm(networks[a]['actor_params']) for a in networks.keys()]} {[optax.global_norm(networks[a]['critic_params']) for a in networks.keys()]} "
        # )

    # Next steps: Is it training?? / Nice way to log results.


if __name__ == "__main__":
    app.run(main)
