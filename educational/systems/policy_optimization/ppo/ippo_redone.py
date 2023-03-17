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

"""Independent multi-agent JAX PPO. NOte that this implementation 
uses shared network weights between all agents."""

# MAJOR TODOS for this code: 
# 1. Use dm_env api instead fo gym. 
# 2. Heterogeneous agent support.
# 3. Different networks per agent.

import jax.numpy as jnp 
import numpy as np 
import jax 
import haiku as hk
import optax
import distrax
import rlax
import chex
import time 
import gym 
from typing import Optional, Any, Tuple 
from mava.utils.environments import debugging_utils
from mava.wrappers.debugging_envs import DebuggingEnvWrapper

from ppo_utils import (
    create_buffer, 
    add, 
    reset_buffer,
    BufferData, 
    PPOSystemState, 
    NetworkParams,
    OptimiserStates,
)

jit_add = jax.jit(add)

# Constants: 
HORIZON = 200 
CLIP_EPSILON = 0.2 
POLICY_LR = 0.005
CRITIC_LR = 0.005
DISCOUNT_GAMMA = 0.99 
GAE_LAMBDA = 0.95
NUM_EPOCHS = 3
NUM_MINIBATCHES = 8 
MAX_GLOBAL_NORM = 0.5
ADAM_EPS = 1e-5
POLICY_LAYER_SIZES = [64, 64]
CRITIC_LAYER_SIZES = [64, 64]

MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

NORMALISE_ADVANTAGE = True
ADD_ENTROPY_LOSS = True

ALGORITHM = "ff_indep_ppo"
LOG = False

@chex.dataclass
class EnvironmentConfig:
    env_name: str = "simple_spread"
    seed: int = 42
    type: str = "debug"
    action_space: str = "discrete"

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

# TODO: Assuming fully homogeneous agents here. 
# Handle this later on to be per agent. 

env, env_config = make_environment()

class MavaEnvToGym(DebuggingEnvWrapper): 

    def __init__(
        self,
        environment: DebuggingEnvWrapper,
        return_state_info: bool = False,
    ):
        super().__init__(environment=environment)

        self.return_state_info = return_state_info
        self.env = environment
        # TODO: This is hardcoded for the debug env. 
        single_agent_action_dim = environment.action_spec()["agent_0"].num_values
        full_obs_size = env.observation_spec()["agent_0"].observation.shape[0]
        self.action_space = gym.spaces.Discrete(single_agent_action_dim)
        self.n_agents = 3 
        self.observation_space = gym.spaces.Box(
            np.ones(full_obs_size) * -np.inf, 
            np.ones(full_obs_size) * np.inf, 
            (full_obs_size,), np.float32)

    def reset(self,):
        
        time_step = self.environment.reset()
        observations = [time_step.observation[key].observation for key in time_step.observation.keys()] 

        return observations



    def step(self, env_actions):

        actions = { f"agent_{i}":action for i, action in enumerate(env_actions)}
        time_step = self.env.step(actions)
        
        next_obs = [time_step.observation[key].observation for key in time_step.observation.keys()]  
        rewards = [time_step.reward[agent] for agent in time_step.reward.keys()] 

        terminals = [time_step.observation[key].terminal for key in time_step.observation.keys()]
        dones = [bool(terminals[i][0]) for i in range(len(terminals))]

        done = None 
        info = None 

        return next_obs, rewards, dones, info 



env = MavaEnvToGym(env)


num_actions = env.action_space.n
num_agents = env.n_agents
observation_dim = env.observation_space.shape[0]


# Make networks 

def make_networks(
    num_actions: int, 
    policy_layer_sizes: list = POLICY_LAYER_SIZES, 
    critic_layer_sizes: list = CRITIC_LAYER_SIZES, ):

    @hk.without_apply_rng
    @hk.transform
    def policy_network(x):

        return hk.nets.MLP(policy_layer_sizes + [num_actions])(x)

    @hk.without_apply_rng
    @hk.transform
    def critic_network(x):

        return hk.nets.MLP(critic_layer_sizes + [1])(x) 

    return policy_network, critic_network

policy_network, critic_network = make_networks(num_actions=num_actions)

# Create network params 

dummy_obs_data = jnp.zeros(observation_dim, dtype=jnp.float32)
networks_key, policy_init_key, critic_init_key = jax.random.split(networks_key, 3)

policy_params = policy_network.init(policy_init_key, dummy_obs_data)
critic_params = critic_network.init(critic_init_key, dummy_obs_data)

network_params = NetworkParams(
    policy_params=policy_params, 
    critic_params=critic_params,
)

# Create optimisers and states
policy_optimiser = optax.chain(
      optax.clip_by_global_norm(MAX_GLOBAL_NORM),
      optax.adam(learning_rate = POLICY_LR, eps = ADAM_EPS),
    )
critic_optimiser = optax.chain(
    optax.clip_by_global_norm(MAX_GLOBAL_NORM),
    optax.adam(learning_rate = CRITIC_LR, eps = ADAM_EPS),
    )


policy_optimiser_state = policy_optimiser.init(policy_params)
critic_optimiser_state = critic_optimiser.init(critic_params)

# Better idea is probably a high level Policy and Critic state. 

optimiser_states = OptimiserStates(
    policy_state=policy_optimiser_state, 
    critic_state=critic_optimiser_state, 
)

# Initialise buffer 
buffer_state = create_buffer(
    buffer_size=HORIZON, 
    num_agents=num_agents, 
    num_envs=1, 
    observation_dim=observation_dim, 
)

system_state = PPOSystemState(
    buffer=buffer_state, 
    actors_key=actors_key, 
    networks_key=networks_key, 
    network_params=network_params, 
    optimiser_states=optimiser_states, 
) 

@jax.jit
@chex.assert_max_traces(n=1)
def choose_action(
    logits,  
    actors_key,
    ):
    
    actors_key, sample_key = jax.random.split(actors_key)

    dist = distrax.Categorical(logits=logits)

    action, logprob = dist.sample_and_log_prob(
        seed = sample_key, 
    )
    entropy = dist.entropy()

    return actors_key, action, logprob, entropy

def policy_loss(
    policy_params, 
    states, 
    actions, 
    old_log_probs, 
    advantages, 
    entropies_):

    logits = policy_network.apply(policy_params, states)
    dist = distrax.Categorical(logits=logits)

    new_log_probs = dist.log_prob(value=actions)

    logratio = new_log_probs - old_log_probs
    ratio = jnp.exp(logratio)

    # Policy loss
    loss_term_1 = -advantages * ratio
    loss_term_2 = -advantages * jnp.clip(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
    loss = jnp.maximum(loss_term_1, loss_term_2).mean() 
    if ADD_ENTROPY_LOSS: 
        loss -= 0.01 * jnp.mean(entropies_)

    # jax.debug.print("policy loss {x}", x= loss)

    return loss

def critic_loss(
    critic_params, 
    states, 
    returns
    ):

    new_values = jnp.squeeze(critic_network.apply(critic_params, states))
    
    loss = 0.5 * ((new_values - returns) ** 2).mean()
    # jax.debug.print("critic loss {x}", x= loss)
    return loss

@jax.jit
@chex.assert_max_traces(n=1)
def update_policy(system_state: PPOSystemState, advantages, mb_idx): 

    for agent in range(num_agents): 
        states_ = jnp.squeeze(system_state.buffer.states[:,:,agent,:])[mb_idx]
        old_log_probs_ = jnp.squeeze(system_state.buffer.log_probs[:,:,agent])[mb_idx]
        actions_ = jnp.squeeze(system_state.buffer.actions[:,:,agent,:])[mb_idx]
        entropies_ = jnp.squeeze(system_state.buffer.entropy[:,:,agent])[mb_idx]
        advantages_ = advantages[:, agent][mb_idx]

        if NORMALISE_ADVANTAGE: 
            advantages_ = (advantages_ - jnp.mean(advantages_)) / (jnp.std(advantages_) + 1e-5)
        
        policy_optimiser_state = system_state.optimiser_states.policy_state
        policy_params = system_state.network_params.policy_params

        grads = jax.grad(policy_loss)(
            policy_params, 
            states_, 
            actions_, 
            old_log_probs_, 
            advantages_,
            entropies_,
        )

        updates, new_policy_optimiser_state = policy_optimiser.update(grads, policy_optimiser_state)
        new_policy_params = optax.apply_updates(policy_params, updates)

        system_state.optimiser_states.policy_state = new_policy_optimiser_state
        system_state.network_params.policy_params = new_policy_params

    return system_state

@jax.jit
@chex.assert_max_traces(n=1)
def update_critic(
    system_state: PPOSystemState, 
    returns,
    mb_idx,
): 

    for agent in range(num_agents): 
        states_ = jnp.squeeze(system_state.buffer.states[:,:,agent,:])[mb_idx]
        returns_ = returns[:, agent][mb_idx]
        
        critic_optimiser_state = system_state.optimiser_states.critic_state
        critic_params = system_state.network_params.critic_params

        grads = jax.grad(critic_loss)(
            critic_params, 
            states_, 
            returns_,
        )

        updates, new_critic_optimiser_state = critic_optimiser.update(grads, critic_optimiser_state)
        new_critic_params = optax.apply_updates(critic_params, updates)

        system_state.optimiser_states.critic_state = new_critic_optimiser_state
        system_state.network_params.critic_params = new_critic_params

    return system_state

# NOTE: Can terminate episode if one agent is done. Doesn't have to be all agents. 

global_step = 0
episode = 0 
log_data = {}
while global_step < 100_000: 

    team_done = False 
    obs = env.reset()
    obs = jnp.array(obs, dtype=jnp.float32) 
    episode_return = 0
    episode_step = 0 
    start_time = time.time()
    while not team_done: 
        
        # For stepping the environment
        step_joint_action = jnp.empty(num_agents, dtype=jnp.int32)

        # Data to append to buffer
        act_joint_action = jnp.empty((num_agents,1), dtype=jnp.int32)
        act_values = jnp.empty((num_agents), dtype=jnp.float32)
        act_log_probs = jnp.empty((num_agents), dtype=jnp.float32)
        act_entropies = jnp.empty((num_agents), dtype=jnp.float32)

        for agent in range(num_agents):
            # logits = policy_network.apply(system_state.network_params.policy_params, jnp.array(obs[agent], dtype=jnp.float32))
            logits = policy_network.apply(system_state.network_params.policy_params, obs[agent])
            actors_key = system_state.actors_key
            actors_key, action, logprob, entropy = choose_action(logits, actors_key)
            system_state.actors_key = actors_key

            # value = jnp.squeeze(critic_network.apply(system_state.network_params.critic_params, jnp.array(obs[agent], dtype=jnp.float32)))
            value = jnp.squeeze(critic_network.apply(system_state.network_params.critic_params, obs[agent]))

            step_joint_action = step_joint_action.at[agent].set(action)
            
            act_joint_action = act_joint_action.at[agent, 0].set(action)
            act_values = act_values.at[agent].set(value)
            act_log_probs = act_log_probs.at[agent].set(logprob)
            act_entropies = act_entropies.at[agent].set(entropy)

        # Covert action to int in order to step the env. 
        # Can also handle in the wrapper
        obs_, reward, done, _ = env.step(step_joint_action.tolist())  
        obs_ = jnp.array(obs_, dtype=jnp.float32)     

        team_done = all(done)
        global_step += 1 # TODO: With vec envs this should be more. 
        
        # NB: Correct shapes here. 
        data = BufferData(
            state = jnp.expand_dims(jnp.array(obs, dtype=jnp.float32), axis=0), 
            action = jnp.expand_dims(act_joint_action, axis=0), 
            reward = jnp.expand_dims(jnp.array(reward, dtype=jnp.float32), axis=0), 
            done = jnp.expand_dims(jnp.array(done, dtype=bool), axis=0), 
            log_prob = jnp.expand_dims(act_log_probs, axis=0), 
            value = jnp.expand_dims(act_values, axis=0), 
            entropy = jnp.expand_dims(act_entropies, axis=0)
        )

        buffer_state = system_state.buffer 
        buffer_state = add(buffer_state, data)
        # buffer_state = add(buffer_state, data)
        system_state.buffer = buffer_state

        obs = obs_ 
        episode_return += jnp.mean(jnp.array(reward, dtype=jnp.float32))
        episode_step += 1 
        
        if global_step % (HORIZON + 1) == 0: 
            
            advantages = jnp.empty_like(jnp.squeeze(system_state.buffer.rewards)[:-1], dtype=jnp.float32)
            returns = jnp.empty_like(jnp.squeeze(system_state.buffer.rewards)[:-1], dtype=jnp.float32)

            for agent in range(num_agents): 
                
                advantage = rlax.truncated_generalized_advantage_estimation(
                    r_t = jnp.squeeze(system_state.buffer.rewards[:,:,agent])[:-1],
                    discount_t = (1 - jnp.squeeze(system_state.buffer.dones[:,:,agent]))[:-1] * DISCOUNT_GAMMA,
                    lambda_ = GAE_LAMBDA, 
                    values = jnp.squeeze(system_state.buffer.values[:,:,agent]),
                    stop_target_gradients=True
                )

                advantage = jax.lax.stop_gradient(advantage)
                # Just not sure how to index the values here. 
                return_ = advantage + jnp.squeeze(system_state.buffer.values[:,:,agent])[:-1]
                return_ = jax.lax.stop_gradient(return_)

                advantages = advantages.at[:, agent].set(advantage)
                returns = returns.at[:, agent].set(return_)

            # TODO: 
            # 1. VMAP over advantage and return calculations
            # 2. Scan the epoch update
            # 3. Scan / vmap over agents in the loss. 

            for _ in range(NUM_EPOCHS):
                
                # Create data minibatches 
                # Generate random numbers 
                networks_key, sample_idx_key = jax.random.split(system_state.networks_key)
                system_state.actors_key = networks_key

                idxs = jax.random.permutation(key = sample_idx_key, x=HORIZON)
                mb_idxs = jnp.split(idxs, NUM_MINIBATCHES)

                for mb_idx in mb_idxs:
                    system_state = update_policy(system_state, advantages, mb_idx)
                    system_state = update_critic(system_state, returns, mb_idx)
                
                
            buffer_state = reset_buffer(buffer_state) 
            system_state.buffer = buffer_state

    sps = episode_step / (time.time() - start_time)

    episode += 1
    if episode % 1 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {jnp.round(episode_return, 3)}, SPS: {int(sps)}")   