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
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from absl import app, flags

from mava import specs as mava_specs
from mava.utils.environments import debugging_utils, smac_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("system", "test agent", "What agent is running.")
flags.DEFINE_string(
    "base_dir",
    os.path.expanduser("~") + "/mava",
    "Base dir to store experiment data e.g. checkpoints.",
)

# TODO: clean up alot of these stuffs we are not using


@dataclass
class InitConfig:
    seed: int = 42879
    learning_rate: float = 1e-4
    buffer_size: int = 10000
    min_epsilon: float = 0.1
    max_epsilon: float = 1
    duration: float = 10000 # duration of exploration
    total_num_steps: int = 1500000
    batch_size: int = 960
    update_frequency: int = 50
    training_frequency: int = 10
    tau: float = 1.0
    logging_frequency: int = 5
    exp_name: str = str(datetime.now())


@dataclass
class EnvironmentConfig:
    env_name: str = "3m" # "simple_spread"
    seed: int = 42678
    type: str = "smac" # "debug"
    action_space: str = "discrete"


@dataclass
class SystemConfig:
    name: str = "random"
    seed: int = 42789


class TrainingState(NamedTuple):
    params: Dict[str, hk.Params]
    target_params: Dict[str, hk.Params]
    opt_state: Dict[str, optax.OptState]
    training_steps: int


class TransitionBatch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    actions: Any
    observation: Any
    next_observation: Any
    reward: Any
    done: Any
    legal_actions: Any


def from_singles(action, obs, next_obs, reward, done) -> TransitionBatch:

    agents = action.keys()
    actions = {agent: jnp.array([action[agent]]) for agent in agents}
    rewards = {agent: jnp.array([reward[agent]]) for agent in agents}
    dones = {agent: jnp.array([done], dtype=bool) for agent in agents}
    obs_ = {agent: obs[agent].observation for agent in agents}
    next_obs_ = {agent: next_obs[agent].observation for agent in agents}
    legal_actions = {agent: next_obs[agent].legal_actions for agent in agents}

    return TransitionBatch(
        actions=actions,
        observation=obs_,
        next_observation=next_obs_,
        reward=rewards,
        done=dones,
        legal_actions=legal_actions,
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
    config: EnvironmentConfig = EnvironmentConfig(),
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
    elif config.type == "smac":
        env, _ = smac_utils.make_environment(map_name=config.env_name)
    else:
        raise ValueError(f"Config type {config.type} not yet supported.")
    return env  # , config


def make_system(
    environment_spec: mava_specs.MAEnvironmentSpec,
) -> Tuple[Any, SystemConfig]:
    """Inits and returns system/networks.

    Args:
        config : system config.

    Returns:
        system.
    """
    prng = jax.random.PRNGKey(478678)
    q_key, _ = jax.random.split(prng)
    agent_specs = environment_spec.get_agent_environment_specs()

    # add more options to this
    def make_q_net(hidden_layer_sizes):
        def make_actor(x):
            return hk.nets.MLP(hidden_layer_sizes)(x)

        return hk.without_apply_rng(hk.transform(make_actor))

    def make_networks():
        networks = {}

        agents = list(agent_specs.keys())
        net_keys = [agents[0]] # agent_0

        # Try a shared network with key agent_0 as net_key
        for net_key in net_keys:
            spec = agent_specs[net_key]
            num_actions = spec.actions.num_values
            q_net = make_q_net([64, 64, num_actions])
            obs_dim = spec.observations.observation.shape
            obs_type = spec.observations.observation.dtype
            dummy_obs = jnp.zeros(shape=obs_dim, dtype=obs_type)

            q_network_params = q_net.init(q_key, dummy_obs)
            networks[net_key] = {}
            networks[net_key]["actor_params"] = q_network_params
            networks[net_key]["actor_net"] = q_net

        return networks

    return make_networks()


class ReplayBuffer:
    def __init__(self, max_len=100, seed=0):

        self.buffer = deque(maxlen=max_len)
        self.num_items = 0
        random.seed(seed)

    def add(self, data):
        self.buffer.extend([data])
        self.num_items = len(self.buffer)

    def sample(self, batch_size: int) -> TransitionBatch:
        transitions = random.sample(self.buffer, batch_size)
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *transitions)

    @property
    def size(self):
        return self.num_items


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main(_: Any) -> None:
    """Template for educational system implementations.

    Args:
        _ : unused param - for absl.
    """

    # Init env and system.
    config = init()
    env = make_environment()
    test_env = make_environment()
    env_spec = mava_specs.MAEnvironmentSpec(env)
    agent_specs = env_spec.get_agent_environment_specs()
    networks = make_system(env_spec)
    replay_buffer = ReplayBuffer(config.buffer_size)

    # set up logger
    time_delta = 1
    logger = logger_utils.make_logger(
        directory=f"{FLAGS.base_dir}/{config.exp_name}",
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=str(datetime.now()),
        time_delta=time_delta,
        label="iqdn_system",
    )

    # Initialise optimisers states and params
    optimisers = {}
    initial_params = {}
    initial_opt_state = {}

    for net_key in networks.keys():
        optimisers[net_key] = optax.chain(
            optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-config.learning_rate)
        )
        initial_params[net_key] = networks[net_key]["actor_params"]
        initial_opt_state[net_key] = optimisers[net_key].init(initial_params[net_key])

    state = TrainingState(initial_params, initial_params, initial_opt_state, 0)

    total_num_steps = config.total_num_steps
    min_epsilon, max_epsilon = config.min_epsilon, config.max_epsilon

    test_episodes = 5 #config.update_frequency
    num_agents = len(agent_specs)

    @jax.jit
    def sample_fn(observation, q_params):
        action = {}
        
        agents = list(observation.keys())
        net_key = agents[0] # agent_0

        for agent_key, agent_olt in observation.items():
            q_network = networks[net_key]["actor_net"]
            params = q_params[net_key]
            agent_observation = agent_olt.observation
            q_values = q_network.apply(params, agent_observation) 
            q_values = jnp.where(agent_olt.legal_actions==1.0, q_values, -jnp.inf)
            action[agent_key] = q_values.argmax(axis=-1)

        return action

    @jax.jit
    def update(state: TrainingState, batch: TransitionBatch):

        loss = {}
        q_preds_dict = {}
        q_params_dict = {}
        opt_state_dict = {}

        (actions, obs, next_obs, rewards, dones, legal_actions) = (
            batch.actions,
            batch.observation,
            batch.next_observation,
            batch.reward,
            batch.done,
            batch.legal_actions,
        )

        agents = list(dones.keys())
        net_key = agents[0] # agent_0

        def mse_loss(params, target_params, actions, obs, next_obs, rewards, dones, mask):
            # since the networks are homogenous
            q_tm1 = networks[net_key]["actor_net"].apply(params, obs)
            q_t_value = networks[net_key]["actor_net"].apply(target_params, next_obs)
            q_t_selector = networks[net_key]["actor_net"].apply(params, next_obs)

            q_t_selector = jnp.where(mask == 1.0, q_t_selector, -jnp.inf)

            batch_double_q_learning_loss_fn = jax.vmap(
                rlax.double_q_learning, (0, 0, 0, 0, 0, 0, None)
            )
            
            discounts = (1 - dones) * 0.99

            error = batch_double_q_learning_loss_fn(
                q_tm1,
                actions.squeeze(),
                rewards.squeeze(),
                discounts.squeeze(),
                q_t_value,
                q_t_selector,
                True,
            )

            loss = jax.numpy.mean(rlax.l2_loss(error))
            return loss, q_tm1.squeeze()

        for agent_key in agents:
            q_params = state.params[net_key]
            q_params_target = state.target_params[net_key]
            
            actions_ = actions[agent_key]
            obs_ = obs[agent_key]
            next_obs_ = next_obs[agent_key]
            rewards_ = rewards[agent_key]
            dones_ = dones[agent_key]
            mask = legal_actions[agent_key]

            (loss[agent_key], q_preds_dict[agent_key]), grads = jax.value_and_grad(
                mse_loss, has_aux=True
            )(q_params, q_params_target, actions_, obs_, next_obs_, rewards_, dones_, mask)

            updates, opt_state_dict[net_key] = optimisers[net_key].update(
                grads, state.opt_state[net_key]
            )
            q_params_dict[net_key] = optax.apply_updates(q_params, updates)

            state = TrainingState(q_params_dict, state.target_params, opt_state_dict, state.training_steps)
        
        state = TrainingState(q_params_dict, state.target_params, opt_state_dict, state.training_steps+1)

        return loss, q_preds_dict, state

    def test(env, num_episodes, q_params):
        score = np.zeros(num_agents)
        steps_ran =0
        for _ in range(num_episodes):
            timestep = env.reset()
            if type(timestep) == tuple:
                timestep, _ = timestep
            while not timestep.last():
                actions = sample_fn(timestep.observation, q_params)
                timestep = env.step(actions)
                if type(timestep) == tuple:
                    timestep, _ = timestep
                score += np.array(list(timestep.reward.values()))
                steps_ran += 1

        return sum(score / num_episodes), steps_ran

    rng_key = jax.random.PRNGKey(config.seed)
    score = np.zeros(num_agents)
    episode_scores = np.zeros(config.logging_frequency)
    global_num_steps = 0
    eval_num_steps = 0
    episode = 0
    target_params_updated_atleast_once = False

    while global_num_steps < total_num_steps:

        rng_key, episode_key = jax.random.split(rng_key)

        timestep = env.reset()
        if type(timestep) == tuple:
            timestep, _ = timestep
        
        while not timestep.last():

            epsilon = linear_schedule(
                  max_epsilon, min_epsilon, config.duration, global_num_steps
                )

            episode_key, action_key = jax.random.split(episode_key)
            
            if jax.random.uniform(action_key) < epsilon:
                actions = {
                    agent: np.random.choice(len(timestep.observation[agent].legal_actions), 
                                p=timestep.observation[agent].legal_actions/np.sum(timestep.observation[agent].legal_actions))
                    for agent in agent_specs.keys()
                }
            else:
                actions = sample_fn(timestep.observation, state.params)

            new_timestep = env.step(actions)
            if type(new_timestep) == tuple:
                new_timestep, _ = new_timestep

            transition_data = from_singles(
                actions,
                timestep.observation,
                new_timestep.observation,
                new_timestep.reward,
                timestep.last(),
            )

            replay_buffer.add(transition_data)

            timestep = new_timestep

            score += np.array(list(new_timestep.reward.values()))

            global_num_steps += 1
            
            # train if time to train
            if global_num_steps > config.batch_size:

                # train for a certain number of iterations
                # for _ in range(config.training_frequency):
                batch = replay_buffer.sample(config.batch_size)
                loss, q_values, state = update(state, batch)

                loss_dict = {
                    f"losses/td_loss-{agent}": jax.device_get(loss[agent])
                    for agent in agent_specs.keys()
                }
                qvalues_dict = {
                    f"losses/q_values-{agent}": jax.device_get(q_values[agent]).mean()
                    for agent in agent_specs.keys()
                }

                loss_dict["losses/training_steps"] = state.training_steps
                
            # update the network
            if state.training_steps % config.update_frequency == 0 and state.training_steps > 0:
                state = state._replace(
                    target_params=optax.incremental_update(
                        state.params, state.target_params, config.tau
                    )
                )
                target_params_updated_atleast_once = True

        episode_scores[episode%config.logging_frequency] = np.sum(score)
        score = np.zeros(num_agents)
        episode += 1

        if episode % config.logging_frequency == 0 and target_params_updated_atleast_once:
            test_score, steps_ran = test(test_env, test_episodes, state.params)
            eval_num_steps += steps_ran

            train_score = np.mean(episode_scores[-test_episodes:])

            exec_stats = env.get_stats()
            eval_stats = test_env.get_stats()

            qvalues_dict["losses/avg_train_return"] = train_score/num_agents
            qvalues_dict["losses/avg_test_return"] = test_score/num_agents
            loss_dict["epsilon"] = epsilon
            loss_dict["eval_steps"] = eval_num_steps
            if exec_stats:
                loss_dict["eval_win_rate"] = eval_stats["cumulative_win_rate"]
                loss_dict["eval_battles_draw"] = eval_stats["battles_draw"]
                loss_dict["eval_battles_won"] = eval_stats["battles_won"]
                loss_dict["eval_battles_game"] = eval_stats["battles_game"]
                loss_dict["executor_win_rate"] = exec_stats["cumulative_win_rate"]
                loss_dict["executor_battles_draw"] = exec_stats["battles_draw"]
                loss_dict["executor_battles_won"] = exec_stats["battles_won"]
                loss_dict["executor_battles_game"] = exec_stats["battles_game"]
            
            loss_dict["executor_steps"] = global_num_steps

            write_dict = loss_dict | qvalues_dict
            logger.write(write_dict)


if __name__ == "__main__":
    app.run(main)
