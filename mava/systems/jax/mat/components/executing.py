from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from acme.jax import utils

from mava.components.jax.executing.action_selection import (
    ExecutorSelectAction,
    ExecutorSelectActionConfig,
)
from mava.core_jax import SystemExecutor
from mava.types import OLT
from mava.utils.jax_tree_utils import stack_trees, index_stacked_tree


class MatExecutorActionSelection(ExecutorSelectAction):
    def __init__(
        self, config: ExecutorSelectActionConfig = ExecutorSelectActionConfig()
    ):
        super().__init__(config)

    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        # TODO (sasha): do this once and put it in the store
        agent_spec = list(executor.store.environment_spec.get_agent_specs().values())[0]

        executor.store.actions_info = {}
        executor.store.policies_info = {}

        batch_size = 1
        num_agents = len(executor.store.observations)
        # only works for discrete action spaces
        action_dim = agent_spec.actions.num_values

        # setting up initial token for no actions
        executor.store.previous_actions = jnp.zeros(
            (batch_size, num_agents, action_dim + 1)
        )
        # for every agent's action in the batch, set initial action to 1
        # TODO (sasha): 1 seems like a bad value as it is an actual action -
        #  except that this is an extra dim, so...?
        #  ---------------------------------------------------------------
        #  why does it only set at agent=0 action=0, but when setting current action, it sets from
        #  1:end for all agent dims?
        executor.store.previous_actions.at[:, 0, 0].set(1)

        # TODO (sasha): this only allows for shared actions
        network = executor.store.networks["networks"]["network_agent"]

        # [idea] TODO (sasha): should I stack this on the last dim because want agents to be
        #         treated as channels? (what about embedding then?)
        stacked_obs = stack_observations(list(executor.store.observations.values()))
        encoded_obs = network.encode_observations(
            utils.add_batch_dim(stacked_obs.observation)
        )
        stacked_encoded_obs = OLT(
            encoded_obs, stacked_obs.legal_actions, stacked_obs.terminal
        )

        for agent in executor.store.observations.keys():
            action_info, policy_info = executor.select_action(
                agent, stacked_encoded_obs
            )
            # TODO (sasha): ideally this shouldn't be here but otherwise everything complains
            #  that actions are int32's and reverb wants int64
            executor.store.actions_info[agent] = np.int64(action_info)
            executor.store.policies_info[agent] = policy_info

    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        # TODO (sasha): assert that you are only using shared weights and the same net
        #  for all agents
        rng_key, executor.store.key = jax.random.split(executor.store.key)
        network = executor.store.networks["networks"]["network_agent"]
        agent_ind = int(executor.store.agent.split("_")[1])

        # TODO (sasha): do this once and put it in the store
        agent_spec = list(executor.store.environment_spec.get_agent_specs().values())[0]
        action_dim = agent_spec.actions.num_values

        observation = executor.store.observation.observation
        legals = index_stacked_tree(executor.store.observation, agent_ind).legal_actions

        executor.store.action_info, executor.store.policy_info = network.get_action(
            observation,
            executor.store.previous_actions,
            agent_ind,
            rng_key,
            legals,
        )

        # TODO (sasha): why only set from 1:?
        action = jax.nn.one_hot(executor.store.action_info, action_dim)
        executor.store.previous_actions.at[:, agent_ind + 1, 1:].set(action)


def stack_observations(observations: List[OLT]):
    return stack_trees(observations)
