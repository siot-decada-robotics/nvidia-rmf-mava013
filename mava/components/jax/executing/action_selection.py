# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

"""Execution components for system builders"""

from dataclasses import dataclass
from typing import Any, Callable, Dict

import acme.jax.utils as utils
import jax
import jax.numpy as jnp
from acme.jax import utils

from mava.components.jax import Component
from mava.core_jax import SystemExecutor
from mava.systems.jax.mamcts.mcts import MCTS, MaxDepth, RecurrentFn, RootFn, TreeSearch
from mava.systems.jax.mamcts.networks import LearnedModelNetworks, MAMCTSNetworks


@dataclass
class ExecutorSelectActionProcessConfig:
    pass


class FeedforwardExecutorSelectAction(Component):
    def __init__(
        self,
        config: ExecutorSelectActionProcessConfig = ExecutorSelectActionProcessConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    # Select actions
    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """Summary"""
        executor.store.actions_info = {}
        executor.store.policies_info = {}
        for agent, observation in executor.store.observations.items():
            action_info, policy_info = executor.select_action(agent, observation)
            executor.store.actions_info[agent] = action_info
            executor.store.policies_info[agent] = policy_info

    # Select action
    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        """Summary"""

        agent = executor.store.agent
        network = executor.store.networks["networks"][
            executor.store.agent_net_keys[agent]
        ]

        observation = utils.add_batch_dim(executor.store.observation.observation)

        rng_key, executor.store.key = jax.random.split(executor.store.key)

        # TODO (dries): We are currently using jit in the networks per agent.
        # We can also try jit over all the agents in a for loop. This would
        # allow the jit function to save us even more time.
        executor.store.action_info, executor.store.policy_info = network.get_action(
            observation,
            rng_key,
            utils.add_batch_dim(executor.store.observation.legal_actions),
        )

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "executor_select_action"


@dataclass
class MCTSConfig:
    root_fn: RootFn = None
    recurrent_fn: RecurrentFn = None
    search: TreeSearch = None
    environment_model: Any = None
    num_simulations: int = 10
    evaluator_num_simulations: int = 50
    max_depth: MaxDepth = None
    other_search_params: Callable[[None], Dict[str, Any]] = lambda: {}
    evaluator_other_search_params: Callable[[None], Dict[str, Any]] = lambda: {}


class MCTSFeedforwardExecutorSelectAction(FeedforwardExecutorSelectAction):
    """MCTS action selection"""

    def __init__(
        self,
        config: MCTSConfig = MCTSConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        super().__init__(config)

    def on_execution_init_start(self, executor: SystemExecutor) -> None:

        if None in [self.config.root_fn, self.config.recurrent_fn, self.config.search]:
            raise ValueError("Required arguments for MCTS config have not been given")

        self.mcts = MCTS(self.config)

    # TODO figure out how to pass agent ids since it is a string
    # Select action
    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        """Summary"""

        agent = executor.store.agent
        network = executor.store.networks["networks"][
            executor.store.agent_net_keys[agent]
        ]

        rng_key, executor.store.key = jax.random.split(executor.store.key)

        observation = utils.add_batch_dim(executor.store.observation.observation)
        action_mask = executor.store.observation.legal_actions

        if self.config.environment_model is None and isinstance(
            network, LearnedModelNetworks
        ):
            executor.store.environment_state_history[agent].append(
                executor.store.observation.observation
            )

            def pad_observation_history(stacked_observation_history):
                padded_hist_size = (
                    self.config.history_size - stacked_observation_history.shape[2]
                )
                padded_obs = jnp.zeros(
                    (*stacked_observation_history.shape[0:-1], 1),
                    stacked_observation_history.dtype,
                )
                padded_obs = jnp.repeat(padded_obs, padded_hist_size, axis=-1)
                stacked_observation_history = jnp.concatenate(
                    [padded_obs, stacked_observation_history], axis=-1
                )
                return stacked_observation_history

            stacked_observation_history = jnp.stack(
                executor.store.environment_state_history[agent], -1
            )
            stacked_observation_history = jax.lax.cond(
                stacked_observation_history.shape[-1] < self.config.history_size,
                pad_observation_history,
                lambda x: x,
                stacked_observation_history,
            )

            stacked_observation_history = utils.add_batch_dim(
                stacked_observation_history
            )

            (
                executor.store.action_info,
                executor.store.policy_info,
            ) = self.mcts.learned_get_action(
                network,
                rng_key,
                stacked_observation_history,
                executor.store.is_evaluator,
                action_mask,
            )

            executor.store.action_history[agent].append(executor.store.action_info)

        elif self.config.environment_model is not None and isinstance(
            network, MAMCTSNetworks
        ):
            (
                executor.store.action_info,
                executor.store.policy_info,
            ) = self.mcts.get_action(
                network.forward_fn,
                network.params,
                rng_key,
                executor.store.environment_state,
                observation,
                agent,
                executor.store.is_evaluator,
                action_mask,
            )
        else:
            raise NotImplementedError(
                "Currently Monte Carlo Tree Search requires an environment model or a LearnedModelNetwork"
            )

    @staticmethod
    def config_class() -> Callable:
        return MCTSConfig
