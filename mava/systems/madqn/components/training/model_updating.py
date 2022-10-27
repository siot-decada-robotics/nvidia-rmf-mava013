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

"""Trainer components for system updating."""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import tree
from acme.jax import networks as networks_lib
from acme.jax import utils
from jax.random import KeyArray
from optax._src import base as optax_base

from mava import constants
from mava.components.training import Utility
from mava.components.training.base import BatchDQN
from mava.core_jax import SystemTrainer


# @dataclass
# class MADQNMinibatchUpdateConfig:
#     pass
#
#
# class MADQNMinibatchUpdate(Utility):
#     def __init__(
#         self,
#         config: MADQNMinibatchUpdateConfig = MADQNMinibatchUpdateConfig(),
#     ):
#         """_summary_
#
#         Args:
#             config : _description_.
#         """
#         self.config = config
#
#     def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
#         """_summary_"""
#
#         def model_update_minibatch(
#             carry: Tuple[networks_lib.Params, networks_lib.Params, optax.OptState],
#             minibatch: BatchDQN,
#         ) -> Tuple[Tuple[Any, Any, optax.OptState], Dict[str, Any]]:
#             """Performs model update for a single minibatch."""
#             params, target_params, opt_states = carry
#
#             # Calculate the gradients and agent metrics.
#             gradients, agent_metrics = trainer.store.grad_fn(
#                 params,
#                 target_params,
#                 minibatch.observations,
#                 minibatch.next_observations,
#                 minibatch.actions,
#                 minibatch.discounts,
#                 minibatch.rewards,
#             )
#
#             # Update the networks and optimizers.
#             metrics = {}
#             # new_params = {}
#             # new_opt_states = {}
#             # new_target_params = copy.deepcopy(target_params)
#             for agent_key in trainer.store.trainer_agents:
#                 agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
#                 # Apply updates
#                 # TODO (dries): Use one optimizer per network type here and not
#                 # just one.
#                 updates, opt_states[agent_net_key] = trainer.store.optimizer.update(
#                     gradients[agent_key], opt_states[agent_net_key]
#                 )
#                 params[agent_net_key] = optax.apply_updates(
#                     params[agent_net_key], updates
#                 )
#
#                 agent_metrics[agent_key]["norm_grad"] = optax.global_norm(
#                     gradients[agent_key]
#                 )
#                 agent_metrics[agent_key]["norm_updates"] = optax.global_norm(updates)
#                 metrics[agent_key] = agent_metrics
#             return (params, target_params, opt_states), agent_metrics  # it
#             # was
#             # metrics before, but it doesnt make sense!
#
#         trainer.store.minibatch_update_fn = model_update_minibatch
#
#     @staticmethod
#     def name() -> str:
#         """_summary_
#
#         Returns:
#             _description_
#         """
#         return "minibatch_update_fn"
#
#     @staticmethod
#     def config_class() -> Callable:
#         """_summary_"""
#         return MADQNMinibatchUpdateConfig


@dataclass
class MADQNEpochUpdateConfig:
    num_epochs: int = 2  # not used in the implementation whichout scan
    batch_size: int = 128
    # learning_rate: float = 1e-3  # This should be a param of optim
    # adam_epsilon: float = 1e-5  # This should be a param of optim
    target_update_period = 10
    # max_gradient_norm: float = jnp.inf  # This should be a param of optim
    # optimizer: Optional[optax_base.GradientTransformation] = (None,)


class MADQNEpochUpdate(Utility):
    def __init__(
        self,
        config: MADQNEpochUpdateConfig = MADQNEpochUpdateConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        trainer.store.num_epochs = self.config.num_epochs

        # TODO (sasha): remove, ensure optimizer initialized somewhere else (only once)
        # if not self.config.optimizer:
        #     trainer.store.optimizer = optax.chain(
        #         optax.clip_by_global_norm(self.config.max_gradient_norm),
        #         optax.adam(self.config.learning_rate),
        #     )
        # else:
        #     trainer.store.optimizer = self.config.optimizer
        #
        # # Initialize optimizers.
        # trainer.store.opt_states = {}
        # for net_key in trainer.store.networks["networks"].keys():
        #     trainer.store.opt_states[net_key] = trainer.store.optimizer.init(
        #         trainer.store.networks["networks"][net_key].params
        #     )  # pytype: disable=attribute-error

        # TODO (sasha): re jit this
        # @jax.jit
        def model_update_epoch(
            carry: Tuple[Any, Any, KeyArray, Any, Any, Any, optax.OptState, BatchDQN],
            unused_t: Tuple[()],
        ) -> Tuple[
            Tuple[KeyArray, Any, Any, optax.OptState, BatchDQN],
            Dict[str, jnp.ndarray],
        ]:
            """Performs model updates based on one epoch of data."""
            key, params, target_params, opt_states, batch, steps, probs, keys = carry

            # Calculate the gradients and agent metrics.
            gradients, agent_metrics = trainer.store.grad_fn(
                params,
                target_params,
                batch.observations,
                batch.next_observations,
                batch.actions,
                batch.discounts,
                batch.rewards,
                probs,
                keys,
            )

            # Calcluating priorities
            priorities = agent_metrics["joint"][0].priorities

            # Average priorities because replay table stores all agents observations as a single entry
            for i in range(1, len(agent_metrics["joint"])):
                priorities += agent_metrics["joint"][i].priorities

            priorities /= len(agent_metrics["joint"])

            # TODO: Finish assigning priorities
            keys, priorities = tree.map_structure(
                # Fetch array and combine device and batch dimensions.
                lambda x: utils.fetch_devicearray(x).reshape((-1,) + x.shape[2:]),
                (keys, priorities),
            )

            # trainer.store.data_server_client.mutate_priorities(
            #     table="trainer", updates=dict(zip(keys, priorities))
            # )
            priority_updates = (keys, priorities)

            # Update the networks and optimizers.
            metrics = {}
            new_params = {}
            new_target_params = {}
            new_opt_states = {}

            steps += 1  # type: ignore

            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                # Apply updates
                # TODO (dries): Use one optimizer per network type here and not
                # just one.
                (
                    updates,
                    new_opt_states[agent_net_key],
                ) = trainer.store.policy_optimiser.update(
                    gradients[agent_key],
                    opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],
                )
                new_params[agent_net_key] = optax.apply_updates(
                    params[agent_net_key], updates
                )

                agent_metrics[agent_key]["norm_grad"] = optax.global_norm(
                    gradients[agent_key]
                )
                agent_metrics[agent_key]["norm_updates"] = optax.global_norm(updates)
                metrics[agent_key] = agent_metrics

                new_target_params[agent_net_key] = optax.periodic_update(
                    new_params[agent_net_key],
                    target_params[agent_net_key],
                    steps,
                    self.config.target_update_period,
                )

            return (
                (
                    new_params,
                    new_target_params,
                    new_opt_states,
                    batch,
                    steps,
                ),
                metrics,
                priority_updates,
            )

        trainer.store.epoch_update_fn = model_update_epoch

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "epoch_update"

    @staticmethod
    def config_class() -> Callable:
        """_summary_"""
        return MADQNEpochUpdateConfig
