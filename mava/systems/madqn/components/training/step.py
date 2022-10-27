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

"""Trainer components for gradient step calculations."""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
import reverb
import tree
from acme.jax import utils
from jax import jit

from mava import constants
from mava.components.training.base import BatchDQN, TrainingStateDQN, Step
from mava.core_jax import SystemTrainer


@dataclass
class MADQNStepConfig:
    target_update_period: int = 10
    discounts: float = 0.99  # this is defined somewhere else, I guess in transition.


class MADQNStep(Step):
    def __init__(
        self,
        config: MADQNStepConfig = MADQNStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        # TODO (sasha): jit
        # @jit
        # @chex.assert_max_traces(n=1)
        def sgd_step(
            states: TrainingStateDQN,
            sample: reverb.ReplaySample,
        ) -> Tuple[TrainingStateDQN, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""
            # Extract the data.
            data = sample.data
            keys, probs, *_ = sample.info
            keys, probs = jnp.array(keys), jnp.array(probs)

            # TODO (sasha): why in the world must I explicitly make this stuff jnp arrays?
            #  it should already be, but they seem to be tf.Tensors
            (
                observations,
                new_observations,
                actions,
                rewards,
                discounts,
                _,
            ) = jax.tree_map(
                lambda x: jnp.array(x),
                (
                    data.observations,
                    data.next_observations,
                    data.actions,
                    data.rewards,
                    data.discounts,
                    data.extras,
                ),
            )

            discounts = tree.map_structure(
                lambda x: x * self.config.discounts, discounts
            )

            trajectories = BatchDQN(
                observations=observations,
                next_observations=new_observations,
                actions=actions,
                rewards=rewards,
                discounts=discounts,
            )

            trajectories = jax.tree_map(lambda x: jnp.array(x), trajectories)
            batch = trajectories
            next_rng_key, rng_key = jax.random.split(states.random_key)

            (
                (
                    new_params,
                    new_target_params,
                    new_opt_states,
                    _,
                    steps,
                ),
                metrics,
                priority_updates,
            ) = trainer.store.epoch_update_fn(
                (
                    rng_key,
                    states.params,
                    states.target_params,
                    states.opt_states,
                    batch,
                    states.steps,
                    probs,
                    keys,
                ),
                {},
            )

            # Update the training states.
            new_states = TrainingStateDQN(
                params=new_params,
                target_params=new_target_params,
                opt_states=new_opt_states,
                random_key=next_rng_key,
                steps=steps,
            )

            # Set the metrics
            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["norm_params"] = optax.global_norm(new_states.params)
            metrics["observations_mean"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_map(
                        lambda x: jnp.abs(jnp.mean(x, axis=(0, 1))), observations
                    ),
                    num_batch_dims=0,
                )
            )
            metrics["observations_std"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_map(lambda x: jnp.std(x, axis=(0, 1)), observations),
                    num_batch_dims=0,
                )
            )
            metrics["rewards_mean"] = jnp.mean(
                utils.batch_concat(rewards, num_batch_dims=0)
            )
            metrics["rewards_std"] = jnp.std(
                utils.batch_concat(rewards, num_batch_dims=0)
            )

            return new_states, metrics, priority_updates

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            networks = trainer.store.networks
            target_networks = trainer.store.target_networks
            params = {
                net_key: networks[net_key].policy_params for net_key in networks.keys()
            }
            target_params = {
                net_key: target_networks[net_key].policy_params
                for net_key in target_networks.keys()
            }

            # trainer.store.policy_opt_states[net_key] = {
            #     constants.OPT_STATE_DICT_KEY: builder.store.policy_optimiser.init(
            #         builder.store.networks[net_key].policy_params
            #     )
            # }  # pytype: disable=attribute-error
            opt_states = trainer.store.policy_opt_states  # trainer.store.opt_states
            random_key, trainer.store.base_key = jax.random.split(
                trainer.store.base_key
            )
            # keys, probs, *_ = sample.info

            steps = trainer.store.trainer_counts["trainer_steps"]

            states = TrainingStateDQN(
                params=params,
                target_params=target_params,
                opt_states=opt_states,
                random_key=random_key,
                steps=steps,
            )

            new_states, metrics, priority_updates = sgd_step(states, sample)

            # priority_updates is a tuple of reverb keys and new priorities
            # converts device arrays to lists
            prio_updates_list = map(lambda x: x.tolist(), priority_updates)
            # TODO (sasha): "trainer_0" is hard coded, need to update priorities per table
            # udpating the reverb table's priorities
            trainer.store.data_server_client.mutate_priorities(
                table="trainer_0",
                updates=dict(zip(*prio_updates_list)),
            )

            # Set the new variables
            # TODO (dries): key is probably not being store correctly.
            # The variable client might lose reference to it when checkpointing.
            # We also need to add the optimizer and random_key to the variable
            # server.
            trainer.store.base_key = new_states.random_key

            networks = trainer.store.networks
            target_networks = trainer.store.target_networks

            params = {
                net_key: networks[net_key].policy_params for net_key in networks.keys()
            }

            # TODO (sasha): this loop seems to have no effect?
            # Updating the networks:
            for net_key in params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.networks[net_key].policy_params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.params[net_key][param_key]

            # Update the optimizer
            for net_key in params.keys():
                # This needs to be in the loop to not lose the reference.
                # TODO (sasha): this could be an issue if not passing back new opt states as expected
                trainer.store.policy_opt_states[net_key][
                    constants.OPT_STATE_DICT_KEY
                ] = new_states.opt_states[
                    net_key
                ]  # [constants.OPT_STATE_DICT_KEY]

            # Update the target networks
            target_params = {
                net_key: target_networks[net_key].policy_params
                for net_key in target_networks.keys()
            }

            # TODO (sasha): this loop seems to have no effect?
            for net_key in target_params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.target_networks[net_key].policy_params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.target_params[net_key][param_key]

            # Set the metrics
            trainer.store.metrics = metrics

            trainer.store.trainer_counts["trainer_steps"] = new_states.steps

            return metrics

        trainer.store.step_fn = step

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "sgd_step"

    @staticmethod
    def config_class() -> Callable:
        """_summary_"""
        return MADQNStepConfig
