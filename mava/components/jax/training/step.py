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

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import tree
from acme.jax import utils
from chex import Array, Scalar
from haiku._src.basic import merge_leading_dims
from jax import jit
from mava.adders.base import DEFAULT_PRIORITY_TABLE

from mava.components.jax import Component
from mava.components.jax.training import Batch, Step, TrainingState
from mava.components.jax.training.base import MCTSBatch, MCTSLearnedModelBatch
from mava.core_jax import SystemTrainer
from mava.systems.jax.mamcts.learned_model_utils import (
    inv_value_transform,
    logits_to_scalar,
)


@dataclass
class DefaultStepConfig:
    random_key: int = 42


class DefaultStep(Component):
    def __init__(
        self,
        config: DefaultStepConfig = DefaultStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_step(self, trainer: SystemTrainer) -> None:
        """Does a step of SGD and logs the results."""

        # Do a batch of SGD.
        sample = next(trainer.store.dataset_iterator)
        results = trainer.store.step_fn(sample)

        # Update our counts and record it.
        # counts = self._counter.increment(steps=1) # TODO: add back in later

        # TODO (dries): Confirm that this is the correctly place to put the
        # variable client code.
        timestamp = time.time()
        elapsed_time = (
            timestamp - trainer.store.timestamp
            if hasattr(trainer.store, "timestamp")
            else 0
        )
        trainer.store.timestamp = timestamp

        trainer.store.trainer_parameter_client.add_async(
            {"trainer_steps": 1, "trainer_walltime": elapsed_time},
        )

        # Update the variable source and the trainer.
        trainer.store.trainer_parameter_client.set_and_get_async()

        # Add the trainer counts.
        results.update(trainer.store.trainer_counts)

        # Write to the loggers.
        trainer.store.trainer_logger.write({**results})

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "step"


@dataclass
class MAPGWithTrustRegionStepConfig:
    discount: float = 0.99


class MAPGWithTrustRegionStep(Step):
    def __init__(
        self,
        config: MAPGWithTrustRegionStepConfig = MAPGWithTrustRegionStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_init_start(self, trainer: SystemTrainer) -> None:
        # Note (dries): Assuming the batch and sequence dimensions are flattened.
        trainer.store.full_batch_size = trainer.store.sample_batch_size * (
            trainer.store.sequence_length - 1
        )

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        @jit
        def sgd_step(
            states: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""

            # Extract the data.
            data = sample.data

            observations, actions, rewards, termination, extra = (
                data.observations,
                data.actions,
                data.rewards,
                data.discounts,
                data.extras,
            )

            discounts = tree.map_structure(
                lambda x: x * self.config.discount, termination
            )

            behavior_log_probs = extra["policy_info"]

            networks = trainer.store.networks["networks"]

            def get_behavior_values(
                net_key: Any, reward: Any, observation: Any
            ) -> jnp.ndarray:
                o = jax.tree_map(
                    lambda x: jnp.reshape(x, [-1] + list(x.shape[2:])), observation
                )
                _, behavior_values = networks[net_key].network.apply(
                    states.params[net_key], o
                )
                behavior_values = jnp.reshape(behavior_values, reward.shape[0:2])
                return behavior_values

            agent_nets = trainer.store.trainer_agent_net_keys
            behavior_values = {
                key: get_behavior_values(
                    agent_nets[key], rewards[key], observations[key].observation
                )
                for key in agent_nets.keys()
            }

            # Vmap over batch dimension
            batch_gae_advantages = jax.vmap(trainer.store.gae_fn, in_axes=0)

            advantages = {}
            target_values = {}
            for key in rewards.keys():
                advantages[key], target_values[key] = batch_gae_advantages(
                    rewards[key], discounts[key], behavior_values[key]
                )

            # Exclude the last step - it was only used for bootstrapping.
            # The shape is [num_sequences, num_steps, ..]
            observations, actions, behavior_log_probs, behavior_values = jax.tree_map(
                lambda x: x[:, :-1],
                (observations, actions, behavior_log_probs, behavior_values),
            )

            trajectories = Batch(
                observations=observations,
                actions=actions,
                advantages=advantages,
                behavior_log_probs=behavior_log_probs,
                target_values=target_values,
                behavior_values=behavior_values,
            )

            # Concatenate all trajectories. Reshape from [num_sequences, num_steps,..]
            # to [num_sequences * num_steps,..]
            agent_0_t_vals = list(target_values.values())[0]
            assert len(agent_0_t_vals) > 1
            num_sequences = agent_0_t_vals.shape[0]
            num_steps = agent_0_t_vals.shape[1]
            batch_size = num_sequences * num_steps
            assert batch_size % trainer.store.num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, trainer.store.num_minibatches)
            batch = jax.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            (new_key, new_params, new_opt_states, _,), metrics = jax.lax.scan(
                trainer.store.epoch_update_fn,
                (states.random_key, states.params, states.opt_states, batch),
                (),
                length=trainer.store.num_epochs,
            )

            # Set the metrics
            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["norm_params"] = optax.global_norm(states.params)
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
            metrics["rewards_mean"] = jax.tree_map(
                lambda x: jnp.mean(jnp.abs(jnp.mean(x, axis=(0, 1)))), rewards
            )
            metrics["rewards_std"] = jax.tree_map(
                lambda x: jnp.std(x, axis=(0, 1)), rewards
            )

            new_states = TrainingState(
                params=new_params, opt_states=new_opt_states, random_key=new_key
            )
            return new_states, metrics

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            networks = trainer.store.networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            opt_states = trainer.store.opt_states
            random_key, _ = jax.random.split(trainer.store.key)

            states = TrainingState(
                params=params, opt_states=opt_states, random_key=random_key
            )

            new_states, metrics = sgd_step(states, sample)

            # Set the new variables
            # TODO (dries): key is probably not being store correctly.
            # The variable client might lose reference to it when checkpointing.
            # We also need to add the optimizer and random_key to the variable
            # server.
            trainer.store.key = new_states.random_key

            networks = trainer.store.networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            for net_key in params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.networks["networks"][net_key].params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.params[net_key][param_key]

                # Update the optimizer
                # This needs to be in the loop to not lose the reference.
                trainer.store.opt_states[net_key] = new_states.opt_states[net_key]

            return metrics

        trainer.store.step_fn = step

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "step_fn"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPGWithTrustRegionStepConfig


@dataclass
class MAMCTSStepConfig(MAPGWithTrustRegionStepConfig):
    pass


class MAMCTSStep(Step):
    def __init__(
        self,
        config: MAMCTSStepConfig = MAMCTSStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_init_start(self, trainer: SystemTrainer) -> None:
        # Note (dries): Assuming the batch and sequence dimensions are flattened.
        trainer.store.full_batch_size = trainer.store.sample_batch_size * (
            trainer.store.sequence_length
        )

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        @jit
        def sgd_step(
            states: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""

            # Extract the data.
            data = sample.data

            observations, actions, rewards, termination, extra = (
                data.observations,
                data.actions,
                data.rewards,
                data.discounts,
                data.extras,
            )

            discounts = tree.map_structure(
                lambda x: x * self.config.discount, termination
            )

            search_policies = {}
            for agent_key in extra["policy_info"].keys():
                search_policies[agent_key] = extra["policy_info"][agent_key][
                    "search_policies"
                ]

            networks = trainer.store.networks["networks"]

            def get_bootstrap_values(
                net_key: Any, reward: Any, observation: Any
            ) -> jnp.ndarray:
                merged_obs = jax.tree_map(
                    lambda x: merge_leading_dims(x, 2), observation
                )

                _, bootstrap_values = networks[net_key].forward_fn(
                    states.params[net_key], merged_obs
                )
                bootstrap_values = logits_to_scalar(bootstrap_values)
                bootstrap_values = inv_value_transform(bootstrap_values)
                bootstrap_values = jnp.reshape(bootstrap_values, reward.shape[0:2])
                return bootstrap_values

            agent_nets = trainer.store.trainer_agent_net_keys
            bootstrap_values = {
                key: get_bootstrap_values(
                    agent_nets[key], rewards[key], observations[key].observation
                )
                for key in agent_nets.keys()
            }

            # Vmap over batch dimension
            batch_n_step_returns = jax.vmap(
                trainer.store.n_step_fn, in_axes=(0, 0, 0, None, None)
            )

            # TODO shift as is done in the old way
            zeros = jnp.zeros_like(list(bootstrap_values.values())[0])
            # Shift the bootstrapping values up by one
            bootstrap_values = jax.tree_map(
                lambda x: jnp.concatenate(
                    [x[:, 1:], jnp.expand_dims(zeros[:, -1], -1)], -1
                ),
                bootstrap_values,
            )

            target_values = {}
            for key in rewards.keys():
                target_values[key] = batch_n_step_returns(
                    rewards[key],
                    discounts[key],
                    bootstrap_values[key],
                    self.config.n_step,
                    self.config.lambda_t,
                )

            trajectories = MCTSBatch(
                observations=observations,
                search_policies=search_policies,
                target_values=target_values,
            )

            # Concatenate all trajectories. Reshape from [num_sequences, num_steps,..]
            # to [num_sequences * num_steps,..]
            agent_0_t_vals = list(target_values.values())[0]
            assert len(agent_0_t_vals) > 1
            num_sequences = agent_0_t_vals.shape[0]
            num_steps = agent_0_t_vals.shape[1]
            batch_size = num_sequences * num_steps
            assert batch_size % trainer.store.num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, trainer.store.num_minibatches)

            batch = jax.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            (new_key, new_params, new_opt_states, _,), metrics = jax.lax.scan(
                trainer.store.epoch_update_fn,
                (states.random_key, states.params, states.opt_states, batch),
                (),
                length=trainer.store.num_epochs,
            )

            # Set the metrics
            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["norm_params"] = optax.global_norm(states.params)
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
            metrics["rewards_mean"] = jax.tree_map(
                lambda x: jnp.mean(jnp.abs(jnp.mean(x, axis=(0, 1)))), rewards
            )
            metrics["rewards_std"] = jax.tree_map(
                lambda x: jnp.std(x, axis=(0, 1)), rewards
            )

            new_states = TrainingState(
                params=new_params, opt_states=new_opt_states, random_key=new_key
            )
            return new_states, metrics

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            networks = trainer.store.networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            opt_states = trainer.store.opt_states
            random_key, _ = jax.random.split(trainer.store.key)

            states = TrainingState(
                params=params, opt_states=opt_states, random_key=random_key
            )
            new_states, metrics = sgd_step(states, sample)

            # Set the new variables
            # TODO (dries): key is probably not being store correctly.
            # The variable client might lose reference to it when checkpointing.
            # We also need to add the optimizer and random_key to the variable
            # server.
            trainer.store.key = new_states.random_key

            networks = trainer.store.networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            for net_key in params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.networks["networks"][net_key].params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.params[net_key][param_key]

                # Update the optimizer
                # This needs to be in the loop to not lose the reference.
                trainer.store.opt_states[net_key] = new_states.opt_states[net_key]

            return metrics

        trainer.store.step_fn = step

    @staticmethod
    def config_class() -> Callable:
        return MAMCTSStepConfig


@dataclass
class MAMCTSLearnedModelStepConfig(MAMCTSStepConfig):
    unroll_steps: int = 5


class MAMCTSLearnedModelStep(Step):
    def __init__(
        self,
        config: MAMCTSLearnedModelStepConfig = MAMCTSLearnedModelStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_init_start(self, trainer: SystemTrainer) -> None:
        # Note (dries): Assuming the batch and sequence dimensions are flattened.
        trainer.store.full_batch_size = trainer.store.sample_batch_size

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        @jit
        def sgd_step(
            states: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""

            # Extract the data.
            data = sample.data

            keys, sequence_priorities, *_ = sample.info

            

            observations, actions, rewards, termination, extra = (
                data.observations,
                data.actions,
                data.rewards,
                data.discounts,
                data.extras,
            )

            discounts = tree.map_structure(
                lambda x: x * self.config.discount, termination
            )

            search_policies = {}
            search_values = {}
            observation_history = {}
            for agent_key in extra["policy_info"].keys():
                search_policies[agent_key] = extra["policy_info"][agent_key][
                    "search_policies"
                ]
                search_values[agent_key] = extra["policy_info"][agent_key][
                    "search_values"
                ]
                observation_history[agent_key] = extra["policy_info"][agent_key][
                    "observation_history"
                ]

            agent_nets = trainer.store.trainer_agent_net_keys
            networks = trainer.store.networks["networks"]

            def get_predicted_values(
                net_key: Any, reward: Any, observation_history: Any
            ) -> jnp.ndarray:
                merged_obs_hist = jax.tree_map(
                    lambda x: merge_leading_dims(x, 2), observation_history
                )
                merged_root_embeddings = networks[net_key].representation_fn(
                    states.params[net_key]["representation"], merged_obs_hist
                )

                _, predicted_value_logits = networks[net_key].prediction_fn(
                    states.params[net_key]["prediction"], merged_root_embeddings
                )
                predicted_values = logits_to_scalar(predicted_value_logits)
                predicted_values = inv_value_transform(predicted_values)
                predicted_values = jnp.reshape(predicted_values, reward.shape[0:2])
                return predicted_values

            predicted_values = {
                key: get_predicted_values(
                    agent_nets[key], rewards[key], observation_history[key]
                )
                for key in agent_nets.keys()
            }

            bootstrap_values = {key: search_values[key] for key in agent_nets.keys()}

            # Vmap over batch dimension
            batch_n_step_returns = jax.vmap(
                trainer.store.n_step_fn, in_axes=(0, 0, 0, None, None)
            )

            # Shift the bootstrapping values up by one
            bootstrap_values = jax.tree_map(
                lambda x: x[:, 1:],
                bootstrap_values,
            )

            (
                observations,
                search_policies,
                rewards,
                actions,
                observation_history,
                discounts,
                predicted_values,
            ) = jax.tree_map(
                lambda x: x[:, :-1],
                (
                    observations,
                    search_policies,
                    rewards,
                    actions,
                    observation_history,
                    discounts,
                    predicted_values,
                ),
            )

            target_values = {}
            for key in rewards.keys():
                target_values[key] = batch_n_step_returns(
                    rewards[key],
                    discounts[key],
                    bootstrap_values[key],
                    self.config.n_step,
                    self.config.lambda_t,
                )

            state_priorities = jax.tree_map(
                lambda pred_val, target_val: jnp.abs(pred_val - target_val),
                predicted_values,
                target_values,
            )

            def get_start_indices(rng_key, termination, probabilities):
                end_of_sequence = (
                    jnp.squeeze(
                        jnp.argwhere(
                            termination == 0, size=1, fill_value=termination.shape[-1]
                        )
                    )
                    - self.config.unroll_steps
                )
                end_of_sequence = jax.lax.cond(
                    end_of_sequence < 0, lambda: jnp.int32(0), lambda: end_of_sequence
                )

                probabilities = probabilities * jnp.concatenate(
                    (jnp.array([1.0]), termination[:-1])
                )

                # Sample a state according to priorities
                sampled_state_index = jnp.squeeze(
                    jax.random.categorical(rng_key, probabilities, axis=-1, shape=())
                )
                

                return sampled_state_index

            def cut_trajectory(field, start_index):
                start_other_dims = [0] * len(field.shape[1:])
                end_other_dims = field.shape[1:]
                return jax.lax.dynamic_slice(
                    field,
                    (start_index, *start_other_dims),
                    (self.config.unroll_steps, *end_other_dims),
                )

            batch_size = list(rewards.values())[0].shape[0]

            new_key, *rng_keys = jax.random.split(states.random_key, batch_size + 1)
            index_keys = jnp.array(rng_keys)
            indices = jax.tree_map(
                lambda term, probs: jax.vmap(get_start_indices, in_axes=(0, 0, 0))(
                    index_keys, term, probs
                ),
                discounts,
                state_priorities,
            )

            search_policies = jax.tree_map(
                lambda field, index: jax.vmap(cut_trajectory, in_axes=(0, 0))(
                    field, index
                ),
                search_policies,
                indices,
            )
            rewards = jax.tree_map(
                lambda field, index: jax.vmap(cut_trajectory, in_axes=(0, 0))(
                    field, index
                ),
                rewards,
                indices,
            )
            actions = jax.tree_map(
                lambda field, index: jax.vmap(cut_trajectory, in_axes=(0, 0))(
                    field, index
                ),
                actions,
                indices,
            )
            observation_history = jax.tree_map(
                lambda field, index: jax.vmap(cut_trajectory, in_axes=(0, 0))(
                    field, index
                ),
                observation_history,
                indices,
            )
            discounts = jax.tree_map(
                lambda field, index: jax.vmap(cut_trajectory, in_axes=(0, 0))(
                    field, index
                ),
                discounts,
                indices,
            )
            target_values = jax.tree_map(
                lambda field, index: jax.vmap(cut_trajectory, in_axes=(0, 0))(
                    field, index
                ),
                target_values,
                indices,
            )

            trajectories = MCTSLearnedModelBatch(
                search_policies=search_policies,
                target_values=target_values,
                rewards=rewards,
                actions=actions,
                observation_history=observation_history,
                priorities=sequence_priorities,
            )

            # Concatenate all trajectories. Reshape from [num_sequences, num_steps,..]
            # to [num_sequences * num_steps,..]
            agent_0_t_vals = list(target_values.values())[0]
            assert len(agent_0_t_vals) >= 1
            num_sequences = agent_0_t_vals.shape[0]
            batch_size = num_sequences
            assert batch_size % trainer.store.num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, trainer.store.num_minibatches)

            (new_key, new_params, new_opt_states, _,), metrics = jax.lax.scan(
                trainer.store.epoch_update_fn,
                (new_key, states.params, states.opt_states, trajectories),
                (),
                length=trainer.store.num_epochs,
            )

           
            priorities = None
            for agent_key in metrics:
                if priorities is None:
                    priorities = metrics[agent_key][agent_key].pop("priorities")
                else:
                    priorities += metrics[agent_key][agent_key].pop("priorities")

            # Set the metrics
            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["norm_params"] = optax.global_norm(states.params)
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
            metrics["rewards_mean"] = jax.tree_map(
                lambda x: jnp.mean(jnp.abs(jnp.mean(x, axis=(0, 1)))), rewards
            )
            metrics["rewards_std"] = jax.tree_map(
                lambda x: jnp.std(x, axis=(0, 1)), rewards
            )

            new_states = TrainingState(
                params=new_params, opt_states=new_opt_states, random_key=new_key
            )
            return new_states, metrics, keys, priorities

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            networks = trainer.store.networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            opt_states = trainer.store.opt_states
            random_key, _ = jax.random.split(trainer.store.key)

            states = TrainingState(
                params=params, opt_states=opt_states, random_key=random_key
            )
            new_states, metrics, keys, priorities = sgd_step(states, sample)
   
            keys, priorities = tree.map_structure(utils.fetch_devicearray,(keys, jnp.squeeze(priorities)))
            priority_updates = dict(zip(keys, priorities))
            
            
            for table_key in trainer.store.table_network_config.keys():
                trainer.store.data_server_client.mutate_priorities(table=table_key,updates=priority_updates)
            
            # Set the new variables
            # TODO (dries): key is probably not being store correctly.
            # The variable client might lose reference to it when checkpointing.
            # We also need to add the optimizer and random_key to the variable
            # server.
            trainer.store.key = new_states.random_key

            networks = trainer.store.networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            for net_key in params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.networks["networks"][net_key].params

                for model_param in net_params.keys():
                    for param_key in net_params[model_param].keys():
                        net_params[model_param][param_key] = new_states.params[net_key][
                            model_param
                        ][param_key]

                # Update the optimizer
                # This needs to be in the loop to not lose the reference.
                trainer.store.opt_states[net_key] = new_states.opt_states[net_key]
                trainer.store.networks["networks"][net_key].update_inner_params()

            return metrics

        trainer.store.step_fn = step

    @staticmethod
    def config_class() -> Callable:
        return MAMCTSLearnedModelStepConfig

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "step_fn"
