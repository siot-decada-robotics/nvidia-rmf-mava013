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
from functools import partial
from typing import Dict, List, Tuple, Type

import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
from jax import jit

from mava import constants
from mava.callbacks import Callback
from mava.components.training.base import DQNTrainingState, TrainingState
from mava.components.training.step import Step
from mava.core_jax import SystemTrainer
from mava.systems.idrqn.components.training.step import IRDQNStepConfig


class VDNStep(Step):
    def __init__(
        self,
        config: IRDQNStepConfig = IRDQNStepConfig(),
    ):
        """Component defines the MAPGWithTrustRegion SGD step.

        Args:
            config: MAPGWithTrustRegionStepConfig.
        """
        self.config = config

    def on_building_init_end(self, builder) -> None:
        super().on_building_init_end(builder)
        builder.store.trainer_iter = 0

    # flake8: noqa: C901
    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """Define and store the SGD step function for MAPGWithTrustRegion.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """
        trainer.store.step_fn = partial(self.step, trainer=trainer)

    @partial(jit, static_argnames=["self", "trainer"])
    def sgd_step(
        self,
        states: DQNTrainingState,
        sample: reverb.ReplaySample,
        trainer: SystemTrainer,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
        """Performs a minibatch SGD step.

        Args:
            states: Training states (network params and optimiser states).
            sample: Reverb sample.
            trainer: SystemTrainer.

        Returns:
            Tuple[new state, metrics].
        """
        observations, actions, rewards, discounts, extras = self.get_data(sample)

        params, target_params = self.get_params_from_state(states)
        opt_states = self.extract_opt_state(states)

        grads, grad_metrics = self.grad(
            trainer,
            trainer.store.policy_grad_fn,
            params,
            target_params,
            observations,
            actions,
            rewards,
            discounts,
            extras,
        )

        params, target_params, opt_states, update_metrics = self.update_policies(
            trainer, grads, params, target_params, opt_states, states.trainer_iteration
        )

        metrics = self.format_metrics(grad_metrics, update_metrics)

        new_states = self.make_training_state(
            params,
            target_params,
            opt_states,
            states.random_key,
            states.trainer_iteration,
        )
        return new_states, metrics

    def step(
        self, sample: reverb.ReplaySample, trainer: SystemTrainer
    ) -> Tuple[Dict[str, jnp.ndarray]]:
        """Step over the reverb sample and update the parameters / optimiser states.

        Args:
            sample: Reverb sample.

        Returns:
            Metrics from SGD step.
        """

        params, target_params = self.get_params_from_store(trainer)

        opt_states = self.get_opt_states(trainer)

        _, random_key = jax.random.split(trainer.store.base_key)
        trainer.store.trainer_iter += 1
        states = self.make_training_state(
            params, target_params, opt_states, random_key, trainer.store.trainer_iter
        )

        new_states, metrics = self.sgd_step(states, sample, trainer)

        self.update_store(new_states, trainer)

        return metrics

    # ------------------- Step utility methods -------------------
    def get_params_from_store(self, trainer: SystemTrainer):
        networks = trainer.store.networks

        policy_params = {
            net_key: networks[net_key].policy_params for net_key in networks.keys()
        }
        target_policy_params = {
            net_key: networks[net_key].target_policy_params
            for net_key in networks.keys()
        }

        return policy_params, target_policy_params

    def get_opt_states(self, trainer: SystemTrainer):
        return trainer.store.policy_opt_states

    def make_training_state(
        self, policy_params, target_params, opt_states, random_key, trainer_iteration
    ):
        return DQNTrainingState(
            policy_params=policy_params,
            target_policy_params=target_params,
            policy_opt_states=opt_states,
            random_key=random_key,
            trainer_iteration=trainer_iteration,
        )

    def update_store(self, new_states: DQNTrainingState, trainer: SystemTrainer):
        trainer.store.base_key = new_states.random_key

        for net_key in trainer.store.networks.keys():
            # UPDATING THE PARAMETERS IN THE NETWORK IN THE STORE

            # The for loop below is needed to not lose the param reference.
            net_params = trainer.store.networks[net_key].policy_params
            for param_key in net_params.keys():
                net_params[param_key] = new_states.policy_params[net_key][param_key]

            target_net_params = trainer.store.networks[net_key].target_policy_params
            for param_key in target_net_params.keys():
                target_net_params[param_key] = new_states.target_policy_params[net_key][
                    param_key
                ]

            # Update the policy optimiser
            # The opt_states need to be wrapped in a dict so as not to lose
            # the reference.
            trainer.store.policy_opt_states[net_key][
                constants.OPT_STATE_DICT_KEY
            ] = new_states.policy_opt_states[net_key][constants.OPT_STATE_DICT_KEY]

    # ------------------- sgd_step utils -------------------
    def get_data(self, sample: reverb.ReplaySample):
        # Extract the data.
        data = sample.data

        return (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.extras,  # could be None, as we don't use extras
        )

    def get_params_from_state(self, states: DQNTrainingState):
        return states.policy_params, states.target_policy_params

    def extract_opt_state(self, states: DQNTrainingState):
        return states.policy_opt_states

    def grad(
        self,
        trainer,
        grad_fn,
        policy_params,
        target_policy_params,
        observations,
        actions,
        rewards,
        discounts,
        extras,
    ):
        return grad_fn(
            trainer=trainer,
            params=policy_params,
            target_params=target_policy_params,
            policy_states=extras["policy_states"],
            env_states=None,
            observations=observations,
            actions=actions,
            rewards=rewards,
            discounts=discounts,
        )

    def update_policies(
        self,
        trainer: SystemTrainer,
        grads,
        params,
        target_params,
        opt_states,
        trainer_iter,
    ):
        metrics = {}
        for agent_net_key in trainer.store.trainer_agent_net_keys.values():
            # Update the policy networks and optimisers.
            (
                policy_updates,
                opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],
            ) = trainer.store.policy_optimiser.update(
                grads[agent_net_key],
                opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],
            )
            params[agent_net_key] = optax.apply_updates(
                params[agent_net_key], policy_updates
            )

        # update target q net
        target_params = optax.periodic_update(
            params,
            target_params,
            trainer_iter,
            self.config.target_update_period,
        )

        return params, target_params, opt_states, metrics

    def format_metrics(self, grad_metrics, update_metrics):
        return jax.tree_map(jnp.mean, {**grad_metrics, **update_metrics})

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        Returns:
            List of required component classes.
        """
        return Step.required_components()  # TODO
