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
import abc
from functools import partial
import time
from dataclasses import dataclass
from distutils.command.config import config
from typing import Any, Dict, List, Tuple, Type

import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import tree
from acme.jax import utils
from jax import jit

import mava.components.building.adders  # To avoid circular imports
import mava.components.training.model_updating  # To avoid circular imports
from mava import constants
from mava.callbacks import Callback
from mava.components.training.advantage_estimation import GAE
from mava.components.training.base import Batch, QmixTrainingState, TrainingState
from mava.components.training.step import Step
from mava.core_jax import SystemTrainer
from mava.utils.jax_training_utils import denormalize, normalize


@dataclass
class IRDQNStepConfig:
    target_update_period: int = 100


class QmixStep(Step):
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
        states: QmixTrainingState,
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
        mixer_params = trainer.store.mixing_net.hyper_params
        target_mixer_params = trainer.store.mixing_net.target_hyper_params

        policies = (policy_params, mixer_params)
        targets = (target_policy_params, target_mixer_params)

        return policies, targets

    def get_opt_states(self, trainer: SystemTrainer):
        return trainer.store.policy_opt_states, trainer.store.mixer_opt_state

    def make_training_state(
        self, policy_params, target_params, opt_states, random_key, trainer_iteration
    ):
        q_policy_params, mixer_params = policy_params
        q_target_policy_params, target_mixer_params = target_params
        policy_opt_states, mixer_opt_state = opt_states

        return QmixTrainingState(
            policy_params=q_policy_params,
            target_policy_params=q_target_policy_params,
            mixer_params=mixer_params,
            target_mixer_params=target_mixer_params,
            policy_opt_states=policy_opt_states,
            mixer_opt_state=mixer_opt_state,
            random_key=random_key,
            trainer_iteration=trainer_iteration,
        )

    def update_store(self, new_states: QmixTrainingState, trainer: SystemTrainer):
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

        mixer_param_names = [
            "hyper_w1_params",
            "hyper_w2_params",
            "hyper_b1_params",
            "hyper_b2_params",
        ]
        for param_name in mixer_param_names:
            params = trainer.store.mixing_net.hyper_params[param_name]
            for param_key in params.keys():
                params[param_key] = new_states.mixer_params[param_name][param_key]

            target_params = trainer.store.mixing_net.target_hyper_params[param_name]
            for param_key in params.keys():
                target_params[param_key] = new_states.target_mixer_params[param_name][
                    param_key
                ]

        # TODO (sasha): I don't think this will update properly
        # trainer.store.mixing_net.hyper_params = new_states.mixer_params
        # trainer.store.mixing_net.target_hyper_params = new_states.mixer_params

        trainer.store.mixer_opt_state[
            constants.OPT_STATE_DICT_KEY
        ] = new_states.mixer_opt_state[constants.OPT_STATE_DICT_KEY]

    # ------------------- sgd_step utils -------------------
    def get_data(self, sample: reverb.ReplaySample):
        # Extract the data.
        data = sample.data

        return (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.extras,
        )

    def get_params_from_state(self, states: QmixTrainingState):
        return (states.policy_params, states.mixer_params,), (
            states.target_policy_params,
            states.target_mixer_params,
        )

    def extract_opt_state(self, states: QmixTrainingState):
        return states.policy_opt_states, states.mixer_opt_state

    def grad(
        self,
        grad_fn,
        policy_params,
        target_policy_params,
        observations,
        actions,
        rewards,
        discounts,
        extras,
    ):
        q_policy_params, mixer_params = policy_params
        q_target_policy_params, target_mixer_params = target_policy_params
        return grad_fn(
            q_policy_params,
            mixer_params,
            q_target_policy_params,
            target_mixer_params,
            extras["policy_states"],
            extras["s_t"],
            observations,
            actions,
            rewards,
            discounts,
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
        # TODO (sasha): this should possibly return metrics also?
        policy_gradients, mixer_gradients = grads

        q_policy_params, mixer_params = params
        q_target_policy_params, target_mixer_params = target_params

        policy_opt_states, mixer_opt_state = opt_states

        metrics = {}
        for agent_net_key in trainer.store.trainer_agent_net_keys.values():
            # Update the policy networks and optimisers.
            (
                policy_updates,
                policy_opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],
            ) = trainer.store.policy_optimiser.update(
                policy_gradients[agent_net_key],
                policy_opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],
            )
            q_policy_params[agent_net_key] = optax.apply_updates(
                q_policy_params[agent_net_key], policy_updates
            )

        # Mixer update
        (
            mixer_updates,
            mixer_opt_state[constants.OPT_STATE_DICT_KEY],
        ) = trainer.store.mixer_optimiser.update(
            mixer_gradients,
            mixer_opt_state[constants.OPT_STATE_DICT_KEY],
        )
        mixer_params = optax.apply_updates(mixer_params, mixer_updates)

        # update target q net
        q_target_policy_params = rlax.periodic_update(
            q_policy_params,
            q_target_policy_params,
            trainer_iter,
            self.config.target_update_period,
        )

        # update target mixer net
        target_mixer_params = rlax.periodic_update(
            mixer_params,
            target_mixer_params,
            trainer_iter,
            self.config.target_update_period,
        )

        return (
            (q_policy_params, mixer_params),
            (
                q_target_policy_params,
                target_mixer_params,
            ),
            (policy_opt_states, mixer_opt_state),
            metrics,
        )

    def format_metrics(self, grad_metrics, update_metrics):
        return jax.tree_map(jnp.mean, {**grad_metrics, **update_metrics})

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        Returns:
            List of required component classes.
        """
        return Step.required_components()  # TODO
