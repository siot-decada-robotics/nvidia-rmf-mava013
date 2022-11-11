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
from typing import List, Type

import jax
import jax.numpy as jnp
import optax
import rlax

from mava import constants
from mava.callbacks import Callback
from mava.components.training.base import QmixTrainingState
from mava.components.training.step import Step
from mava.core_jax import SystemTrainer
from mava.systems.vdn.components.training import VDNStep


class QmixStep(VDNStep):
    # ------------------- Step utility methods -------------------
    def get_params_from_store(self, trainer: SystemTrainer):
        policy_params, target_policy_params = super().get_params_from_store(trainer)

        mixer_params = trainer.store.mixing_net.hyper_params
        target_mixer_params = trainer.store.mixing_net.target_hyper_params

        policies = (policy_params, mixer_params)
        targets = (target_policy_params, target_mixer_params)

        return policies, targets

    def get_opt_states(self, trainer: SystemTrainer):
        return super().get_opt_states(trainer), trainer.store.mixer_opt_state

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
        super().update_store(new_states, trainer)  # updates policy

        # Mixer updates

        # TODO (sasha): I don't think this will update properly
        # trainer.store.mixing_net.hyper_params = new_states.mixer_params
        # trainer.store.mixing_net.target_hyper_params = new_states.mixer_params
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

        trainer.store.mixer_opt_state[
            constants.OPT_STATE_DICT_KEY
        ] = new_states.mixer_opt_state[constants.OPT_STATE_DICT_KEY]

    # ------------------- sgd_step utils -------------------
    def get_params_from_state(self, states: QmixTrainingState):
        return (
            (
                states.policy_params,
                states.mixer_params,
            ),
            (
                states.target_policy_params,
                states.target_mixer_params,
            ),
        )

    def extract_opt_state(self, states: QmixTrainingState):
        return super().extract_opt_state(states), states.mixer_opt_state

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
        # Separating out policy and mixer
        policy_gradients, mixer_gradients = grads

        q_policy_params, mixer_params = params
        q_target_policy_params, target_mixer_params = target_params

        policy_opt_states, mixer_opt_state = opt_states

        metrics = {}  # no metrics...yet

        # Policy update
        (
            q_policy_params,
            q_target_policy_params,
            policy_opt_states,
            policy_metrics,
        ) = super().update_policies(
            trainer,
            policy_gradients,
            q_policy_params,
            q_target_policy_params,
            policy_opt_states,
            trainer_iter,
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
            {**policy_metrics, **metrics},
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
