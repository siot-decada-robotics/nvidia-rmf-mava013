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

"""Trainer components for calculating losses."""
import haiku as hk
import jax
import jax.numpy as jnp

from mava.systems.vdn.components.training.loss import VDNLoss


class QmixLoss(VDNLoss):
    def get_policy_params(self, params, target_params):
        policy_params, _ = params
        target_policy_params, _ = target_params

        return policy_params, target_policy_params

    def get_mixer_params(self, params, target_params):
        _, mixer_params = params
        _, target_mixer_params = target_params

        return mixer_params, target_mixer_params

    def mix(self, trainer, env_states, q_tm1, q_t, mixer_params, target_mixer_params):
        mixer = trainer.store.mixing_net

        mixed_q_tm1 = mixer.forward(env_states[:, :-1], q_tm1, mixer_params)
        mixed_q_t = mixer.forward(env_states[:, 1:], q_t, target_mixer_params)

        return mixed_q_tm1, mixed_q_t

    def grad(
        self,
        loss_fn,
        params,
        target_params,
        policy_states,
        env_states,
        observations,
        actions,
        rewards,
        discounts,
        trainer,
    ):
        return jax.grad(loss_fn, has_aux=True)(
            params,
            target_params,
            policy_states,
            env_states,
            observations,
            actions,
            rewards,
            discounts,
            trainer,
        )
