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
import abc
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Type

import haiku as hk
import jax
import jax.numpy as jnp
import rlax

from mava.callbacks import Callback
from mava.components import Component, training
from mava.components.training.losses import Loss
from mava.core_jax import SystemTrainer
from mava.systems.idqn.components.training import IDQNLossConfig
from haiku._src.basic import merge_leading_dims


class QmixLoss(Loss):
    def __init__(
        self,
        config: IDQNLossConfig = IDQNLossConfig(),
    ):
        """Component defines a MAPGWithTrustRegionClipping loss function.

        Args:
            config: MAPGTrustRegionClippingLossConfig.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Create and store MAPGWithTrustRegionClippingLoss loss function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def policy_loss_grad_fn(
            policy_params: Any,
            hyper_params: Any,
            target_policy_params: Any,
            target_hyper_params: Any,
            policy_states: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            rewards: Dict[str, jnp.ndarray],
            discounts: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios.

            Args:
                policy_params: policy network parameters.
                observations: agent observations.
                actions: actions the agents took.
                behaviour_log_probs: Log probabilities of actions taken by
                    current policy in the environment.
                advantages: advantage estimation values per agent.

            Returns:
                Tuple[policy gradients, policy loss information]
            """
            policy_grads = {}

            def policy_loss_fn(
                params: Any,
                target_params: Any,
                env_states: Any,
                all_policy_states: Any,
                all_observations: Any,
                all_actions: jnp.ndarray,
                all_rewards: jnp.ndarray,
                all_discounts: jnp.ndarray,
            ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                """Inner policy loss function: see outer function for parameters."""
                policy_params, hyper_params = params
                target_policy_params, target_hyper_params = target_params
                num_agents = len(actions)
                b, t = list(actions.values())[0].shape[:2]

                all_q_tm1 = jnp.zeros((b, t, num_agents, 1), dtype=jnp.float32)
                all_q_t = jnp.zeros_like(all_q_tm1)
                rewards = jnp.zeros_like(all_q_tm1)
                discounts = jnp.zeros_like(all_q_tm1)

                for i, agent_key in enumerate(trainer.store.trainer_agents):
                    agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                    network = trainer.store.networks[agent_net_key]

                    observations = all_observations[agent_key].observation
                    masks = all_observations[agent_key].legal_actions
                    policy_states = all_policy_states[agent_key]

                    # Use the state at the start of the sequence and unroll the policy.
                    policy_net_core = lambda x, y: network.forward(
                        policy_params, [x, y]
                    )
                    target_policy_net_core = lambda x, y: network.forward(
                        target_policy_params, [x, y]
                    )

                    online_qs, _ = hk.static_unroll(
                        policy_net_core,
                        observations,
                        policy_states[:, 0],
                        time_major=False,
                    )

                    target_qs, _ = hk.static_unroll(
                        target_policy_net_core,
                        observations,
                        policy_states[:, 0],
                        time_major=False,
                    )

                    q_tm1 = online_qs[:, :-1]
                    q_t_selector = online_qs[:, 1:]
                    q_t_value = target_qs[:, 1:]

                    q_t_selector = jnp.where(
                        masks[:, 1:] == 1.0, q_t_selector, -99999  # TODO
                    )

                    num_actions = q_t_value.shape[-1]
                    next_actions = jnp.argmax(q_t_selector, keepdims=True)
                    one_hot_next_actions = jax.nn.one_hot(next_actions, num_actions)
                    q_t = jnp.sum(
                        q_t_value * one_hot_next_actions, axis=-1, keepdims=True
                    )

                    actions = all_actions[agent_key]
                    one_hot_actions = jax.nn.one_hot(actions, num_actions)
                    q_tm1 = jnp.sum(q_tm1 * one_hot_actions, axis=-1, keepdims=True)

                    all_q_tm1.at[:, :, i].set(q_tm1)
                    all_q_t.at[:, :, i].set(q_t)

                    rewards.at[:, :, i].set(all_rewards[agent_key])
                    discounts.at[:, :, i].set(all_discounts[agent_key])

                rewards = jnp.sum(rewards, axis=2, keepdims=True)
                discounts = jnp.sum(discounts, axis=2, keepdims=True)

                mixed_q_tm1 = network.mixing(hyper_params, env_states, q_tm1)
                mixed_q_t = network.mixing(target_hyper_params, env_states, q_t)

                target = jax.lax.stop_gradient(
                    rewards + discounts * self.config.gamma * mixed_q_t
                )
                error = 0.5 * (mixed_q_tm1 - target) ** 2
                loss = jnp.mean(error)
                # TODO grad for mixing net?

                return loss, {}  # loss_info_policy

            policy_grads, loss_info_policy = jax.grad(policy_loss_fn, has_aux=True)(
                (policy_params, hyper_params),
                (target_policy_params, target_hyper_params),
                policy_states,
                observations,
                actions,
                rewards,
                discounts,
            )

            return policy_grads, loss_info_policy

        # Save the gradient funcitons.
        trainer.store.policy_grad_fn = policy_loss_grad_fn
