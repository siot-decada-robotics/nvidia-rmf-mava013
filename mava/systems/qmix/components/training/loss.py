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
from haiku._src.basic import merge_leading_dims

from mava.callbacks import Callback
from mava.components import Component, training
from mava.components.training.losses import Loss
from mava.core_jax import SystemTrainer
from mava.systems.idqn.components.training.loss import IDQNLossConfig


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
            mixer_params: Any,
            target_policy_params: Any,
            target_mixer_params: Any,
            policy_states: Any,
            env_states: Any,
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

            def policy_loss_fn(
                params: Any,
                target_params: Any,
                all_policy_states: Any,
                env_states: Any,
                all_observations: Any,
                all_actions: jnp.ndarray,
                all_rewards: jnp.ndarray,
                all_discounts: jnp.ndarray,
            ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                """Inner policy loss function: see outer function for parameters."""
                # unpack parameters
                policy_params, mixer_params = params
                target_policy_params, target_mixer_params = target_params

                num_agents = len(all_actions)
                b, t = list(all_actions.values())[0].shape[:2]

                q_tm1 = jnp.zeros((b, t - 1, num_agents, 1), dtype=jnp.float32)
                q_t = jnp.zeros_like(q_tm1)
                rewards = jnp.zeros_like(q_tm1)
                discounts = jnp.zeros_like(q_tm1)

                mixer = trainer.store.mixing_net

                for i, agent_key in enumerate(trainer.store.trainer_agents):
                    agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                    network = trainer.store.networks[agent_net_key]

                    observations = all_observations[agent_key].observation
                    masks = all_observations[agent_key].legal_actions
                    policy_states = all_policy_states[agent_key]

                    # Use the state at the start of the sequence and unroll the policy.
                    policy_net_core = lambda obs, states: network.forward(
                        policy_params[agent_net_key], obs, states
                    )
                    target_policy_net_core = lambda obs, states: network.forward(
                        target_policy_params[agent_net_key], obs, states
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

                    agent_q_tm1 = online_qs[:, :-1]
                    q_t_selector = online_qs[:, 1:]
                    q_t_value = target_qs[:, 1:]

                    q_t_selector = jnp.where(
                        masks[:, 1:] == 1.0, q_t_selector, -99999  # TODO
                    )

                    num_actions = q_t_value.shape[-1]
                    next_actions = jnp.argmax(q_t_selector, axis=-1)
                    one_hot_next_actions = jax.nn.one_hot(next_actions, num_actions)
                    agent_q_t = jnp.sum(
                        q_t_value * one_hot_next_actions, axis=-1, keepdims=True
                    )

                    actions = all_actions[agent_key][:, :-1]
                    one_hot_actions = jax.nn.one_hot(actions, num_actions)
                    agent_q_tm1 = jnp.sum(
                        agent_q_tm1 * one_hot_actions, axis=-1, keepdims=True
                    )

                    q_tm1 = q_tm1.at[:, :, i].set(agent_q_tm1)
                    q_t = q_t.at[:, :, i].set(agent_q_t)

                    agent_reward = jnp.expand_dims(
                        all_rewards[agent_key][:, :-1], axis=-1
                    )
                    agent_discount = jnp.expand_dims(
                        all_discounts[agent_key][:, :-1], axis=-1
                    )
                    rewards = rewards.at[:, :, i].set(agent_reward)
                    discounts = discounts.at[:, :, i].set(agent_discount)

                rewards = jnp.mean(rewards, axis=2)
                # discounts should be zero or one
                discounts = jnp.mean(discounts, axis=2)

                mixed_q_tm1 = mixer.forward(env_states[:, :-1], q_tm1, mixer_params)
                mixed_q_t = mixer.forward(env_states[:, 1:], q_t, target_mixer_params)

                # mixed_q_tm1 = jnp.sum(q_tm1, axis=2)
                # mixed_q_t = jnp.sum(q_t, axis=2)

                target = jax.lax.stop_gradient(
                    rewards + discounts * self.config.gamma * mixed_q_t
                )
                error = 0.5 * (mixed_q_tm1 - target) ** 2
                loss = jnp.mean(error)

                return loss, {"total_loss": loss}

            policy_grads, loss_info_policy = jax.grad(policy_loss_fn, has_aux=True)(
                (policy_params, mixer_params),
                (target_policy_params, target_mixer_params),
                policy_states,
                env_states,
                observations,
                actions,
                rewards,
                discounts,
            )

            return policy_grads, loss_info_policy

        # Save the gradient funcitons.
        trainer.store.policy_grad_fn = policy_loss_grad_fn
