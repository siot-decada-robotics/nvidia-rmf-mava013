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

import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import rlax
from acme.agents.jax.dqn import learning_lib
from jax import jit
from jax.config import config

from mava.components.training.base import Loss
from mava.core_jax import SystemTrainer


@dataclass
class MADQNLossConfig:
    max_abs_reward: float = 1.0
    gamma: float = 0.99
    importance_sampling_exponent: float = 0.6


class MADQNLoss(Loss):
    """Deep q learning.

    This matches the original DQN loss: https://arxiv.org/abs/1312.5602.
    It differs by two aspects that improve it on the optimization side
    - it uses a square loss instead of the Huber one.
    """

    def __init__(
        self,
        config: MADQNLossConfig = MADQNLossConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Creates the grad function of the loss and adds it to the trainer.store."""

        @chex.assert_max_traces(n=1)
        def loss_grad_fn(
            trainer_network: Any,  # TODO (sasha): similar to PPO we should access networks from the store
            params: Any,
            target_params: Any,
            observations: Any,
            next_observations: Any,
            actions: Dict[str, jnp.ndarray],
            discounts: Any,
            rewards: Any,
            probs: Any,
            keys: Any,
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Gradient of loss.

            Args:
                params: parameters of all models.
                target_params: parameters of all target model.
                observations: observations of all the agents.
                next_observations: next observations of all the agents.
                actions: actions of all the agents.
                discounts: discounts of all the agents.
                rewards: rewards of all the agents.

            Returns:
                grads: gradients of loss with respect to all the parameters.
                extra: extra information.
            """
            grads = {}
            loss = {}
            loss_info = {}

            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer_network[agent_net_key]

                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.
                def loss_fn(
                    params: Any,
                    target_params: Any,
                    observations: Any,
                    next_observations: Any,
                    actions: jnp.ndarray,
                    discount: Any,
                    rewards: Any,
                    next_legal_actions: Any,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    # Forward pass.
                    _, logits_tm1, atoms_tm1 = network.forward_fn(params, observations)
                    _, logits_t, atoms_t = network.forward_fn(
                        target_params, next_observations
                    )
                    q_t_selector, _, _ = network.forward_fn(params, next_observations)

                    # Masking illegal actions with min float, not sure if this has any effect?
                    # q_t_selector = jnp.where(
                    #    next_legal_actions.astype(bool),
                    #    q_t_selector,
                    #    jnp.finfo(q_t_selector.dtype).min,
                    # )

                    d_t = (discount * self.config.gamma).astype(jnp.float32)
                    # Cast and clip rewards.
                    r_t = jnp.clip(
                        rewards,
                        -self.config.max_abs_reward,
                        self.config.max_abs_reward,
                    ).astype(jnp.float32)

                    # Compute categorical double Q-learning loss.
                    batch_loss_fn = jax.vmap(
                        rlax.categorical_double_q_learning,
                        in_axes=(None, 0, 0, 0, 0, None, 0, 0),
                    )
                    batch_error = batch_loss_fn(
                        atoms_tm1,
                        logits_tm1,
                        actions,
                        r_t,
                        d_t,
                        atoms_t,
                        logits_t,
                        q_t_selector,
                    )

                    batch_loss = rlax.l2_loss(batch_error)
                    # batch_loss = rlax.huber_loss(td_error)

                    # What are probs: looks like they are priorities currently in replay buff
                    importance_weights = (1.0 / probs).astype(jnp.float32)
                    importance_weights **= self.config.importance_sampling_exponent
                    importance_weights /= jnp.max(importance_weights)

                    # Weigthing loss by probability transition was chosen
                    loss = jnp.mean(importance_weights * batch_loss)
                    reverb_update = learning_lib.ReverbUpdate(
                        keys=keys,
                        priorities=jnp.abs(batch_error).astype(jnp.float64),
                    )
                    loss_info = {"loss_total": loss, "reverb_updates": reverb_update}
                    return loss, loss_info

                (loss[agent_key], loss_info[agent_key]), grads[
                    agent_key
                ] = jax.value_and_grad(loss_fn, has_aux=True)(
                    params[agent_net_key],
                    target_params[agent_net_key],
                    observations[agent_key].observation,
                    next_observations[agent_key].observation,
                    actions[agent_key],
                    discounts[agent_key],
                    rewards[agent_key],
                    next_observations[agent_key].legal_actions,
                )

                # TODO (sasha): is this not already in loss info?
                if agent_key == "agent_0":
                    loss_info["joint"] = [loss_info[agent_key]["reverb_updates"]]
                else:
                    loss_info["joint"].append(loss_info[agent_key]["reverb_updates"])
                loss_info["total_loss"] = loss[agent_key]

            return grads, loss_info

        # Save the gradient function.
        trainer.store.grad_fn = jax.jit(
            functools.partial(loss_grad_fn, trainer.store.networks)
        )

    @staticmethod
    def config_class() -> Callable:
        """Returns the config class for this loss."""
        return MADQNLossConfig

    @staticmethod
    def name() -> str:
        """Returns name of the component."""
        return "loss"
