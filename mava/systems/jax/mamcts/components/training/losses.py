from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import rlax
from haiku._src.basic import merge_leading_dims

from mava.components.jax.training.base import Loss
from mava.core_jax import SystemTrainer
from mava.systems.jax.mamcts.learned_model_utils import (
    logits_to_scalar,
    scalar_to_two_hot,
    scale_gradient,
    value_transform,
)


@dataclass
class MAMCTSLossConfig:
    L2_regularisation_coeff: float = 0.0001
    value_cost: float = 1.0


class MAMCTSLoss(Loss):
    """MAMCTS Loss - essentially a decentralised AlphaZero loss"""

    def __init__(
        self,
        config: MAMCTSLossConfig = MAMCTSLossConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def loss_grad_fn(
            params: Any,
            observations: Any,
            search_policies: Dict[str, jnp.ndarray],
            target_values: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """TODO add description"""

            grads = {}
            loss_info = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks["networks"][agent_net_key]

                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.
                def loss_fn(
                    params: Any,
                    observations: Any,
                    search_policies: jnp.ndarray,
                    target_values: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

                    logits, value_logits = network.forward_fn(params, observations)

                    # Transform the target values into logits
                    target_values = value_transform(target_values)
                    target_values_logits = scalar_to_two_hot(
                        target_values, network._num_bins
                    )
                    target_values_logits = jax.lax.stop_gradient(target_values_logits)

                    # Compute the policy loss
                    policy_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            search_policies, logits
                        )
                    )

                    # Compute the value loss
                    value_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            target_values_logits, value_logits
                        )
                    )

                    # Entropy regulariser.
                    l2_regularisation = sum(
                        jnp.sum(jnp.square(parameter))
                        for parameter in jax.tree_leaves(params)
                    )

                    total_loss = (
                        policy_loss
                        + value_loss * self.config.value_cost
                        + l2_regularisation * self.config.L2_regularisation_coeff
                    )

                    loss_info = {
                        "loss_total": total_loss,
                        "loss_policy": policy_loss,
                        "loss_value": value_loss,
                        "loss_regularisation_term": l2_regularisation,
                    }

                    return total_loss, loss_info

                grads[agent_key], loss_info[agent_key] = jax.grad(
                    loss_fn, has_aux=True
                )(
                    params[agent_net_key],
                    observations[agent_key].observation,
                    search_policies[agent_key],
                    target_values[agent_key],
                )
            return grads, loss_info

        # Save the gradient funciton.
        trainer.store.grad_fn = loss_grad_fn

    @staticmethod
    def config_class() -> Callable:
        return MAMCTSLossConfig


@dataclass
class MAMCTSLearnedModelLossConfig(MAMCTSLossConfig):
    importance_sampling_exponent: float = 0.5


class MAMCTSLearnedModelLoss(Loss):
    """MAMCTS Loss - essentially a decentralised AlphaZero loss"""

    def __init__(
        self,
        config: MAMCTSLearnedModelLossConfig = MAMCTSLearnedModelLossConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def loss_grad_fn(
            params: Any,
            search_policies: Dict[str, jnp.ndarray],
            target_values: Dict[str, jnp.ndarray],
            rewards: Dict[str, jnp.ndarray],
            actions: Dict[str, jnp.ndarray],
            observation_history: Dict[str, jnp.ndarray],
            priorities: Any,
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """TODO add description"""

            grads = {}
            loss_info = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks["networks"][agent_net_key]

                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.
                def loss_fn(
                    params: Any,
                    search_policies: jnp.ndarray,
                    target_values: jnp.ndarray,
                    rewards: jnp.ndarray,
                    actions: jnp.ndarray,
                    observation_history: jnp.ndarray,
                    priorities: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

                    # Get batch initial root embeddings
                    initial_observation_history = observation_history[:, 0]

                    root_embeddings = network.representation_fn(
                        params["representation"], initial_observation_history
                    )

                    def dynamics_step(action, prev_state) -> Tuple[Any, Any]:
                        """Run one step of the RNN.
                        Args:
                        inputs: An arbitrarily nested structure.
                        prev_state: Previous core state.
                        Returns:
                        A tuple with two elements ``output, next_state``. ``output`` is an
                        arbitrarily nested structure. ``next_state`` is the next core state, this
                        must be the same shape as ``prev_state``."""

                        (
                            new_embedding,
                            reward_logits,
                        ) = network.dynamics_fn(params["dynamics"], prev_state, action)

                        new_embedding = scale_gradient(new_embedding, 0.5)
                        return reward_logits, new_embedding

                    # unroll and get the predicted reward logits and embeddings
                    predicted_rewards_logits, predicted_embeddings = hk.dynamic_unroll(
                        dynamics_step,
                        actions,
                        root_embeddings,
                        time_major=False,
                        return_all_states=True,
                    )

                    # Transform the rewards into logits
                    rewards = value_transform(rewards)
                    rewards_logits = scalar_to_two_hot(rewards, network._num_bins)

                    # Transform the target values into logits
                    target_values = value_transform(target_values)
                    target_values_logits = scalar_to_two_hot(
                        target_values, network._num_bins
                    )
                    target_values_logits = jax.lax.stop_gradient(target_values_logits)

                    # Add the initial root embedding to the sequence of generated embeddings
                    predicted_embeddings = jnp.concatenate(
                        [
                            jnp.expand_dims(root_embeddings, 1),
                            predicted_embeddings[:, 0:-1],
                        ],
                        axis=1,
                    )

                    # Get the policy and value logits for each of the generated embeddings
                    logits, value_logits = network.prediction_fn(
                        params["prediction"],
                        merge_leading_dims(predicted_embeddings, 2),
                    )

                    # Compute the policy loss
                    policy_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            merge_leading_dims(search_policies, 2), logits
                        ),
                        axis=-1,
                    )

                    # Compute the value loss
                    value_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            merge_leading_dims(target_values_logits, 2), value_logits
                        ),
                        axis=-1,
                    )

                    # Compute the reward loss
                    reward_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            merge_leading_dims(rewards_logits, 2),
                            merge_leading_dims(predicted_rewards_logits, 2),
                        ),
                        axis=-1,
                    )

                    # Entropy regulariser.
                    l2_regularisation = sum(
                        jnp.sum(jnp.square(parameter))
                        for parameter in jax.tree_leaves(params)
                    )

                    # Scale the gradients by 1/N where N is sequence length
                    sequence_length = actions.shape[-1]
                    policy_loss = scale_gradient(policy_loss, 1 / sequence_length)
                    value_loss = scale_gradient(value_loss, 1 / sequence_length)
                    reward_loss = scale_gradient(reward_loss, 1 / sequence_length)

                    batch_loss = (
                        policy_loss + self.config.value_cost * value_loss + reward_loss
                    )

                    # Importance weighting.
                    importance_weights = (1.0 / priorities).astype(jnp.float32)
                    importance_weights **= self.config.importance_sampling_exponent
                    importance_weights /= jnp.max(importance_weights)

                    total_loss = jnp.mean(importance_weights * batch_loss)

                    # Calculate new sequence priorities
                    batch_size = actions.shape[0]
                    predicted_value_scalar = logits_to_scalar(value_logits)
                    priorities = jnp.abs(
                        predicted_value_scalar - merge_leading_dims(target_values, 2)
                    ).reshape(batch_size, sequence_length)
                    priorities = jnp.squeeze(jnp.max(priorities, axis=-1))

                    loss_info = {
                        "loss_total": total_loss,
                        "loss_policy": policy_loss,
                        "loss_value": value_loss,
                        "loss_reward": reward_loss,
                        "loss_regularisation_term": l2_regularisation,
                        "priorities": priorities,
                    }

                    return total_loss, loss_info

                grads[agent_key], loss_info[agent_key] = jax.grad(
                    loss_fn, has_aux=True
                )(
                    params[agent_net_key],
                    search_policies[agent_key],
                    target_values[agent_key],
                    rewards[agent_key],
                    actions[agent_key],
                    observation_history[agent_key],
                    priorities,
                )
            return grads, loss_info

        # Save the gradient funciton.
        trainer.store.grad_fn = loss_grad_fn

    @staticmethod
    def config_class() -> Callable:
        return MAMCTSLearnedModelLossConfig

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "loss_fn"
