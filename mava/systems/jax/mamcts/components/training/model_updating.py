from dataclasses import dataclass
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from acme.jax import networks as networks_lib
from jax.random import KeyArray

from mava.components.jax.training import Utility
from mava.components.jax.training.model_updating import (
    EpochUpdate,
    MAPGEpochUpdateConfig,
    MAPGMinibatchUpdateConfig,
    MinibatchUpdate,
)
from mava.core_jax import SystemTrainer


class MCTSBatch(NamedTuple):
    """A batch of MAMCTS data; all shapes are expected to be [B, ...]."""

    observations: Any
    search_policies: Any
    target_values: Any


class MAMUBatch(NamedTuple):
    """A batch of MAMCTS data; all shapes are expected to be [B, ...]."""

    search_policies: Any
    target_values: Any
    rewards: Any
    actions: Any
    observation_history: Any
    priorities: Any


class MAMUEpochUpdate(EpochUpdate):
    def __init__(
        self,
        config: MAPGEpochUpdateConfig = MAPGEpochUpdateConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        trainer.store.num_epochs = self.config.num_epochs
        trainer.store.num_minibatches = self.config.num_minibatches

        def model_update_epoch(
            carry: Tuple[KeyArray, Any, optax.OptState, MAMUBatch],
            unused_t: Tuple[()],
        ) -> Tuple[
            Tuple[KeyArray, Any, optax.OptState, MAMUBatch],
            Dict[str, jnp.ndarray],
        ]:

            """Performs model updates based on one epoch of data."""
            key, params, opt_states, batch = carry

            new_key, subkey = jax.random.split(key)

            # TODO (dries): This assert is ugly. Is there a better way to do this check?
            # Maybe using a tree map of some sort?
            # shapes = jax.tree_map(
            #         lambda x: x.shape[0]==trainer.store.full_batch_size, batch
            #     )
            # assert ...

            assert (
                list(batch.rewards.values())[0].shape[0]
                == trainer.store.full_batch_size
            )

            permutation = jax.random.permutation(subkey, trainer.store.full_batch_size)

            shuffled_batch = jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_map(
                lambda x: jnp.reshape(
                    x, [self.config.num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            (new_params, new_opt_states), metrics = jax.lax.scan(
                trainer.store.minibatch_update_fn,
                (params, opt_states),
                minibatches,
                length=self.config.num_minibatches,
            )

            return (new_key, new_params, new_opt_states, batch), metrics

        trainer.store.epoch_update_fn = model_update_epoch

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPGEpochUpdateConfig


@dataclass
class MAMCTSMinibatchUpdateConfig(MAPGMinibatchUpdateConfig):
    pass


class MAMCTSMinibatchUpdate(MinibatchUpdate):
    def __init__(
        self,
        config: MAMCTSMinibatchUpdateConfig = MAMCTSMinibatchUpdateConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        if not self.config.optimizer:
            trainer.store.optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.max_gradient_norm),
                optax.scale_by_adam(eps=self.config.adam_epsilon),
                optax.scale(-self.config.learning_rate),
            )
        else:
            trainer.store.optimizer = self.config.optimizer

        # Initialize optimizers.
        trainer.store.opt_states = {}
        for net_key in trainer.store.networks["networks"].keys():
            trainer.store.opt_states[net_key] = trainer.store.optimizer.init(
                trainer.store.networks["networks"][net_key].params
            )  # pytype: disable=attribute-error

        def model_update_minibatch(
            carry: Tuple[networks_lib.Params, optax.OptState], minibatch: MCTSBatch
        ) -> Tuple[Tuple[Any, optax.OptState], Dict[str, Any]]:
            """Performs model update for a single minibatch."""
            params, opt_states = carry

            # Calculate the gradients and agent metrics.
            gradients, agent_metrics = trainer.store.grad_fn(
                params,
                minibatch.observations,
                minibatch.search_policies,
                minibatch.target_values,
            )

            # Update the networks and optimizors.
            metrics = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                # Apply updates
                # TODO (dries): Use one optimizer per network type here and not
                # just one.
                updates, opt_states[agent_net_key] = trainer.store.optimizer.update(
                    gradients[agent_key], opt_states[agent_net_key]
                )
                params[agent_net_key] = optax.apply_updates(
                    params[agent_net_key], updates
                )

                agent_metrics[agent_key]["norm_grad"] = optax.global_norm(
                    gradients[agent_key]
                )
                agent_metrics[agent_key]["norm_updates"] = optax.global_norm(updates)
                metrics[agent_key] = agent_metrics
            return (params, opt_states), metrics

        trainer.store.minibatch_update_fn = model_update_minibatch

    @staticmethod
    def config_class() -> Callable:
        return MAMCTSMinibatchUpdateConfig


class MAMUMinibatchUpdate(MAMCTSMinibatchUpdate):
    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        if not self.config.optimizer:
            trainer.store.optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.max_gradient_norm),
                optax.scale_by_adam(eps=self.config.adam_epsilon),
                optax.scale(-self.config.learning_rate),
            )
        else:
            trainer.store.optimizer = self.config.optimizer

        # Initialize optimizers.
        trainer.store.opt_states = {}
        for net_key in trainer.store.networks["networks"].keys():
            trainer.store.opt_states[net_key] = trainer.store.optimizer.init(
                trainer.store.networks["networks"][net_key].params
            )  # pytype: disable=attribute-error

        def model_update_minibatch(
            carry: Tuple[networks_lib.Params, optax.OptState],
            minibatch: MAMUBatch,
        ) -> Tuple[Tuple[Any, optax.OptState], Dict[str, Any]]:
            """Performs model update for a single minibatch."""
            params, opt_states = carry

            # Calculate the gradients and agent metrics.
            gradients, agent_metrics = trainer.store.grad_fn(
                params,
                minibatch.search_policies,
                minibatch.target_values,
                minibatch.rewards,
                minibatch.actions,
                minibatch.observation_history,
                minibatch.priorities,
            )

            # Update the networks and optimizors.
            metrics = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                # Apply updates
                # TODO (dries): Use one optimizer per network type here and not
                # just one.
                updates, opt_states[agent_net_key] = trainer.store.optimizer.update(
                    gradients[agent_key], opt_states[agent_net_key]
                )
                params[agent_net_key] = optax.apply_updates(
                    params[agent_net_key], updates
                )

                agent_metrics[agent_key]["norm_grad"] = optax.global_norm(
                    gradients[agent_key]
                )
                agent_metrics[agent_key]["norm_updates"] = optax.global_norm(updates)
                metrics[agent_key] = agent_metrics
            return (params, opt_states), metrics

        trainer.store.minibatch_update_fn = model_update_minibatch
