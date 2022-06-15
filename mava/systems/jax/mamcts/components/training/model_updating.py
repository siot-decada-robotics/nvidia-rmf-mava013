from dataclasses import dataclass
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from acme.jax import networks as networks_lib
from jax.random import KeyArray

from mava.components.jax.component import Component
from mava.components.jax.training import Utility
from mava.components.jax.training.model_updating import (
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
    """A batch of MAMU data; all shapes are expected to be [B, ...]."""

    search_policies: Any
    target_values: Any
    rewards: Any
    actions: Any
    observation_history: Any
    priorities: Any


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


@dataclass
class MAMUUpdateConfig(MAPGMinibatchUpdateConfig):
    pass


class MAMUUpdate(Component):
    def __init__(
        self,
        config: MAMUUpdateConfig = MAMUUpdateConfig(),
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

        def model_update(
            carry: Tuple[KeyArray, Any, optax.OptState, MAMUBatch]
        ) -> Tuple[
            Tuple[KeyArray, Any, optax.OptState, MAMUBatch],
            Dict[str, jnp.ndarray],
        ]:

            """Performs model updates based on one sample of data."""
            key, params, opt_states, batch = carry

            new_key, subkey = jax.random.split(key)

            # Calculate the gradients and agent metrics.
            gradients, agent_metrics = trainer.store.grad_fn(
                params,
                batch.search_policies,
                batch.target_values,
                batch.rewards,
                batch.actions,
                batch.observation_history,
                batch.priorities,
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

            return (new_key, params, opt_states, batch), metrics

        trainer.store.update_fn = model_update

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "model_update"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAMUUpdateConfig
