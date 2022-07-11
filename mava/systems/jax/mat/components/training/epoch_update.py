from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax.random import KeyArray

from mava.components.jax.training import Batch
from mava.components.jax.training.model_updating import (
    EpochUpdate,
    MAPGEpochUpdateConfig,
)
from mava.core_jax import SystemTrainer


class MatEpochUpdate(EpochUpdate):
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
            carry: Tuple[KeyArray, Any, optax.OptState, Batch],
            unused_t: Tuple[()],
        ) -> Tuple[
            Tuple[KeyArray, Any, optax.OptState, Batch],
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
            # TODO (sasha): I shouldn't have to duplicate this class, the only thing that doesn't
            #  work is this assert
            assert (
                batch.observations.observation[:, :, 0].shape[0]
                == trainer.store.action_batch_size
            )

            permutation = jax.random.permutation(
                subkey, trainer.store.action_batch_size
            )
            # TODO: problem is here because of adv's different shape it is getting batched
            #  differently in minibatch. Check the shape of obs/actions b4 and after
            #  and then see how PPO does batching
            shuffled_batch = batch
            # shuffled_batch = jax.tree_map(
            #     lambda x: jnp.take(x, permutation, axis=0), batch
            # )
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
