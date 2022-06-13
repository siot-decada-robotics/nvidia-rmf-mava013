import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import chex
import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from dm_env import specs as dm_specs
from jax import jit

from mava import specs as mava_specs
from mava.utils.jax_training_utils import action_mask_categorical_policies

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


@dataclasses.dataclass
class MatNetworks:
    """TODO: Add description here."""

    def __init__(
        self,
        encoder,
        decoder,
        encoder_params: networks_lib.Params,
        decoder_params: networks_lib.Params,
        log_prob: Optional[networks_lib.LogProbFn] = None,
        entropy: Optional[EntropyFn] = None,
        sample: Optional[networks_lib.SampleFn] = None,
    ) -> None:
        """TODO: Add description here."""
        self.encoder = encoder
        self.decoder = decoder

        self.encoder_params = encoder_params
        self.decoder_params = decoder_params

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
            key: networks_lib.PRNGKey,
            mask: chex.Array = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """TODO: Add description here."""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            distribution, _ = self.network.apply(params, observations)
            if mask is not None:
                distribution = action_mask_categorical_policies(distribution, mask)

            actions = jax.numpy.squeeze(distribution.sample(seed=key))
            log_prob = distribution.log_prob(actions)

            return actions, log_prob

        self.forward_fn = forward_fn

    def get_actions(
        self,
        observations: networks_lib.Observation,
        key: networks_lib.PRNGKey,
        mask: chex.Array = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Gets the actions for all agents in an auto-regressive manner"""
        batch_size, num_agents, action_dim = observations.shape

        # setting up initial token for no actions
        shifted_action = jnp.zeros((batch_size, num_agents, action_dim + 1))
        shifted_action[:, 0, 0] = 1

        output_action = jnp.zeros((batch_size, num_agents, 1))
        output_logp = jnp.zeros_like(output_action)

        obs_rep, _ = self.encoder.apply(self.encoder_params, observations)

        for i in range(len(observations)):
            dist = self.decoder.apply(
                self.decoder_params, (observations, shifted_action)
            )
            if mask is not None:
                dist = action_mask_categorical_policies(dist, mask)

            action = dist.sample(seed=key)
            logp = dist.log_prob(action)

            output_action[:, i, :] = action
            output_logp[:, i, :] = logp
            if i + 1 < num_agents:
                # TODO (sasha): should we feed this in or the raw distribution?
                shifted_action[:, i + 1, 1:] = jax.nn.one_hot(
                    action, num_classes=action_dim
                )

        return output_action, {"log_prob": output_logp}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        _, value = self.encoder.apply(self.encoder_params, observations)
        return value


class Decoder(hk.Module):
    def __init__(self, n_block, n_emb, n_head, n_agent):
        super().__init__(name="decoder")
        # TODO (sasha): initialize as done in repo
        hk.Linear(n_emb, with_bias=False)
