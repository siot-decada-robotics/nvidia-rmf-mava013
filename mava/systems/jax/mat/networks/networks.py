import dataclasses
import functools
import math
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
from mava.systems.jax.mat.networks.transformer import Encoder, Decoder

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


@dataclasses.dataclass
class MatEncoderNetwork:
    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
    ):
        self.network = network
        self.params = params

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
        ):
            """
            Encoder forward function - performs attention over all agents observations and
            returns the attended state and value of each state
            """
            values, encoded_state = self.network.apply(params, observations)
            return values, encoded_state

        self.forward_fn = forward_fn


@dataclasses.dataclass
class MatDecoderNetwork:
    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
    ):
        self.network = network
        self.params = params

        @jit
        def forward_fn(
            params,
            encoded_observations: networks_lib.Observation,
            previous_actions: jnp.ndarray,
            agent_ind: int,
            key: networks_lib.PRNGKey,
            mask: chex.Array = None,
        ):
            """
            Encoder forward function - performs attention over all agents observations and
            returns the attended state and value of each state
            """
            action_dist = self.network.apply(
                params, previous_actions, encoded_observations
            )
            # decoder produces sequence of actions for all agents, so must select the action for
            # only the current agent
            action_dist._logits = action_dist.logits[:, agent_ind, :]

            if mask is not None:
                action_dist = action_mask_categorical_policies(action_dist, mask)

            action = jnp.squeeze(action_dist.sample(seed=key))
            log_prob = action_dist.log_prob(action)

            # TODO (sasha): might want to return action distrib to append to previous actions
            return action, log_prob

        self.forward_fn = forward_fn


@dataclasses.dataclass
class MatNetworks:
    """TODO: Add description here."""

    def __init__(self, encoder: MatEncoderNetwork, decoder: MatDecoderNetwork) -> None:
        """TODO: Add description here."""
        self.encoder = encoder
        self.decoder = decoder

        # TODO (sasha): custom trainer to work with dict of params (from mamu)
        self.params = {"encoder": encoder.params, "decoder": decoder.params}
        # mamu minibatch update step?
        # mamu sgd step
        # mamu loss fn

    def get_action(
        self,
        encoded_observations: networks_lib.Observation,
        previous_actions: jnp.ndarray,
        agent_ind: int,
        key: networks_lib.PRNGKey,
        mask: chex.Array = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Gets the actions for all agents in an auto-regressive manner"""

        action, logp = self.decoder.forward_fn(
            self.decoder.params,
            encoded_observations,
            previous_actions,
            agent_ind,
            key,
            mask,
        )

        return action, {"log_prob": jnp.squeeze(logp)}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        _, value = self.encoder.network.apply(self.params["encoder"], observations)
        return value

    def encode_observations(self, observations: networks_lib.Observation):
        """TODO: Add description here."""
        _, encoded_obs = self.encoder.forward_fn(self.params["encoder"], observations)
        return encoded_obs


def make_network(spec, n_agents, obs_net, rng_key):
    dummy_batch_size = 10
    action_len = spec.actions.num_values
    obs_shape = spec.observations.observation.shape

    dummy_obs = jnp.zeros((dummy_batch_size, n_agents, *obs_shape))
    # +1 to action dim because it of initial token to indicate start of step
    dummy_prev_actions = jnp.zeros((dummy_batch_size, n_agents, action_len + 1))

    enc = lambda x: Encoder(1, 1)(obs_net(x))
    enc_forward = hk.without_apply_rng(hk.transform(enc))
    enc_params = enc_forward.init(rng_key, dummy_obs)
    v, dummy_encoded_obs = enc_forward.apply(enc_params, dummy_obs)

    # TODO (sasha): n blocks and n heads as params
    dec = lambda act, obs: Decoder(1, 1, action_len)(act, obs)
    dec_forward = hk.without_apply_rng(hk.transform(dec))
    dec_params = dec_forward.init(rng_key, dummy_prev_actions, dummy_encoded_obs)

    return MatNetworks(
        encoder=MatEncoderNetwork(network=enc_forward, params=enc_params),
        decoder=MatDecoderNetwork(network=dec_forward, params=dec_params),
    )


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    rng_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    obs_net=functools.partial(utils.batch_concat, num_batch_dims=2),
):
    specs = environment_spec.get_agent_specs()
    n_agents = len(specs)

    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks = {
        net_key: make_network(
            specs[net_key],
            n_agents,
            obs_net,
            rng_key=rng_key,
        )
        for net_key in specs.keys()
    }

    return {
        "networks": networks,
    }
