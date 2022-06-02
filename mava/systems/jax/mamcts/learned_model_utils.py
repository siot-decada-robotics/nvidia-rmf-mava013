from typing import Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import acme.jax.utils as utils
from rlax._src.transforms import (
    signed_hyperbolic,
    signed_parabolic,
    transform_from_2hot,
    transform_to_2hot,
)


def scale_gradient(g: chex.Array, scale: float):
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)


def scalar_to_two_hot(scalar: chex.Array, num_bins: int):
    min_value = ((num_bins - 1) // 2) * -1
    max_value = min_value * -1
    return transform_to_2hot(scalar, min_value, max_value, num_bins)


def logits_to_scalar(logits: chex.Array):
    """The inverse of the scalar_to_two_hot function above."""
    num_bins = logits.shape[-1]
    min_value = ((num_bins - 1) // 2) * -1
    max_value = min_value * -1
    probs = jax.nn.softmax(logits, axis=-1)
    return transform_from_2hot(probs, min_value, max_value, num_bins)


def value_transform(x: chex.Array, epsilon: float = 1e-3):
    """A non-linear value transformation for variance reduction. Ref: https://arxiv.org/abs/1805.11593."""
    return signed_hyperbolic(x, epsilon)


def inv_value_transform(x: chex.Array, epsilon: float = 1e-3):
    """The inverse of the non-linear value transformation above."""
    return signed_parabolic(x, epsilon)


def normalise_encoded_state(encoded_state: chex.Array, epsilon: float = 1e-5):
    min_encoded_state = jnp.min(encoded_state, axis=-1, keepdims=True)
    max_encoded_state = jnp.max(encoded_state, axis=-1, keepdims=True)
    scale_encoded_state = max_encoded_state - min_encoded_state
    scale_encoded_state = jnp.where(
        scale_encoded_state < epsilon,
        scale_encoded_state + epsilon,
        scale_encoded_state,
    )
    encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
    return encoded_state_normalized


def actions_to_tiles(
    action_array: chex.Array, tile_shape: Sequence[int], num_actions: int
):

    tiled_actions = jnp.ones((action_array.shape[0], *tile_shape, 1))
    tiled_actions = action_array[:, None, None, None] * tiled_actions / num_actions

    return tiled_actions  # Batch x tile x 1


def join_flattened_observation_action_history(
    stacked_observation_history: chex.Array,
    stacked_action_history: chex.Array,
    num_actions: int,
):  
    """Process and concatenate observation and action history for single dimension observations"""
    
    stacked_observation_history = utils.batch_concat(stacked_observation_history)
    
   
    stacked_action_history = hk.one_hot(stacked_action_history, num_actions)

    full_history = jnp.concatenate(
        (
            stacked_observation_history.reshape(1, -1),
            stacked_action_history.reshape(1, -1),
        ),
        axis=-1,
    )

    return full_history


def join_non_flattened_observation_action_history(
    stacked_observation_history: chex.Array,
    stacked_action_history: chex.Array,
    num_actions: int,
):
    """Process and concatenate observation and action history for non-single dimension observations"""
    stacked_observation_history = jnp.transpose(
        stacked_observation_history,
        axes=(*range(1, len(stacked_observation_history.shape)), 0),
    )  # expects no batch dimension
    stacked_action_history = actions_to_tiles(
        stacked_action_history, stacked_observation_history.shape[:-1], num_actions
    )
    stacked_action_history = jnp.squeeze(stacked_action_history, axis=-1)
    stacked_action_history = jnp.transpose(
        stacked_action_history, axes=(*range(1, len(stacked_action_history.shape)), 0)
    )

    full_history = jnp.concatenate(
        (stacked_observation_history, stacked_action_history), axis=-1
    )

    return full_history


def pad_history(
    stacked_observation_history: chex.Array,
    stacked_action_history: chex.Array,
    history_size: int,
):
    """Add zero padding to observation and action history"""

    padded_hist_size = history_size - stacked_observation_history.shape[0]

    padded_obs = jnp.zeros(
        (1, *stacked_observation_history.shape[1:]), stacked_observation_history.dtype
    )

    padded_obs = jnp.repeat(padded_obs, padded_hist_size, axis=0)

    stacked_observation_history = jnp.concatenate(
        [padded_obs, stacked_observation_history], axis=0
    )

    padded_actions = jnp.zeros(
        (1, *stacked_action_history.shape[1:]),
        stacked_action_history.dtype,
    )

    padded_actions = jnp.repeat(padded_actions, padded_hist_size, axis=0)

    stacked_action_history = jnp.concatenate(
        [padded_actions, stacked_action_history], axis=0
    )

    return stacked_observation_history, stacked_action_history
