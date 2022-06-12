from functools import partial
from typing import Sequence, Tuple

import acme.jax.utils as utils
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from rlax._src.transforms import (
    signed_hyperbolic,
    signed_parabolic,
    transform_from_2hot,
    transform_to_2hot,
)


def scale_gradient(g: chex.Array, scale: float) -> chex.Array:
    """Scale the gradient of an associated value.

    Args:
        g : Any array or value that has a gradient associated with it.
        scale : scalar value to scale the gradient by.

    Returns:
        Array: same value as g but with gradient scaled.
    """
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)


def scalar_to_two_hot(scalar: chex.Array, num_bins: int) -> chex.Array:
    """Convert a scalar to a two_hot representation i.e. a vector.

    Args:
        scalar : the scalar value.
        num_bins : the size of the vector.

    Returns:
        Array: the vector representing the scalar value.
    """
    min_value = ((num_bins - 1) // 2) * -1
    max_value = min_value * -1
    return transform_to_2hot(scalar, min_value, max_value, num_bins)


def logits_to_scalar(logits: chex.Array) -> chex.Array:
    """Convert a vector to a scalar. I.e. inverse of scalar_to_two_hot.

    Args:
        logits : the vector to be converted into a scalar.

    Returns:
        Array: a scalar value.
    """
    num_bins = logits.shape[-1]
    min_value = ((num_bins - 1) // 2) * -1
    max_value = min_value * -1
    probs = jax.nn.softmax(logits, axis=-1)
    return transform_from_2hot(probs, min_value, max_value, num_bins)


def value_transform(x: chex.Array, epsilon: float = 1e-3) -> chex.Array:
    """A non-linear value transformation for variance reduction. Ref: https://arxiv.org/abs/1805.11593.

    Args:
        x : the array to transform.
        epsilon : value used in transform. Defaults to 1e-3.

    Returns:
        chex.Array: transformed array.
    """
    return signed_hyperbolic(x, epsilon)


def inv_value_transform(x: chex.Array, epsilon: float = 1e-3):
    """The inverse of the non-linear value transformation.

    Args:
        x : the array to transform.
        epsilon : value used in transform. Defaults to 1e-3.

    Returns:
        chex.Array: transformed array.
    """
    return signed_parabolic(x, epsilon)


def normalise_encoded_state(
    encoded_state: chex.Array, epsilon: float = 1e-5
) -> chex.Array:
    """Normalise an encoded state using the last dimension.

    Args:
        encoded_state : The encoded state to normalise.
        epsilon : minimum value allowed in encoded state before epsilon is added.

    Returns:
        normalised encoded state.
    """
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


@partial(jax.jit, static_argnames=["tile_shape", "normalise"])
def actions_to_tiles(
    action_array: chex.Array,
    tile_shape: Sequence[int],
    num_actions: int,
    normalise: bool = False,
    shift_actions_by: int = 20,
) -> chex.Array:
    """Converts an array of actions into planes and normalises them.

    Args:
        action_array : Array of actions.
        tile_shape : shape of plane to put actions into
        num_actions : number of actions. used to normalise between 0 and 1.

    Returns:
        returns the tiled actions in the shape [tile,batch]
    """
    num_actions = jax.lax.cond(
        normalise, lambda: num_actions + shift_actions_by, lambda: 1
    )

    tiled_actions = (
        jax.vmap(lambda x: jnp.full(tile_shape, x), out_axes=(-1))(action_array)
        + shift_actions_by / num_actions
    )

    return tiled_actions


@jax.jit
def join_non_flattened_observation_action_history(
    stacked_observation_history: chex.Array,
    stacked_action_history: chex.Array,
    num_actions: int,
) -> chex.Array:
    """Process and concatenate a history of observations and actions into the shape [observation shape, history size*2].
    Actions are tiled and stacked on top of the observation history in the last dimension.

    Args:
        stacked_observation_history : A stack of observations in the shape [history, observation shape]
        stacked_action_history : A stack of actions in the shape [history,]
        num_actions : the number of available actions. Used for normalisation.

    Returns:
        Concatenated observation and action history. In the shape [observation shape, history size*2].
    """
    stacked_observation_history = jnp.transpose(
        stacked_observation_history,
        axes=(*range(1, len(stacked_observation_history.shape)), 0),
    )  # expects no batch dimension

    stacked_action_history = actions_to_tiles(
        stacked_action_history, stacked_observation_history.shape[:-1], num_actions
    )

    full_history = jnp.concatenate(
        (stacked_observation_history, stacked_action_history), axis=-1
    )

    return full_history


@partial(jax.jit, static_argnames=["history_size"])
def pad_history(
    stacked_observation_history: chex.Array,
    stacked_action_history: chex.Array,
    history_size: int,
) -> Tuple[chex.Array, chex.Array]:
    """Add zero padding to observation and action history

    Args:
        stacked_observation_history : The current observation history.
        stacked_action_history : The current action history.
        history_size : The desired history size.

    Returns:
        A padded observation history and a padded action history.
    """

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
