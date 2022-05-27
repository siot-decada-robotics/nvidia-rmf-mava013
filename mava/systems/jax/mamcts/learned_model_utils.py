from typing import Sequence

import chex
import jax
import jax.numpy as jnp


def scale_gradient(g, scale: float):
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)


def scalar_to_two_hot(x: chex.Array, num_bins: int):
    """A categorical representation of real values. Ref: https://www.nature.com/articles/s41586-020-03051-4.pdf."""
    max_val = (num_bins - 1) // 2
    x = jnp.clip(x, -max_val, max_val)
    x_low = jnp.floor(x).astype(jnp.int32)
    x_high = jnp.ceil(x).astype(jnp.int32)
    p_high = x - x_low
    p_low = 1.0 - p_high
    idx_low = x_low + max_val
    idx_high = x_high + max_val
    cat_low = jax.nn.one_hot(idx_low, num_bins) * p_low[..., None]
    cat_high = jax.nn.one_hot(idx_high, num_bins) * p_high[..., None]
    return cat_low + cat_high


def logits_to_scalar(logits: chex.Array):
    """The inverse of the scalar_to_two_hot function above."""
    num_bins = logits.shape[-1]
    max_val = (num_bins - 1) // 2
    x = jnp.sum((jnp.arange(num_bins) - max_val) * jax.nn.softmax(logits), axis=-1)
    return x


def value_transform(x: chex.Array, epsilon: float = 1e-3):
    """A non-linear value transformation for variance reduction. Ref: https://arxiv.org/abs/1805.11593."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + epsilon * x


def inv_value_transform(x: chex.Array, epsilon: float = 1e-3):
    """The inverse of the non-linear value transformation above."""
    return jnp.sign(x) * (
        ((jnp.sqrt(1 + 4 * epsilon * (jnp.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon))
        ** 2
        - 1
    )


def action_to_tile(
    action_array: chex.Array, tile_shape: Sequence[int], num_actions: int
):
    action_tile_plane = jnp.ones((*tile_shape, 1), float)
    tiled_action_one_hot = action_array * action_tile_plane / num_actions
    return tiled_action_one_hot
