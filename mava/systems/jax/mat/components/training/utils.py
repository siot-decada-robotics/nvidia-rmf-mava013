from typing import NamedTuple, Any

import jax.numpy as jnp


def merge_agents_to_sequence(a):
    """merges the agents and the sequence dimension (dim 1 and 2)"""
    # TODO (sasha): explain why order is F
    return jnp.reshape(
        a, (a.shape[0], a.shape[1] * a.shape[2], *a.shape[3:]), order="F"
    )


def unmerge_agents_to_sequence(a, num_agents):
    """unmerges the agents and the sequence dimension"""
    return jnp.reshape(a, (a.shape[0], -1, num_agents, *a.shape[2:]), order="F")


class MatBatch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    observations: Any
    actions: Any
    advantages: Any

    # Target value estimate used to bootstrap the value function.
    target_values: Any

    # Value estimate and action log-prob at behavior time.
    behavior_values: Any
    behavior_log_probs: Any

    terminals: Any
