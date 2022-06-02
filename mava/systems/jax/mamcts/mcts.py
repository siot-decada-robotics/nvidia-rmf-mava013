import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import acme.jax.utils as utils
import chex
import jax
import jax.numpy as jnp
import mctx
import numpy as np
from haiku import Params
from jax import jit

from mava.utils.id_utils import EntityId
from mava.utils.tree_utils import (
    add_batch_dim_tree,
    apply_fun_tree,
    remove_batch_dim_tree,
)
from mava.wrappers.env_wrappers import EnvironmentModelWrapper

RecurrentState = Any
RootFn = Callable[[Params, chex.PRNGKey, Any], mctx.RootFnOutput]
RecurrentState = Any
RecurrentFn = Callable[
    [Params, chex.PRNGKey, chex.Array, RecurrentState],
    Tuple[mctx.RecurrentFnOutput, RecurrentState],
]
MaxDepth = Optional[int]
SearchOutput = mctx.PolicyOutput[Union[mctx.GumbelMuZeroExtraData, None]]
TreeSearch = Callable[
    [Params, chex.PRNGKey, mctx.RootFnOutput, RecurrentFn, int, MaxDepth], SearchOutput
]


class MCTS:
    """TODO: Add description here."""

    def __init__(self, config) -> None:
        """TODO: Add description here."""
        self.config = config

    def get_action(
        self,
        forward_fn,
        params,
        rng_key,
        env_state,
        observation,
        agent_info,
        is_evaluator,
        action_mask,
    ):
        """TODO: Add description here."""

        num_simulations = (
            self.config.evaluator_num_simulations
            if is_evaluator
            else self.config.num_simulations
        )
        search_kwargs = (
            self.config.evaluator_other_search_params()
            if is_evaluator
            else self.config.other_search_params()
        )

        search_out = self.environment_model_search(
            forward_fn,
            params,
            rng_key,
            env_state,
            observation,
            agent_info,
            num_simulations,
            action_mask,
            **search_kwargs,
        )
        action = jnp.squeeze(search_out.action.astype(jnp.int32))
        search_policy = jnp.squeeze(search_out.action_weights)

        return (
            action,
            {"search_policies": search_policy},
        )

    @functools.partial(
        jit,
        static_argnames=[
            "self",
            "forward_fn",
            "agent_info",
            "num_simulations",
            "search_kwargs",
        ],
    )
    def environment_model_search(
        self,
        forward_fn,
        params,
        rng_key,
        env_state,
        observation,
        agent_info,
        num_simulations,
        root_action_mask,
        **search_kwargs,
    ):
        """TODO: Add description here."""

        root = self.config.root_fn(forward_fn, params, rng_key, env_state, observation)

        def recurrent_fn(params, rng_key, action, embedding):

            return self.config.recurrent_fn(
                self.config.environment_model,
                forward_fn,
                params,
                rng_key,
                action,
                embedding,
                agent_info,
            )

        root_invalid_actions = utils.add_batch_dim(1 - root_action_mask) + jnp.array(
            [[1, 0, 0, 0, 0]]
        )

        search_output = self.config.search(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=root_invalid_actions,
            max_depth=self.config.max_depth,
            **search_kwargs,
        )

        return search_output

    def learned_get_action(
        self,
        representation_fn,
        dynamics_fn,
        prediction_fn,
        params,
        rng_key,
        observation_history,
        is_evaluator,
        root_action_mask,
    ):
        """TODO: Add description here."""

        num_simulations = (
            self.config.evaluator_num_simulations
            if is_evaluator
            else self.config.num_simulations
        )
        search_kwargs = (
            self.config.evaluator_other_search_params()
            if is_evaluator
            else self.config.other_search_params()
        )

        search_out, predicted_root_value = self.learned_model_search(
            representation_fn,
            dynamics_fn,
            prediction_fn,
            params,
            rng_key,
            observation_history,
            num_simulations,
            root_action_mask,
            **search_kwargs,
        )
        action = jnp.squeeze(search_out.action.astype(jnp.int32))
        search_policy = jnp.squeeze(search_out.action_weights)
        search_value = jnp.squeeze(search_out.search_tree.node_values[:, 0])

        return (
            action,
            {
                "search_policies": search_policy,
                "search_values": search_value,
                "observation_history": jnp.squeeze(observation_history, 0),
                "predicted_values": jnp.squeeze(predicted_root_value),
            },
        )

    @functools.partial(
        jit,
        static_argnames=[
            "self",
            "representation_fn",
            "dynamics_fn",
            "prediction_fn",
            "num_simulations",
            "search_kwargs",
        ],
    )
    @functools.partial(chex.assert_max_traces, n=4)
    def learned_model_search(
        self,
        representation_fn,
        dynamics_fn,
        prediction_fn,
        params,
        rng_key,
        observation_history,
        num_simulations,
        root_action_mask,
        **search_kwargs,
    ):
        """TODO: Add description here."""

        root = self.config.root_fn(
            representation_fn, prediction_fn, params, rng_key, observation_history
        )

        def recurrent_fn(params, rng_key, action, embedding):

            return self.config.recurrent_fn(
                dynamics_fn,
                prediction_fn,
                params,
                rng_key,
                action,
                embedding,
            )

        root_invalid_actions = utils.add_batch_dim(1 - root_action_mask) + jnp.array(
            [[1, 0, 0, 0, 0]]
        )

        search_output = self.config.search(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=root_invalid_actions,
            max_depth=self.config.max_depth,
            **search_kwargs,
        )

        return search_output, root.value
