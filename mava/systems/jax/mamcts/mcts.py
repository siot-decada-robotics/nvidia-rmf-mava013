import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import acme.jax.utils as utils
import chex
import jax.numpy as jnp
import mctx
from haiku import Params
from jax import jit

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
    """Monte Carlo Search Tree Class

    Provides functionality for MAMCTS and MAMU System tree searches"""

    def __init__(self, config) -> None:
        """Initialise MCTS Class.

        Args:
            config : MCTSConfig - instantiated in the MCTSActionSelection executor"""

        self.config = config

    def get_mamcts_action(
        self,
        forward_fn,
        params,
        rng_key,
        env_state,
        observation,
        agent_info,
        is_evaluator,
        root_action_mask,
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        """Performs a tree search and gets a MAMCTS agent's action and policy information.

        Args:
            forward_fn : a PredictionNetwork's forward function - takes in an observation and parameters and returns a policy and value.
            params : the neural network parameters used in the forward_fn.
            rng_key : a pseudo random number key
            env_state : the current environment state
            observation : the current agent observation
            agent_info : the current agent's information - ID
            is_evaluator : whether or not the current agent is an evaluator or an executor - specifies the number of simulations to use.
            root_action_mask : the current agent's mask for legal actions

        Return:
            a selected action, dictionary containing relevant policy information for training"""

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
            root_action_mask,
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
    ) -> SearchOutput:
        """Perform the MCTS for an MAMCTS agent"""

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

        root_invalid_actions = utils.add_batch_dim(
            1 - root_action_mask
        )  # + jnp.array([[1, 0, 0, 0, 0]])

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

    def mamu_get_action(
        self,
        representation_fn,
        dynamics_fn,
        prediction_fn,
        params,
        rng_key,
        observation_history,
        is_evaluator,
        root_action_mask,
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        """Performs a tree search and gets a MAMU agent's action and policy information.

        Args:
            representation_fn : a RepresentationNetwork's forward function - takes in an observation history and parameters and returns a initial root embedding.
            dynamics_fn : a DynamicsNetwork's forward function - takes in an embedding, action and parameters and returns a new embedding and reward.
            prediction_fn : a PredictionNetwork's forward function - takes in an embedding and parameters and returns a policy and value.
            params : the neural network parameters used in the three functions.
            rng_key : a pseudo random number key
            env_state : the current environment state
            observation_history : the current agent's observation history
            is_evaluator : whether or not the current agent is an evaluator or an executor - specifies the number of simulations to use.
            root_action_mask : the current agent's mask for legal actions

        Return:
            a selected action, dictionary containing relevant policy information for training"""

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
    ) -> Tuple[SearchOutput, chex.Array]:
        """Perform the MCTS for an MAMU agent"""

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

        if root_action_mask is not None:
            root_invalid_actions = utils.add_batch_dim(
                1 - root_action_mask
            )  # + jnp.array([[1, 0, 0, 0, 0]])
        else:
            root_invalid_actions = jnp.zeros_like(root.prior_logits)

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
