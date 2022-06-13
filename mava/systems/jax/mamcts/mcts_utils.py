from typing import Callable, List

import jax
import jax.numpy as jnp
import mctx
from acme.jax import utils
from chex import Array
from mctx._src.policies import _mask_invalid_actions

from mava.systems.jax.mamcts.learned_model_utils import (
    inv_value_transform,
    logits_to_scalar,
)
from mava.systems.jax.mamcts.networks import MAMUNetworks, PredictionNetwork
from mava.utils.id_utils import EntityId
from mava.utils.tree_utils import add_batch_dim_tree, remove_batch_dim_tree, stack_trees
from mava.wrappers.env_wrappers import MAMCTSWrapper


class MAMU:
    def learned_root_fn() -> Callable:
        """A simple root_fn to generate a root state's prior probabilities and value

        Return:
            Callable function used by the MCTS component"""

        def root_fn(
            representation_fn, prediction_fn, params, rng_key, observation_history
        ):

            embedding = representation_fn(
                observation_history=observation_history, params=params["representation"]
            )

            prior_logits, values = prediction_fn(
                observations=embedding, params=params["prediction"]
            )

            values = logits_to_scalar(values)
            values = inv_value_transform(values)

            return mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=values,
                embedding=embedding,
            )

        return root_fn

    def learned_recurrent_fn(discount_gamma: float = 1.0) -> Callable:
        """Creates a recurrent function used by the MCTS component - this setting makes
        other agents all select a default action in an individual agents tree search"""

        def recurrent_fn(
            dynamics_fn,
            prediction_fn,
            params,
            rng_key,
            action,
            embedding,
        ) -> mctx.RecurrentFnOutput:

            new_embedding, reward = dynamics_fn(
                previous_embedding=embedding, action=action, params=params["dynamics"]
            )
            reward = logits_to_scalar(reward)
            reward = inv_value_transform(reward)

            prior_logits, values = prediction_fn(
                observations=new_embedding, params=params["prediction"]
            )
            values = logits_to_scalar(values)
            values = inv_value_transform(values)

            return (
                mctx.RecurrentFnOutput(
                    reward=reward.reshape(
                        new_embedding.shape[0],
                    ),
                    discount=jnp.float32(discount_gamma).reshape(
                        new_embedding.shape[0],
                    ),
                    prior_logits=prior_logits,
                    value=values.reshape(
                        new_embedding.shape[0],
                    ),
                ),
                new_embedding,
            )

        return recurrent_fn


class MAMCTS:
    def environment_root_fn() -> Callable:
        """A simple root_fn to generate a root state's prior probabilities and value

        Return:
            Callable function used by the MCTS component"""

        def root_fn(forward_fn, params, rng_key, env_state, observation):

            prior_logits, values = forward_fn(observations=observation, params=params)
            values = logits_to_scalar(values)
            values = inv_value_transform(values)

            return mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=values,
                embedding=add_batch_dim_tree(env_state),
            )

        return root_fn

    def default_action_recurrent_fn(
        default_action: int, discount_gamma: float = 0.99
    ) -> Callable:
        """Creates a recurrent function used by the MCTS component - this setting makes
        other agents all select a default action in an individual agents tree search"""

        def recurrent_fn(
            environment_model: MAMCTSWrapper,
            forward_fn: PredictionNetwork,
            params,
            rng_key,
            action,
            env_state,
            agent_info,
        ) -> mctx.RecurrentFnOutput:
            agent_list = environment_model.get_possible_agents()

            actions = {agent_id: default_action for agent_id in agent_list}

            actions[agent_info] = jnp.squeeze(action)

            env_state = remove_batch_dim_tree(env_state)

            next_state, timestep, _ = environment_model.step(env_state, actions)

            observation = environment_model.get_observation(next_state, agent_info)

            prior_logits, values = forward_fn(
                observations=utils.add_batch_dim(observation), params=params
            )
            values = logits_to_scalar(values)
            values = inv_value_transform(values)

            agent_mask = utils.add_batch_dim(
                environment_model.get_agent_mask(next_state, agent_info)
            )

            prior_logits = _mask_invalid_actions(prior_logits, agent_mask)

            reward = timestep.reward[agent_info].reshape(
                1,
            )

            discount = (
                timestep.discount[agent_info].reshape(
                    1,
                )
                * discount_gamma
            )

            return (
                mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=discount,
                    prior_logits=prior_logits,
                    value=values,
                ),
                add_batch_dim_tree(next_state),
            )

        return recurrent_fn

    def random_action_recurrent_fn(discount_gamma: float = 0.99) -> Callable:
        """Creates a recurrent function used by the MCTS component - this setting makes
        other agents all select random actions in an individual agents tree search"""

        def recurrent_fn(
            environment_model: MAMCTSWrapper,
            forward_fn,
            params,
            rng_key,
            action,
            env_state,
            agent_info,
        ) -> mctx.RecurrentFnOutput:
            agent_list = environment_model.get_possible_agents()

            rng_key, *agent_action_keys = jax.random.split(rng_key, len(agent_list) + 1)

            actions = {
                agent_id: jax.random.randint(
                    agent_rng_key,
                    (),
                    minval=0,
                    maxval=environment_model.action_spec()[agent_info].num_values,
                )
                for agent_rng_key, agent_id in zip(agent_action_keys, agent_list)
            }

            actions[agent_info] = jnp.squeeze(action)

            env_state = remove_batch_dim_tree(env_state)

            next_state, timestep, _ = environment_model.step(env_state, actions)

            observation = environment_model.get_observation(next_state, agent_info)

            prior_logits, values = forward_fn(
                observations=utils.add_batch_dim(observation), params=params
            )
            values = logits_to_scalar(values)
            values = inv_value_transform(values)

            agent_mask = utils.add_batch_dim(
                environment_model.get_agent_mask(next_state, agent_info)
            )

            prior_logits = _mask_invalid_actions(prior_logits, agent_mask)

            reward = timestep.reward[agent_info].reshape(
                1,
            )

            discount = (
                timestep.discount[agent_info].reshape(
                    1,
                )
                * discount_gamma
            )

            return (
                mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=discount,
                    prior_logits=prior_logits,
                    value=values,
                ),
                add_batch_dim_tree(next_state),
            )

        return recurrent_fn

    def greedy_policy_recurrent_fn(discount_gamma: float = 0.99) -> Callable:
        """Creates a recurrent function used by the MCTS component - this setting makes
        other agents all select the greedy action, according to the searching agent's policy, in an individual agents tree search"""

        def recurrent_fn(
            environment_model: MAMCTSWrapper,
            forward_fn,
            params,
            rng_key,
            action,
            env_state,
            agent_info,
        ) -> mctx.RecurrentFnOutput:
            agent_list = environment_model.get_possible_agents()

            stacked_agents = stack_trees(agent_list)

            env_state = remove_batch_dim_tree(env_state)

            prev_observations = jax.vmap(
                environment_model.get_observation, in_axes=(None, 0)
            )(env_state, stacked_agents)

            prev_prior_logits, _ = forward_fn(
                observations=prev_observations, params=params
            )

            other_agent_masks = jax.vmap(
                environment_model.get_agent_mask, in_axes=(None, 0)
            )(env_state, stacked_agents)

            prev_prior_logits = jax.vmap(_mask_invalid_actions, in_axes=(0, 0))(
                prev_prior_logits, other_agent_masks
            )

            agent_actions = jnp.argmax(prev_prior_logits, -1)

            actions = {agent_id: agent_actions[agent_id.id] for agent_id in agent_list}

            actions[agent_info] = jnp.squeeze(action)

            next_state, timestep, _ = environment_model.step(env_state, actions)

            observation = environment_model.get_observation(next_state, agent_info)

            prior_logits, values = forward_fn(
                observations=utils.add_batch_dim(observation),
                params=params,
            )
            values = logits_to_scalar(values)
            values = inv_value_transform(values)

            agent_mask = utils.add_batch_dim(
                environment_model.get_agent_mask(next_state, agent_info)
            )

            prior_logits = _mask_invalid_actions(prior_logits, agent_mask)

            reward = timestep.reward[agent_info].reshape(
                1,
            )

            discount = (
                timestep.discount[agent_info].reshape(
                    1,
                )
                * discount_gamma
            )

            return (
                mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=discount,
                    prior_logits=prior_logits,
                    value=values,
                ),
                add_batch_dim_tree(next_state),
            )

        return recurrent_fn
