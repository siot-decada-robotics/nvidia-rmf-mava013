import abc
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import tree
from acme.jax import utils
from jax import jit

from mava.components.jax.training import Batch, Step, TrainingState
from mava.components.jax.training.step import MAPGWithTrustRegionStepConfig
from mava.core_jax import SystemTrainer
from mava.types import OLT
from mava.utils.jax_tree_utils import stack_trees


def _interleave_arrays(arrays: List[np.ndarray]):
    """Utility method to interleave arrays generalized from:
    https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays

    Given: [[1,3,5], [2,4,6]]
    Output: [1,2,3,4,5,6]

    Args:
        arrays: list of numpy arrays to interleave

    Returns: Values from array interleaved
    """
    n_arrays = len(arrays)
    array_shape = arrays[0].shape
    array_dtype = arrays[0].dtype

    interleaved = jnp.empty(
        (array_shape[0] * n_arrays, *array_shape[1:]), dtype=array_dtype
    )
    for i, array in enumerate(arrays):
        # set every n_arrays'th element to a value from array
        interleaved.at[i::n_arrays].set(array)

    return interleaved


class MatStep(Step):
    def __init__(
        self,
        config: MAPGWithTrustRegionStepConfig = MAPGWithTrustRegionStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        # @jit
        def sgd_step(
            states: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""
            # Extract the data.
            observations, actions, rewards, termination, extra = (
                sample.data.observations,
                sample.data.actions,
                sample.data.rewards,
                sample.data.discounts,
                sample.data.extras,
            )

            n_agents = len(observations)

            discounts = tree.map_structure(
                lambda x: x * self.config.discount, termination
            )

            behavior_log_probs = extra["policy_info"]

            networks = trainer.store.networks["networks"]

            def get_values_and_encode_obs(
                net_key: Any, observation: Any
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
                batch_size, num_sequences = observation.shape[:2]
                o = jax.tree_map(
                    lambda x: jnp.reshape(x, [-1] + list(x.shape[2:])), observation
                )

                behavior_values, encoded_obs = networks[net_key].encoder.forward_fn(
                    states.params[net_key]["encoder"], o
                )
                # behavior_values = jnp.reshape(behavior_values, reward.shape[0:2])
                behavior_values = jnp.reshape(
                    behavior_values,
                    (batch_size, num_sequences, *behavior_values.shape[1:]),
                )
                encoded_obs = jnp.reshape(
                    encoded_obs, (batch_size, num_sequences, *encoded_obs.shape[1:])
                )
                return encoded_obs, behavior_values

            # TODO (sasha): I don't know why it is necessary to explicitely convert observations to
            #  OLTs (they should already be) but if I don't I get a type mismatch with:
            #  'tensorflow.python.saved_model.nested_structure_coder.OLT'
            olts = map(lambda olt: OLT(**olt._asdict()), list(observations.values()))
            observations = stack_trees(
                olts, axis=-1
            )  # (batch, sequence, agents, obs...)
            # TODO (sasha): dict.values might be returning different order for each of these, get
            #  order once and then get values as [d[key] for key in keys]
            actions = stack_trees(list(actions.values()), axis=-1)
            discounts = stack_trees(list(discounts.values()), axis=-1)
            rewards = stack_trees(list(rewards.values()), axis=-1)
            behavior_log_probs = stack_trees(list(behavior_log_probs.values()), axis=-1)
            print(f"actions:{actions.shape}")

            # Need to stack observations here because need each agent in order to perform
            # inference
            # current shape is (batch, sequence, row, col)
            # need to be (batch, sequence, agent, row, col)
            agent_nets = trainer.store.trainer_agent_net_keys

            encoded_obs, behavior_values = get_values_and_encode_obs(
                list(agent_nets.values())[0], observations.observation
            )
            # # TODO (sasha): there has to be a cleaner way to do this
            # encoded_obs = [values_and_obs[key][0] for key in agent_nets.keys()]
            # behavior_values = {key: values_and_obs[key][1] for key in agent_nets.keys()}

            # Vmap over batch dimension
            # TODO (sasha): also vmap over agent dim?
            #  swap agent dim to front and double vmap
            batch_gae_advantages = jax.vmap(trainer.store.gae_fn, in_axes=0)
            advantages = {}
            target_values = {}
            for i in range(n_agents):
                # shape of rewards is 3!? Want it to be 20
                batch_gae_advantages(
                    rewards[i], discounts[i], behavior_values[i]
                )

            # Exclude the last step - it was only used for bootstrapping.
            # The shape is [num_sequences, num_steps, ..]
            observations, actions, behavior_log_probs, behavior_values = jax.tree_map(
                lambda x: x[:, :-1],
                (observations, actions, behavior_log_probs, behavior_values),
            )

            trajectories = Batch(
                observations=observations,
                actions=actions,
                advantages=advantages,
                behavior_log_probs=behavior_log_probs,
                target_values=target_values,
                behavior_values=behavior_values,
            )

            # Concatenate all trajectories. Reshape from [num_sequences, num_steps,..]
            # to [num_sequences * num_steps,..]
            agent_0_t_vals = list(target_values.values())[0]
            assert len(agent_0_t_vals) > 1
            num_sequences = agent_0_t_vals.shape[0]
            num_steps = agent_0_t_vals.shape[1]
            batch_size = num_sequences * num_steps
            assert batch_size % trainer.store.num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, trainer.store.num_minibatches)
            batch = jax.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            (new_key, new_params, new_opt_states, _,), metrics = jax.lax.scan(
                trainer.store.epoch_update_fn,
                (states.random_key, states.params, states.opt_states, batch),
                (),
                length=trainer.store.num_epochs,
            )

            # Set the metrics
            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["norm_params"] = optax.global_norm(states.params)
            metrics["observations_mean"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_map(
                        lambda x: jnp.abs(jnp.mean(x, axis=(0, 1))), observations
                    ),
                    num_batch_dims=0,
                )
            )
            metrics["observations_std"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_map(lambda x: jnp.std(x, axis=(0, 1)), observations),
                    num_batch_dims=0,
                )
            )
            metrics["rewards_mean"] = jax.tree_map(
                lambda x: jnp.mean(jnp.abs(jnp.mean(x, axis=(0, 1)))), rewards
            )
            metrics["rewards_std"] = jax.tree_map(
                lambda x: jnp.std(x, axis=(0, 1)), rewards
            )

            new_states = TrainingState(
                params=new_params, opt_states=new_opt_states, random_key=new_key
            )
            return new_states, metrics

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            networks = trainer.store.networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            opt_states = trainer.store.opt_states
            random_key, _ = jax.random.split(trainer.store.key)

            states = TrainingState(
                params=params, opt_states=opt_states, random_key=random_key
            )

            new_states, metrics = sgd_step(states, sample)

            # Set the new variables
            # TODO (dries): key is probably not being store correctly.
            # The variable client might lose reference to it when checkpointing.
            # We also need to add the optimizer and random_key to the variable
            # server.
            trainer.store.key = new_states.random_key

            networks = trainer.store.networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            for net_key in params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.networks["networks"][net_key].params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.params[net_key][param_key]

                # Update the optimizer
                # This needs to be in the loop to not lose the reference.
                trainer.store.opt_states[net_key] = new_states.opt_states[net_key]

            return metrics

        trainer.store.step_fn = step

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPGWithTrustRegionStepConfig
