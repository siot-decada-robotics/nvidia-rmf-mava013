import abc
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import rlax
import tree
from acme.jax import utils
from jax import jit
from haiku._src.basic import merge_leading_dims
from jax.random import KeyArray

from mava.components.jax.training import Batch, Step, TrainingState, Loss
from mava.components.jax.training.losses import MAPGTrustRegionClippingLossConfig
from mava.components.jax.training.model_updating import (
    MAPGEpochUpdateConfig,
    EpochUpdate,
)
from mava.components.jax.training.step import MAPGWithTrustRegionStepConfig
from mava.core_jax import SystemTrainer
from mava.types import OLT
from mava.utils.jax_tree_utils import stack_trees


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

    def on_training_init_start(self, trainer: SystemTrainer) -> None:
        # Note (dries): Assuming the batch and sequence dimensions are flattened.
        trainer.store.full_batch_size = trainer.store.sample_batch_size * (
            trainer.store.sequence_length - 1
        )

        # TODO: surely some of the asserts for batch size can be done in here

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
                o = jax.tree_map(lambda x: merge_leading_dims(x, 2), observation)

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
                return encoded_obs, jnp.squeeze(behavior_values)

            # TODO (sasha): I don't know why it is necessary to explicitely convert observations to
            #  OLTs (they should already be) but if I don't I get a type mismatch with:
            #  'tensorflow.python.saved_model.nested_structure_coder.OLT'
            olts = map(lambda olt: OLT(**olt._asdict()), list(observations.values()))
            observations = stack_trees(
                olts, axis=2
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
            print(behavior_values.shape)
            print(observations.observation.shape)

            # Vmap over batch dimension
            # TODO (sasha): also vmap over agent dim?
            #  swap agent dim to front and double vmap
            batch_gae_advantages = jax.vmap(jax.vmap(trainer.store.gae_fn, in_axes=0))
            # need to transpose to (batch, agents, sequence) so that the vmap performs gae over
            # sequence dim
            advantages, target_values = batch_gae_advantages(
                jnp.transpose(rewards, (0, 2, 1)),
                jnp.transpose(discounts, (0, 2, 1)),
                jnp.transpose(behavior_values, (0, 2, 1)),
            )

            # transpose back to standard (batch, sequence, agents) view
            advantages = jnp.transpose(advantages, (0, 2, 1))
            target_values = jnp.transpose(target_values, (0, 2, 1))

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
            batch = jax.tree_map(lambda x: merge_leading_dims(x, 2), trajectories)
            batch_size = batch.actions.shape[0]
            # TODO (sasha): surely we can calculate this once and not do it every trainer update?
            assert batch_size % trainer.store.num_minibatches == 0, (
                f"Num minibatches must divide batch size. Got batch_size={batch_size}"
                f" num_minibatches={trainer.store.num_minibatches}."
            )

            print(
                f"o:{batch.observations.observation.shape}\n"
                f"a:{batch.actions.shape}\n"
                f"logp:{batch.behavior_log_probs.shape}\n"
                f"vals:{batch.behavior_values.shape}\n"
                f"adv:{batch.advantages.shape}\n"
                f"tvals:{batch.target_values.shape}"
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


# TODO probably put in own file
@dataclass
class MatLossConfig(MAPGTrustRegionClippingLossConfig):
    num_actions: int = 5  # TODO better way to get this


class MatLoss(Loss):
    def __init__(
        self,
        config: MatLossConfig = MatLossConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def loss_grad_fn(
            params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            behaviour_log_probs: Dict[str, jnp.ndarray],
            target_values: Dict[str, jnp.ndarray],
            advantages: Dict[str, jnp.ndarray],
            behavior_values: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios."""

            grads = {}
            loss_info = {}

            agent_key = trainer.store.trainer_agents[0]
            agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
            network = trainer.store.networks["networks"][agent_net_key]

            def loss_fn(
                params: Any,
                observations: Any,
                actions: jnp.ndarray,
                behaviour_log_probs: jnp.ndarray,
                target_values: jnp.ndarray,
                advantages: jnp.ndarray,
                behavior_values: jnp.ndarray,
            ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

                # setting up actions to be passed through model as previous actions
                prev_action_shape = (*actions.shape, self.config.num_actions + 1)
                one_hot_actions = jax.nn.one_hot(actions, self.config.num_actions)
                prev_actions = jnp.zeros(prev_action_shape)
                prev_actions.at[:, 0, 0].set(1)
                prev_actions.at[:, :, 1:].set(one_hot_actions)

                # [idea] TODO (sasha): how would it work if we used the old encoded observation
                #         which were calculated in the step fn when calculating behaviour vals
                values, encoded_obs = network.encoder.network.apply(
                    params["encoder"], observations
                )
                values = jnp.squeeze(values)

                distribution_params = network.decoder.network.apply(
                    params["decoder"], prev_actions, encoded_obs
                )

                # copy ppo logp + entropy funcs
                log_probs = distribution_params.log_prob(actions)
                # log_probs = network.log_prob(distribution_params, actions)
                # entropy = network.entropy(distribution_params)
                entropy = distribution_params.entropy()
                # Compute importance sampling weights:
                # current policy / behavior policy.
                print(f"logp:{log_probs.shape}")
                rhos = jnp.exp(log_probs - behaviour_log_probs)
                clipping_epsilon = self.config.clipping_epsilon

                batch_pg_loss = jax.vmap(
                    rlax.clipped_surrogate_pg_loss, in_axes=(1, 1, None)
                )
                policy_loss = batch_pg_loss(rhos, advantages, clipping_epsilon)

                # Value function loss. Exclude the bootstrap value
                print(f"pol loss:{policy_loss.shape}")
                unclipped_value_error = target_values - values
                unclipped_value_loss = unclipped_value_error**2

                if self.config.clip_value:
                    # Clip values to reduce variablility during critic training.
                    clipped_values = behavior_values + jnp.clip(
                        values - behavior_values,
                        -clipping_epsilon,
                        clipping_epsilon,
                    )
                    clipped_value_error = target_values - clipped_values
                    clipped_value_loss = clipped_value_error**2
                    value_loss = jnp.mean(
                        jnp.fmax(unclipped_value_loss, clipped_value_loss), axis=0
                    )
                else:
                    value_loss = jnp.mean(unclipped_value_loss, axis=0)

                # Entropy regulariser.
                entropy_loss = -jnp.mean(entropy, axis=0)

                total_loss = (
                    policy_loss
                    + value_loss * self.config.value_cost
                    + entropy_loss * self.config.entropy_cost
                )

                loss_info = {
                    "loss_total": total_loss,
                    "loss_policy": policy_loss,
                    "loss_value": value_loss,
                    "loss_entropy": entropy_loss,
                }
                print(f"value loss:{value_loss.shape}")
                print(f"ent loss:{entropy_loss.shape}")
                print(f"tls {total_loss.shape}")

                return jnp.mean(total_loss), jax.tree_map(jnp.mean, loss_info)

            # TODO (sasha): this is not the correct solution, it's going to apply the avg loss
            #  3 times. Better to remake the update class and do it once.
            grad, info = jax.grad(loss_fn, has_aux=True)(
                params[agent_net_key],
                observations.observation,
                actions,
                behaviour_log_probs,
                target_values,
                advantages,
                behavior_values,
            )

            for agent_key in trainer.store.trainer_agents:
                grads[agent_key], loss_info[agent_key] = grad, info

            return grads, loss_info

        # Save the gradient function.
        trainer.store.grad_fn = loss_grad_fn

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MatLossConfig


class MatEpochUpdate(EpochUpdate):
    def __init__(
        self,
        config: MAPGEpochUpdateConfig = MAPGEpochUpdateConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        trainer.store.num_epochs = self.config.num_epochs
        trainer.store.num_minibatches = self.config.num_minibatches

        def model_update_epoch(
            carry: Tuple[KeyArray, Any, optax.OptState, Batch],
            unused_t: Tuple[()],
        ) -> Tuple[
            Tuple[KeyArray, Any, optax.OptState, Batch],
            Dict[str, jnp.ndarray],
        ]:
            """Performs model updates based on one epoch of data."""
            key, params, opt_states, batch = carry

            new_key, subkey = jax.random.split(key)

            # TODO (dries): This assert is ugly. Is there a better way to do this check?
            # Maybe using a tree map of some sort?
            # shapes = jax.tree_map(
            #         lambda x: x.shape[0]==trainer.store.full_batch_size, batch
            #     )
            # assert ...
            # TODO (sasha): I shouldn't have to duplicate this class, the only thing that doesn't
            #  work is this assert
            assert (
                batch.observations.observation[:, :, 0].shape[0]
                == trainer.store.full_batch_size
            )

            permutation = jax.random.permutation(subkey, trainer.store.full_batch_size)

            shuffled_batch = jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_map(
                lambda x: jnp.reshape(
                    x, [self.config.num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            (new_params, new_opt_states), metrics = jax.lax.scan(
                trainer.store.minibatch_update_fn,
                (params, opt_states),
                minibatches,
                length=self.config.num_minibatches,
            )

            return (new_key, new_params, new_opt_states, batch), metrics

        trainer.store.epoch_update_fn = model_update_epoch

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPGEpochUpdateConfig
