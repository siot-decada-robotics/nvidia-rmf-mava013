from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import reverb
import tree
from acme.jax import utils
from haiku._src.basic import merge_leading_dims

from mava.components.jax.training import Batch, Step, TrainingState

from mava.components.jax.training.step import MAPGWithTrustRegionStepConfig
from mava.core_jax import SystemTrainer
from mava.types import OLT
from mava.utils.jax_tree_utils import stack_trees

from mava.systems.jax.mat.components.training.utils import (
    merge_agents_to_sequence,
    unmerge_agents_to_sequence,
)


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
        trainer.store.adv_batch_size = (
            trainer.store.sample_batch_size
            * trainer.store.sequence_length
            * len(trainer.store.agents)
            - 1
        )
        trainer.store.action_batch_size = (
            trainer.store.sample_batch_size * trainer.store.sequence_length
        )

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        def sgd_step(
            states: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""
            print("SGD STEP!")
            # Extract the data.
            observations, actions, rewards, termination, extra = (
                sample.data.observations,
                sample.data.actions,
                sample.data.rewards,
                sample.data.discounts,
                sample.data.extras,
            )

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

                behavior_values = jnp.reshape(
                    behavior_values,
                    (batch_size, num_sequences, *behavior_values.shape[1:]),
                )
                encoded_obs = jnp.reshape(
                    encoded_obs, (batch_size, num_sequences, *encoded_obs.shape[1:])
                )
                # TODO (sasha): this squeeze prevents MAT working on single agent envs
                return encoded_obs, jnp.squeeze(behavior_values)

            # TODO (sasha): I don't know why it is necessary to explicitly convert observations to
            #  OLTs (they should already be) but if I don't I get a type mismatch with:
            #  'tensorflow.python.saved_model.nested_structure_coder.OLT'
            agent_keys = trainer.store.trainer_agents
            olts = map(
                lambda olt: OLT(**olt._asdict()),
                [observations[key] for key in agent_keys],
            )
            # Need to stack observations here because need each agent in order to perform
            # inference
            # current shape is (batch, sequence, row, col)
            # need to be (batch, sequence, agent, row, col)
            observations = stack_trees(olts, axis=2)
            actions = stack_trees([actions[key] for key in agent_keys], axis=-1)
            discounts = stack_trees([discounts[key] for key in agent_keys], axis=-1)
            rewards = stack_trees([rewards[key] for key in agent_keys], axis=-1)
            behavior_log_probs = stack_trees(
                [behavior_log_probs[key] for key in agent_keys], axis=-1
            )

            agent_nets = trainer.store.trainer_agent_net_keys
            encoded_obs, behavior_values = get_values_and_encode_obs(
                list(agent_nets.values())[0], observations.observation
            )

            # Vmap over batch and agent dimension
            batch_gae_advantages = jax.vmap(trainer.store.gae_fn, in_axes=0)

            advantages, target_values = batch_gae_advantages(
                merge_agents_to_sequence(rewards),
                merge_agents_to_sequence(discounts),
                merge_agents_to_sequence(behavior_values),
            )

            # need to get advantages and target values to be the same shape as everything else so
            # adding dummy value and unmerging. Will remove this after shuffling and
            # minibatch selection.
            advantages = jnp.concatenate(
                [advantages, jnp.zeros((advantages.shape[0], 1))], axis=1
            )
            advantages = unmerge_agents_to_sequence(
                advantages, len(trainer.store.agents)
            )

            target_values = jnp.concatenate(
                [target_values, jnp.zeros((target_values.shape[0], 1))], axis=1
            )
            target_values = unmerge_agents_to_sequence(
                target_values, len(trainer.store.agents)
            )

            # print(f"rew:{rewards.shape}")
            # print(f"merged+stacked:{merge_agents_to_sequence(rewards).shape}")
            print(f"adv|vals shape:{advantages.shape}|{target_values.shape}")

            # TODO (sasha) elsewhere: Exclude the last step - it was only used for bootstrapping.
            # The shape is [num_sequences, num_steps, ..]
            # behavior_log_probs, behavior_values = jax.tree_map(
            #     lambda x: x[:, :-1],
            #     (
            #         merge_agents_to_sequence(behavior_log_probs),
            #         merge_agents_to_sequence(behavior_values),
            #     ),
            # )

            # TODO (sasha): currently advs and target vals are in a diff shape:
            #  (batch, seq*n_agents) instead of
            #  (batch, seq, n_agents)
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
