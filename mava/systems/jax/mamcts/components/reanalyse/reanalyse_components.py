from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import acme.jax.utils as utils
import chex
import dm_env
import jax
import jax.numpy as jnp
import reverb
import tree
from acme.adders.reverb import utils as acme_utils
from haiku._src.basic import merge_leading_dims

from mava.adders import reverb as reverb_adders
from mava.adders.reverb.base import Step
from mava.components.jax.building.parameter_client import BaseParameterClient
from mava.components.jax.component import Component
from mava.core_jax import SystemBuilder, SystemParameterServer
from mava.systems.jax.mamcts.components.reanalyse.base import ReanalyseComponent
from mava.systems.jax.mamcts.reanalyse_worker import ReanalyseWorker
from mava.systems.jax.parameter_client import ParameterClient
from mava.utils.tree_utils import index_stacked_tree


@dataclass
class ReanalyseParameterClientConfig:
    pass


class ReanalyseParameterClient(ReanalyseComponent, BaseParameterClient):
    def __init__(
        self, config: ReanalyseParameterClientConfig = ReanalyseParameterClientConfig()
    ) -> None:
        self.config = config

    def on_reanalyse_worker_init_start(self, reanalyse_worker: ReanalyseWorker) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        # Create parameter client
        params = {}
        set_keys = []
        get_keys = []

        trainer_networks = reanalyse_worker.store.trainer_networks[
            reanalyse_worker.store.trainer_id
        ]
        for net_type_key in reanalyse_worker.store.networks.keys():
            for net_key in reanalyse_worker.store.networks[net_type_key].keys():
                params[f"{net_type_key}-{net_key}"] = reanalyse_worker.store.networks[
                    net_type_key
                ][net_key].params
                if net_key in set(trainer_networks):
                    set_keys.append(f"{net_type_key}-{net_key}")
                else:
                    get_keys.append(f"{net_type_key}-{net_key}")

        count_names, params = self._set_up_count_parameters(params=params)

        get_keys.extend(count_names)
        reanalyse_worker.store.reanalyse_counts = {
            name: params[name] for name in count_names
        }

        # Create parameter client
        parameter_client = None
        if reanalyse_worker.store.parameter_server_client:
            parameter_client = ParameterClient(
                client=reanalyse_worker.store.parameter_server_client,
                parameters=params,
                get_keys=get_keys,
                set_keys=set_keys,
            )

            # Get all the initial parameters
            parameter_client.get_all_and_wait()

        reanalyse_worker.store.reanalyse_parameter_client = parameter_client

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "reanalyse_parameter_client"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """
        Optional class which specifies the
        dataclass/config object for the component.
        """
        return ReanalyseParameterClientConfig


@dataclass
class ReanalyseUpdateConfig:
    pass


class ReanalyseUpdate(ReanalyseComponent):
    def __init__(
        self,
        config: ReanalyseUpdateConfig = ReanalyseUpdateConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config
        self.step_counter = 0

    def on_reanalyse_worker_init(self, reanalyse_worker: ReanalyseWorker) -> None:
        self.mcts = reanalyse_worker.store.mcts
        self.networks = reanalyse_worker.store.networks["networks"]
        reanalyse_worker.store.trainer_table_entry = (
            reanalyse_worker.store.table_network_config[
                reanalyse_worker.store.trainer_id
            ]
        )
        reanalyse_worker.store.trainer_agents = reanalyse_worker.store.agents[
            : len(reanalyse_worker.store.trainer_table_entry)
        ]
        reanalyse_worker.store.trainer_agent_net_keys = {
            agent: reanalyse_worker.store.trainer_table_entry[a_i]
            for a_i, agent in enumerate(reanalyse_worker.store.trainer_agents)
        }
        self.agent_nets = reanalyse_worker.store.trainer_agent_net_keys

        self.num_simulations = reanalyse_worker.store.num_simulations

        self.priority_fns = {
            table_key: lambda step: self.muzero_priority(step)
            for table_key in reanalyse_worker.store.table_network_config.keys()
        }

    def on_reanalyse_worker_step_start(self, reanalyse_worker: ReanalyseWorker) -> None:
        reanalyse_worker.store.reanalyse_parameter_client.get_async()

    @partial(jax.jit, static_argnames=["self"])
    def muzero_priority(self, step) -> chex.Array:
        """Calculates the sequence priority.

        Args:
            step : a reverb step.

        Returns:
            the priority for the sequence.
        """
        extras = step.extras["policy_info"]
        priority = 0
        for agent in extras:
            priority += jnp.abs(
                extras[agent]["search_values"] - extras[agent]["predicted_values"]
            )
        priority = jnp.max(priority, -1)
        return priority

    @partial(jax.jit, static_argnames=["self"])
    def reanalyse_data(self, sample, rng_key):
        data = sample.data
        extra = data.extras

        observation_history = {}
        for agent_key in extra["policy_info"].keys():
            observation_history[agent_key] = extra["policy_info"][agent_key][
                "observation_history"
            ]

        updated_search_policies = {}
        updated_search_values = {}
        updated_predicted_root_values = {}
        for net_key in self.agent_nets:
            batch_size = observation_history[net_key].shape[0]
            seq_size = observation_history[net_key].shape[1]
            network = self.networks[self.agent_nets[net_key]]
            full_history = merge_leading_dims(observation_history[net_key], 2)
            full_history = jax.numpy.expand_dims(full_history, 1)
            rng_keys = jax.random.split(rng_key, len(full_history))
            # TODO action masking
            search_output, predicted_root_value = jax.vmap(
                self.mcts.learned_model_search,
                in_axes=(None, None, None, None, 0, 0, None, None),
                out_axes=(0, 0),
            )(
                network.representation_fn,
                network.dynamics_fn,
                network.prediction_fn,
                network.params,
                rng_keys,
                full_history,
                self.num_simulations,
                None,
            )

            updated_search_policies[net_key] = jnp.squeeze(
                search_output.action_weights.reshape(batch_size, seq_size, -1)
            )
            updated_search_values[net_key] = jnp.squeeze(
                search_output.search_tree.node_values[:, :, 0].reshape(
                    batch_size, seq_size, -1
                )
            )
            updated_predicted_root_values[net_key] = jnp.squeeze(
                predicted_root_value.reshape(batch_size, seq_size, -1)
            )

        return (
            updated_search_policies,
            updated_search_values,
            updated_predicted_root_values,
        )

    def on_reanalyse_worker_step(self, reanalyse_worker: ReanalyseWorker) -> None:
        # TODO mask values updated values after discounts
        sample = next(reanalyse_worker.store.dataset_iterator)

        keys, _, *_ = sample.info
        rng_key, reanalyse_worker.store.key = jax.random.split(
            reanalyse_worker.store.key
        )
        (
            updated_search_policies,
            updated_search_values,
            updated_predicted_root_values,
        ) = self.reanalyse_data(sample, rng_key)

        for table_key in reanalyse_worker.store.table_network_config.keys():
            reanalyse_worker.store.data_server_client.mutate_priorities(
                table=table_key, deletes=keys
            )

        termination_mask = jax.tree_map(
            lambda discount: jnp.concatenate(
                [jnp.ones_like(discount)[:, 0, None], discount[:, :-1]], -1
            ),
            sample.data.discounts,
        )

        for batch_index in range(reanalyse_worker.store.reanalyse_sample_batch_size):
            with reanalyse_worker.store.data_server_client.trajectory_writer(
                num_keep_alive_refs=reanalyse_worker.store.sequence_length
            ) as writer:
                for sequence_index in range(reanalyse_worker.store.sequence_length):

                    observations = index_stacked_tree(
                        sample.data.observations, (batch_index, sequence_index)
                    )

                    actions = index_stacked_tree(
                        sample.data.actions, (batch_index, sequence_index)
                    )

                    rewards = index_stacked_tree(
                        sample.data.rewards, (batch_index, sequence_index)
                    )

                    discounts = index_stacked_tree(
                        sample.data.discounts, (batch_index, sequence_index)
                    )

                    index_termination_mask = index_stacked_tree(
                        termination_mask, (batch_index, sequence_index)
                    )

                    extras = index_stacked_tree(
                        sample.data.extras, (batch_index, sequence_index)
                    )

                    start_of_episode = sample.data.start_of_episode[
                        batch_index, sequence_index
                    ]

                    indexed_updated_search_policies = index_stacked_tree(
                        updated_search_policies, (batch_index, sequence_index)
                    )
                    indexed_updated_search_values = index_stacked_tree(
                        updated_search_values, (batch_index, sequence_index)
                    )
                    indexed_updated_predicted_root_values = index_stacked_tree(
                        updated_predicted_root_values, (batch_index, sequence_index)
                    )

                    indexed_updated_search_policies = jax.tree_map(
                        lambda val, mask: val * mask,
                        indexed_updated_search_policies,
                        index_termination_mask,
                    )
                    indexed_updated_search_values = jax.tree_map(
                        lambda val, mask: val * mask,
                        indexed_updated_search_values,
                        index_termination_mask,
                    )
                    indexed_updated_predicted_root_values = jax.tree_map(
                        lambda val, mask: val * mask,
                        indexed_updated_predicted_root_values,
                        index_termination_mask,
                    )

                    for agent in extras["policy_info"].keys():
                        extras["policy_info"][agent][
                            "search_policies"
                        ] = indexed_updated_search_policies[agent]
                        extras["policy_info"][agent][
                            "search_values"
                        ] = indexed_updated_search_values[agent]
                        extras["policy_info"][agent][
                            "predicted_values"
                        ] = indexed_updated_predicted_root_values[agent]

                    trajectory_step = {
                        "observations": observations,
                        "actions": actions,
                        "rewards": rewards,
                        "discounts": discounts,
                        "start_of_episode": start_of_episode,
                        "extras": extras,
                    }

                    writer.append(trajectory_step)

                for table_key in reanalyse_worker.store.table_network_config.keys():

                    trajectory = tree.map_structure(lambda x: x[:], writer.history)

                    # Pack the history into a base.Step structure and get numpy converted
                    # variant for priotiy computation.
                    trajectory = Step(**trajectory)

                    table_priorities = acme_utils.calculate_priorities(
                        self.priority_fns, trajectory
                    )

                    writer.create_item(
                        table=f"{table_key}_reanalyse",
                        priority=table_priorities[table_key],
                        trajectory=trajectory,
                    )
                    writer.flush()

        self.step_counter += 1
        print(f"Reanalyse Step : {self.step_counter}")

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "reanalyse_update"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """
        Optional class which specifies the
        dataclass/config object for the component.
        """
        return ReanalyseUpdateConfig
