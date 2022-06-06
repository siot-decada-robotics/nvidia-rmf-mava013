from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import acme.jax.utils as utils
import dm_env
import jax
import reverb
import tree
from haiku._src.basic import merge_leading_dims

from mava.adders import reverb as reverb_adders
from mava.components.jax.building.parameter_client import BaseParameterClient
from mava.components.jax.component import Component
from mava.core_jax import SystemBuilder, SystemParameterServer
from mava.systems.jax.mamcts.components.reanalyse.base import ReanalyseComponent
from mava.systems.jax.mamcts.reanalyse_worker import ReanalyseWorker
from mava.systems.jax.parameter_client import ParameterClient
from mava.utils.tree_utils import index_stacked_tree

# TODO WORK IN PROGRESS


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
class ReanalyseDatasetConfig:
    reanalyse_sample_batch_size: int = 256
    reanalyse_max_in_flight_samples_per_worker: int = 512
    reanalyse_num_workers_per_iterator: int = -1
    reanalyse_max_samples_per_stream: int = -1
    reanalyse_rate_limiter_timeout_ms: int = -1
    reanalyse_get_signature_timeout_secs: Optional[int] = None


class ReanalyseDataset(ReanalyseComponent):
    def __init__(
        self, config: ReanalyseDatasetConfig = ReanalyseDatasetConfig()
    ) -> None:
        self.config = config

    def on_reanalyse_worker_init_start(self, reanalyse_worker: ReanalyseWorker) -> None:

        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=reanalyse_worker.store.data_server_client.server_address,
            table=reanalyse_worker.store.trainer_id,
            max_in_flight_samples_per_worker=2
            * self.config.reanalyse_sample_batch_size,
            num_workers_per_iterator=self.config.reanalyse_num_workers_per_iterator,
            max_samples_per_stream=self.config.reanalyse_max_samples_per_stream,
            rate_limiter_timeout_ms=self.config.reanalyse_rate_limiter_timeout_ms,
            get_signature_timeout_secs=self.config.reanalyse_get_signature_timeout_secs,
        )

        # Add batch dimension.
        dataset = dataset.batch(
            self.config.reanalyse_sample_batch_size, drop_remainder=True
        )
        reanalyse_worker.store.reanalyse_sample_batch_size = (
            self.config.reanalyse_sample_batch_size
        )

        reanalyse_worker.store.dataset_iterator = dataset.as_numpy_iterator()

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "reanalyse_dataset"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """
        Optional class which specifies the
        dataclass/config object for the component.
        """
        return ReanalyseDatasetConfig


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

    def on_reanalyse_worker_step_start(self, reanalyse_worker: ReanalyseWorker) -> None:
        reanalyse_worker.store.reanalyse_parameter_client.get_async()

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
            updated_search_policies[net_key] = search_output.action_weights
            updated_search_values[net_key] = search_output.search_tree.node_values[:, 0]
            updated_predicted_root_values[net_key] = predicted_root_value

        return (
            updated_search_policies,
            updated_search_values,
            updated_predicted_root_values,
        )

    def on_reanalyse_worker_step(self, reanalyse_worker: ReanalyseWorker) -> None:
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
        return
        with reanalyse_worker.store.data_server_client.trajectory_writer(
            num_keep_alive_refs=3
        ) as writer:
            for batch_index in range(
                reanalyse_worker.store.reanalyse_sample_batch_size
            ):
                for sequence_index in range(reanalyse_worker.store.sequence_length):
                    observation = index_stacked_tree(
                        sample.data.observations, (batch_index, sequence_index)
                    )
                    action = index_stacked_tree(
                        sample.data.actions, (batch_index, sequence_index)
                    )
                    reward = index_stacked_tree(
                        sample.data.rewards, (batch_index, sequence_index)
                    )
                    discount = index_stacked_tree(
                        sample.data.discounts, (batch_index, sequence_index)
                    )
                    extras = index_stacked_tree(
                        sample.data.extras, (batch_index, sequence_index)
                    )
                    for agent in extras["policy_info"].keys():
                        extras["policy_info"][agent][
                            "search_policies"
                        ] = updated_search_policies[agent]
                        extras["policy_info"][agent][
                            "search_values"
                        ] = updated_search_values[agent]
                        extras["policy_info"][agent][
                            "predicted_values"
                        ] = updated_predicted_root_values[agent]

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
