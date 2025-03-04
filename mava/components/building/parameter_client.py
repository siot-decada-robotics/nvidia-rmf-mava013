# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parameter client for system builders"""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Set, Tuple, Type

import numpy as np
from chex import Array

from mava.callbacks import Callback
from mava.components import Component
from mava.components.building.best_checkpointer import BestCheckpointer
from mava.components.normalisation.base_normalisation import BaseNormalisation
from mava.components.training.trainer import BaseTrainerInit
from mava.core_jax import SystemBuilder
from mava.systems import ParameterClient


class BaseParameterClient(Component):
    def _set_up_count_parameters(
        self, params: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Registers parameters to count and store.

        Counts trainer_steps, trainer_walltime, evaluator_steps,
        evaluator_episodes, executor_episodes, executor_steps.

        Args:
            params: Network parameters.

        Returns:
            Tuple of count parameters and network parameters.
        """
        add_params = {
            "trainer_steps": np.array(0, dtype=np.int32),
            "trainer_walltime": np.array(0, dtype=np.float32),
            "evaluator_steps": np.array(0, dtype=np.int32),
            "evaluator_episodes": np.array(0, dtype=np.int32),
            "executor_episodes": np.array(0, dtype=np.int32),
            "executor_steps": np.array(0, dtype=np.int32),
        }
        params.update(add_params)
        return list(add_params.keys()), params

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        BaseTrainerInit required to set up builder.store.networks
        and builder.store.trainer_networks.

        Returns:
            List of required component classes.
        """
        return [BaseTrainerInit]


@dataclass
class ExecutorParameterClientConfig:
    executor_parameter_update_period: int = 200


class ExecutorParameterClient(BaseParameterClient):
    def __init__(
        self,
        config: ExecutorParameterClientConfig = ExecutorParameterClientConfig(),
    ) -> None:
        """Component creates a parameter client for the executor.

        Args:
            config: ExecutorParameterClientConfig.
        """

        self.config = config

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """Create and store the executor parameter client.

        Gets network parameters from store and registers them for tracking.
        Also works for the evaluator.

        Args:
            builder: SystemBuilder.
        """
        # Create policy parameters
        params: Dict[str, Any] = {}
        # Executor does not explicitly set variables i.e. it adds to count variables
        # and hence set_keys is empty
        set_keys: List[str] = []
        get_keys: List[str] = []

        net_keys, net_params = self.get_network_parameters(builder.store)
        params.update(net_params)
        get_keys.extend(net_keys)

        # Create observations' normalisation parameters
        if builder.has(BaseNormalisation):
            params["norm_params"] = builder.store.norm_params
            get_keys.append("norm_params")

        if (
            builder.store.is_evaluator
            and builder.has(BestCheckpointer)
            and builder.store.global_config.checkpoint_best_perf
        ):
            params["best_checkpoint"] = builder.store.best_checkpoint
            set_keys.append("best_checkpoint")

        count_names, params = self._set_up_count_parameters(params=params)

        get_keys.extend(count_names)

        builder.store.executor_counts = {name: params[name] for name in count_names}

        parameter_client = None
        if builder.store.parameter_server_client:
            # Create parameter client
            parameter_client = ParameterClient(
                server=builder.store.parameter_server_client,
                parameters=params,
                multi_process=builder.store.global_config.multi_process,
                get_keys=get_keys,
                set_keys=set_keys,
                update_period=self.config.executor_parameter_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning parameters before running the environment loop.
            parameter_client.get_and_wait()

        builder.store.executor_parameter_client = parameter_client

    def get_network_parameters(
        self, store: SimpleNamespace
    ) -> Tuple[List[str], Dict[str, Array]]:
        """Returns: network keys and parameters"""
        params = {}
        net_keys = []
        for agent_net_key in store.networks.keys():
            policy_param_key = f"policy_network-{agent_net_key}"
            params[policy_param_key] = store.networks[agent_net_key].policy_params
            net_keys.append(policy_param_key)

        return net_keys, params

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "executor_parameter_client"


class ActorCriticExecutorParameterClient(ExecutorParameterClient):
    def get_network_parameters(
        self, store: SimpleNamespace
    ) -> Tuple[List[str], Dict[str, Array]]:
        """Returns: network keys and parameters"""
        params = {}
        net_keys = []
        for agent_net_key in store.networks.keys():
            policy_param_key = f"policy_network-{agent_net_key}"
            params[policy_param_key] = store.networks[agent_net_key].policy_params
            net_keys.append(policy_param_key)

            critic_param_key = f"critic_network-{agent_net_key}"
            params[critic_param_key] = store.networks[agent_net_key].critic_params
            net_keys.append(critic_param_key)

        return net_keys, params


@dataclass
class TrainerParameterClientConfig:
    trainer_parameter_update_period: int = 5


class TrainerParameterClient(BaseParameterClient):
    def __init__(
        self,
        config: TrainerParameterClientConfig = TrainerParameterClientConfig(),
    ) -> None:
        """Component creates a parameter client for the trainer.

        Args:
            config: TrainerParameterClientConfig.
        """

        self.config = config

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """Create and store the trainer parameter client.

        Gets network parameters from store and registers them for tracking.

        Args:
            builder: SystemBuilder.
        """
        # Create parameter client
        params: Dict[str, Any] = {}
        set_keys: List[str] = []
        get_keys: List[str] = []
        # TODO (dries): Only add the networks this trainer is working with.
        # Not all of them.
        trainer_networks = builder.store.trainer_networks[builder.store.trainer_id]
        get_keys, set_keys, params = self.get_network_parameters(
            builder.store, set(trainer_networks)
        )

        # Add observations' normalisation parameters
        if builder.has(BaseNormalisation):
            params["norm_params"] = builder.store.norm_params
            set_keys.append("norm_params")

        count_names, params = self._set_up_count_parameters(params=params)

        get_keys.extend(count_names)
        builder.store.trainer_counts = {name: params[name] for name in count_names}

        # Create parameter client
        parameter_client = None
        if builder.store.parameter_server_client:
            parameter_client = ParameterClient(
                server=builder.store.parameter_server_client,
                parameters=params,
                multi_process=builder.store.global_config.multi_process,
                get_keys=get_keys,
                set_keys=set_keys,
                update_period=self.config.trainer_parameter_update_period,
            )

            # Get all the initial parameters
            parameter_client.get_all_and_wait()

        builder.store.trainer_parameter_client = parameter_client

    def get_network_parameters(
        self, store: SimpleNamespace, trainer_networks: Set[str]
    ) -> Tuple[List[str], List[str], Dict[str, Array]]:
        """Gets keys for this trainers networks, other trainers networks and params"""
        params = {}
        set_keys = []
        get_keys = []
        for net_key in store.networks.keys():
            params[f"policy_network-{net_key}"] = store.networks[net_key].policy_params

            if net_key in trainer_networks:
                set_keys.append(f"policy_network-{net_key}")
            else:
                get_keys.append(f"policy_network-{net_key}")

            params[f"policy_opt_state-{net_key}"] = store.policy_opt_states[net_key]
            set_keys.append(f"policy_opt_state-{net_key}")

        return get_keys, set_keys, params

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "trainer_parameter_client"


class ActorCriticTrainerParameterClient(TrainerParameterClient):
    def get_network_parameters(
        self, store: SimpleNamespace, trainer_networks: Set[str]
    ) -> Tuple[List[str], List[str], Dict[str, Array]]:
        """Gets keys for this trainers networks, other trainers networks and params"""
        params = {}
        set_keys = []
        get_keys = []
        for net_key in store.networks.keys():
            params[f"policy_network-{net_key}"] = store.networks[net_key].policy_params
            params[f"critic_network-{net_key}"] = store.networks[net_key].critic_params

            if net_key in trainer_networks:
                set_keys.append(f"policy_network-{net_key}")
                set_keys.append(f"critic_network-{net_key}")
            else:
                get_keys.append(f"policy_network-{net_key}")
                get_keys.append(f"critic_network-{net_key}")

            params[f"policy_opt_state-{net_key}"] = store.policy_opt_states[net_key]
            params[f"critic_opt_state-{net_key}"] = store.critic_opt_states[net_key]
            set_keys.append(f"policy_opt_state-{net_key}")
            set_keys.append(f"critic_opt_state-{net_key}")

        return get_keys, set_keys, params

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "trainer_parameter_client"
