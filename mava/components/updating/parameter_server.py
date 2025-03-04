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

"""Parameter server Component for Mava systems."""
import abc
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import numpy as np
from chex import Array

from mava.callbacks import Callback
from mava.components.building.networks import Networks
from mava.components.component import Component
from mava.core_jax import SystemParameterServer
from mava.utils.lp_utils import termination_fn


@dataclass
class ParameterServerConfig:
    non_blocking_sleep_seconds: int = 10
    experiment_path: str = "~/mava/"
    json_path: Optional[str] = None


class ParameterServer(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: ParameterServerConfig = ParameterServerConfig(),
    ) -> None:
        """Component defining hooks to override when creating a parameter server."""
        self.config = config

    @abc.abstractmethod
    def on_parameter_server_init_start(self, server: SystemParameterServer) -> None:
        """Register parameters and network params to track."""
        pass

    # Get
    @abc.abstractmethod
    def on_parameter_server_get_parameters(self, server: SystemParameterServer) -> None:
        """Fetch the parameters from the server specified in the store."""
        pass

    # Set
    @abc.abstractmethod
    def on_parameter_server_set_parameters(self, server: SystemParameterServer) -> None:
        """Set the parameters in the server to the values specified in the store."""
        pass

    # Add
    @abc.abstractmethod
    def on_parameter_server_add_to_parameters(
        self, server: SystemParameterServer
    ) -> None:
        """Increment the server parameters by the amount specified in the store."""
        pass

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "parameter_server"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        Networks required to set up server.store.network_factory.

        Returns:
            List of required component classes.
        """
        return [Networks]


class DefaultParameterServer(ParameterServer):
    def __init__(
        self,
        config: ParameterServerConfig = ParameterServerConfig(),
    ) -> None:
        """Default Mava parameter server.

        Registers count parameters and network params for tracking.
        Handles the getting, setting, and adding of parameters.

        Args:
            config: ParameterServerConfig.
            calculate_absolute_metric: Flag to stop terminating the
        system before the calculation of the absolute metric
        """
        self.config = config
        self.calculate_absolute_metric = False

    def on_parameter_server_init_start(self, server: SystemParameterServer) -> None:
        """Register parameters and network params to track.

        Args:
            server: SystemParameterServer.
        """
        networks = server.store.network_factory()

        # Store net_keys
        server.store.agents_net_keys = list(networks.keys())

        # Create parameters
        server.store.parameters = {
            "trainer_steps": np.zeros(1, dtype=np.int32),
            "trainer_walltime": np.zeros(1, dtype=np.float32),
            "evaluator_steps": np.zeros(1, dtype=np.int32),
            "evaluator_episodes": np.zeros(1, dtype=np.int32),
            "executor_episodes": np.zeros(1, dtype=np.int32),
            "executor_steps": np.zeros(1, dtype=np.int32),
        }
        server.store.parameters.update(
            self._get_network_parameters(server.store, networks)
        )

        server.store.experiment_path = self.config.experiment_path

        # Interrupt the system flag
        server.store.parameters["terminate"] = False

        # Interrupt the system in case all the executors failed
        server.store.parameters["num_executor_failed"] = 0

    # Get
    def on_parameter_server_get_parameters(self, server: SystemParameterServer) -> None:
        """Fetch the parameters from the server specified in the store.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        # server.store._param_names set by Parameter Server
        names: Union[str, Sequence[str]] = server.store._param_names

        if type(names) == str:
            get_params = server.store.parameters[names]  # type: ignore
        else:
            get_params = {}
            for var_key in names:
                get_params[var_key] = server.store.parameters[var_key]
        server.store.get_parameters = get_params

        # Interrupt the system flag
        if server.store.parameters["terminate"]:
            termination_fn(server)

        # Interrupt the system in case all the executors failed
        if server.store.num_executors == server.store.parameters["num_executor_failed"]:
            termination_fn(server)

    # Set
    def on_parameter_server_set_parameters(self, server: SystemParameterServer) -> None:
        """Set the parameters in the server to the values specified in the store.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        # server.store._set_params set by Parameter Server
        params: Dict[str, Any] = server.store._set_params
        names = params.keys()

        for var_key in names:
            assert var_key in server.store.parameters
            if type(server.store.parameters[var_key]) == tuple:
                raise NotImplementedError
                # # Loop through tuple
                # for var_i in range(len(server.store.parameters[var_key])):
                #     server.store.parameters[var_key][var_i].assign(params[var_key][var_i])
            else:
                server.store.parameters[var_key] = params[var_key]

    # Add
    def on_parameter_server_add_to_parameters(
        self, server: SystemParameterServer
    ) -> None:
        """Increment the server parameters by the amount specified in the store.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        # server.store._add_to_params set by Parameter Server
        params: Dict[str, Any] = server.store._add_to_params
        names = params.keys()

        for var_key in names:
            assert var_key in server.store.parameters
            server.store.parameters[var_key] += params[var_key]

    def _get_network_parameters(
        self, store: SimpleNamespace, networks: Dict
    ) -> Dict[str, Array]:
        parameters = {}
        for agent_net_key in networks.keys():
            agent_net = networks[agent_net_key]
            parameters[f"policy_network-{agent_net_key}"] = agent_net.policy_params
            parameters[f"policy_opt_state-{agent_net_key}"] = store.policy_opt_states[
                agent_net_key
            ]

        return parameters


class ActorCriticParameterServer(DefaultParameterServer):
    def _get_network_parameters(
        self, store: SimpleNamespace, networks: Dict
    ) -> Dict[str, Array]:
        parameters = {}
        for agent_net_key in networks.keys():
            agent_net = networks[agent_net_key]
            parameters[f"policy_network-{agent_net_key}"] = agent_net.policy_params
            parameters[f"critic_network-{agent_net_key}"] = agent_net.critic_params
            parameters[f"policy_opt_state-{agent_net_key}"] = store.policy_opt_states[
                agent_net_key
            ]
            parameters[f"critic_opt_state-{agent_net_key}"] = store.critic_opt_states[
                agent_net_key
            ]

        return parameters
