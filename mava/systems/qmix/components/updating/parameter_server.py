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
from typing import Any, Dict, List, Sequence, Type, Union

import numpy as np

from mava.callbacks import Callback
from mava.components.building.networks import Networks
from mava.components.component import Component
from mava.core_jax import SystemParameterServer
from mava.utils.lp_utils import termination_fn
from mava.components.updating.parameter_server import (
    DefaultParameterServer,
    ParameterServerConfig,
)


class QmixParameterServer(DefaultParameterServer):
    def __init__(
        self,
        config: ParameterServerConfig = ParameterServerConfig(),
    ) -> None:
        """Default Mava parameter server.

        Registers count parameters and network params for tracking.
        Handles the getting, setting, and adding of parameters.

        Args:
            config: ParameterServerConfig.
        """
        self.config = config

    def on_parameter_server_init_start(self, server: SystemParameterServer) -> None:
        """Register parameters and network params to track.

        Args:
            server: SystemParameterServer.
        """
        # TODO (sasha): I shouldn't have to rewrite all this code just to change this one line
        networks, _ = server.store.network_factory()

        # Create parameters
        server.store.parameters = {
            "trainer_steps": np.zeros(1, dtype=np.int32),
            "trainer_walltime": np.zeros(1, dtype=np.float32),
            "evaluator_steps": np.zeros(1, dtype=np.int32),
            "evaluator_episodes": np.zeros(1, dtype=np.int32),
            "executor_episodes": np.zeros(1, dtype=np.int32),
            "executor_steps": np.zeros(1, dtype=np.int32),
        }
        # Network parameters
        for agent_net_key in networks.keys():
            server.store.parameters[f"policy_network-{agent_net_key}"] = networks[
                agent_net_key
            ].policy_params
            server.store.parameters[
                f"policy_opt_state-{agent_net_key}"
            ] = server.store.policy_opt_states[agent_net_key]

        server.store.experiment_path = self.config.experiment_path

        # Interrupt the system in case evaluator or trainer failed
        server.store.parameters["evaluator_or_trainer_failed"] = False

        # Interrupt the system in case all the executors failed
        server.store.parameters["num_executor_failed"] = 0
