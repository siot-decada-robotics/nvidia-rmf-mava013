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

"""Custom components for IPPO system."""
from types import SimpleNamespace

import numpy as np

from mava.components.building import ExtrasSpec
from mava.core_jax import SystemBuilder


class ExtrasLogProbSpec(ExtrasSpec):
    def __init__(
        self,
        config: SimpleNamespace = SimpleNamespace(),
    ):
        """Class that adds log probs to the extras spec

        Args:
            config : SimpleNamespace
        """
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Create extra specs after builder has been initialised

        Args:
            builder: SystemBuilder

        Returns:
            None.

        """
        agent_specs = builder.store.ma_environment_spec.get_agent_environment_specs()
        builder.store.extras_spec = {"policy_info": {}}

        for agent, _ in agent_specs.items():
            # Make dummy log_probs
            builder.store.extras_spec["policy_info"][agent] = np.ones(
                shape=(), dtype=np.float32
            )

        # Add the networks keys to extras.
        net_spec = self.get_network_keys(
            builder.store.unique_net_keys,
            builder.store.ma_environment_spec.get_agent_ids(),
        )
        builder.store.extras_spec.update(net_spec)

        # Get the policy state specs
        networks = builder.store.network_factory()
        net_states = {}
        for agent_key, network_key in builder.store.agent_net_keys.items():
            init_state = networks[network_key].get_init_state()
            if init_state is not None:
                net_states[agent_key] = init_state

        if net_states:
            net_spec = {"policy_states": net_states}

        builder.store.extras_spec.update(net_spec)
