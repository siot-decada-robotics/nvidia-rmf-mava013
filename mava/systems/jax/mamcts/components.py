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

"""Custom components for MAMCTS system."""
from dataclasses import dataclass

import jax.numpy as jnp
from dm_env import specs

from mava.components.jax import Component
from mava.core_jax import SystemBuilder


@dataclass
class ExtraSearchPolicySpecConfig:
    pass


class ExtraSearchPolicySpec(Component):
    def __init__(
        self,
        config: ExtraSearchPolicySpecConfig = ExtraSearchPolicySpecConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """[summary]"""

        agent_specs = builder.store.environment_spec.get_agent_specs()

        builder.store.extras_spec = {"policy_info": {}}

        for agent, spec in agent_specs.items():

            # Make dummy specs
            builder.store.extras_spec["policy_info"][agent] = {
                "search_policies": jnp.ones(
                    shape=(spec.actions.num_values,), dtype=jnp.float32
                ),
            }

        # Add the networks keys to extras.
        int_spec = specs.DiscreteArray(len(builder.store.unique_net_keys))
        agents = builder.store.environment_spec.get_agent_ids()
        net_spec = {"network_keys": {agent: int_spec for agent in agents}}
        builder.store.extras_spec.update(net_spec)

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "extras_search_policy"

    @staticmethod
    def config_class():
        return ExtraSearchPolicySpecConfig


@dataclass
class ExtraLearnedSearchPolicySpecConfig:
    history_size: int = 10


class ExtraLearnedSearchPolicySpec(Component):
    def __init__(
        self,
        config: ExtraLearnedSearchPolicySpecConfig = ExtraLearnedSearchPolicySpecConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        builder.store.history_size = self.config.history_size

        agent_specs = builder.store.environment_spec.get_agent_specs()

        builder.store.extras_spec = {"policy_info": {}}

        for agent, spec in agent_specs.items():

            # Make dummy specs
            builder.store.extras_spec["policy_info"][agent] = {
                "search_policies": jnp.ones(
                    shape=(spec.actions.num_values,), dtype=jnp.float32
                ),
                "search_values": jnp.ones(shape=(), dtype=jnp.float32),
                "observation_history": jnp.ones(
                    shape=(
                        *spec.observations.observation.shape,
                        self.config.history_size,
                    ),
                    dtype=spec.observations.observation.dtype,
                ),
            }

        # Add the networks keys to extras.
        int_spec = specs.DiscreteArray(len(builder.store.unique_net_keys))
        agents = builder.store.environment_spec.get_agent_ids()
        net_spec = {"network_keys": {agent: int_spec for agent in agents}}
        builder.store.extras_spec.update(net_spec)

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "extras_search_policy"

    @staticmethod
    def config_class():
        return ExtraLearnedSearchPolicySpecConfig
