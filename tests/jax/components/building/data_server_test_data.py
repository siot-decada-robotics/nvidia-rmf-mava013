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

"""Tests for config class for Jax-based Mava systems"""


from typing import Dict, List

import reverb

from mava.components.jax.building import adders
from mava.components.jax.building.base import SystemInit
from mava.components.jax.building.data_server import (
    OffPolicyDataServer,
    OnPolicyDataServer,
)
from mava.components.jax.building.environments import EnvironmentSpec
from mava.components.jax.building.reverb import (
    MinSizeRateLimiter,
    SampleToInsertRateLimiter,
)
from tests.jax.mocks import (  # MockedEnvSpec,
    MockOffPolicyDataServer,
    MockOnPolicyDataServer,
    make_fake_environment_factory,
)

transition_adder_data_server_test_cases: List[Dict] = [
    {
        "component": {
            "environment_spec": EnvironmentSpec,
            "system_init": SystemInit,
            "rate_limiter": MinSizeRateLimiter,
            "data_server_adder_signature": adders.ParallelTransitionAdderSignature,
            "data_server": MockOffPolicyDataServer,
        },
        "system_config": {
            "sampler": reverb.selectors.Uniform(),
            "remover": reverb.selectors.Lifo(),
            "max_size": 500,
            "min_data_server_size": 100,
            "max_times_sampled": 12,
            "environment_factory": make_fake_environment_factory(),
        },
    },
    {
        "component": {
            "environment_spec": EnvironmentSpec,
            "system_init": SystemInit,
            "data_server_adder_signature": adders.ParallelTransitionAdderSignature,
            "data_server": MockOnPolicyDataServer,
        },
        "system_config": {
            "max_queue_size": 50,
            "environment_factory": make_fake_environment_factory(),
        },
    },
    {
        "component": {
            "environment_spec": EnvironmentSpec,
            "system_init": SystemInit,
            "rate_limiter": SampleToInsertRateLimiter,
            "data_server_adder_signature": adders.ParallelTransitionAdderSignature,
            "data_server": OffPolicyDataServer,
        },
        "system_config": {
            "sampler": reverb.selectors.MinHeap(),
            "remover": reverb.selectors.Lifo(),
            "max_size": 500,
            "min_data_server_size": 100,
            "max_times_sampled": 12,
            "environment_factory": make_fake_environment_factory(),
        },
    },
    {
        "component": {
            "environment_spec": EnvironmentSpec,
            "system_init": SystemInit,
            "data_server_adder_signature": adders.ParallelTransitionAdderSignature,
            "data_server": OnPolicyDataServer,
        },
        "system_config": {
            "max_queue_size": 50,
            "environment_factory": make_fake_environment_factory(),
        },
    },
]
