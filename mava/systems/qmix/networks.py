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

"""Jax IPPO system networks."""
from typing import Dict

import haiku as hk
import jax
from acme.jax import utils

from mava import specs as mava_specs
from mava.systems.qmix.qmix_network import MixingNetwork


def make_mixing_network(
    environment_spec: mava_specs.MAEnvironmentSpec,
    base_key: jax.random.PRNGKey,
    agent_net_keys: Dict[str, str],
    hidden_dim: int = 64,
    output_dim: int = 32,
) -> MixingNetwork:
    num_agents = len(agent_net_keys)

    @hk.without_apply_rng
    @hk.transform
    def hyper_w1_net(env_states):
        net = hk.Sequential(
            [hk.Linear(hidden_dim), jax.nn.relu, hk.Linear(output_dim * num_agents)]
        )
        return net(env_states)

    @hk.without_apply_rng
    @hk.transform
    def hyper_w2_net(env_states):
        net = hk.Sequential([hk.Linear(hidden_dim), jax.nn.relu, hk.Linear(output_dim)])
        return net(env_states)

    @hk.without_apply_rng
    @hk.transform
    def hyper_b1_net(env_states):
        return hk.Linear(output_dim)(env_states)

    @hk.without_apply_rng
    @hk.transform
    def hyper_b2_net(env_states):
        net = hk.Sequential([hk.Linear(output_dim), jax.nn.relu, hk.Linear(1)])
        return net(env_states)

    dummy_env_state = utils.zeros_like(environment_spec.get_extras_specs()["s_t"])

    w1_key, w2_key, b1_key, b2_key = jax.random.split(base_key, 4)

    hyper_w1_params = hyper_w1_net.init(w1_key, dummy_env_state)  # type: ignore
    hyper_w2_params = hyper_w2_net.init(w2_key, dummy_env_state)  # type: ignore
    hyper_b1_params = hyper_b1_net.init(b1_key, dummy_env_state)  # type: ignore
    hyper_b2_params = hyper_b2_net.init(b2_key, dummy_env_state)  # type: ignore

    return MixingNetwork(
        hyper_w1_net=hyper_w1_net,
        hyper_w1_params=hyper_w1_params,
        hyper_w2_net=hyper_w2_net,
        hyper_w2_params=hyper_w2_params,
        hyper_b1_net=hyper_b1_net,
        hyper_b1_params=hyper_b1_params,
        hyper_b2_net=hyper_b2_net,
        hyper_b2_params=hyper_b2_params,
        output_dim=output_dim,
    )
