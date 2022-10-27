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

"""Jax MADQN system networks."""
from typing import Any, Dict, List, Optional, Sequence

import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from dm_env import specs as dm_specs

from mava import specs as mava_specs
from mava.systems.madqn.DQNNetworks import DQNNetworks

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray


class C51DuellingMLP(hk.Module):
    """A Duelling MLP Q-network."""

    def __init__(
        self,
        num_actions: int,
        hidden_sizes: Sequence[int],
        v_min: float,
        v_max: float,
        w_init: Optional[hk.initializers.Initializer] = None,
        num_atoms: int = 51,
    ):
        super().__init__(name="duelling_q_network")

        self._value_mlp = hk.nets.MLP([*hidden_sizes, num_atoms], w_init=w_init)
        self._advantage_mlp = hk.nets.MLP(
            [*hidden_sizes, num_actions * num_atoms], w_init=w_init
        )
        self.num_actions = num_actions
        self._num_atoms = 51
        self._atoms = jnp.linspace(v_min, v_max, self._num_atoms)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the duelling network.

        Args:
          inputs: 2-D tensor of shape [batch_size, embedding_size].

        Returns:
          q_values: 2-D tensor of action values of shape [batch_size, num_actions]
        """

        # Compute value & advantage for dueling.
        value = self._value_mlp(inputs)  # [B, 1]
        advantages = self._advantage_mlp(inputs)  # [B, A]

        # Advantages have zero mean.
        # dueling part
        advantages -= jnp.mean(advantages, axis=-1, keepdims=True)  # [B, A]
        advantages = advantages.reshape(-1, self._num_atoms, self.num_actions)
        value = jax.numpy.expand_dims(value, axis=-1)
        q_logits = value + advantages  # [B, A]

        # Distributional Part
        q_logits = q_logits.reshape(-1, self.num_actions, self._num_atoms)
        q_dist = jax.nn.softmax(q_logits)
        # print(q_dist.shape)
        # print(self._atoms.shape)
        # exit()
        q_values = jnp.sum(q_dist * self._atoms, axis=2)
        q_values = jax.lax.stop_gradient(q_values)
        return q_values, q_logits, self._atoms


def make_DQN_network(
    network: networks_lib.FeedForwardNetwork, params: Dict[str, jnp.ndarray]
) -> DQNNetworks:
    """TODO: Add description here."""
    return DQNNetworks(network=network, params=params)


def make_networks(
    spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    v_max: int,
    v_min: int,
    policy_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    num_atoms: int = 51,
) -> DQNNetworks:
    """TODO: Add description here."""
    if isinstance(spec.actions, specs.DiscreteArray):
        return make_discrete_networks(
            environment_spec=spec,
            key=key,
            policy_layer_sizes=policy_layer_sizes,
            v_max=v_max,
            v_min=v_min,
            num_atoms=num_atoms,
        )
    else:
        raise NotImplementedError("Only discrete actions are implemented for MADQN.")


def make_discrete_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    v_max: int,
    v_min: int,
    num_atoms: int = 51,
) -> DQNNetworks:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    # TODO (dries): Investigate if one forward_fn function is slower
    # than having a policy_fn and critic_fn. Maybe jit solves
    # this issue. Having one function makes obs network calculations
    # easier.
    def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        policy_value_network = hk.Sequential(
            [
                utils.batch_concat,
                networks_lib.LayerNormMLP(policy_layer_sizes, activate_final=True),
                # networks_lib.DiscreteValued(num_actions)
                C51DuellingMLP(
                    num_actions,
                    policy_layer_sizes,
                    v_max=v_max,
                    v_min=v_min,
                    num_atoms=num_atoms,
                ),
            ]
        )
        return policy_value_network(inputs)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create DQNNetworks to add functionality required by the agent.
    return make_DQN_network(network=forward_fn, params=params)


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    base_key: List[int],
    v_max: int,
    v_min: int,
    net_spec_keys: Dict[str, str] = {},
    policy_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    num_atoms: int = 51,
) -> Dict[str, DQNNetworks]:
    """Description here"""

    # Create agent_type specs.
    specs = environment_spec.get_agent_environment_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_networks(
            specs[net_key],
            key=base_key,
            policy_layer_sizes=policy_layer_sizes,
            v_max=v_max,
            v_min=v_min,
            num_atoms=num_atoms,
        )

    return networks
