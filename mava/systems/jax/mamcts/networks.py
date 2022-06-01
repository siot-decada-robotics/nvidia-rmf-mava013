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

"""Jax MAMCTS system networks."""
import dataclasses
import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from dm_env import specs as dm_specs
from jax import jit

from mava import specs as mava_specs
from mava.systems.jax.mamcts.learned_model_utils import (
    inv_value_transform,
    logits_to_scalar,
    normalise_encoded_state,
)
from mava.systems.jax.mamcts.network_components import (
    DynamicsNet,
    PredictionNet,
    RepresentationNet,
)

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


class PredictionNetworks:
    """TODO: Add description here."""

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
        num_bins: int,
    ) -> None:
        """TODO: Add description here."""
        self.network = network
        self.params = params
        self._num_bins = num_bins

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """TODO: Add description here."""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            logits, value_logits = self.network.apply(params, observations)

            return logits, value_logits

        self.forward_fn = forward_fn

    def get_logits(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        logits, _ = self.forward_fn(self.params, observations)

        return logits

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        _, value_logits = self.forward_fn(self.params, observations)
        value = logits_to_scalar(value_logits)
        value = inv_value_transform(value)
        return value

    def get_logits_and_value(
        self, observations: networks_lib.Observation
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """TODO: Add description here."""
        logits, value_logits = self.forward_fn(self.params, observations)
        value = logits_to_scalar(value_logits)
        value = inv_value_transform(value)
        return logits, value

def make_environment_model_prediction_network(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    num_bins: int,
    output_init_scale: float = 1.0,
    use_v2: bool = True,
    fully_connected=False,
    prediction_layers=(256, 256, 256),
) -> PredictionNetworks:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    if fully_connected:

        def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:

            base_network = networks_lib.LayerNormMLP(prediction_layers)
            policy_network = hk.Linear(num_actions)
            value_network = hk.Linear(num_bins)

            inputs = base_network(inputs)

            return policy_network(inputs), value_network(inputs)

        size = environment_spec.observations.observation.shape[0]
        for s in environment_spec.observations.observation.shape[1:]:
            size += size * s

        dummy_obs = jnp.zeros(size)
    else:

        def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
            embedding_network = hk.Embed(128,8)
            inputs = embedding_network(inputs.astype(int))
            policy_value_network = PredictionNet(
                num_actions=num_actions,
                num_bins=num_bins,
                output_init_scale=output_init_scale,
                use_v2=use_v2,
            )
            return policy_value_network(inputs)

        dummy_obs = jnp.zeros(environment_spec.observations.observation.shape)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    network_key, key = jax.random.split(key)

    dummy_obs = utils.add_batch_dim(dummy_obs)

    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return PredictionNetworks(network=forward_fn, params=params, num_bins=num_bins)

def make_prediction_network(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    num_bins: int,
    output_init_scale: float = 1.0,
    use_v2: bool = True,
    representation_net=None,
    dynamics_net=None,
    fully_connected=False,
    prediction_layers=(256, 256, 256),
) -> PredictionNetworks:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    if fully_connected:

        def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:

            base_network = networks_lib.LayerNormMLP(prediction_layers)
            policy_network = hk.Linear(num_actions)
            value_network = hk.Linear(num_bins)

            inputs = base_network(inputs)

            return policy_network(inputs), value_network(inputs)

        size = environment_spec.observations.observation.shape[0]
        for s in environment_spec.observations.observation.shape[1:]:
            size += size * s

        dummy_obs = jnp.zeros(
            ((size + num_actions) * int(representation_net.observation_history_size),)
        )
    else:

        def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:

            policy_value_network = PredictionNet(
                num_actions=num_actions,
                num_bins=num_bins,
                output_init_scale=output_init_scale,
                use_v2=use_v2,
            )
            return policy_value_network(inputs)

        dummy_obs = dummy_obs = jnp.zeros(
            (
                *environment_spec.observations.observation.shape,
                int(representation_net.observation_history_size * 2),
            )
        )

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    network_key, key = jax.random.split(key)

    dummy_obs = utils.add_batch_dim(dummy_obs)

    dummy_root_embedding = representation_net.forward_fn(
        representation_net.params, dummy_obs
    )
    dummy_action = jnp.ones((), int)
    dummy_action = utils.add_batch_dim(dummy_action)

    dummy_embedding, _ = dynamics_net.forward_fn(
        dynamics_net.params, dummy_root_embedding, dummy_action
    )

    params = forward_fn.init(network_key, dummy_embedding)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return PredictionNetworks(network=forward_fn, params=params, num_bins=num_bins)

def make_environment_model_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    rng_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    num_bins: int = 601,
    output_init_scale: float = 1.0,
    use_v2: bool = True,
) -> Dict[str, Any]:
    """Description here"""

    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_environment_model_prediction_network(
            specs[net_key],
            key=rng_key,
            num_bins=num_bins,
            output_init_scale=output_init_scale,
            use_v2=use_v2,
        )

    return {
        "networks": networks,
    }


@dataclasses.dataclass
class RepresentationNetwork:
    """TODO: Add description here."""

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
        observation_history_size: int,
    ) -> None:
        """TODO: Add description here."""
        self.network = network
        self.params = params
        self.observation_history_size = observation_history_size

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            observation_history: networks_lib.Observation,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """TODO: Add description here."""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            root_embedding = self.network.apply(params, observation_history)
            root_embedding = normalise_encoded_state(root_embedding)
            return root_embedding

        self.forward_fn = forward_fn


def make_representation_network(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    channels: int = 64,
    use_v2: bool = True,
    observation_history_size=10,
    fully_connected=False,
    encoding_size=100,
    representation_layers=(256, 256, 256),
) -> RepresentationNetwork:
    """TODO: Add description here."""
    num_actions = environment_spec.actions.num_values

    if fully_connected:

        def forward_fn(
            observation_history: jnp.ndarray,
        ) -> networks_lib.FeedForwardNetwork:

            representation_network = networks_lib.LayerNormMLP(
                (*representation_layers, encoding_size)
            )
            initial_state = representation_network(observation_history)
            return initial_state

        size = environment_spec.observations.observation.shape[0]
        for s in environment_spec.observations.observation.shape[1:]:
            size += size * s

        dummy_obs = jnp.zeros(((size + num_actions) * int(observation_history_size),))

    else:

        def forward_fn(
            observation_history: jnp.ndarray,
        ) -> networks_lib.FeedForwardNetwork:

            representation_network = RepresentationNet(channels=channels, use_v2=use_v2)
            initial_state = representation_network(observation_history)
            return initial_state

        dummy_obs = dummy_obs = jnp.zeros(
            (
                *environment_spec.observations.observation.shape,
                int(observation_history_size * 2),
            )
        )

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return RepresentationNetwork(
        network=forward_fn,
        params=params,
        observation_history_size=observation_history_size,
    )


@dataclasses.dataclass
class DynamicsNetwork:
    """TODO: Add description here."""

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
    ) -> None:
        """TODO: Add description here."""
        self.network = network
        self.params = params

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            previous_embedding: networks_lib.NetworkOutput,
            action: networks_lib.Action,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """TODO: Add description here."""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            embedding, reward_logits = self.network.apply(
                params, previous_embedding, action
            )
            embedding = normalise_encoded_state(embedding)
            return embedding, reward_logits

        self.forward_fn = forward_fn


def make_dynamics_network(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    num_bins: int = 601,
    output_init_scale: float = 1.0,
    use_v2: bool = True,
    representation_net=None,
    fully_connected=False,
    encoding_size=100,
    dynamics_layers=(256, 256, 256),
) -> DynamicsNetwork:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    if fully_connected:

        def forward_fn(
            prev_embedding: jnp.ndarray, action: jnp.ndarray
        ) -> networks_lib.FeedForwardNetwork:
            action_one_hot = hk.one_hot(action, num_actions)
            inputs = jnp.concatenate([prev_embedding, action_one_hot], axis=-1)
            dynamics_network = networks_lib.LayerNormMLP(
                (*dynamics_layers, encoding_size)
            )
            rewards_network = networks_lib.LayerNormMLP((*dynamics_layers, num_bins))

            next_state = dynamics_network(inputs)

            rewards = rewards_network(inputs)

            return next_state, rewards

        size = environment_spec.observations.observation.shape[0]
        for s in environment_spec.observations.observation.shape[1:]:
            size += size * s

        dummy_obs = jnp.zeros(
            ((size + num_actions) * int(representation_net.observation_history_size),)
        )

    else:

        def forward_fn(
            prev_embedding: jnp.ndarray, action: jnp.ndarray
        ) -> networks_lib.FeedForwardNetwork:

            dynamics_network = DynamicsNet(
                num_bins=num_bins,
                output_init_scale=output_init_scale,
                use_v2=use_v2,
                num_actions=num_actions,
            )

            next_state, reward_logits = dynamics_network(prev_embedding, action)

            return next_state, reward_logits

        dummy_obs = dummy_obs = jnp.zeros(
            (
                *environment_spec.observations.observation.shape,
                int(representation_net.observation_history_size * 2),
            )
        )
        
    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = utils.add_batch_dim(dummy_obs)

    dummy_root_embedding = representation_net.forward_fn(
        representation_net.params, dummy_obs
    )
    dummy_action = jnp.ones((), int)
    dummy_action = utils.add_batch_dim(dummy_action)

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_root_embedding, dummy_action)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return DynamicsNetwork(
        network=forward_fn,
        params=params,
    )


class LearnedModelNetworks:
    """TODO: Add description here."""

    def __init__(
        self,
        prediction_network: PredictionNetworks,
        dynamics_network: DynamicsNetwork,
        representation_network: RepresentationNetwork,
    ) -> None:
        """TODO: Add description here."""
        self._num_bins = prediction_network._num_bins

        self.prediction_network = prediction_network
        self.dynamics_network = dynamics_network
        self.representation_network = representation_network
        self.params = {
            "prediction": self.prediction_network.params,
            "dynamics": self.dynamics_network.params,
            "representation": self.representation_network.params,
        }

        self.prediction_fn = self.prediction_network.forward_fn
        self.dynamics_fn = self.dynamics_network.forward_fn
        self.representation_fn = self.representation_network.forward_fn

    def update_inner_params(self):
        self.prediction_network.params = self.params["prediction"]
        self.dynamics_network.params = self.params["dynamics"]
        self.representation_network.params = self.params["representation"]

    def get_policy_value(self, embedding):
        logits, value_logits = self.prediction_fn(self.params["prediction"], embedding)
        value = logits_to_scalar(value_logits)
        value = inv_value_transform(value)
        return logits, value

    def get_next_state_and_reward(self, embedding, action):
        next_state, reward_logits = self.dynamics_fn(
            self.params["dynamics"], embedding, action
        )
        reward = logits_to_scalar(reward_logits)
        reward = inv_value_transform(reward)
        return next_state, reward

    def get_root_state(self, observation_history):
        return self.representation_fn(
            self.params["representation"], observation_history
        )

    # def normalise_encoded_state(self, state_embedding, epsilon = 1e-5):
    #      # Scale encoded state between [0, 1] (See paper appendix Training)
    #     min_state_embedding = jnp.min(state_embedding,axis=1,keepdims=True)
    #     max_state_embedding = jnp.max(state_embedding,axis=1,keepdims=True)
    #     scale_state_embedding = max_state_embedding - min_state_embedding
    #     scale_state_embedding[scale_state_embedding < epsilon] += epsilon
    #     state_embedding_normalized = (
    #         state_embedding - min_state_embedding
    #     ) / scale_state_embedding
    #     return state_embedding_normalized


def make_learned_model_networks(
    spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    channels: int = 64,
    num_bins: int = 601,
    output_init_scale: float = 1.0,
    use_v2: bool = True,
    observation_history_size: int = 10,
    fully_connected=False,
    encoding_size=100,
    representation_layers=(256, 256, 256),
    dynamics_layers=(256, 256, 256),
    prediction_layers=(256, 256, 256),
) -> LearnedModelNetworks:
    """TODO: Add description here."""

    representation_net = make_representation_network(
        environment_spec=spec,
        key=key,
        channels=channels,
        use_v2=use_v2,
        observation_history_size=observation_history_size,
        fully_connected=fully_connected,
        encoding_size=encoding_size,
        representation_layers=representation_layers,
    )

    dynamics_net = make_dynamics_network(
        environment_spec=spec,
        key=key,
        num_bins=num_bins,
        output_init_scale=output_init_scale,
        use_v2=use_v2,
        representation_net=representation_net,
        fully_connected=fully_connected,
        encoding_size=encoding_size,
        dynamics_layers=dynamics_layers,
    )

    prediction_net = make_prediction_network(
        environment_spec=spec,
        key=key,
        num_bins=num_bins,
        output_init_scale=output_init_scale,
        representation_net=representation_net,
        dynamics_net=dynamics_net,
        fully_connected=fully_connected,
        prediction_layers=prediction_layers,
    )

    return LearnedModelNetworks(
        prediction_network=prediction_net,
        dynamics_network=dynamics_net,
        representation_network=representation_net,
    )


def make_default_learned_model_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    rng_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    channels: int = 64,
    num_bins: int = 601,
    output_init_scale: float = 1.0,
    use_v2: bool = True,
    observation_history_size: int = 10,
    fully_connected=False,
    encoding_size=100,
    representation_layers=(256, 256, 256),
    dynamics_layers=(256, 256, 256),
    prediction_layers=(256, 256, 256),
) -> Dict[str, Any]:
    """Description here"""

    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_learned_model_networks(
            specs[net_key],
            key=rng_key,
            channels=channels,
            num_bins=num_bins,
            output_init_scale=output_init_scale,
            use_v2=use_v2,
            observation_history_size=observation_history_size,
            fully_connected=fully_connected,
            encoding_size=encoding_size,
            representation_layers=representation_layers,
            dynamics_layers=dynamics_layers,
            prediction_layers=prediction_layers,
        )

    return {
        "networks": networks,
    }
