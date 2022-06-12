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

"""Jax MAMCTS and MAMU system networks."""
import dataclasses
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
    SimpleDynamicsNet,
    SimplePredictionNet,
    SimpleRepresentationNet,
)

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


def identity(x):
    return x


class PredictionNetwork:
    """Networks class used for policy and value predictions"""

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
        num_bins: int,
    ) -> None:
        """Create a PredictionNetwork object.

        Args:
            network: an actual haiku network object.
            params: the parameters for the network object.
            num_bins: the number of bins the prediction network uses.
        """
        self.network = network
        self.params = params
        self._num_bins = num_bins

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """PredictionNetwork forward function - returns the logits and value logits of an input"""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            logits, value_logits = self.network.apply(params, observations)

            return logits, value_logits

        self.forward_fn = forward_fn

    def get_logits(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """Get only the logits of an input"""
        logits, _ = self.forward_fn(self.params, observations)

        return logits

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """Get only the value of an input and convert it to a scalar value"""
        _, value_logits = self.forward_fn(self.params, observations)
        value = logits_to_scalar(value_logits)
        value = inv_value_transform(value)
        return value

    def get_logits_and_value(
        self, observations: networks_lib.Observation
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get both the logits and scalar value prediction of an input"""
        logits, value_logits = self.forward_fn(self.params, observations)
        value = logits_to_scalar(value_logits)
        value = inv_value_transform(value)
        return logits, value


@dataclasses.dataclass
class RepresentationNetwork:
    """RepresentationNetwork class for a learned model system"""

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
        observation_history_size: int,
    ) -> None:
        """Create a representation network"""
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


class MAMUNetworks:
    """MAMU Networks class - Provides functionality and encapsulates all networks for a MAMU system"""

    def __init__(
        self,
        prediction_network: PredictionNetwork,
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
        self.history_size = self.representation_network.observation_history_size

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


def make_mamcts_prediction_network(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    num_bins: int,
    base_prediction_layers: Sequence[int],
    value_prediction_layers: Sequence[int],
    policy_prediction_layers: Sequence[int],
    observation_net=utils.batch_concat,
) -> PredictionNetwork:
    """Create a prediction network for a mamcts system."""

    num_actions = environment_spec.actions.num_values

    def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        inputs = observation_net(inputs)
        policy_value_network = SimplePredictionNet(
            base_prediction_layers=base_prediction_layers,
            value_prediction_layers=value_prediction_layers,
            policy_prediction_layers=policy_prediction_layers,
            num_actions=num_actions,
            num_bins=num_bins,
        )
        return policy_value_network(inputs)

    dummy_obs = jnp.zeros(environment_spec.observations.observation.shape)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    network_key, key = jax.random.split(key)

    dummy_obs = utils.add_batch_dim(dummy_obs)

    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PredictionNetwork to add functionality required by the agent.
    return PredictionNetwork(network=forward_fn, params=params, num_bins=num_bins)


def make_default_mamcts_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    rng_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    num_bins: int = 601,
    base_prediction_layers: Sequence[int] = [256],
    value_prediction_layers: Sequence[int] = [256],
    policy_prediction_layers: Sequence[int] = [256],
    observation_net=utils.batch_concat,
) -> Dict[str, Any]:
    """Make default networks for a mamcts system"""

    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_mamcts_prediction_network(
            specs[net_key],
            key=rng_key,
            num_bins=num_bins,
            base_prediction_layers=base_prediction_layers,
            value_prediction_layers=value_prediction_layers,
            policy_prediction_layers=policy_prediction_layers,
            observation_net=observation_net,
        )

    return {
        "networks": networks,
    }


def make_mamu_prediction_network(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    num_bins: int,
    representation_net,
    dynamics_net,
    base_prediction_layers,
    value_prediction_layers,
    policy_prediction_layers,
    observation_net=utils.batch_concat,
) -> PredictionNetwork:
    """Make a mamu prediction network"""

    num_actions = environment_spec.actions.num_values

    def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        inputs = observation_net(inputs)
        policy_value_network = SimplePredictionNet(
            base_prediction_layers=base_prediction_layers,
            num_actions=num_actions,
            num_bins=num_bins,
            value_prediction_layers=value_prediction_layers,
            policy_prediction_layers=policy_prediction_layers,
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

    # Create PredictionNetwork to add functionality required by the agent.
    return PredictionNetwork(network=forward_fn, params=params, num_bins=num_bins)


def make_mamu_representation_network(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    observation_history_size: int,
    encoding_size: int,
    representation_layers: Sequence[int],
    observation_net=utils.batch_concat,
) -> RepresentationNetwork:
    """Make a mamu representation network"""

    def forward_fn(
        observation_history: jnp.ndarray,
    ) -> networks_lib.FeedForwardNetwork:
        observation_history = observation_net(observation_history)
        observation_history = jax.nn.elu(observation_history)
        representation_network = SimpleRepresentationNet(
            representation_layers=representation_layers, encoding_size=encoding_size
        )
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

    # Create RepresentationNetwork to add functionality required by the agent.
    return RepresentationNetwork(
        network=forward_fn,
        params=params,
        observation_history_size=observation_history_size,
    )


def make_mamu_dynamics_network(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    representation_net,
    num_bins: int,
    encoding_size: int,
    base_transition_layers: Sequence[int],
    dynamics_layers: Sequence[int],
    reward_layers: Sequence[int],
    observation_net,
) -> DynamicsNetwork:
    """Make a mamu dynamics network"""

    num_actions = environment_spec.actions.num_values

    def forward_fn(
        prev_embedding: jnp.ndarray, action: jnp.ndarray
    ) -> networks_lib.FeedForwardNetwork:
        prev_embedding = observation_net(prev_embedding)

        dynamics_network = SimpleDynamicsNet(
            base_transition_layers=base_transition_layers,
            dynamics_layers=dynamics_layers,
            reward_layers=reward_layers,
            num_bins=num_bins,
            num_actions=num_actions,
            encoding_size=encoding_size,
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

    # Create DynamicsNetwork to add functionality required by the agent.
    return DynamicsNetwork(
        network=forward_fn,
        params=params,
    )


def make_mamu_networks(
    spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    num_bins: int,
    observation_history_size,
    encoding_size,
    representation_layers,
    base_transition_layers,
    dynamics_layers,
    reward_layers,
    base_prediction_layers,
    value_prediction_layers,
    policy_prediction_layers,
    representation_obs_net,
    dynamics_obs_net,
    prediction_obs_net,
) -> MAMUNetworks:
    """Make all three mamu networks"""

    representation_net = make_mamu_representation_network(
        environment_spec=spec,
        key=key,
        observation_history_size=observation_history_size,
        encoding_size=encoding_size,
        representation_layers=representation_layers,
        observation_net=representation_obs_net,
    )

    dynamics_net = make_mamu_dynamics_network(
        environment_spec=spec,
        key=key,
        num_bins=num_bins,
        representation_net=representation_net,
        encoding_size=encoding_size,
        base_transition_layers=base_transition_layers,
        dynamics_layers=dynamics_layers,
        reward_layers=reward_layers,
        observation_net=dynamics_obs_net,
    )

    prediction_net = make_mamu_prediction_network(
        environment_spec=spec,
        key=key,
        num_bins=num_bins,
        representation_net=representation_net,
        dynamics_net=dynamics_net,
        base_prediction_layers=base_prediction_layers,
        value_prediction_layers=value_prediction_layers,
        policy_prediction_layers=policy_prediction_layers,
        observation_net=prediction_obs_net,
    )

    return MAMUNetworks(
        prediction_network=prediction_net,
        dynamics_network=dynamics_net,
        representation_network=representation_net,
    )


def make_default_mamu_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    rng_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    num_bins: int = 601,
    observation_history_size: int = 1,
    encoding_size: int = 100,
    representation_layers: Sequence[int] = (256, 256),
    base_transition_layers: Sequence[int] = [256],
    dynamics_layers: Sequence[int] = [256],
    reward_layers: Sequence[int] = [256],
    base_prediction_layers: Sequence[int] = (256, 256),
    value_prediction_layers=[16],
    policy_prediction_layers=[16],
    representation_obs_net=identity,
    dynamics_obs_net=identity,
    prediction_obs_net=identity,
) -> Dict[str, Any]:
    """Make the default mamu networks"""

    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_mamu_networks(
            specs[net_key],
            key=rng_key,
            num_bins=num_bins,
            observation_history_size=observation_history_size,
            encoding_size=encoding_size,
            representation_layers=representation_layers,
            base_transition_layers=base_transition_layers,
            reward_layers=reward_layers,
            dynamics_layers=dynamics_layers,
            base_prediction_layers=base_prediction_layers,
            value_prediction_layers=value_prediction_layers,
            policy_prediction_layers=policy_prediction_layers,
            representation_obs_net=representation_obs_net,
            dynamics_obs_net=dynamics_obs_net,
            prediction_obs_net=prediction_obs_net,
        )

    return {
        "networks": networks,
    }
