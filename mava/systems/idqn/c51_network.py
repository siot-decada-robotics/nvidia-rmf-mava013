import copy
import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import chex
import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
from acme.jax import networks as networks_lib
from distrax import Categorical  # type: ignore
from acme import specs
from mava import specs as mava_specs
from acme.jax import utils

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
        self._num_atoms = num_atoms
        self._atoms = jnp.linspace(v_min, v_max, self._num_atoms)

    def __call__(
        self, inputs: jnp.ndarray
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
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


@dataclasses.dataclass
class C51DuellingNetwork:
    def __init__(
        self,
        policy_params: networks_lib.Params,
        network: C51DuellingMLP,
    ) -> None:
        """A container for IDQN networks.

        Holds target and main network

        Args:
            policy_params: parameters of the policy network
            network: structure of the policy network

        Return:
            IDQNNetwork
        """
        self.policy_params: networks_lib.Params = policy_params
        self.target_policy_params: networks_lib.Params = copy.deepcopy(policy_params)
        self.policy_network: C51DuellingMLP = network

        def forward_fn(
            policy_params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
        ) -> jnp.ndarray:
            """Get Q values from the network given observations

            Args:
                policy_params: parameters of the policy network
                observations: agent observations

            Returns: Q-values of all actions in the current state
            """
            return self.policy_network.apply(policy_params, observations)

        self.forward = forward_fn

    def get_action(
        self,
        params: networks_lib.Params,
        observations: networks_lib.Observation,
        epsilon: float,
        base_key: jax.random.KeyArray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get actions from policy network given observations.

        Args:
            policy_params: parameters of the policy network
            observations: agent observations
            epsilon: probability that the agent takes a random action
            base_key: jax random key
            mask: action mask of the legal actions

        Returns:
            an action to take in the current state
        """
        q_values, _, _ = self.forward(params, observations)
        masked_q_values = jnp.where(mask == 1.0, q_values, jnp.finfo(jnp.float32).min)

        greedy_actions = masked_q_values == jnp.max(masked_q_values)
        greedy_actions_probs = greedy_actions / jnp.sum(greedy_actions)

        random_action_probs = mask / jnp.sum(mask)

        weighted_gready_probs = (1 - epsilon) * greedy_actions_probs
        weighted_rand_probs = epsilon * random_action_probs
        combined_probs = weighted_gready_probs + weighted_rand_probs

        action_dist = Categorical(probs=combined_probs)
        return action_dist.sample(seed=base_key)

    def get_params(
        self,
    ) -> Dict[str, networks_lib.Params]:
        """Return current params of the target and policy network.

        Returns:
            policy and target policy params.
        """
        return {
            "policy_network": self.policy_params,
            "target_policy_network": self.target_policy_params,
        }
    
def make_c51_network(
    environment_spec: specs.EnvironmentSpec,
    base_key: jax.random.KeyArray,
    policy_layer_sizes: Sequence[int],
    v_max: int,
    v_min: int,
    num_atoms: int = 51,
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    observation_network: Optional[Callable] = None,
):
    num_actions = environment_spec.actions.num_values

    @hk.without_apply_rng
    @hk.transform
    def policy_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        # Add the observation network and an MLP network.
        value_network = []
        if observation_network is not None:
            value_network.append(observation_network)

        value_network.append(
            hk.nets.MLP(
                (*policy_layer_sizes, num_actions),
                activation=activation_function,
            ),
        )
        value_network.append(
            C51DuellingMLP(
                num_actions,
                policy_layer_sizes,
                v_max=v_max,
                v_min=v_min,
                num_atoms=num_atoms,
            )
        )

        return hk.Sequential(value_network)(inputs)

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    base_key, network_key = jax.random.split(base_key)

    policy_params = policy_fn.init(network_key, dummy_obs)

    base_key, network_key = jax.random.split(base_key)

    return C51DuellingNetwork(
        network=policy_fn,
        policy_params=policy_params,
    )


def make_c51_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    base_key: jax.random.KeyArray,
    v_min: int,
    v_max: int,
    num_atoms: int = 51,
    net_spec_keys: Dict[str, str] = {},
    policy_layer_sizes: Sequence[int] = (
        64,
        64,
    ),
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    observation_network: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create default IDQN networks (one per agent)

    Args:
        environment_spec: mava multi-agent environment spec
        agent_net_keys: dictionary specifying which networks are
            used by which agent
        base_key: jax random key to be used for network initialization
        net_spec_keys: keys for each agent network
        policy_layer_sizes: policy network layers
        activation_function: activation function to be used for network hidden layers.
        observation_network: optional network for processing observations

    Returns:
        networks: IDQN networks created to given spec
    """

    # Create agent_type specs.
    specs = environment_spec.get_agent_environment_specs()
    if not net_spec_keys:
        specs = {
            agent_net_keys[agent_key]: specs[agent_key] for agent_key in specs.keys()
        }
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_c51_network(
            specs[net_key],
            base_key=base_key,
            v_min=v_min,
            v_max=v_max,
            num_atoms=num_atoms,
            policy_layer_sizes=policy_layer_sizes,
            activation_function=activation_function,
            observation_network=observation_network,
        )

    return networks
