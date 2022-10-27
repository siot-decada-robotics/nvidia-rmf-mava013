import dataclasses
from typing import Dict, Tuple

import chex
import jax.numpy as jnp
from acme.jax import networks as networks_lib
from jax import jit

from mava.systems.madqn.components.executing.epsilon_greedy import EpsilonGreedyWithMask


@dataclasses.dataclass
class DQNNetworks:
    """A Class for DQN network.

    Args:
        network: pure function defining the feedforward neural network
        params: the parameters of the network.
    """

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
    ) -> None:
        """Instantiate the class."""
        self.network = network
        self.policy_params = params  # TODO (sasha): change this back to params once ValueBasedTrainerInit has been created

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray], observations: networks_lib.Observation
        ) -> jnp.ndarray:
            """Forward evaluation of the network.

            Args:
                params: the network parameters
                observations: the observation

            Returns:
                the results of the feedforward evaluation, i.e. the q-values in DQN
            """
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            q_values, q_logits, atoms = self.network.apply(params, observations)

            return q_values, q_logits, atoms

        self.forward_fn = forward_fn

    def get_action(
        self,
        observation: networks_lib.Observation,
        params: networks_lib.Params,
        key: networks_lib.PRNGKey,
        epsilon: float,
        mask: chex.Array,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Taking actions using epsilon greedy approach.

        Args:
            observations: the observations
            key: the random number generator key
            epsilon: the epsilon value for the epsilon-greedy approach. If 0.0, then the
             action is taken deterministically.

        Returns:
            the actions and a dictionary with q-values
        """
        action_values, _, _ = self.forward_fn(params, observation)
        actions = EpsilonGreedyWithMask(
            preferences=action_values, epsilon=epsilon, mask=mask  # type: ignore
        ).sample(seed=key)
        assert len(actions) == 1, "Only one action is allowed."
        actions = jnp.array(actions, dtype=jnp.int64)
        actions = jnp.squeeze(actions)

        return actions, {"action_values": jnp.squeeze(action_values)}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """Get the value of the network.

        Args:
            observations: the observations
        Returns:
            the feedforward values of the network, i.e. the Q-values in DQN.
        """
        q_value = self.network.apply(self.policy_params, observations)
        return q_value

    def get_params(
        self,
    ) -> Dict[str, jnp.ndarray]:
        """Return current params.

        Returns:
            policy and critic params.
        """
        return {
            "value_network": self.policy_params,
        }
