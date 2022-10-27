from types import SimpleNamespace

import abc
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Type

import jax
from acme.jax import networks as networks_lib
from acme.jax import utils


from mava.components.executing.action_selection import ExecutorSelectAction
from mava.core_jax import SystemExecutor
from mava.types import NestedArray


class FeedforwardExecutorSelectActionValueBased(ExecutorSelectAction):
    """Feedforward executor that selects actions based on the q-values.
    TODO: this class has method on_execution_select_actions which identical to the same
     method in FeedforwardExecutorSelectAction. They can be children of a parent class.
    """

    def __init__(
        self,
        config: SimpleNamespace = SimpleNamespace(),
    ) -> None:
        """_summary_
        Args:
            config : a config is passed, works with an empty config as well.
        """
        self.config = config

    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """Select actions for each agent and save info in store.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """

        # Dict with params per network
        current_agent_params = {
            network: executor.store.networks[network].get_params()
            for network in executor.store.agent_net_keys.values()
        }
        (
            executor.store.actions_info,
            executor.store.policies_info,
            executor.store.base_key,
        ) = executor.store.select_actions_fn(
            executor.store.observations, current_agent_params, executor.store.base_key
        )

    def on_execution_init_end(self, executor: SystemExecutor) -> None:
        """Create function that is used to select actions.

        Args:
            executor : SystemExecutor.

        Returns:
            None.
        """
        networks = executor.store.networks
        agent_net_keys = executor.store.agent_net_keys

        def select_action(
            observation: NestedArray,
            current_params: NestedArray,
            network: Any,
            base_key: networks_lib.PRNGKey,
        ) -> Tuple[NestedArray, NestedArray, networks_lib.PRNGKey]:
            """Action selection across a single agent.

            Args:
                observation : The observation for the current agent.
                current_params : The parameters for current agent's network.
                network : The network object used by the current agent.
                key : A JAX prng key.

            Returns:
                action info, policy info and new key.
            """
            epsilon = executor.store.epsilon_scheduler.epsilon
            base_key, action_key = jax.random.split(base_key)

            observation_data = utils.add_batch_dim(observation.observation)

            action_info, policy_info = network.get_action(
                observation=observation_data,
                params=current_params["value_network"],
                key=action_key,
                epsilon=epsilon,
                mask=utils.add_batch_dim(observation.legal_actions),
            )

            return action_info, policy_info, base_key

        def select_actions(
            observations: Dict[str, NestedArray],
            current_params: Dict[str, NestedArray],
            base_key: networks_lib.PRNGKey,
        ) -> Tuple[
            Dict[str, NestedArray], Dict[str, NestedArray], networks_lib.PRNGKey
        ]:
            """Select actions across all agents - this is jitted below.

            Args:
                observations : The observations for all the agents.
                current_params : The parameters for all the agents.
                base_key : A JAX prng_key.

            Returns:
                action info, policy info and new prng key.
            """
            actions_info, policies_info = {}, {}
            # TODO Look at tree mapping this forloop.
            # Since this is jitted, compiling a forloop with lots of agents could take
            # long, we should vectorize this.
            for agent, observation in observations.items():
                network = networks[agent_net_keys[agent]]
                actions_info[agent], policies_info[agent], base_key = select_action(
                    observation=observation,
                    current_params=current_params[agent_net_keys[agent]],
                    network=network,
                    base_key=base_key,
                )
            return actions_info, policies_info, base_key

        executor.store.select_actions_fn = jax.jit(select_actions)
