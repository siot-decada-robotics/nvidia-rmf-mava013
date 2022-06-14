import jax
from acme.jax import utils

from mava.components.jax.executing.action_selection import (
    ExecutorSelectAction,
    ExecutorSelectActionConfig,
)
from mava.core_jax import SystemExecutor


class MatExecutorActionSelection(ExecutorSelectAction):
    def __init__(
        self, config: ExecutorSelectActionConfig = ExecutorSelectActionConfig()
    ):
        super().__init__(config)

    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        executor.store.actions_info = {}
        executor.store.policies_info = {}

        # todo (sasha): do this as in the paper impl (shifted_action)
        executor.store.previous_actions = []

        encoded_observations = utils.add_batch_dim(
            get_obs_encoder(executor)(
                stack_observations(list(executor.store.observations.values()))
            )
        )

        for agent in executor.store.observations.keys():
            action_info, policy_info = executor.select_action(
                agent, encoded_observations
            )
            executor.store.actions_info[agent] = action_info
            executor.store.policies_info[agent] = policy_info

    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        # TODO (sasha): assert that you are only using shared weights and the same net
        #  for all agents
        network = executor.store.networks["decoder"]

        observation = executor.store.observation.observation
        rng_key, executor.store.key = jax.random.split(executor.store.key)

        executor.store.action_info, executor.store.policy_info = network.get_action(
            observation,
            executor.store.previous_actions,
            rng_key,
            executor.store.observation.legal_actions[executor.store.agent],
        )

        # todo (sasha): do this as in the paper impl (shifted_action)
        executor.store.previous_actions.append(executor.store.action_info)


def stack_observations(observations):
    # Stack inside an OLT
    pass


def get_obs_encoder(executor: SystemExecutor):
    # TODO (sasha): make networks store like this
    return executor.store.networks["networks"].encoder
