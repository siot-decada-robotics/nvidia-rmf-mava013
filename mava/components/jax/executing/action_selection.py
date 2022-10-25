


class FeedforwardExecutorSelectActionValueBased(ExecutorSelectAction):
    """Feedforward executor that selects actions based on the q-values.

    TODO: this class has method on_execution_select_actions which identical to the same
     method in FeedforwardExecutorSelectAction. They can be children of a parent class.
    """

    def __init__(
        self,
        config: ExecutorSelectActionConfig = ExecutorSelectActionConfig(),
    ) -> None:
        """_summary_

        Args:
            config : a config is passed, works with an empty config as well.
        """
        self.config = config

    # Select actions
    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """Select actions for all the agents."""
        executor.store.actions_info = {}
        executor.store.policies_info = {}
        for agent, observation in executor.store.observations.items():
            action_info, policy_info = executor.select_action(agent, observation)
            executor.store.actions_info[agent] = action_info
            executor.store.policies_info[agent] = policy_info

    # Select action
    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        """Compute the action for the agent."""

        agent = executor.store.agent
        network = executor.store.networks["networks"][
            executor.store.agent_net_keys[agent]
        ]

        # observation = executor.store.observation.observation.reshape((1, -1))
        observation = utils.add_batch_dim(executor.store.observation.observation)
        rng_key, executor.store.key = jax.random.split(executor.store.key)

        # TODO (dries): We are currently using jit in the networks per agent.
        # We can also try jit over all the agents in a for loop. This would
        # allow the jit function to save us even more time.
        epsilon = executor.store.epsilon_scheduler.epsilon

        executor.store.action_info, executor.store.policy_info = network.get_action(
            observation,
            rng_key,
            epsilon=epsilon,
            mask=utils.add_batch_dim(executor.store.observation.legal_actions),
        )
