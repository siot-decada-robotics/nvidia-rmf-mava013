import time
from typing import Any, Dict, Optional, Tuple

import acme
import dm_env
import numpy as np
from acme.utils import counting, loggers

import mava
from mava.types import Action
from mava.utils.jax_tree_utils import index_stacked_tree
from mava.utils.training_utils import check_count_condition
from mava.utils.wrapper_utils import (
    SeqTimestepDict,
    convert_seq_timestep_and_actions_to_parallel,
    generate_zeros_from_spec,
)


class VectorizedParallelEnvironmentLoop(acme.core.Worker):
    """A parallel MARL environment loop.

    This takes `Environment` and `Executor` instances and coordinates their
    interaction. Executors are updated if `should_update=True`. This can be used as:
        loop = EnvironmentLoop(environment, executor)
        loop.run(num_episodes)
    A `Counter` instance can optionally be given in order to maintain counts
    between different Mava components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.
    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger from acme. A string `label` can be passed
    to easily change the label associated with the default logger; this is ignored
    if a `Logger` instance is given.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        executor: mava.core.Executor,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        should_update: bool = True,
        label: str = "parallel_environment_loop",
    ):
        """Parallel environment loop init

        Args:
            environment: an environment
            executor: a Mava executor
            counter: an optional counter. Defaults to None.
            logger: an optional counter. Defaults to None.
            should_update: should update. Defaults to True.
            label: optional label. Defaults to "sequential_environment_loop".
        """
        # Internalize agent and environment.
        self._environment = environment
        self._executor = executor
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)

        self._should_update = should_update
        self._running_statistics: Dict[str, float] = {}

        # We need this to schedule evaluation/test runs
        self._last_evaluator_run_t = -1

    def _get_actions(self, timestep: dm_env.TimeStep) -> Any:
        return self._executor.select_actions(timestep.observation)

    def _get_running_stats(self) -> Dict:
        return self._running_statistics

    def _compute_step_statistics(self, rewards: Dict[str, float]) -> None:
        pass

    def _compute_episode_statistics(
        self,
        episode_returns: Dict[str, float],
        episode_steps: int,
        start_time: float,
    ) -> None:
        pass

    def get_counts(self) -> counting.Counter:
        """Get latest counts"""
        if hasattr(self._executor, "_counts"):
            counts = self._executor._counts
        else:
            counts = self._counter.get_counts()
        return counts

    def record_counts(self, episode_steps: int) -> counting.Counter:
        """Record latest counts"""
        # Record counts.
        if hasattr(self._executor, "_counts"):
            loop_type = "evaluator" if self._executor._evaluator else "executor"

            if hasattr(self._executor, "_variable_client"):
                self._executor._variable_client.add_async(
                    [f"{loop_type}_episodes", f"{loop_type}_steps"],
                    {
                        f"{loop_type}_episodes": 1,
                        f"{loop_type}_steps": episode_steps,
                    },
                )
            else:
                self._executor._counts[f"{loop_type}_episodes"] += 1
                self._executor._counts[f"{loop_type}_steps"] += episode_steps

            counts = self._executor._counts
        else:
            counts = self._counter.increment(episodes=1, steps=episode_steps)

        return counts

    def run_episode(self) -> loggers.LoggingData:
        """Run one episode.

        Each episode is a loop which interacts first with the environment to get a
        dictionary of observations and then give those observations to the executor
        in order to retrieve an action for each agent in the system.

        Returns:
            An instance of `loggers.LoggingData`.
        """

        # Reset any counts and start the environment.
        start_time = time.time()
        episode_steps = 0

        timestep = self._environment.reset()

        if type(timestep) == tuple:
            # TODO fix for this case -> output is a list of tuples
            timestep, env_extras = timestep
        else:
            env_extras = {}

        # Make the first observation.
        self._executor.observe_first(timestep, extras=env_extras)

        # For evaluation, this keeps track of the total undiscounted reward
        # for each agent accumulated during the episode.
        rewards: Dict[str, float] = {}
        episode_returns: Dict[str, float] = {}
        for agent, spec in self._environment.reward_spec().items():
            rewards.update({agent: generate_zeros_from_spec(spec)})
            episode_returns.update({agent: generate_zeros_from_spec(spec)})

        # Run an episode.
        # TODO probably stack timesteps and join the STEP_TYPE
        while not all(timestep.step_type == dm_env.StepType.LAST):

            # Generate an action from the agent's policy and step the environment.
            actions = self._get_actions(timestep)

            if type(actions) == tuple:
                # Return other action information
                # e.g. the policy information.
                env_actions, _ = actions
            else:
                env_actions = actions

            timestep = self._environment.step(env_actions)
            # print(f"BEGIN----------------:\n{timestep}")
            if type(timestep) == tuple:
                timestep, env_extras = timestep
            else:
                env_extras = {}

            rewards = timestep.reward

            # Have the agent observe the timestep and let the actor update itself.
            self._executor.observe(
                actions, next_timestep=timestep, next_extras=env_extras
            )

            if self._should_update:
                self._executor.update()

            # Book-keeping.
            episode_steps += self._executor.store.global_config.num_environments

            if hasattr(self._executor, "after_action_selection"):
                if hasattr(self._executor, "_counts"):
                    loop_type = "evaluator" if self._executor._evaluator else "executor"
                    total_steps_before_current_episode = self._executor._counts[
                        f"{loop_type}_steps"
                    ].numpy()
                else:
                    total_steps_before_current_episode = self._counter.get_counts().get(
                        "executor_steps", 0
                    )
                current_step_t = total_steps_before_current_episode + episode_steps
                self._executor.after_action_selection(current_step_t)

            # TODO: vmap?
            rewards = index_stacked_tree(rewards, 0)
            self._compute_step_statistics(rewards)

            for agent, reward in rewards.items():
                episode_returns[agent] = episode_returns[agent] + reward

        self._compute_episode_statistics(
            episode_returns,
            episode_steps,
            start_time,
        )
        if self._get_running_stats():
            return self._get_running_stats()
        else:

            counts = self.record_counts(episode_steps)

            # Collect the results and combine with counts.
            steps_per_second = episode_steps / (time.time() - start_time)
            result = {
                "episode_length": episode_steps,
                "mean_episode_return": np.mean(list(episode_returns.values())),
                "steps_per_second": steps_per_second,
            }
            result.update(counts)
            return result

    def run_episode_and_log(self) -> loggers.LoggingData:
        """_summary_"""

        results = self.run_episode()
        self._logger.write(results)
        return results

    def run(
        self, num_episodes: Optional[int] = None, num_steps: Optional[int] = None
    ) -> None:
        """Perform the run loop.

        Run the environment loop either for `num_episodes` episodes or for at
        least `num_steps` steps (the last episode is always run until completion,
        so the total number of steps may be slightly more than `num_steps`).
        At least one of these two arguments has to be None.
        Upon termination of an episode a new episode will be started. If the number
        of episodes and the number of steps are not given then this will interact
        with the environment infinitely.

        Args:
            num_episodes: number of episodes to run the loop for.
            num_steps: minimal number of steps to run the loop for.

        Raises:
            ValueError: If both 'num_episodes' and 'num_steps' are not None.
        """

        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episode_count: int, step_count: int) -> bool:
            return (num_episodes is not None and episode_count >= num_episodes) or (
                num_steps is not None and step_count >= num_steps
            )

        def should_run_loop(eval_condtion: Tuple) -> bool:
            """Check if the eval loop should run in current step.

            Args:
                eval_condition : tuple containing interval key and count.

            Returns:
                a bool indicatings if eval should run.
            """
            should_run_loop = False
            eval_interval_key, eval_interval_count = eval_condition
            counts = self.get_counts()
            if counts:
                count = counts.get(eval_interval_key)
                # We run eval loops around every eval_interval_count (not exactly every
                # eval_interval_count due to latency in getting updated counts).
                should_run_loop = (
                    (count - self._last_evaluator_run_t) / eval_interval_count
                ) >= 1.0
                if should_run_loop:
                    self._last_evaluator_run_t = int(count)
                    print(
                        "Running eval loop at executor step: "
                        + f"{self._last_evaluator_run_t}"
                    )
            return should_run_loop

        episode_count, step_count = 0, 0

        # Currently, we only use intervals for eval loops.
        environment_loop_schedule = (
            self._executor._evaluator and self._executor._interval
        )
        if environment_loop_schedule:
            eval_condition = check_count_condition(self._executor._interval)

        while not should_terminate(episode_count, step_count):
            if (not environment_loop_schedule) or should_run_loop(eval_condition):
                result = self.run_episode()
                episode_count += 1
                step_count += result["episode_length"]
                # Log the given results.
                self._logger.write(result)
            else:
                # Note: We assume that the evaluator will be running less
                # than once per second.
                time.sleep(1)
            # We need to get the latest counts if we are using eval intervals.
            if environment_loop_schedule:
                self._executor.update()
