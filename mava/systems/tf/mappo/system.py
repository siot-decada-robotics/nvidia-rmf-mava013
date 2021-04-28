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

"""MAPPO system implementation."""
import dataclasses
from typing import Callable, Dict, Iterator, Optional

import reverb
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils
from acme.utils import counting, loggers

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.systems import system
from mava.systems.builders import SystemBuilder
from mava.systems.tf.mappo import execution, training


@dataclasses.dataclass
class MAPPOConfig:
    """Configuration options for the MAPPO system
    Args:
        environment_spec: description of the actions, observations, etc.
        networks: the online Q network (the one being optimized)
        sequence_length: ...
        sequence_period: ...
        entropy_cost: ...
        baseline_cost: ...
        max_abs_reward: ...
        batch_size: batch size for updates.
        max_queue_size: maximum queue size.
        learning_rate: learning rate for the q-network update.
        discount: discount to use for TD updates.
        logger: logger object to be used by learner.
        max_gradient_norm: used for gradient clipping.
        replay_table_name: string indicating what name to give the replay table.
    """

    environment_spec: specs.EnvironmentSpec
    networks: Dict[str, snt.RNNCore]
    sequence_length: int
    sequence_period: int
    counter: counting.Counter = None
    logger: loggers.Logger = None
    shared_weights: bool = False
    discount: float = 0.99
    lambda_gae: float = 0.95
    max_queue_size: int = 100_000
    batch_size: int = 16
    critic_learning_rate: float = 1e-3
    policy_learning_rate: float = 1e-3
    entropy_cost: float = 0.01
    baseline_cost: float = 0.5
    max_abs_reward: Optional[float] = None
    max_gradient_norm: Optional[float] = None
    checkpoint: bool = True
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


class MAPPOBuilder(SystemBuilder):
    """Builder for MAPPO which constructs individual components of the system."""

    """Defines an interface for defining the components of an MARL system.
      Implementations of this interface contain a complete specification of a
      concrete MARL system. An instance of this class can be used to build an
      MARL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(self, config: MAPPOConfig):
        """Args:
        config: Configuration options for the MAPPO system."""

        self._config = config

        """ _agents: a list of the agent specs (ids).
            _agent_types: a list of the types of agents to be used."""
        self._agents = self._config.environment_spec.get_agent_ids()
        self._agent_types = self._config.environment_spec.get_agent_types()

    def make_replay_table(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> reverb.Table:
        """Create tables to insert data into."""
        agent_specs = environment_spec.get_agent_specs()

        core_states = {}
        logits = {}
        for agent in self._agents:
            core_states[agent] = self._config.networks[agent].initial_state(1)
            num_actions = agent_specs[agent].actions.num_values
            logits[agent] = tf.ones(shape=(1, num_actions), dtype=tf.float32)

        extras_spec = {"core_states": core_states, "logits": logits}

        extras_spec = tf2_utils.squeeze_batch_dim(extras_spec)

        return reverb.Table.queue(
            name=self._config.replay_table_name,
            max_size=self._config.max_queue_size,
            signature=reverb_adders.ParallelSequenceAdder.signature(
                environment_spec, extras_spec=extras_spec
            ),
        )

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the system."""
        # dataset = datasets.make_reverb_dataset(
        #     server_address=replay_client.server_address,
        #     batch_size=self._config.batch_size,
        #     sequence_length=self._config.sequence_length,
        # )
        # NOTE we dont do the above to avoid interleaving and prefetching.
        # See Alex's note on Ridl
        dataset = reverb.ReplayDataset.from_table_signature(
            server_address=replay_client.server_address,
            table=self._config.replay_table_name,
            max_in_flight_samples_per_worker=1,
            sequence_length=self._config.sequence_length,
            emit_timesteps=False,
        )
        dataset = dataset.batch(self._config.batch_size, drop_remainder=True)
        return iter(dataset)

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment.
        Args:
          replay_client: Reverb Client which points to the replay server.
        """
        return reverb_adders.ParallelSequenceAdder(
            client=replay_client,
            period=self._config.sequence_period,
            sequence_length=self._config.sequence_length,
        )

    def make_executor(
        self,
        networks: Dict[str, snt.RNNCore],
        adder: Optional[adders.ParallelAdder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ) -> core.Executor:
        """Create an executor instance.
        Args:
          policy_networks: A struct of instance of all the different policy networks;
           this should be a callable
            which takes as input observations and returns actions.
          adder: How data is recorded (e.g. added to replay).
          variable_source: A source providing the necessary executor parameters.
        """
        shared_weights = self._config.shared_weights

        variable_client = None
        if variable_source:
            agent_keys = self._agent_types if shared_weights else self._agents

            # Create policy variables
            variables = {}
            for agent in agent_keys:
                variables[agent] = networks[agent].variables

            # Get new policy variables
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={"policy": variables},
                update_period=1000,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # Create the actor which defines how we take actions.
        return execution.MAPPORecurrentExecutor(
            networks=networks,
            shared_weights=shared_weights,
            variable_client=variable_client,
            adder=adder,
        )

    def make_trainer(
        self,
        networks: Dict[str, snt.RNNCore],
        dataset: Iterator[reverb.ReplaySample],
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[types.NestedLogger] = None,
        checkpoint: bool = False,
    ) -> core.Trainer:
        """Creates an instance of the trainer.
        Args:
          networks: struct describing the networks needed by the trainer; this can
            be specific to the trainer in question.
          dataset: iterator over samples from replay.
          replay_client: client which allows communication with replay, e.g. in
            order to update priorities.
          counter: a Counter which allows for recording of counts (trainer steps,
            executor steps, etc.) distributed throughout the system.
          logger: Logger object for logging metadata.
          checkpoint: bool controlling whether the trainer checkpoints itself.
        """
        agents = self._agents
        agent_types = self._agent_types
        shared_weights = self._config.shared_weights
        discount = self._config.discount
        lambda_gae = self._config.lambda_gae
        max_gradient_norm = self._config.max_gradient_norm
        critic_learning_rate = self._config.critic_learning_rate
        policy_learning_rate = self._config.policy_learning_rate
        entropy_cost = self._config.entropy_cost
        baseline_cost = self._config.baseline_cost
        max_abs_reward = self._config.max_abs_reward

        # The learner updates the parameters (and initializes them).
        trainer = training.MAPPOTrainer(
            agents=agents,
            agent_types=agent_types,
            shared_weights=shared_weights,
            networks=networks,
            dataset=dataset,
            counter=counter,
            logger=logger,
            discount=discount,
            lambda_gae=lambda_gae,
            critic_learning_rate=critic_learning_rate,
            policy_learning_rate=policy_learning_rate,
            entropy_cost=entropy_cost,
            baseline_cost=baseline_cost,
            max_abs_reward=max_abs_reward,
            max_gradient_norm=max_gradient_norm,
        )
        return trainer


class MAPPO(system.System):
    """MAPPO system.
    This implements a single-process MAPPO system. This is an actor-critic based
    system that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policies of each agent
    (and as a result the behavior) by sampling uniformly from this buffer.
    """

    def __init__(
        self,
        environment_spec: specs.MAEnvironmentSpec,
        networks: Dict[str, snt.Module],
        sequence_length: int,
        sequence_period: int,
        shared_weights: bool = False,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        discount: float = 0.99,
        lambda_gae: float = 0.95,
        max_queue_size: int = 100_000,
        batch_size: int = 16,
        critic_learning_rate: float = 1e-3,
        policy_learning_rate: float = 1e-3,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        max_abs_reward: Optional[float] = None,
        max_gradient_norm: Optional[float] = None,
        checkpoint: bool = True,
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
    ):
        """Initialize the system.
        Args:
        environment_spec: description of the actions, observations, etc.
        networks: the online Q network (the one being optimized)
        sequence_length: ...
        sequence_period: ...
        entropy_cost: ...
        baseline_cost: ...
        max_abs_reward: ...
        batch_size: batch size for updates.
        max_queue_size: maximum queue size.
        learning_rate: learning rate for the q-network update.
        discount: discount to use for TD updates.
        logger: logger object to be used by learner.
        max_gradient_norm: used for gradient clipping.
        replay_table_name: string indicating what name to give the replay table."""

        builder = MAPPOBuilder(
            MAPPOConfig(
                environment_spec=environment_spec,
                networks=networks,
                sequence_length=sequence_length,
                sequence_period=sequence_period,
                shared_weights=shared_weights,
                counter=counter,
                logger=logger,
                discount=discount,
                lambda_gae=lambda_gae,
                max_queue_size=max_queue_size,
                batch_size=batch_size,
                critic_learning_rate=critic_learning_rate,
                policy_learning_rate=policy_learning_rate,
                entropy_cost=entropy_cost,
                baseline_cost=baseline_cost,
                max_abs_reward=max_abs_reward,
                max_gradient_norm=max_gradient_norm,
                checkpoint=checkpoint,
                replay_table_name=replay_table_name,
            )
        )

        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_table = builder.make_replay_table(environment_spec=environment_spec)
        self._server = reverb.Server([replay_table], port=None)
        replay_client = reverb.Client(f"localhost:{self._server.port}")

        self._can_sample: Callable[[], bool] = lambda: replay_table.can_sample(
            batch_size
        )

        # The adder is used to insert observations into replay.
        adder = builder.make_adder(replay_client)

        # The dataset provides an interface to sample from replay.
        dataset = builder.make_dataset_iterator(replay_client)

        # Create the executor which defines how we take actions.
        executor = builder.make_executor(networks, adder)

        # The learner updates the parameters (and initializes them).
        trainer = builder.make_trainer(
            networks, dataset, counter=counter, logger=logger, checkpoint=checkpoint
        )

        super().__init__(
            executor=executor,
            trainer=trainer,
            min_observations=batch_size,
            observations_per_step=batch_size,
        )

    def update(self) -> None:
        learner_step = False
        while self._can_sample():
            self._trainer.step()
            learner_step = True
        if learner_step:
            self._executor.update()
