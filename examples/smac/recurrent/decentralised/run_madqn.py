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


import functools
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence, Union

import launchpad as lp
import sonnet as snt
import tensorflow as tf
from absl import app, flags
from acme import types

from mava import specs as mava_specs
from mava.components.tf import networks
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.wrappers.env_preprocess_wrappers import ConcatAgentIdToObservation, ConcatPrevActionToObservation
from mava.components.tf.networks.epsilon_greedy import EpsilonGreedy
from mava.systems.tf import madqn
from mava.utils import lp_utils
from mava.utils.environments import pettingzoo_utils
from mava.utils.loggers import logger_utils
from mava.utils.enums import ArchitectureType
from mava.utils.environments.smac_utils import make_environment



FLAGS = flags.FLAGS
flags.DEFINE_string(
    "map_name",
    "3m",
    "Starcraft 2 micromanagement map name (str).",
)

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


def main(_: Any) -> None:
    """Example running recurrent MADQN on multi-agent Starcraft 2 (SMAC) environment."""
    # environment
    environment_factory = functools.partial(
        make_environment, env_name=FLAGS.map_name
    )

    # Networks.
    network_factory = lp_utils.partial_kwargs(
        madqn.make_default_networks,
        architecture_type=ArchitectureType.recurrent
    )

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_dir = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

    # Log every [log_every] seconds.
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=FLAGS.mava_id,
        time_delta=log_every,
    )

    # distributed program
    program = madqn.MADQN(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=1,
        exploration_scheduler_fn=LinearExplorationScheduler(
            epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=8e-6
        ),
        optimizer=snt.optimizers.RMSProp(
            learning_rate=0.0005, epsilon=0.00001, decay=0.99
        ),
        checkpoint_subpath=checkpoint_dir,
        batch_size=32,
        executor_variable_update_period=200,
        target_update_period=100,
        max_gradient_norm=20.0,
        sequence_length=60,
        period=60,
        min_replay_size=32,
        max_replay_size=5000,
        samples_per_insert=1,
        evaluator_interval={"executor_episodes": 2},
        termination_condition={"executor_steps": 3_000_000},
        trainer_fn=madqn.training.MADQNRecurrentTrainer,
        executor_fn=madqn.execution.MADQNRecurrentExecutor,
    ).build()

    # launch
    local_resources = lp_utils.to_device(
        program_nodes=program.groups.keys(), nodes_on_gpu=["trainer"]
    )
    lp.launch(
        program,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal",
        local_resources=local_resources,
    )


if __name__ == "__main__":
    app.run(main)
