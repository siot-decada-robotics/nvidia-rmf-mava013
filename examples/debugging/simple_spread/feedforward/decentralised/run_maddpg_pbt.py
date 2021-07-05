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

"""Example running feedforward MADDPG on debug MPE environments."""

import functools
from datetime import datetime
from typing import Any

import launchpad as lp
import sonnet as snt
from absl import app, flags
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mava.systems.tf import maddpg_scaled
from mava.systems.tf.maddpg import make_default_networks
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "simple_spread",
    "Debugging environment name (str).",
)
flags.DEFINE_string(
    "action_space",
    "continuous",
    "Environment action space type (str).",
)
flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava/", "Base dir to store experiments.")


def main(_: Any) -> None:

    # environment
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name=FLAGS.env_name,
        action_space=FLAGS.action_space,
    )

    # networks
    network_factory = lp_utils.partial_kwargs(make_default_networks)

    # loggers
    # set custom logger config for each process
    # -- log trainer,executor and evaluator to TF
    # -- log only evaluator to terminal
    log_every = 10
    logger_config = {
        "trainer": {
            "directory": FLAGS.base_dir,
            "to_terminal": False,
            "to_tensorboard": True,
            "time_stamp": FLAGS.mava_id,
            "time_delta": log_every,
        },
        "executor": {
            "directory": FLAGS.base_dir,
            "to_terminal": False,
            "to_tensorboard": True,
            "time_stamp": FLAGS.mava_id,
            "time_delta": log_every,
        },
        "evaluator": {
            "directory": FLAGS.base_dir,
            "to_terminal": True,
            "to_tensorboard": True,
            "time_stamp": FLAGS.mava_id,
            "time_delta": log_every,
        },
    }

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_dir = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

    # distributed program
    program = maddpg_scaled.MADDPG(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_config=logger_config,
        num_executors=2,
        num_trainers=3,
        shared_weights=False,
        trainer_net_config={
            "trainer_0": ["agent_0", "agent_3"],
            "trainer_1": ["agent_1", "agent_4"],
            "trainer_2": ["agent_2", "agent_5"],
            "trainer_3": ["agent_6", "agent_9"],
            "trainer_4": ["agent_7", "agent_8"],
        },
        do_pbt=True,
        num_agents_in_population=10,
        policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        checkpoint_subpath=checkpoint_dir,
        max_gradient_norm=40.0,
    ).build()

    # launch
    gpu_id = -1
    env_vars = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    local_resources = {
        "trainer": [],
        "evaluator": PythonProcess(env=env_vars),
        "executor": PythonProcess(env=env_vars),
    }
    lp.launch(
        program,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal",
        local_resources=local_resources,
    )


if __name__ == "__main__":
    app.run(main)
