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

"""Example running MAPPO on debug MPE environments."""
import functools
from datetime import datetime
from typing import Any

import optax
from absl import app, flags
from pcb_mava import pcb_grid_utils
from pyvirtualdisplay import Display

from mava.components.jax.building.environments import MonitorExecutorEnvironmentLoop
from mava.systems.jax import mappo
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "simple_spread",
    "Debugging environment name (str).",
)
flags.DEFINE_string(
    "action_space",
    "discrete",
    "Environment action space type (str).",
)

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")

import jax.numpy as jnp
import haiku as hk
from acme.jax.networks.atari import DeepAtariTorso


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    display = Display(visible=False, size=(600, 600))
    display.start()
    # Environment.
    env_factory = functools.partial(
        pcb_grid_utils.make_environment,
        size=8,
        num_agents=3,
        step_timeout=50,
        reward_per_timestep=-0.03,
        continued_rewards=False,
        mava=True,
        mava_stats=True,
        render=True,
    )
    # Networks.
    def network_factory(*args, **kwargs):
        def obs_net_forward(x):
            # orig_shape = x.shape
            # x = jnp.reshape(x, (-1, *orig_shape[2:]))
            x = hk.Embed(128, 8)(jnp.int32(x))
            x = DeepAtariTorso()(x)
            # x = jnp.reshape(x, (*orig_shape[:2], -1))

            return x

        return mappo.make_default_networks(
            *args,
            observation_network=obs_net_forward,
            **kwargs,
        )

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    experiment_path = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

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

    # Optimizer.
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Create the system.
    system = mappo.MAPPOSystem()

    system.build(
        environment_factory=env_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=experiment_path,
        optimizer=optimizer,
        run_evaluator=True,
        sample_batch_size=5,
        num_epochs=15,
        num_executors=1,
        multi_process=True,
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)
