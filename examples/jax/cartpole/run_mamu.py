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

"""Example running MAMU on Jax Cartpole."""
import functools
from datetime import datetime
from typing import Any

import gym
import mctx
import optax
import reverb
from absl import app, flags
from acme.jax import utils

from mava.components.jax.building.environments import ParallelExecutorEnvironmentLoop
from mava.systems.jax import mamcts
from mava.systems.jax.mamcts.mcts_utils import MAMU
from mava.utils.environments.JaxEnvironments.jax_env_utils import make_jax_cartpole
from mava.utils.loggers import logger_utils
from mava.wrappers.environment_loop_wrappers import (
    JAXDetailedEpisodeStatistics,
    JAXMonitorEnvironmentLoop,
)
from mava.wrappers.gym_env_debug import GymWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


GAME_HISTORY_SIZE = 1


def network_factory(*args, **kwargs):

    return mamcts.make_default_mamu_networks(
        num_bins=21,
        observation_history_size=GAME_HISTORY_SIZE,
        representation_layers=[16],
        base_transition_layers=[16],
        dynamics_layers=[16],
        reward_layers=[16],
        base_prediction_layers=[16],
        value_prediction_layers=[16],
        policy_prediction_layers=[16],
        encoding_size=8,
        representation_obs_net=utils.batch_concat,
        dynamics_obs_net=utils.batch_concat,
        prediction_obs_net=utils.batch_concat,
        *args,
        **kwargs,
    )


def make_gym_cartpole(evaluation: bool = False):
    return GymWrapper(gym.make("CartPole-v1"))


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    # Create the system.
    system = mamcts.MAMUSystem()

    # Environment.
    environment_factory = functools.partial(
        make_gym_cartpole,
    )

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_subpath = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

    # Log every [log_every] seconds.
    log_every = 1
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
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale(-1),
        optax.scale_by_schedule(optax.exponential_decay(0.02, 1000, 0.8)),
    )

    system.update(ParallelExecutorEnvironmentLoop)

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        run_evaluator=True,
        sample_batch_size=128,
        num_executors=4,
        multi_process=True,
        root_fn=MAMU.learned_root_fn(),
        recurrent_fn=MAMU.learned_recurrent_fn(discount_gamma=0.997),
        search=functools.partial(mctx.muzero_policy, dirichlet_alpha=0.25),
        num_simulations=50,
        rng_seed=0,
        n_step=20,
        discount=0.997,
        value_cost=1.0,
        # executor_stats_wrapper_class=JAXDetailedEpisodeStatistics,  # For Jax Envs
        # evaluator_stats_wrapper_class=JAXMonitorEnvironmentLoop,
        sequence_length=20,
        period=20,
        unroll_steps=10,
        max_size=5000000,
        importance_sampling_exponent=0.5,
        priority_exponent=0.5,
        terminal="gnome-terminal-tabs",
        executor_parameter_update_period=300,
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)
