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

"""Example running MAMCTS on debug MPE environments."""
import functools
from datetime import datetime
from typing import Any

import haiku as hk
import jax
import mctx
import optax
import reverb
from absl import app, flags
from acme.jax import utils
from acme.jax.networks import base
from acme.jax.networks.atari import DeepAtariTorso
from mctx import RecurrentFnOutput, RootFnOutput
from pcb_mava.pcb_grid_utils import make_jax_env

from mava.components.jax.building.environments import ParallelExecutorEnvironmentLoop
from mava.systems.jax import mamcts, mappo
from mava.systems.jax.mamcts.mcts_utils import EnvironmentModel, LearnedModel
from mava.utils.debugging.environments.jax.debug_env.new_debug_env import DebugEnv
from mava.utils.loggers import logger_utils
from mava.wrappers.environment_loop_wrappers import (
    DetailedEpisodeStatistics,
    JAXDetailedEpisodeStatistics,
    JAXDetailedPerAgentStatistics,
    JAXMonitorEnvironmentLoop,
)
from mava.wrappers.gym_env_debug import GymWrapper

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "debug_env",
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


GAME_HISTORY_SIZE = 1
FULLY_CONNECTED = False


def network_factory(*args, **kwargs):

    return mamcts.make_default_learned_model_networks(
        num_bins=21,
        use_v2=True,
        channels=64,
        observation_history_size=GAME_HISTORY_SIZE,
        output_init_scale=1.0,
        fully_connected=FULLY_CONNECTED,
        representation_layers=(),
        dynamics_layers=(16,),
        prediction_layers=(16,),
        encoding_size=8,
        *args,
        **kwargs,
    )


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    # Create the system.
    system = mamcts.MAMCTSLearnedModelSystem()

    # Environment.
    environment_factory = functools.partial(make_jax_env, rows=6, cols=6, num_agents=1)

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
        optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.scale(-1e-3)
    )

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        run_evaluator=True,
        sample_batch_size=128,
        num_minibatches=1,
        num_epochs=1,
        num_executors=8,
        multi_process=True,
        root_fn=LearnedModel.learned_root_fn(),
        recurrent_fn=LearnedModel.learned_recurrent_fn(discount_gamma=0.997),
        search=mctx.muzero_policy,
        num_simulations=50,
        rng_seed=0,
        n_step=20,
        discount=0.997,
        history_size=GAME_HISTORY_SIZE,
        fully_connected=FULLY_CONNECTED,
        value_cost=0.25,
        executor_stats_wrapper_class=JAXDetailedEpisodeStatistics,  # For Jax Envs
        sequence_length=37,
        period=37,
        unroll_steps=10,
        max_size=500 * 20,
        importance_sampling_exponent=0.3,
        sampler=functools.partial(reverb.selectors.Prioritized, priority_exponent=0.3),
        terminal="gnome-terminal-tabs",
        num_reanalyse_workers=0,
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)
