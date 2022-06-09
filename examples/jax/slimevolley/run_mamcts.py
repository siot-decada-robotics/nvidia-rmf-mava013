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
import mctx
import optax
from absl import app, flags
from acme.jax import utils
from acme.jax.networks.atari import DeepAtariTorso
from mctx import RecurrentFnOutput, RootFnOutput

from mava.systems.jax import mamcts
from mava.systems.jax.mamcts.mcts_utils import EnvironmentModel
from mava.utils.environments.JaxEnvironments.jax_env_utils import make_slimevolley_env
from mava.utils.loggers import logger_utils
from mava.wrappers.environment_loop_wrappers import (
    JAXDetailedEpisodeStatistics,
    JAXDetailedPerAgentStatistics,
    JAXMonitorEnvironmentLoop,
)

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


def network_factory(*args, **kwargs):

    return mamcts.make_default_mamcts_networks(
        num_bins=21,
        base_prediction_layers=(256, 256),
        *args,
        **kwargs,
    )


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    # Environment.
    environment_factory = functools.partial(
        make_slimevolley_env, is_multi_agent=True, is_cooperative=True
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
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-3)
    )

    # Create the system.
    system = mamcts.MAMCTSSystem()

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        run_evaluator=True,
        sample_batch_size=128,
        num_minibatches=8,
        num_epochs=4,
        num_executors=6,
        multi_process=True,
        root_fn=EnvironmentModel.environment_root_fn(),
        recurrent_fn=EnvironmentModel.greedy_policy_recurrent_fn(discount_gamma=1.0),
        search=mctx.gumbel_muzero_policy,
        environment_model=environment_factory(),
        num_simulations=50,
        evaluator_num_simulations=50,
        evaluator_other_search_params=lambda: {"gumbel_scale": 0.0},
        rng_seed=0,
        n_step=10,
        discount=0.99,
        executor_stats_wrapper_class=JAXDetailedEpisodeStatistics,
        evaluator_stats_wrapper_class=JAXMonitorEnvironmentLoop,
        terminal="gnome-terminal-tabs",
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)
