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

from mava.systems.jax import mat
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


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name=FLAGS.env_name,
        action_space=FLAGS.action_space,
    )

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_subpath = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

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
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Create the system.
    system = mat.MatSystem()

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=mat.make_default_networks,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
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

# n_agents = 5
# batch_size = 10
#
# rng = jax.random.PRNGKey(1)
# enc = lambda x: trf.Encoder(1, 1)(x)
# enc_forward = hk.without_apply_rng(hk.transform(enc))
#
# dummy_input = jnp.zeros((batch_size, n_agents, 8 * 8))
#
# enc_params = enc_forward.init(rng, dummy_input)
#
# v, dummy_obs_rep = enc_forward.apply(enc_params, dummy_input)
# print('here', dummy_obs_rep.shape)
# dummy_actions = jnp.zeros((batch_size, n_agents, 5,))
#
# dec = lambda act, obs: trf.Decoder(1, 64, 1, 5)(act, obs)
# dec_forward = hk.without_apply_rng(hk.transform(dec))
# dec_params = dec_forward.init(rng, dummy_actions, dummy_obs_rep)
