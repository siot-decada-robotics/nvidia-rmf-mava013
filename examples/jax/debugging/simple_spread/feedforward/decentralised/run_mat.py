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

import haiku as hk
import jax.numpy as jnp
import optax
from absl import app, flags
from acme.jax.networks.atari import DeepAtariTorso
from pcb_mava import pcb_grid_utils
from pyvirtualdisplay import Display

from mava.components.jax.building.environments import MonitorExecutorEnvironmentLoop
from mava.systems.jax import mat
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
    display = Display(visible=False, size=(600, 600))
    display.start()
    # Environment.
    env_factory = functools.partial(
        pcb_grid_utils.make_environment,
        size=6,
        num_agents=2,
        mava=True,
        render=True,
    )

    def network_factory(*args, **kwargs):
        def obs_net_forward(x):
            orig_shape = x.shape
            x = jnp.reshape(x, (-1, *orig_shape[2:]))
            x = hk.Embed(128, 8)(jnp.int32(x))
            x = DeepAtariTorso()(x)
            x = jnp.reshape(x, (*orig_shape[:2], -1))

            return x

        return mat.make_default_networks(
            *args,
            obs_net=obs_net_forward,
            **kwargs,
        )

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    experiment_path = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

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
        optax.clip_by_global_norm(0.5), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Create the system.
    system = mat.MatSystem()
    system.update(MonitorExecutorEnvironmentLoop)

    # Build the system.
    system.build(
        environment_factory=env_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=experiment_path,
        optimizer=optimizer,
        run_evaluator=True,
        sample_batch_size=5,
        num_epochs=5,
        num_minibatches=1,
        num_executors=1,
        multi_process=True,
        record_every=1,
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
