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

import jax.random
import optax
from absl import app, flags

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

from mava.systems.jax.mat.networks import transformer as trf
import jax.numpy as jnp
import haiku as hk


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    n_agents = 5
    batch_size = 10

    rng = jax.random.PRNGKey(1)
    enc = lambda x: trf.Encoder(1, 1)(x)
    enc_forward = hk.without_apply_rng(hk.transform(enc))

    dummy_input = jnp.zeros((batch_size, n_agents, 8 * 8))

    enc_params = enc_forward.init(rng, dummy_input)

    v, dummy_obs_rep = enc_forward.apply(enc_params, dummy_input)
    print('here', dummy_obs_rep.shape)
    dummy_actions = jnp.zeros((batch_size, n_agents, 5,))

    dec = lambda act, obs: trf.Decoder(1, 64, 1, 5)(act, obs)
    dec_forward = hk.without_apply_rng(hk.transform(dec))
    dec_params = dec_forward.init(rng, dummy_actions, dummy_obs_rep)



if __name__ == "__main__":
    app.run(main)
