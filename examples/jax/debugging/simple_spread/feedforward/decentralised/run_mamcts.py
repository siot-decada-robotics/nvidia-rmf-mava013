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

import chex
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
from absl import app, flags
from jumanji.jax.pcb_grid.env import PcbGridEnv
from jumanji.jax.pcb_grid.types import State
from mctx import RecurrentFnOutput, RootFnOutput

from mava.systems.jax import mamcts
from mava.utils.debugging.environments.jax.debug_env.new_debug_env import DebugEnv
from mava.utils.environments import debugging_utils
from mava.utils.id_utils import EntityId
from mava.utils.loggers import logger_utils
from mava.wrappers.JaxDebugEnvWrapper import DebugEnvWrapper

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


def make_environment(rows=5, cols=5, evaluation: bool = None, num_agents: int = 2):

    return DebugEnvWrapper(
        DebugEnv(
            rows,
            cols,
            num_agents,
            reward_for_connection=1.0,
            reward_for_blocked=-1.0,
            reward_per_timestep=-1.0 / (rows + cols),
        )
    )


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    # Environment.
    environment_factory = functools.partial(
        make_environment,
    )

    def root_fn(forward_fn, params, key, env_state, observation):

        prior_logits, values = forward_fn(observations=observation, params=params)

        return RootFnOutput(
            prior_logits=prior_logits.logits,
            value=values,
            embedding=jax.tree_map(lambda x: jnp.expand_dims(x, 0), env_state),
        )

    def recurrent_fn(
        environment_model: DebugEnvWrapper,
        forward_fn,
        params,
        rng_key,
        action,
        env_state,
        agent_info,
    ) -> RecurrentFnOutput:

        agent_index = EntityId.from_string(agent_info).id

        actions = {f"type-0-id-{i}": 0 for i in range(environment_model.num_agents)}

        actions[agent_info] = jnp.squeeze(action)

        env_state = jax.tree_map(lambda x: jnp.squeeze(x, 0), env_state)

        next_state, timestep, _ = environment_model.step(env_state, actions)

        observation = environment_model.get_obs(next_state.grid)

        prior_logits, values = forward_fn(
            observations=observation[agent_index].reshape(1, -1), params=params
        )

        reward = timestep.reward[agent_info].reshape(
            1,
        )
        discount = timestep.discount[agent_info].reshape(
            1,
        )

        return (
            RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits.logits,
                value=values,
            ),
            jax.tree_map(lambda x: jnp.expand_dims(x, 0), next_state),
        )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return mamcts.make_default_networks(
            policy_layer_sizes=(254, 254, 254),
            critic_layer_sizes=(512, 512, 256),
            *args,
            **kwargs,
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
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
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
        sample_batch_size=256,
        num_minibatches=4,
        num_epochs=5,
        num_executors=1,
        multi_process=True,
        root_fn=root_fn,
        recurrent_fn=recurrent_fn,
        search=mctx.gumbel_muzero_policy,
        environment_model=environment_factory(),
        num_simulations=20,
        rng_seed=0,
        learning_rate=0.01,
        n_step=5,
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)
