import copy
import dataclasses
from typing import Dict

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from acme.jax import networks as networks_lib
from acme.jax import utils
from chex import dataclass

from mava.specs import MAEnvironmentSpec


@dataclasses.dataclass
class MixingNetwork:
    # TODO make more general mixing with bigger hypernet
    def __init__(
        self,
        hyper_w1_net,
        hyper_w1_params,
        hyper_w2_net,
        hyper_w2_params,
        hyper_b1_net,
        hyper_b1_params,
        hyper_b2_net,
        hyper_b2_params,
        output_dim: int,
    ) -> None:
        self.hyper_params = {
            "hyper_w1_params": hyper_w1_params,
            "hyper_w2_params": hyper_w2_params,
            "hyper_b1_params": hyper_b1_params,
            "hyper_b2_params": hyper_b2_params,
        }

        self.target_hyper_params = copy.deepcopy(self.hyper_params)

        self.hyper_w1_net = hyper_w1_net
        self.hyper_w2_net = hyper_w2_net
        self.hyper_b1_net = hyper_b1_net
        self.hyper_b2_net = hyper_b2_net

        self.output_dim = output_dim
        # TODO (sasha): params first
        def forward_fn(env_states, agent_q_values, hyper_params):
            agent_q_values = jnp.squeeze(agent_q_values)
            b, t, num_agents = agent_q_values.shape[:3]
            env_states = jnp.ones_like(env_states)
            agent_q_values = jnp.reshape(agent_q_values, (-1, 1, num_agents))
            env_states = jnp.reshape(env_states, (-1, env_states.shape[-1]))

            w1 = hyper_w1_net.apply(hyper_params["hyper_w1_params"], env_states)
            w1 = jnp.abs(jnp.reshape(w1, (-1, num_agents, self.output_dim)))

            b1 = hyper_b1_net.apply(hyper_params["hyper_b1_params"], env_states)
            b1 = jnp.reshape(b1, (-1, 1, self.output_dim))

            w2 = hyper_w2_net.apply(hyper_params["hyper_w2_params"], env_states)
            w2 = jnp.abs(jnp.reshape(w2, (-1, self.output_dim, 1)))

            b2 = hyper_b2_net.apply(hyper_params["hyper_b2_params"], env_states)
            b2 = jnp.reshape(b2, (-1, 1, 1))

            hidden = jax.nn.elu(jnp.matmul(agent_q_values, w1) + b1)
            output = jnp.matmul(hidden, w2) + b2
            return jnp.reshape(output, (b, t, 1))

        self.forward = forward_fn
