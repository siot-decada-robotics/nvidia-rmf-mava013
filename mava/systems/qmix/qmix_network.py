import copy
import dataclasses
from typing import Dict

from acme.jax import networks as networks_lib
import jax.numpy as jnp
import jax
import distrax
import haiku as hk
from mava.specs import EnvironmentSpec

from acme.jax import utils


@dataclasses.dataclass
class HyperParams:
    hyper_w1_params: networks_lib.Params
    hyper_w2_params: networks_lib.Params
    hyper_b1_params: networks_lib.Params
    hyper_b2_params: networks_lib.Params


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
        self.hyper_params = HyperParams(
            hyper_w1_params,
            hyper_w2_params,
            hyper_b1_params,
            hyper_b2_params,
        )
        self.target_hyper_params = copy.deepcopy(self.hyper_params)

        self.hyper_w1_net = hyper_w1_net
        self.hyper_w2_net = hyper_w2_net
        self.hyper_b1_net = hyper_b1_net
        self.hyper_b2_net = hyper_b2_net

        self.output_dim = output_dim

        def forward_fn(env_states, agent_q_values, hyper_params: HyperParams):
            b, t, num_agents = agent_q_values.shape[:3]

            agent_q_values = jnp.reshape(agent_q_values, (-1, 1, num_agents))
            env_states = jnp.reshape(env_states, (-1, env_states.shape[-1]))

            w1 = hyper_w1_net.apply(hyper_params.hyper_w1_params, env_states)
            w1 = jnp.abs(jnp.reshape(w1, (-1, num_agents, self.output_dim)))

            b1 = hyper_b1_net.apply(hyper_params.hyper_b1_params, env_states)
            b1 = jnp.reshape(b1, (-1, 1, self.output_dim))

            w2 = hyper_w2_net.apply(hyper_params.hyper_w2_params, env_states)
            w2 = jnp.abs(jnp.reshape(w2, (-1, self.output_dim, 1)))

            b2 = hyper_b2_net.apply(hyper_params.hyper_b2_params, env_states)
            b2 = jnp.reshape(b2, (-1, 1, 1))

            hidden = jax.nn.elu(jnp.matmul(agent_q_values, w1) + b1)
            output = jnp.matmul(hidden, w2) + b2
            return jnp.reshape(output, (b, t, 1))

        self.forward = forward_fn


# @dataclasses.dataclass
# class QmixNetwork:
#     def __init__(
#         self,
#         policy_params,
#         policy_init_state,
#         policy_network,
#         # mixing_network: MixingNetwork,
#     ) -> None:
#         self.policy_params: networks_lib.Params = policy_params
#         self.target_policy_params: networks_lib.Params = copy.deepcopy(policy_params)
#         self.policy_init_state = policy_init_state
#         self.policy_network: networks_lib.FeedForwardNetwork = policy_network
#         # self.mixing_network: MixingNetwork = mixing_network
#
#         def forward_fn(
#             policy_params: Dict[str, jnp.ndarray],
#             policy_state: Dict[str, jnp.ndarray],
#             observations: networks_lib.Observation,
#         ):
#             return self.policy_network.apply(
#                 policy_params, [observations, policy_state]
#             )
#
#         self.forward = forward_fn
#
#     # TODO possibly create a container for all params
#     def mixing(self, hyper_params: HyperParams, env_states, agent_q_values):
#         return self.mixing_network.forward(
#             env_states,
#             agent_q_values,
#             hyper_params.hyper_w1_params,
#             hyper_params.hyper_w2_params,
#             hyper_params.hyper_b1_params,
#             hyper_params.hyper_b2_params,
#         )
#
#     def get_action(
#         self, params, policy_state, observations, epsilon, base_key, mask: jnp.array
#     ):
#         q_values, new_policy_state = self.forward(params, policy_state, observations)
#         masked_q_values = jnp.where(mask == 1.0, q_values, -99999)  # todo
#
#         greedy_actions = masked_q_values == jnp.max(masked_q_values)
#         greedy_actions_probs = greedy_actions / jnp.sum(greedy_actions)
#
#         random_action_probs = mask / jnp.sum(mask)
#
#         combined_probs = (
#             1 - epsilon
#         ) * greedy_actions_probs + epsilon * random_action_probs
#
#         action_dist = distrax.Categorical(probs=combined_probs)
#         return action_dist.sample(seed=base_key), new_policy_state
#
#     def get_params(
#         self,
#     ) -> Dict[str, jnp.ndarray]:
#         """Return current params.
#
#         Returns:
#             policy and target policy params.
#         """
#         return {
#             "policy_network": self.policy_params,
#             "target_policy_network": self.target_policy_params,
#             "hyper_params": self.mixing_network.hyper_params,
#             "target_hyper_params": self.mixing_network.target_hyper_params,
#         }
#
#     def get_init_state(self):
#         return self.policy_init_state


def make_mixing_network(
    environment_spec: EnvironmentSpec,
    hidden_dim: int,
    output_dim: int,
    num_agents: int,
    network_key,
) -> MixingNetwork:
    @hk.without_apply_rng
    @hk.transform
    def hyper_w1_net(env_states):
        net = hk.Sequential(
            [hk.Linear(hidden_dim), jax.nn.relu, hk.Linear(output_dim * num_agents)]
        )
        return net(env_states)

    @hk.without_apply_rng
    @hk.transform
    def hyper_w2_net(env_states):
        net = hk.Sequential([hk.Linear(hidden_dim), jax.nn.relu, hk.Linear(output_dim)])
        return net(env_states)

    @hk.without_apply_rng
    @hk.transform
    def hyper_b1_net(env_states):
        return hk.Linear(output_dim)(env_states)

    @hk.without_apply_rng
    @hk.transform
    def hyper_b2_net(env_states):
        net = hk.Sequential([hk.Linear(output_dim), jax.nn.relu, hk.Linear(1)])
        return net(env_states)

    dummy_env_state = utils.zeros_like(environment_spec.extras_spec()["s_t"])

    hyper_w1_params = hyper_w1_net.init(network_key, dummy_env_state)  # type: ignore
    hyper_w2_params = hyper_w2_net.init(network_key, dummy_env_state)  # type: ignore
    hyper_b1_params = hyper_b1_net.init(network_key, dummy_env_state)  # type: ignore
    hyper_b2_params = hyper_b2_net.init(network_key, dummy_env_state)  # type: ignore

    return MixingNetwork(
        hyper_w1_net=hyper_w1_net,
        hyper_w1_params=hyper_w1_params,
        hyper_w2_net=hyper_w2_net,
        hyper_w2_params=hyper_w2_params,
        hyper_b1_net=hyper_b1_net,
        hyper_b1_params=hyper_b1_params,
        hyper_b2_net=hyper_b2_net,
        hyper_b2_params=hyper_b2_params,
        output_dim=output_dim,
    )
