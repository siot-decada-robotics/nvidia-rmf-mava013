from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import rlax

from mava.components.jax.training import Loss
from mava.components.jax.training.losses import MAPGTrustRegionClippingLossConfig
from mava.core_jax import SystemTrainer
from mava.systems.jax.mat.components.training.utils import (
    merge_agents_to_sequence,
    unmerge_agents_to_sequence,
)


@dataclass
class MatLossConfig(MAPGTrustRegionClippingLossConfig):
    num_actions: int = 5  # TODO better way to get this


class MatLoss(Loss):
    def __init__(
        self,
        config: MatLossConfig = MatLossConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        print("ON TRAINING LOSS FN!")

        def loss_grad_fn(
            params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            behaviour_log_probs: Dict[str, jnp.ndarray],
            target_values: Dict[str, jnp.ndarray],
            advantages: Dict[str, jnp.ndarray],
            behavior_values: Dict[str, jnp.ndarray],
            terminals: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios."""

            grads = {}
            loss_info = {}

            agent_key = trainer.store.trainer_agents[0]
            agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
            network = trainer.store.networks["networks"][agent_net_key]

            def loss_fn(
                params: Any,
                observations: Any,
                actions: jnp.ndarray,
                behaviour_log_probs: jnp.ndarray,
                target_values: jnp.ndarray,
                advantages: jnp.ndarray,
                behavior_values: jnp.ndarray,
                terminals: jnp.ndarray,
            ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                # print(f"loss fn actions shape 3:{actions.shape}")

                # setting up actions to be passed through model as previous actions
                prev_action_shape = (*actions.shape, self.config.num_actions + 1)
                one_hot_actions = jax.nn.one_hot(actions, self.config.num_actions)
                prev_actions = jnp.zeros(prev_action_shape)
                prev_actions.at[:, 0, 0].set(1)
                prev_actions.at[:, :, 1:].set(one_hot_actions)

                # [idea] TODO (sasha): how would it work if we used the old encoded observation
                #         which were calculated in the step fn when calculating behaviour vals
                values, encoded_obs = network.encoder.network.apply(
                    params["encoder"], observations
                )
                values = jnp.squeeze(values)

                distribution_params = network.decoder.network.apply(
                    params["decoder"], prev_actions, encoded_obs
                )

                # print(f"distp:{distribution_params.shape}")
                # copy ppo logp + entropy funcs
                log_probs = distribution_params.log_prob(actions)
                entropy = distribution_params.entropy()

                batch_size = 5
                num_agents = 2
                num_non_terminal = jnp.sum(terminals)

                # TODO instead of doing all this stuff just pass in terminals and mask it all out
                # unmerge batch and sequence dim and merge sequence and agent dim
                # log_probs = jnp.ravel(
                #     merge_agents_to_sequence(
                #         jnp.reshape(log_probs, (batch_size, -1, num_agents))
                #     )[:, :-1]
                # )
                # entropy = jnp.ravel(
                #     merge_agents_to_sequence(
                #         jnp.reshape(entropy, (batch_size, -1, num_agents))
                #     )[:, :-1]
                # )
                # behaviour_log_probs = jnp.ravel(
                #     merge_agents_to_sequence(
                #         jnp.reshape(
                #             behaviour_log_probs,
                #             (batch_size, -1, num_agents),
                #         )
                #     )[:, :-1]
                # )
                # behavior_values = jnp.ravel(
                #     merge_agents_to_sequence(
                #         jnp.reshape(behavior_values, (batch_size, -1, num_agents))
                #     )[:, :-1]
                # )
                #
                # behavior_values = jnp.ravel(
                #     merge_agents_to_sequence(
                #         jnp.reshape(behavior_values, (batch_size, -1, num_agents))
                #     )[:, :-1]
                # )
                # # TODO this isn't working it should == 0
                # check = merge_agents_to_sequence(
                #     jnp.reshape(advantages, (batch_size, -1, num_agents))
                # )

                # TODO (sasha): check in the middle of this proccess that once you get
                #  (batch, sequence * num_agents) that x[:,-1]==0
                # advantages = jnp.ravel(
                #     merge_agents_to_sequence(
                #         jnp.reshape(advantages, (batch_size, -1, num_agents))
                #     )[:, :-1]
                # )
                # values = jnp.ravel(
                #     merge_agents_to_sequence(
                #         jnp.reshape(values, (batch_size, -1, num_agents))
                #     )[:, :-1]
                # )
                # target_values = jnp.ravel(
                #     merge_agents_to_sequence(
                #         jnp.reshape(target_values, (batch_size, -1, num_agents))
                #     )[:, :-1]
                # )

                log_probs = jnp.ravel(log_probs)
                behaviour_log_probs = jnp.ravel(behaviour_log_probs)
                advantages = jnp.ravel(advantages)
                terminals = jnp.ravel(terminals)
                behavior_values = jnp.ravel(behavior_values)
                values = jnp.ravel(values)
                target_values = jnp.ravel(target_values)
                entropy = jnp.ravel(entropy)

                print(
                    f"--------------LOSS--------------\n"
                    f"adv:{advantages.shape}\n"
                    f"vals:{values.shape}\n"
                    f"bvals:{behavior_values.shape}\n"
                    f"ent:{entropy.shape}\n"
                    f"lp:{log_probs.shape}\n"
                    f"blp:{behaviour_log_probs.shape}\n"
                    f"targ v:{target_values.shape}\n"
                    f"term:{terminals.shape}\n"
                    f"n non term:{num_non_terminal.shape}"
                )


                # Compute importance sampling weights:
                # current policy / behavior policy.
                rhos = jnp.exp(log_probs - behaviour_log_probs)
                clipping_epsilon = self.config.clipping_epsilon

                # TODO this means, but we have zeros!
                policy_loss = rlax.clipped_surrogate_pg_loss(
                    rhos, advantages * terminals, clipping_epsilon
                )

                # Value function loss. Exclude the bootstrap value
                # TODO both of these need to be masked
                unclipped_value_error = target_values - values
                unclipped_value_loss = unclipped_value_error**2

                if self.config.clip_value:
                    # Clip values to reduce variablility during critic training.
                    clipped_values = behavior_values + jnp.clip(
                        values - behavior_values,
                        -clipping_epsilon,
                        clipping_epsilon,
                    )
                    clipped_value_error = target_values - clipped_values
                    clipped_value_loss = clipped_value_error**2
                    clipped_value_loss = clipped_value_loss * terminals
                    value_loss = jnp.fmax(unclipped_value_loss, clipped_value_loss)

                else:
                    value_loss = unclipped_value_loss

                value_loss = jnp.sum(value_loss) / num_non_terminal

                # Entropy regulariser.
                entropy_loss = -jnp.sum(entropy * terminals) / num_non_terminal

                print(policy_loss.shape, value_loss.shape, entropy_loss.shape)
                total_loss = (
                    policy_loss
                    + value_loss * self.config.value_cost
                    + entropy_loss * self.config.entropy_cost
                )

                loss_info = {
                    "loss_total": total_loss,
                    "loss_policy": policy_loss,
                    "loss_value": value_loss,
                    "loss_entropy": entropy_loss,
                }

                # loss = (num_agents,) therefore need to reduce it
                # TODO (sasha): Options:
                #  [x] mean and apply 3 times - didn't learn -1.5
                #  [ ] mean and apply once
                #  [ ] sum and apply once
                #  [x] index and apply 3 times - learnt, but not well -0.9
                #  [ ] flatten and apply once -> try this next
                #  [ ] change adv calc to diff btw agent's V and sum of prev agent's Vs
                #  [x] put grads in a dict {"encoder":grad,"decoder":grad} for optax.update
                return total_loss, loss_info

            grad, loss_info = jax.grad(loss_fn, has_aux=True)(
                params[agent_net_key],
                observations.observation,
                actions,
                behaviour_log_probs,
                target_values,
                advantages,
                behavior_values,
                terminals,
            )

            return grad, loss_info

        # Save the gradient function.
        trainer.store.grad_fn = loss_grad_fn

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MatLossConfig
