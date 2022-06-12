from functools import partial

import chex
import jax
import jax.numpy as jnp

from mava.components.jax.building.adders import AdderPriority
from mava.core_jax import SystemBuilder


class MuzeroAdderPriority(AdderPriority):
    """Muzero adder priority. Sets up the priority function when adding new data to the replay buffer."""

    @partial(jax.jit, static_argnames=["self"])
    def muzero_priority(self, step) -> chex.Array:
        """Calculates the sequence priority.

        Args:
            step : a reverb step.

        Returns:
            the priority for the sequence.
        """
        extras = step.extras["policy_info"]
        priority = 0
        for agent in extras:
            priority += jnp.abs(
                extras[agent]["search_values"] - extras[agent]["predicted_values"]
            )
        priority = jnp.max(priority, -1)
        return priority

    def on_building_executor_adder_priority(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.store.priority_fns = {
            table_key: lambda step: self.muzero_priority(step)
            for table_key in builder.store.table_network_config.keys()
        }
