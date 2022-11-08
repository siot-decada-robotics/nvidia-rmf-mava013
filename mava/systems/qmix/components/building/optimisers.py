from dataclasses import dataclass
from typing import Optional

import optax
from optax._src import base as optax_base
from mava.components.building.optimisers import Optimisers

from mava.core_jax import SystemBuilder

@dataclass
class MixerOptimisersConfig:
    mixer_learning_rate: float = 1e-3
    mixer_adam_epsilon: float = 1e-5
    mixer_max_gradient_norm: float = 0.5
    mixer_optimiser: Optional[optax_base.GradientTransformation] = None

class MixerOptimiser(Optimisers):
    def __init__(
        self,
        config: MixerOptimisersConfig = MixerOptimisersConfig(),
    ):
        """Component defines the default way to initialise optimisers.

        Args:
            config: DefaultOptimisers.
        """
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Create and store the optimisers.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        # Build the optimiser function here
        if not self.config.mixer_optimiser:
            builder.store.mixer_optimiser = optax.chain(
                optax.clip_by_global_norm(self.config.mixer_max_gradient_norm),
                optax.scale_by_adam(eps=self.config.mixer_adam_epsilon),
                optax.scale(-self.config.mixer_learning_rate),
            )
        else:
            builder.store.mixer_optimiser = self.config.mixer_optimiser

    
    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "mixer_optimiser" 