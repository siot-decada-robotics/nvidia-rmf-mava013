from dataclasses import dataclass
from typing import Callable

import jax

from mava import constants
from mava.components import Component
from mava.core_jax import SystemBuilder


@dataclass
class MixerInitConfig:
    # no good default here should we assert this is never None?
    mixer_factory: Callable = None


class MixerInit(Component):
    def __init__(self, config: MixerInitConfig = MixerInitConfig()) -> None:
        super().__init__(config)

    def on_building_init_start(self, builder: SystemBuilder):
        # Following convention, but these two methods could probably just be one
        builder.store.base_key, network_key = jax.random.split(builder.store.base_key)

        builder.store.mixer_factory = lambda: self.config.mixer_factory(
            environment_spec=builder.store.ma_environment_spec,
            agent_net_keys=builder.store.agent_net_keys,
            base_key=network_key,
        )

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        builder.store.mixing_net = builder.store.mixer_factory()

        # Wrap opt_states in a mutable type (dict) since optax return an immutable tuple
        builder.store.mixer_opt_state = {
            constants.OPT_STATE_DICT_KEY: builder.store.mixer_optimiser.init(
                builder.store.mixing_net.hyper_params
            )
        }

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "mixer_init"
