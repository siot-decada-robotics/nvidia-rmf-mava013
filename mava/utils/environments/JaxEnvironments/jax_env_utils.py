from mava.utils.environments.JaxEnvironments.jax_cartpole import JaxCartPole
from mava.utils.environments.JaxEnvironments.jax_ma_waterworld import (
    MultiAgentWaterWorld,
)
from mava.utils.environments.JaxEnvironments.Jax_ma_waterworld_wrapper import (
    MultiAgentWaterworldWrapper,
)
from mava.utils.environments.JaxEnvironments.jax_slime_volley import SlimeVolley
from mava.utils.environments.JaxEnvironments.jax_slime_volley_wrapper import (
    SlimeVolleyWrapper,
)


def make_slimevolley_env(
    evaluation: bool = False,
    max_steps: int = 3000,
    is_multi_agent: bool = False,
    is_cooperative: bool = False,
):
    return SlimeVolleyWrapper(
        SlimeVolley(max_steps=max_steps),
        is_multi_agent=is_multi_agent,
        is_cooperative=is_cooperative,
    )


def make_ma_waterworld_env(
    evaluation: bool = False,
    num_agents: int = 16,
    num_items: int = 100,
    max_steps: int = 1000,
):
    return MultiAgentWaterworldWrapper(
        MultiAgentWaterWorld(
            num_agents=num_agents, num_items=num_items, max_steps=max_steps
        )
    )


def make_jax_cartpole(evaluation: bool = False):
    return JaxCartPole()
