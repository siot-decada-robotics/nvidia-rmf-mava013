import copy
from typing import List, Dict, Any, Callable, Optional

from chex import dataclass

from mava.components import Component
from mava.core_jax import SystemExecutor

# TODO (sasha): this is so much boilerplate for what this does, it probably doesn't need to be
#  a component
def find_extras(store: Any, keys: List[str]) -> Dict:
    """Finds information from the store related to the given keys

    Args:
        store: the store of the executor.
        keys: the keys to look for in the store.

    Returns:
        a dictionary with (modified) keys and the values in store.
    """
    user_defined_extras = {}
    for key in keys:
        key_in_store = copy.deepcopy(key)
        modified_key = copy.deepcopy(key)
        if key == "network_keys":
            modified_key = "network_int_keys"
            key_in_store = "network_int_keys_extras"
        # has no effect...
        # if key == "policy_info":
        #     key_in_store = "policy_info"
        value = store.__getattribute__(key_in_store)
        # value = store.extras_spec[key_in_store]
        user_defined_extras.update({modified_key: value})
    return user_defined_extras


@dataclass
class ExtrasFinderConfig:
    extras_finder: Callable[[Any, List[str]], Dict] = find_extras


class ExtrasFinder(Component):
    def __init__(self, config: ExtrasFinderConfig = ExtrasFinderConfig()):
        """Creating Extras from Store of the executor at its current state."""
        self.config = config

    def on_execution_init(self, executor: SystemExecutor) -> None:
        """The function for finding extras that are added to store."""
        executor.store.extras_finder = self.config.extras_finder

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "extras_finder"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Returns class config."""
        return ExtrasFinderConfig
