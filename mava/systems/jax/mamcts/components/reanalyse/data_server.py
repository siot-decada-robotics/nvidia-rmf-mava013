from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import reverb

from mava import specs
from mava.components.jax.building.data_server import OffPolicyDataServer, OffPolicyDataServerConfig
from mava.components.jax.building.reverb import Remover, Sampler, SamplerConfig
from mava.components.jax.component import Component
from mava.core_jax import SystemBuilder

@dataclass
class ReanalyseOffPolicyDataServer(OffPolicyDataServerConfig):
    reanalyse_max_size: int = 100000
    reanalyse_max_times_sampled: int = 1

class ReanalyseOffPolicyDataServer(OffPolicyDataServer):
    def table(
        self,
        table_key: str,
        environment_spec: specs.MAEnvironmentSpec,
        extras_spec: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """_summary_

        Args:
            table_key : _description_
            environment_spec : _description_
            extras_spec : _description_
            builder : _description_
        Returns:
            _description_
        """
        if builder.store.__dict__.get("sequence_length"):
            signature = builder.store.adder_signature_fn(
                environment_spec, builder.store.sequence_length, extras_spec
            )
        else:
            signature = builder.store.adder_signature_fn(environment_spec, extras_spec)

        # Check to see if the table is a reanalyse table
        reanalyse: bool = table_key.split("_")[-1] == "reanalyse"

        table = reverb.Table(
            name=table_key,
            sampler=builder.store.sampler_fn()
            if not reanalyse
            else builder.store.reanalyse_sampler_fn(),
            remover=builder.store.remover_fn()
            if not reanalyse
            else builder.store.reanalyse_remover_fn(),
            max_size=self.config.max_size if not reanalyse else self.config.reanalyse_max_size,
            rate_limiter=builder.store.rate_limiter_fn(),
            signature=signature,
            max_times_sampled=self.config.max_times_sampled if not reanalyse else self.config.reanalyse_max_times_sampled,
        )
        return table

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return ReanalyseOffPolicyDataServer
    

class ReanalyseSampler(Sampler):
    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "data_server_reanalyse_sampler"


class LIFOReanalyseSampler(ReanalyseSampler):
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """_summary_"""

        def sampler_fn() -> reverb.selectors:
            return reverb.selectors.Lifo()

        builder.store.reanalyse_sampler_fn = sampler_fn

class FIFOReanalyseSampler(ReanalyseSampler):
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """_summary_"""

        def sampler_fn() -> reverb.selectors:
            return reverb.selectors.Fifo()

        builder.store.reanalyse_sampler_fn = sampler_fn

class UniformReanalyseSampler(ReanalyseSampler):
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """_summary_"""

        def sampler_fn() -> reverb.selectors:
            return reverb.selectors.Uniform()

        builder.store.reanalyse_sampler_fn = sampler_fn


class ReanalyseRemover(Remover):
    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "data_server_reanalyse_remover"


class FIFOReanalyseRemover(ReanalyseRemover):
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """_summary_"""

        def remover_fn() -> reverb.selectors:
            return reverb.selectors.Fifo()

        builder.store.reanalyse_remover_fn = remover_fn
