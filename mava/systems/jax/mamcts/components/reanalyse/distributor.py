from dataclasses import dataclass
from typing import Callable, Optional

import jax

from mava.components.jax.building.distributor import Distributor, DistributorConfig
from mava.core_jax import SystemBuilder
from mava.systems.jax.launcher import Launcher, NodeType
from mava.systems.jax.mamcts.reanalyse_worker import ReanalyseWorker


@dataclass
class ReanalyseDistributorConfig(DistributorConfig):
    num_reanalyse_workers: int = 1


class ReanalyseDistributor(Distributor):
    def __init__(
        self, config: ReanalyseDistributorConfig = ReanalyseDistributorConfig()
    ):
        if isinstance(config.nodes_on_gpu, str):
            config.nodes_on_gpu = [config.nodes_on_gpu]
        self.config = config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.store.program = Launcher(
            multi_process=self.config.multi_process,
            nodes_on_gpu=self.config.nodes_on_gpu,
            name=self.config.distributor_name,
            terminal=self.config.terminal,
        )

        # tables node
        data_server = builder.store.program.add(
            builder.data_server,
            node_type=NodeType.reverb,
            name="data_server",
        )

        # variable server node
        parameter_server = builder.store.program.add(
            builder.parameter_server,
            node_type=NodeType.corrier,
            name="parameter_server",
        )
        distributor_key = jax.random.PRNGKey(self.config.rng_seed)
        distributor_key, executor_seed_key, evaluator_seed_key = jax.random.split(
            distributor_key, 3
        )
        executor_seeds = jax.random.randint(
            executor_seed_key,
            (self.config.num_executors,),
            minval=-jax.numpy.inf,
            maxval=jax.numpy.inf,
        )

        # executor nodes
        for executor_id in range(self.config.num_executors):
            builder.store.program.add(
                builder.executor,
                [
                    f"executor_{executor_id}",
                    data_server,
                    parameter_server,
                    executor_seeds[executor_id],
                ],
                node_type=NodeType.corrier,
                name="executor",
            )

        if self.config.run_evaluator:
            evaluator_seed = jax.random.randint(
                evaluator_seed_key, (), minval=-jax.numpy.inf, maxval=jax.numpy.inf
            )
            # evaluator node
            builder.store.program.add(
                builder.executor,
                ["evaluator", data_server, parameter_server, evaluator_seed],
                node_type=NodeType.corrier,
                name="evaluator",
            )

        # trainer nodes
        for trainer_id in builder.store.trainer_networks.keys():
            builder.store.program.add(
                builder.trainer,
                [trainer_id, data_server, parameter_server],
                node_type=NodeType.corrier,
                name="trainer",
            )

        for reanalyse_id in range(self.config.num_reanalyse_workers):
            for trainer_id in builder.store.trainer_networks.keys():
                # Reanalyse Node
                builder.store.program.add(
                    ReanalyseWorker,
                    [
                        reanalyse_id,
                        trainer_id,
                        data_server,
                        parameter_server,
                        builder.store,
                        builder.store.reanalyse_components,
                    ],
                    node_type=NodeType.corrier,
                    name="reanalyse_worker",
                )

        if not self.config.multi_process:
            builder.store.system_build = builder.store.program.get_nodes()

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return ReanalyseDistributorConfig
