from typing import Callable, Optional

from chex import dataclass

from mava.components.jax.building.environments import (
    ExecutorEnvironmentLoopConfig,
    ParallelExecutorEnvironmentLoop,
)
from mava.core_jax import SystemBuilder
from mava.wrappers.offline_environment_logger import MAOfflineEnvironmentSequenceLogger


@dataclass
class EvaluatorOfflineLoggingConfig(ExecutorEnvironmentLoopConfig):
    offline_sequence_length: int = 1000
    offline_sequence_period: int = 1000
    offline_logdir: str = "./offline_env_logs"
    offline_label: str = "offline_logger"
    offline_min_sequences_per_file: int = 100


class EvaluatorOfflineLogging(ParallelExecutorEnvironmentLoop):
    def __init__(
        self,
        config: EvaluatorOfflineLoggingConfig = EvaluatorOfflineLoggingConfig(),
    ):
        """Component logs evaluator trajectories to create offline datasets.

        Args:
            config: EvaluatorOfflineLoggingConfig.
        """
        self.config = config

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """Log sequences of experience to file.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        env = builder.store.global_config.environment_factory(
            evaluation=False
        )  # type: ignore

        if builder.store.is_evaluator:
            env = MAOfflineEnvironmentSequenceLogger(
                environment=env,
                sequence_length=self.config.offline_sequence_length,
                period=self.config.offline_sequence_period,
                logdir=self.config.offline_logdir,
                label=self.config.offline_label,
                min_sequences_per_file=self.config.offline_min_sequences_per_file,
            )

        builder.store.executor_environment = env

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return EvaluatorOfflineLoggingConfig
