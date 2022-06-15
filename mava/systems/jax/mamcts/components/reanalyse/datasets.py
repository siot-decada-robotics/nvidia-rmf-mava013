from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import reverb
import tensorflow as tf
from pygame import init

from mava.components.jax.building.datasets import (
    TrajectoryDataset,
    TrajectoryDatasetConfig,
)
from mava.core_jax import SystemBuilder
from mava.systems.jax.mamcts.components.reanalyse.base import ReanalyseComponent
from mava.systems.jax.mamcts.reanalyse_worker import ReanalyseWorker


@dataclass
class ReanalyseTrainerTrajectoryDatasetConfig(TrajectoryDatasetConfig):
    reanalyse_fraction: float = 0.0


class ReanalyseTrainerTrajectoryDataset(TrajectoryDataset):
    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """

        self.config.reanalyse_fraction = np.clip(self.config.reanalyse_fraction, 0, 1)

        actor_dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=builder.store.data_server_client.server_address,
            table=builder.store.trainer_id,
            max_in_flight_samples_per_worker=2 * self.config.sample_batch_size,
            num_workers_per_iterator=self.config.num_workers_per_iterator,
            max_samples_per_stream=self.config.max_samples_per_stream,
            rate_limiter_timeout_ms=self.config.rate_limiter_timeout_ms,
            get_signature_timeout_secs=self.config.get_signature_timeout_secs,
            # max_samples=self.config.max_samples,
        )
        if builder.store.num_reanalyse_workers > 0:
            reanalyse_dataset = reverb.TrajectoryDataset.from_table_signature(
                server_address=builder.store.data_server_client.server_address,
                table=f"{builder.store.trainer_id}_reanalyse",
                max_in_flight_samples_per_worker=2 * self.config.sample_batch_size,
                num_workers_per_iterator=self.config.num_workers_per_iterator,
                max_samples_per_stream=self.config.max_samples_per_stream,
                rate_limiter_timeout_ms=self.config.rate_limiter_timeout_ms,
                get_signature_timeout_secs=self.config.get_signature_timeout_secs,
                # max_samples=self.config.max_samples,
            )
            reanalyse_dataset = reanalyse_dataset.batch(
                self.config.sample_batch_size, drop_remainder=True
            )
        else:
            self.config.reanalyse_fraction == 0

        # Add batch dimension.
        actor_dataset = actor_dataset.batch(
            self.config.sample_batch_size, drop_remainder=True
        )

        if self.config.reanalyse_fraction > 0 and self.config.reanalyse_fraction < 1:
            dataset = tf.data.Dataset.sample_from_datasets(
                datasets=[actor_dataset, reanalyse_dataset],
                weights=[
                    1 - self.config.reanalyse_fraction,
                    self.config.reanalyse_fraction,
                ],
                seed=None,
                stop_on_empty_dataset=False,
            )

        elif self.config.reanalyse_fraction == 0:
            dataset = actor_dataset
        else:
            dataset = reanalyse_dataset

        builder.store.sample_batch_size = self.config.sample_batch_size

        builder.store.dataset_iterator = dataset.as_numpy_iterator()

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return ReanalyseTrainerTrajectoryDatasetConfig


@dataclass
class ReanalyseActorDatasetConfig:
    reanalyse_sample_batch_size: int = 32
    reanalyse_max_in_flight_samples_per_worker: int = 512
    reanalyse_num_workers_per_iterator: int = -1
    reanalyse_max_samples_per_stream: int = -1
    reanalyse_rate_limiter_timeout_ms: int = -1
    reanalyse_get_signature_timeout_secs: Optional[int] = None


class ReanalyseActorDataset(ReanalyseComponent):
    def __init__(
        self, config: ReanalyseActorDatasetConfig = ReanalyseActorDatasetConfig()
    ) -> None:
        self.config = config

    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        # Add duplicate table keys for reanalyse
        if builder.store.num_reanalyse_workers > 0:
            reanalyse_tables = {}
            for table_key in builder.store.table_network_config.keys():
                reanalyse_tables[
                    f"{table_key}_reanalyse"
                ] = builder.store.table_network_config[table_key]

            builder.store.table_network_config = {
                **builder.store.table_network_config,
                **reanalyse_tables,
            }

    def on_reanalyse_worker_init_start(self, reanalyse_worker: ReanalyseWorker) -> None:

        # Sample from actor table
        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=reanalyse_worker.store.data_server_client.server_address,
            table=reanalyse_worker.store.trainer_id,
            max_in_flight_samples_per_worker=2
            * self.config.reanalyse_sample_batch_size,
            num_workers_per_iterator=self.config.reanalyse_num_workers_per_iterator,
            max_samples_per_stream=self.config.reanalyse_max_samples_per_stream,
            rate_limiter_timeout_ms=self.config.reanalyse_rate_limiter_timeout_ms,
            get_signature_timeout_secs=self.config.reanalyse_get_signature_timeout_secs,
        )

        # Add batch dimension.
        dataset = dataset.batch(
            self.config.reanalyse_sample_batch_size, drop_remainder=True
        )
        reanalyse_worker.store.reanalyse_sample_batch_size = (
            self.config.reanalyse_sample_batch_size
        )

        reanalyse_worker.store.actor_dataset_iterator = dataset.as_numpy_iterator()

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "reanalyse_dataset"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """
        Optional class which specifies the
        dataclass/config object for the component.
        """
        return ReanalyseActorDatasetConfig
