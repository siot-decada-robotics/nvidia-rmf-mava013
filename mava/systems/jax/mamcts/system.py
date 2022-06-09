# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Jax MAMCTS and MAMU systems."""
from typing import Any, Tuple

from mava.components.jax import building, executing, training, updating
from mava.specs import DesignSpec
from mava.systems.jax import System
from mava.systems.jax.mamcts.components.executing.action_selection import (
    MCTSFeedforwardExecutorSelectAction,
)
from mava.systems.jax.mamcts.components.extra.adder_priority import MuzeroAdderPriority
from mava.systems.jax.mamcts.components.extra.extra_specs import (
    ExtraLearnedSearchPolicySpec,
    ExtraSearchPolicySpec,
)
from mava.systems.jax.mamcts.components.reanalyse.distributor import (
    ReanalyseDistributor,
)
from mava.systems.jax.mamcts.components.reanalyse.reanalyze_components import (
    ReanalyseDataset,
    ReanalyseParameterClient,
    ReanalyseUpdate,
)
from mava.systems.jax.mamcts.components.training.losses import MAMCTSLoss, MAMULoss
from mava.systems.jax.mamcts.components.training.model_updating import (
    MAMCTSMinibatchUpdate,
    MAMUEpochUpdate,
    MAMUMinibatchUpdate,
)
from mava.systems.jax.mamcts.components.training.n_step_bootstrapped_returns import (
    NStepBootStrappedReturns,
)
from mava.systems.jax.mamcts.components.training.step import MAMCTSStep, MAMUStep
from mava.systems.jax.mamcts.config import MAMCTSDefaultConfig


class MAMCTSSystem(System):
    def design(self) -> Tuple[DesignSpec, Any]:
        """MAMCTS System - uses environment model."""

        # Set the default configs
        default_params = MAMCTSDefaultConfig()

        # Default system processes
        # System initialization
        system_init = DesignSpec(
            environment_spec=building.EnvironmentSpec, system_init=building.SystemInit
        ).get()

        # Default system processes
        # Executor
        executor_process = DesignSpec(
            executor_init=executing.ExecutorInit,
            executor_observe=executing.FeedforwardExecutorObserve,
            executor_select_action=MCTSFeedforwardExecutorSelectAction,
            executor_adder=building.ParallelSequenceAdder,
            executor_environment_loop=building.JAXParallelExecutorEnvironmentLoop,
            networks=building.DefaultNetworks,
        ).get()

        # Trainer
        trainer_process = DesignSpec(
            trainer_init=training.TrainerInit,
            n_step_fn=NStepBootStrappedReturns,
            loss=MAMCTSLoss,
            epoch_update=training.MAPGEpochUpdate,
            minibatch_update=MAMCTSMinibatchUpdate,
            sgd_step=MAMCTSStep,
            step=training.DefaultTrainerStep,
            trainer_dataset=building.TrajectoryDataset,
        ).get()

        # Data Server
        data_server_process = DesignSpec(
            data_server=building.OnPolicyDataServer,
            data_server_adder_signature=building.ParallelSequenceAdderSignature,
            extras_spec=ExtraSearchPolicySpec,
        ).get()

        # Parameter Server
        parameter_server_process = DesignSpec(
            parameter_server=updating.DefaultParameterServer,
            executor_parameter_client=building.ExecutorParameterClient,
            trainer_parameter_client=building.TrainerParameterClient,
        ).get()

        system = DesignSpec(
            **system_init,
            **data_server_process,
            **parameter_server_process,
            **executor_process,
            **trainer_process,
            distributor=building.Distributor,
            logger=building.Logger,
        )
        return system, default_params


class MAMUSystem(System):
    def design(self) -> Tuple[DesignSpec, Any]:
        """MAMCTS System that learns an environment model."""

        # Set the default configs
        default_params = MAMCTSDefaultConfig()

        # Default system processes
        # System initialization
        system_init = DesignSpec(
            environment_spec=building.EnvironmentSpec, system_init=building.SystemInit
        ).get()

        # Default system processes
        # Executor
        executor_process = DesignSpec(
            executor_init=executing.ExecutorInit,
            executor_observe=executing.FeedforwardExecutorObserve,
            executor_select_action=MCTSFeedforwardExecutorSelectAction,
            executor_adder=building.ParallelSequenceAdder,
            executor_environment_loop=building.JAXParallelExecutorEnvironmentLoop,
            networks=building.DefaultNetworks,
        ).get()

        # Trainer
        trainer_process = DesignSpec(
            trainer_init=training.TrainerInit,
            n_step_fn=NStepBootStrappedReturns,
            loss=MAMULoss,
            epoch_update=MAMUEpochUpdate,
            minibatch_update=MAMUMinibatchUpdate,
            sgd_step=MAMUStep,
            step=training.DefaultTrainerStep,
            trainer_dataset=building.TrajectoryDataset,
        ).get()

        # Data Server
        data_server_process = DesignSpec(
            adder_priority=MuzeroAdderPriority,
            extras_spec=ExtraLearnedSearchPolicySpec,
            rate_limiter=building.SampleToInsertRateLimiter,
            data_server=building.OffPolicyDataServer,
            data_server_adder_signature=building.ParallelSequenceAdderSignature,
        ).get()

        # Parameter Server
        parameter_server_process = DesignSpec(
            parameter_server=updating.DefaultParameterServer,
            executor_parameter_client=building.ExecutorParameterClient,
            trainer_parameter_client=building.TrainerParameterClient,
        ).get()

        system = DesignSpec(
            **system_init,
            **data_server_process,
            **parameter_server_process,
            **executor_process,
            **trainer_process,
            reanalyse_update=ReanalyseUpdate,
            reanalyse_dataset=ReanalyseDataset,
            reanalyse_parameter_client=ReanalyseParameterClient,
            distributor=ReanalyseDistributor,
            logger=building.Logger,
        )
        return system, default_params
