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

"""Jax MADQN system."""
from typing import Any, Tuple

from mava.components import building, executing, training, updating
from mava.components.executing.base import ExecutorTargetNetInit
from mava.specs import DesignSpec
from mava.systems import System
from mava.systems.madqn.components import executing as dqn_executing
from mava.systems.madqn.components import training as dqn_training
from mava.systems.madqn.components.utils import ExtrasActionInfo
from mava.systems.madqn.components.extras_finder import ExtrasFinder
from mava.systems.madqn.config import MADQNDefaultConfig


class MADQNSystem(System):
    def design(self) -> Tuple[DesignSpec, Any]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        # Set the default configs
        default_params = MADQNDefaultConfig()

        system_init = DesignSpec(
            environment_spec=building.EnvironmentSpec,
            system_init=building.FixedNetworkSystemInit,
        ).get()

        # Default system processes
        # Executor
        executor_process = DesignSpec(
            executor_init=executing.ExecutorInit,
            executor_target_network_init=ExecutorTargetNetInit,
            extras_finder=ExtrasFinder,
            executor_observe=executing.FeedforwardExecutorObserve,
            executor_select_action=dqn_executing.FeedforwardExecutorSelectActionValueBased,
            executor_environment_loop=building.ParallelExecutorEnvironmentLoop,
            executor_scheduler=executing.EpsilonScheduler,
            networks=building.DefaultNetworks,
        ).get()

        # Trainer
        trainer_process = DesignSpec(
            trainer_init=training.SingleTrainerInit,
            step=training.DefaultTrainerStep,
            sgd_step=dqn_training.MADQNStep,
            loss=dqn_training.MADQNLoss,
            epoch_update=dqn_training.MADQNEpochUpdate,
            trainer_dataset=building.TransitionDataset,
            optimisers=building.SingleOptimiser,  # for ppo this is in executor, which seems weird
        ).get()

        # Data Server
        data_server_process = DesignSpec(
            adder_priority=building.adders.UniformAdderPriority,
            data_server=building.OffPolicyDataServer,
            executor_adder=building.ParallelTransitionAdder,
            data_server_rate_limiter=building.SampleToInsertRateLimiter,
            data_server_adder_signature=building.ParallelTransitionAdderSignature,
            extras_spec=ExtrasActionInfo,
            # data_server_sampler=building.reverb_components.UniformSampler,
            data_server_sampler=building.reverb_components.PrioritySampler,
            data_server_remover=building.reverb_components.FIFORemover,
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
