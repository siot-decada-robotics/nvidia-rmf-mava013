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

"""Jax IPPO system."""
from typing import Any, Tuple

from mava.components.jax import building, executing, training, updating
from mava.components.jax.building.guardrails import ComponentDependencyGuardrails
from mava.specs import DesignSpec
from mava.systems.jax import System
from mava.systems.jax.ippo.components import ExtrasLogProbSpec
from mava.systems.jax.ippo.config import IPPODefaultConfig


class IPPOSystem(System):
    @staticmethod
    def design() -> Tuple[DesignSpec, Any]:
        """System design for IPPO with single optimiser.

        Args:
            None.

        Returns:
            system: design spec for IPPO
            default_params: default IPPO configuration
        """
        # Set the default configs
        default_params = IPPODefaultConfig()

        # Default system processes
        # System initialization
        system_init = DesignSpec(
            environment_spec=building.EnvironmentSpec,
            system_init=building.FixedNetworkSystemInit,
        ).get()

        # Executor
        executor_process = DesignSpec(
            executor_init=executing.ExecutorInit,
            executor_observe=executing.FeedforwardExecutorObserve,
            executor_select_action=executing.FeedforwardExecutorSelectAction,
            executor_adder=building.ParallelSequenceAdder,
            adder_priority=building.UniformAdderPriority,
            executor_environment_loop=building.ParallelExecutorEnvironmentLoop,
            networks=building.DefaultNetworks,
        ).get()

        # Trainer
        trainer_process = DesignSpec(
            trainer_init=training.SingleTrainerInit,
            gae_fn=training.GAE,
            loss=training.MAPGWithTrustRegionClippingLoss,
            epoch_update=training.MAPGEpochUpdate,
            minibatch_update=training.MAPGMinibatchUpdate,
            sgd_step=training.MAPGWithTrustRegionStep,
            step=training.DefaultTrainerStep,
            trainer_dataset=building.TrajectoryDataset,
        ).get()

        # Data Server
        data_server_process = DesignSpec(
            data_server=building.OnPolicyDataServer,
            data_server_adder_signature=building.ParallelSequenceAdderSignature,
            extras_spec=ExtrasLogProbSpec,
        ).get()

        # Parameter Server
        parameter_server_process = DesignSpec(
            parameter_server=updating.DefaultParameterServer,
            executor_parameter_client=building.ExecutorParameterClient,
            trainer_parameter_client=building.TrainerParameterClient,
            termination_condition=updating.CountConditionTerminator,
        ).get()

        system = DesignSpec(
            **system_init,
            **data_server_process,
            **parameter_server_process,
            **executor_process,
            **trainer_process,
            distributor=building.Distributor,
            logger=building.Logger,
            component_dependency_guardrails=ComponentDependencyGuardrails,
        )
        return system, default_params


class IPPOSystemSeparateNetworks(System):
    @staticmethod
    def design() -> Tuple[DesignSpec, Any]:
        """System design for PPO with separate policy and critic networks.

        Returns:
            system callback components, default system parameters
        """

        # Get the generic IPPO system setup.
        system, default_params = IPPOSystem.design()

        # Update trainer components with seperate networks
        # TODO (dries): Investigate whether the names (below) are necessary or if they can be removed.
        system.set("loss", training.MAPGWithTrustRegionClippingLossSeparateNetworks)
        system.set("minibatch_update", training.MAPGMinibatchUpdateSeparateNetworks)
        system.set("sgd_step", training.MAPGWithTrustRegionStepSeparateNetworks)
        system.set("epoch_update", training.MAPGEpochUpdateSeparateNetworks)

        # Update parameter server components with seperate networks
        # TODO (dries): See if we can somehow reuse the same parameter client and server components
        # as is used in the shared networks system. We can then remove the below components.
        system.set("parameter_server", updating.ParameterServerSeparateNetworks)
        system.set(
            "executor_parameter_client",
            building.ExecutorParameterClientSeparateNetworks,
        )
        system.set(
            "trainer_parameter_client", building.TrainerParameterClientSeparateNetworks
        )

        return system, default_params

class IPPOSystemRecurrentPolicy(System):
    def design(self) -> Tuple[DesignSpec, Any]:
        """System design for PPO with a recurrent policy.

        Returns:
            system callback components, default system parameters
        """

        # Get the IPPOSystemSeparateNetworks system setup.
        system, default_params = IPPOSystemSeparateNetworks.design()

        # Update trainer components with seperate networks
        system.set("executor_select_action", executing.RecurrentExecutorSelectAction)
        system.set("executor_observe", executing.RecurrentExecutorObserve)

        return system, default_params
