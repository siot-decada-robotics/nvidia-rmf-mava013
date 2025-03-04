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

"""Wrapper classes for Mava systems."""
from mava.wrappers.debugging_envs import DebuggingEnvWrapper
from mava.wrappers.env_wrappers import ParallelEnvWrapper
from mava.wrappers.environment_loop_wrappers import (
    DetailedEpisodeStatistics,
    DetailedPerAgentStatistics,
    MonitorParallelEnvironmentLoop,
)
from mava.wrappers.pettingzoo import PettingZooParallelEnvWrapper

try:
    # The user might not have installed Flatland
    from mava.wrappers.flatland import FlatlandEnvWrapper
except ModuleNotFoundError:
    pass

try:
    # The user might not have installed SMAC
    from mava.wrappers.smac import SMACWrapper
except ModuleNotFoundError:
    pass

from mava.wrappers.saveable import SaveableWrapper
from mava.wrappers.system_trainer_statistics import (
    DetailedTrainerStatistics,
    ScaledDetailedTrainerStatistics,
)
