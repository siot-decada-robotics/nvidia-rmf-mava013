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

"""Jax MAMCTS system."""
from mava.systems.jax.mamcts.components.executing.action_selection import (
    MCTSFeedforwardExecutorSelectAction,
)
from mava.systems.jax.mamcts.components.extra.adder_priority import MuzeroAdderPriority
from mava.systems.jax.mamcts.components.extra.extra_specs import (
    ExtraLearnedSearchPolicySpec,
    ExtraSearchPolicySpec,
)
from mava.systems.jax.mamcts.components.training.losses import MAMCTSLoss, MAMULoss
from mava.systems.jax.mamcts.components.training.model_updating import (
    MAMCTSMinibatchUpdate,
    MAMUBatch,
    MCTSBatch,
)
from mava.systems.jax.mamcts.components.training.n_step_bootstrapped_returns import (
    NStepBootStrappedReturns,
)
from mava.systems.jax.mamcts.components.training.step import MAMCTSStep, MAMUStep
from mava.systems.jax.mamcts.networks import (
    make_default_mamcts_networks,
    make_default_mamu_networks,
)
from mava.systems.jax.mamcts.system import MAMCTSSystem, MAMUSystem
