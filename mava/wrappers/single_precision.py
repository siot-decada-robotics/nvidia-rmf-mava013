# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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


from typing import Any, Dict

import dm_env
import numpy as np
import tree
from acme import specs, types, wrappers
from acme.wrappers.single_precision import _convert_value

from mava import types as mava_types


class SinglePrecisionWrapper(wrappers.SinglePrecisionWrapper):
    """Wrapper which converts environments from double- to single-precision.

    Adapted version of https://github.com/deepmind/acme/blob/master/acme/wrappers/single_precision.py # noqa: E501
    to work with our marl envs.
    """

    def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        """Convert timestep to single precision.

        Args:
            timestep : standard timestep.

        Returns:
            single precision timestep.
        """
        # Extras
        if type(timestep) == tuple:
            timestep, extras = timestep
            return (
                timestep._replace(
                    reward=_convert_value(timestep.reward),
                    discount=_convert_value(timestep.discount),
                    observation=_convert_value(timestep.observation),
                ),
                _convert_value(extras),
            )
        else:
            return timestep._replace(
                reward=_convert_value(timestep.reward),
                discount=_convert_value(timestep.discount),
                observation=_convert_value(timestep.observation),
            )

    def extras_spec(self) -> Dict[str, specs.BoundedArray]:
        """Convert discount spec to single precision.

        Returns:
            single prec extra spec.
        """
        return _convert_spec(self._environment.extras_spec())

    def observation_spec(self) -> Dict[str, mava_types.OLT]:
        """Convert timestep to single precision.

        Returns:
            single prec obs sepc.
        """
        return _convert_spec(self._environment.observation_spec())


def _convert_spec(nested_spec: types.NestedSpec) -> types.NestedSpec:
    """Convert a nested spec."""

    def _convert_single_spec(spec: specs.Array) -> Any:
        """Convert a single spec."""
        if spec.dtype == "O":
            # Pass StringArray objects through unmodified.
            return spec
        if np.issubdtype(spec.dtype, np.float64):
            dtype = np.float32
        elif np.issubdtype(spec.dtype, np.int64):
            dtype = np.int32
        else:
            dtype = spec.dtype

        if hasattr(spec, "replace"):
            return spec.replace(dtype=dtype)
        else:
            # Usually for legal actions
            spec = np.dtype(dtype).type(spec)  # type: ignore
            return spec

    return tree.map_structure(_convert_single_spec, nested_spec)
