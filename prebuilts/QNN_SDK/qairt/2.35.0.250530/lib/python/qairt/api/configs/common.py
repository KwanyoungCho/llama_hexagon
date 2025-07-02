# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import warnings
from enum import Enum
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypeAlias, Union

import numpy as np
import numpy.typing as npt
from numpy import ndarray
from pydantic import BaseModel, ConfigDict, GetPydanticSchema, InstanceOf, computed_field
from pydantic.dataclasses import dataclass

from qti.aisw.tools.core.modules.api.definitions.common import (
    OpPackageIdentifier,
    ProfilingData,
    ProfilingLevel,
    ProfilingOption,
    Target,
)

# TODO: This file is a partial duplicate of qti.aisw.tools.core.modules.api.definitions.common.py
# It should be removed once the modules files are shared with the QAIRT SDK


class AISWBaseModel(BaseModel):
    """Internal variation of a BaseModel"""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_attribute_docstrings=True,
        protected_namespaces=(),
    )


class BackendType(str, Enum):
    """
    Enum representing backend types that are supported by a module.
    """

    CPU = "CPU"
    GPU = "GPU"
    HTP = "HTP"
    HTP_MCP = "HTP_MCP"
    AIC = "AIC"
    LPAI = "LPAI"

    @classmethod
    def offline_preparable_backends(cls):
        return [cls.HTP, cls.AIC, cls.HTP_MCP]

    @classmethod
    def quantizable_backends(cls):
        return [cls.CPU, cls.HTP, cls.AIC, cls.HTP_MCP, cls.LPAI]

    @classmethod
    def from_id(cls, backend_id: int) -> "BackendType":
        _ID_TO_BACKEND = {3: cls.CPU, 4: cls.GPU, 6: cls.HTP, 11: cls.HTP_MCP, 8: cls.AIC, 12: cls.LPAI}
        return _ID_TO_BACKEND[backend_id]

    @staticmethod
    def is_valid_backend(backend_str: str):
        return any(backend_str == backend for backend in BackendType._member_map_)

    @staticmethod
    def backend_to_id(backend_str: str):
        return BackendType(backend_str).id

    @property
    def id(self):
        _BACKEND_TO_ID = {"CPU": 3, "GPU": 4, "HTP": 6, "HTP_MCP": 11, "AIC": 8, "LPAI": 12}

        return _BACKEND_TO_ID[self._value_]


# --------------------------- Input Types --------------------- #
InputListInput = str | PathLike
NamedTensorMapping = Dict[str, np.ndarray]
ExecutionInputData = Union[
    InputListInput,
    np.ndarray,
    NamedTensorMapping,
]

# Code to enable validation on numpy array
PydanticNDArray: TypeAlias = Annotated[
    npt.NDArray, GetPydanticSchema(lambda _s, h: h(InstanceOf[np.ndarray]))
]


@dataclass
class ExecutionResult:
    """
    The result of executing a Model or CompiledModel object.
    """

    data: Optional[Dict[str, PydanticNDArray] | Sequence[Dict[str, PydanticNDArray]]] = None
    """ The output data from the execution. It contains a data field which is a dictionary of output names to
        numpy arrays or a sequence of dictionaries of output names to numpy arrays."""

    profiling_data: Optional[ProfilingData] = None
    """ The profiling data generated during execution."""

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, output_name: str) -> npt.NDArray:
        if self.data is None:
            raise TypeError("Cannot get item from data. Data is None.")

        # Legacy support for multiple inference
        if isinstance(self.data, Sequence):
            raise TypeError("Cannot get item from data of type: Sequence. Use self.data")

        return self.data[output_name]

    def __iter__(self):
        if self.data is None:
            raise TypeError("Cannot iterate through data. Data is None.")

        # Legacy support for multiple inference
        if isinstance(self.data, Sequence):
            warnings.warn(
                " Data is of type sequence. Iterator will return a sequence of dictionaries."
                " Use self.data for key, value pairs."
            )
            return iter(self.data)

        return iter(self.data.items())


class DspArchitecture(str, Enum):
    v66 = "v66"
    v68 = "v68"
    v69 = "v69"
    v73 = "v73"
    v75 = "v75"
    v79 = "v79"
    v81 = "v81"

    @classmethod
    def list_options(cls):
        """Returns a list of all DSP architecture options"""
        return [option.value for option in cls]


class PerfProfile(str, Enum):
    LOW_BALANCED = "low_balanced"
    BALANCED = "balanced"
    DEFAULT = "default"
    HIGH_PERFORMANCE = "high_performance"
    SUSTAINED_HIGH_PERFORMANCE = "sustained_high_performance"
    BURST = "burst"
    EXTREME_POWER_SAVER = "extreme_power_saver"
    LOW_POWER_SAVER = "low_power_saver"
    POWER_SAVER = "power_saver"
    HIGH_POWER_SAVER = "high_power_saver"
    SYSTEM_SETTINGS = "system_settings"
    NO_USER_INPUT = "no_user_input"
    CUSTOM = "custom"
    INVALID = "invalid"

    @classmethod
    def list_options(cls):
        """Returns a list of all performance profile options"""
        return [option.value for option in cls]
