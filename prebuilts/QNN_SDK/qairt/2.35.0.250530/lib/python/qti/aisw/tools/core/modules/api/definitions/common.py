# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DeviceCredentials,
    DevicePlatformType,
    RemoteDeviceIdentifier,
)


"""
This module contains the definition for AISWBaseModel class which is a pydantic class derived
from BaseModel, and AISWVersion which is a pydantic class that stores fields that categorize a
semantic versioning scheme.
"""


class AISWBaseModel(BaseModel):
    """Internal variation of a BaseModel"""

    model_config = ConfigDict(extra='forbid', validate_assignment=True, protected_namespaces=())


class OpPackageIdentifier(AISWBaseModel):
    """Defines the custom op package parameters for the net runner module
    """
    package_path: Union[str, PathLike]
    interface_provider: str
    target_name: Optional[str] = None
    cpp_stl_path: Optional[Union[str, PathLike]] = None


class AISWVersion(AISWBaseModel):
    """A dataclass that conveys when modifications are made to a module's interface
    or its properties.
    """

    _MAJOR_VERSION_MAX = 15
    _MINOR_VERSION_MAX = 40
    _PATCH_VERSION_MAX = 15
    _PRE_RELEASE_MAX_LENGTH = 26

    major: int = Field(ge=0, le=_MAJOR_VERSION_MAX)  # Backwards incompatible changes to a module
    minor: int = Field(ge=0, le=_MINOR_VERSION_MAX)  # Backwards compatible changes
    patch: int = Field(ge=0, le=_PATCH_VERSION_MAX)  # Backwards compatible bug fixes
    pre_release: str = Field(default="", max_length=_PRE_RELEASE_MAX_LENGTH)

    @model_validator(mode='after')
    def check_allowed_sem_ver(self):
        """Sanity checks a version to ensure it is not all zeros

        Raises:
            ValueError if no version is set
        """
        if self.major == self.minor == self.patch == 0:
            raise ValueError(f'Version: {self.__repr__()} is not allowed')
        return self

    def __str__(self):
        """Formats the version as a string value: "major.minor.patch"
        or "major.minor.patch" if the release tag is set
        """
        if not self.pre_release:
            return f'{self.major}.{self.minor}.{self.patch}'
        return f'{self.major}.{self.minor}.{self.patch}-{self.pre_release}'


class QNNCommonConfig(AISWBaseModel):
    """Specifies the shared parameters supported by both the Context-bin generator and the Net Runner Module.

    Attributes:
        log_level: Specifies max logging level to be set
        set_output_tensors: Specifies a comma-separated list of intermediate output tensor names, for which the outputs
                                              will be written in addition to final graph output tensors
        profiling_level: Option to Enable Profiling
        profiling_option: Option to Set profiling options
        platform_options: Specifies values to pass as platform options
    """
    log_level: Optional[str] = None
    set_output_tensors: Optional[List[str]] = None
    profiling_level: Optional[str] = None
    profiling_option: Optional[str] = None
    platform_options: Optional[Union[str, Dict[str, str]]] = None


class BackendType(str, Enum):
    """Enum representing backend types that are supported by a module.
    """

    CPU = 'CPU'
    GPU = 'GPU'
    HTP = 'HTP'
    HTP_MCP = 'HTP_MCP'
    AIC = 'AIC'
    LPAI = 'LPAI'

    @classmethod
    def offline_preparable_backends(cls):
        return [cls.HTP, cls.AIC, cls.HTP_MCP, cls.LPAI]

    @classmethod
    def quantizable_backends(cls):
        return [cls.CPU, cls.HTP, cls.AIC, cls.HTP_MCP, cls.LPAI]


class Target(AISWBaseModel):
    """Defines the type of device to be used by the module, optionally including device identifiers
    and connection parameters for remote devices.

    Attributes:
        type (DevicePlatformType): The type of device platform to be used
        identifier (Optional[RemoteDeviceIdentifier]): The identifier of the device.
                                                        Defaults to None.
        credentials (Optional[DeviceCredentials]): The credentials for the device. Defaults to
        None.
        soc_model (Optional[str]): The soc name of the device ex: SA8295. Defaults to
        None.
    """
    type: DevicePlatformType
    identifier: Optional[RemoteDeviceIdentifier] = None
    credentials: Optional[DeviceCredentials] = None
    soc_model: Optional[str] = None


class Model(AISWBaseModel):
    """Describes a model in a form that can be consumed by a module. Only one field should be set based
    on the model's format

    Attributes:
        qnn_model_library_path (Optional[Union[str, PathLike]]: Path to a QNN model library
        context_binary_path (Optional[Union[str, PathLike]]): Path to a context binary
        dlc_path (Optional[Union[str, PathLike]]): Path to a DLC
    """
    qnn_model_library_path: Optional[Union[str, PathLike]] = None
    context_binary_path: Optional[Union[str, PathLike]] = None
    dlc_path: Optional[Union[str, PathLike]] = None

    @model_validator(mode='after')
    def validate_one_field_set(self) -> 'Model':
        """Validates that only a single model type can be provided per Model instance.

        Args:
            self (Model): The instance of Model.

        Returns:
            Model: The instance of Model if it contains a single model type.

        Raises:
            ValueError: If no model types or > 1 model type was provided
        """
        if len(self.model_fields_set) != 1:
            raise ValueError("Exactly one Model field must be set")
        return self


class ProfilingLevel(str, Enum):
    """Enum representing profiling levels that are supported by a module.
    """
    BASIC = 'basic'
    DETAILED = 'detailed'
    CLIENT = 'client'
    BACKEND = 'backend'


class ProfilingOption(str, Enum):
    """Enum representing profiling options that are supported by a module.
    """
    OPTRACE = 'optrace'


class ProfilingData(AISWBaseModel):
    """Defines a module's profiling output.

    Attributes:
        profiling_log (Path): A path to the generated profiling log.
        backend_profiling_artifacts: An optional list of paths to any backend-specific profiling
                                     artifacts that were generated.
    """
    profiling_log: PathLike
    backend_profiling_artifacts: Optional[List[Path]] = None
