# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from numpy.typing import NDArray
from pydantic import ConfigDict, DirectoryPath, Field, FilePath, field_validator, model_validator
from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    InputSample,
    QuantizerInputArguments,
    RemoteHostDetails,
)
from qti.aisw.accuracy_debugger.framework_runner.framework_factory import get_framework_type
from qti.aisw.accuracy_debugger.inference_engine.qairt_inference_engine import (
    validate_backend_platform,
    validate_float_fallback,
)
from qti.aisw.accuracy_debugger.snooping.factory import get_snooper_class
from qti.aisw.accuracy_debugger.utils.constants import Algorithm
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    BackendType,
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
)
from qti.aisw.tools.core.modules.context_bin_gen import GenerateConfig
from qti.aisw.tools.core.modules.net_runner import InferenceConfig
from qti.aisw.tools.core.utilities.comparators.comparator import Comparator
from qti.aisw.tools.core.utilities.comparators.mse import MSEComparator
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class ModelSnooperInputConfig(AISWBaseModel):
    """Defines input arguments for Accuracy Debugger

    Attributes:
        input_model : Path to the source model/dlc/bin file.
        input_sample : List of InputSample objects or dictionary of tensor name to numpy array.
        algorithm: Algorithm to use to debug the model.
        converter_arguments: Input arguments required by the converter module.
        quantizer_arguments: Input arguments required by the quantizer module.
        context_bin_gen_arguments: Input arguments required by context_bin_gen module.
        context_bin_backend_extension: Backend extension config for context binary generator.
        offline_prepare: Boolean to indicate offline prepare of graph.
        net_run_arguments: Input arguments required by the netrunner module.
        net_run_backend_extension: Backend extension config for net-runner.
        comparators: List of comparators to use in verification stage.
        working_directory: Path to the directory to store the results.
        backend : Type of Backend.
        platform : Platform of target device like, android, x86_64_linux, etc.
        soc_model : Name of SOC model on target device.
        remote_host_details: Details of the remote host.
        golden_reference_path : Directory containing golden reference outputs.
        retain_compilation_artifacts: Flag to retain the compilation artifacts. Default is False.
        dump_output_tensors: Boolean to indicate whether to dump output tensors.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_model: FilePath
    input_sample: list[InputSample] | dict[str, NDArray]
    algorithm: Algorithm = Algorithm.ONESHOT
    converter_arguments: Optional[ConverterInputArguments] = None
    quantizer_arguments: Optional[QuantizerInputArguments] = None
    context_bin_gen_arguments: Optional[GenerateConfig] = None
    context_bin_backend_extension: Optional[FilePath | dict] = None
    offline_prepare: Optional[bool] = None
    net_run_arguments: Optional[InferenceConfig] = None
    net_run_backend_extension: Optional[FilePath | dict] = None
    comparators: List[Comparator] = [MSEComparator()]
    working_directory: Optional[DirectoryPath] = Field(default=None, validate_default=True)
    backend: BackendType
    platform: DevicePlatformType
    soc_model: str = ""
    remote_host_details: Optional[RemoteHostDetails] = None
    golden_reference_path: Optional[DirectoryPath] = None
    retain_compilation_artifacts: bool = False
    dump_output_tensors: bool = False

    @model_validator(mode="before")
    @classmethod
    def set_offline_prepare(cls, values):
        """If offline_prepare is None, set it based on backend."""
        backend = values.get("backend")
        offline_prepare = values.get("offline_prepare")

        # Enable offline prepare if backend supports.
        if offline_prepare is None and backend in BackendType.offline_preparable_backends():
            values["offline_prepare"] = True
        return values

    @field_validator("working_directory")
    @classmethod
    def validate_directory(cls, working_directory):
        """Validation and Initialization for the working directory"""
        if working_directory is None:
            working_directory = Path.cwd() / "working_directory"
            working_directory.mkdir(parents=True, exist_ok=True)
        return working_directory

    @field_validator("input_model", mode="after")
    @classmethod
    def validate_input_model(cls, input_model):
        """Validation for the type of input_model provided"""
        _ = get_framework_type(input_model)
        return input_model

    @model_validator(mode="after")
    def validate_backend_platform(self):
        """Validations for combination of backend and platform."""
        validate_backend_platform(self.backend, self.platform, self.offline_prepare)
        return self

    @model_validator(mode="after")
    def validate_float_fallback(self):
        """Validation for the float fallback"""
        validate_float_fallback(self.converter_arguments, self.quantizer_arguments)
        return self


class ModelSnooperOutputConfig(AISWBaseModel):
    """Defines Accuracy Debugger output format

    Attributes:
        snooping_report: Report generated by the snooping algorithm executed.
    """

    snooping_report: FilePath


class ModelSnooperSchemaV1(ModuleSchema):
    """Schema for accuracy debugger module."""

    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)
    _BACKENDS = None
    name: Literal["ModelSnooperModule"] = "ModelSnooperModule"
    path: Path = Path(__file__)
    arguments: ModelSnooperInputConfig
    outputs: Optional[ModelSnooperOutputConfig] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


class ModelSnooper(Module):
    """User interface class for accuracy debugger API."""

    _SCHEMA = ModelSnooperSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger: logging.Logger = None) -> None:
        """Initialize Debugger module

        Args:
            logger: A logger instance to be used by the ModelSnooper module
        """
        if logger is None:
            log_area = LogAreas.register_log_area("Accuracy Debugger")
            self.logger = QAIRTLogger.register_area_logger(area=log_area, level=logging.INFO)
        else:
            self.logger = logger
        super().__init__(logger)

    def run(self, config: ModelSnooperInputConfig) -> ModelSnooperOutputConfig:
        """Run the accuracy debugger module
        Args:
             config: Accuracy debugger input configuration.

        Returns:
             ModelSnooperOutputConfig: Accuracy debugger output configuration
             with the snooping report.
        """
        snooping_cls = get_snooper_class(config.algorithm)
        snooping_obj = snooping_cls(config.input_model, logger=self.logger)

        output = snooping_obj.run(
            input_tensors=config.input_sample,
            backend=config.backend,
            platform=config.platform,
            converter_args=config.converter_arguments,
            quantizer_args=config.quantizer_arguments,
            context_bin_args=config.context_bin_gen_arguments,
            context_bin_backend_extension=config.context_bin_backend_extension,
            offline_prepare=config.offline_prepare,
            net_runner_args=config.net_run_arguments,
            net_run_backend_extension=config.net_run_backend_extension,
            remote_host_details=config.remote_host_details,
            soc_model=config.soc_model,
            comparators=config.comparators,
            working_directory=config.working_directory,
            golden_reference_path=config.golden_reference_path,
            retain_compilation_artifacts=config.retain_compilation_artifacts,
            dump_output_tensors=config.dump_output_tensors,
        )
        output_config = ModelSnooperOutputConfig(snooping_report=output)
        return output_config

    def get_logger(self) -> Any:
        """This should return an instance of the logger that is used.

        Returns:
            The logger used by the module.
        """
        return self._logger

    def properties(self) -> Dict[str, Any]:
        return self._schema.model_json_schema()

    @property
    def _schema(self):
        return self._SCHEMA

    def enable_debug(self, debug_level: int, **kwargs) -> None:
        """Enable debugging behaviour for the module

        Args:
            debug_level: The level of debugging to be enabled.
        """
        # TODO: Update this method based on QAIRTLogging utility.
        pass
