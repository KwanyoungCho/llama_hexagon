# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import logging
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import qti.aisw.core.model_level_api as mlapi
from pydantic import Field, field_validator
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    BackendType,
    Model,
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
    OpPackageIdentifier,
    ProfilingData,
    ProfilingLevel,
    ProfilingOption,
    Target,
    expect_module_compliance,
)
from qti.aisw.tools.core.modules.api.definitions.common import QNNCommonConfig
from qti.aisw.tools.core.modules.api.utils.configure_backend import create_backend, get_supported_backends
from qti.aisw.tools.core.modules.api.utils.model_level_api import create_mlapi_model

from qti.aisw.tools.core.utilities.qairt_logging import QAIRTLogger


ContextBinaryGeneratorCacheKey = Tuple[BackendType, str, str]

class GenerateConfig(QNNCommonConfig):
    """
    Defines supported context binary generation parameters that are implemented by backend-agnostic
    tools, therefore are applicable to all backends.
    """
    enable_intermediate_outputs: Optional[bool] = None
    op_packages: Optional[List[OpPackageIdentifier]] = None
    input_output_tensor_mem_type: Optional[str] = 'raw'


class ContextBinGenArgConfig(AISWBaseModel):
    """
    Defines all possible arguments for context binary generation.

    Backend-specific parameters should be passed via config file or config dict. Any duplicate
    arguments specified in the config dict will override identical arguments in the config file.

    If target is not provided, the backend will choose a sane default based on its typical
    workflows, e.g. QNN HTP will generate on the host by default, but QNN GPU will generate on
    Android since offline preparation is not supported.
    """
    backend: BackendType
    backend_config_file: Optional[Union[str, PathLike]] = None
    backend_config_dict: Optional[Dict[str, Any]] = None
    target: Optional[Target] = None
    model: Model | List[Model]
    output_dir: Optional[Union[str, PathLike]] = "./output/"
    output_filename: Optional[str] = None
    backend_specific_filename: Optional[str] = None
    generate_config: Optional[GenerateConfig] = Field(default_factory=GenerateConfig)

    @field_validator('model', mode="after")
    @classmethod
    def validate_model_type(cls, value: Model | List[Model]) -> 'Model':
        if isinstance(value, list) and not all(value_.dlc_path for value_ in value):
            raise ValueError("Multiple models are only supported for DLCs")
        models = [value] if isinstance(value, Model) else value
        for model in models:
            if model.context_binary_path:
                raise ValueError('Cannot generate a context binary from an existing context binary')
        return value


class ContextBinGenOutputConfig(AISWBaseModel):
    """
    Defines context binary generation output format
    """
    context_binary: Model
    backend_binary_path: Optional[str] = None
    profiling_data: Optional[ProfilingData] = None

    @field_validator('context_binary', mode="after")
    @classmethod
    def validate_model_type(cls, value: Model) -> 'Model':
        if not value.context_binary_path:
            raise ValueError('Context binary generation output must be a context binary')
        return value


class ContextBinGenModuleSchema(ModuleSchema):
    _BACKENDS = get_supported_backends()
    _VERSION = ModuleSchemaVersion(major=0, minor=6, patch=0)

    name: Literal["ContextBinGenModule"] = "ContextBinGenModule"
    path: Path = Path(__file__)
    arguments: ContextBinGenArgConfig
    outputs: Optional[ContextBinGenOutputConfig] = None
    backends: List[str] = _BACKENDS


@expect_module_compliance
class ContextBinGen(Module):
    _SCHEMA = ContextBinGenModuleSchema

    def __init__(self, persistent: bool = False, logger: logging.Logger = None):
        """
        Initializes a ContextBinGen module instance

        Args:
            persistent (bool): Indicates that objects initialized for a particular
            (backend, target, model) tuple should persist between calls to generate() so setup steps
            (e.g. pushing backend + model artifacts to a remote device) only need to be performed
            once instead of each time a binary is generated.

            logger (any): A logger instance to be used by the ContextBinGen module
        """
        if logger:
            self._logger = QAIRTLogger.get_logger("ContextBinGenLogger", parent_logger=logger)
        else:
            self._logger = QAIRTLogger.get_logger(
                "ContextBinGenLogger", level="INFO", formatter_val="extended", handler_list=["dev_console"]
            )
        self._persistent: bool = persistent
        self._context_binary_generator_cache: Dict[ContextBinaryGeneratorCacheKey,
                                                   mlapi.ContextBinaryGenerator] = {}

    def properties(self) -> Dict[str, Any]:
        return self._SCHEMA.model_json_schema()

    def get_logger(self) -> Any:
        return self._logger

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        pass

    def generate(self, config: ContextBinGenArgConfig) -> ContextBinGenOutputConfig:
        """
        Generates a context binary from a given model using the specified backend on the provided
        target.

        Args:
            config (ContextBinGenArgConfig): Arguments of context binary generation where a model,
            backend, and target are specified, as well as any other miscellaneous parameters.

        Returns:
            ContextBinGenOutputConfig: A structure containing a path to the generated context binary
            as an instance of a Model with the context_binary_path populated.

        Examples: The example below generates a context binary for HTP on x86 linux
            >>> model = Model(dlc_path="/path/to/dlc")
            >>> generate_config = GenerateConfig()
            >>> context_bin_gen_arg_config = ContextBinGenArgConfig(backend=BackendType.HTP,
            >>>                                                     model=model,
            >>>                                                     output_dir='./htp_output/',
            >>>                                                     generate_config=generate_config)
            >>> context_bin_gen = ContextBinGen()
            >>> output_config = context_bin_gen.generate(context_bin_gen_arg_config)
        """
        self._logger.debug(f'Generating context with config: {config}')

        # check if a ContextBinaryGenerator with the same (backend, target, model) tuple is in the
        # cache to avoid redundant setup done during ContextBinaryGenerator initialization
        target_str = config.target.model_dump_json() if config.target else ""

        # Get string for config.model (used to create generator cache key below)
        if isinstance(config.model, list):
            model_dump_hash = f"{(model_.dump_json() for model_ in config.model)}"
        else:
            model_dump_hash = config.model_dump_json()

        context_binary_generator_cache_key = (config.backend,
                                              target_str,
                                              model_dump_hash)
        context_bin_gen = \
            self._context_binary_generator_cache.get(context_binary_generator_cache_key)
        if context_bin_gen is None:
            backend_instance = create_backend(config.backend,
                                                 config.target,
                                                 config.backend_config_file,
                                                 config.backend_config_dict)

            # TODO: To keep existing behavior with minimal change, setting mlapi_models to list
            # or single model. This can be confusing long term, as such it should be changed as part
            # of https://jira-dc.qualcomm.com/jira/browse/AISW-121075
            if not isinstance(config.model, list):
                mlapi_models = create_mlapi_model(config.model)
            else:
                mlapi_models = [create_mlapi_model(model_) for model_ in config.model]

            op_packages = config.generate_config.op_packages if config.generate_config else None
            if op_packages:
                for op_package in op_packages:
                    backend_instance.register_op_package(op_package.package_path,
                                                      op_package.interface_provider,
                                                      op_package.target,
                                                      op_package.cpp_stl_path)
            context_bin_gen = mlapi.ContextBinaryGenerator(backend_instance, mlapi_models)
            if self._persistent:
                # if persistence enabled, store the ContextBinaryGenerator in a dict so it is not
                # garbage collected which would undo the initialization
                self._context_binary_generator_cache[context_binary_generator_cache_key] = \
                    context_bin_gen

        output_binary, output_backend_binary_path = context_bin_gen.generate(output_path=config.output_dir,
                                                 output_filename=config.output_filename,
                                                 backend_specific_filename=config.backend_specific_filename,
                                                 config=config.generate_config)

        profiling_data = context_bin_gen.get_profiling_data()
        profiling_data = profiling_data[0] if profiling_data else None

        profiling_output = None
        if profiling_data is not None:
            profiling_output = ProfilingData(
                profiling_log=profiling_data.profiling_log,
                backend_profiling_artifacts=profiling_data.backend_profiling_artifacts
            )

        context_binary = Model(context_binary_path=output_binary.binary_path)
        return ContextBinGenOutputConfig(context_binary=context_binary,
                                         backend_binary_path=output_backend_binary_path,
                                         profiling_data=profiling_output)
