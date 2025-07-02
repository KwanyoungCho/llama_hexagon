# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
ExecutionConfig class.
"""

from typing import Dict, List, Literal, Optional, Union

from pydantic import field_validator, model_validator

from qairt.api.configs import DevicePlatformType
from qairt.api.configs.common import AISWBaseModel, PerfProfile
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.modules.api import OpPackageIdentifier, ProfilingLevel, ProfilingOption

_execute_config_logger = get_logger("qairt.execute")

supported_initialize_execute_platforms = {
    DevicePlatformType.X86_64_LINUX,
    DevicePlatformType.X86_64_WINDOWS_MSVC,
    DevicePlatformType.WOS,
}


class ExecutionConfig(AISWBaseModel):
    """
    Pydantic class of parameters for model execution
    """

    debug: Optional[bool] = None
    """
    Specifies that output from all layers of the network will be saved.
    """

    use_native_output_data: Optional[bool] = None
    """
    Specifies that the output files will be generated
    in the data type native to the graph or in floating point.
    """

    use_native_input_data: Optional[bool] = None
    """
    Specifies that the input files will be parsed in the data type
    native to the graph. If not specified, input files will be parsed in floating point.
    Note that options use_native_input_data and native_input_tensor_names are mutually exclusive.
    """

    native_input_tensor_names: Optional[List[str]] = None
    """
    List of input tensor names,for which the input files
    would be read/parsed in native format. Note that
    options use_native_input_data and native_input_tensor_names are mutually exclusive.
    """

    op_packages: Optional[List[OpPackageIdentifier]] = None

    """
    Provide a comma-separated list of op packages, interface
    providers, and, optionally, targets to register. Valid values
    for target are CPU and HTP.
    """

    perf_profile: Optional[PerfProfile] = None
    """
    Specifies performance profile to be used. Valid settings are
    low_balanced, balanced, default, high_performance, sustained_high_performance, burst,
    low_power_saver, power_saver, high_power_saver, extreme_power_saver and system_settings.
    """

    synchronous: Optional[bool] = None
    """
    Specifies the way graphs should be executed.
    """

    batch_multiplier: Optional[str] = None
    """
    Specifies the value with which the batch value in input and
    output tensors dimensions will be multiplied. The modified input and output tensors will be
    used only during the execute graphs.
    """

    set_output_tensors: Optional[List[str]] = None
    """
    List of intermediate output tensor names,
    for which the outputs will be written in addition to final graph output tensors.
    Note that options debug and set_output_tensors are mutually exclusive.
    """

    platform_options: Optional[Union[str, Dict[str, str]]] = None
    """
    Specifies values to pass as platform options.
    """

    profiling_level: Optional[ProfilingLevel] = None
    """ Profiling levels: options are "basic", "backend", "detailed" and "client".
        This field should be set within a profiler context. """

    profiling_option: Optional[Literal["optrace"]] = None
    """ Profiling options: "optrace. This field should be set within a profiler context. """

    use_mmap: Optional[bool] = False
    """
    Specifies whether to use mmap for memory allocation.
    """

    log_level: Optional[str] = None
    """
    Log level for the executor. Standard logging levels are supported.
    """

    duration: Optional[float] = None
    """
    Specifies the duration of the graph execution in seconds.
    Loops over the input_list until this amount of time has transpired.
    """

    num_inferences: Optional[int] = None
    """
    Specifies the number of inferences. Loops over the input_list until
    the specified number of inferences has transpired.
    """

    @model_validator(mode="after")
    def _validate_num_inference_and_duration_exclusivity(self):
        """Validate that num_inferences and duration are not set simultaneously."""
        if self.num_inferences and self.duration:
            raise ValueError("num_inferences and duration cannot be set simultaneously.")
        return self

    @model_validator(mode="after")
    def _validate_input_tensor_arguments(self):
        """Validate that use_native_input_files and native_input_tensor_names are not
        set simultaneously."""
        if self.use_native_input_data and self.native_input_tensor_names:
            raise ValueError(
                "use_native_input_files and native_input_tensor_names cannot be set simultaneously."
            )
        return self

    @model_validator(mode="after")
    def _validate_set_output_tensors_and_debug_arguments(self):
        """Validate that set_output_tensors and debug are not set simultaneously."""
        if self.set_output_tensors and self.debug:
            self.debug = False
            _execute_config_logger.warning(
                "The 'set_output_tensors' and 'debug' option cannot be set simultaneously. "
                "'debug' option has been changed to false."
            )
        return self

    @field_validator("log_level", mode="after")
    @classmethod
    def set_default_log_level(cls, value):
        if value is None:
            # Import here to avoid global import issues
            from qti.aisw.tools.core.utilities.qairt_logging import QAIRTLogger

            return QAIRTLogger.get_default_logging_level("qairt.compile").lower()
        return value
