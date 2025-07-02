# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import contextlib
from typing import Dict

import numpy as np
import torch

from qairt.api.converter.converter_config import CalibrationConfig, InputTensorConfig
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.modules.converter.converter_module import (
    InputTensorConfig as ModuleInputTensorConfig,
)
from qti.aisw.tools.core.modules.converter.converter_module import (
    OutputTensorConfig as ModuleOutputTensorConfig,
)

_convert_logger = get_logger("qairt.convert")


def _rename_arg(args: Dict, api_arg: str, module_arg: str) -> None:
    """
    Renames an argument from API naming to module naming.

    Args:
        args (Dict): The arguments dict containing the API arguments.
        api_arg (str): The name of the API argument.
        module_arg (str): The name of the module argument.

    Returns:
        None: Modifies args dictionary in-place
    """
    args[module_arg] = args.pop(api_arg)


def _input_tensor_config_arg_adapter(input_tensor_dict: Dict) -> ModuleInputTensorConfig:
    """Converts InputTensorConfig dict to ModuleInputTensorConfig object"""
    input_tensors_args_map = {
        "shape": "desired_input_shape",
        "datatype": "source_model_input_datatype",
        "layout": "source_model_input_layout",
    }

    # Rename input_tensor args
    for api_arg, module_arg in input_tensors_args_map.items():
        if api_arg in input_tensor_dict:
            _rename_arg(input_tensor_dict, api_arg, module_arg)

    # Handle desired_input_shape arg
    shape = input_tensor_dict.get("desired_input_shape")
    if shape:
        if isinstance(shape, torch.Size):
            shape = tuple(shape)
        tensor_rank = len(shape)
        shape = ",".join(str(dim) for dim in shape) if shape else None
        input_tensor_dict["desired_input_shape"] = shape

    # Handle source_model_input_datatype arg
    datatype = input_tensor_dict.get("source_model_input_datatype")
    if datatype:
        if isinstance(datatype, torch.dtype):
            input_tensor_dict["source_model_input_datatype"] = str(datatype).split(".")[-1]
        elif isinstance(datatype, np.dtype):
            input_tensor_dict["source_model_input_datatype"] = str(datatype)

    layout_aliases = {
        "channels_first": {3: "NCF", 4: "NCHW", 5: "NCDHW"},
        "channels_last": {3: "NFC", 4: "NHWC", 5: "NDHWC"},
    }

    layout = input_tensor_dict.get("source_model_input_layout")
    if (layout in layout_aliases) and (shape is None):
        _convert_logger.warning(
            f"{layout} cannot be resolved without a shape provided. Defaulting to None..."
        )
        input_tensor_dict["source_model_input_layout"] = None
    elif layout == "channels_first":
        input_tensor_dict["source_model_input_layout"] = layout_aliases["channels_first"][tensor_rank]
    elif layout == "channels_last":
        input_tensor_dict["source_model_input_layout"] = layout_aliases["channels_last"][tensor_rank]

    # Create module-level InputTensorConfig
    return ModuleInputTensorConfig(**input_tensor_dict)


def _converter_config_arg_adapter(extra_args: Dict) -> Dict:
    """Converts extra args names to converter module internal names"""

    converter_config_args_map = {
        "input_tensor_config": "input_tensors",
        "output_tensor_names": "output_tensors",
        "float_precision": "float_bitwidth",
        "float_bias_precision": "float_bias_bitwidth",
        "batch": "onnx_batch",
        "define_symbol": "onnx_define_symbol",
        "defer_loading": "onnx_defer_loading",
        "op_package_lib": "converter_op_package_lib",
    }

    for api_arg, module_arg in converter_config_args_map.items():
        if api_arg in extra_args:
            _rename_arg(extra_args, api_arg, module_arg)

    # Handle input_tensors args
    input_tensors = []
    for input_tensor in extra_args.get("input_tensors", []):
        module_input_tensor_config = _input_tensor_config_arg_adapter(input_tensor)
        input_tensors.append(module_input_tensor_config)
    if input_tensors:
        extra_args["input_tensors"] = input_tensors

    # Handle output_tensors args
    output_tensors = [ModuleOutputTensorConfig(name=name) for name in extra_args.get("output_tensors", [])]
    if output_tensors:
        extra_args["output_tensors"] = output_tensors

    return extra_args


def _calibration_config_arg_adapter(calibration_config: CalibrationConfig) -> Dict:
    """Converts calibration config attributes to quantizer module internal names"""

    quantizer_config_args_map = {
        "act_precision": "act_bitwidth",
        "bias_precision": "bias_bitwidth",
        "weights_precision": "weights_bitwidth",
        "param_calibration_method": "param_quantizer_calibration",
        "act_calibration_method": "act_quantizer_calibration",
        "per_channel_quantization": "use_per_channel_quantization",
        "per_row_quantization": "use_per_row_quantization",
        "per_row_quantization_bias": "enable_per_row_quantized_bias",
    }

    quantizer_args = calibration_config.model_dump()

    for api_arg, module_arg in quantizer_config_args_map.items():
        if api_arg in quantizer_args:
            _rename_arg(quantizer_args, api_arg, module_arg)

    for key in ["dataset", "batch_size", "num_of_samples"]:
        quantizer_args.pop(key)

    return quantizer_args


@contextlib.contextmanager
def disable_spawn_process_and_exec():
    """Apply temporary patch to spawn_process_and_exec and restore original method"""

    import qti.aisw.converters.common.utils.framework_utils as fw_utils

    def _spawn_process_and_exec_override(func, *args, **kwargs):
        kwargs.pop("process_name", "Process")
        res = func(*args, **kwargs)
        status = res is not None
        return status, res

    # Patch spawn_process_and_exec
    spawn_process_and_exec_name = "spawn_process_and_exec"
    original_spawn_process_and_exec = getattr(fw_utils, spawn_process_and_exec_name)
    patched_spawn_process_and_exec = _spawn_process_and_exec_override

    try:
        setattr(fw_utils, spawn_process_and_exec_name, patched_spawn_process_and_exec)
        yield

    finally:
        setattr(fw_utils, spawn_process_and_exec_name, original_spawn_process_and_exec)
