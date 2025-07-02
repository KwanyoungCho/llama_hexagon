# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""ConverterConfig class"""

from os import PathLike
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

try:
    from torch import Size, dtype
    from torch.utils.data import DataLoader, Dataset

    ShapeTypes = Union[Tuple[int, ...], Size]
    DatatypeTypes = Union[str, np.dtype, dtype]
    DatasetTypes = Union[str, PathLike, List, DataLoader, Dataset]
except ImportError:
    # mypy complains of multiple types assignments
    ShapeTypes = Union[Tuple[int, ...]]  # type: ignore
    DatatypeTypes = Union[str, np.dtype]  # type: ignore
    DatasetTypes = Union[str, PathLike, List]  # type: ignore

from typing_extensions import TypedDict

from qairt.api.configs.common import AISWBaseModel


class InputTensorConfig(TypedDict, total=False):
    """
    TypedDict of input tensor configuration. Any of the keys can be omitted, except name.
    """

    name: str
    """
    Name of tensor. This is required.
    """

    shape: ShapeTypes
    """
    Shape of input tensor. Default is None.
    """

    datatype: DatatypeTypes
    """
    Data type of input tensor. Default is float32.
    """

    layout: Optional[str]
    """
    Layout of each input tensor. Valid layouts include "channels_first", "channels_last", "NCDHW", "NDHWC",
    "NCHW", "NHWC", "HWIO", "OIHW", "NFC", "NCF", "NTF", "TNF", "NF", "NC", "F", "NONTRIVIAL".
    Default is None.
    """


class ConverterConfig(AISWBaseModel):
    """
    Pydantic class of parameters for model conversion.
    """

    # Override model_config for this class
    model_config = AISWBaseModel.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "extra": "allow"})

    input_tensor_config: Optional[List[InputTensorConfig]] = None
    """
    A list of input tensor configurations containing the name, shape, datatype, and layout of the input
    tensors.
    """

    output_tensor_names: Optional[List[str]] = None
    """
    A list of output tensor names.
    """

    float_precision: Literal[32, 16] = 32
    """
    The floating point precision to use for the model. Note the floating point precision will be applied to
    all tensors (including static tensors). Users should ensure that the precision is supported by each
    operation according to QAIRT spec.
    """

    float_bias_precision: Literal[16, 32] = 32
    """
    Option to select the precision to use for float bias tensor.
    """

    preserve_io_datatype: Optional[Union[str, List[str]]] = None
    """
    Set this option to maintain the source framework datatype for input and output tensors. This option
    is particularly useful when sequences of unsupported static operations are present. To preserve datatype
    for all input and output tensors, use preserve_io_datatype = "all". For select input and output tensors,
    use preserve_io_datatype = ["input1", "output1", ...]
    """

    onnx_simplification: bool = True
    """
    Do not attempt to simplify the model automatically. This may prevent some models from converting when
    sequences of unsupported static operations are present. Default is True.
    """

    batch: Optional[int] = None
    """
    The batch dimension override. This will take the first dimension of all inputs and treat it as a batch
    dim, overriding it with the value provided here.
    """

    define_symbol: Optional[List[Tuple[str, int]]] = None
    """
    Option to override specific input dimension symbols.
    """

    defer_loading: bool = False
    """
    Option to have the model not load weights. If False, the model will be loaded eagerly.
    """

    enable_framework_trace: bool = False
    """
    Use this option to enable converter to trace the o/p tensor change information.
    """

    op_package_config: Optional[List[str | PathLike]] = None

    """
    List of absolute paths to a Qnn Op Package XML configuration file that contains user defined custom
    operations.
    """

    op_package_lib: Optional[List[str | PathLike]] = None
    """
    List of absolute paths to converter op package library compiled by the OpPackage generator.
    """


class CalibrationConfig(AISWBaseModel):
    """
    Configuration for calibration process.
    """

    # Override model_config for this class
    model_config = AISWBaseModel.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "protected_namespaces": ()})

    dataset: Optional[DatasetTypes] = None
    """
    The dataset to be used for calibration.
        It can be a string, a PathLike object, or a list of datasets.
    """

    batch_size: int = 1
    """
    The size of the batch to be used during calibration.
            Default is 1.
    """

    num_of_samples: int = 512
    """
    The number of samples to be used for calibration.
            Default is 512.
    """

    act_precision: Literal[8, 16] = 8
    """
    Integer precision value to use while quantizing activations
    """

    bias_precision: Literal[8, 32] = 8
    """
    Precision value to use while quantizing biases
    """

    weights_precision: Literal[4, 8, 16] = 8
    """
    Precision value to use while quantizing weights
    """

    param_calibration_method: str = "min-max"
    """
    Calibration method to use for parameters. Valid methods are "min-max", "sqnr", "entropy", "mse", "percentile"
    """

    act_calibration_method: str = "min-max"
    """
    Calibration method to use for activations. Valid methods are "min-max", "sqnr", "entropy", "mse", "percentile"
    """

    per_channel_quantization: bool = True
    """
    Enable per channel quantization for convolution based op weights
    """

    per_row_quantization: bool = False
    """
    Enable per row quantization for Matmul and FullyConnected ops
    """

    per_row_quantization_bias: bool = False
    """
    Enable per row quantization of bias for FullyConnected ops, when weights are per-row quantized
    """
