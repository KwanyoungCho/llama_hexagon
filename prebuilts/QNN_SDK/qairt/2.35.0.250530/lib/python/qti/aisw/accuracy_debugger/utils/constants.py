# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from enum import Enum

from qti.aisw.tools.core.modules.api import BackendType
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.tools.core.utilities.framework.utils.constants import OnnxFrameworkInfo


class Algorithm(str, Enum):
    ONESHOT = "oneshot"
    LAYERWISE = "layerwise"
    CUMULATIVE = "cumulative_layerwise"


class Component(Enum):
    SNOOPING = "snooping"


class DataType(Enum):
    INTERMEDIATE_OUT_DATATYPE = "float32"


class MaxLimits(Enum):
    max_file_name_size = 255

    # Model has to fit in one DSP of 3.75 GB = 3840 MB
    # To be on safe side we will leave 840MB has buffer and use 3000MB
    max_model_size_with_intermediates = 3000


MATH_INVARIANT_OPS = [
    "cast",
    "constant",
    "reshape",
    "shape",
    "squeeze",
    "transpose",
    "unsqueeze",
    "maxpool",
    "flatten",
    "resize",
    "expand",
    "tile",
    "convert",
    "branch",
    "gather",
    "split",
    "compress",
    "stridedslice",
]

# Relu op types in onnx and tensorflow, if in future some new type is found -> add here
RELU_OPS = ["clip", "relu", "relu6", "leakyrelu", "prelu", "thresholdedrelu", "leaky_relu"]

supported_backends = [
    BackendType.AIC,
    BackendType.HTP,
    BackendType.CPU,
    BackendType.HTP_MCP,
    BackendType.GPU,
]

supported_platforms = [DevicePlatformType.ANDROID, DevicePlatformType.X86_64_LINUX]

supported_frameworks = [OnnxFrameworkInfo.name]
