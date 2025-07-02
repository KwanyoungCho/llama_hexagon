# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from .base import WorkerGroup
from .onnxrt import OnnxRTWorkerGroup
from .qairt import QAIRTWorkerGroup
from .tensorflow import TensorflowRTWorkerGroup
from .torchscript import TorchScriptWorkerGroup

__all__ = ["WorkerGroup", "OnnxRTWorkerGroup", "QAIRTWorkerGroup", "TensorflowRTWorkerGroup","TorchScriptWorkerGroup" ]