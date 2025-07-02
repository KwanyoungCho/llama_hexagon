# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from typing import Callable, Dict, List

from qairt.api.configs.common import BackendType
from qairt.api.transformer.model_transformer_config import QuantizationStage
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations import (
    MHA2SHA,
    AttentionMask1Dto2D,
    OptimizeKVCache,
    ReplaceLinearWithConv,
    ReplacePosIdWithRoPE,
    SplitModel,
    UnpackQKV,
)

backend_to_transformation_map: Dict[BackendType, Dict[QuantizationStage, List[Callable]]] = {
    BackendType.HTP: {
        QuantizationStage.PRE_QUANT: [UnpackQKV, ReplaceLinearWithConv],
        QuantizationStage.POST_QUANT: [
            OptimizeKVCache,
            ReplacePosIdWithRoPE,
            # AttentionMask1Dto2D,
            SplitModel,
            MHA2SHA,
        ],
    },
    BackendType.CPU: {},
    BackendType.GPU: {},
    BackendType.LPAI: {},
}

transformation_name_to_config_attr = {
    "AttentionMask1Dto2D": "attention_mask_1d_to_2d",
    "ReplaceLinearWithConv": "replace_linear_with_conv",
    "ReplacePosIdWithRoPE": "replace_pos_id_with_rope",
    "UnpackQKV": "unpack_qkv",
    "OptimizeKVCache": "optimize_kv_cache",
    "MHA2SHA": "mha_to_sha",
    "SplitModel": "split_model",
}
