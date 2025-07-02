# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
Module for preparing a model for inference using the Genie Composer module.
"""

import pathlib
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Optional

from qairt.api.compiled_model import CompiledModel
from qairt.api.configs.common import BackendType
from qairt.modules.cache_module import CacheModule
from qairt.utils.loggers import get_logger
from qti.aisw.genai.qnn_genai_transformer_composer_backend import (
    GGMLFileType,
    OutputFile,
    Params,
    convert_model_names,
    convert_to_output_type,
    getConfigFromSDK,
    load_some_model,
)

logger = get_logger(__name__)


class QuantizationLevel(Enum):
    """An enum for the quantization levels supported by the Genie Composer module."""

    Z4 = "Z4"
    Z4_FP32 = "Z4_FP32"
    Z4_FP16 = "Z4_FP16"
    Z4_BF16 = "Z4_BF16"
    Q4 = "Q4"
    Z8 = "Z8"
    FP32 = "FP32"


class GeniePreparationModule:
    """A class for preparing a model for inference using the Genie Composer module."""

    QUANTIZATION_LEVEL_TO_GGML_FILE_TYPE = {
        QuantizationLevel.Z4: GGMLFileType.MostlyZ4,
        QuantizationLevel.Z4_FP16: GGMLFileType.Z4_FP16,
        QuantizationLevel.Z4_BF16: GGMLFileType.Z4_BF16,
        QuantizationLevel.Q4: GGMLFileType.MostlyQ4_0_32,
        QuantizationLevel.Z8: GGMLFileType.MostlyZ8,
        QuantizationLevel.FP32: GGMLFileType.AllF32,
    }

    @classmethod
    def prepare(
        cls,
        framework_model_path: str | PathLike,
        quantization_level: QuantizationLevel = QuantizationLevel.FP32,
        outfile: str | PathLike = pathlib.Path.cwd(),
    ) -> CompiledModel:
        """
        Prepare a model for inference using the Genie Composer module.

        Args:
            framework_model_path (Path): path to the framework model to be prepared
            quantization_level (Optional[QuantizationLevel]): quantization level to use for the model, defaults to FP32 (no quantization)
            outfile (Optional[str]): path to the output file, defaults to "outfile.bin"
        Returns:
            CompiledModel object containing the prepared model
        """

        model_config_path = pathlib.Path(framework_model_path) / "config.json"
        config_path = getConfigFromSDK(model_config_path)
        model_plus = load_some_model(framework_model_path)
        params = Params.load(model_plus, config_path)
        params.ftype = cls.QUANTIZATION_LEVEL_TO_GGML_FILE_TYPE[quantization_level]
        params.f_rope_factor_short = None
        params.f_rope_factor_long = None
        params.rope_attn_factor = None
        model = model_plus.model
        model = convert_model_names(model, params, config_path)
        model = convert_to_output_type(model, GGMLFileType.AllF32)
        outfile = pathlib.Path(outfile)
        if outfile.is_dir():
            outfile = outfile / "outfile.bin"

        OutputFile.write_all(outfile, params, model, quantization_level)

        return CompiledModel(CacheModule.load(path=outfile), BackendType.CPU)
