# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import inspect
from typing import Optional, Union

import onnx

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.defs.mha2sha_transformed_model import (
    Mha2ShaTransformedModel,
)

from .attn_mask_1d_to_2d import AdaptAttentionMask
from .kvcache_optimizer import KVCacheOptimizer
from .linear2conv import LinearToConv
from .mha2sha import MHA2SHAConverter
from .onnx_splitter import split_onnx
from .RoPE import RoPE
from .unpackQKV import UnpackQKV as _UnpackQKV
from .utils.base_arch_configs import mha2sha_base_arch_flags, splitter_base_arch_flags

from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger

mt_log_area = LogAreas.register_log_area("Model_Transformations")
mt_logger = QAIRTLogger.register_area_logger(
            mt_log_area, level="INFO", formatter_val="extended", handler_list=["dev_console"]
        )

def AttentionMask1Dto2D(
    model: onnx.ModelProto,
    encodings: Optional[dict] = None,
    *,
    log_level: str = "info",
) -> None:
    """
    Modifies the model and encodings in-place to adapt the attention mask from 1D to 2D.

    This function applies the `AdaptAttentionMask` transformation to the given model and encodings.
    The transformation is performed in-place, meaning the original model and encodings are modified directly.

    Args:
        model (onnx.ModelProto): The ONNX model to be modified.
        encodings (Optional[dict]): Aimet encodings for the model.
            The encodings will be modified when a corresponding modification is done on the graph.
    Keyword-only Args:
        log_level (str): The logging level to be used during the transformation. Default is "info".

    Returns:
        None
    """
    AdaptAttentionMask(model, encodings, log_level).apply()


def OptimizeKVCache(
    model: onnx.ModelProto,
    *,
    transpose_keycache: bool = True,
    output_new_key_value_only: bool = True,
    log_level: str = "info",
) -> None:
    """
    Optimizes the key-value cache in the given ONNX model.

    This function modifies the model in-place to optimize the key-value cache.
    It can transpose the key cache and configure the model to output only the new key-value pairs.

    Args:
        model (onnx.ModelProto): The ONNX model to be optimized.
    Keyword-only Args:
        transpose_keycache (bool): If True, the key cache will be transposed. Default is True.
        output_new_key_value_only (bool): If True, the model will be configured to output only the
            new key-value pairs. Default is True.
        log_level (str): The logging level to be used during the optimization. Default is "info".

    Returns:
        None
    """
    KVCacheOptimizer(model, transpose_keycache, output_new_key_value_only, log_level).run()


def MHA2SHA(
    model: onnx.ModelProto,
    base_arch: str = "",
    encodings: Optional[dict] = None,
    *,
    is_llm_model: bool = True,
    is_gqa_model: bool = False,
    is_lora_model: bool = False,
    lora_adapters: Optional[Union[str, list[dict]]] = None,
    lora_tensor_names: Optional[Union[str, list[str]]] = None,
    lora_alpha_from_input: bool = False,
    is_prepared_model: bool = False,
    optimize_o_proj: bool = True,
    handle_alibi: bool = False,
    handle_past_key_value: bool = False,
    handle_rope_ops: bool = False,
    strict_rope_pattern: bool = True,
    build_ar: int | None = None,
    disable_auto_attn_finder: bool = False,
    skip_verification: bool = False,
    is_linear_to_conv_converted: bool = True,  # Linear2Conv is mostly assumed to be run before mha2sha
    is_nchw_aligned: bool = True,
    log_level: str = "info",
) -> Mha2ShaTransformedModel:
    """
    Converts Multi-Head Attention (MHA) to Single-Head Attention (SHA) in the given ONNX model.

    This function modifies the model in-place to convert MHA to SHA.

    Args:
        model (onnx.ModelProto): The ONNX model to be modified.
        base_arch (str): The base architecture of the model. Default is an empty string.
        encodings (Optional[dict]): Aimet encodings for the model.
            The encodings will be modified when a corresponding modification is done on the graph.
    Keyword-only Arguments:
        is_llm_model (bool): Indicates if the model is a large language model. Default is True.
        is_gqa_model (bool): Indicates if the model uses GQA (Generalized Query Attention).
            Default is False.
        is_lora_model (bool): Indicates if the model uses LoRA (Low-Rank Adaptation).
            Default is False.
        lora_adapters (str, list[dict]): List of LoRA adapters to be used or path to a yaml file with list of lora adapters
            * If `lora_adapters` is a list of dictionaries, each dictionary in the list should have the following structure:
                {
                    "name": <adapter/usecase name>,
                    "lora_weights": <path to safetensor file for the adapter>,
                    "quant_overrides": <path to AIMET encodings file for the adapter>,
                }
            * If `lora_adapters` is a path to a .yaml file, the schema for the yaml file should be as follows:
                # Start config
                use_case:  # List of use-cases
                    - name:                 <usecase_1/adapter_1 name>
                      lora_weights:         <path to safetensor file for adapter_1>
                      quant_overrides:      <path to AIMET encodings file for adapter_1>
                    - name:                 <usecase_2/adapter_2 name>
                      lora_weights:         <path to safetensor file for adapter_2>
                      quant_overrides:      <path to AIMET encodings file for adapter_2>
                    ...
            Default is None
        lora_tensor_names (str, list[str]): List of LoRA tensor names.
            Default is None
        lora_alpha_from_input (bool): If True, LoRA alpha values are taken from input.
            Default is False.
        is_prepared_model (bool): Indicates if the model is already prepared. Default is False.
        optimize_o_proj (bool): If True, optimizes the output projection. Default is True.
        handle_alibi (bool): If True, handles alibi bias. Default is False.
        handle_past_key_value (bool): If True, handles past key-value pairs. Default is False.
        handle_rope_ops (bool): If True, handles RoPE (Rotary Position Embedding) operations.
            Default is False.
        strict_rope_pattern (bool): If True, enforces strict RoPE patterns. Default is True.
        build_ar (int): Build AR (Auto-Regressive) configuration. Default is 0.
        disable_auto_attn_finder (bool): If True, disables automatic attention finder.
            Default is False.
        skip_verification (bool): If True, skips verification steps. Default is False.
        is_linear_to_conv_converted (bool): Indicates if Linear2Conv conversion has been done. Default is True.
        is_nchw_aligned (bool): Indicates if the model is NCHW aligned. Default is True.
        log_level (str): The logging level to be used during the conversion. Default is "info".

    Returns:
        Object of type(dataclass) Mha2ShaTransformedModel
    """

    local_vars = locals()

    signature = inspect.signature(MHA2SHA)

    kwargs = {}

    if base_arch_flags := mha2sha_base_arch_flags.get(base_arch):
        # If the following conditions hold true:
        #    1. An argument has been explicitly passed
        #    2. And it doesn't match the value in the base_arch flags
        # Then, print a warning
        for name, param in signature.parameters.items():
            local_value = local_vars[name]
            default_value = param.default
            if flag_value := base_arch_flags.get(name):
                if local_value != default_value:  # Overridden
                    if local_value != flag_value:  # And set to an unexpected value
                        mt_logger.warning(
                            f"The default value of '{name}' is '{flag_value}' for base_arch={base_arch}, but {name}={local_value}"
                        )
                        kwargs[name] = local_value
                else:
                    kwargs[name] = flag_value
            else:
                kwargs[name] = local_value

    else:
        kwargs = local_vars

    return MHA2SHAConverter(**kwargs).convert()


def ReplaceLinearWithConv(model: onnx.ModelProto, *, log_level: str = "info") -> None:
    """
    Replaces linear layers with convolutional layers in the given ONNX model.

    This function modifies the model in-place to replace linear layers with convolutional layers.

    Args:
        model (onnx.ModelProto): The ONNX model to be modified.
        log_level (str): The logging level to be used during the replacement. Default is "info".

    Returns:
        None
    """
    LinearToConv(model, log_level).replace()


def ReplacePosIdWithRoPE(
    model: onnx.ModelProto,
    base_arch: str = "",
    encodings: Optional[dict] = None,
    *,
    log_level: str = "info",
) -> None:
    """
    Replaces positional IDs with Rotary Position Embeddings (RoPE) in the given ONNX model.

    This function modifies the model in-place to replace positional IDs with RoPE.

    Args:
        model (onnx.ModelProto): The ONNX model to be modified.
        base_arch (str): The base architecture of the model. Default is an empty string.
        encodings (Optional[dict]): Aimet encodings for the model.
            The encodings will be modified when a corresponding modification is done on the graph.
    Keyword-only args:
        log_level (str): The logging level to be used during the replacement. Default is "info".

    Returns:
        None
    """
    RoPE(model, base_arch, encodings, log_level).perform_RoPE()


def UnpackQKV(model: onnx.ModelProto, log_level: str = "info") -> None:
    """
    Unpacks the QKV (Query, Key, Value) tensors in the given ONNX model.

    This function modifies the model in-place to unpack the QKV tensors.

    Args:
        model (onnx.ModelProto): The ONNX model to be modified.
        log_level (str): The logging level to be used during the unpacking. Default is "info".

    Returns:
        None
    """
    _UnpackQKV(model, log_level).unpack()


def SplitModel(
    model: Union[str, onnx.ModelProto],
    base_arch: str = "",
    encodings: Optional[dict] = None,
    *,
    split_embedding: bool = False,
    split_lm_head: bool = False,
    num_splits: int = -1,
    skip_verification: bool = False,
    log_level: str = "info"
) -> Union[list[str], list[onnx.ModelProto]]:
    """
    Splits the given ONNX model into multiple sub-models.

    This function splits the model in-place into the specified number of sub-models.
    It supports splitting embeddings and language model heads.

    Args:
        model (onnx.ModelProto): The ONNX model to be split.
        base_arch (str): The base architecture of the model. Default is an empty string.
        encodings (Optional[dict]): Aimet encodings for the model.
            The encodings will be modified when a corresponding modification is done on the graph.
    Keyword-only Args:
        num_splits (int): The number of splits to be made. Default is 1.
        split_embedding (bool): If True, splits the embeddings. Default is False.
        split_lm_head (bool): If True, splits the language model head. Default is False.
        skip_verification (bool): If True, skip ONNXRT verification of comparing 
            full model outputs to splits outputs. Default is False.
        log_level (str): The logging level to be used during the splitting. Default is "info".

    Returns:
        list[onnx.ModelProto]: A list of the resulting sub-models.
    """
    local_vars = locals()

    signature = inspect.signature(SplitModel)

    kwargs = {}

    if base_arch_flags := splitter_base_arch_flags.get(base_arch):
        # If the following conditions hold true:
        #    1. An argument has been explicitly passed
        #    2. And it doesn't match the value in the base_arch flags
        # Then, print a warning
        for name, param in signature.parameters.items():
            local_value = local_vars[name]
            default_value = param.default
            if flag_value := base_arch_flags.get(name):
                if local_value != default_value and local_value != flag_value:  # Overridden and set to an unexpected value
                    mt_logger.warning(
                        f"The default value of '{name}' is '{flag_value}' for base_arch={base_arch}, but {name}={local_value}"
                    )
                    kwargs[name] = local_value
                else:
                    kwargs[name] = flag_value
            else:
                kwargs[name] = local_value

        num_splits = kwargs['num_splits']
        split_embedding = kwargs['split_embedding']
        split_lm_head = kwargs['split_lm_head']
    else:
        if num_splits == -1: # Unchanged
            raise AttributeError(f"Either `base_arch` should be set to one of the following values: {list(splitter_base_arch_flags.keys())}, or `num_splits` should be passed")

    min_splits = 1 + int(split_embedding) + int(split_lm_head)

    if num_splits < min_splits:
        raise AttributeError(f"Invalid value for `num_splits`: {num_splits}. The minimum number of splits when `split_embedding={split_embedding}` and `split_lm_head={split_lm_head}` is {min_splits}")

    if num_splits == 1:
        mt_logger.warning("`num_splits=1` results in a no-op. Returning the original model")
        return [model]

    return split_onnx(model, encodings, num_splits=num_splits, split_embedding=split_embedding, split_lm_head=split_lm_head, skip_verification=skip_verification, log_level=log_level) 
