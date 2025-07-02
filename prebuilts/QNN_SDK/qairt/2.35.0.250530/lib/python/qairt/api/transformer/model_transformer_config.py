# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
ModelTransformerConfig class.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union


class QuantizationStage(Enum):
    """
    Represents the quantization stage for running the transformations.

    Attributes:
        PRE_QUANT (str): Indicates the stage before quantization.
        POST_QUANT (str): Indicates the stage after quantization.
    """

    PRE_QUANT = "PRE_QUANT"
    POST_QUANT = "POST_QUANT"


@dataclass
class AttentionMask1Dto2DConfig:
    """
    Configuration for converting 1D attention mask to 2D.
    """

    log_level: str = "info"
    """
    Level/Severity of events to log.
    """


@dataclass
class ReplaceLinearWithConvConfig:
    """
    Configuration for replacing linear layers with convolutional layers.
    """

    log_level: str = "info"
    """
    Level/Severity of events to log.
    """


@dataclass
class ReplacePosIdWithRoPEConfig:
    """
    Configuration for replacing positional IDs with RoPE.
    """

    log_level: str = "info"
    """
    Level/Severity of events to log.
    """


@dataclass
class UnpackQKVConfig:
    """
    Configuration for unpacking QKV (Query, Key, Value) in attention layers.
    """

    log_level: str = "info"
    """
    Level/Severity of events to log.
    """


@dataclass
class OptimizeKVCacheConfig:
    """
    Configuration for optimizing key-value cache in attention layers.
    """

    transpose_keycache: bool = True
    """
    Transpose the new key and output the transposed key cache for all attention layers.
    """
    output_new_key_value_only: bool = True
    """
    Return only the new key value to reduce memory traffic.
    """
    log_level: str = "info"
    """
    Level/Severity of events to log.
    """


@dataclass
class MHA2SHAConfig:
    """
    Configuration for converting Multi-Head Attention (MHA) to Single-Head Attention (SHA) in models.
    """

    is_llm_model: bool = True
    """
    Whether the model is an LLM (as opposed to an LVM model).
    """
    is_gqa_model: bool = False
    """
    Indicates if the model has GQA architecture.
    """
    is_lora_model: bool = False
    """
    Indicates if the model has LoRA adapters.
    """
    lora_adapters: Optional[Union[str, list[dict]]] = None
    """
    List of LoRA adapters.
    """
    lora_tensor_names: Optional[Union[str, list[str]]] = None
    """
    List of LoRA tensor names
    """
    lora_alpha_from_input: bool = False
    """
    LoRA alpha from model input. Use this option only if is_lora_model is also set.
    """
    is_prepared_model: bool = False
    """
    Enable changes for prepared model.
    """
    optimize_o_proj: bool = True
    """
    Optimize SHA head concat -> o_proj pattern.
    """
    handle_alibi: bool = False
    """
    Model has ALiBi position embeddings.
    """
    handle_past_key_value: bool = False
    """
    Enable handling of past key/value in LLMs.
    """
    handle_rope_ops: bool = False
    """
    Enable handling of RoPE ops.
    """
    strict_rope_pattern: bool = True
    """
    Strictly enforce Golden RoPE pattern.
    """
    build_ar: Optional[int] = None
    """
    Builds a SHA model with a different AR provided. AR value must be >=1.
    """
    disable_auto_attn_finder: bool = False
    """
    Disable auto attention finder in step 5.
    """
    skip_verification: bool = False
    """
    Does not run extra steps for verification of the model and encoding mappings.
    """
    is_linear_to_conv_converted: bool = True
    """
    Indicates if Linear2Conv conversion has been done.
    """
    is_nchw_aligned: bool = True
    """
    Indicates if the model is NCHW aligned.
    """
    log_level: str = "info"
    """
    Level/Severity of events to log.
    """


@dataclass
class SplitModelConfig:
    """
    Configuration for splitting a model into multiple subgraphs.
    """

    num_splits: int = 1
    """
    Number of splits we want to divide the model up into.
    """
    split_embedding: bool = False
    """
    Split embedding into its own subgraph/model.
    """
    split_lm_head: bool = False
    """
    Split LM head into its own subgraph/model.
    """
    skip_verification: bool = False
    """
    Skip ONNXRT verification of comparing full model outputs to splits outputs
    """
    log_level: str = "info"
    """
    Level/Severity of events to log.
    """


@dataclass
class ModelTransformerConfig:
    """
    Parent configuration for all transformation settings.

    Attributes:
        attention_mask_1d_to_2d (AttentionMask1Dto2DConfig): Configuration for converting 1D attention mask to 2D.
        replace_linear_with_conv (ReplaceLinearWithConvConfig): Configuration for replacing linear layers with convolutional layers.
        replace_pos_id_with_rope (ReplacePosIdWithRoPEConfig): Configuration for replacing positional IDs with RoPE.
        unpack_qkv (UnpackQKVConfig): Configuration for unpacking QKV (Query, Key, Value) in attention layers.
        optimize_kv_cache (OptimizeKVCacheConfig): Configuration for optimizing key-value cache in attention layers.
        mha_to_sha (MHA2SHAConfig): Configuration for converting Multi-Head Attention (MHA) to Single-Head Attention (SHA) in models.
        split_model (SplitModel): Configuration for splitting a model into multiple subgraphs.
    """

    attention_mask_1d_to_2d: Optional[AttentionMask1Dto2DConfig] = field(
        default_factory=AttentionMask1Dto2DConfig
    )
    replace_linear_with_conv: Optional[ReplaceLinearWithConvConfig] = field(
        default_factory=ReplaceLinearWithConvConfig
    )
    replace_pos_id_with_rope: Optional[ReplacePosIdWithRoPEConfig] = field(
        default_factory=ReplacePosIdWithRoPEConfig
    )
    unpack_qkv: Optional[UnpackQKVConfig] = field(default_factory=UnpackQKVConfig)
    optimize_kv_cache: Optional[OptimizeKVCacheConfig] = field(default_factory=OptimizeKVCacheConfig)
    mha_to_sha: Optional[MHA2SHAConfig] = field(default_factory=MHA2SHAConfig)
    split_model: Optional[SplitModelConfig] = field(default_factory=SplitModelConfig)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ModelTransformerConfig":
        # Mapping of transformation names to their corresponding configuration classes
        transformation_config_classes = {
            "attention_mask_1d_to_2d": AttentionMask1Dto2DConfig,
            "replace_linear_with_conv": ReplaceLinearWithConvConfig,
            "replace_pos_id_with_rope": ReplacePosIdWithRoPEConfig,
            "unpack_qkv": UnpackQKVConfig,
            "optimize_kv_cache": OptimizeKVCacheConfig,
            "mha_to_sha": MHA2SHAConfig,
            "split_model": SplitModelConfig,
        }

        # Initialize TransformConfig dynamically
        transform_config_kwargs = {}
        for key, config_class in transformation_config_classes.items():
            if key in config_dict:
                transform_config_kwargs[key] = config_class(**config_dict[key])

        return cls(**transform_config_kwargs)
