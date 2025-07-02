# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
import os
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from qti.aisw.tools.core.modules.api import AISWBaseModel


class GenieConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, os.PathLike):
            return str(obj)
        return super().default(obj)


class VersionedModel(AISWBaseModel):
    version: int = 1
    model_config = ConfigDict(populate_by_name=True)


class QnnHtpBackend(VersionedModel):
    use_mmap: bool = Field(False, alias="use-mmap")
    spill_fill_bufsize: int = Field(0, alias="spill-fill-bufsize")
    mmap_budget: int = Field(40, alias="mmap-budget")
    poll: bool
    pos_id_dim: Optional[int] = Field(None, alias="pos-id-dim")
    cpu_mask: str = Field(default="0x00", alias="cpu-mask")
    kv_dim: Optional[int] = Field(None, alias="kv-dim")
    kv_update_method: Optional[str] = Field(None, alias="kv-update-method")
    rope_theta: Optional[int] = Field(None, alias="rope-theta")
    allow_async_init: Optional[bool] = Field(None, alias="allow-async-init")
    enable_graph_switching: Optional[bool] = Field(None, alias="enable-graph-switching")


class QnnGenAiTransformerBackend(VersionedModel):
    use_mmap: Optional[bool] = Field(None, alias="use-mmap")
    n_logits: Optional[int] = Field(None, alias="n-logits")
    n_layer: Optional[int] = Field(None, alias="n-layer")
    n_embd: Optional[int] = Field(None, alias="n-embd")
    n_heads: Optional[int] = Field(None, alias="n-heads")


class EngineBackendType(str, Enum):
    QNN_GEN_AI_TRANSFORMER = "QnnGenAiTransformer"
    QNN_HTP = "QnnHtp"


class EngineBackend(VersionedModel):
    type: EngineBackendType = EngineBackendType.QNN_GEN_AI_TRANSFORMER
    QnnGenAiTransformer: Optional[QnnGenAiTransformerBackend] = None
    QnnHtp: Optional[QnnHtpBackend] = None
    extensions: Optional[str | os.PathLike] = None

    @model_validator(mode="after")
    def check_type(self) -> Self:
        if self.type == EngineBackendType.QNN_GEN_AI_TRANSFORMER:
            if self.QnnGenAiTransformer is None:
                raise ValueError(f"QnnGenAiTransformer must be provided when type is: {self.type.value}")
        elif self.QnnGenAiTransformer is not None:
            raise ValueError(
                "QnnGenAiTransformer should only be provided when type is: "
                f"{EngineBackendType.QNN_GEN_AI_TRANSFORMER.value}"
            )

        if self.type == EngineBackendType.QNN_HTP:
            if self.QnnHtp is None:
                raise ValueError(f"QnnHtp must be provided when type is: {self.type.value}")
        elif self.QnnHtp is not None:
            raise ValueError(
                f"QnnHtp should only be provided when type is: {EngineBackendType.QNN_HTP.value}"
            )

        return self


class LoraConfigAdapter(VersionedModel):
    name: str
    bin_sections: Optional[List[str | os.PathLike]] = Field(alias="bin-sections")
    path: Optional[str | os.PathLike] = None


class LoraConfig(VersionedModel):
    alpha_tensor_name: Optional[str] = Field(None, alias="alpha-tensor-name")
    lora_version: Optional[int] = Field(None, alias="lora-version")
    adapters: List[LoraConfigAdapter]


class ModelBinary(VersionedModel):
    ctx_bins: List[str | os.PathLike] = Field(alias="ctx-bins")
    lora: Optional[LoraConfig] = None


class ModelLibrary(VersionedModel):
    model_bin: str | os.PathLike = Field(alias="model-bin")
    lora: Optional[LoraConfig] = None


class RopeType(str, Enum):
    LLAMA3 = "llama3"
    DEFAULT = "default"
    LONG_ROPE = "longrope"


class RopeScaling(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    rope_type: Optional[RopeType] = Field(None, alias="rope-type")
    factor: Optional[float] = None
    low_freq_factor: Optional[float] = Field(None, alias="low-freq-factor")
    high_freq_factor: Optional[float] = Field(None, alias="high-freq-factor")
    original_max_position_embeddings: Optional[int] = Field(None, alias="original-max-position-embeddings")
    short_factor: Optional[List[float]] = Field(None, alias="short-factor")
    long_factor: Optional[List[float]] = Field(None, alias="long-factor")


class PositionalEncodingType(str, Enum):
    ROPE = "rope"
    ABSOLUTE = "absolute"
    ALIBI = "alibi"


class PositionalEncoding(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Optional[PositionalEncodingType] = None
    rope_dim: Optional[int] = Field(None, alias="rope-dim")
    rope_theta: Optional[int] = Field(None, alias="rope-theta")
    rope_scaling: Optional[RopeScaling] = Field(None, alias="rope-scaling")


class EngineModelType(str, Enum):
    LIBRARY = "library"
    BINARY = "binary"


class EngineModel(VersionedModel):
    type: EngineModelType = EngineModelType.LIBRARY
    library: Optional[ModelLibrary] = None
    binary: Optional[ModelBinary] = None
    positional_encoding: Optional[PositionalEncoding] = None


class DialogEngine(VersionedModel):
    n_threads: int = Field(6, alias="n-threads")
    backend: EngineBackend = Field(default_factory=EngineBackend)
    model: EngineModel = Field(default_factory=EngineModel)


class Context(VersionedModel):
    bos_token: int = Field(0, alias="bos-token")
    eos_token: int | List[int] = Field(0, alias="eos-token")
    eot_token: Optional[int] = Field(None, alias="eot-token")
    n_vocab: int = Field(0, alias="n-vocab")
    size: int = 512
    pad_token: Optional[int] = Field(None, alias="pad-token")


class Sampler(VersionedModel):
    seed: Optional[int] = None
    temp: Optional[float] = None
    top_k: Optional[int] = Field(None, alias="top-k")
    top_p: Optional[float] = Field(None, alias="top-p")
    greedy: Optional[bool] = None
    type: Optional[str] = None
    callback_name: Optional[str] = Field(None, alias="callback-name")


class Tokenizer(VersionedModel):
    path: str | os.PathLike = ""


class DialogEmbeddingDataType(str, Enum):
    FLOAT32 = "float32"
    NATIVE = "native"


class DialogEmbedding(VersionedModel):
    size: int
    datatype: Optional[DialogEmbeddingDataType] = None


class SsdQ1(VersionedModel):
    ssd_version: int = Field(1, alias="ssd-version")
    forecast_token_count: int = Field(alias="forecast-token-count")
    forecast_prefix: int = Field(alias="forecast-prefix")
    forecast_prefix_name: str | os.PathLike = Field(alias="forecast-prefix-name")
    branches: List[int]
    n_streams: Optional[int] = Field(None, alias="n-streams")
    p_threshold: Optional[float] = Field(None, alias="p-threshold")


class DialogType(str, Enum):
    BASIC = "basic"
    SSD_Q1 = "ssd-q1"


class Dialog(VersionedModel):
    type: DialogType = DialogType.BASIC
    context: Context
    tokenizer: Tokenizer
    engine: DialogEngine
    stop_sequence: Optional[List[str]] = Field(None, alias="stop-sequence")
    max_num_tokens: Optional[int] = Field(None, alias="max-num-tokens")
    sampler: Optional[Sampler] = None
    ssd_q1: Optional[SsdQ1] = Field(None, alias="ssd-q1")
    embedding: Optional[DialogEmbedding] = None


class GenieConfig(AISWBaseModel):
    """
    top level config object for genie config
    """

    dialog: Dialog

    def export(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)

    def __str__(self):
        return json.dumps(self.export(), indent=2, cls=GenieConfigEncoder)
