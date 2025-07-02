# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import pathlib
from os import PathLike
from typing import Optional

from pydantic import field_validator

from qairt.api.configs.common import AISWBaseModel


class PositionalEncodings(AISWBaseModel):
    """Base class for configuration information for models that use positional encodings."""


class RotationalPositionalEncodings(PositionalEncodings):
    """Rotational positional encodings"""

    theta: float
    dim: int


class AbsolutePositionalEncodings(PositionalEncodings):
    """Absolute positional encodings"""

    # TODO: Need to understand what (if any) attributes belong to this encoding option


class AlibiPositionalEncodings(PositionalEncodings):
    """Alibi positional encodings"""

    # TODO: Need to understand what (if any) attributes belong to this encoding option


class GenAIConfig(AISWBaseModel):
    """
    GenAIConfig holds common configuration information for the Generative AI Model, needed for Genie
    execution.  Common attributes (present in all subclasses):
    """

    tokenizer_path: str | PathLike
    """
    The path to the tokenizer.  Must point to an existing file.
    """
    context_length: int
    """context length"""

    n_vocab: int
    """The number of tokens in the vocabulary, which is also the first dimension of the embeddings matrix"""

    n_heads: Optional[int] = None
    """The number of attention heads used in the multi-head attention layers of the model"""

    n_layer: Optional[int] = None
    """The number of blocks in the model"""

    n_embd: Optional[int] = None
    """The hidden size of the model"""

    bos_token: int
    """The id of the _beginning-of-stream_ token."""

    eos_token: int | list[int]
    """The id of the _end-of-stream_ token."""

    eot_token: Optional[int] = None
    """The id of the _end-of-turn_ token."""

    positional_encodings: Optional[PositionalEncodings] = None
    """An object describing the positional encodings"""

    """
    These are properties of the model that genie may use, but I'm not sure how to get them:
    kv-dim
    n_logits: int
    """

    kv_dim: Optional[int] = None

    @field_validator("tokenizer_path")
    def validate_tokenizer_path(cls, v):
        path = pathlib.Path(v)
        if not path.resolve().is_file():
            raise FileNotFoundError(f"The tokenizer_path '{v}' does not point to an existing file.")
        return v
