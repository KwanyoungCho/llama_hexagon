# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import List, Optional

from pydantic import Field

from qairt.api.configs.common import AISWBaseModel


class EncodingsInfo(AISWBaseModel):
    """
    Represents the encodings information of a tensor.
    """

    bitwidth: int = Field(description="The number of bits used for quantization")
    max: float = Field(description="The maximum value of the quantized range")
    min: float = Field(description="The minimum value of the quantized range")
    scale: float = Field(
        description="The scale factor used in quantization. This factor is used to map the original values to the quantized range."
    )
    offset: int = Field(
        description="The offset used in quantization. This is typically used to shift the range of values."
    )
    is_symmetric: bool = Field(description="Indicates whether the quantization is symmetric.")
    is_fixed_point: bool = Field(description="Indicates whether the quantization uses fixed-point arithmetic")


class TensorInfo(AISWBaseModel):
    """
    Represents an input or output tensor in a graph.
    """

    name: str = Field(description="The unique name of the tensor, used to identify it in the graph.")
    dimensions: List[int] = Field(description="List representation of the tensor dimensions")
    data_type: str = Field(description="The string representation of the tensor dimensions")
    is_quantized: bool = Field(default=False, description="Detemines if a model is quantized.")
    encodings: Optional[EncodingsInfo] = Field(
        default=None, description="Encapsulates the quantization encoding details for a tensor"
    )


class GraphInfo(AISWBaseModel):
    """
    Represents one graph in the cache, with name
    and lists of input and output tensors.
    """

    name: str = Field(description="The name or identifier of the graph.")
    inputs: List[TensorInfo] = Field(description="A list of input tensors for the graph.")
    outputs: List[TensorInfo] = Field(description="A list of output tensors for the graph.")
