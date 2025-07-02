# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from typing import List, Literal, Optional

from pydantic import Field, FilePath, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from qti.aisw.tools.core.modules.api.definitions.common import AISWBaseModel
from qti.aisw.tools.core.modules.converter import (
    ConverterInputConfig,
    QuantizerInputConfig,
)
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    RemoteDeviceInfo,
)


class InputSample(AISWBaseModel):
    name: str
    dimensions: Optional[List[int]]
    raw_file: FilePath
    data_type: Optional[str] = None


class LayerOptions(AISWBaseModel):
    add_layer_outputs: Optional[List[str]] = []
    add_layer_types: Optional[List[str]] = []
    skip_layer_types: Optional[List[str]] = []
    skip_layer_outputs: Optional[List[str]] = []
    start_layer: Optional[str] = None
    end_layer: Optional[str] = None


class ConverterInputArguments(ConverterInputConfig):
    input_network: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    dry_run: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    output_path: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    float_bitwidth: Optional[Literal[32, 16]] = Field(
        default=None,
        description="Convert the graph to specified float bitwidth.",
    )
    onnx_batch: SkipJsonSchema[int] = Field(default=None, init=False, exclude=True)

    @field_validator("input_network")
    @classmethod
    def validate_framework(cls, v):
        pass

    @model_validator(mode="after")
    def validate_input_arguments(self):
        return self


class QuantizerInputArguments(QuantizerInputConfig):
    input_dlc: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    output_dlc: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    backend_info: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)


class RemoteHostDetails(RemoteDeviceInfo):
    platform_type: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
