# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from qti.aisw.tools.core.modules.api import AISWBaseModel


class ProfileRecordEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class EventType(Enum):
    GENIE_DIALOG_CREATE = "GenieDialog_create"
    GENIE_DIALOG_QUERY = "GenieDialog_query"


class Event(AISWBaseModel):
    type: EventType
    duration: int
    start: int
    stop: int


class Record(AISWBaseModel):
    value: Union[int, float]
    unit: str


class DialogCreateEvent(Event):
    init_time: Optional[Record] = Field(None, alias="init-time")


class DialogQueryEvent(Event):
    num_prompt_tokens: Optional[Record] = Field(None, alias="num-prompt-tokens")
    prompt_processing_rate: Optional[Record] = Field(None, alias="prompt-processing-rate")
    time_to_first_token: Optional[Record] = Field(None, alias="time-to-first-token")
    num_generated_tokens: Optional[Record] = Field(None, alias="num-generated-tokens")
    token_generation_rate: Optional[Record] = Field(None, alias="token-generation-rate")
    token_generation_time: Optional[Record] = Field(None, alias="token-generation-time")
    lora_adapter_switching_time: Optional[Record] = Field(None, alias="lora-adapter-switching-time")


class ComponentType(Enum):
    DIALOG = "dialog"


class Component(AISWBaseModel):
    name: str
    type: ComponentType
    events: List[Union[DialogCreateEvent, DialogQueryEvent]]


class ArtifactType(Enum):
    GENIE_PROFILE = "GENIE_PROFILE"


class Version(AISWBaseModel):
    major: int
    minor: int
    patch: int


class Header(AISWBaseModel):
    header_version: Version
    version: Version
    artifact_type: ArtifactType


class GenieProfileRecord(AISWBaseModel):
    header: Header
    metadata: Optional[Dict[str, Any]] = None
    components: List[Component]

    def export(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)

    def __str__(self):
        return json.dumps(self.export(), indent=2, cls=ProfileRecordEncoder)
