# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field, model_serializer

from qairt.api.configs.common import AISWBaseModel


class Report(AISWBaseModel):
    """
    Base class for reports.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """ Unique identifier for the report. """

    timestamp: datetime = Field(default_factory=datetime.now)
    """ Timestamp when the report was created. """

    data: Optional[Any] = None
    """ Data associated with the report. """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    @model_serializer
    def serialize(self) -> Dict[str, Any]:
        """
        Custom serialization logic for the profiling report, including sub-reports.
        """
        return {"id": self.id, "timestamp": self.timestamp.isoformat(), "data": self.data}

    def dump(self, path: str | os.PathLike) -> str:
        """Dumps json to a file of choice"""
        path = Path(path)
        if os.path.isdir(path):
            path /= f"report_{self.id}.json"

        with open(path, "w+") as f:
            serialized_output = self.serialize()
            json.dump(serialized_output.get("data"), f, indent=4)

        return str(path)

    @model_serializer(when_used="json")
    def dump_json(self, **kwargs) -> str:
        """
        Generate a JSON string report for the profiling report and its sub-reports.

        Args:
            **kwargs: Additional keyword arguments passed to json.dumps.

        Returns:
            str: JSON-formatted string.
        """
        serialized_output = self.serialize()
        return json.dumps(serialized_output.get("data"), indent=4, **kwargs)
