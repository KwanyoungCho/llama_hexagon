# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
import os
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic.dataclasses import dataclass

from qairt.api.bases.report_base import Report
from qairt.api.configs import ProfilingData, ProfilingLevel, ProfilingOption
from qairt.utils.asset_utils import AssetType, check_asset_type
from qti.aisw.core.model_level_api.utils.qnn_profiling import (
    generate_optrace_profiling_output,
    profiling_log_to_dict,
)


class EventType(str, Enum):
    INFERENCE = "inference"
    COMPILE = "compile"
    CUSTOM = "custom"


@dataclass
class FunctionProfilingEvent:
    id: int

    name: str

    type: EventType  # e.g. "timing", "memory", "log"

    data: Optional[Any] = None  # e.g. unparsed data, JSON, plain text, etc

    level: Optional[ProfilingLevel] = None
    """ Profiling level (e.g., 'basic', 'detailed', 'linting')."""

    option: Optional[ProfilingOption] = None
    """ Profiling option: "op trace" """


class ProfilingReport(Report):
    """
    Base class for profiling reports. Supports sub-reports for more granular profiling.
    """

    level: Optional[ProfilingLevel] = None
    """ Profiling level (e.g., 'basic', 'detailed', 'linting')."""

    option: Optional[ProfilingOption] = None
    """ Profiling option: "op trace" """


class OpTraceReport(ProfilingReport):
    """
    Base class for profiling reports. Supports sub-reports for more granular profiling.
    """

    data: List[Dict[str, Any]] | Dict[str, Any]

    level: ProfilingLevel = "detailed"
    """ Profiling level "detailed" """

    option: ProfilingOption = "optrace"
    """ Profiling option: "op trace" """

    summary: Optional[Report] = None
    """ A summary of the op trace report """

    def dump_qhas(self, path: str | os.PathLike) -> str:
        if not self.summary:
            raise RuntimeError("QHAS report was not generated")
        return self.summary.dump(path)


class ReportGenerator:
    """Class responsible for transforming profiling events into profiling reports"""

    def generate_report(self, event: FunctionProfilingEvent) -> ProfilingReport:
        """Generate a report from an event."""
        raise NotImplementedError("Generate method not implemented.")

    def generate_reports(self, events: Iterable[FunctionProfilingEvent]) -> List[ProfilingReport]:
        """Generate a list of reports from a list of events."""
        return [self.generate_report(event) for event in events]


class ProfileLogGenerator(ReportGenerator):
    def generate_report(self, event: FunctionProfilingEvent) -> ProfilingReport:
        if not isinstance(event.data, ProfilingData):
            # unknown profiling event type
            raise TypeError(f"Profiling data: {event.data} is not valid")

        parsed_profiling_data = self._resolve_profiling_data(event.data)
        return ProfilingReport(level=event.level, option=event.option, data=parsed_profiling_data)

    @staticmethod
    def _resolve_profiling_data(profiling_data: ProfilingData) -> Dict:
        # validate log file
        profiling_log_file = profiling_data.profiling_log
        if not check_asset_type(AssetType.PROFILING_LOG, profiling_log_file):
            raise TypeError(f"Profiling log: {profiling_log_file} is not valid")

        profiling_log_data = profiling_log_to_dict(profiling_log_file)

        return profiling_log_data


class OpTraceGenerator(ReportGenerator):
    def generate_report(self, event: FunctionProfilingEvent) -> OpTraceReport:
        """Generate a profiling report from a list of profiling events."""

        if not isinstance(event.data, ProfilingData):
            # unknown profiling event type
            raise TypeError(f"Profiling data: {event.data} is not valid")

        if event.option == ProfilingOption.OPTRACE:
            parsed_profiling_data = self._resolve_profiling_data(event.data)
            op_trace_report = OpTraceReport(data=parsed_profiling_data[0])

            if len(parsed_profiling_data) > 1:
                op_trace_report.summary = ProfilingReport(data=parsed_profiling_data[1])

            return op_trace_report
        else:
            raise TypeError(f"Profiling data: {event.data} is not a valid op trace event")

    @staticmethod
    def _resolve_profiling_data(profiling_data: ProfilingData) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Resolve profiling data into profiling log data and op trace data.

        Args:
            profiling_data (Any): The profiling data to resolve.

        Returns:
            List[Dict]: The op trace data.
            str: QHAS analysis html summary file path.

        Raises:
            TypeError: If the profiling log or op trace output is not valid.
        """

        # validate log file
        profiling_log_file = profiling_data.profiling_log
        if not check_asset_type(AssetType.PROFILING_LOG, profiling_log_file):
            raise TypeError(f"Profiling log: {profiling_log_file} is not valid")

        if not profiling_data.backend_profiling_artifacts:
            raise ValueError("No op trace raw data found.")

        schematic_binary = profiling_data.backend_profiling_artifacts[0]

        if not check_asset_type(AssetType.SCHEMATIC_BIN, schematic_binary):
            raise TypeError(f"Op trace output: {schematic_binary} is not valid")

        with TemporaryDirectory(prefix="optrace") as temp_dir:
            output_dir = Path(temp_dir) / "optrace_output"

            optrace_json, qhas_json = generate_optrace_profiling_output(
                schematic_binary, profiling_log_file, output_dir, qhas_output_type="json"
            )
            with open(optrace_json) as f:
                op_trace_data = json.load(f)

            with open(qhas_json) as f:
                qhas_summary = json.load(f)

        return op_trace_data, qhas_summary
