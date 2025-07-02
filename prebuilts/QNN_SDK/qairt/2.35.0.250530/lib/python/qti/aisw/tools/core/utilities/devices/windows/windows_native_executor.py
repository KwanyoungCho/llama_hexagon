# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from os import getcwd
from typing import List, Optional

from qti.aisw.tools.core.utilities.devices.api.executor import *
from qti.aisw.tools.core.utilities.devices.api.native_executor import NativeExecutor
from qti.aisw.tools.core.utilities.devices.utils import subprocess_helper

# Import QAIRT logging utilities
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class WindowsNativeExecutor(NativeExecutor):
    """Class for interacting with a local Windows Arm Linux (WOS) Device"""
    log_area = LogAreas.register_log_area("WindowsNativeExecutor")
    _logger = QAIRTLogger.register_area_logger(area=log_area, level="INFO")

    def __init__(self):
        super().__init__(logger=self._logger)

    def execute(self, command: str, args: Optional[List[str]] = None, shell: bool = True,
                 cwd: str = getcwd()) -> DeviceReturn:
        """Execute the command on a Windows X86_64 local host
        Args:
            command (str): The command to run.
            args (Optional[List[str]], optional): List of arguments for the command.
                Defaults to None.
            shell (bool, optional): Whether to use shell. Defaults to True.
            cwd (str): Current working directory. Defaults to os.getcwd()

        Returns:
            DeviceReturn Union[DeviceCompletedProcess, DeviceFailedProcess]
        """
        return subprocess_helper.execute(command, args, powershell=True, cwd=cwd, shell=shell)
