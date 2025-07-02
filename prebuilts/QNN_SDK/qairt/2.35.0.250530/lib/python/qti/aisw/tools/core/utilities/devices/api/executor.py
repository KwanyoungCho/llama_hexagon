# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from os import PathLike
from typing import Any, List, Optional,  Union
from abc import abstractmethod, ABC
from qti.aisw.tools.core.utilities.devices.utils.device_code import *
from qti.aisw.tools.core.utilities.devices.protocol_helpers.protocol_helper import ProtocolHelper


class Executor(ABC):
    """
    Base class that enables interaction between a Device and Protocol Helper.

    The role of this object is to abstract the details involved in calling
    device protocols, especially in cases where connections must be established
    before each call or different protocols are needed for IO/execution.
    """

    def __init__(self, protocol_helper: Optional[ProtocolHelper] = None):
        """
        Instantiates an executor

        Args:
            protocol_helper: A protocol helper class.
        """
        self.protocol_helper = protocol_helper

    @property
    def protocol_helper(self) -> ProtocolHelper:
        """
        Returns the protocol helper associated with this executor
        """
        return self._protocol_helper

    @protocol_helper.setter
    def protocol_helper(self, protocol_helper: ProtocolHelper):
        """
        Set the protocol helper for the executor

        Args:
            protocol_helper: The new protocol helper
        """
        self._protocol_helper = protocol_helper

    @abstractmethod
    def push(self, src: str | PathLike, dst: str | PathLike) -> DeviceReturn:
        """
        Abstract method to push a file from local storage to remote device

        Args:
            src(str | PathLike): Source file to be pushed
            dst(str | PathLike): Destination directory on the remote device

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """

    @abstractmethod
    def pull(self, src: str | PathLike, dst: str | PathLike) -> DeviceReturn:
        """
        Abstract method to push a file from local storage to remote device

        Args:
            src(str | PathLike): Source file to be pushed
            dst(str | PathLike): Destination directory on the remote device

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """

    @abstractmethod
    def execute(self, command: str, args: Optional[List[str]] = None, **kwargs) -> DeviceReturn:
        """
        Abstract method to execute a command on the remote device

        Args:
            command(str): Command to be executed
            args(Optional[List[str]]): List of arguments to be passed to the command
                Defaults to None

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """

    @abstractmethod
    def get_available_devices(self, *args, **kwargs) -> Any:
        """
        Abstract method to get available devices connected to the system
        """
