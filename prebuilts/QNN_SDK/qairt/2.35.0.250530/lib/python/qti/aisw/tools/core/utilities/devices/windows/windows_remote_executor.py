# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from os import PathLike
from typing import Dict, List, Optional, Type, Union

from qti.aisw.tools.core.utilities.devices.api.executor import DevicePlatformError, DeviceReturn, Executor
from qti.aisw.tools.core.utilities.devices.protocol_helpers.ssh import ProtocolHelper, SSHProtocolHelper

# Import QAIRT logging utilities
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class WindowsRemoteExecutor(Executor):
    """Class for interacting with a WOS Remote device via SSH"""
    log_area = LogAreas.register_log_area("WindowsRemoteExecutor")
    _logger = QAIRTLogger.register_area_logger(area=log_area, level="INFO")

    def __init__(self, hostname: str, username: Optional[str] = None,
                 password: Optional[str] = None, *, port: int = 0, timeout: Optional[int] = None):
        super().__init__()
        self._protocol_helper = SSHProtocolHelper
        self._ssh_client = None
        self._sftp_client = None
        self.hostname = hostname.strip()
        self.username = username
        self.password = password
        if ssh_client := self._protocol_helper.setup_ssh_connection(self.hostname,
                                                                   self.username,
                                                                   self.password,
                                                                   port=port,
                                                                   timeout=timeout):
            self._ssh_client = ssh_client
            self._sftp_client = self._protocol_helper.get_ftp_client(ssh_client)
        else:
            raise DevicePlatformError(f"WOS Remote: Could not setup an ssh connection for: "
                                      f"{self.hostname}")

    @property
    def protocol_helper(self) -> Type[ProtocolHelper]:
        """Returns the protocol helper associated with this executor"""
        return self._protocol_helper

    @protocol_helper.setter
    def protocol_helper(self, protocol_helper: Dict[str, ProtocolHelper]):
        self._protocol_helper = protocol_helper

    def push(self, src: Union[str, PathLike], dst: Union[str, PathLike]) -> DeviceReturn:
        """Method to push files to the device.

        Args:
            src (Union[str, PathLike]): The source path on the local machine.
            dst (Union[str, PathLike]): The destination path on the device.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        return self._protocol_helper.push(src, str(dst), client=self._sftp_client)

    def pull(self, src: Union[str, PathLike], dst: Union[str, PathLike]) -> DeviceReturn:
        """Method to pull files from the device. Pull is analogous to push in that the
        device is the local host.

        Args:
            src (Union[str, PathLike]): The source path on the device.
            dst (Union[str, PathLike]): The destination path on the local machine.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        return self._protocol_helper.pull(str(src), dst, client=self._sftp_client)

    def make_directory(self, dir_name: Union[str, PathLike]) -> DeviceReturn:
        """Method to make a directory on an WOS Remote device.

        Args:
            dir_name (Union[str, PathLike]): The name of the directory to create.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        return self._protocol_helper.make_directory(str(dir_name), client=self._sftp_client)

    def close(self) -> Optional[DeviceReturn]:
        """Method to close any open connections that were created by this class instance
        Returns:
            Optional[DeviceReturn]: A DeviceReturn object representing the result of closing the
            connections or none if no open connections exist
        """
        if self._sftp_client:
            self._sftp_client.close()  # close the sftp client
        if not self._ssh_client or not self._ssh_client.get_transport().is_active():
            return None
        return self._protocol_helper.close(self._ssh_client)

    def remove(self, target_path: Union[str, PathLike]) -> DeviceReturn:
        """Method to remove a file or directory from the device.

        Args:
            target_path (Union[str, PathLike]): The path of the file or directory to remove.

        Returns:
            DeviceReturn: The device code after removing the file or directory.
        """
        return self._protocol_helper.remove(str(target_path), client=self._sftp_client)

    def execute(self, command: str, args: Optional[List[str]] = None, **kwargs) -> DeviceReturn:
        """Method to execute a command on the WOS Remote device.

        Args:
            command (str): The command to execute on the device.
            args (Optional[List[str]]): The arguments for the command. Defaults to None.

        Returns:
            DeviceReturn Union[DeviceCompletedProcess, DeviceFailedProcess]
        """
        if args is None:
            args = []
        command_args_str = f"{command} {' '.join(args)}"
        return self._protocol_helper.execute(command_args_str, client=self._ssh_client)

    @classmethod
    def get_available_devices(cls, destination: str, *_destinations) -> List[str]:
        """Retrieve a list of available devices from the specified destination(s).

        Args:
            destination (str): The primary destination to query for available devices.
            *_destinations (str): Additional destinations to query for available devices.

        Returns:
            List[str]: A list of available device identifiers or names.
        """
        available_devices = SSHProtocolHelper.get_available_devices(destination, *_destinations)
        return list(available_devices)
