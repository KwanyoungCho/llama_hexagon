# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import re
from typing import List, Optional

# TODO: Handle qti imports in a common location
from qti.aisw.tools.core.modules.api.definitions import AISWBaseModel
from qti.aisw.tools.core.utilities.devices.android.android_device import AndroidDevice
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    ConnectionType,
    DeviceCredentials,
    DeviceEnvironmentContext,
    DeviceIdentifier,
    DeviceInfo,
    DevicePlatformType,
    RemoteDeviceIdentifier,
    RemoteDeviceInfo,
    SocDetails,
)
from qti.aisw.tools.core.utilities.devices.api.device_factory import DeviceFactory


class Device(AISWBaseModel):
    """
    An object that captures information about the intended device.

    Supported Platform types are:
    - DevicePlatformType.ANDROID
      DevicePlatformType.LINUX_EMBEDDED
      DevicePlatformType.QNX
      DevicePlatformType.WOS
      DevicePlatformType.X86_64_LINUX
      DevicePlatformType.X86_64_WINDOWS_MSVC

    For remote connections, additional information such as hostname or ip, and credentials
    may be required.

    A remote connection requiring a username may be specified as follows:

    .. code-block:: python
        device = Device(type=DevicePlatformType.ANDROID)
        device_with_id = Device(type=DevicePlatformType.ANDROID,
                                identifier=RemoteDeviceIdentifier(serial_id="a1234bc"))

    Attributes:
        type (DevicePlatformType): The type of device platform to be used
        identifier (Optional[RemoteDeviceIdentifier]): The identifier of the device.
                                                        Defaults to None.
        credentials (Optional[DeviceCredentials]): The credentials for the device. Defaults to
        None.
    """

    type: DevicePlatformType
    identifier: Optional[RemoteDeviceIdentifier | str] = None
    credentials: Optional[DeviceCredentials] = None

    _info: Optional[RemoteDeviceInfo | DeviceInfo] = None

    @property
    def info(self) -> DeviceInfo:
        if self._info:
            pass
        elif self.identifier and isinstance(self.identifier, RemoteDeviceIdentifier):
            self._info = RemoteDeviceInfo(
                platform_type=self.type, identifier=self.identifier, credentials=self.credentials
            )
        else:
            self._info = DeviceInfo(platform_type=self.type, identifier=self.identifier)

        return self._info

    def get_chipset(self) -> str:
        """Returns the chipset associated with this device. Note that only
        Android devices are supported."""
        if self.type == DevicePlatformType.ANDROID:
            device_factory_instance = DeviceFactory.create_device(self.info)
            chipset = device_factory_instance.get_soc_name()
            # TODO: Remove Hack for 8650 missing in map
            if "UNKNOWN" in chipset:
                # retrieve soc id
                soc_id = device_factory_instance.executor.query_soc_id()
                if soc_id == "577":
                    chipset = "SM8650"
            return chipset
        else:
            print(f"Could not resolve chipset for device of type: {self.type} ")
            return ""

    def __str__(self):
        return f"{str(self.model_dump(exclude_none=True, exclude_unset=True, exclude={'credentials'}))}"


def soc_details_from_str(specs_str):
    return get_soc_details(specs_str)[0]


def populate_soc_details_from_factory(soc_detail: SocDetails, backend: str = "HTP"):
    """Populates soc detail with additional information given a backend. This function
    will override any preset values."""

    if not soc_detail.chipset:
        print(f"Could not determine soc details without chipset")
        return False

    try:
        soc_details_from_factory = DeviceFactory.get_device_soc_details(backend, soc_detail.chipset)
        soc_detail.dsp_arch = soc_details_from_factory.dsp_arch
        soc_detail.model = soc_details_from_factory.model
        soc_detail.num_of_hvx_threads = soc_details_from_factory.num_of_hvx_threads
        soc_detail.vtcm_size_in_mb = soc_details_from_factory.vtcm_size_in_mb
        soc_detail.supports_fp16 = soc_details_from_factory.supports_fp16
    except Exception as e:
        print(f"Could not determine soc details due to: {e}")
        return False
    return True


def get_soc_details(specs_str: str) -> List[SocDetails]:
    """Transforms a spec str into a device spec object

    Matches one or more instances of either:
        1. chipset:abc-123 e.x chipset:SM8550
        2. dsp_arch:v123;soc_model:123|456 e.x dsp_arch:v73;soc_model:43|50
        3. chipset:abc-123;dsp_arch:v123;soc_model:123|456 e.x chipset:SM8550;dsp_arch:v73;soc_model:43|50

    Returns:
        A list of soc detail objects

    Raises:
        ValueError: If the specs string is not valid or no matches were found
    """

    pattern = r"(chipset:[\w-]+|dsp_arch:[\w-]+;soc_model:\d+(\|\d+)*)"

    # Find all matches in the input string
    matches = re.findall(pattern, specs_str)

    # Create a list of SocDetails instances from the matches
    soc_details_default = lambda: SocDetails(
        chipset="", model="", dsp_arch=0, num_of_hvx_threads=0, vtcm_size_in_mb=0, supports_fp16=False
    )
    soc_details = soc_details_default()

    if not matches and specs_str:
        raise ValueError("Input does not match expected format")

    soc_details_list = []

    # To illustrate the code below consider the example:
    # chipset:SM8550;dsp_arch:v73;soc_model:60
    # The matches are: [("chipset:SM8550", ""), (";dsp_arch:v73;soc_model:69", '')]
    # The first pass through adds chipset to a partially formed soc detail.
    # The second pass populates the same soc detail with a dsp arch and soc model.
    for match in matches:
        entry = match[0]

        if entry.startswith("chipset:"):
            # if chipset or dsp arch has already been found,
            # then create a new soc detail
            if soc_details.chipset or soc_details.dsp_arch:
                soc_details_list.append(soc_details)
                soc_details = soc_details_default()

            soc_details.chipset = entry.split(":")[1]

        elif entry.startswith("dsp_arch:"):
            dsp_arch_part, soc_model_part = entry.split(";")
            dsp_arch = dsp_arch_part.split(":")[1]
            soc_model = soc_model_part.split(":")[1]

            # TODO: Remove when dsp arch is a str by default
            idx = dsp_arch.find("v")
            if idx != -1:
                dsp_arch = dsp_arch[idx + 1 :]

            soc_details.dsp_arch = int(dsp_arch)
            soc_details.model = soc_model

            if not soc_details.model:
                raise ValueError("Soc model should be provided along with dsp arch")

            if not soc_details.chipset:
                # if chipset is not populated
                soc_details_list.append(soc_details)
                soc_details = soc_details_default()

    # add final soc details to list
    if soc_details.chipset or soc_details.dsp_arch:
        soc_details_list.append(soc_details)

    return soc_details_list
