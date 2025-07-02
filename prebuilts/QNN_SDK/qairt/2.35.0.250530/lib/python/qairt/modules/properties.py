# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import ABC, abstractmethod
from typing import Any, List

from qairt.api.configs.common import BackendType
from qairt.api.configs.device import DevicePlatformType


class Executable(ABC):
    """
    Mixin class for executable assets.
    """

    _exec_ready = True

    @property
    def executable(self):
        return self._exec_ready

    @abstractmethod
    def supported_backends(self) -> List[BackendType]:
        raise NotImplementedError("Supported backends are not implemented")

    @abstractmethod
    def supported_platforms(self) -> List[DevicePlatformType]:
        raise NotImplementedError("Supported platforms are not implemented")


class Loadable(ABC):
    """Mixin class for loadable assets."""

    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Load method is not implemented")

    def save(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Save method is not implemented")


class AttrNoCopy:
    def __init__(self, value):
        self._value = value

    def __get__(self, instance, owner):
        return self._value

    def __set__(self, instance, value):
        if self._value is None:
            self._value = value
        if self._value != value:
            raise AttributeError("This attribute is read-only.")

    def __delete__(self, instance):
        raise AttributeError("This attribute is read-only.")


class Queryable(ABC):
    """
    Mixin class for retrieving properties. One of info or
    get_property should be implemented.
    """

    @property
    def queryable(self) -> bool:
        """
        Returns True if this object is queryable, implying that either
        self.info or self.get_property is implemented.
        """
        return True

    @property
    def info(self) -> Any:
        """
        Returns an object that contains information about this object.
            Intended for objects with visible read-only properties.
        """
        raise NotImplementedError("info is not implemented.")


class GraphMixin(Queryable):
    pass


class DlcMixin(Executable, Queryable, Loadable):
    pass


class CacheMixin(Queryable, Executable, Loadable):
    pass
