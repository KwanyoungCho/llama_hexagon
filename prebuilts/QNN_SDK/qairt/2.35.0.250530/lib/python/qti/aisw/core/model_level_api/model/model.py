#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from abc import ABC, abstractmethod
from typing import TypeVar


class Model(ABC):
    @abstractmethod
    def __init__(self, name):
        self.name = name


ModelT = TypeVar('ModelT', bound=Model)
