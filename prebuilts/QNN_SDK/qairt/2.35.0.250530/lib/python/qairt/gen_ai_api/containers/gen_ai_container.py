# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import abstractmethod
from os import PathLike
from typing import Optional

from qairt import Device
from qairt.gen_ai_api.executors.gen_ai_executor import GenAIExecutor


class GenAIContainer:
    """
    Final product of a GenAIBuilder constitutes all assets required for completing a generative AI task.
    """

    @abstractmethod
    def save(self, dest: str | PathLike, *, exist_ok: bool = False):
        """
        Serialize container to disk.  Note, this will copy artifacts into the destination directory, and update any
        configurations accordingly.

        Args:
            dest (str | PathLike): Path to save the artifacts
            exist_ok (bool): If True, raise an exception if an artifact already exists.
        """
        raise NotImplementedError(f"'save' has not been implemented for: {self.__class__.__name__}")

    @classmethod
    def load(cls, path: str | PathLike) -> "GenAIContainer":
        """
        Load a GenAIContainer from disk

        Args:
            path (str | PathLike): Path to load a previously serialized GenAIContainer from
        Returns:
            GenAIContainer: newly created instance
        """
        raise NotImplementedError(f"'load' has not been implemented for: {cls.__name__}")

    @abstractmethod
    def get_executor(self, device: Optional[Device] = None, **kwargs) -> GenAIExecutor:
        """
        Create a GenAIExecutor to run on the specified device with the artifacts contained in this container.

        Args:
            device (Optional[Device]): Device to run the executor on
        Returns:
            GenAIExecutor: GenAIExecutor to run use case represented by this container on specified device
        """
        raise NotImplementedError(f"'get_executor' has not been implemented for: {self.__class__.__name__}")
