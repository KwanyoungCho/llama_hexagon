# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging
import os
import shutil
from collections.abc import Iterator, MutableMapping
from enum import Enum
from pathlib import Path
from typing import Dict, TypeVar

from pydantic import BaseModel, computed_field


class AssetType(Enum):
    """
    Asset types supported by QAIRT

    Attributes:
        DLC: A Deep Learning Container (DLC) file.
        CTX_BIN: A context binary file.
        ADAPTER_BIN: A LoRA adapter binary file.
        SCHEMATIC_BIN: A schematic binary file.
        PROFILING_LOG: A profiling log file.
        GENERIC: A generic file.
    """

    DLC = "dlc"
    CTX_BIN = "bin"
    ADAPTER_BIN = "adapter_bin"
    SCHEMATIC_BIN = "schematic_bin"
    PROFILING_LOG = "log"
    GENERIC = "generic"  # purely for type checking on base

    @classmethod
    def validate_asset_type(cls, asset_type: "AssetType"):
        """Validate the given asset type."""
        return asset_type in cls.__members__

    @classmethod
    def get_extension(cls, asset_type: "AssetType"):
        """Return the extension of the given asset type."""
        match asset_type:
            case AssetType.DLC:
                return ".dlc"
            case AssetType.CTX_BIN | AssetType.ADAPTER_BIN | AssetType.SCHEMATIC_BIN:
                return ".bin"
            case AssetType.PROFILING_LOG:
                return ".log"
            case _:
                raise TypeError(f"Unknown asset type: {asset_type}")

    @classmethod
    def from_str(cls, asset_type_str: str, /):
        """
        Returns the AssetType enum corresponding to the given string.
        """
        for asset_type in AssetType.__members__.values():
            if asset_type_str == asset_type.name:
                return asset_type
        raise TypeError(f"Unknown asset type: {asset_type_str}")


def check_asset_type(asset_type: AssetType, *assets: str | os.PathLike) -> bool:
    """
    Checks if all the given assets are of the given type.
    """
    return all(asset_type == get_asset_type(asset) for asset in assets)


def get_asset_type(path: str | os.PathLike) -> AssetType:
    """
    Returns the AssetType of the given asset.
    Args:
        path: The path to the asset.
    """
    asset_path = Path(path).resolve()

    if not asset_path.exists():
        raise FileNotFoundError(f"Given path {asset_path} does not exist.")

    suffix = asset_path.suffix

    if suffix == ".bin":
        # not a great way to do this but not clear how
        # this is expected to be compared
        if "_schematic" in asset_path.name:
            return AssetType.SCHEMATIC_BIN
        elif "_bin" in asset_path.name:
            return AssetType.ADAPTER_BIN
        return AssetType.CTX_BIN
    elif suffix == ".dlc":
        return AssetType.DLC
    elif suffix == ".log":
        return AssetType.PROFILING_LOG

    else:
        raise ValueError(f"Expected known asset type. Received an unknown {suffix} for file: '{path}'")


class Asset(BaseModel):
    """
    Base class for assets.
    """

    _logger = logging.getLogger()
    path: os.PathLike
    delete: bool = False

    @computed_field
    @property
    def type(self) -> AssetType:
        return get_asset_type(self.path)

    def save(self, new_dir: str | os.PathLike):
        new_dir = Path(new_dir).resolve()
        new_path = new_dir / Path(self.path).name
        if new_path.exists():
            self._logger.warning(f"File {new_path} already exists. Overwriting.")
        shutil.copy(self.path, new_path)

    def __del__(self):
        if hasattr(self, "delete") and self.delete and os.path.exists(self.path):
            os.remove(self.path)


KT = TypeVar("KT")


class AssetMapping(MutableMapping[KT, Asset]):
    """
    A custom mutable mapping class that holds a mapping from string keys to Asset instances.
    Enforces asset coexistence rules as specified.
    """

    def __init__(self):
        self._assets: Dict[KT, Asset] = {}

    def __getitem__(self, key: KT) -> Asset:
        return self._assets[key]

    def __setitem__(self, key: KT, value: Asset) -> None:
        if not isinstance(value, Asset):
            raise ValueError("Value must be an instance of Asset.")

        self._assets[key] = value

    def __delitem__(self, key: KT) -> None:
        del self._assets[key]

    def __iter__(self) -> Iterator[KT]:
        return iter(self._assets)

    def __len__(self) -> int:
        return len(self._assets)

    def keys(self):
        return self._assets.keys()

    def values(self):
        return self._assets.values()

    def items(self):
        return self._assets.items()

    def __repr__(self):
        return f"{self.__class__.__name__}({self._assets})"
