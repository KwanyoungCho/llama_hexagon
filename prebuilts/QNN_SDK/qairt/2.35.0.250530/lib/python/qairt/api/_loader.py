# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
from pathlib import Path
from typing import overload

from qairt.api.compiled_model import CompiledModel
from qairt.api.model import Model
from qairt.modules.dlc_module import DlcModule
from qairt.utils.asset_utils import AssetType, check_asset_type
from qairt.utils.exceptions import LoadAssetError


@overload
def load(path: str | os.PathLike) -> CompiledModel: ...


def load(path: str | os.PathLike, **load_args) -> Model:
    """
    Loads assets of the following types:
     - A DLC (.dlc)
     - A binary (.bin)

    Args:
        path (str): Path to the DLC or binary.
        **load_args: Additional arguments for loading a DLC.
                     See `qairt.modules.dlc_module` for details.

    Returns:

         Model: A model object if the path is a valid DLC
         CompiledModel: If the model is a binary or DLC compiled for a QAIRT Backend.

    Raises:
        LoadAssetError: If the path is not a valid DLC or binary.
    """
    try:
        if check_asset_type(AssetType.CTX_BIN, path):
            return CompiledModel.load(str(path), **load_args)
        elif check_asset_type(AssetType.DLC, path):
            # Load the DLC as a module

            module = DlcModule.load(path, **load_args)
            # return true if module has caches
            if len(module.caches) > 0:
                return CompiledModel(module=module)
            else:
                return Model(module=module)
        else:
            raise ValueError("Expected a DLC or binary.")

    except Exception as e:
        raise LoadAssetError(f"Path: {path} could not be loaded as a DLC or Binary.") from e
