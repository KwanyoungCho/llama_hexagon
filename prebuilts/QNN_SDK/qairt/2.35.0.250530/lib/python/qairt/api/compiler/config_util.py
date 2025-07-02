# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
from typing import TYPE_CHECKING, Dict

from pydantic import DirectoryPath

from qairt.api.compiler.backends.aic.config import AicConfigHelper
from qairt.api.compiler.backends.htp.config import HtpConfigHelper
from qairt.api.configs.common import BackendType

if TYPE_CHECKING:
    from qairt.api.compiler.config import CompileConfig


def get_config_api_options_dict(config: "CompileConfig") -> Dict:
    """
    Converts the compile config to a dictionary encoded as a json string.
    """

    if config.backend == BackendType.HTP:
        # TODO: Remove once schema is incorporated
        backend_extensions_dict = HtpConfigHelper.to_backend_extension_dict(
            config.context_custom_configs,
            config.graph_custom_configs,
            config.device_custom_configs,
            config.memory_custom_config,
        )
        shared_library_path = HtpConfigHelper.shared_library_path()
    elif config.backend == BackendType.AIC:
        backend_extensions_dict = AicConfigHelper.to_backend_extension_dict(config.compiler_custom_configs)
        shared_library_path = AicConfigHelper.shared_library_path()
    else:
        backend_extensions_dict = {}
        shared_library_path = ""

    compile_config_dict = {
        "backend_extensions": {"shared_library_path": shared_library_path, "config_file_path": ""}
    }

    # add non-schema compliant value of config as a dictionary
    compile_config_dict["backend_extensions"]["config_dict"] = backend_extensions_dict

    return compile_config_dict


# TODO: Files could be avoided on local host
def _get_config_api_options_json_file(config, output_dir: DirectoryPath, prefix="") -> str:
    # save backend extension dict to a file
    if prefix:
        prefix = f"{prefix}_"

    compile_cfg_dict = get_config_api_options_dict(config)
    if backend_cfg := compile_cfg_dict["backend_extensions"]["config_dict"]:
        backend_ext_file_path = output_dir / f"{prefix}{config.backend.lower()}_extensions.json"
        with open(str(backend_ext_file_path), "w+") as fp:
            json.dump(backend_cfg, fp, indent=4)
        compile_cfg_dict["backend_extensions"]["config_file_path"] = str(backend_ext_file_path)

    # remove the config dict from the compile config dict before saving to file
    del compile_cfg_dict["backend_extensions"]["config_dict"]

    compile_config_file_path = output_dir / f"{prefix}compile_config.json"
    with open(str(compile_config_file_path), "w+") as fp:
        json.dump(compile_cfg_dict, fp, indent=4)

    return str(compile_config_file_path)
