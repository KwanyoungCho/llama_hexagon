# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
from pathlib import Path

# TODO: Warnings should be errors but in the short term, we need
# allow some unit tests to pass without QAIRT SDK.

modeltools = None


try:
    from qti.aisw.converters.common import modeltools as qairt_tools_cpp
except ImportError:
    try:
        # May need this import for QAIRT
        from qti.aisw.dlc_utils import modeltools as qairt_tools_cpp
    except ImportError:
        print("WARNING: Unable to import qti.aisw.converters.common.modeltools")


try:
    from qti.aisw.tools.core import modules as qti_modules
    from qti.aisw.tools.core.modules import api as qti_module_api
    from qti.aisw.tools.core.modules import context_bin_gen

    file_path = os.path.abspath(
        os.path.join(qti_modules.__path__[0], "..", "..", "..", "..", "..", "..", "..")
    )
    sdk_root_path = Path(file_path).resolve()
    QNN_SDK_ROOT = os.getenv("QNN_SDK_ROOT", sdk_root_path)  # set qnn sdk root if unset
except ImportError:
    QNN_SDK_ROOT = ""
    print("WARNING: QNN_SDK_ROOT environment variable is not set.")
