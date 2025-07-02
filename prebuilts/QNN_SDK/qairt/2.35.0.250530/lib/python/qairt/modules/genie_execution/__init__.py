# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qairt.utils import loggers

logger = loggers.get_logger(name=__name__)
try:
    from qti.aisw.genai import genie
except ImportError as e:
    logger.warn(f'Failed to import libPyGenie. Exception: "{e}". Loading dummy stand-in instead.')
    from qairt.modules.genie_execution import dummy_lib_genie as genie
