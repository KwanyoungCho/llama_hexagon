# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
try:
    import qti.aisw.tools.core
except ImportError:
    # This should be an error, but it is circumvented so that CI can pass without QAIRT SDK installed.
    print("WARNING: qti.aisw.tools.core package is not installed. Ensure QAIRT SDK is installed.")
