# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os

qacc_logger = logging.getLogger('qacc')
qacc_file_logger = logging.getLogger('qacc_logfile')

# limiting logging for tensor flow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
