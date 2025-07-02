# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from enum import Enum


class COMPARATORS(str, Enum):
    L1NORM = "l1 norm"
    L2NORM = "l2 norm"
    AVERAGE = "average"
    COSINE = "cosine"
    STANDARD_DEVIATION = "standard deviation"
    MSE = "mse"
    SNR = "snr"
    KLD = "kl divergence"
    RTOL_ATOL = "rtol_atol"
    L1_ERROR = "l1_error"
    MSE_REL = "mse rel"
    TOPK = "topk"
    ADJUSTED_RTOL_ATOL = "adjusted_rtol_atol"
    MAE = "mae"


class TensorShapeError(Exception):
    pass


class ComparisonError(Exception):
    pass
