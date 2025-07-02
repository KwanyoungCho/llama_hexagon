# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.snooping.cumulative_layerwise_snooping import (
    CumlativeLayerwiseSnooper,
)
from qti.aisw.accuracy_debugger.snooping.layerwise_snooping import LayerwiseSnooper
from qti.aisw.accuracy_debugger.snooping.oneshot_layerwise import OneshotLayerwiseSnooper
from qti.aisw.accuracy_debugger.snooping.snooper import Snooper
from qti.aisw.accuracy_debugger.utils.constants import Algorithm


def get_snooper_class(algorithm: Algorithm) -> Snooper:
    """Returns the snooping class based on the algorithm.

    Args:
        algorithm: Enum specifying algorithm.

    Returns:
        Snooper: class of the snooping algorithm.

    Raises:
        NotImplementedError if algorithm is not supported.
    """
    if algorithm == Algorithm.ONESHOT:
        return OneshotLayerwiseSnooper
    if algorithm == Algorithm.LAYERWISE:
        return LayerwiseSnooper
    if algorithm == Algorithm.CUMULATIVE:
        return CumlativeLayerwiseSnooper
    raise NotImplementedError(f"Algorithm '{algorithm}' not implemented.")
