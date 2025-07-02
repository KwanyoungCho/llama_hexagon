# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from dataclasses import dataclass, field

from qti.aisw.accuracy_debugger.graph_op.op import Op


@dataclass
class TargetOp(Op):
    """Represents an operator in the target DLC by extending Op class."""

    data_type: str = None
    static_tensors: list = field(default_factory=list)
