# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.graph_op.op import Op

class FrameworkOp(Op):
    def __init__(self, name) -> None:
        super().__init__(name)
