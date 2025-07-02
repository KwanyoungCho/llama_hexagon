# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.graph_op.op import Op

class TargetOp(Op):

    def __init__(self, name) -> None:
        super().__init__(name)
        self._data_type = None
        self._static_tensor_names = []

    def set_data_type(self, dtype):
        self._data_type = dtype

    def set_static_tensors(self, static_tensors):
        self._static_tensor_names = static_tensors

    def get_data_type(self):
        return self._data_type

    def get_static_tensors(self):
        return self._static_tensor_names
