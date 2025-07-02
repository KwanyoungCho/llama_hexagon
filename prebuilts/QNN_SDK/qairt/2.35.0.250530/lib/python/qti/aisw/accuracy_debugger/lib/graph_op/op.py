# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

class Op:
    def __init__(self, name) -> None:
        self._name = name
        self._type = None
        self._outputs = []
        self._inputs = []
        self._children_ops = []
        self._parent_ops = []

    def set_outputs(self, outputs: list):
        self._outputs = outputs

    def set_inputs(self, inputs: list):
        self._inputs = inputs

    def set_op_type(self, type: str):
        self._type = type

    def set_children_ops(self, children_ops: list):
        self._children_ops.extend(children_ops)

    def set_parent_ops(self, parent_ops: list):
        self._parent_ops.extend(parent_ops)

    def get_name(self):
        return self._name

    def get_outputs(self):
        return self._outputs

    def get_inputs(self):
        return self._inputs

    def get_op_type(self):
        return self._type

    def get_children_ops(self):
        return self._children_ops

    def get_parent_ops(self):
        return self._parent_ops
