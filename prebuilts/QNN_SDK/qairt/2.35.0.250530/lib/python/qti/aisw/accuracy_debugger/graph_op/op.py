# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


from dataclasses import dataclass, field


@dataclass
class Op:
    """Represents a generic operator in a computational graph.

    This class defines the basic attributes and functionalities of an operator, including
    its name, type, inputs, outputs, and relationships with other operations.
    """

    name: str
    op_type: str = None
    outputs: list = field(default_factory=list)
    inputs: list = field(default_factory=list)
    _children_ops: list = field(default_factory=list)
    _parent_ops: list = field(default_factory=list)

    @property
    def children_ops(self) -> list:
        return self._children_ops

    @children_ops.setter
    def children_ops(self, children_ops: list) -> None:
        self._children_ops.extend(children_ops)

    @property
    def parent_ops(self) -> list:
        return self._parent_ops

    @parent_ops.setter
    def parent_ops(self, parent_ops: list) -> None:
        self._parent_ops.extend(parent_ops)
