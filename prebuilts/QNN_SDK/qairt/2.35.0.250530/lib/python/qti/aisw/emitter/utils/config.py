# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" Input Config for Emitter """
from dataclasses import dataclass, field
from typing import Dict, List

from qti.aisw.converters.common import ir_graph


@dataclass
class CustomOpInfo:
    op_type_to_module: Dict[str, str] = field(default_factory=dict)
    custom_module_paths: List[str] = field(default_factory=list)


def is_custom_ir_op(op: ir_graph.IrOp) -> bool:
    """
    Checks whether the op is a custom op or QNN Op. All the QNN IrOp will have the package name as "qti.aisw".
    :param op: IrOp which needs to be checked.
    :return bool: True if given IrOp is a custom op otherwise False
    """
    return op.attrs.get_string('packageName') != 'qti.aisw'
