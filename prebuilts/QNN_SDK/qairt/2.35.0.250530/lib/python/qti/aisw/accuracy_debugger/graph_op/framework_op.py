# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from dataclasses import dataclass

from qti.aisw.accuracy_debugger.graph_op.op import Op


@dataclass
class FrameworkOp(Op):
    """Represents an operator in the framework graph by extending Op class."""
