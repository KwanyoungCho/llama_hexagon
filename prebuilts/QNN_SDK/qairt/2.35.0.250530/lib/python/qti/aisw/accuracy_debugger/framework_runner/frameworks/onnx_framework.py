# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from logging import Logger

from onnx import ModelProto
from qti.aisw.accuracy_debugger.graph_op.framework_op import FrameworkOp
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_framework import OnnxFramework
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model_helper import (
    OnnxModelHelper,
)


class CustomOnnxFramework(OnnxFramework):
    """Class representing the Onnx framework.

    This class provides methods for loading and creating connectd graph
    """

    def __init__(self, logger: Logger):
        """Init function for onnx framework class
        Args:
            logger (Logger): A python Logger instance
        """
        super().__init__(logger)

    def get_output_tensor_names(self, model: ModelProto) -> list:
        """Get the output tensor names from the model

        Args:
            model(ModelProto) : The loaded model.

        Returns:
            list: A list of output tensor names
        """
        output_tensor_names = OnnxModelHelper.get_output_names(model)
        return output_tensor_names

    def create_connected_graph(self, model: ModelProto) -> dict:
        """Create a map of node name to Op object

        Args:
            model(ModelProto) : The loaded model.

        Returns:
            dict: A dictionary mapping node names to their corresponding Op objects
        """
        graph = model.graph
        connected_graph = {}
        for idx, inp in enumerate(graph.input):
            name = f"input_{idx}"
            op = FrameworkOp(name)
            op.inputs = []
            op.outputs = [inp.name]
            op.op_type = "input"
            connected_graph[name] = op

        for idx, node in enumerate(graph.node):
            name = node.name if node.name else f"op_{idx}"
            op = FrameworkOp(name)
            op.inputs = node.input
            op.outputs = node.output
            op.op_type = node.op_type
            connected_graph[name] = op

        # Now set the children and parent ops for each op
        for _, node1 in connected_graph.items():
            for _, node2 in connected_graph.items():
                for output in node1.outputs:
                    if output in node2.inputs:
                        # node1 -> node2
                        node1.children_ops = [node2]
                        node2.parent_ops = [node1]

        return connected_graph
