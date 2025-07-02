# ==============================================================================
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# Not a contribution.
# ==============================================================================

import onnx
import onnx_graphsurgeon as gs

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model_helper import OnnxModelHelper
from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger

fold_constants_log_area = LogAreas.register_log_area("fold_constants")


def fold_constants(model: onnx.ModelProto, log_level="info") -> onnx.ModelProto:
    """Fold constants in `model` and run shape inference"""

    logger = QAIRTLogger.register_area_logger(fold_constants_log_area,
                                              level=log_level,
                                              formatter_val="extended",
                                              handler_list=["dev_console"])

    def _const_fold_pass(_model: onnx.ModelProto) -> onnx.ModelProto:
        graph = gs.import_onnx(_model)

        graph.fold_constants()

        _model = gs.export_onnx(graph.cleanup())

        try:
            _model = OnnxModelHelper.symbolic_shape_inference(_model)
        except ImportError:
            _model = OnnxModelHelper.shape_inference(_model)

        return _model

    def get_num_nodes(_model):
        def _get_num_graph_nodes(graph):
            num_nodes = len(graph.node)
            for node in graph.node:
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        num_nodes += _get_num_graph_nodes(attr.g)
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        for subgraph in attr.graphs:
                            num_nodes += _get_num_graph_nodes(subgraph)
            return num_nodes

        return _get_num_graph_nodes(_model.graph)

    init_num_nodes = get_num_nodes(model)
    prefold_num_nodes = init_num_nodes
    postfold_num_nodes = -1

    pass_num = 0

    while prefold_num_nodes != postfold_num_nodes:
        logger.info(f"Folding Constants | Pass {pass_num + 1}")
        pass_num += 1
        prefold_num_nodes = get_num_nodes(model)

        try:
            model = _const_fold_pass(model)
        except Exception as e:
            logger.error(f"Constant folding pass failed. Skipping subsequent passes.\nNote: Error was:\n{e}")
            break
        else:
            postfold_num_nodes = get_num_nodes(model)
            nodes_folded = prefold_num_nodes - postfold_num_nodes
            logger.info(
                f"Original: {prefold_num_nodes} | After folding: {postfold_num_nodes} | {nodes_folded} Node{'s' if nodes_folded != 1 else ''} folded\n"
            )

    logger.info(f"Ran {pass_num} constant folding passes")
    logger.info(
        f"Original Nodes: {init_num_nodes} | After folding: {postfold_num_nodes} | Folded {init_num_nodes - postfold_num_nodes} Nodes\n"
    )

    return model
