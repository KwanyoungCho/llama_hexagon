# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import onnx
import onnxsim
import functools
from typing import Dict, Optional, List, Union, Any
from dataclasses import dataclass
import copy
import contextlib


@dataclass(frozen=True)
class PrunedInitializerInfo:
    """
    Data carrier containing initializer to be added and identity node to be removed in ONNX graph
    """
    initializer: onnx.TensorProto
    identity_node: onnx.NodeProto


def prepare_torch_to_onnx_export_args(onnx_export_args: Optional[Dict[str, Any]],
                             input_names: Optional[str], output_names: Optional[str]):
    """
    Prepare onnx export args into a dictionary to be later used when exporting to onnx.

    :param onnx_export_args: Starting onnx export args
    :param input_names: Input names to use for the model which appear in the prepared model inputs
    :param output_names: Output names to use for the model which appear in the onnx model exported during model prepare
    :return: Onnx export args as a dictionary for use in torch.onnx.export
    """
    if onnx_export_args is None:
        onnx_export_args = {}

    if input_names is not None:
        onnx_export_args['input_names'] = input_names
    if output_names is not None:
        onnx_export_args['output_names'] = output_names
    if 'opset_version' not in onnx_export_args or onnx_export_args['opset_version'] is None:
        onnx_export_args['opset_version'] = 17
    if 'do_constant_folding' not in onnx_export_args:
        onnx_export_args['do_constant_folding'] = True

    return onnx_export_args


def _get_all_initializers(onnx_graph: onnx.GraphProto, initializers: Union[List[onnx.TensorProto], None] = None) \
        -> List[onnx.TensorProto]:
    """
    Get all initializer names in the onnx graph. Also recursively gets initializer names for subgraphs of the graph.
    :param onnx_graph: Onnx graph to get initializer names for
    :param initializers: List of initializer names in the graph
    :return List of initializer names in the graph
    """
    if initializers is None:
        initializers = []
    for initializer in onnx_graph.initializer:
        initializers.append(initializer)
    for node in onnx_graph.node:
        for attribute in node.attribute:
            if getattr(attribute, 'g').name != '':
                _get_all_initializers(attribute.g, initializers)
    return initializers


def save_initializer_restored_onnx_graph(original_model_path: str,
                                         restored_model_path: str):
    """
    Load original ONNX model path and save restored ONNX model to specific path

    :param original_model_path: Path where the original ONNX artifact was stored
    :param restored_model_path: Path to store restored ONNX artifact
    """
    model = onnx.load(original_model_path)
    restored_model = restore_onnx_graph_initializers(model, inplace=True)
    save_as_external_data = model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF
    onnx.save(restored_model, restored_model_path, save_as_external_data=save_as_external_data)


def restore_onnx_graph_initializers(model: onnx.ModelProto,
                                    inplace: bool = False) -> onnx.ModelProto:
    """
    Copy original model and restore its pruned initializers

    :param model: Original ONNX ModelProto
    :param inplace: Whether to modify ModelProto by inplace manner or not
    :return: Initializer restored ONNX ModelProto
    """
    # pylint: disable=protected-access, no-member
    if not inplace:
        model = copy.deepcopy(model)

    onnx_graph = model.graph

    initializers = _get_all_initializers(onnx_graph)
    initializer_names = [initializer.name for initializer in initializers]
    pruned_initializer_map = _get_pruned_initializer_map(
        onnx_graph, initializers, initializer_names
    )

    for node in onnx_graph.node:
        for input_tensor in node.input:
            _restore_pruned_initializer(
                onnx_graph, input_tensor, pruned_initializer_map
            )

    # Remove all the detached "Identity" type nodes
    for pruned_initializer_info in pruned_initializer_map.values():
        onnx_graph.node.remove(pruned_initializer_info.identity_node)
    return model


def _get_pruned_initializer_map(onnx_graph: onnx.GraphProto,
                                initializers: List[onnx.TensorProto],
                                initializer_names: List[str]) -> Dict[str, PrunedInitializerInfo]:
    """
    Find pruned ONNX initializers by iterating Identity nodes

    :param onnx_graph: ONNX graph
    :param initializers: List of ONNX initializers
    :param initializer_names: List of model initializer names
    :return: Dictionary with output of identity node as key and PrunedInitializerInfo as value
    """
    pruned_initializer_map = {}
    for node in onnx_graph.node:
        if node.op_type == "Identity" and node.input[0] in initializer_names:
            index = initializer_names.index(node.input[0])
            initializer = copy.deepcopy(initializers[index])
            pruned_initializer_map[node.output[0]] = PrunedInitializerInfo(
                initializer, node
            )

    return pruned_initializer_map


def _restore_pruned_initializer(onnx_graph: onnx.GraphProto,
                                input_tensor: str,
                                pruned_initializer_map: Dict[str, PrunedInitializerInfo],
                                new_initializer_name: Optional[str] = None):
    """
    Create new Initializer to restore pruned Initializer

    :param onnx_graph: ONNX graph
    :param input_tensor: Input tensor name
    :param pruned_initializer_map: Dictionary with output of identity node as key and PrunedInitializerInfo as value
    :param new_initializer_name: Name for new initializer
    """
    if result := pruned_initializer_map.get(input_tensor):
        new_initializer = result.initializer
        new_initializer.name = new_initializer_name or input_tensor
        onnx_graph.initializer.append(new_initializer)

@contextlib.contextmanager
def disable_onnxsim_optimizers(skipped_optimizers: Optional[List[str]]) -> contextlib.AbstractContextManager:
    """Apply temporary monkey patch to onnxsim::simplify and restore original method

    Args:
        skipped_optimizers: optimizer names to disable during onnx simplification
    """
    method_name = 'simplify'
    original_method = getattr(onnxsim, method_name)
    patched_method = functools.partial(onnxsim.simplify,
                                       skipped_optimizers=skipped_optimizers)
    try:
        setattr(onnxsim, method_name, patched_method)
        yield
    finally:
        setattr(onnxsim, method_name, original_method)