# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.graph_op.target_op import TargetOp
from qti.aisw.accuracy_debugger.utils.constants import RELU_OPS


def get_common_parent_activations(
    current_activation: str,
    graph1_activation_op_map: dict,
    graph2_activation_op_map: dict,
    ignore_activations: set,
) -> set:
    """Given the op activation in graph1, it returns the set of parent op
    activations which is present in both graph1 and graph2. This is important
    incase of supergroups.

    Args:
        current_activation (str): activation in the framework graph
        graph1_activation_op_map (dict): graph1_activation_op_map with
            graph1_activations as key and graph1_op as value
        graph2_activation_op_map (dict): graph2_activation_op_map with
            graph2_activations as key and graph2_op as value
        ignore_activations (set): set of activations which has to be ignored
            while finding the common parent activations. for e.g. for conv, bn,
            relu qnn_net_json and qairt_encodings_json gives out activation encodings for both bn
            and relu user may want to delete bn encodings for some optimizations.

    Returns:
        set: Set of parent activations
    """
    if current_activation in (graph2_activation_op_map.keys() - ignore_activations):
        return set([current_activation])
    parent_activations = set()
    graph1_op = graph1_activation_op_map[current_activation]
    for input_name in graph1_op.inputs:
        # some input_name can be params
        if input_name in graph1_activation_op_map:
            activations = get_common_parent_activations(
                input_name, graph1_activation_op_map, graph2_activation_op_map, ignore_activations
            )
            parent_activations.update(activations)
    return parent_activations


def trace_subgraph(
    subgraph_input_names: set,
    subgraph_output_name: str,
    graph1_activation_op_map: dict,
    graph2_activation_op_map: dict = dict(),
    ignore_activations: set = set(),
    visited_activations: dict = dict(),
) -> tuple[set, set]:
    """Trace out all the intermediate tensor names in graph1 which are
    part of the subgraph that starts with subgraph_input_names and ends with
    subgraph_output_name but also present in the graph2

    Args:
        subgraph_input_names (set): set of subgraph input names
        subgraph_output_name (str): one of the final output of the subgraph
        graph1_activation_op_map (dict): graph1_activation_op_map with
                            graph1_activations as key and graph1_op as value
        graph2_activation_op_map (dict, optional): graph2_activation_op_map with
                            graph2_activations as key and graph2_op as value. Defaults to dict().
        ignore_activations (set, optional): set of activations which has to be ignored
                            while finding the common parent activations. for e.g. for conv, bn,
                            relu qnn_net_json and qairt_encodings_json gives out activation
                            encodings for both bn and relu user may want to delete bn encodings for
                            some optimizations. Defaults to set().
        visited_activations (dict, optional): memoization dicionary to store the subgraph inputs
                            that can be visited from the given activation. Helps tracing the
                            subgraph faster. Defaults to dict().

    Returns:
        tuple[set, set]: set of intermediate tensors, set of subgraph input tensors visited.
    """
    # ----------------------->>>--> op -------------->
    # subgraph_input_tensors            tensor_name
    # If the subgraph_output_name is already traced, return the subgraph_inputs that
    # can be traced from here
    if subgraph_output_name in visited_activations:
        return set(), visited_activations[subgraph_output_name]
    # If we hit any of the subgraph inputs, return ()
    # input_tensors are not part of intermediate tensors
    if subgraph_output_name in subgraph_input_names:
        return set(), set([subgraph_output_name])
    subgraph_tensor_names = set()
    visited_inputs = set()
    graph_op = graph1_activation_op_map[subgraph_output_name]
    for input_name in graph_op.inputs:
        # some input_name can be params
        if input_name in graph1_activation_op_map:
            partial_subgraph_tensors, partial_visited_inputs = trace_subgraph(
                subgraph_input_names,
                input_name,
                graph1_activation_op_map,
                graph2_activation_op_map,
                ignore_activations,
                visited_activations,
            )
            subgraph_tensor_names.update(partial_subgraph_tensors)
            visited_inputs.update(partial_visited_inputs)
            visited_activations[input_name] = partial_visited_inputs
    # If any of the subgraph inputs are visited, we can say that the path traced is part of the
    # subgraph, else this subgraph_output_name does not belong to the subgraph.
    # If current subgraph_output_name needs to be ignored while tracing
    if visited_inputs and subgraph_output_name not in ignore_activations:
        # If graph2 given, then trace out only common activations
        if graph2_activation_op_map:
            if subgraph_output_name in graph2_activation_op_map:
                subgraph_tensor_names.update([subgraph_output_name])
            else:
                return subgraph_tensor_names, visited_inputs
        else:
            # If graph2 not given, add subgraph_output_name to traced subgraph
            subgraph_tensor_names.update([subgraph_output_name])
    else:
        # None of the subgraph inputs are visited -> subgraph_output_name not part of the
        # subgraph. or subgraph_output_name part of ignore_activations
        return subgraph_tensor_names, visited_inputs
    return subgraph_tensor_names, visited_inputs


def get_subgraph(
    subgraph_input_names: set,
    subgraph_output_names: set,
    graph1_activation_op_map: dict,
    graph2_activation_op_map: dict = dict(),
    ignore_activations: set = set(),
) -> tuple[set, set, set]:
    """Find subgraph intermediate tensor names in graph1. It identifies the subgraph
    that starts with subgraph_input_names and ends with subgraph_output_names, and
    checks if they are present in graph2 (if graph2 is not empty).

    Args:
        subgraph_input_names (set): Set of subgraph input names.
        subgraph_output_names (set): Set of subgraph output names.
        graph1_activation_op_map (dict): Activation to op mapping for graph1.
        graph2_activation_op_map (dict, optional): Activation to op mapping for graph2.
                                                   Defaults to an empty dict().
        ignore_activations (set, optional): Set of activations to ignore while finding
                                            common parent activations. Defaults to an empty set.

    Returns:
        tuple[set, set, set]: Set of intermediate tensors, set of visited subgraph input tensors,
                              and set of visited graph output names.

    Raises:
        ValueError: If either subgraph_input_names or subgraph_output_names is empty.
    """
    # Check if subgraph input_names or output_names are empty
    if not subgraph_input_names or not subgraph_output_names:
        return set(), set(), set()
    subgraph_activations = set()
    visited_inputs = set()
    visited_outputs = set()
    visited_activations = dict()

    for subgraph_output_name in subgraph_output_names:
        # Find the partial debug graph between inputs and the subgraph output
        partial_subgraph_activations, partial_visited_inputs = trace_subgraph(
            subgraph_input_names,
            subgraph_output_name,
            graph1_activation_op_map,
            graph2_activation_op_map,
            ignore_activations,
            visited_activations,
        )
        subgraph_activations.update(partial_subgraph_activations)
        if partial_subgraph_activations:
            visited_outputs.update([subgraph_output_name])
            visited_inputs.update(partial_visited_inputs)

    return subgraph_activations, visited_inputs, visited_outputs


def is_part_of_relu_supergroup(target_op: TargetOp, framework_activation_op_map: dict) -> bool:
    """Determines whether target_op is part of the relu supergroup (conv -> relu or bn -> relu)

    Args:
        target_op (TargetOp): object of TargetOp
        framework_activation_op_map (dict): dictionary with target activation
                as keys and target op as value
    Returns:
        bool: True, if target_op is part of relu supergroup else False
    """
    # target does not fuse {conv, relu} and {bn, relu}

    # Validate target_op has exactly one child operation and one output
    if len(target_op.children_ops) != 1 or len(target_op.outputs) != 1:
        return False

    next_op = target_op.children_ops[0]
    # For the next_op, validate following:
    # 1. It is of type ElementWiseNeuron
    # 2. It has exactly one input and output
    # 3. The output must be also present in the framework graph
    # If it is otherwise, this is not relu supergroup
    if (
        next_op.op_type.lower() != "elementwiseneuron"
        or len(next_op.outputs) != 1
        or len(next_op.inputs) != 1
        or next_op.outputs[0] not in framework_activation_op_map
    ):
        return False

    # conv(100) -> bn(101) -> relu(102) -> transpose(103), gets fused
    # conv(100) -> relu(103) in target. Check whether relu is there after conv/bn or not.
    framework_activations, _, _ = get_subgraph(
        target_op.outputs, next_op.outputs, framework_activation_op_map
    )
    if not any(
        framework_activation_op_map[activation].op_type.lower() in RELU_OPS
        for activation in framework_activations
    ):
        return False

    return True


def get_supergroup_activations(
    framework_activation_op_map: dict, target_activation_op_map: dict
) -> set:
    """Finds out the set of framework activations which are part of
    some supergroup. The last activation of the supergroup is not
    included.

    Args:
        framework_activation_op_map (dict): dictionary with framework activation
                            as keys and framework op as value
        target_activation_op_map (dict): dictionary with target activation
                            as keys and target op as value

    Returns:
        set: set of activation which are part of conv-bn-relu/bn-relu/conv-relu/conv-mul-add-relu/
                            conv-mul-add fusion pattern
    """
    supergroup_activations = set()
    for activation_name in target_activation_op_map:
        if activation_name in framework_activation_op_map:
            target_op = target_activation_op_map[activation_name]
            target_op_type = target_op.op_type.lower()
            if (
                "conv" in target_op_type and target_op_type != "convert"
            ) or "batchnorm" in target_op_type:
                if is_part_of_relu_supergroup(target_op, framework_activation_op_map):
                    supergroup_activations.update(target_op.outputs)
    return supergroup_activations


def get_invalidate_activations(activations: list, graph_activation_op_map: dict) -> set:
    """For each activation in activations checks whether it is present
    in the graph_activation_op_map

    Args:
        activations (list): list of activations
        graph_activation_op_map (dict): dictionary of graph activations as keys
        and graph op as value

    Returns:
        set: set of activation which are not present in the graph
    """
    invalid_activations = set()
    for activation in activations:
        if activation not in graph_activation_op_map:
            invalid_activations.update([activation])
    return invalid_activations


def validate_inputs_outputs(
    input_names: list, output_names: list, graph_activation_op_map: dict
) -> None:
    """Validates whether the input_names and output_names belongs to the
    graph or not.

    Args:
        input_names (list): list of input tensor names
        output_names (list): list of output tensor names
        graph_activation_op_map (dict): dictionary of graph activations as keys
        and graph op as value

    Raises:
        Exception: If any of the input or output tensor does not belong to the graph
    """
    invalid_input_tensors = get_invalidate_activations(input_names, graph_activation_op_map)
    invalid_output_tensors = get_invalidate_activations(output_names, graph_activation_op_map)
    exception_msg = []
    if invalid_input_tensors:
        exception_msg.append(f"Invalid Input Tensor: {str(invalid_input_tensors)}")
    if invalid_output_tensors:
        exception_msg.append(f"Invalid Output Tensor: {str(invalid_output_tensors)}")
    if exception_msg:
        raise Exception("\n".join(exception_msg))


def get_topological_order(graph_activation_op_map: dict) -> list:
    """Finds out the topological sort of the activations in the graph
    on the basis of level order traversal.

    Args:
        graph_activation_op_map (dict): graph activation as keys and graph_op as value

    Returns:
        list: list of topological sorted activation
    """
    in_degree = {
        activation: len(graph_op.parent_ops)
        for activation, graph_op in graph_activation_op_map.items()
    }
    zero_degree_ops = [
        graph_op for _, graph_op in graph_activation_op_map.items() if not graph_op.parent_ops
    ]
    topological_sort = []
    while zero_degree_ops:
        zero_degree_op = zero_degree_ops.pop(0)
        for output_name in zero_degree_op.outputs:
            # Incase of mult-output node like split, there will be multiple
            # occurances of zero_degree_op in zero_degree_ops.
            if output_name not in topological_sort:
                topological_sort.append(output_name)
        for children_op in zero_degree_op.children_ops:
            for children_output_name in children_op.outputs:
                in_degree[children_output_name] -= 1
                if in_degree[children_output_name] == 0:
                    zero_degree_ops.append(children_op)
    return topological_sort
