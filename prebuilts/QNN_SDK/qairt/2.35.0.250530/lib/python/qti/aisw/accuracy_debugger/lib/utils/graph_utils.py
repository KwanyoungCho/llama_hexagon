# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.graph_op.target_op import TargetOp
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import RELU_OPS


def get_common_parent_activations(current_activation: str, graph1_activation_op_map: dict,
                                  graph2_activation_op_map: dict, ignore_activations: set = set()) -> set:
    '''
    Given the op activation in graph1, it returns the set of parent op
    activations which is present in both graph1 and graph2. This is important
    incase of supergroups.

    :param current_activation: activation in the framework graph
    :param graph1_activation_op_map: graph1_activation_op_map with
                            graph1_activations as key and graph1_op as value
    :param graph2_activation_op_map: graph2_activation_op_map with
                            graph2_activations as key and graph2_op as value
    :param ignore_activations: set of activations which has to be ignored
        while finding the common parent activations. for e.g. for conv, bn,
    relu qnn_net_json and qairt_encodings_json gives out activation encodings for both bn
    and relu user may want to delete bn encodings for some optimizations.
    '''

    if current_activation in (graph2_activation_op_map.keys() - ignore_activations):
        return set([current_activation])

    parent_activations = set()
    graph1_op = graph1_activation_op_map[current_activation]
    for input_name in graph1_op.get_inputs():
        # some input_name can be params
        if input_name in graph1_activation_op_map:
            activations = get_common_parent_activations(input_name, graph1_activation_op_map,
                                                        graph2_activation_op_map,
                                                        ignore_activations)
            parent_activations.update(activations)

    return parent_activations


def trace_subgraph(
    subgraph_input_names: set, subgraph_output_name: str, graph1_activation_op_map: dict,
    graph2_activation_op_map: dict = dict(), ignore_activations: set = set(),
    visited_activations: dict = dict()
) -> tuple[set, set]:
    '''
    Trace out all the intermediate tensor names in graph1 which are
    part of the subgraph that starts with subgraph_input_names and ends with
    subgraph_output_name but also present in the graph2

    :param subgraph_input_names: set of subgraph input names
    :param subgraph_output_name: one of the final output of the subgraph
    :param graph1_activation_op_map: graph1_activation_op_map with
                            graph1_activations as key and graph1_op as value
    :param graph2_activation_op_map: graph2_activation_op_map with
                            graph2_activations as key and graph2_op as value
    :param ignore_activations: set of activations which has to be ignored
        while finding the common parent activations. for e.g. for conv, bn,
    relu qnn_net_json and qairt_encodings_json gives out activation encodings for both bn
    and relu user may want to delete bn encodings for some optimizations.
    :param visited_activations: memoization dicionary to store the subgraph inputs that
        can be visited from the given activation. Helps tracing the subgraph faster.
    :return subgraph_tensor_names: set of intermediate tensors which are part of the target subgraph
        but also present in the framework subgraph
    :return visited_inputs: set of subgraph input tensors which are visited
    '''
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
    for input_name in graph_op.get_inputs():
        # some input_name can be params
        if input_name in graph1_activation_op_map:
            partial_subgraph_tensors, partial_visited_inputs = trace_subgraph(
                subgraph_input_names, input_name, graph1_activation_op_map,
                graph2_activation_op_map, ignore_activations, visited_activations)
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
    subgraph_input_names: set, subgraph_output_names: set, graph1_activation_op_map: dict,
    graph2_activation_op_map: dict = dict(), ignore_activations: set = set()
) -> tuple[set, set, set]:
    '''
    Find the subgraph intermediate tensor names in graph1 which are
    part of the subgraph that starts with subgraph_input_names and ends with
    subgraph_output_name but also present in the graph2(if graph2 is not empty)

    :param subgraph_input_names: set of subgraph input names
    :param subgraph_output_names: set of subgraph output names
    :param graph1_activation_op_map: graph1_activation_op_map with
                            graph1_activations as key and graph1_op as value
    :param graph2_activation_op_map: graph2_activation_op_map with
                            graph2_activations as key and graph2_op as value
    :param ignore_activations: set of activations which has to be ignored
        while finding the common parent activations. for e.g. for conv, bn,
    relu qnn_net_json and qairt_encodings_json gives out activation encodings for both bn
    and relu user may want to delete bn encodings for some optimizations.
    :return subgraph_tensor_names: set of intermediate tensors which are part of the target subgraph
        but also present in the framework subgraph
    :return visited_inputs: set of subgraph input tensors which are visited
    :return visited_outputs: set of graph output names visited out of debug_graph_output_names
    '''

    # If subgraph input_names or output_names are empty
    if not subgraph_input_names or not subgraph_output_names:
        return set(), set(), set()

    subgraph_activations = set()
    visited_inputs = set()
    visited_outputs = set()
    visited_activations = dict()
    for subgraph_output_name in subgraph_output_names:
        # find partial debug graph between inputs to the graph
        # and output of a partial graph
        partial_subgraph_activations, partial_visited_inputs = trace_subgraph(
            subgraph_input_names, subgraph_output_name, graph1_activation_op_map,
            graph2_activation_op_map, ignore_activations, visited_activations)
        subgraph_activations.update(partial_subgraph_activations)
        if partial_subgraph_activations:
            visited_outputs.update([subgraph_output_name])
            visited_inputs.update(partial_visited_inputs)

    return subgraph_activations, visited_inputs, visited_outputs


def _is_part_of_relu_supergroup(target_op: TargetOp, framework_activation_op_map: dict) -> bool:
    '''
    Finds out the following supergroups that exists in the target graph which starts with target_op:
    1. conv -> relu
    2. bn -> relu

    :param target_op: object of TargetOp
    :param framework_activation_op_map: dictionary with framework activation
            as keys and framework op as value
    :return: True, if target_op is part of relu supergroup else False
    '''
    # target does not fuse {conv, relu} and {bn, relu}

    # If number of children ops of the conv/bn != 1 or outputs != 1
    if len(target_op.get_children_ops()) != 1 or len(target_op.get_outputs()) != 1:
        return False

    next_op = target_op.get_children_ops()[0]
    activations = next_op.get_outputs()
    # If next_op is not of type ElementWiseNeuron
    if next_op.get_op_type().lower() != 'elementwiseneuron':
        return False

    # If number of outputs != 1 or inputs != 1
    if len(activations) != 1 or len(next_op.get_inputs()) != 1:
        return False

    # If acitvations[0] not in framework graph
    if activations[0] not in framework_activation_op_map:
        return False

    # conv(100) -> bn(101) -> relu(102) -> transpose(103), gets fused
    # conv(100) -> relu(103) in target. Check whether relu is there after conv/bn or not.
    framework_activations, _, _ = get_subgraph(target_op.get_outputs(), activations, framework_activation_op_map)
    framework_ops = set([framework_activation_op_map[activation].get_op_type().lower() for activation in framework_activations])
    if framework_ops.isdisjoint(set(RELU_OPS)):
        return False

    return True


def get_supergroup_activations(framework_activation_op_map, target_activation_op_map) -> set:
    '''
        Finds out the set of framework activations which are part of
        some supergroup. The last activation of the supergroup is not
        included.


        :param framework_activation_op_map: dictionary with framework activation
            as keys and framework op as value
        :param target_activation_op_map: dictionary with target activation
            as keys and target op as value
        :return supergroup_activations: set of conv/bin activation which are part of
            conv-bn-relu/bn-relu/conv-relu/conv-mul-add-relu/
            conv-mul-add fusion pattern
        '''

    # Following target graph supergroups are targeted
    # 1. conv -> relu
    # 2. bn -> relu

    supergroup_activations = set()

    for activation_name in target_activation_op_map:
        if activation_name in framework_activation_op_map:
            target_op = target_activation_op_map[activation_name]
            target_op_type = target_op.get_op_type().lower()
            if ('conv' in target_op_type and target_op_type != 'convert' ) or\
                'batchnorm' in target_op_type:
                if _is_part_of_relu_supergroup(target_op, framework_activation_op_map):
                    supergroup_activations.update(target_op.get_outputs())

    return supergroup_activations


def get_invalidate_activations(activations: list, graph_activation_op_map: dict) -> set:
    '''
    for each activation in activations checks whether it is present
    in the graph_activation_op_map

    :param activations: list of activations
    :param graph_activation_op_map: dictionary of graph activations as keys
        and graph op as value

    :return invalid_activations: set of activation which are not present in the
        graph
    '''
    invalid_activations = set()
    for activation in activations:
        if activation not in graph_activation_op_map:
            invalid_activations.update([activation])

    return invalid_activations


def validate_inputs_outputs(input_names: list, output_names: list,
                            graph_activation_op_map: dict) -> None:
    '''
    Validates whether the input_names and output_names belongs to the
    graph or not.

    :param input_names: list of input tensor names
    :param output_names: list of output tensor names
    :param graph_activation_op_map: dictionary of graph activations as keys
        and graph op as value

    :raise Exception: If any of the input or output tensor does not belong to the graph
    '''

    invalid_input_tensors = get_invalidate_activations(input_names, graph_activation_op_map)
    invalid_output_tensors = get_invalidate_activations(output_names, graph_activation_op_map)

    exception_msg = []
    if invalid_input_tensors:
        exception_msg.append(f"Invalid Input Tensor: {str(invalid_input_tensors)}")
    if invalid_output_tensors:
        exception_msg.append(f"Invalid Output Tensor: {str(invalid_output_tensors)}")

    if exception_msg:
        raise Exception('\n'.join(exception_msg))


def get_topological_order(graph_activation_op_map: dict) -> list:
    '''
    Finds out the topological sort of the activations in the graph
    on the basis of level order traversal.

    :param graph_activation_op_map: graph activation as keys and graph_op as value
    :return topological_sort: list of topological sorted activation
    '''

    in_degree = {
        activation: len(graph_op.get_parent_ops())
        for activation, graph_op in graph_activation_op_map.items()
    }
    zero_degree_ops = [
        graph_op for _, graph_op in graph_activation_op_map.items()
        if not graph_op.get_parent_ops()
    ]
    topological_sort = []

    while zero_degree_ops:
        zero_degree_op = zero_degree_ops.pop(0)
        for output_name in zero_degree_op.get_outputs():
            # Incase of mult-output node like split, there will be multiple
            # occurances of zero_degree_op in zero_degree_ops.
            if output_name not in topological_sort:
                topological_sort.append(output_name)
        for children_op in zero_degree_op.get_children_ops():
            for children_output_name in children_op.get_outputs():
                in_degree[children_output_name] -= 1
                if in_degree[children_output_name] == 0:
                    zero_degree_ops.append(children_op)

    return topological_sort
