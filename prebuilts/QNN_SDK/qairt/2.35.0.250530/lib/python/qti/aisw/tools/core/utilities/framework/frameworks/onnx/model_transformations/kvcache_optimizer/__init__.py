# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# Not a contribution.
#
# -----------------------------------------------------------------------------

from typing import List, Tuple, Union

import onnx

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.logger import (
    log_debug,
    log_info,
    log_warning,
    setup_logging,
)


class KVCacheOptimizer:
    def __init__(
        self,
        model: onnx.ModelProto,
        transpose_keycache: bool = True,
        output_new_key_value_only: bool = True,
        log_level: str = "info",
    ) -> None:
        setup_logging(log_level, "KVCacheOptimizer")
        self.model = model
        self.transpose_keycache = transpose_keycache
        self.output_new_key_value_only = output_new_key_value_only

        # Cache some useful node details
        self.nodes = self.model.graph.node
        self._all_inputs = [_input.name for _input in self.model.graph.input]
        self._all_outputs = [_output.name for _output in self.model.graph.output]
        self._key_concat_nodes, self._value_concat_nodes = self._get_kv_concat_nodes()
        self._non_transposed_key_concat_nodes = [
            node for node in self._key_concat_nodes if node.output[0] in self._all_outputs
        ]

    def _get_kv_concat_nodes(self) -> Tuple[List[onnx.NodeProto], List[onnx.NodeProto]]:
        """
        Return the list of concats that concat key and value tensors with past key or past value tensors
        When running transposed_keycache adaptation, the key concat nodes are of interest
        When running output_new_key_value_only adaptation, both key concat and value concat nodes are of interest
        """
        # Assumption is that all concats that have one of the inputs the same as an input to the graph
        # is either a key concat or a value concat
        key_value_concats = [
            node
            for node in self.nodes
            if node.op_type == "Concat"
            and ((node.input[0] in self._all_inputs) ^ (node.input[1] in self._all_inputs))
        ]

        def _is_key_concat(node):
            """
            Pattern matcher to differentiate a key concat from a value concat
            * When looking for value concat, a breadth first search would end at a MatMul where
              where one of the inputs to the MatMul is a from a Softmax node
            * When looking for key concat, a breadth first search ends at the Softmax node before
              a MatMul node as described above
            """

            def _is_matmul_with_softmax_input(_m_node):
                """Check if `node` is a MatMul node where one of it's input nodes is a Softmax"""

                producer = _m_node["producer"]
                _m_node = _m_node["node"]
                if _m_node.op_type == "MatMul":
                    # Consider the other input node
                    other_input = [_ip for _ip in _m_node.input if _ip not in producer.output][0]
                    input_node = self._find_node_by_output_name(other_input)
                    while input_node.op_type != "Softmax":
                        if len(input_node.input) > 1:
                            return False
                        input_node = self._find_node_by_output_name(input_node.input[0])

                    return input_node.op_type == "Softmax"
                return False

            queue = [{"node": node, "producer": None}]
            while queue:
                curr_node = queue.pop(0)
                _curr = curr_node["node"]
                if _curr.op_type == "Softmax":
                    # Is a key concat
                    return True
                elif _is_matmul_with_softmax_input(curr_node):
                    # Is a value concat
                    return False
                consumer_nodes = self._find_nodes_by_input_name(_curr.output[0])
                for consumer in consumer_nodes:
                    queue.append({"node": consumer, "producer": _curr})

            return False

        key_concats = [node for node in key_value_concats if _is_key_concat(node)]
        value_concats = [node for node in key_value_concats if node not in key_concats]

        return key_concats, value_concats

    def _find_nodes_by_input_name(self, input_name: str) -> List[onnx.NodeProto]:
        """Get list of nodes which have `input_name` as one of its inputs"""
        return [node for node in self.nodes if input_name in node.input]

    def _find_node_by_output_name(self, output_name: str) -> Union[onnx.NodeProto | None]:
        """Get the node with `output_name` as one of its outputs"""
        for node in self.nodes:
            if output_name in node.output:
                return node
        return None

    def _transposed_key_cache(self):
        """
        Currently, the model transposes the key tensors after concatenating new key with past key cache
        As a result, the entire keycache needs to be transposed on every inference

        For example, if the current size of key cache is (p, d) and the size of the new key tensor is (t, d),
        the new key tensor after concatenating is (p + d, t)
        This tensor will be transposed on every inference for every layer

        Instead transpose the key cache so that it's new shape is (d, p) and tranpose the new key tensor
        before concatenating. This way, there is no need to transpose the key cache everytime

        To avoid this, the optimization does the following:
            1. Transpose the new key tensor before concatenating with the past key tensor
            2. Output this keycache, which is transposed
        """

        def _remove_transpose(transpose: onnx.NodeProto):
            """Remove the `transpose` node from the onnx graph"""
            # Bridge the input and consumers of the transpose node
            transpose_input_node = self._find_node_by_output_name(transpose.input[0])

            transpose_consumers = self._find_nodes_by_input_name(transpose.output[0])

            for transpose_consumer in transpose_consumers:
                index = None
                for i, input_name in enumerate(transpose_consumer.input):
                    if input_name == transpose.output[0]:
                        index = i
                        break
                transpose_consumer.input[index] = transpose_input_node.output[0]
            self.nodes.remove(transpose)

        def _swap_last_dims(node: onnx.ValueInfoProto) -> None:
            """Since the past key cache is transposed, swap the dims of the corresponding input and output nodes"""
            dim = node.type.tensor_type.shape.dim
            last_dim = dim[-1].dim_value
            dim[-1].dim_value = dim[-2].dim_value
            dim[-2].dim_value = last_dim

        def _get_downstream_transpose(start_node: onnx.NodeProto):
            """Get the next `Transpose` node in the topological ordering of nodes"""

            if start_node.op_type == "Transpose":
                return start_node

            queue = []
            visited = set()
            transpose_node = None

            queue.extend(start_node.output)

            while queue and not transpose_node:
                curr_output = queue.pop(0)
                if curr_output not in visited:
                    visited.add(curr_output)

                    downstream_nodes = self._find_nodes_by_input_name(curr_output)

                    for node in downstream_nodes:
                        if node.op_type == "Transpose":
                            transpose_node = node
                            break
                        else:
                            queue.extend(node.output)

            return transpose_node

        log_info(
            f"Found {len(self._non_transposed_key_concat_nodes)} `Concat` nodes that concat new key tensor with past key cache"
        )
        log_info(
            "Modifying the graph so that the new key tensor is transposed before concatenating with past key cache."
        )
        log_info("The resulting model outputs transposed key cache\n")

        for concat_node in self._non_transposed_key_concat_nodes:
            other_input = [_inp for _inp in concat_node.input if _inp not in self._all_inputs]
            past_key_input = [_inp for _inp in concat_node.input if _inp in self._all_inputs]

            transpose = _get_downstream_transpose(concat_node)

            if transpose:
                perm_attr = [attr for attr in transpose.attribute if attr.name == "perm"][0]
                new_transpose = onnx.helper.make_node(
                    transpose.op_type,
                    inputs=other_input,
                    outputs=transpose.output,
                    name=transpose.name,
                    perm=perm_attr.ints,
                )

                _remove_transpose(transpose)

                # Modify the inputs
                other_input_index = list(concat_node.input).index(other_input[0])
                concat_node.input[other_input_index] = new_transpose.output[0]

                # Add the new transpose before the concat op
                self.nodes.insert(list(self.nodes).index(concat_node), new_transpose)

                # Update attributes of the concat node to concat along the last dim
                axis_attr = [attr for attr in concat_node.attribute if attr.name == "axis"]
                if axis_attr:
                    axis_attr[0].i = -1

                # Modify the dims of past_key_tensor
                past_key_node = [
                    input_tensor
                    for input_tensor in self.model.graph.input
                    if input_tensor.name == past_key_input[0]
                ][0]
                _swap_last_dims(past_key_node)

                # Modify the dims of past_key_output
                past_key_output_node = [
                    output_tensor
                    for output_tensor in self.model.graph.output
                    if output_tensor.name == concat_node.output[0]
                ][0]
                _swap_last_dims(past_key_output_node)

            else:
                log_warning(f"Could not find downstream transpose for {concat_node.name}. Skipping")

    def _output_new_key_value_only(self):
        """
        * In the original model, the entire key cache is returned as a model output
        * But this is redundant, as the only difference is the new key
        * This adaptation modifies the inputs/outputs such that only the new key is returned
          instead of the key tensor state with all past key tensor concatenated with new key
          to optimize memory utilization
        """

        def _get_key_and_transpose_input(concat_node: onnx.NodeProto):
            """
            Return:
              i. key input - input tensor to the graph
              ii. transpose input - Assuming transpose_kvcache is run. Else, the other input to concat
            """
            for input_tensor in self.model.graph.input:
                input_1 = None
                input_2 = None
                if concat_node.input[0] == input_tensor.name:
                    input_1 = input_tensor
                    input_2 = self._find_node_by_output_name(concat_node.input[1])
                elif concat_node.input[1] == input_tensor.name:
                    input_2 = self._find_node_by_output_name(concat_node.input[0])
                    input_1 = input_tensor
                if input_1 and input_2:
                    return input_1, input_2

            return None, None

        log_debug(
            f"Number of past_key and past_value nodes: {len(self._key_concat_nodes + self._value_concat_nodes)}"
        )

        modified = False

        key_concat_names = [node.name for node in self._key_concat_nodes]

        for concat in self._key_concat_nodes + self._value_concat_nodes:
            kv_input, other_input = _get_key_and_transpose_input(concat)

            # This is the output name of the concat node
            # Which is also the output name of the graph(example - past_key_0_out)
            # This should instead be the output name of the new key input node
            # Cache the name and replace it later

            if kv_input is None or other_input is None:
                continue

            if (
                other_input.output[0] not in self._all_outputs
            ):  # If true, the optimization has already been done
                if not modified:
                    log_info(
                        "Modifying past_key and past_value output tensors to only output new key value \n"
                    )

                concat_output_name = concat.output[0]

                # Modify the input name of the concat node that consumes transpose(or the other input)
                # Do this before modifying the output name of the transpose
                # Cache the index here because concat inputs will be modified
                # But we still need the index to modify the right input to concat
                transpose_input_index = list(concat.input).index(other_input.output[0])

                # The new output will be the output of the transpose node prior to concat
                other_input.output[0] = concat_output_name

                # Modify the name of the concat output
                new_concat_output = concat.name + "_output_0"
                concat.output[0] = new_concat_output

                # Modify the output name of all the consumers of the concat node
                # to reflect the change in name of the concat node
                for consumer_node in self._find_nodes_by_input_name(concat_output_name):
                    concat_input_index = 0
                    for i, input_tensor in enumerate(consumer_node.input):
                        if input_tensor == concat_output_name:
                            concat_input_index = i
                            break
                    consumer_node.input[concat_input_index] = new_concat_output

                concat.input[transpose_input_index] = concat_output_name

                # Modify the dimensions
                kv_output = [
                    output_tensor
                    for output_tensor in self.model.graph.output
                    if output_tensor.name == concat_output_name
                ]
                if kv_output:
                    kv_output = kv_output[0]

                    if self.transpose_keycache and concat.name in key_concat_names:
                        index_to_modify = -1
                    else:
                        index_to_modify = -2

                    output_dim_value = kv_output.type.tensor_type.shape.dim[index_to_modify].dim_value
                    input_dim_value = kv_input.type.tensor_type.shape.dim[index_to_modify].dim_value

                    kv_output.type.tensor_type.shape.dim[index_to_modify].dim_value = (
                        output_dim_value - input_dim_value
                    )

                    modified = True

        if not modified:
            log_warning("'Output new key value only' optimization not run")
            log_warning("All key/value outputs are new key/values only\n")

    def run(self):
        log_debug("Key concat nodes:\n", self._key_concat_nodes)
        log_debug("Value concat nodes:\n", self._value_concat_nodes)
        if self.transpose_keycache:
            if not self._non_transposed_key_concat_nodes:
                log_warning(
                    'No non-transposed key concat nodes found. Skipping "transpose keycache" adaptation \n'
                )
            else:
                self._transposed_key_cache()
        if self.output_new_key_value_only:
            if not self._key_concat_nodes and not self._value_concat_nodes:
                log_warning(
                    'No key and value concat nodes found. Skipping "output only new key value" adaptation \n'
                )
            else:
                self._output_new_key_value_only()
