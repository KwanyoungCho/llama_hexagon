# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# Not a contribution.
#
# -----------------------------------------------------------------------------

from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
import onnx
from onnx import helper, numpy_helper

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.onnx import (
    NodeNotFoundError,
    get_next_node_up_based_on_cond,
    get_next_node_down_based_on_cond,
    get_node_by_input_name,
    get_node_by_output_name,
    get_node_input_constant_op_value,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.logger import (
    log_debug,
    log_error,
    log_warning,
    log_info,
    setup_logging,
)


class UnpackQKV:
    """
    Pattern 1:
             Input Tensor [T x N]
                     |
                     V
            +-----------------------------------+
            | Linear [N x 3K] (Matmul or Gemm)  | (Packed)
            +-----------------------------------+
                     |
                     V
                  .......

                     |
                     |
                     V
            +------------------+
        ----|     Split (3)    |----
        |   +------------------+   |
        |            |             |
        V            V             V
    +-------+    +-------+     +-------+
    | Node1 |    | Node2 |     | Node3 |
    +-------+    +-------+     +-------+


    Pattern 2:
                   Input Tensor [T x N]
                         |
                         V
                +-----------------------------------+
                | Linear [N x 3K] (Matmul or Gemm)  |
                +-----------------------------------+
                         |
                         V
                      .......
                         |
                         V
                +------------------------+
            ----|   Node (any op type)   |----
            |   +------------------------+  |
            |             |                 |
            V             V                 V
    +-----------+   +-----------+   +-----------+
    | Gather(0) |   | Gather(1) |   | Gather(2) |
    +-----------+   +-----------+   +-----------+
          |               |              |
          V               V              V
      +-------+       +-------+      +-------+
      | Node1 |       | Node2 |      | Node3 |
      +-------+       +-------+      +-------+

    Result:
                      Input Tensor [T x N]
                              |
                              |
            ------------------------------------
            |                 |                |
            V                 V                V
    +-------------+   +-------------+   +-------------+
    | Linear[NxK] |   | Linear[NxK] |   | Linear[NxK] |
    +-------------+   +-------------+   +-------------+
          |                  |                 |
          V                  V                 V
      +-------+          +-------+         +-------+
      | Node1 |          | Node2 |         | Node3 |
      +-------+          +-------+         +-------+

    """

    def __init__(self, model: onnx.ModelProto, log_level: str = "info"):
        self.model = model
        self.nodes = model.graph.node
        self.initializers = model.graph.initializer
        self.val_info = model.graph.value_info
        setup_logging(log_level)
        self.populate_mappings()

    def populate_mappings(self) -> None:
        """
        Helper function to populate the graph parsing mappings.
        """
        self.producer_map = {
            output: node for node in self.nodes for output in node.output
        }
        self.init_map = {init.name: init for init in self.initializers}
        self.val_info_map = {val.name: val for val in self.val_info}
        self.get_node_by_input_name = get_node_by_input_name(self.model)
        self.get_node_by_output_name = get_node_by_output_name(self.model)

    def _get_node_constant(self, input: str) -> Optional[np.ndarray]:
        """
        Return constant or initializer 1D array of given input or
        return array(-1) if input is not a constant
        """
        if input in self.init_map:
            arr = numpy_helper.to_array(self.init_map[input])
            return np.atleast_1d(arr)
        elif self.producer_map[input].op_type == "Constant":
            arr = numpy_helper.to_array(self.producer_map[input].attribute[0].t)
            return np.atleast_1d(arr)
        else:
            return None

    def _find_split_by_3(self) -> List[onnx.NodeProto]:
        """Find all split nodes that split a tensor into three equal sized tensors"""

        found_splits = [
            node
            for node in self.nodes
            if (node.op_type == "Split")
            and len(node.output) == 3
            # Ensure all splits values are constant input
            and self._get_node_constant(node.input[1]) is not None
            # All splits have the same size
            and len(set(self._get_node_constant(node.input[1]))) == 1
        ]

        return found_splits

    def _find_three_gather_pattern(
        self,
    ) -> List[Tuple[onnx.NodeProto, List[onnx.NodeProto]]]:
        """
        Find all patterns where the output of a node is consumed by three gathers
        and the gathers have indices equal to 0,1,2 respectively
        """

        gather_map = defaultdict(list)

        # find all nodes that have the output consumed by 3 Gathers
        all_gathers = [node for node in self.nodes if node.op_type == "Gather"]
        for gather in all_gathers:
            producer = self.producer_map.get(gather.input[0])
            if producer:
                gather_map[producer.output[0]].append(gather)

        gather_items = list()
        # Filter the map such that the gathers have indices 0, 1, 2
        for node_output, gathers in gather_map.items():
            gather_set = set()
            for g in gathers:
                index_input = self._get_node_constant(g.input[1])
                if index_input is None:
                    # skip if index input is not a constant
                    continue
                # only consider gather nodes with 1D index input
                if len(index_input.shape) == 1:
                    gather_set.add(index_input[0])
            if gather_set == {0, 1, 2}:
                # Sort by gather index -> q,k,v
                # Append a tuple of format (producer_node, list(gather nodes))
                gather_items.append(
                    (
                        self.producer_map[node_output],
                        sorted(gathers, key=lambda g: index_input[0]),
                    )
                )
        return gather_items

    def _get_unpacked_linears(
        self, packed_linear: onnx.NodeProto, unpacked_output_names: List[str]
    ) -> List[onnx.NodeProto]:
        """
        :param packed_linear: Linear node to be unpacked into q,k and v projections
        :unpacked_output_names: Output name to assign to each unpacked linear
        """

        is_gemm = packed_linear.op_type == "Gemm"

        weight = self.init_map[packed_linear.input[1]]
        if weight.dims[1] % 3 != 0:
            log_error(f"packed QKV dim {weight.dims[1]} cannot be divided by 3")
            return []

        # Split the weight (and bias if Gemm) into three
        weight_array = numpy_helper.to_array(weight)
        split_weight = np.split(weight_array, 3, 1)

        if is_gemm:
            bias = self.init_map[packed_linear.input[2]]
            bias_array = numpy_helper.to_array(bias)
            split_bias = np.split(bias_array, 3)

        unpacked_linears = []

        # Construct linears for unpacked q,k and v projections
        for i, name in enumerate(["_q_proj", "_k_proj", "_v_proj"]):
            weight_name = packed_linear.input[1] + name
            if weight_name not in self.init_map:
                weight_init = helper.make_tensor(
                    name=weight_name,
                    data_type=weight.data_type,
                    dims=split_weight[i].shape,
                    vals=split_weight[i],
                )
                self.initializers.append(weight_init)
                self.init_map[weight_name] = weight_init

            node_inputs = [packed_linear.input[0], weight_name]

            if is_gemm:
                bias_name = packed_linear.input[2] + name
                if bias_name not in self.init_map:
                    bias_init = helper.make_tensor(
                        name=bias_name,
                        data_type=bias.data_type,
                        dims=split_bias[i].shape,
                        vals=split_bias[i],
                    )
                    self.initializers.append(bias_init)
                    self.init_map[bias_name] = bias_init

                node_inputs.append(bias_name)

            unpacked_linear = helper.make_node(
                op_type=packed_linear.op_type,
                inputs=node_inputs,
                outputs=[unpacked_output_names[i]],
                name=packed_linear.name + name,
            )
            unpacked_linears.append(unpacked_linear)

        return unpacked_linears

    def _get_all_nodes_between(
        self, start: onnx.NodeProto, end: onnx.NodeProto
    ) -> List[onnx.NodeProto]:
        """Get all nodes between `start` and `end` (inclusive)"""

        start_index = list(self.nodes).index(start)
        end_index = list(self.nodes).index(end) + 1

        return self.nodes[start_index:end_index]

    def unpack(self):
        """Unpack QKV Linear into q,k and v projections"""
        gather_map = self._find_three_gather_pattern()
        found_splits = self._find_split_by_3()

        if not gather_map and not found_splits:
            log_warning(
                "Could not find any packed QKV patterns, returning original model."
            )
            return self.model, False

        log_info(
            f"Found {len(gather_map) + len(found_splits)} packed QKV projections. Replacing with unpacked projections."
        )

        # Initialize nodes_to_remove with all gathers
        nodes_to_remove = []
        for _, gathers in gather_map:
            nodes_to_remove.extend(gathers)

        unpack_gather = [
            (node, [g.output[0] for g in gathers]) for node, gathers in gather_map
        ]

        unpack_split = [(split, list(split.output)) for split in found_splits]

        for key_node, output_names in unpack_gather + unpack_split:
            try:
                packed_linear = get_next_node_up_based_on_cond(
                    key_node,
                    get_node_by_output_name(self.model),
                    node_found_cond=lambda n: (
                        n.op_type == "MatMul" or n.op_type == "Gemm"
                    ),
                    # packed Conv is currently unsupported so end search if Conv is encountered
                    # TODO add support for packed Conv and remove below
                    node_end_search_cond=lambda n: n.op_type == "Conv",
                )
            except NodeNotFoundError:
                log_info(
                    "Packed MatMul or Gemm node not found for corresponding split/gather nodes,"
                    " checking next potential match."
                )
                continue

            new_reshape_shape = None
            reshape_after_gather_or_split = None
            if packed_linear and packed_linear.op_type == "MatMul":
                try:
                    # Look for reshape node after gather or split node that splits C dims
                    # into num_heads and head_dims
                    reshape_after_gather_or_split = get_next_node_down_based_on_cond(
                        self.get_node_by_output_name[output_names[0]],
                        self.get_node_by_input_name,
                        node_found_cond=lambda n: n.op_type == "Reshape",
                        node_end_search_cond=lambda n: n.op_type == "MatMul",
                    )
                except NodeNotFoundError:
                    log_debug(
                        "No reshape found after matched gather/split node. So skipping "
                        "addition of reshape & transpose nodes after unpacked QKV to handle "
                        "(num_heads*head_dims) -> (num_heads, head_dims) split"
                    )

                if reshape_after_gather_or_split is None:
                    # If there is a reshape after gather or split then the shapes after unpacked QKV
                    # will be taken care by it so skip below addition of reshape & transpose nodes after unpacked QKV
                    try:
                        # Look for reshape node after packed linear that splits C dims into num_heads and head_dims
                        # between packed QKV and split or gather
                        reshape_node = get_next_node_down_based_on_cond(
                            packed_linear,
                            self.get_node_by_input_name,
                            node_found_cond=lambda n: n.op_type == "Reshape",
                            node_end_search_cond=lambda n: n
                            == self.get_node_by_output_name[output_names[0]],
                        )
                        reshape_input_shape = get_node_input_constant_op_value(
                            reshape_node, self.get_node_by_output_name, self.init_map
                        )
                        if val_info := self.val_info_map.get(packed_linear.output[0]):
                            packed_linear_shape = [
                                d.dim_value for d in val_info.type.tensor_type.shape.dim
                            ]

                            # match [B, seq_len, ...] of packed linear output with [B, seq_len, ...] reshape shape input
                            # if it matches then reshape is reshaping C dim as [B, seq_len , num_heads, head_dim]
                            if (
                                packed_linear_shape[:2] == reshape_input_shape[:2]
                            ).all():
                                # create new reshape shape input as [B, seq_len, num_heads, head_dim]
                                new_reshape_shape = np.append(
                                    packed_linear_shape[:2], reshape_input_shape[-2:]
                                )
                    except NodeNotFoundError:
                        log_debug(
                            "No reshape found between packed QKV and QK matmul. So skipping "
                            "addition of reshape & transpose nodes after unpacked QKV to handle "
                            "(num_heads*head_dims) -> (num_heads, head_dims) split"
                        )

            if packed_linear:
                # Remove all nodes between packed_linear and parent node of 3 gather (inclusive of both)
                nodes_to_remove += self._get_all_nodes_between(packed_linear, key_node)

                # Unpacked linears will have the output name equal to the outputs of split or gathers
                # This way, the consumers node inputs will be the output of unpacked linears
                unpacked_linears = self._get_unpacked_linears(
                    packed_linear, output_names
                )

                insert_index = list(self.nodes).index(packed_linear)

                if val_info := self.val_info_map.get(packed_linear.output[0]):
                    # Get unpacked linear output shape
                    unpacked_linear_shape = [
                        d.dim_value for d in val_info.type.tensor_type.shape.dim
                    ]
                    unpacked_linear_shape[-1] = unpacked_linear_shape[-1] // 3
                    for unpacked_linear in unpacked_linears:
                        gather_or_split_output_shape = [
                            d.dim_value
                            for d in self.val_info_map.get(
                                unpacked_linear.output[0]
                            ).type.tensor_type.shape.dim
                        ]
                        # if shape of new reshape node's shape does not match unpacked linear output
                        # then add reshape after unpacked linear
                        if new_reshape_shape is not None and not np.array_equal(
                            unpacked_linear_shape, new_reshape_shape
                        ):
                            reshape_init = onnx.helper.make_tensor(
                                name=f"{unpacked_linear.name}_Reshape_Constant",
                                vals=new_reshape_shape,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(new_reshape_shape)],
                            )
                            self.initializers.append(reshape_init)
                            original_unpack_linear_output = unpacked_linear.output[0]
                            unpacked_linear.output[0] = f"{unpacked_linear.name}_output"
                            reshape_output_name = f"{unpacked_linear.name}_Reshape"

                            reshape_node = onnx.helper.make_node(
                                "Reshape",
                                inputs=[unpacked_linear.output[0], reshape_init.name],
                                outputs=[reshape_output_name],
                                name=f"{unpacked_linear.name}_Reshape",
                            )
                            # insert reshape after unpacked linear inserted at insert_index
                            self.nodes.insert(insert_index + 1, reshape_node)

                            # if shape of gather/split output does not match the new reshape node's shape
                            # then add transpose [B, seq_len, num_heads, head_dim] -> [B, num_heads, seq_len, head_dim]
                            if not np.array_equal(
                                gather_or_split_output_shape, new_reshape_shape
                            ):
                                transpose_node = onnx.helper.make_node(
                                    "Transpose",
                                    inputs=[reshape_output_name],
                                    outputs=[original_unpack_linear_output],
                                    name=f"{unpacked_linear.name}_Transpose",
                                    perm=[0, 2, 1, 3],
                                )
                                # insert transpose after unpacked linear inserted at insert_index
                                self.nodes.insert(insert_index + 2, transpose_node)

                        # Update the unpacked linear output shapes to reflect the new shape
                        unpacked_linear_val_info = helper.make_tensor_value_info(
                            unpacked_linear.output[0],
                            val_info.type.tensor_type.elem_type,
                            shape=unpacked_linear_shape,
                        )
                        self.val_info.append(unpacked_linear_val_info)

                    # Remove val info of the old linear
                    self.val_info.remove(val_info)

                # Insert the unpacked linears in reverse order at the original index
                # of the packed linear to maintain topological sorting
                for unpacked_linear in unpacked_linears[::-1]:
                    self.nodes.insert(insert_index, unpacked_linear)

        for node in nodes_to_remove:
            try:
                self.nodes.remove(node)
            except Exception:
                continue

        return self.model, True
