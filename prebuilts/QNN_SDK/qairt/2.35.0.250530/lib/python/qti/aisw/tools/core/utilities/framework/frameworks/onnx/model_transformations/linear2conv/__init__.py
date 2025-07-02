# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# Not a contribution.
#
# -----------------------------------------------------------------------------
"""Adaptation to replace Gemm and MatMul with Conv"""

import onnx
from tqdm import tqdm

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.logger import (
    log_error,
    log_info,
    log_warning,
    setup_logging,
)


class LinearToConv:
    def __init__(self, model: onnx.ModelProto, log_level: str = "info") -> None:
        self.model = model
        self.nodes = model.graph.node
        self.initializers = model.graph.initializer

        # TODO: Common utils
        self.init_map = {init.name: init for init in model.graph.initializer}
        self.input_map = {inp.name: inp for inp in model.graph.input}
        self.output_map = {out.name: out for out in model.graph.output}
        self.val_map = {val.name: val for val in model.graph.value_info}

        self.tensor_dim_map = {}
        for key in ["value_info", "input", "output"]:
            self.tensor_dim_map.update(
                {
                    val.name: [d.dim_value for d in val.type.tensor_type.shape.dim]
                    for val in getattr(model.graph, key)
                }
            )

        setup_logging(log_level, "LinearToConv")

    def _is_graph_output(self, node):
        """Returns True if `name` is a graph output name"""
        for out in node.output:
            if out in self.output_map:
                return True
        return False

    def _get_transA_transB_attr(self, node: onnx.NodeProto):
        """Return the attribute values of transA and transB (if set)"""
        try:
            transA = bool(onnx.helper.get_node_attr_value(node, "transA"))
        except ValueError:
            transA = False

        # If transB is set, weights are NOT transposed
        try:
            transB = bool(onnx.helper.get_node_attr_value(node, "transB"))
        except ValueError:
            transB = False

        return transA, transB

    def replace(self) -> bool:
        """Replace all Linear nodes with an equivalent Conv node"""

        log_info("Replacing MatMul/Gemm linears with Convolutions...")

        # Linear node if
        # 1. op_type is MatMul or Gemm AND
        # 2. Atleast one of the inputs is an initializer
        linear_nodes = [
            node
            for node in self.nodes
            if node.op_type in ["MatMul", "Gemm"] and node.input[1] in self.init_map
        ]

        if len(linear_nodes) == 0:
            log_warning("No MatMul/Gemm linear nodes found! Skipping...")
            return True

        log_info(
            f"Found {len(linear_nodes)} linear nodes to be replaced with Convolutions..."
        )

        for node in tqdm(linear_nodes):
            is_gemm = node.op_type == "Gemm"

            # If transA is set, first transpose is NOT added
            # If transB is set, weights are NOT transposed
            transA, transB = self._get_transA_transB_attr(node)

            # Preserve this for encodings
            linear_output_name = node.output[0]

            # Cache the index where the node should be inserted
            insert_index = list(self.nodes).index(node)

            try:
                input_shapes = self.tensor_dim_map[node.input[0]]
                output_shapes = self.tensor_dim_map[node.output[0]]
            except KeyError:
                log_error(
                    f"""Unable to determine input and/or output tensor shapes of node: {node.name}. Skipping..."""
                )
                continue

            weight = self.init_map[node.input[1]]

            in_features = input_shapes[-2]
            k_channels = weight.dims[0]
            out_features = weight.dims[1]

            weight_array = onnx.numpy_helper.to_array(weight)
            if transB:
                k_channels, out_features = out_features, k_channels
            else:
                weight_array = weight_array.T

            node_list = []
            init_list = []

            # Maintain the prev node input after creating each node
            prev_node_out = node.input[0]

            # 1. Reshape 1
            # *******************************************************************************************************
            reshape_1_init = onnx.helper.make_tensor(
                name=f"{node.name}_Reshape_1_Constant",
                vals=[-1, in_features, 1, k_channels],
                data_type=onnx.TensorProto.INT64,
                dims=[4],
            )
            init_list.append(reshape_1_init)

            reshape_1_node = onnx.helper.make_node(
                "Reshape",
                name=f"{node.name}_Reshape_1",
                inputs=[prev_node_out, reshape_1_init.name],
                outputs=[f"{node.name}_Reshape_1_output"],
            )
            node_list.append(reshape_1_node)
            prev_node_out = reshape_1_node.output[0]

            # 2. Transpose 1
            # *******************************************************************************************************
            # Transpose input only if transA is not set
            if not transA:
                transpose_1_node = onnx.helper.make_node(
                    "Transpose",
                    name=f"{node.name}_Transpose_1",
                    inputs=[prev_node_out],
                    outputs=[f"{node.name}_Transpose_1_output"],
                    perm=[0, 3, 2, 1],
                )
                node_list.append(transpose_1_node)
                prev_node_out = transpose_1_node.output[0]

            # 3. Conv
            # *******************************************************************************************************
            weight_init = onnx.numpy_helper.from_array(
                weight_array[..., None, None], name=weight.name
            )
            init_list.append(weight_init)

            conv_inputs = [prev_node_out, weight_init.name]
            if is_gemm:
                # Add the bias input as-is
                conv_inputs.append(node.input[2])

            conv_node = onnx.helper.make_node(
                "Conv",
                name=f"{node.name}_Conv",
                inputs=conv_inputs,
                outputs=[linear_output_name],
                kernel_shape=[1, 1],
                pads=[0, 0, 0, 0],
                strides=[1, 1],
                dilations=[1, 1],
            )
            node_list.append(conv_node)
            prev_node_out = conv_node.output[0]

            # TODO: Automate this
            try:
                val_info = self.val_map[linear_output_name]
                self.model.graph.value_info.remove(val_info)
                linear_val_info = onnx.helper.make_tensor_value_info(
                    linear_output_name,
                    val_info.type.tensor_type.elem_type,
                    [1, out_features, 1, in_features],
                )
                self.model.graph.value_info.append(linear_val_info)
            except (KeyError, ValueError, AttributeError):
                pass

            # Add Transpose->Reshape only if there are no downstream Liners with no reshaping ops in-between
            # 4. Transpose 2
            # *******************************************************************************************************
            transpose_2_node = onnx.helper.make_node(
                "Transpose",
                name=f"{node.name}_Transpose_2",
                inputs=[prev_node_out],
                outputs=[f"{node.name}_Transpose_2_out"],
                perm=[0, 3, 2, 1],
            )
            node_list.append(transpose_2_node)
            prev_node_out = transpose_2_node.output[0]

            # 5. Reshape 2
            # *******************************************************************************************************
            # Insert a reshape op if output shape is not 4
            if len(output_shapes) != 4:
                if len(output_shapes) == 3:
                    reshape_2_shape = [-1, in_features, out_features]
                else:
                    reshape_2_shape = [in_features, out_features]

                reshape_2_init = onnx.helper.make_tensor(
                    name=f"{node.name}_Reshape_2_Constant",
                    vals=reshape_2_shape,
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(reshape_2_shape)],
                )
                init_list.append(reshape_2_init)

                reshape_2_node = onnx.helper.make_node(
                    "Reshape",
                    inputs=[prev_node_out, reshape_2_init.name],
                    outputs=[f"{node.name}_Reshape_2_output"],
                )
                node_list.append(reshape_2_node)
                prev_node_out = reshape_2_node.output[0]

            # Update the output name if node is the output node
            if self._is_graph_output(node):
                self.output_map[linear_output_name].name = prev_node_out

            # Update consumer inputs if Transpose->Reshape is added after Conv
            elif prev_node_out != node.input[0]:
                # TODO: Common utils
                # Update input names of all consumers
                for consumer in self.nodes:
                    for i, _inp in enumerate(consumer.input):
                        if _inp == linear_output_name:
                            consumer.input[i] = prev_node_out
                            break

            # Defer graph updates to the end of the function

            # First, remove the linear node and weight from the graph
            self.initializers.remove(weight)
            self.nodes.remove(node)

            # Insert at the index of current linear node, in reverse orderto
            # to maintain topological sorting
            self.initializers.extend(init_list)
            for _node in node_list[::-1]:
                self.nodes.insert(insert_index, _node)

        log_info(f"Replaced {len(linear_nodes)} linear nodes with Convolutions\n")
        return True
