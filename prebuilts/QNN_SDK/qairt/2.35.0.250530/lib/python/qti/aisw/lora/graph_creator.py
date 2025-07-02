# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import onnx
from qti.aisw.lora.preprocessing import *
import numpy as np
from qti.aisw.converters.common.utils.converter_utils import *


class MaxRankAttachPoint:
    def __init__(self, op):
        self.op = op
        self.lora_tensor_names = LoraTensorNames()
        self.lora_node_names = LoraNodeNames()
        self.alpha_separation_gather_indices_tensor = None
        self.alpha_separation_gather_indices_tensor_name = None
        self.lora_a_weight_shape = None
        self.lora_b_weight_shape = None


#  Class to create Max rank graph
class MaxRankGraphCreator:
    def __init__(self, base_graph_path, attach_point_info_map, base_indices_map, alpha_vector_size, **kargs):
        self.attach_point_info_map = attach_point_info_map
        self.alpha_vector_size = alpha_vector_size
        self.base_graph_path = base_graph_path
        self.base_indices_map = base_indices_map.attach_pt_indices
        self.model = None # Onnx ModelProto
        self.graph = None # Onnx GraphProto
        self.attach_pts = {}
        self.alpha_input_tensor = None
        self.consumers_map = {}
        self.nodes_map = {}


    def load_graph(self):
        self.model = onnx.load(self.base_graph_path)
        self.graph = self.model.graph

        for node in self.graph.node:
            self.nodes_map[node.name] = node
            inputs = node.input
            for inp in inputs:
                if inp in self.consumers_map:
                    self.consumers_map[inp].append(node)
                else:
                    self.consumers_map[inp] = [node]


    def _get_node_name(self, base_node_name, op_type, suffix=None):
        if suffix is None:
            return f"{base_node_name}_{op_type}"
        else:
            return f"{base_node_name}_{op_type}_{suffix}"


    def _get_output_tensor_name(self, base_node_name, op_type):
        return f"{base_node_name}_{op_type}_output"


    def _get_lora_tensor_names(self, node, op_type):
        return LoraTensorNames(
            lora_a_weight_name=self._get_node_name(f"{node.name}_loraA", op_type, ".weight"),
            lora_a_act=self._get_output_tensor_name(f"{node.name}_loraA", op_type),
            mul_scale="",
            mul=self._get_output_tensor_name(f"{node.name}_lora", 'Mul'),
            lora_b_weight_name=self._get_node_name(f"{node.name}_loraB", op_type, ".weight"),
            lora_b_act=self._get_output_tensor_name(f"{node.name}_loraB", op_type),
            add=self._get_output_tensor_name(f"{node.name}_lora", 'Add')
        )


    def _get_lora_node_names(self, node, op_type):
        return LoraNodeNames(
            lora_a=self._get_node_name(node.name, f"loraA_{op_type}"),
            lora_b=self._get_node_name(node.name, f"loraB_{op_type}"),
            mul=self._get_node_name(node.name, "lora_Mul"),
            add=self._get_node_name(node.name, "lora_Add")
        )


    def _validate_attach_point_op(self, node):
        if node.op_type not in ['Conv', 'MatMul']:
            raise ValueError(f"Unsupported op_type: {node.op_type}.  Only Conv and MatMul are supported as lora attach-points.")


    def find_attach_pt_ops(self):
        """
        Traverse the graph and store attach-point ops.
        Inputs:
        - graph
        - Map of {attach-point-name, AttachPointInfo}
        Outputs:
        - Updated {attach-point-name, MaxRankAttachPoint} map
        """
        for attach_point_name, attach_point_info in self.attach_point_info_map.items():
            node = self.nodes_map.get(attach_point_name)  # Use the node map
            if node:
                self._validate_attach_point_op(node)
                max_rank_attach_point = MaxRankAttachPoint(node)
                op_name = node.op_type
                max_rank_attach_point.lora_tensor_names = self._get_lora_tensor_names(node, op_name)
                max_rank_attach_point.lora_node_names = self._get_lora_node_names(node, op_name)
                max_rank_attach_point.alpha_separation_gather_indices_tensor_name = f"{attach_point_name}_indices"
                self.attach_pts[attach_point_name] = max_rank_attach_point
            else:
                # Throw error if the attach point is not present
                raise ValueError("Invalid Base Graph: Unable to find the attach point {} in the base graph."
                                 .format(attach_point_name))


    def _create_lora_a_conv(self, max_rank_attach_point, attach_point_info, input_tensor, output_tensor, weight_shape_map):
        loraA_weight_name = max_rank_attach_point.lora_tensor_names.lora_a_weight_name
        base_layer_weight_shape = weight_shape_map[max_rank_attach_point.op.input[1]]
        input_channel = base_layer_weight_shape[1]
        loraA_weight_array = np.zeros((attach_point_info.max_rank, input_channel, base_layer_weight_shape[2], base_layer_weight_shape[3])).astype(np.float32)
        max_rank_attach_point.lora_a_weight_shape = loraA_weight_array.shape
        loraA_weight = onnx.helper.make_tensor(loraA_weight_name, onnx.TensorProto.FLOAT, loraA_weight_array.shape, loraA_weight_array)
        self.graph.initializer.extend([loraA_weight])
        lora_A_conv_node_name = max_rank_attach_point.lora_node_names.lora_a
        lora_A_conv_op = onnx.helper.make_node('Conv',
                                            [input_tensor, loraA_weight_name],
                                            [max_rank_attach_point.lora_tensor_names.lora_a_act],
                                            name=lora_A_conv_node_name)
        for attr in max_rank_attach_point.op.attribute:
            copy_attr = copy.deepcopy(attr)
            lora_A_conv_op.attribute.append(copy_attr)
        return max_rank_attach_point.lora_tensor_names.lora_a_act, lora_A_conv_op


    def _create_mul_node(self, lora_A_conv_node_output, max_rank_attach_point):
        mul_name = max_rank_attach_point.lora_node_names.mul
        mul_output = max_rank_attach_point.lora_tensor_names.mul
        mul = onnx.helper.make_node('Mul',
                                    [lora_A_conv_node_output],
                                    [mul_output],
                                    name=mul_name)
        return mul_output, mul


    def _create_lora_b_conv(self, max_rank_attach_point, attach_point_info, mul_name_output, output_tensor, weight_shape_map):
        loraB_weight_name = max_rank_attach_point.lora_tensor_names.lora_b_weight_name
        base_layer_weight_shape = weight_shape_map[max_rank_attach_point.op.input[1]]
        output_channel = base_layer_weight_shape[0]
        loraB_weight_array = np.zeros((output_channel, attach_point_info.max_rank, base_layer_weight_shape[2], base_layer_weight_shape[3])).astype(np.float32)
        max_rank_attach_point.lora_b_weight_shape = loraB_weight_array.shape
        loraB_weight = onnx.helper.make_tensor(loraB_weight_name, onnx.TensorProto.FLOAT, loraB_weight_array.shape, loraB_weight_array)
        self.graph.initializer.extend([loraB_weight])
        lora_Bconv_node_name = max_rank_attach_point.lora_node_names.lora_b
        lora_B_conv_op = onnx.helper.make_node('Conv',
                                            [mul_name_output, loraB_weight_name],
                                            [max_rank_attach_point.lora_tensor_names.lora_b_act],
                                            name=lora_Bconv_node_name)
        for attr in max_rank_attach_point.op.attribute:
            copy_attr = copy.deepcopy(attr)
            lora_B_conv_op.attribute.append(copy_attr)
        return max_rank_attach_point.lora_tensor_names.lora_b_act, lora_B_conv_op


    def _create_add_node(self, output_tensor, lora_Bconv_node_name_output, max_rank_attach_point):
        add_node_name = max_rank_attach_point.lora_node_names.add
        add_output = max_rank_attach_point.lora_tensor_names.add
        add = onnx.helper.make_node('Add',
                                    [output_tensor, lora_Bconv_node_name_output],
                                    [add_output],  # Update the output to point to the Add node
                                    name=add_node_name)
        return add


    def add_max_rank_lora_branch(self):
        """
        Add Lora branches with max-rank for each attach-point.
        Inputs:
        - graph
        - Map of {attach-point-name, AttachPointInfo}
        Outputs:
        - Updated {attach-point-name, MaxRankAttachPoint} map
        - Updated graph
        """
        weight_shape_map = {}
        for weight in self.graph.initializer:
            weight_shape_map[weight.name] = weight.dims
        for attach_point_name, attach_point_info in self.attach_point_info_map.items():
            max_rank_attach_point = self.attach_pts[attach_point_name]
            if max_rank_attach_point.op.op_type == 'Conv':
                input_tensor = max_rank_attach_point.op.input[0]
                output_tensor = max_rank_attach_point.op.output[0]
                base_layer_weight_shape = weight_shape_map[max_rank_attach_point.op.input[1]]
                input_channel = base_layer_weight_shape[1]
                output_channel = base_layer_weight_shape[0]

                lora_A_conv_node_output, lora_A_conv_op = self._create_lora_a_conv(max_rank_attach_point, attach_point_info, input_tensor, output_tensor, weight_shape_map)
                mul_node_output, mul_op = self._create_mul_node(lora_A_conv_node_output, max_rank_attach_point)
                lora_Bconv_node_name_output, lora_B_conv_op = self._create_lora_b_conv(max_rank_attach_point, attach_point_info, mul_node_output, output_tensor, weight_shape_map)
                add_op = self._create_add_node(output_tensor, lora_Bconv_node_name_output, max_rank_attach_point)

                new_nodes = [lora_A_conv_op, mul_op, lora_B_conv_op, add_op]
                attach_index = self.find_node_index(self.graph, max_rank_attach_point.op.name)
                for i, new_node in enumerate(new_nodes):
                    self.graph.node.insert(attach_index + 1 + i, new_node)

                for node in new_nodes:
                    for inp in node.input:
                        if inp in self.consumers_map:
                            self.consumers_map[inp].append(node)
                        else:
                            self.consumers_map[inp] = [node]

                for consumer in self.consumers_map[output_tensor]:
                    for i, input_tensor in enumerate(consumer.input):
                        if input_tensor == output_tensor:
                            consumer.input[i] = add_op.output[0]


    def find_node_index(self, graph, node_name):
        for index, node in enumerate(graph.node):
            if node.name == node_name:
                return index
        return -1  # Return -1 if the node is not found


    def create_alpha_input_vector(self):
        # Create a new input tensor for the alpha input vector
        alpha_input_tensor = onnx.helper.make_tensor_value_info(
            name="lora_alpha",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[1, self.alpha_vector_size]
        )
        self.graph.input.insert(0, alpha_input_tensor)
        # Create a pad node to pad the alpha input vector
        pads = onnx.helper.make_tensor(self._get_node_name(alpha_input_tensor.name, "pads"), onnx.TensorProto.INT64, [4], [0, 0, 0, 1])
        self.graph.initializer.append(pads)
        pad_value = onnx.helper.make_tensor(self._get_node_name(alpha_input_tensor.name, "pad_value"), onnx.TensorProto.FLOAT, [1], [0.0])
        self.graph.initializer.append(pad_value)
        pad_node = onnx.helper.make_node(
            "Pad",
            inputs=[alpha_input_tensor.name, pads.name, pad_value.name],
            outputs=[self._get_node_name(alpha_input_tensor.name, "padded")],
            mode="constant"
        )
        self.graph.node.insert(0, pad_node)
        # Define the shape of the padded_alpha_input output
        padded_alpha_input_value_info = onnx.helper.make_tensor_value_info(
            name=self._get_node_name(alpha_input_tensor.name, "padded"),
            elem_type=onnx.TensorProto.FLOAT,
            shape=[1, self.alpha_vector_size + 1]
        )
        self.graph.value_info.insert(0, padded_alpha_input_value_info)


    def add_alpha_scattering_graph(self):
        for attach_point_name, attach_point_info in self.attach_point_info_map.items():
            max_rank_attach_point = self.attach_pts[attach_point_name]
            indices_tensor_name = max_rank_attach_point.alpha_separation_gather_indices_tensor_name
            max_rank_attach_point.alpha_separation_gather_indices_tensor = onnx.helper.make_tensor(
                    name=indices_tensor_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=[1, attach_point_info.max_rank],
                    vals=np.full((1, attach_point_info.max_rank), 1, dtype=np.int64)
                )

            self.graph.initializer.extend([max_rank_attach_point.alpha_separation_gather_indices_tensor])
            # Store the gathered indices tensor in the attach point object
            max_rank_attach_point.alpha_separation_gather_indices_tensor = self.graph.initializer[-1]

            gather_node = onnx.helper.make_node(
                "Gather",
                inputs=["lora_alpha_padded", max_rank_attach_point.alpha_separation_gather_indices_tensor.name],
                outputs=[self._get_output_tensor_name(attach_point_name, "gather")],
                axis=1
            )
            # Add a reshape node to reshape the gather output
            reshape_shape_tensor = onnx.helper.make_tensor(
                name=self._get_node_name(attach_point_name, "reshape", "shape"),
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=[1, attach_point_info.max_rank, 1, 1]
            )
            self.graph.initializer.extend([reshape_shape_tensor])
            reshape_node = onnx.helper.make_node(
                "Reshape",
                inputs=[gather_node.output[0], reshape_shape_tensor.name],
                outputs=[self._get_output_tensor_name(attach_point_name, "reshape")]
            )

            # Find the Mul node that is connected to the attach point
            mul_index = self.find_node_index(self.graph, max_rank_attach_point.lora_node_names.mul)
            mul_node = self.graph.node[mul_index]

            # Insert the new nodes in the correct order
            new_nodes = [gather_node, reshape_node]
            for i, new_node in enumerate(new_nodes):
                self.graph.node.insert(mul_index - len(new_nodes) + i, new_node)

            # Update the inputs of the Mul node to include the alpha scattering graph
            mul_node.input.append(reshape_node.output[0])
            max_rank_attach_point.lora_tensor_names.mul_scale = reshape_node.output[0]


    # Update the indices in the graph to base concurrency
    def update_indices_in_graph(self):
        for max_rank_attach_point in self.attach_pts.values():
            att_pt_gather_indices = self.base_indices_map[max_rank_attach_point.op.name]
            max_rank_attach_point.alpha_separation_gather_indices_tensor.ClearField('int64_data')
            max_rank_attach_point.alpha_separation_gather_indices_tensor.int64_data.extend(att_pt_gather_indices.alpha_indices[0])


    def create_max_rank_lora_graph(self):
        try:
            self.find_attach_pt_ops()

            log_debug("Adding max rank Lora branch...")
            self.add_max_rank_lora_branch()
            log_debug("Added max rank Lora branch successfully.")

            self.create_alpha_input_vector()
            log_debug("Adding alpha scattering graph...")

            self.add_alpha_scattering_graph()
            log_debug("Added alpha scattering graph successfully.")

            log_debug("Updating indices in graph...")
            self.update_indices_in_graph()
            log_debug("Updated indices in graph successfully.")

        except Exception as e:
            log_error(f"Error creating max rank Lora graph: {e}")
            # You can also log the error or raise a custom exception here
            sys.exit(1)


    def get_graph(self):
        return self.model