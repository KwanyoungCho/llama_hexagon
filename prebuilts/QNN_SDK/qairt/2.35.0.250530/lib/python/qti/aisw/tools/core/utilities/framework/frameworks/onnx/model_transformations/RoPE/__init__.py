# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# Not a contribution.
#
# -----------------------------------------------------------------------------

import onnx
from onnx.onnx_pb import ModelProto
from onnx import helper
import rich

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.logger import (
    log_warning,
    log_verbose,
    log_debug,
    log_info,
    setup_logging,
)

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.encodings import (
    AimetEncodings,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.pretty_print import (
    create_rich_table,
    bold_text,
    PrettyPrintConstants,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.utils import (
    get_head_num_and_dims,
)
from .model_utils import model_utils


class RoPE:
    def __init__(
        self,
        model: ModelProto,
        base_arch: str = "",
        encodings: dict | None = None,
        log_level: str = "info",
    ):
        self.model = model
        self.base_arch = base_arch
        setup_logging(log_level)

        self.node_map = {id(node): node for node in model.graph.node}
        self.graph_inputs = {input.name: input for input in model.graph.input}
        self.graph_outputs = set([output.name for output in model.graph.output])
        self.value_info = {
            value_info.name: value_info for value_info in self.model.graph.value_info
        }
        self.consumer_map = {}
        for node in model.graph.node:
            for input in node.input:
                if input in self.consumer_map:
                    self.consumer_map[input].append(id(node))
                else:
                    self.consumer_map[input] = [id(node)]

        self.utils = model_utils(
            model, self.node_map, self.graph_inputs, self.consumer_map, self.value_info
        )
        self.skip_match_adds = set()

        if encodings:
            self.aimet_enc = AimetEncodings(encodings)
        else:
            self.aimet_enc = None

        self.table = create_rich_table(
            title=bold_text(
                "RoPE Adaptation Results:", color=PrettyPrintConstants.Q_BLUE
            ),
            headers=["RoPE Number", "New Nodes"],
            positions=[0.18, 1.0],
            alignment=["left", "left"],
        )

    def match_pos_id_pattern(self, node_id):
        """
        Finds the position id pattern.

        looks for this specific patten:

        Slice
          |
         Neg     Slice
          |        |
        Concat Unsqueeze   Unsqueeze
           |       |           |
           └─ Mul ─┘          Mul
               |               |
               └───── Add ─────┘

        :param node_id: The id of the node we want to check.

        :return: Whether the node is the base of a position id chunk and the id's of the real and imag branches
        """
        # Find 2 Muls that feed into an Add.
        found_muls = self.utils.match_pattern(node_id, ["Mul", "Mul"])
        if found_muls:
            # Find a Mul that consumes the output of both an Unsqueeze and a Concat (sin)
            # and a Mul that consumes an Unsqeeze and something else (cos).
            for i in range(2):
                found_unsqueeze_concat = self.utils.match_pattern(
                    found_muls[i], ["Unsqueeze", "Concat"]
                )
                found_unsqueeze = self.utils.match_pattern(
                    found_muls[abs(i - 1)], ["Unsqueeze"]
                )
                if found_unsqueeze_concat and found_unsqueeze:
                    # The sin branch of the pos_id tree should have a Slice and Neg that feed into the concat.
                    found_slice_neg = self.utils.match_pattern(
                        found_unsqueeze_concat[1], ["Slice", "Neg"]
                    )
                    if found_slice_neg:
                        # Find a Slice that feeds into the Neg.
                        found_slice = self.utils.match_pattern(
                            found_slice_neg[1], ["Slice"]
                        )
                        if found_slice:
                            return (
                                True,
                                found_slice_neg[0],
                                found_slice[0],
                                self.node_map[found_muls[abs(i - 1)]].output[0],
                                self.node_map[found_muls[i]].output[0],
                            )
        return False, -1, -1, "", ""

    def create_rope(self, real, imag, RoPE_count, original_add_out, prefix=None):
        """
        Creates the RoPE ops and strings them together. After creation, the ops are inserted into
        the graph in the corresponding index such that the sequence is preserved.

        Builds this structure:
        Pre-existing nodes indicated with a *

        position_ids_sin   Slice*       Slice*   position_ids_cos
                │            │            │              │
                └────────┬───)────────────)───┐          │
                         │   │            │   │          │
                         │   │   ┌────┬───)───)──────────┘
                         │   │   │    │   │   │
                         │   │  Mul ──)───┴─ Mul
                         │   │   │    │       │
                        Mul ─┴───)── Mul      │
                         │       │    │       │
                         └─ Add ─┘    └─ Sub ─┘
                            │            │
                            └── Concat ──┘

        :param real: The id of the real Unsqueeze node that we need to connect the RoPE to.
        :param imag: The id of the imag Unsqueeze node that we need to connect the RoPE to.
        :param RoPE_count: Which RoPE structure this is.
        :param original_add_out: The output of the original add that we will be replacing. if it
                                 is a graph output, it's best to not change the name and just have
                                 the concat reuse the tensor.
        :param prefix: Prefix of all nodes that will be created in this op.

        :return: All the output tensors of the new RoPE structure
        """

        real_out_name = self.node_map[real].output[0]
        imag_out_name = self.node_map[imag].output[0]

        # The index that we insert the new ops in needs to be after both of the Slice ops.
        seq = [id(node) for node in self.model.graph.node]
        insert_index = max(seq.index(real), seq.index(imag)) + 1

        sub_cos_mul_name = self.utils.create_node_name("Mul", prefix)
        sub_sin_mul_name = self.utils.create_node_name("Mul", prefix)
        sub_cos_mul_out = sub_cos_mul_name + "_output_0"
        sub_sin_mul_out = sub_sin_mul_name + "_output_0"
        sub_cos_mul = helper.make_node(
            "Mul",
            inputs=[real_out_name, "position_ids_cos"],
            outputs=[sub_cos_mul_out],
            name=sub_cos_mul_name,
        )
        sub_sin_mul = helper.make_node(
            "Mul",
            inputs=[imag_out_name, "position_ids_sin"],
            outputs=[sub_sin_mul_out],
            name=sub_sin_mul_name,
        )

        sub_name = self.utils.create_node_name("Sub", prefix)
        sub_out = sub_name + "_output_0"
        sub = helper.make_node(
            "Sub",
            inputs=[sub_cos_mul_out, sub_sin_mul_out],
            outputs=[sub_out],
            name=sub_name,
        )

        add_cos_mul_name = self.utils.create_node_name("Mul", prefix)
        add_sin_mul_name = self.utils.create_node_name("Mul", prefix)
        add_cos_mul_out = add_cos_mul_name + "_output_0"
        add_sin_mul_out = add_sin_mul_name + "_output_0"
        add_cos_mul = helper.make_node(
            "Mul",
            inputs=[imag_out_name, "position_ids_cos"],
            outputs=[add_cos_mul_out],
            name=add_cos_mul_name,
        )
        add_sin_mul = helper.make_node(
            "Mul",
            inputs=[real_out_name, "position_ids_sin"],
            outputs=[add_sin_mul_out],
            name=add_sin_mul_name,
        )

        add_name = self.utils.create_node_name("Add", prefix)
        add_out = add_name + "_output_0"
        add = helper.make_node(
            "Add",
            inputs=[add_cos_mul_out, add_sin_mul_out],
            outputs=[add_out],
            name=add_name,
        )
        self.skip_match_adds.add(id(add))

        concat_name = self.utils.create_node_name("Concat", prefix)
        concat_out = (
            concat_name + "_output_0"
            if original_add_out not in self.graph_outputs
            else original_add_out
        )
        concat = helper.make_node(
            "Concat",
            inputs=[sub_out, add_out],
            outputs=[concat_out],
            name=concat_name,
            axis=3,
        )

        new_nodes = [
            concat,
            sub,
            sub_cos_mul,
            sub_sin_mul,
            add,
            add_cos_mul,
            add_sin_mul,
        ]

        self.table.add_row(
            str(RoPE_count + 1), ", ".join([node.name for node in new_nodes])
        )

        for node in new_nodes:
            self.model.graph.node.insert(insert_index, node)

        return (
            concat.output[0],
            add_out,
            sub_out,
            add_cos_mul_out,
            add_sin_mul_out,
            sub_cos_mul_out,
            sub_sin_mul_out,
        )

    def copy_encodings(
        self,
        original_add,
        original_mul_cos,
        original_mul_sin,
        new_mul_real0,
        new_mul_real1,
        new_mul_imag0,
        new_mul_imag1,
        new_add,
        new_sub,
        new_concat,
        copy_concat_out,
    ):
        """
        Copies the encodings from the original tensors to the new tensors.
        The output of the new Mul ops will have their encodings copied from the old Mul ops. The
        remaining new ops will all have the same encodings as the original Add op.

        :param original_add: The name of the original tensor that is the output of the Add.
        :param original_mul_cos: The name of the original tensor that is the output of the Mul on
                                 the cos branch.
        :param original_mul_sin: The name of the original tensor that is the output of the Mul on
                                 the sin branch.
        :param new_mul_real0: The name of the new tensor that is consumed by the new Add and is the
                              output of the Mul which consumes position_ids_cos. This tensor will
                              have the same encodings as original_mul_cos.
        :param new_mul_real1: The name of the new tensor that is consumed by the new Sub and is the
                              output of the Mul which consumes position_ids_sin. This tensor will
                              have the same encodings as original_mul_cos.
        :param new_mul_imag0: The name of the new tensor that is consumed by the new Add and is the
                              output of the Mul which consumes position_ids_sin. This tensor will
                              have the same encodings as original_mul_sin.
        :param new_mul_imag1: The name of the new tensor that is consumed by the new Sub and is the
                              output of the Mul which consumes position_ids_cos. This tensor will
                              have the same encodings as original_mul_sin.
        :param new_add: The name of the new tensor that is the output of the new Add. This tensor
                        will have the same encodings as original_add.
        :param new_sub: The name of the new tensor that is the output of the new Sub. This tensor
                        will have the same encodings as original_add.
        :param new_concat: The name of the new tensor that is the output of the new Concat. This
                           tensor will have the same encodings as original_add.
        :param copy_concat_out: If the concat already has the same tensor name as the original
                                add output tensor, there's no need to copy the encodings.
        """
        if self.aimet_enc:
            self.aimet_enc.copy_encoding(original_mul_cos, new_mul_real0)
            self.aimet_enc.copy_encoding(original_mul_cos, new_mul_real1)

            self.aimet_enc.copy_encoding(original_mul_sin, new_mul_imag0)
            self.aimet_enc.copy_encoding(original_mul_sin, new_mul_imag1)

            self.aimet_enc.copy_encoding(original_add, new_add)
            self.aimet_enc.copy_encoding(original_add, new_sub)
            if copy_concat_out:
                self.aimet_enc.copy_encoding(original_add, new_concat)

    def compute_head_dim(self, real):
        """
        Finds the head_dim which is required to calculate the proper dimensions for
        position_id_sin/cos.

        :param real: The id of the real Slice which can be used to find the
                     Matmul/Conv -> (Transpose) -> Reshape pattern.

        :return: The head dimension if the structure is found. Otherwise, -1
        """
        found, reshape = self.utils.find_first_of_above(real, "Reshape")
        reshape_dims = self.utils.get_reshape_constant(reshape)
        if found:
            if reshape_dims is None:
                log_warning(
                    f"Could not properly get shape input for {self.node_map[reshape].name}"
                )
                return -1

            # Matmul head:
            found_matmul, _ = self.utils.find_first_of_above(reshape, "MatMul")
            if found_matmul:
                log_debug(f"Identified head dim of {reshape_dims[3]}")
                return reshape_dims[3]

            # Conv head:
            found_conv, conv = self.utils.find_first_of_above(reshape, "Conv")
            if found_conv:
                # Need to check if there is an intermediate transpose
                upstream, _ = self.utils.get_all_upstream(reshape, set([conv]))
                if "Transpose" in upstream.values():
                    # Intermediate transpose
                    _, transpose_before_reshape = self.utils.find_first_of_above(
                        reshape, "Transpose"
                    )
                    # Transpose after reshape
                    _, transpose_after_reshape = self.utils.find_first_of_below(
                        reshape, "Transpose"
                    )

                    _, head_dim = get_head_num_and_dims(
                        reshape_dims,
                        self.node_map[transpose_before_reshape],
                        self.node_map[transpose_after_reshape],
                    )

                    return head_dim

        return -1

    def perform_RoPE(self):
        """
        Performs RoPE on the model. Finds the structure described in the match_pos_id_pattern
        function and replaces it along with any unecessary upstream nodes/tensors with the
        structure described in the create_rope function.

        :return: 1. An onnx ModelProto that has the RoPE structure if successful. Otherwise, the
                   original model is returned
                 2. A boolean value indicating whether applying the RoPE adaptation was successful.
        """
        if self.base_arch != "llama2" and self.base_arch != "llama3":
            log_warning("Skipping RoPE adaptation.")
            return self.model, False

        modified = False

        position_ids_sin = None
        position_ids_cos = None

        nodes_to_remove = set()
        inputs_to_remove = set()

        num_matches = 0

        for node in self.model.graph.node:
            if node.op_type == "Add" and id(node) not in self.skip_match_adds:
                self.skip_match_adds.add(id(node))
                match, real, imag, cos_mul_out, sin_mul_out = self.match_pos_id_pattern(
                    id(node)
                )
                if match:
                    add_out = node.output[0]
                    add_out_is_graph_out = add_out in self.graph_outputs
                    modified = True
                    # Need to first create the inputs.
                    if not position_ids_sin and not position_ids_cos:
                        head_dim = self.compute_head_dim(real)
                        if head_dim < 0:
                            raise ValueError("Could not properly get head_dim")
                        input_shapes = [
                            1,
                            1,
                            self.utils.get_seq_len(),
                            int(head_dim / 2),
                        ]
                        position_ids_sin = helper.make_tensor_value_info(
                            "position_ids_sin", onnx.TensorProto.FLOAT, input_shapes
                        )
                        position_ids_cos = helper.make_tensor_value_info(
                            "position_ids_cos", onnx.TensorProto.FLOAT, input_shapes
                        )
                        log_debug(
                            f"position_id_cos/sin inputs created with shape {input_shapes}"
                        )

                        # Add the new inputs to the graph
                        self.model.graph.input.extend(
                            [position_ids_sin, position_ids_cos]
                        )

                    # Get all the upstream nodes/inputs that aren't needed.
                    real_upstream, _ = self.utils.get_all_upstream(real)
                    imag_upstream, _ = self.utils.get_all_upstream(imag, set([real]))
                    real_imag_upstream = set(real_upstream.keys()).union(
                        imag_upstream.keys()
                    )
                    upstream_nodes_to_remove, upstream_inputs_to_remove = (
                        self.utils.get_all_upstream(id(node), real_imag_upstream)
                    )
                    nodes_to_remove.update(upstream_nodes_to_remove.keys())
                    inputs_to_remove.update(upstream_inputs_to_remove)

                    (
                        rope_out,
                        r_add_out,
                        r_sub_out,
                        r_add_cos_mul_out,
                        r_add_sin_mul_out,
                        r_sub_cos_mul_out,
                        r_sub_sin_mul_out,
                    ) = self.create_rope(real, imag, num_matches, add_out)
                    log_verbose(
                        f"RoPE structure formed and added to graph in place of Add with name {node.name}"
                    )
                    num_matches += 1

                    if self.aimet_enc:
                        self.copy_encodings(
                            add_out,
                            cos_mul_out,
                            sin_mul_out,
                            r_add_cos_mul_out,
                            r_sub_sin_mul_out,
                            r_add_sin_mul_out,
                            r_sub_cos_mul_out,
                            r_add_out,
                            r_sub_out,
                            rope_out,
                            not add_out_is_graph_out,
                        )

                    # If the output is not a graph output, we need to reconnect the RoPE output to
                    # the graph.
                    if not add_out_is_graph_out:
                        for consumer in self.consumer_map[add_out]:
                            consumer_node = self.node_map[consumer]
                            add_output_index = list(consumer_node.input).index(add_out)
                            consumer_node.input.insert(add_output_index, rope_out)
                            consumer_node.input.remove(add_out)

        for node_to_remove in nodes_to_remove:
            for output in self.node_map[node_to_remove].output:
                if output in self.value_info:
                    self.model.graph.value_info.remove(self.value_info[output])
            self.model.graph.node.remove(self.node_map[node_to_remove])

        for input_to_remove in inputs_to_remove:
            self.model.graph.input.remove(self.graph_inputs[input_to_remove])

        if modified:
            # if self.aimet_enc:
            #     self.aimet_enc.save_encoding()
            console = rich.console.Console(highlight=True)
            console.print(self.table, overflow="fold")
            log_debug(
                f"Removed Layers:\n{', '.join([self.node_map[node].name for node in nodes_to_remove])}"
            )
            log_debug(
                f"Removed Inputs:\n{', '.join([self.graph_inputs[input].name for input in inputs_to_remove])}"
            )
            log_info("RoPE adaptation complete.")

        else:
            log_warning(
                "Could not identify correct position_ids pattern. Skipping RoPE adaptation."
            )

        return self.model, modified
