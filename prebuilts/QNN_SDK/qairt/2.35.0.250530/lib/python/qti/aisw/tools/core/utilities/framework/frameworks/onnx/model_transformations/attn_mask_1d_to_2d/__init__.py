# ==============================================================================
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# Not a contribution.
# ==============================================================================

from typing import Dict, List, Tuple

import onnx
from onnx import helper
from onnx.onnx_pb import ModelProto

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.clean import (
    clean_model,
    topological_sort,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.onnx import (
    get_children,
    get_node_by_input_name,
    get_node_by_output_name,
    get_node_mappings,
    get_parent,
    get_pattern_start_end_nodes,
    get_shape_from_value_info_proto,
    get_value_info_proto_mapping,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.encodings import (
    AimetEncodings,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.logger import (
    log_debug,
    log_error,
    log_info,
    log_warning,
    setup_logging,
)

from .attn_mask_patterns import attention_patterns


class AdaptAttentionMask:
    """
    This Pass is applicable for transformer based decoder models with bert style or
    kv style of execution.
    This will remove the causal mask which is generated inside the model and replace
    all usage of that with attention_mask graph input. Note that the original
    attention_mask input which is [batch, seq_k] is now changed to [batch, 1, seq_q,
    seq_k]. With respect to KV style execution seq_k refers to full sequence
    containing past sequence and current sequence tokens and seq_q refers to
    current sequence tokens. For Bert style execution seq_k and seq_q both refer to
    current sequence tokens.

    Assumptions:
    - Model should have an input named "attention_mask" with shape [batch, seq_k].

    For Bert model  : seq_k = seq_q = whole context / full sequence.
    For KV model    : seq_q = current sequence
                    : seq_k = full sequence = past_seq + current sequence

    GPT2 pattern
        Before:
                                    (batch, num_heads, seq_q, seq_k)
                                                |
                                                v
                                ...... MatMul -> Div -> Where \
                                                               Add -> Softmax
            attention_mask -> Unsqueeze -> Cast -> Sub -> Mul /
                ^
                |
            (batch, seq_k)

        After:
                                    (batch, num_heads, seq_q, seq_k)
                                                |
                                                v
                                ..... MatMul -> Div \
                                                    Add -> Softmax
            attention_mask ------------------------ /
                ^
                |
            (batch, 1, seq_q, seq_k)

    Llama7b pattern
        Before:
                                                                        (batch, num_heads, seq_q, seq_k)
                                                                                    |
                                                                                    v
                                ........................................ MatMul -> Div \
                                                                                        Add -> Softmax
            attention_mask -> Unsqueeze -> Equal -> Add -> Where -> ScatterND -> Slice /
                ^
                |
            (batch, seq_q)

        After:
            (batch, num_heads, seq_q, seq_k)
                        |
                        v
                ..... MatMul -> Div \
                                    Add -> Softmax
            attention_mask -------- /
                ^
                |
            (batch, 1, seq_q, seq_k)

    Baichuan pattern
        Before:
                                                                (batch, num_heads, seq_q, seq_k)
                                                                            |
                                                                            v
                            ........................................ MatMul -> Div \
                                                                                    Add -> Max -> Softmax
                                                         /- Cast -\                /
            attention_mask -> Unsqueeze -> Expand -> Sub ----------> Where -> Add /
                ^
                |
            (batch, seq_q)

        After:
            (batch, num_heads, seq_q, seq_k)
                        |
                        v
                ..... MatMul -> Div \
                                    Add -> Softmax
            attention_mask -------- /
                ^
                |
            (batch, 1, seq_q, seq_k)
    """

    def __init__(
        self,
        model: ModelProto,
        encodings: dict | None = None,
        log_level: str = "info",
    ):
        """
        Constructor for Attention Mask Adaptation.

        :param ModelProto model: Onnx model on which adaptation is to be applied.
        :param str log_level: Granularity of the logs, defaults to "info"
        :param str encodings_file: Path of quantization encodings file,
            defaults to None.
        """
        self.model = model
        self.log_level = log_level
        setup_logging(log_level)
        self.populate_mappings()

        if encodings:
            self.aimet_enc = AimetEncodings(encodings)
        else:
            self.aimet_enc = None

    def populate_mappings(self) -> None:
        """
        Helper function to populate the graph parsing mappings.
        """
        self.node_mapping = get_node_mappings(self.model)
        self.val_info_mapping = get_value_info_proto_mapping(self.model)
        self.node_by_output_name = get_node_by_output_name(self.model)
        self.node_by_input_name = get_node_by_input_name(self.model)

    def identify_node_to_be_removed(self, identified_pattern: Dict) -> List[int]:
        """
        Identify nodes to be removed from identified pattern.
        E.g. For GPT model the identified pattern is
            [MatMul, Div, Where, Add, Softmax] and from that Where node shall be
            removed. This function gives index of that node in the
            identified_pattern.

        :param Dict identified_pattern: Identified pattern in the model.
        :raises RuntimeError: If pattern doesn't have correct remove_nodes
            information.
        :return List[int]: List of indices of nodes to be removed from
            identified_pattern["pattern"].
        """
        for pattern in attention_patterns:
            if pattern["pattern"] == identified_pattern:
                remove_nodes = pattern["remove_nodes"]
                if len(remove_nodes) == 0:
                    return []
                if not isinstance(remove_nodes, List):
                    raise RuntimeError("Unsupported format provided for remove_node field.")
                if isinstance(remove_nodes[0], int):
                    remove_nodes_idx = remove_nodes
                elif isinstance(remove_nodes[0], str):
                    remove_nodes_idx = []
                    for remove_node in remove_nodes:
                        try:
                            idx = identified_pattern.index(remove_nodes[0])
                            if idx < 0:
                                idx = len(identified_pattern) + idx
                            remove_nodes_idx.append(idx)
                        except ValueError as e:
                            raise RuntimeError(
                                f"Can't identify node '{remove_node}' to be "
                                "removed from identified pattern "
                                f": {identified_pattern}"
                            ) from e
                else:
                    raise RuntimeError("Unsupported format provided for remove_nodes field.")
                break
        return remove_nodes_idx

    def remove_redundant_pattern_nodes(
        self,
        identified_pattern: Dict,
        start_node_names: List[str],
        end_node_names: List[str],
    ) -> bool:
        """
        Removes the connections of the redundant pattern nodes in the graph.

        :param Dict identified_pattern: Identified pattern in the model.
        :param List[str] start_node_names: List of pattern start node names.
        :param List[str] end_node_names: List of pattern end node names.
        :return bool: Status indicating success of the operation.
        """
        try:
            remove_nodes_idx = self.identify_node_to_be_removed(identified_pattern)
            if (0 in remove_nodes_idx) or ((len(identified_pattern) - 1) in remove_nodes_idx):
                log_error("Can't remove start node or end node of the detected pattern.")
                return False
        except Exception as e:
            log_error(f"Unable to remove redundant nodes due to exception: {e}")
            return False

        if len(remove_nodes_idx) == 0:
            return True
        nodes_to_be_removed = []
        for start_node_name, end_node_name in zip(start_node_names, end_node_names):
            start_node = self.node_mapping[start_node_name]
            nodes_children = get_children(start_node, self.node_by_input_name)

            stack = [[_n, 1] for _n in nodes_children]
            visited_nodes = []
            while len(stack) != 0:
                _node, _depth = stack.pop()
                if _node in visited_nodes:
                    continue
                visited_nodes.append(_node)
                if _depth in remove_nodes_idx:
                    nodes_to_be_removed.append(_node)
                for _child_node in get_children(_node, self.node_by_input_name):
                    if _child_node.name == end_node_name:
                        continue
                    stack.append([_child_node, _depth + 1])

        # No need to remove the nodes here. Just remove some connections.
        # We will call cleanup at the end to remove dangling nodes.
        for node_to_remove in nodes_to_be_removed:
            par_node = get_parent(node_to_remove, self.node_by_output_name)
            par_node = [p for p in par_node if p.op_type != "Constant"]
            if len(par_node) != 1:
                log_error(
                    "Can't remove node '{node_to_remove.name}' as it "
                    "doesn't have single non-constant parent node."
                )
                return False
            if len(node_to_remove.output) != 1:
                log_error(f"Can't remove node '{node_to_remove.name}' as it " "doesn't have single output.")
                return False
            new_input = par_node[0].output[0]
            old_input = node_to_remove.output[0]
            children_nodes = get_children(node_to_remove, self.node_by_input_name)

            for child_node in children_nodes:
                for idx, child_input in enumerate(child_node.input):
                    if child_input == old_input:
                        child_node.input[idx] = new_input

        return True

    def reachable_graph_input(self, tensor_name: str) -> List[str]:
        """
        Get the reachable graph input from the given tensor name.

        :param str tensor_name: Name of the tensor from where the graph input
            is to be found.
        :return List[str]: List of names of graph inputs required for
            tensor_name.
        """
        node = self.node_by_output_name[tensor_name]
        stack = [node]
        visited_nodes = []
        reachable_graph_inputs = []
        graph_inputs = [ip.name for ip in self.model.graph.input]
        while len(stack) != 0:
            _node = stack.pop()
            if _node in visited_nodes:
                continue
            visited_nodes.append(_node)

            _par_nodes = get_parent(_node, self.node_by_output_name)
            if _node.op_type != "Constant":
                for _node_ip in _node.input:
                    if _node_ip in graph_inputs:
                        reachable_graph_inputs.append(_node_ip)
            stack.extend(_par_nodes)

        return reachable_graph_inputs

    def update_attention_mask_input_shape_dtype(self, attn_mask_input: str, qk_matmul_tensor: str) -> None:
        """
        Update attention mask graph input from [batch, full_seq (seq_k)] shape
        to [batch, 1, curr_seq (seq_q), full_seq (seq_k)] shape. The dtype also
        should be updated to float32.

        :param str attn_mask_input: Name of attention mask graph input.
        :param str qk_matmul_tensor: Q<matmul>K tensor name.
        :raises RuntimeError: If model doesn't have any shapes for
            qk_matmul_tensor.
        """
        if qk_matmul_tensor not in self.val_info_mapping:
            raise RuntimeError(f"No shape information available for tensor: {qk_matmul_tensor}")

        qk_matmul_tensor_val_info = self.val_info_mapping[qk_matmul_tensor]
        qk_matmul_tensor_shape = get_shape_from_value_info_proto(qk_matmul_tensor_val_info)

        # attn_mask_shape       : [batch, full_seq (seq_k)]
        # qk_matmul_shape       : [batch, num_heads, curr_seq (seq_q), full_seq (seq_k)]
        # new_attn_mask_shape   : [batch, 1, curr_seq (seq_q), full_seq (seq_k)]
        new_attn_mask_shape = [
            qk_matmul_tensor_shape[0],
            1,
            qk_matmul_tensor_shape[2],
            qk_matmul_tensor_shape[3],
        ]
        for idx, model_ip in enumerate(self.model.graph.input):
            if model_ip.name == attn_mask_input:
                new_input = helper.make_tensor_value_info(
                    attn_mask_input, onnx.TensorProto.FLOAT, new_attn_mask_shape
                )
                self.model.graph.input.remove(model_ip)
                self.model.graph.input.insert(idx, new_input)

    def replace_attn_mask_connections(self, attn_mask_input: str, processed_attn_mask_tensor: str) -> None:
        """
        Replace the attention mask graph input connections. This will replace
        the connections from processed_attn_mask_tensor to its children nodes
        with attn_mask_input to those children nodes.

        :param str attn_mask_input: Attention mask graph input name.
        :param str processed_attn_mask_tensor: Processed attention mask tensor
            which gets fed to Add node.
        :param NodeProto attn_mask_add_node: Add node reference.
        """
        children_nodes = self.node_by_input_name[processed_attn_mask_tensor]

        for child_node in children_nodes:
            for idx, node_ip in enumerate(child_node.input):
                if node_ip == processed_attn_mask_tensor:
                    child_node.input[idx] = attn_mask_input

        attn_mask_input_children = self.node_by_input_name[attn_mask_input]
        for node in attn_mask_input_children:
            for node_ip in node.input:
                if node_ip == attn_mask_input:
                    node.input.remove(node_ip)

    def identify_qk_and_attn_tensors(
        self, identified_pattern: Dict, start_node_name: str, end_node_name: str
    ) -> Tuple:
        """
        Identifies q<matmul>k tensor and processed attention mask tensor.

        :param Dict identified_pattern: Identified pattern in the model.
        :param List[str] start_node_name: Name of the start node in identified
            pattern.
        :param List[str] end_node_name: Name of the end node in identified
            pattern.
        :raises RuntimeError: If the processed attention mask or q<matmul>k
            tensor are not identified correctly.
        :return Tuple: Tuple of 2 values.
            - First value represents q<matmul>k tensor name.
            - Second value represents processed attention mask tensor name.
        """
        model_inputs = [inp.name for inp in self.model.graph.input]
        end_node = self.node_mapping[end_node_name]

        stack = [[end_node, len(identified_pattern) - 1]]
        visited_nodes = []
        processed_attn_mask_tensor = None
        qk_matmul_tensor = None
        while len(stack) != 0:
            _node, pattern_idx = stack.pop()
            if _node in visited_nodes:
                continue
            visited_nodes.append(_node)

            if (len(_node.input) == 2) and (_node.op_type == "Add"):
                for _node_input in _node.input:
                    if _node_input in model_inputs:
                        # During single updation of attention mask, we will be
                        # changing all the childrens of the existing attention
                        # mask. Hence for other instances of matched
                        # pattern, we won't be able to find
                        # processed_attn_mask_tensor as an intermediate tensor
                        # but processed_attn_mask_tensor has already been made
                        # as attention_mask.
                        processed_attn_mask_tensor = _node_input
                    else:
                        if _node_input not in self.node_by_output_name:
                            continue
                        _par_node = self.node_by_output_name[_node_input]
                        if _par_node.op_type != identified_pattern[pattern_idx - 1]:
                            processed_attn_mask_tensor = _node_input
                        elif _par_node.op_type == identified_pattern[pattern_idx - 1]:
                            qk_matmul_tensor = _node_input
                        else:
                            continue
                # Break the while loop.
                break

            _par_nodes = get_parent(_node, self.node_by_output_name)
            for _par_node in _par_nodes:
                if _par_node.name == start_node_name:
                    continue
                stack.append([_par_node, pattern_idx - 1])

        if (processed_attn_mask_tensor is None) or (qk_matmul_tensor is None):
            raise RuntimeError("Failed to get the processed attention mask tensor name.")
        return qk_matmul_tensor, processed_attn_mask_tensor

    def apply(self) -> ModelProto:
        """
        Apply Attention Mask 1d to 2d model adaptation.

        :raises RuntimeError: If the attention_mask graph input is not
            identified correctly.
        :return ModelProto: Update onnx model reference.
        """
        log_info("Applying attention mask 1d to 2d model adaptation.")
        try:
            identified_pattern, start_node_names, end_node_names = get_pattern_start_end_nodes(
                self.model, attention_patterns=attention_patterns
            )
        except Exception:
            log_warning("Unable to find required patterns. Skipping attention mask adaptation.")
            return self.model, False

        model_inputs = [inp.name for inp in self.model.graph.input]
        log_debug(f"Identified pattern: {identified_pattern}")
        for start_node_name, end_node_name in zip(start_node_names, end_node_names):
            log_debug(f"Start node name of the identified pattern: {start_node_name}")
            log_debug(f"End node name of the identified pattern: {end_node_name}")
            qk_matmul_tensor, processed_attn_mask_tensor = self.identify_qk_and_attn_tensors(
                identified_pattern, start_node_name, end_node_name
            )
            if processed_attn_mask_tensor in model_inputs:
                log_debug(
                    "Processed attention mask tensor : "
                    f"{processed_attn_mask_tensor} is already update to "
                    "be an attention mask graph input."
                )
                continue

            reachable_inputs = self.reachable_graph_input(processed_attn_mask_tensor)
            if len(reachable_inputs) != 1:
                log_error(f"Unable to find attention mask input from {processed_attn_mask_tensor}")
                return self.model, False

            attn_mask_input = reachable_inputs[0]

            attn_mask_children = self.node_by_input_name[attn_mask_input]
            if len(attn_mask_children) != 1:
                log_error("Attention mask input shall have only single child node.")
                return self.model, False

            try:
                self.update_attention_mask_input_shape_dtype(attn_mask_input, qk_matmul_tensor)
            except Exception as e:
                log_error(f"Unable to update shapes for attention mask input due to exception: {e}")
                return self.model, False

            self.replace_attn_mask_connections(attn_mask_input, processed_attn_mask_tensor)

            if self.aimet_enc:
                self.aimet_enc.copy_encoding(processed_attn_mask_tensor, attn_mask_input)
                # self.aimet_enc.save_encoding()

        status = self.remove_redundant_pattern_nodes(identified_pattern, start_node_names, end_node_names)
        if not status:
            return self.model, False

        clean_model(self.model)
        topological_sort(self.model)

        self.populate_mappings()

        return self.model, True
