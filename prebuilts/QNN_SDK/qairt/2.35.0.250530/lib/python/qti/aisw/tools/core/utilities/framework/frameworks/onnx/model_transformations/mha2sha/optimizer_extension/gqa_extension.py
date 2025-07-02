# -----------------------------------------------------------------------------
#
# Qualcomm Technologies, Inc. Proprietary
# (c) 2024 Qualcomm Technologies, Inc. All rights reserved.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
# -----------------------------------------------------------------------------
from typing import Any, Dict, Tuple

from onnx import numpy_helper
from onnx.onnx_pb import NodeProto

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.onnx import (
    NodeNotFoundError,
    get_next_node_down_based_on_cond,
    get_node_input_constant_op_value,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.utils import (
    BranchType,
    get_head_num_and_dims,
    sha_node_name_basis,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.logger import (
    log_debug,
)

mha2sha_hf_model_optimizer = Any  # Causes circular import


class GqaExtension:
    """Extenstion helpers for mha2sha_optimzer to bridge Morpheus pipeline code base and v1.0.0 release."""

    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """
        self.mha2sha_optim = mha2sha_optim

    def get_kv_group_head_num(self, k_proj_end_node, key_matmul_node, qk_matmul_node):
        """
        Get head_num for kv branch by search
        the transpose(Potential) -> reshape pattern.

        :param k_proj_end_node: last node of k projection
                    - for k-projection with lora: this is lora_add node
                    - for k-projection without lora: this is key_matmul_node
        :param key_matmul_node: key matmul
        :param qk_matmul_node: qk_matmul_node
        :return number_of_heads: kv branch head num
        """
        matmul_input_shape = None
        for tensor in self.mha2sha_optim.model.graph.value_info:
            if tensor.name == qk_matmul_node.input[1]:
                matmul_input_shape = list(tensor.type.tensor_type.shape.dim)

        if matmul_input_shape is None:
            raise ValueError("Unable to evaluate QK matmuls input[0] shape. Please run shape inference.")

        reshape_node = None
        reshape_input_shape = None
        transpose_before_reshape = None
        transpose_after_reshape = None

        try:
            # Use the first reshape node after k matmul before qk_matmul
            reshape_node = get_next_node_down_based_on_cond(
                k_proj_end_node,
                self.mha2sha_optim.get_node_by_input_name,
                node_found_cond=lambda n: n.op_type == "Reshape",
                node_end_search_cond=lambda n: n == qk_matmul_node,
            )

            reshape_input_shape = get_node_input_constant_op_value(
                reshape_node,
                self.mha2sha_optim.get_node_by_output_name,
                self.mha2sha_optim.get_initializer_by_name,
            )
        except NodeNotFoundError:
            log_debug(
                "No reshape node found in K branch whose input shape is used to compute number of heads"
            )

        try:
            if self.mha2sha_optim.mha_conv and self.mha2sha_optim.nchw_aligned:
                if reshape_node:
                    # collecting transpose between k_matmul and k_reshape
                    transpose_before_reshape = get_next_node_down_based_on_cond(
                        key_matmul_node,
                        self.mha2sha_optim.get_node_by_input_name,
                        node_found_cond=lambda n: n.op_type == "Transpose",
                        node_end_search_cond=lambda n: n == reshape_node,
                    )
        except NodeNotFoundError:
            log_debug(
                "No transpose node found between K matmul and K reshape used to compute number of heads"
            )

        try:
            if reshape_node:
                # collecting transpose between k reshape and qk_matmul
                transpose_after_reshape = get_next_node_down_based_on_cond(
                    reshape_node,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "Transpose",
                    node_end_search_cond=lambda n: n == qk_matmul_node,
                )

        except NodeNotFoundError:
            log_debug(
                "No transpose node found between K reshape and QK matmul used to compute number of heads"
            )

        try:
            if reshape_node is None:
                # collecting transpose between k matmul and qk_matmul
                transpose_after_reshape = get_next_node_down_based_on_cond(
                    key_matmul_node,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "Transpose",
                    node_end_search_cond=lambda n: n == qk_matmul_node,
                )
        except NodeNotFoundError:
            log_debug(
                "No transpose node found between K reshape and QK matmul used to compute number of heads"
            )

        if (reshape_node is None and transpose_after_reshape is not None) or (
            transpose_after_reshape is None and reshape_node is not None
        ):
            # Matmul input key shape [B, head_num, head_dim, h*w]
            number_of_heads = matmul_input_shape[-3].dim_value
        else:
            number_of_heads, _ = get_head_num_and_dims(
                reshape_input_shape, transpose_before_reshape, transpose_after_reshape
            )

        return number_of_heads

    def get_qkv_info(
        self,
        qk_matmul_node: NodeProto,
        qkv_matmul_node: NodeProto,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Function responsible for collecting QKV information.

        :param qk_matmul_node: The MatMul of where the Query and Key branches join.
        :param qkv_matmaul_node: The MatMul of where the Query, Key, and Key branches join.

        :return dquery: Dict - it contains matmul (node, initializers)
                and add (node, initializers).
        :return dkey: Dict - it contains matmul (node, initializers)
                and add (node, initializers).
        :return dvalue: Dict - it contains matmul (node, initializers)
                and add (node, initializers).
        """
        # Find Add op that adds up baselinear and lora output
        if self.mha2sha_optim.mha_conv:
            query_proj, key_proj, value_proj = self.mha2sha_optim._mha_conv_extension.get_qkv_conv(
                qk_matmul_node, qkv_matmul_node
            )
            query_initializer = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(query_proj)
            key_initializer = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(key_proj)
            value_initializer = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(value_proj)

        else:
            query_proj, key_proj, value_proj = self.mha2sha_optim._mha_conv_extension.get_qkv_matmuls(
                qk_matmul_node, qkv_matmul_node
            )
            query_initializer = numpy_helper.to_array(self.get_initializer_by_name[query_proj.input[1]])
            key_initializer = numpy_helper.to_array(self.get_initializer_by_name[key_proj.input[1]])
            value_initializer = numpy_helper.to_array(self.get_initializer_by_name[value_proj.input[1]])

        dquery = {"matmul_node": query_proj, "matmul_init": query_initializer}

        dkey = {"matmul_node": key_proj, "matmul_init": key_initializer}

        dvalue = {"matmul_node": value_proj, "matmul_init": value_initializer}

        self.kv_head_num = self.get_kv_group_head_num(key_proj, qk_matmul_node)

        return dquery, dkey, dvalue

    def _reset_cached_keys_value_nodes(self):
        """GQA model cache sha key and value Matmul nodes."""
        self.key_groups_list = []
        self.value_groups_list = []

    def reset_and_pre_build_kv_proj_nodes(
        self,
        info_dict,
        ns,
        head_dim,
        key_matmul_inp,
        value_matmul_inp,
        sha_base_attn_node_list,
    ):
        """
        Reset self.key_groups_list and self.value_groups_list. Then create all k, v proj nodes before
        creating SHA. Add rope node if needed.
        :key_groups_list: each key has shape [B, 1, seq_len, head_dim], [B, 1, head_dim, seq_len] if handle_rope_ops
        :value_groups_list: each value has shape [B, head_dim, 1, seq_len], [B, 1, seq_len, head_dim] if handle_rope_ops
        """
        assert self.mha2sha_optim.mha_conv, "support only mha-conv model"

        self._reset_cached_keys_value_nodes()
        for head_num in range(self.kv_head_num):
            # Create weight names
            if (
                "matmul_init" in info_dict["key"].keys()
                and (conv_weight_init := info_dict["key"]["matmul_init"]) is not None
            ):
                key_conv = self.mha2sha_optim._mha_conv_extension.create_single_conv(
                    conv_weight_init,
                    ns,
                    head_num,
                    head_dim,
                    key_matmul_inp,
                    suffix=None,
                    branch_type=BranchType.K,
                    bias_init=info_dict["key"].get("matmul_init_bias", None),
                )
            else:
                raise ValueError("key matmul weight is None")

            if (
                "matmul_init" in info_dict["value"].keys()
                and (conv_weight_init := info_dict["value"]["matmul_init"]) is not None
            ):
                value_conv = self.mha2sha_optim._mha_conv_extension.create_single_conv(
                    conv_weight_init,
                    ns,
                    head_num,
                    head_dim,
                    value_matmul_inp,
                    suffix=None,
                    branch_type=BranchType.V,
                    bias_init=info_dict["value"].get("matmul_init_bias", None),
                )
            else:
                raise ValueError("value matmul weight is None")

            sha_base_attn_node_list.k_matmul.append(key_conv)
            sha_base_attn_node_list.v_matmul.append(value_conv)

            if self.mha2sha_optim.lora_model:
                key_conv = self.mha2sha_optim._lora_extension.attach_single_lora_adaptor(
                    info_dict["key"],
                    ns,
                    head_num,
                    head_dim,
                    key_conv,
                    BranchType.K,
                    sha_base_attn_node_list.lora_k,
                )
                value_conv = self.mha2sha_optim._lora_extension.attach_single_lora_adaptor(
                    info_dict["value"],
                    ns,
                    head_num,
                    head_dim,
                    value_conv,
                    BranchType.V,
                    sha_base_attn_node_list.lora_v,
                )

            if self.mha2sha_optim.handle_rope_ops:
                cos_node = info_dict["rope_cos_model_input"]
                sin_node = info_dict["rope_sin_model_input"]
                key_conv = self.mha2sha_optim._rope_extension.create_llama_rope_node(
                    key_conv, cos_node, sin_node, head_dim, BranchType.K
                )

            self.key_groups_list.append(key_conv)
            self.value_groups_list.append(value_conv)

        if self.mha2sha_optim.handle_past_key_value and self.mha2sha_optim.mha_conv:
            # Pre-transpose K, V for LLM-conv model and concate past_key and value to new key and value
            self.post_process_kv_for_rope(info_dict, ns)

    def post_process_kv_for_rope(self, info_dict, ns):
        """
        Post process k input and q input for rope, reshape k to [B, 1, D, L] and reshape v to [B, 1, L, D].
        create self.key_groups_list_to_return and self.value_groups_list_to_return: list of k, v for model
        output.

        :param self.key_groups_list: [B, 1, L, D]
        :param self.value_groups_list: [B, D, 1, L]

        :update self.key_groups_list: [B, 1, D, L]
        :update self.value_groups_list: [B, 1, L, D]
        :update self.key_groups_list_to_return: past_key model output [B, 1, D, L]
        :update self.value_groups_list_to_return: past_value model output [B, D, 1, L]
        """
        key_groups_list_temp = []
        value_groups_list_temp = []
        # add_past_key_value_for_llama_sha expects v in [B, 1, L, D] or [B, D, 1, L] is NCHW aligned
        self.key_groups_list_to_return = []
        self.value_groups_list_to_return = self.value_groups_list

        for kv_head_num, (key_node, value_node) in enumerate(
            zip(self.key_groups_list, self.value_groups_list)
        ):
            _, _, propose_sha_key_name, propose_sha_value_name = sha_node_name_basis(ns, kv_head_num)

            # Transpose K: [B, 1, L, D] -> [B, 1, D, L]
            key_transpose = self.mha2sha_optim._op_factory.get_transpose_op(
                key_node, [0, 1, 3, 2], propose_sha_key_name + "_Transpose"
            )
            self.mha2sha_optim.model.graph.node.append(key_transpose)
            self.key_groups_list_to_return.append(key_transpose)  # return post transpose pre concat keys

            # Transpose V: [B, D, 1, L] -> [B, 1, L, D]
            value_transpose = self.mha2sha_optim._op_factory.get_transpose_op(
                value_node, [0, 2, 3, 1], propose_sha_value_name + "_Transpose"
            )
            self.mha2sha_optim.model.graph.node.append(value_transpose)

            past_key_inp = None
            past_value_inp = None
            # Concate past key input and past value output
            if self.mha2sha_optim.kv_cache:
                past_key_inp = info_dict["key"]["past_input_name"]
                past_value_inp = info_dict["value"]["past_input_name"]
                # concat_past_key_value_input expects key_inp: [B, 1, D, L]; value_inp = [B, 1, L, D]
                key_transpose, value_transpose = (
                    self.mha2sha_optim._past_kv_extension.concat_past_key_value_input(
                        past_key_inp,
                        past_value_inp,
                        key_transpose,
                        value_transpose,
                        kv_head_num,
                    )
                )
            self.past_value_inp = past_value_inp

            key_groups_list_temp.append(key_transpose)
            value_groups_list_temp.append(value_transpose)

        self.key_groups_list = key_groups_list_temp
        self.value_groups_list = value_groups_list_temp

    def create_sha_convs_with_gqa_rope(
        self,
        info_dict,
        ns,
        head_num,
        head_dim,
        query_matmul_inp,
        key_matmul_inp,
        value_matmul_inp,
        sha_encoding_name_dict,
    ):
        return self.mha2sha_optim._mha_conv_extension.create_sha_conv_with_rope(
            info_dict,
            ns,
            head_num,
            head_dim,
            query_matmul_inp,
            key_matmul_inp,
            value_matmul_inp,
            sha_encoding_name_dict,
        )
