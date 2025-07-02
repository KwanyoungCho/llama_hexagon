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
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

from onnx import numpy_helper
from onnx.onnx_pb import NodeProto

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.encoding_mapper_utils import (
    NodeMappingDict,
    create_activation_node_mapping_dict,
    update_sha_tensor_to_node_mapping_dict,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.onnx import (
    NodeNotFoundError,
    get_least_commom_ancestor_with_verified_pathway,
    get_mul_value,
    get_next_node_up_based_on_cond,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.utils import (
    BranchType,
    ExtensionTypes,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.logger import (
    log_debug,
    log_warning,
)

mha2sha_hf_model_optimizer = Any  # Causes circular import

LORA_BRANCH_PREFIX = [ExtensionTypes.LORA + "_" + qkv for qkv in ["q", "k", "v"]]
LORA_ACTIVATION_ENCODING_KEYS = [
    "lora_b",
    "lora_alpha",
    "lora_add",
]
LORA_PARAM_ENCODING_KEYS = ["lora_b_param"]


def list_field():
    return field(default_factory=list)


@dataclass
class LoraNode:
    """Defines nodes to record for lora models."""

    lora_a: Optional[Union[NodeProto, list[NodeProto]]] = list_field() 
    lora_b: Optional[Union[NodeProto, list[NodeProto]]] = list_field()
    lora_add: Optional[Union[NodeProto, list[NodeProto]]] = list_field()
    lora_alpha: Optional[Union[NodeProto, list[NodeProto]]] = list_field() 
    base_linear: Optional[NodeProto] = None
    lora_alpha_index: Union[int, list[int]] = list_field()
    lora_alpha_value: Union[float, list[float]] = list_field()
    is_proj_end: bool = False

    def __str__(self):
        # useful for printing and debuging
        return (
            f"LoraNode(\n"
            f"    lora_a: '{self.lora_a.name if self.lora_a else None}',\n"
            f"    lora_b: '{self.lora_b.name if self.lora_b else None}',\n"
            f"    lora_add: '{self.lora_add.name if self.lora_add else None}',\n"
            f"    lora_alpha: '{self.lora_alpha.name if self.lora_alpha else None}',\n"
            f"    base_linear: '{self.base_linear.name if self.base_linear else None}'\n"
            f")"
        )

    def __repr__(self):
        return str(self)


def create_branch_lora_encoding_mapping_dict(lora_node, prefix):
    """create Dict[str, NodeMappingDict] for lora nodes."""
    return {
        f"{prefix}_lora_b": create_activation_node_mapping_dict(
            input_1_name=lora_node.lora_b.input[0], output_name=lora_node.lora_b.output[0]
        ),
        f"{prefix}_lora_add": create_activation_node_mapping_dict(
            lora_node.lora_add.input[0], lora_node.lora_add.input[1], lora_node.lora_add.output[0]
        ),
        f"{prefix}_lora_alpha": create_activation_node_mapping_dict(
            lora_node.lora_alpha.input[0], lora_node.lora_alpha.input[1], lora_node.lora_alpha.output[0]
        ),
        f"{prefix}_lora_b_param": NodeMappingDict(
            mha_mapping_name_list=["mha_param_name"],
            sha_mapping_name_list=["sha_param_name"],
            mapping_name_dict={"mha_param_name": lora_node.lora_b.input[1], "sha_param_name": None},
        ),
    }


def create_lora_encoding_mapping_dict(info_dict: dict):
    """
    Create a Dict[str, NodeMappingDict] for lora nodes which will be used in EncodingMapper.
    :param mha_rope_dict: Collected MHA rope input outpue tensor names in optimizer.
    :return LoraEncodingMappingDict:
    """
    lora_dict = {}
    m = {"lora_q": "query", "lora_k": "key", "lora_v": "value"}
    for prefix in LORA_BRANCH_PREFIX:
        lora_nodes = info_dict[m[prefix]].get("lora_nodes", [])
        for lora_node in lora_nodes:
            enc = create_branch_lora_encoding_mapping_dict(lora_node, prefix)
            if enc:
                lora_dict.update(enc)

    return lora_dict


def update_lora_sha_encoding_name_to_lora_encoding_mapping_dict(
    lora_encoding_mapping_dict, sha_base_attn_node_list
):
    """
    Update sha LORA encoding names to EncodingMappingDict.
    :param lora_encoding_mapping_dict: encoding_mapping_dict.lora: Dict[str, NodeMappingDict]
    :param q_sha_lora_node: A name dict for sha nodes created when optimizer split and create nodes for sha
    :param k_sha_lora_node: A name dict for sha nodes created when optimizer split and create nodes for sha
    :param v_sha_lora_node: A name dict for sha nodes created when optimizer split and create nodes for sha
    """
    for lora_branch_prefix in LORA_BRANCH_PREFIX:
        for lora_node in getattr(sha_base_attn_node_list, lora_branch_prefix):
            # Check lora b for exsitance. Empty lora_b list should retuen False.
            if lora_node.lora_b:
                # Handle activation encodings
                for lora_node_name in LORA_ACTIVATION_ENCODING_KEYS:
                    update_sha_tensor_to_node_mapping_dict(
                        node_mapping_dict=lora_encoding_mapping_dict[
                            lora_branch_prefix + "_" + lora_node_name
                        ],
                        sha_node_list=getattr(lora_node, lora_node_name),
                    )

                for lora_node_name in LORA_PARAM_ENCODING_KEYS:
                    update_sha_tensor_to_node_mapping_dict(
                        node_mapping_dict=lora_encoding_mapping_dict[
                            lora_branch_prefix + "_" + lora_node_name
                        ],
                        sha_node_list=getattr(lora_node, "lora_b"),
                    )


class LoraExtension:
    """Extenstion helpers for mha2sha_optimzer to bridge Morpheus pipeline code base and v1.0.0 release."""

    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """
        self.mha2sha_optim = mha2sha_optim
        self.map_lora_encoding = True

    def reset_sha_encoding_name_list(self):
        """
        Reset mha sha names for LORA tensors.
        """
        self.map_lora_encoding = True
        self.q_sha_lora_node = LoraNode()
        self.k_sha_lora_node = LoraNode()
        self.v_sha_lora_node = LoraNode()

    def update_sha_lora_node(self, branch_type, lora_b, lora_alpha, lora_add):
        """Update sha node"""
        if branch_type == BranchType.Q:
            sha_lora_node = self.q_sha_lora_node
        elif branch_type == BranchType.K:
            sha_lora_node = self.k_sha_lora_node
        elif branch_type == BranchType.V:
            sha_lora_node = self.v_sha_lora_node

        sha_lora_node.lora_b.append(lora_b)
        sha_lora_node.lora_add.append(lora_add)
        sha_lora_node.lora_alpha.append(lora_alpha)

    def _get_node_constant(self, input: str) -> float:
        """
        Return constant or initializer 1D array of given input or
        return array(-1) if input is not a constant
        """
        if input in self.mha2sha_optim.get_initializer_by_name:
            arr = numpy_helper.to_array(self.mha2sha_optim.get_initializer_by_name[input])
            return arr.flatten()[0]
        elif (node := self.mha2sha_optim.get_node_by_output_name[input]).op_type == "Constant":
            arr = numpy_helper.to_array(node.attribute[0].t)
            return arr.flatten()[0]
        else:
            return None

    def _capture_lora_structure(
        self, add_lora: NodeProto, prev_node: Optional[NodeProto] = None, base_linear: Optional[NodeProto] = None
    ):
        matmul_op_type = "Conv"
        allow_types = {"Transpose", "Reshape"}

        def _find_upstream_linear(start_tensor, allow_types_ext=set()):
            try:
                start_node = self.mha2sha_optim.get_node_by_output_name[start_tensor]
            except KeyError:
                raise NodeNotFoundError
            return get_next_node_up_based_on_cond(
                start_node=start_node,
                get_node_by_output_name=self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == matmul_op_type,
                node_end_search_cond=lambda n: n.op_type not in allow_types | allow_types_ext,
            )

        def _find_upstream_mul(start_tensor, allow_types_ext=set()):
            try:
                start_node = self.mha2sha_optim.get_node_by_output_name[start_tensor]
            except KeyError:
                raise NodeNotFoundError

            return get_next_node_up_based_on_cond(
                start_node=start_node,
                get_node_by_output_name=self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == "Mul",
                node_end_search_cond=lambda n: n.op_type not in allow_types | allow_types_ext,
            )

        if not base_linear:
            # add_lora's inputs should come from two different nodes
            prev_node_0 = self.mha2sha_optim.get_node_by_output_name.get(add_lora.input[0], None)
            prev_node_1 = self.mha2sha_optim.get_node_by_output_name.get(add_lora.input[1], None)
            if prev_node_0 is None or prev_node_1 is None:
                return
            if prev_node_0.name == prev_node_1.name:
                return

            try:
                base_linear = _find_upstream_linear(add_lora.input[0])
                lora_b = _find_upstream_linear(add_lora.input[1])

            except NodeNotFoundError:
                return

            base_linear_init = self.mha2sha_optim.get_initializer_by_name[base_linear.input[1]]
            lora_b_init = self.mha2sha_optim.get_initializer_by_name[lora_b.input[1]]

            # base linear input should have higher rank then lora_b input
            # ONNX conv weight is [O, I, kH, kW]
            if base_linear_init.dims[1] <= lora_b_init.dims[1]:
                base_linear, lora_b = lora_b, base_linear

        else:
            # Lora V3, handle all branches
            # Skip the base linear branch
            if add_lora.input[0] == prev_node.output[0]:
                lora_branch_input = add_lora.input[1]
            else:
                lora_branch_input = add_lora.input[0]

            try:
                lora_b = _find_upstream_linear(lora_branch_input)
            except NodeNotFoundError:
                # An "Add" that is not from a LoRA branch. Break here
                return

        try:
            lora_a = _find_upstream_linear(lora_b.input[0], {"Mul"})

        except NodeNotFoundError:
            return

        lca = get_least_commom_ancestor_with_verified_pathway(
            self.mha2sha_optim.get_node_by_output_name[lora_a.input[0]],
            self.mha2sha_optim.get_node_by_output_name[base_linear.input[0]],
            self.mha2sha_optim,
            pathway_nodes_verifier=lambda n: n.op_type in allow_types,
        )
        if lca is None:
            log_debug(
                "Ignoring candiate lora structure, reason: LCA not found\n    "
                f"base_linear:'{base_linear.name}', "
                f"lora_a:'{lora_a.name}', "
                f"lora_b:'{lora_b.name}'"
            )
            return

        lora_alpha = _find_upstream_mul(lora_b.input[0])
        if lora_alpha:
            if lora_alpha_value := self._get_node_constant(lora_alpha.input[0]) is None:
                if lora_alpha_value := self._get_node_constant(lora_alpha.input[1]) is None:
                    # Potential LoraV3 case where input to Mul is a `Reshape` node
                    if self.mha2sha_optim.get_node_by_output_name[lora_alpha.input[0]].op_type == "Reshape":
                        lora_alpha_index = 0
                    elif self.mha2sha_optim.get_node_by_output_name[lora_alpha.input[1]].op_type == "Reshape":
                        lora_alpha_index = 1
                    else:
                        raise ValueError(
                            "At-least one of the input to LoRA branch Mul should be a constant or output of a Reshape node"
                        )
                else:
                    lora_alpha_index = 1
            else:
                lora_alpha_index = 0

        lora_node = LoraNode(
            base_linear=base_linear,
            lora_a=lora_a,
            lora_b=lora_b,
            lora_add=add_lora,
            lora_alpha=lora_alpha,
            lora_alpha_index=lora_alpha_index,
            lora_alpha_value=lora_alpha_value,
        )

        return lora_node

    def verify_and_capture_lora_structure(self, add_lora):
        """
        Find base linear, lora_b and lora_b.
            x-----------
            |           |
            |         lora_a
            |           |
        base linear   Mul alpha
            |           |
            |         lora_b
            |           |
            Add--------/

        the lora-structure will be verified.
        If the verification failed, None will be returned.

        combine "lora verification" and "lora nodes matching" into one function
        since the majority of their logic is identical.
        """

        all_lora_nodes = []
        prev_node = None
        base_linear = None

        while lora_node := self._capture_lora_structure(add_lora, prev_node, base_linear):
            all_lora_nodes.append(lora_node)

            try:
                next_node = self.mha2sha_optim.get_node_by_input_name[add_lora.output[0]][0]
                if next_node.op_type != "Add":
                    break

                prev_node = add_lora
                add_lora = next_node
                base_linear = lora_node.base_linear
            except (KeyError, IndexError):
                break

        if all_lora_nodes:
            all_lora_nodes[-1].is_proj_end = True

        return all_lora_nodes

    def get_qkv_info(
        self, qk_matmul_node: NodeProto, qkv_matmul_node: NodeProto, lora_nodes: list[LoraNode]
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
        assert self.mha2sha_optim.mha_conv, "Support mha-conv lora at the moment"

        proj_op_type = "Conv" if self.mha2sha_optim.mha_conv else "MatMul"

        # - when lora exists, q_conv_candidate may be lora's conv or base_linear's conv
        #   but that dosen't matter
        # - when lora dosen't exist, q_conv_candidate is the base_linear's conv
        qkv_info = {}
        qkv_branch_types = [BranchType.Q, BranchType.K, BranchType.V]

        qkv_to_tensor_map = {
            BranchType.Q: self.mha2sha_optim.get_node_by_output_name[qk_matmul_node.input[0]],
            BranchType.K: self.mha2sha_optim.get_node_by_output_name[qk_matmul_node.input[1]],
            BranchType.V: self.mha2sha_optim.get_node_by_output_name[qkv_matmul_node.input[1]],
        }

        for branch_type in qkv_branch_types:
            conv = get_next_node_up_based_on_cond(
                qkv_to_tensor_map[branch_type],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == proj_op_type,
            )

            # Match the matmul to branch, could be lora_b or base linear
            # EAFP
            try:
                match = [node for node in lora_nodes if conv == node.lora_b][0]
                base_linear = match.base_linear
            except IndexError:
                # Base linear
                base_linear = conv

            branch_lora_nodes = [node for node in lora_nodes if base_linear == node.base_linear]

            if branch_lora_nodes:
                proj_end = [node for node in branch_lora_nodes if node.is_proj_end][0]
                conv_initializer = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(base_linear)
                conv_bias_initializer = self.mha2sha_optim._mha_conv_extension.get_conv_bias_in_OI(
                    base_linear
                )
                qkv_info[branch_type] = {
                    "matmul_node": base_linear,
                    "matmul_init": conv_initializer,
                    "matmul_init_bias": conv_bias_initializer,
                    "lora_nodes": branch_lora_nodes,
                    "proj_end": proj_end.lora_add,
                }

            else:
                log_warning(f"No lora adapter(s) found on branch {branch_type.name}")

        return qkv_info[BranchType.Q], qkv_info[BranchType.K], qkv_info[BranchType.V]

    def attach_single_lora_adaptor(
        self,
        branch_info_dict,
        ns,
        head_num,
        head_dim,
        proj_output,  # qkv output
        branch_type,
        sha_lora_nodes,
    ):
        """Attach lora adpator from lora out to one of qkv conv"""

        for i, lora_node in enumerate(branch_info_dict["lora_nodes"]):
            # Skip over transpose-reshape after lora A

            lora_alpha_inp = lora_node.lora_alpha.input[lora_node.lora_alpha_index]

            # Either of the two conditions holds
            # 1. Lora alpha is from input
            # 2. Lora alpha is from a Reshape node (LoraV3 case)
            # Then simply add the node
            lora_alpha = lora_node.lora_alpha 

            lora_b_input = lora_alpha.output[0]
            lora_b_init = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(lora_node.lora_b)

            lora_b_conv = self.mha2sha_optim._mha_conv_extension.create_single_conv(
                lora_b_init,
                ns,
                head_num,
                head_dim,
                lora_b_input,
                suffix=f"lora_b",
                branch_type=branch_type,
            )

            proj_output = self.mha2sha_optim._op_factory.get_add_op(proj_output, lora_b_conv)
            self.mha2sha_optim.model.graph.node.append(proj_output)

            sha_lora_nodes[i].lora_alpha.append(lora_alpha)
            sha_lora_nodes[i].lora_b.append(lora_b_conv)
            sha_lora_nodes[i].lora_add.append(proj_output)

            # Save sha lora tensors
            # Mul output
            self.mha2sha_optim.sha_lora_tensor_names.add(lora_alpha.output[0])

            # Lora A Conv activation and weights
            self.mha2sha_optim.sha_lora_tensor_names.add(lora_node.lora_a.output[0])
            self.mha2sha_optim.sha_lora_tensor_names.add(lora_node.lora_a.input[1])
            try:  # Bias
                self.mha2sha_optim.sha_lora_tensor_names.add(lora_node.lora_a.input[2])
            except IndexError:
                pass

            # Lora B Conv activation and weights
            self.mha2sha_optim.sha_lora_tensor_names.add(lora_b_conv.output[0])
            self.mha2sha_optim.sha_lora_tensor_names.add(lora_b_conv.input[1])
            try:  # Bias
                self.mha2sha_optim.sha_lora_tensor_names.add(lora_b_conv.input[2])
            except IndexError:
                pass

            # LoRA V3 Gather indices
            try:
                lora_alpha_inp_node = self.mha2sha_optim.get_node_by_output_name[lora_alpha_inp]
                if lora_alpha_inp_node.op_type == "Reshape":
                    reshape_input = self.mha2sha_optim.get_node_by_output_name.get(
                        lora_alpha_inp_node.input[0]
                    )
                    if reshape_input and reshape_input.op_type == "Gather":
                        self.mha2sha_optim.sha_lora_tensor_names.add(reshape_input.input[1])
            except KeyError:
                pass

        return proj_output

    def attach_lora_adapters(
        self, info_dict, ns, head_num, head_dim, sha_base_attn_node_list, query_inp, key_inp, value_inp
    ):
        """Attach lora adpator from lora out to qkv conv"""

        def _get(branch, inp):
            m = {"query": BranchType.Q, "key": BranchType.K, "value": BranchType.V}
            m2 = {"query": "lora_q", "key": "lora_k", "value": "lora_v"}
            return self.attach_single_lora_adaptor(
                info_dict[branch],
                ns,
                head_num,
                head_dim,
                inp,
                m[branch],
                getattr(sha_base_attn_node_list, m2[branch]),
            )

        query_inp = _get("query", query_inp)
        key_inp = _get("key", key_inp)
        value_inp = _get("value", value_inp)

        return query_inp, key_inp, value_inp

    def create_sha_conv_lora_rope(
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

    def create_sha_conv_lora(
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
        return self.mha2sha_optim.create_sha(
            info_dict,
            ns,
            head_num,
            head_dim,
            query_matmul_inp,
            key_matmul_inp,
            value_matmul_inp,
            sha_encoding_name_dict,
        )
