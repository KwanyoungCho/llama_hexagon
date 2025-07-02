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
"""Helping functions for mha2sha optimizer"""

from enum import Enum

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.logger import (
    log_warning,
)

from .onnx import (
    get_initializer_mappings,
    get_node_by_input_name,
    get_node_by_output_name,
    get_node_mappings,
    get_value_info_proto_mapping,
)


class BranchType(Enum):
    Q = 1
    K = 2
    V = 3


class ExtensionTypes(str, Enum):
    BASE_ATTN = "base_attn"
    GQA = "gqa"
    LORA = "lora"
    PAST_KEY_VALUE = "past_key_value"
    ROPE = "rope"


def sha_node_name_basis(attn_num, head_num):
    """
    attn_num: attention number in pattern.
    head_num: head number in mha.
    """
    propose_sha_name = f"attn_{attn_num}_head_{head_num}"
    propose_sha_query_name = propose_sha_name + "_query"
    propose_sha_key_name = propose_sha_name + "_key"
    propose_sha_value_name = propose_sha_name + "_value"

    return (
        propose_sha_name,
        propose_sha_query_name,
        propose_sha_key_name,
        propose_sha_value_name,
    )


def update_all_mapping_dicts(optimizer):
    """Helper function to update mappings to nodes.
    Updates all the mapping dictionaries such as `get_initializer_by_name`. These need
    to be updated as nodes are added to the graph and are not yet know.
    """
    optimizer.get_node_by_node_name = get_node_mappings(optimizer.model)
    optimizer.get_initializer_by_name = get_initializer_mappings(optimizer.model)
    optimizer.get_initializer_idx_by_name = {
        n.name: idx for idx, n in enumerate(optimizer.model.graph.initializer)
    }
    optimizer.get_value_info_by_name = get_value_info_proto_mapping(optimizer.model)

    optimizer.get_node_by_input_name = get_node_by_input_name(optimizer.model)
    optimizer.get_node_by_output_name = get_node_by_output_name(optimizer.model)

    optimizer.node_name_mapping_dict = {}
    optimizer.tensor_name_set = {node_op for node in optimizer.model.graph.node for node_op in node.output}
    optimizer.tensor_name_set.update(
        *(set([node_inp for node_inp in node.input]) for node in optimizer.model.graph.node)
    )

    optimizer.mha_model_input_names = [_input.name for _input in optimizer.model.graph.input]
    optimizer.mha_model_output_names = [_output.name for _output in optimizer.model.graph.output]
    optimizer.mha_model_input_names_index_dict = {
        _input.name: i for i, _input in enumerate(optimizer.model.graph.input)
    }


def get_head_num_and_dims(reshape_input_shape, inter_transpose, transpose_after_reshape):
    # conv -> [B, head_num*head_dim, 1, seq_len]/[B, C, H, W]
    number_of_heads = 0
    if inter_transpose:
        permute_axis = inter_transpose.attribute[0].ints
        if len(permute_axis) == 3:
            if permute_axis[1] == 1:
                # permute -> [B,  C, seq_len]
                # reshape -> [B, head_num, head_dim, seq_len]
                number_of_heads = reshape_input_shape[-3]
                head_dim = reshape_input_shape[-2]
            elif permute_axis[2] == 1:
                # permute -> [B, seq_len, C]
                # reshape -> [B, seq_len, head_num, head_dim] or [B, head_num, head_dim, seq_len]
                if permute_axis[1] == 2:
                    if transpose_after_reshape:
                        transpose_after_reshape_permute_axis = transpose_after_reshape.attribute[0].ints
                        if len(transpose_after_reshape_permute_axis) == 3:
                            # [seq_len, head_num, head_dim] -> [head_num, seq_len, head_dim]
                            if transpose_after_reshape_permute_axis == [1, 0, 2]:
                                number_of_heads = reshape_input_shape[-2]
                                head_dim = reshape_input_shape[-1]
                            else:
                                raise ValueError(
                                    f"Unexpected permute axis {transpose_after_reshape_permute_axis} of transpose after "
                                    f"reshape in Q branch. Failed to calculate number of heads"
                                )
                        elif len(transpose_after_reshape_permute_axis) == 4:
                            # [B, seq_len, head_num, head_dim] -> [B, head_num, seq_len, head_dim]
                            if transpose_after_reshape_permute_axis == [0, 2, 1, 3]:
                                number_of_heads = reshape_input_shape[-2]
                                head_dim = reshape_input_shape[-1]
                            else:
                                raise ValueError(
                                    f"Unexpected permute axis {transpose_after_reshape_permute_axis} of transpose after "
                                    f"reshape in Q branch. Failed to calculate number of heads"
                                )
                        else:
                            raise ValueError(
                                "Only 3D or 4D input supported for transpose after reshape in Q branch."
                                "Failed to calculate number of heads"
                            )
                    else:
                        # [B, head_num, seq_len, head_dim]
                        number_of_heads = reshape_input_shape[-3]
                        head_dim = reshape_input_shape[-1]
                else:
                    raise ValueError(
                        f"Unexpected permute axis {permute_axis} of transpose before reshape in Q branch. "
                        f"Failed to calculate number of heads"
                    )
            else:
                raise ValueError(
                    f"Unexpected permute axis {permute_axis} of transpose before reshape in Q branch. "
                    f"Failed to calculate number of heads"
                )

        elif len(permute_axis) == 4:
            if permute_axis[1] == 1:
                # permute -> [B,  C, 1, seq_len]
                # reshape -> [B, head_num, head_dim, seq_len]
                number_of_heads = reshape_input_shape[-3]
                head_dim = reshape_input_shape[-2]
            elif permute_axis[2] == 1:
                # permute -> [B, seq_len, C, 1]/[B, 1, C, seq_len]
                # reshape -> [B, seq_len, head_num, head_dim] or [B, head_num, head_dim, seq_len]
                if permute_axis[1] == 2:  # Case [B, 1, C, seq_len] -> [B, head_num, head_dim, seq_len]
                    number_of_heads = reshape_input_shape[-3]
                    head_dim = reshape_input_shape[-2]
                elif permute_axis[3] == 2:  # Case [B, seq_len, dim, 1] -> [B, seq_len, head_num, head_dim]
                    number_of_heads = reshape_input_shape[-2]
                    head_dim = reshape_input_shape[-1]
                else:
                    raise ValueError(f"Got Transpose({permute_axis}) -> Reshape({reshape_input_shape}) pattern,\
            and expect when transposing C to dim=2 will have H in NCHW transposed to dim = 1 or 3. Can't determine head_num.")
            elif permute_axis[3] == 1:
                # permute -> [B, seq_len, 1, C]
                # reshape -> [B, seq_len, head_num, head_dim]
                if transpose_after_reshape:
                    transpose_after_reshape_permute_axis = transpose_after_reshape.attribute[0].ints
                    if transpose_after_reshape_permute_axis == [0, 2, 1, 3]:
                        number_of_heads = reshape_input_shape[-2]
                        head_dim = reshape_input_shape[-1]
                    else:
                        raise ValueError(
                            f"Unexpected permute axis {transpose_after_reshape_permute_axis} of transpose after"
                            f" reshape in Q branch. Failed to calculate number of heads"
                        )
                else:
                    # [B, head_num, seq_len, head_dim]
                    number_of_heads = reshape_input_shape[-3]
                    head_dim = reshape_input_shape[-1]
            else:
                raise ValueError(f"expecting permute C in NCHW to axis 1, 2, or 3, but got {permute_axis}")
        else:
            raise ValueError(
                "Only 3D or 4D input supported for transpose before reshape in Q branch."
                "Failed to calculate number of heads"
            )
    else:
        if transpose_after_reshape:
            transpose_after_reshape_permute_axis = transpose_after_reshape.attribute[0].ints
            if len(transpose_after_reshape_permute_axis) == 3:
                # [seq_len, head_num, head_dim] -> [head_num, seq_len, head_dim]
                if transpose_after_reshape_permute_axis == [1, 0, 2]:
                    number_of_heads = reshape_input_shape[-2]
                    head_dim = reshape_input_shape[-1]
                else:
                    raise ValueError(
                        f"Unexpected permute axis {transpose_after_reshape_permute_axis} of transpose after"
                        f" reshape in Q branch. Failed to calculate number of heads"
                    )

            elif len(transpose_after_reshape_permute_axis) == 4:
                # [B, seq_len, head_num, head_dim] -> [B, head_num, seq_len, head_dim]
                if transpose_after_reshape_permute_axis == [0, 2, 1, 3]:
                    number_of_heads = reshape_input_shape[-2]
                    head_dim = reshape_input_shape[-1]

                elif transpose_after_reshape_permute_axis == [0, 1, 3, 2]:
                    # [B, head_num, head_dim, seq_len] -> [B, head_num, seq_len, head_dim]
                    number_of_heads = reshape_input_shape[-3]
                    head_dim = reshape_input_shape[-2]
                else:
                    raise ValueError(
                        f"Unexpected permute axis {transpose_after_reshape_permute_axis} of transpose after"
                        f" reshape in Q branch. Failed to calculate number of heads"
                    )
            else:
                raise ValueError(
                    "Only 3D or 4D input supported for transpose after reshape in Q branch."
                    "Failed to calculate number of heads"
                )
        else:
            # reshape -> [B, head_num, head_dim, seq_len]
            number_of_heads = reshape_input_shape[-3]  # [B, head_num, head_dim, seq_len]
            head_dim = reshape_input_shape[-2]

    if number_of_heads > 64:
        log_warning(
            f'Number of heads identified "{number_of_heads}" is too large! Reconsider checking the model.'
        )

    if number_of_heads < 1:
        raise ValueError(f"Invalid number of heads {number_of_heads} computed")

    return number_of_heads, head_dim
