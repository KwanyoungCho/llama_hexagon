# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import re
import onnx
import math
import itertools
import transformers
import numpy as np

from typing import List
from transformers.integrations.ggml import generate_float_encodings, FLOAT_BW
from onnx import helper, numpy_helper, TensorProto, ModelProto

MODEL_TYPE_TO_ARCH = {"llama": "LlamaForCausalLM"}

MODEL_TYPE_TO_TOKENIZER = {"llama": transformers.LlamaTokenizerFast}

GGUF_TO_ONNX_TENSOR = {
    "llama": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj.MatMul",
        "ffn_down": "mlp.down_proj.MatMul",
        "ffn_gate": "mlp.gate_proj.MatMul",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "attn.q_proj.MatMul",
        "attn_v": "attn.v_proj.MatMul",
        "attn_k": "attn.k_proj.MatMul",
        "attn_output": "attn.o_proj.MatMul",
        "output.weight": "lm_head.MatMul.weight",
        "output_norm": "model.layers.{max_block}.final_norm_layernorm",
    }
}

ONNX_TENSOR_NAME_STRINGS = {
    "llama_final_layernorm": "final_norm_layernorm",
    "llama_SkipLayerNorm": "SkipLayerNorm",
    "llama_LayerNorm": "LayerNorm",
    "llama_name_seqlens_k": "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0",
    "llama_GroupQueryAttention": "GroupQueryAttention",
    "llama_qkv_proj": "qkv_proj",
    "model_mask_reduce_sum": "/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum",
    "model_mask_sub": "/model/attn_mask_reformat/attn_mask_subgraph/Sub"
}

def update_symbolic_shape_with_value(model: ModelProto, model_config: dict, batch_size: int):
    """
        Utility function to update an onnx model inputs, outputs and intermediate value_info shapes.
        All symbolic shapes for the above-mentioned tensors are replaced with constant values.
    """
    batch_size_value = batch_size
    sequence_length_value = 1
    total_sequence_length_value = model_config["max_position_embeddings"]
    past_sequence_length_value = model_config["max_position_embeddings"]
    model_input_shape_dict = {
        "batch_size": batch_size_value,
        "sequence_length": sequence_length_value,
        "total_sequence_length": total_sequence_length_value,
        "past_sequence_length": past_sequence_length_value
    }

    def update_symbolic_values(value_infos):
        for value_info in value_infos:
            if value_info.type.HasField("tensor_type") and value_info.type.tensor_type.HasField("shape"):
                for cur_dim in value_info.type.tensor_type.shape.dim:
                    if cur_dim.HasField("dim_param") and cur_dim.dim_param in model_input_shape_dict.keys():
                        dict_key = cur_dim.dim_param
                        cur_dim.Clear()
                        cur_dim.dim_value = model_input_shape_dict[dict_key]

    update_symbolic_values(model.graph.input)
    update_symbolic_values(model.graph.output)
    update_symbolic_values(model.graph.value_info)


def decompose_layernorms(model: ModelProto, model_config: dict):
    """
        Utility function to decompose the layer norm ops in ort-genai onnx model with constituent onnx ops.
        1. SkipSimplifiedLayerNormalization -> Elementwise Add + SimplifiedLayerNormalization
        2. SimplifiedLayerNormalization -> x/Sqrt(RedSum(x^2) + eps) * gamma_weight
    """
    hidden_size = model_config["hidden_size"]
    # Update SkipSimplifiedLayerNorm = Elementwise Add and SimplifiedLayerNorm
    for node in model.graph.node:
        if node.op_type == "SkipSimplifiedLayerNormalization":
            # Get input output weight info
            name_sln = node.name
            is_last_layernorm = False
            # Final SkipSLN only has one output
            if ONNX_TENSOR_NAME_STRINGS["llama_final_layernorm"] in name_sln:
                is_last_layernorm = True
            input_name_data_0 = node.input[0]
            input_name_data_1 = node.input[1]
            weight_name_scale = node.input[2]
            eps_data = node.attribute[0].f
            output_name_data_0 = node.output[0]
            # Set Node Names to be added to graph
            split_name_skip_sln = name_sln.split(ONNX_TENSOR_NAME_STRINGS["llama_SkipLayerNorm"])
            node_name_elementwise_add = split_name_skip_sln[0] + "elementwise_add"
            node_name_simplifiedlayernorm = split_name_skip_sln[0] + "LayerNorm"
            if not is_last_layernorm:
                output_name_data_3 = node.output[3]
                node_elementwise_add = helper.make_node("Add", name=node_name_elementwise_add, inputs=[input_name_data_0, input_name_data_1],
                                                        outputs=[output_name_data_3])
                node_simplifiedlayernorm = helper.make_node("SimplifiedLayerNormalization", name=node_name_simplifiedlayernorm, inputs=[output_name_data_3, weight_name_scale],
                                                            outputs=[output_name_data_0])
            else:
                output_name_elemwise_add = node_name_elementwise_add + "/output_0"
                node_elementwise_add = helper.make_node("Add", name=node_name_elementwise_add, inputs=[input_name_data_0, input_name_data_1],
                                                        outputs=[output_name_elemwise_add])
                node_simplifiedlayernorm = helper.make_node("SimplifiedLayerNormalization", name=node_name_simplifiedlayernorm, inputs=[output_name_elemwise_add, weight_name_scale],
                                                            outputs=[output_name_data_0])
                elemwise_add_vi = helper.make_tensor_value_info(output_name_elemwise_add, TensorProto.FLOAT,
                                                                ["batch_size", "sequence_length", hidden_size])
                model.graph.value_info.extend([elemwise_add_vi])
            # Add required attributes to SimplfiedLayerNorm
            eps_attribute = helper.make_attribute("epsilon", eps_data)
            axis_attribute = helper.make_attribute("axis", -1)
            node_simplifiedlayernorm.attribute.extend([eps_attribute, axis_attribute])
            model.graph.node.extend([node_elementwise_add, node_simplifiedlayernorm])
            # Remove Node
            model.graph.node.remove(node)
    # Decompose SimplifiedLayerNormalization into constituent onnx ops
    for node in model.graph.node:
        if node.op_type == 'SimplifiedLayerNormalization':
            # Get input output weight info
            name_sln = node.name
            input_name_data = node.input[0]
            weight_name_scale = node.input[1]
            eps_data = node.attribute[0].f
            output_name_data = node.output[0]
            # Set Node Names to be added to graph
            split_name_sln = name_sln.split(ONNX_TENSOR_NAME_STRINGS["llama_LayerNorm"])
            node_name_pow_value = split_name_sln[0] + 'pow_value'
            node_name_eps_value = split_name_sln[0] + 'eps_value'
            node_name_square_inp = split_name_sln[0] + 'square_inp'
            node_name_reduce_mean = split_name_sln[0] + 'reduce_mean'
            node_name_add_eps = split_name_sln[0] + 'add_eps'
            node_name_sqrt = split_name_sln[0] + 'sqrt'
            node_name_elementwise_div = split_name_sln[0] + 'elementwise_div'
            node_name_elementwise_mul_gamma = split_name_sln[0] + 'elementwise_mul_gamma'
            # Set constants Input
            pow_value = np.array([2], dtype=np.float32)
            eps_value = np.array([eps_data], dtype=np.float32)
            # Set Output names for nodes
            out_name_pow_value = node_name_pow_value + '/output_0'
            out_name_eps_value = node_name_eps_value + '/output_0'
            out_name_square_inp = node_name_square_inp + '/output_0'
            out_name_reduce_mean = node_name_reduce_mean + '/output_0'
            out_name_add_eps = node_name_add_eps + '/output_0'
            out_name_sqrt = node_name_sqrt + '/output_0'
            out_name_elementwise_div = node_name_elementwise_div + '/output_0'
            # Create Decomposed RMSNorm Nodes
            node_constant_pow_value = helper.make_node('Constant', inputs=[], outputs=[out_name_pow_value],
                                                       value=helper.make_tensor(
                                                           name=node_name_pow_value,
                                                           data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')],
                                                           dims=pow_value.shape,
                                                           vals=pow_value.flatten()))
            node_constant_eps_value = helper.make_node('Constant', inputs=[], outputs=[out_name_eps_value],
                                                       value=helper.make_tensor(
                                                           name=node_name_eps_value,
                                                           data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')],
                                                           dims=eps_value.shape,
                                                           vals=eps_value.flatten()))
            node_square_inp = helper.make_node('Pow', name=node_name_square_inp, inputs=[input_name_data, out_name_pow_value],
                                               outputs=[out_name_square_inp])
            node_reduce_mean = helper.make_node('ReduceMean', name=node_name_reduce_mean, inputs=[out_name_square_inp], axes=[2],
                                                outputs=[out_name_reduce_mean])
            node_add_eps = helper.make_node('Add', name=node_name_add_eps, inputs=[out_name_reduce_mean, out_name_eps_value],
                                            outputs=[out_name_add_eps])
            node_sqrt = helper.make_node('Sqrt', name=node_name_sqrt, inputs=[out_name_add_eps],
                                         outputs=[out_name_sqrt])
            node_elementwise_div = helper.make_node('Div', name=node_name_elementwise_div, inputs=[input_name_data, out_name_sqrt],
                                                    outputs=[out_name_elementwise_div])
            node_elementwise_mul_gamma = helper.make_node('Mul', name=node_name_elementwise_mul_gamma, inputs=[out_name_elementwise_div, weight_name_scale],
                                                          outputs=[output_name_data])
            # Create intermediate output tensors and add to graph: Value_Info
            vi_square_inp = helper.make_tensor_value_info(out_name_square_inp, TensorProto.FLOAT,
                                                          ['batch_size', 'sequence_length', hidden_size])
            vi_reduce_mean = helper.make_tensor_value_info(out_name_reduce_mean, TensorProto.FLOAT,
                                                           ['batch_size', 'sequence_length', 1])
            vi_add_eps = helper.make_tensor_value_info(out_name_add_eps, TensorProto.FLOAT,
                                                       ['batch_size', 'sequence_length', 1])
            vi_sqrt = helper.make_tensor_value_info(out_name_sqrt, TensorProto.FLOAT,
                                                    ['batch_size', 'sequence_length', 1])
            vi_elementwise_div = helper.make_tensor_value_info(out_name_elementwise_div, TensorProto.FLOAT,
                                                               ['batch_size', 'sequence_length', hidden_size])
            model.graph.value_info.extend([vi_square_inp, vi_reduce_mean, vi_add_eps,
                                           vi_sqrt, vi_elementwise_div])
            # Add created nodes to graph
            model.graph.node.extend([node_constant_pow_value, node_constant_eps_value,
                                     node_square_inp, node_reduce_mean, node_add_eps, node_sqrt,
                                     node_elementwise_div, node_elementwise_mul_gamma])
            # Remove Node
            model.graph.node.remove(node)


def decompose_gqa(model: ModelProto, model_config: dict, batch_size: int):
    """
        Utility function that decomposes ort-genai generated GroupQueryAttention(GQA) op into constituent ops.
    """
    def update_o_proj_input(model: ModelProto, node_name: str, output_tensor_name: str):
        o_proj_node_name = node_name.replace("reshape_attnv", "o_proj/MatMul")

        for node in model.graph.node:
            if node.name == o_proj_node_name:
                node.input[0] = output_tensor_name

    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_attention_heads"]
    total_seq_len = model_config["max_position_embeddings"]
    num_kv_heads = model_config["num_key_value_heads"] if "num_key_value_heads" in model_config else num_heads
    head_dim = hidden_size // num_heads
    n_rep = num_heads // num_kv_heads
    cur_seq_len = 1
    input_name_seqlens_k = ONNX_TENSOR_NAME_STRINGS["llama_name_seqlens_k"]

    # change dtype of input_ids to int32
    for vi in model.graph.input:
        if vi.name == "input_ids":
            vi.type.tensor_type.elem_type = 6

    # Update kv cache
    for vi in itertools.islice(model.graph.input, 2, None):
        vi_name = vi.name
        vi_name_parts = vi_name.split(".")
        new_vi_name = f"past_{vi_name_parts[-1]}_{vi_name_parts[-2]}_in"
        new_vi_dims = [batch_size, num_kv_heads, head_dim, total_seq_len-cur_seq_len] if vi_name_parts[-1] == "key" else [batch_size, num_kv_heads, total_seq_len-cur_seq_len, head_dim]

        vi.name = new_vi_name
        for idx, dim in enumerate(vi.type.tensor_type.shape.dim):
            dim.dim_value = new_vi_dims[idx]

        for node in model.graph.node:
            if node.op_type == "GroupQueryAttention":
                for idx, node_input in enumerate(node.input):
                    if node_input == vi_name:
                        node.input[idx] = new_vi_name

    for vi in itertools.islice(model.graph.output, 1, None):
        if "present" in vi.name:
            vi_name = vi.name
            vi_name_parts = vi_name.split(".")
            new_vi_name = f"past_{vi_name_parts[-1]}_{vi_name_parts[-2]}_out"
            new_vi_dims = [batch_size, num_kv_heads, head_dim, cur_seq_len] if vi_name_parts[-1] == "key" else [batch_size, num_kv_heads, cur_seq_len, head_dim]

            vi.name = new_vi_name
            for idx, dim in enumerate(vi.type.tensor_type.shape.dim):
                dim.dim_value = new_vi_dims[idx]

            for node in model.graph.node:
                if node.op_type == "GroupQueryAttention":
                    for idx, node_input in enumerate(node.output):
                        if node_input == vi_name:
                            node.output[idx] = new_vi_name

    # Remove exisitng attention mask from input
    input_name_attn_mask = "attention_mask"
    for model_inp in model.graph.input:
        if model_inp.name == input_name_attn_mask:
            model.graph.input.remove(model_inp)

    # Create new attention mask
    attn_mask_new_shape = [batch_size, 1, cur_seq_len, total_seq_len]
    new_attn_mask_input = helper.make_tensor_value_info(input_name_attn_mask, TensorProto.FLOAT, attn_mask_new_shape)

    # Create new branch of ops for 4d attention mask
    node_name_mask_equal = "/model/attn_mask_reformat/attn_mask_subgraph/Equal"
    node_name_mask_equal_cast = "/model/attn_mask_reformat/attn_mask_subgraph/Equal/Cast"
    node_name_mask_squeeze = "/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum/Squeeze"

    initializer_name_mask_equal = "zero_scalar"
    initializer_name_axes = "reduce_axes"

    output_name_mask_equal = node_name_mask_equal + "/output_0"
    output_name_mask_equal_cast = node_name_mask_equal_cast + "/output_0"
    output_name_mask_squeeze = node_name_mask_squeeze + "/output_0"

    initializer_mask_equal = helper.make_tensor(name=initializer_name_mask_equal,
                                                data_type=TensorProto.FLOAT,
                                                dims=[],
                                                vals=[0.0])

    initializer_reducesum_axes = helper.make_tensor(name=initializer_name_axes,
                                                    data_type=TensorProto.INT64,
                                                    dims=[1],
                                                    vals=[-1])

    node_mask_equal = helper.make_node("Equal",
                                       name=node_name_mask_equal,
                                       inputs=[input_name_attn_mask, initializer_name_mask_equal],
                                       outputs=[output_name_mask_equal])

    node_mask_equal_cast = helper.make_node("Cast",
                                            name=node_name_mask_equal_cast,
                                            inputs=[output_name_mask_equal],
                                            outputs=[output_name_mask_equal_cast],
                                            to=TensorProto.INT64)

    reducesum_output_name = None
    reducesum_axes_name = None

    for node in model.graph.node:
        if node.op_type == "ReduceSum" and node.name == ONNX_TENSOR_NAME_STRINGS["model_mask_reduce_sum"]:
            reducesum_axes_name = node.input[1]
            node.input[0] = output_name_mask_equal_cast
            node.input[1] = initializer_name_axes
            new_attr = helper.make_attribute("keepdims", 0)
            node.attribute.append(new_attr)
            reducesum_output_name = node.output[0]

        elif node.op_type == "Sub" and node.name == ONNX_TENSOR_NAME_STRINGS["model_mask_sub"]:
            node.input[0] = output_name_mask_squeeze

    node_mask_squeeze = helper.make_node("Squeeze",
                                         name=node_name_mask_squeeze,
                                         inputs=[reducesum_output_name, reducesum_axes_name],
                                         outputs=[output_name_mask_squeeze])

    for vi in model.graph.value_info:
        if vi.name == reducesum_output_name:
            model.graph.value_info.remove(vi)

    vi_reduce_sum = helper.make_tensor_value_info(reducesum_output_name, TensorProto.INT64, [batch_size, 1, cur_seq_len])

    model.graph.input.insert(1, new_attn_mask_input)
    model.graph.initializer.extend([initializer_mask_equal, initializer_reducesum_axes])
    model.graph.value_info.extend([vi_reduce_sum])
    model.graph.node.extend([node_mask_equal, node_mask_equal_cast, node_mask_squeeze])

    # Create Inputs for RoPE
    pos_ids_cos_name = "position_ids_cos"
    pos_ids_cos_shape = [batch_size, 1, cur_seq_len, head_dim//2]
    pos_ids_cos_input = helper.make_tensor_value_info(pos_ids_cos_name, TensorProto.FLOAT, pos_ids_cos_shape)

    pos_ids_sin_name = "position_ids_sin"
    pos_ids_sin_shape = [batch_size, 1, cur_seq_len, head_dim//2]
    pos_ids_sin_input = helper.make_tensor_value_info(pos_ids_sin_name, TensorProto.FLOAT, pos_ids_sin_shape)

    model.graph.input.insert(2, pos_ids_cos_input)
    model.graph.input.insert(3, pos_ids_sin_input)

    for init in model.graph.initializer:
        if init.name == "sin_cache" or init.name == "cos_cache":
            model.graph.initializer.remove(init)

    unsqueeze_cache_axis = np.array([1], dtype=np.int64)
    shape_q = np.array([batch_size, -1, num_heads, head_dim], dtype=np.int64)
    shape_k = np.array([batch_size, -1, num_kv_heads, head_dim], dtype=np.int64)
    shape_v = np.array([batch_size, -1, num_kv_heads, head_dim], dtype=np.int64)
    div_value = np.array([math.sqrt(head_dim)], dtype=np.float32)
    unsqueeze_axis = np.array([2], dtype=np.int64)
    num_rep_k = np.array([batch_size, num_kv_heads, n_rep, head_dim, total_seq_len], dtype=np.int64)
    num_rep_v = np.array([batch_size, num_kv_heads, n_rep, total_seq_len, head_dim], dtype=np.int64)
    shape_current_k = np.array([-1, num_kv_heads * n_rep, head_dim, total_seq_len], dtype=np.int64)
    shape_current_v = np.array([-1, num_kv_heads * n_rep, total_seq_len, head_dim], dtype=np.int64)
    shape_attnv = np.array([-1, cur_seq_len, num_heads * head_dim], dtype=np.int64)

    # Iterate over nodes
    for node in model.graph.node:
        # check for matmul that has q, k, v weights packed into one tensor
        if node.op_type == 'GroupQueryAttention':
            # Get input output weight info
            name_gqa = node.name
            input_name_q = node.input[0]
            input_name_k = node.input[1]
            input_name_v = node.input[2]

            input_name_past_key = node.input[3]
            input_name_past_value = node.input[4]
            output_name_gqa = node.output[0]
            output_name_present_key = node.output[1]
            output_name_present_value = node.output[2]

            # Set Node Names to be added to graph
            split_name_gqa = name_gqa.split(ONNX_TENSOR_NAME_STRINGS["llama_GroupQueryAttention"])
            node_name_shape_q = split_name_gqa[0] + 'shape_q'
            node_name_shape_k = split_name_gqa[0] + 'shape_k'
            node_name_shape_v = split_name_gqa[0] + 'shape_v'
            node_name_div_value = split_name_gqa[0] + 'div_value'
            node_name_unsqueeze_axis = split_name_gqa[0] + 'unsqueeze_axis'
            node_name_num_rep_k = split_name_gqa[0] + 'num_rep_k'
            node_name_num_rep_v = split_name_gqa[0] + 'num_rep_v'
            node_name_shape_current_k = split_name_gqa[0] + 'shape_current_k'
            node_name_shape_current_v = split_name_gqa[0] + 'shape_current_v'
            node_name_shape_attnv = split_name_gqa[0] + 'shape_attnv'
            node_name_reshape_q = split_name_gqa[0] + 'q_reshape'
            node_name_reshape_k = split_name_gqa[0] + 'k_reshape'
            node_name_reshape_v = split_name_gqa[0] + 'v_reshape'
            node_name_transpose_q = split_name_gqa[0] + 'q_transpose'
            node_name_transpose_k = split_name_gqa[0] + 'k_transpose'
            node_name_transpose_v = split_name_gqa[0] + 'v_transpose'
            node_name_slice_q1 = split_name_gqa[0] + 'slice_q1'
            node_name_slice_q2 = split_name_gqa[0] + 'slice_q2'
            node_name_mul_q1_sin = split_name_gqa[0] + 'mul_q1_sin'
            node_name_mul_q1_cos = split_name_gqa[0] + 'mul_q1_cos'
            node_name_mul_q2_sin = split_name_gqa[0] + 'mul_q2_sin'
            node_name_mul_q2_cos = split_name_gqa[0] + 'mul_q2_cos'
            node_name_add_q1_sin_q2_cos = split_name_gqa[0] + 'add_q1_sin_q2_cos'
            node_name_sub_q1_cos_q2_sin = split_name_gqa[0] + 'sub_q1_cos_q2_sin'
            node_name_concat_q1_q2 = split_name_gqa[0] + 'concat_q1_q2'
            node_name_slice_k1 = split_name_gqa[0] + 'slice_k1'
            node_name_slice_k2= split_name_gqa[0] + 'slice_k2'
            node_name_mul_k1_sin = split_name_gqa[0] + 'mul_k1_sin'
            node_name_mul_k1_cos = split_name_gqa[0] + 'mul_k1_cos'
            node_name_mul_k2_sin = split_name_gqa[0] + 'mul_k2_sin'
            node_name_mul_k2_cos = split_name_gqa[0] + 'mul_k2_cos'
            node_name_add_k1_sin_k2_cos = split_name_gqa[0] + 'add_k1_sin_k2_cos'
            node_name_sub_k1_cos_k2_sin = split_name_gqa[0] + 'sub_k1_cos_k2_sin'
            node_name_concat_k1_k2 = split_name_gqa[0] + 'concat_k1_k2'
            node_name_transpose_rope_k = split_name_gqa[0] + 'k_rope_transpose'
            node_name_concat_k = split_name_gqa[0] + 'k_concat'
            node_name_concat_v = split_name_gqa[0] + 'v_concat'
            node_name_unsqueeze_k = split_name_gqa[0] + 'unsqueeze_k'
            node_name_expand_k = split_name_gqa[0] + 'expand_k'
            node_name_reshape_current_k = split_name_gqa[0] + 'current_k_reshape'
            node_name_unsqueeze_v = split_name_gqa[0] + 'unsqueeze_v'
            node_name_expand_v = split_name_gqa[0] + 'expand_v'
            node_name_reshape_current_v = split_name_gqa[0] + 'current_v_reshape'
            node_name_matmul_qk = split_name_gqa[0] + 'matmul_qk'
            node_name_add_qk_attn_mask = split_name_gqa[0] + 'add_qk_attn_mask'
            node_name_div_qk = split_name_gqa[0] + 'div_qk'
            node_name_softmax_qk = split_name_gqa[0] + 'softmax_qk'
            node_name_matmul_attnv = split_name_gqa[0] + 'matmul_attnv'
            node_name_transpose_attnv = split_name_gqa[0] + 'transpose_attnv'
            node_name_reshape_attnv = split_name_gqa[0] + 'reshape_attnv'

            # Set Output names for nodes
            out_name_shape_q = node_name_shape_q + '/output_0'
            out_name_shape_k = node_name_shape_k + '/output_0'
            out_name_shape_v = node_name_shape_v + '/output_0'
            out_name_div_value = node_name_div_value + '/output_0'
            out_name_unsqueeze_axis = node_name_unsqueeze_axis + '/output_0'

            out_name_num_rep_k = node_name_num_rep_k + '/output_0'
            out_name_num_rep_v = node_name_num_rep_v + '/output_0'
            out_name_shape_current_k = node_name_shape_current_k + '/output_0'
            out_name_shape_current_v = node_name_shape_current_v + '/output_0'
            out_name_shape_attnv = node_name_shape_attnv + '/output_0'
            out_name_reshape_q = node_name_reshape_q + '/output_0'
            out_name_reshape_k = node_name_reshape_k + '/output_0'
            out_name_reshape_v = node_name_reshape_v + '/output_0'

            out_name_transpose_q = node_name_transpose_q + '/output_0'
            out_name_transpose_k = node_name_transpose_k + '/output_0'
            out_name_transpose_v = node_name_transpose_v + '/output_0'

            out_name_slice_q1 = node_name_slice_q1 + '/output_0'
            out_name_slice_q2 = node_name_slice_q2 + '/output_0'
            out_name_mul_q1_sin = node_name_mul_q1_sin + '/output_0'
            out_name_mul_q1_cos = node_name_mul_q1_cos + '/output_0'
            out_name_mul_q2_sin = node_name_mul_q2_sin + '/output_0'
            out_name_mul_q2_cos = node_name_mul_q2_cos + '/output_0'
            out_name_add_q1_sin_q2_cos = node_name_add_q1_sin_q2_cos + '/output_0'
            out_name_sub_q1_cos_q2_sin = node_name_sub_q1_cos_q2_sin + '/output_0'
            out_name_rope_q = node_name_concat_q1_q2 + '/output_0'

            out_name_slice_k1 = node_name_slice_k1 + '/output_0'
            out_name_slice_k2 = node_name_slice_k2 + '/output_0'
            out_name_mul_k1_sin = node_name_mul_k1_sin + '/output_0'
            out_name_mul_k1_cos = node_name_mul_k1_cos + '/output_0'
            out_name_mul_k2_sin = node_name_mul_k2_sin + '/output_0'
            out_name_mul_k2_cos = node_name_mul_k2_cos + '/output_0'
            out_name_add_k1_sin_k2_cos = node_name_add_k1_sin_k2_cos + '/output_0'
            out_name_sub_k1_cos_k2_sin = node_name_sub_k1_cos_k2_sin + '/output_0'
            out_name_rope_k = node_name_concat_k1_k2 + '/output_0'
            out_name_combined_key = node_name_transpose_rope_k + '/output_0'
            out_name_combined_value = node_name_transpose_v + '/output_0'
            out_name_unsqueeze_k = node_name_unsqueeze_k + '/output_0'
            out_name_expand_k = node_name_expand_k + '/output_0'
            out_name_reshape_current_k = node_name_reshape_current_k + '/output_0'
            out_name_unsqueeze_v = node_name_unsqueeze_v + '/output_0'
            out_name_expand_v = node_name_expand_v + '/output_0'
            out_name_reshape_current_v = node_name_reshape_current_v + '/output_0'

            out_name_matmul_qk = node_name_matmul_qk + '/output_0'
            out_name_add_qk_attn_mask = node_name_add_qk_attn_mask + '/output_0'
            out_name_div_qk = node_name_div_qk + '/output_0'
            out_name_softmax_qk = node_name_softmax_qk + '/output_0'
            out_name_matmul_attnv = node_name_matmul_attnv + '/output_0'
            out_name_transpose_attnv = node_name_transpose_attnv + '/output_0'
            out_name_reshape_attnv = node_name_reshape_attnv + '/output_0'

            # Create Decomposed GQA Nodes
            # Reshape Nodes
            node_constant_shape_q = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_q],
                                                     value=helper.make_tensor(
                                                         name=node_name_shape_q,
                                                         data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                         dims=shape_q.shape,
                                                         vals=shape_q.flatten()))
            node_constant_shape_k = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_k],
                                                     value=helper.make_tensor(
                                                         name=node_name_shape_k,
                                                         data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                         dims=shape_k.shape,
                                                         vals=shape_k.flatten()))
            node_constant_shape_v = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_v],
                                                     value=helper.make_tensor(
                                                         name=node_name_shape_v,
                                                         data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                         dims=shape_v.shape,
                                                         vals=shape_v.flatten()))
            node_reshape_q = helper.make_node('Reshape', name=node_name_reshape_q, inputs=[input_name_q, out_name_shape_q],
                                              outputs=[out_name_reshape_q])
            node_reshape_k = helper.make_node('Reshape', name=node_name_reshape_k, inputs=[input_name_k, out_name_shape_k],
                                              outputs=[out_name_reshape_k])
            node_reshape_v = helper.make_node('Reshape', name=node_name_reshape_v, inputs=[input_name_v, out_name_shape_v],
                                              outputs=[out_name_reshape_v])

            # Transpose Nodes
            node_transpose_q = helper.make_node('Transpose', name=node_name_transpose_q, inputs=[out_name_reshape_q],
                                                outputs=[out_name_transpose_q], perm=[0, 2, 1, 3])
            node_transpose_k = helper.make_node('Transpose', name=node_name_transpose_k, inputs=[out_name_reshape_k],
                                                outputs=[out_name_transpose_k], perm=[0, 2, 1, 3])
            node_transpose_v = helper.make_node('Transpose', name=node_name_transpose_v, inputs=[out_name_reshape_v],
                                                outputs=[output_name_present_value], perm=[0, 2, 1, 3])

            # RoPE Nodes Q
            node_name_constant_slice = split_name_gqa[0] + 'slice'
            node_name_start_1 = node_name_constant_slice + '/start_1'
            node_name_start_2 = node_name_constant_slice + '/start_2'
            node_name_end_1 = node_name_constant_slice + '/end_1'
            node_name_end_2 = node_name_constant_slice + '/end_2'
            node_name_slice_axes = node_name_constant_slice + '/axes'
            node_name_slice_steps = node_name_constant_slice + '/steps'

            out_name_start_1 = node_name_start_1 + '/output_0'
            out_name_start_2 = node_name_start_2 + '/output_0'
            out_name_end_1 = node_name_end_1 + '/output_0'
            out_name_end_2 = node_name_end_2 + '/output_0'
            out_name_slice_axes = node_name_slice_axes + '/output_0'
            out_name_slice_steps = node_name_slice_steps + '/output_0'

            node_constant_start_1 = helper.make_node('Constant', inputs=[], outputs=[out_name_start_1],
                                                     value=helper.make_tensor(
                                                         name=node_name_start_1,
                                                         data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                         dims=[1],
                                                         vals=[0]))
            node_constant_end_1 = helper.make_node('Constant', inputs=[], outputs=[out_name_end_1],
                                                   value=helper.make_tensor(
                                                       name=node_name_start_1,
                                                       data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                       dims=[1],
                                                       vals=[head_dim//2]))
            node_constant_start_2 = helper.make_node('Constant', inputs=[], outputs=[out_name_start_2],
                                                     value=helper.make_tensor(
                                                         name=node_name_start_2,
                                                         data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                         dims=[1],
                                                         vals=[head_dim//2]))
            node_constant_end_2 = helper.make_node('Constant', inputs=[], outputs=[out_name_end_2],
                                                   value=helper.make_tensor(
                                                       name=node_name_start_2,
                                                       data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                       dims=[1],
                                                       vals=[head_dim]))
            node_constant_slice_axes = helper.make_node('Constant', inputs=[], outputs=[out_name_slice_axes],
                                                        value=helper.make_tensor(
                                                            name=node_name_slice_axes,
                                                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                            dims=[1],
                                                            vals=[-1]))
            node_constant_slice_steps = helper.make_node('Constant', inputs=[], outputs=[out_name_slice_steps],
                                                         value=helper.make_tensor(
                                                             name=node_name_slice_steps,
                                                             data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                             dims=[1],
                                                             vals=[1]))
            node_slice_q1 = helper.make_node('Slice', name=node_name_slice_q1,
                                             inputs=[out_name_transpose_q,
                                                     out_name_start_1,
                                                     out_name_end_1,
                                                     out_name_slice_axes,
                                                     out_name_slice_steps],
                                             outputs=[out_name_slice_q1])
            node_slice_q2 = helper.make_node('Slice', name=node_name_slice_q2,
                                             inputs=[out_name_transpose_q,
                                                     out_name_start_2,
                                                     out_name_end_2,
                                                     out_name_slice_axes,
                                                     out_name_slice_steps],
                                             outputs=[out_name_slice_q2])
            node_mul_q1_sin = helper.make_node('Mul', name=node_name_mul_q1_sin, inputs=[out_name_slice_q1, pos_ids_sin_name],
                                               outputs=[out_name_mul_q1_sin])
            node_mul_q1_cos = helper.make_node('Mul', name=node_name_mul_q1_cos, inputs=[out_name_slice_q1, pos_ids_cos_name],
                                               outputs=[out_name_mul_q1_cos])
            node_mul_q2_sin = helper.make_node('Mul', name=node_name_mul_q2_sin, inputs=[out_name_slice_q2, pos_ids_sin_name],
                                               outputs=[out_name_mul_q2_sin])
            node_mul_q2_cos = helper.make_node('Mul', name=node_name_mul_q2_cos, inputs=[out_name_slice_q2, pos_ids_cos_name],
                                               outputs=[out_name_mul_q2_cos])
            node_add_q1_sin_q2_cos = helper.make_node('Add', name=node_name_add_q1_sin_q2_cos, inputs=[out_name_mul_q1_sin, out_name_mul_q2_cos],
                                                      outputs=[out_name_add_q1_sin_q2_cos])
            node_sub_q1_cos_q2_sin = helper.make_node('Sub', name=node_name_sub_q1_cos_q2_sin, inputs=[out_name_mul_q1_cos, out_name_mul_q2_sin],
                                                      outputs=[out_name_sub_q1_cos_q2_sin])
            node_concat_q1_q2 = helper.make_node('Concat', name=node_name_concat_q1_q2, inputs=[out_name_sub_q1_cos_q2_sin, out_name_add_q1_sin_q2_cos],
                                                 axis=-1, outputs=[out_name_rope_q])

            # RoPE Nodes K
            node_slice_k1 = helper.make_node('Slice', name=node_name_slice_k1,
                                             inputs=[out_name_transpose_k,
                                                     out_name_start_1,
                                                     out_name_end_1,
                                                     out_name_slice_axes,
                                                     out_name_slice_steps],
                                             outputs=[out_name_slice_k1])
            node_slice_k2 = helper.make_node('Slice', name=node_name_slice_k2,
                                             inputs=[out_name_transpose_k,
                                                     out_name_start_2,
                                                     out_name_end_2,
                                                     out_name_slice_axes,
                                                     out_name_slice_steps],
                                             outputs=[out_name_slice_k2])
            node_mul_k1_sin = helper.make_node('Mul', name=node_name_mul_k1_sin, inputs=[out_name_slice_k1, pos_ids_sin_name],
                                               outputs=[out_name_mul_k1_sin])
            node_mul_k1_cos = helper.make_node('Mul', name=node_name_mul_k1_cos, inputs=[out_name_slice_k1, pos_ids_cos_name],
                                               outputs=[out_name_mul_k1_cos])
            node_mul_k2_sin = helper.make_node('Mul', name=node_name_mul_k2_sin, inputs=[out_name_slice_k2, pos_ids_sin_name],
                                               outputs=[out_name_mul_k2_sin])
            node_mul_k2_cos = helper.make_node('Mul', name=node_name_mul_k2_cos, inputs=[out_name_slice_k2, pos_ids_cos_name],
                                               outputs=[out_name_mul_k2_cos])
            node_add_k1_sin_k2_cos = helper.make_node('Add', name=node_name_add_k1_sin_k2_cos, inputs=[out_name_mul_k1_sin, out_name_mul_k2_cos],
                                                      outputs=[out_name_add_k1_sin_k2_cos])
            node_sub_k1_cos_k2_sin = helper.make_node('Sub', name=node_name_sub_k1_cos_k2_sin, inputs=[out_name_mul_k1_cos, out_name_mul_k2_sin],
                                                      outputs=[out_name_sub_k1_cos_k2_sin])
            node_concat_k1_k2 = helper.make_node('Concat', name=node_name_concat_k1_k2, inputs=[out_name_sub_k1_cos_k2_sin, out_name_add_k1_sin_k2_cos],
                                                 axis=-1, outputs=[out_name_rope_k])
            node_transpose_rope_k = helper.make_node('Transpose', name=node_name_transpose_rope_k, inputs=[out_name_rope_k],
                                                     outputs=[output_name_present_key], perm=[0, 1, 3, 2])

            # Concat Nodes
            node_concat_k = helper.make_node('Concat', name=node_name_concat_k, inputs=[input_name_past_key, output_name_present_key],
                                             axis=-1, outputs=[out_name_combined_key])
            node_concat_v = helper.make_node('Concat', name=node_name_concat_v, inputs=[input_name_past_value, output_name_present_value],
                                             axis=-2, outputs=[out_name_combined_value])
            # K Repetition Node
            node_constant_unsqueeze_axis = helper.make_node('Constant', inputs=[], outputs=[out_name_unsqueeze_axis],
                                                            value=helper.make_tensor(
                                                                name=node_name_unsqueeze_axis,
                                                                data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                                dims=unsqueeze_axis.shape,
                                                                vals=unsqueeze_axis.flatten()))
            node_constant_num_rep_k = helper.make_node('Constant', inputs=[], outputs=[out_name_num_rep_k],
                                                       value=helper.make_tensor(
                                                           name=node_name_num_rep_k,
                                                           data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                           dims=num_rep_k.shape,
                                                           vals=num_rep_k.flatten().tolist()))
            node_constant_shape_current_k = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_current_k],
                                                             value=helper.make_tensor(
                                                                 name=node_name_shape_current_k,
                                                                 data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                                 dims=shape_current_k.shape,
                                                                 vals=shape_current_k.flatten()))
            node_unsqueeze_k = helper.make_node('Unsqueeze', name=node_name_unsqueeze_k, inputs=[out_name_combined_key, out_name_unsqueeze_axis],
                                                outputs=[out_name_unsqueeze_k])
            node_expand_k = helper.make_node('Expand', name=node_name_expand_k, inputs=[out_name_unsqueeze_k, out_name_num_rep_k],
                                             outputs=[out_name_expand_k])
            node_reshape_current_k = helper.make_node('Reshape', name=node_name_reshape_current_k, inputs=[out_name_expand_k, out_name_shape_current_k],
                                                      outputs=[out_name_reshape_current_k])
            # V Repetition Node
            node_constant_shape_current_v = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_current_v],
                                                             value=helper.make_tensor(
                                                                 name=node_name_shape_current_v,
                                                                 data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                                 dims=shape_current_v.shape,
                                                                 vals=shape_current_v.flatten()))
            node_unsqueeze_v = helper.make_node('Unsqueeze', name=node_name_unsqueeze_v, inputs=[out_name_combined_value, out_name_unsqueeze_axis],
                                                outputs=[out_name_unsqueeze_v])
            node_constant_num_rep_v = helper.make_node('Constant', inputs=[], outputs=[out_name_num_rep_v],
                                                       value=helper.make_tensor(
                                                           name=node_name_num_rep_v,
                                                           data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                           dims=num_rep_v.shape,
                                                           vals=num_rep_v.flatten().tolist()))
            node_expand_v = helper.make_node('Expand', name=node_name_expand_v, inputs=[out_name_unsqueeze_v, out_name_num_rep_v],
                                             outputs=[out_name_expand_v])
            node_reshape_current_v = helper.make_node('Reshape', name=node_name_reshape_current_v, inputs=[out_name_expand_v, out_name_shape_current_v],
                                                      outputs=[out_name_reshape_current_v])

            # Q * K'
            node_matmul_qk = helper.make_node('MatMul', name=node_name_matmul_qk, inputs=[out_name_rope_q, out_name_reshape_current_k],
                                              outputs=[out_name_matmul_qk])
            # Constant Div
            node_constant_div_qk = helper.make_node('Constant', inputs=[], outputs=[out_name_div_value],
                                                    value=helper.make_tensor(
                                                        name=node_name_div_value,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')],
                                                        dims=div_value.shape,
                                                        vals=div_value.flatten()))
            node_div_qk = helper.make_node('Div', name=node_name_div_qk, inputs=[out_name_matmul_qk, out_name_div_value],
                                           outputs=[out_name_div_qk])
            node_add_qk_attn_mask = helper.make_node('Add', name=node_name_add_qk_attn_mask, inputs=[out_name_div_qk, input_name_attn_mask],
                                                     outputs=[out_name_add_qk_attn_mask])

            # Softmax Node
            node_softmax_qk = helper.make_node('Softmax', name=node_name_softmax_qk, inputs=[out_name_add_qk_attn_mask],
                                               outputs=[out_name_softmax_qk])
            # Attn * V
            node_matmul_attnv = helper.make_node('MatMul', name=node_name_matmul_attnv, inputs=[out_name_softmax_qk, out_name_reshape_current_v],
                                                 outputs=[out_name_matmul_attnv])
            node_transpose_attnv = helper.make_node('Transpose', name=node_name_transpose_attnv, inputs=[out_name_matmul_attnv],
                                                    outputs=[out_name_transpose_attnv], perm=[0, 2, 1, 3])
            node_constant_shape_attnv = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_attnv],
                                                         value=helper.make_tensor(
                                                             name=node_name_shape_attnv,
                                                             data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                             dims=shape_attnv.shape,
                                                             vals=shape_attnv.flatten()))
            node_reshape_attnv = helper.make_node('Reshape', name=node_name_reshape_attnv, inputs=[out_name_transpose_attnv, out_name_shape_attnv],
                                                  outputs=[out_name_reshape_attnv])

            # Create intermediate output tensors and add to graph: Value_Info
            vi_reshape_q = helper.make_tensor_value_info(out_name_reshape_q, TensorProto.FLOAT,
                                                         [batch_size, 'sequence_length', shape_q[2].item(), shape_q[3].item()])
            vi_reshape_k = helper.make_tensor_value_info(out_name_reshape_k, TensorProto.FLOAT,
                                                         [batch_size, 'sequence_length', shape_k[2].item(), shape_k[3].item()])
            vi_reshape_v = helper.make_tensor_value_info(out_name_reshape_v, TensorProto.FLOAT,
                                                         [batch_size, 'sequence_length', shape_v[2].item(), shape_v[3].item()])
            vi_transpose_q = helper.make_tensor_value_info(out_name_transpose_q, TensorProto.FLOAT,
                                                           [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item()])
            vi_transpose_k = helper.make_tensor_value_info(out_name_transpose_k, TensorProto.FLOAT,
                                                           [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item()])
            vi_transpose_v = helper.make_tensor_value_info(out_name_transpose_v, TensorProto.FLOAT,
                                                           [batch_size, shape_v[2].item(), 'sequence_length',  shape_v[3].item()])
            vi_slice_q1 = helper.make_tensor_value_info(out_name_slice_q1, TensorProto.FLOAT,
                                                        [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2])
            vi_slice_q2 = helper.make_tensor_value_info(out_name_slice_q2, TensorProto.FLOAT,
                                                        [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2])
            vi_mul_q1_sin = helper.make_tensor_value_info(out_name_mul_q1_sin, TensorProto.FLOAT,
                                                          [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2])
            vi_mul_q1_cos = helper.make_tensor_value_info(out_name_mul_q1_cos, TensorProto.FLOAT,
                                                          [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2])
            vi_mul_q2_sin = helper.make_tensor_value_info(out_name_mul_q2_sin, TensorProto.FLOAT,
                                                          [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2])
            vi_mul_q2_cos = helper.make_tensor_value_info(out_name_mul_q2_cos, TensorProto.FLOAT,
                                                          [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2])
            vi_add_q1_sin_q2_cos = helper.make_tensor_value_info(out_name_add_q1_sin_q2_cos, TensorProto.FLOAT,
                                                                 [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2])
            vi_sub_q1_cos_q2_sin = helper.make_tensor_value_info(out_name_sub_q1_cos_q2_sin, TensorProto.FLOAT,
                                                                 [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2])
            vi_rope_q = helper.make_tensor_value_info(out_name_rope_q, TensorProto.FLOAT,
                                                      [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item()])
            vi_slice_k1 = helper.make_tensor_value_info(out_name_slice_k1, TensorProto.FLOAT,
                                                        [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2])
            vi_slice_k2 = helper.make_tensor_value_info(out_name_slice_k2, TensorProto.FLOAT,
                                                        [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2])
            vi_mul_k1_sin = helper.make_tensor_value_info(out_name_mul_k1_sin, TensorProto.FLOAT,
                                                          [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2])
            vi_mul_k1_cos = helper.make_tensor_value_info(out_name_mul_k1_cos, TensorProto.FLOAT,
                                                          [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2])
            vi_mul_k2_sin = helper.make_tensor_value_info(out_name_mul_k2_sin, TensorProto.FLOAT,
                                                          [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2])
            vi_mul_k2_cos = helper.make_tensor_value_info(out_name_mul_k2_cos, TensorProto.FLOAT,
                                                          [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2])
            vi_add_k1_sin_k2_cos = helper.make_tensor_value_info(out_name_add_k1_sin_k2_cos, TensorProto.FLOAT,
                                                                 [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2])
            vi_sub_k1_cos_k2_sin = helper.make_tensor_value_info(out_name_sub_k1_cos_k2_sin, TensorProto.FLOAT,
                                                                 [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2])
            vi_rope_k = helper.make_tensor_value_info(out_name_rope_k, TensorProto.FLOAT,
                                                      [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item()])
            vi_combined_key = helper.make_tensor_value_info(out_name_combined_key, TensorProto.FLOAT,
                                                            [batch_size, num_kv_heads, head_dim, total_seq_len])
            vi_combined_value = helper.make_tensor_value_info(out_name_combined_value, TensorProto.FLOAT,
                                                              [batch_size, num_kv_heads, total_seq_len, head_dim])
            vi_unsqueeze_k = helper.make_tensor_value_info(out_name_unsqueeze_k, TensorProto.FLOAT,
                                                           [batch_size, shape_k[2].item(), 1, shape_k[3].item(), total_seq_len])
            vi_expand_k = helper.make_tensor_value_info(out_name_expand_k, TensorProto.FLOAT,
                                                        [batch_size, shape_k[2].item(), n_rep, shape_k[3].item(), total_seq_len])
            vi_reshape_current_k = helper.make_tensor_value_info(out_name_reshape_current_k, TensorProto.FLOAT,
                                                                 [batch_size, shape_k[2].item() * n_rep, shape_k[3].item(), total_seq_len])
            vi_unsqueeze_v = helper.make_tensor_value_info(out_name_unsqueeze_v, TensorProto.FLOAT,
                                                           [batch_size, shape_v[2].item(), 1, total_seq_len, shape_v[3].item()])
            vi_expand_v = helper.make_tensor_value_info(out_name_expand_v, TensorProto.FLOAT,
                                                        [batch_size, shape_v[2].item(), n_rep,  total_seq_len, shape_v[3].item()])
            vi_reshape_current_v = helper.make_tensor_value_info(out_name_reshape_current_v, TensorProto.FLOAT,
                                                                 [batch_size, shape_v[2].item() * n_rep, total_seq_len, shape_v[3].item()])
            vi_matmul_qk = helper.make_tensor_value_info(out_name_matmul_qk, TensorProto.FLOAT,
                                                         [batch_size, shape_q[2].item(), 'sequence_length', total_seq_len])
            vi_add_qk_attn_mask = helper.make_tensor_value_info(out_name_add_qk_attn_mask, TensorProto.FLOAT,
                                                                [batch_size, shape_q[2].item(), 'sequence_length', total_seq_len])
            vi_div_qk = helper.make_tensor_value_info(out_name_div_qk, TensorProto.FLOAT,
                                                      [batch_size, shape_q[2].item(), 'sequence_length', total_seq_len])
            vi_softmax_qk = helper.make_tensor_value_info(out_name_softmax_qk, TensorProto.FLOAT,
                                                          [batch_size, shape_q[2].item(), 'sequence_length', total_seq_len])
            vi_matmul_attnv = helper.make_tensor_value_info(out_name_matmul_attnv, TensorProto.FLOAT,
                                                            [batch_size, shape_q[2].item(), 'sequence_length', shape_q[3].item()])
            vi_transpose_attnv = helper.make_tensor_value_info(out_name_transpose_attnv, TensorProto.FLOAT,
                                                               [batch_size, 'sequence_length', shape_q[2].item(), shape_q[3].item()])
            vi_reshape_attnv = helper.make_tensor_value_info(out_name_reshape_attnv, TensorProto.FLOAT,
                                                             [batch_size, 'sequence_length', shape_attnv[2].item()])
            gqa_vi = [vi for vi in model.graph.value_info if vi.name == output_name_gqa][0]
            model.graph.value_info.extend([vi_reshape_q, vi_reshape_k, vi_reshape_v,
                                           vi_transpose_q, vi_transpose_k, vi_transpose_v,
                                           vi_slice_q1, vi_slice_q2, vi_mul_q1_sin,
                                           vi_mul_q1_cos, vi_mul_q2_sin, vi_mul_q2_cos,
                                           vi_add_q1_sin_q2_cos, vi_sub_q1_cos_q2_sin, vi_rope_q,
                                           vi_slice_k1, vi_slice_k2, vi_mul_k1_sin, vi_mul_k1_cos,
                                           vi_mul_k2_sin, vi_mul_k2_cos,
                                           vi_add_k1_sin_k2_cos, vi_sub_k1_cos_k2_sin, vi_rope_k,
                                           vi_combined_key, vi_combined_value,
                                           vi_unsqueeze_k, vi_expand_k, vi_reshape_current_k,
                                           vi_unsqueeze_v, vi_expand_v, vi_reshape_current_v,
                                           vi_matmul_qk, vi_add_qk_attn_mask, vi_div_qk, vi_softmax_qk,
                                           vi_matmul_attnv, vi_transpose_attnv, vi_reshape_attnv])
            model.graph.value_info.remove(gqa_vi)

            # Add created nodes to graph
            model.graph.node.extend([node_constant_shape_q, node_constant_shape_k, node_constant_shape_v,
                                     node_reshape_q, node_reshape_k, node_reshape_v,
                                     node_transpose_q, node_transpose_k, node_transpose_v,
                                     node_constant_start_1, node_constant_start_2,
                                     node_constant_end_1, node_constant_end_2,
                                     node_constant_slice_steps, node_constant_slice_axes,
                                     node_slice_q1, node_slice_q2, node_mul_q1_sin, node_mul_q1_cos,
                                     node_mul_q2_sin, node_mul_q2_cos,
                                     node_add_q1_sin_q2_cos, node_sub_q1_cos_q2_sin, node_concat_q1_q2,
                                     node_slice_k1, node_slice_k2, node_mul_k1_sin, node_mul_k1_cos,
                                     node_mul_k2_sin, node_mul_k2_cos,
                                     node_add_k1_sin_k2_cos, node_sub_k1_cos_k2_sin, node_concat_k1_k2,
                                     node_concat_k, node_transpose_rope_k, node_concat_v,
                                     node_constant_unsqueeze_axis, node_constant_num_rep_k,
                                     node_constant_num_rep_v,
                                     node_constant_shape_current_k, node_constant_shape_current_v,
                                     node_unsqueeze_k, node_expand_k, node_reshape_current_k,
                                     node_unsqueeze_v, node_expand_v, node_reshape_current_v,
                                     node_matmul_qk, node_add_qk_attn_mask, node_constant_div_qk,
                                     node_div_qk, node_softmax_qk,
                                     node_matmul_attnv, node_transpose_attnv,
                                     node_constant_shape_attnv, node_reshape_attnv])
            # Remove GQA Node
            model.graph.node.remove(node)

            update_o_proj_input(model, node_name_reshape_attnv, out_name_reshape_attnv)


def unpack_qkv(model: ModelProto, model_config: dict):
    """
        Utility function to subdivide ort-genai generated combined QKV FullyConnected(FC).
        Combined QKV op is split into 3 FC Ops for Q,K and V respectively.
    """
    def update_gqa_inputs(model: ModelProto, node_name: str, output_tensor_name: str, node_type: str):
        gqa_node_name = node_name.replace(f"q_proj/{node_type}", "GroupQueryAttention")

        for node in model.graph.node:
            if node.name == gqa_node_name:
                node.input[0] = output_tensor_name
                node.input[1] = output_tensor_name.replace("q_proj", "k_proj")
                node.input[2] = output_tensor_name.replace("q_proj", "v_proj")

    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config["num_key_value_heads"] if "num_key_value_heads" in model_config else num_heads
    head_dim = hidden_size // num_heads
    n_q = num_heads * head_dim
    n_k = num_kv_heads * head_dim
    n_v = num_kv_heads * head_dim

    # Iterate over nodes
    for node in model.graph.node:

        # check for matmul that has q, k, v weights packed into one tensor
        if node.op_type == "MatMul" and "qkv_proj" in node.name:

            # get input, output and weight names
            name_qkv = node.name

            matmul_layer_idx = re.findall(r"layers\.\d+", name_qkv)[0].split(".")[-1]
            input_name_qkv = node.input[0]
            output_name_qkv = node.output[0]
            weight_name_qkv = node.input[1]

            # get weight initializer
            weight_init_qkv = [initializer for initializer in model.graph.initializer if weight_name_qkv == initializer.name][0]

            # get weight tensor
            weight_tensor_qkv = numpy_helper.to_array(weight_init_qkv)

            # split packed tensor into wq, wk, wv
            weight_tensor_q = np.copy(weight_tensor_qkv[:, : n_q])
            weight_tensor_k = np.copy(weight_tensor_qkv[:, n_q: n_q + n_k])
            weight_tensor_v = np.copy(weight_tensor_qkv[:, n_q + n_k: n_q + n_k + n_v])

            # remove packed tensor
            model.graph.initializer.remove(weight_init_qkv)

            # set wq, wk, wv tensor names
            split_w_qkv = weight_name_qkv.split(ONNX_TENSOR_NAME_STRINGS["llama_qkv_proj"])
            weight_name_q = split_w_qkv[0] + "q_proj" + split_w_qkv[1]
            weight_name_k = split_w_qkv[0] + "k_proj" + split_w_qkv[1]
            weight_name_v = split_w_qkv[0] + "v_proj" + split_w_qkv[1]

            # set names for q, k, v Matmul nodes
            split_name_qkv = name_qkv.split(ONNX_TENSOR_NAME_STRINGS["llama_qkv_proj"])
            node_name_q = split_name_qkv[0] + "q_proj" + split_name_qkv[1]
            node_name_k = split_name_qkv[0] + "k_proj" + split_name_qkv[1]
            node_name_v = split_name_qkv[0] + "v_proj" + split_name_qkv[1]

            # create initializers from split tensors
            init_q = numpy_helper.from_array(weight_tensor_q, weight_name_q)
            init_k = numpy_helper.from_array(weight_tensor_k, weight_name_k)
            init_v = numpy_helper.from_array(weight_tensor_v, weight_name_v)

            # add split tensors to initializers
            model.graph.initializer.extend([init_q, init_k, init_v])

            # set output names for split nodes
            out_name_q = node_name_q + "/output_0"
            out_name_k = node_name_k + "/output_0"
            out_name_v = node_name_v + "/output_0"

            # create split nodes
            node_matmul_q = helper.make_node("MatMul", name=node_name_q, inputs=[input_name_qkv, weight_name_q],
                                             outputs=[out_name_q])

            node_matmul_k = helper.make_node("MatMul", name=node_name_k, inputs=[input_name_qkv, weight_name_k],
                                             outputs=[out_name_k])

            node_matmul_v = helper.make_node("MatMul", name=node_name_v, inputs=[input_name_qkv, weight_name_v],
                                             outputs=[out_name_v])

            # create output tensors and add to graph
            q_vi = helper.make_tensor_value_info(out_name_q, TensorProto.FLOAT,
                                                 ["batch_size", "sequence_length", weight_tensor_q.shape[-1]])
            k_vi = helper.make_tensor_value_info(out_name_k, TensorProto.FLOAT,
                                                 ["batch_size", "sequence_length", weight_tensor_k.shape[-1]])
            v_vi = helper.make_tensor_value_info(out_name_v, TensorProto.FLOAT,
                                                 ["batch_size", "sequence_length", weight_tensor_v.shape[-1]])

            qkv_vi = [vi for vi in model.graph.value_info if vi.name == output_name_qkv][0]

            model.graph.value_info.extend([q_vi, k_vi, v_vi])
            model.graph.value_info.remove(qkv_vi)

            # add split nodes to graph
            model.graph.node.extend([node_matmul_q, node_matmul_k, node_matmul_v])
            model.graph.node.remove(node)

            gqa_in_node_name = node_name_q
            out_tensor_name = out_name_q
            node_type_str = "MatMul"

            for next_node in model.graph.node:
                if next_node.op_type == "Add" and "qkv_proj" in next_node.name and re.findall(r"layers\.\d+", next_node.name)[0].split(".")[-1] == matmul_layer_idx:
                    # Get names of input, output and weight
                    name_qkv_add = next_node.name
                    output_qkv_add = next_node.output[0]
                    weight_qkv_add = next_node.input[1]

                    # get weight initializer
                    weight_init_qkv_add = [initializer for initializer in model.graph.initializer if weight_qkv_add == initializer.name][0]

                    # get weight tensor
                    weight_tensor_qkv_add = numpy_helper.to_array(weight_init_qkv_add)

                    # split packed tensor into wq, wk, wv
                    weight_tensor_q_add = np.copy(weight_tensor_qkv_add[: n_q])
                    weight_tensor_k_add = np.copy(weight_tensor_qkv_add[n_q: n_q + n_k])
                    weight_tensor_v_add = np.copy(weight_tensor_qkv_add[n_q + n_k: n_q + n_k + n_v])

                    # remove packed tensor
                    model.graph.initializer.remove(weight_init_qkv_add)

                    # set wq, wk, wv tensor names
                    split_w_qkv_add = weight_qkv_add.split(ONNX_TENSOR_NAME_STRINGS["llama_qkv_proj"])
                    weight_name_q_add = split_w_qkv_add[0] + "q_proj" + split_w_qkv_add[1]
                    weight_name_k_add = split_w_qkv_add[0] + "k_proj" + split_w_qkv_add[1]
                    weight_name_v_add = split_w_qkv_add[0] + "v_proj" + split_w_qkv_add[1]

                    # set names for q, k, v Add nodes
                    split_name_qkv_add = name_qkv_add.split(ONNX_TENSOR_NAME_STRINGS["llama_qkv_proj"])
                    node_name_q_add = split_name_qkv_add[0] + "q_proj" + split_name_qkv_add[1]
                    node_name_k_add = split_name_qkv_add[0] + "k_proj" + split_name_qkv_add[1]
                    node_name_v_add = split_name_qkv_add[0] + "v_proj" + split_name_qkv_add[1]

                    # create initializers from split tensors
                    init_q_add = numpy_helper.from_array(weight_tensor_q_add, weight_name_q_add)
                    init_k_add = numpy_helper.from_array(weight_tensor_k_add, weight_name_k_add)
                    init_v_add = numpy_helper.from_array(weight_tensor_v_add, weight_name_v_add)

                    # add split tensors to initializers
                    model.graph.initializer.extend([init_q_add, init_k_add, init_v_add])

                    # set output names for split nodes
                    out_name_q_add = node_name_q_add + "/output_0"
                    out_name_k_add = node_name_k_add + "/output_0"
                    out_name_v_add = node_name_v_add + "/output_0"

                    # create split nodes
                    node_add_q = helper.make_node("Add", name=node_name_q_add, inputs=[out_name_q, weight_name_q_add],
                                                  outputs=[out_name_q_add])

                    node_add_k = helper.make_node("Add", name=node_name_k_add, inputs=[out_name_k, weight_name_k_add],
                                                  outputs=[out_name_k_add])

                    node_add_v = helper.make_node("Add", name=node_name_v_add, inputs=[out_name_v, weight_name_v_add],
                                                  outputs=[out_name_v_add])

                    # create output tensors and add to graph
                    q_vi_add = helper.make_tensor_value_info(out_name_q_add, TensorProto.FLOAT,
                                                             ["batch_size", "sequence_length", weight_tensor_q_add.shape[-1]])
                    k_vi_add = helper.make_tensor_value_info(out_name_k_add, TensorProto.FLOAT,
                                                             ["batch_size", "sequence_length", weight_tensor_k_add.shape[-1]])
                    v_vi_add = helper.make_tensor_value_info(out_name_v_add, TensorProto.FLOAT,
                                                             ["batch_size", "sequence_length", weight_tensor_v_add.shape[-1]])

                    qkv_vi_add = [vi for vi in model.graph.value_info if vi.name == output_qkv_add][0]

                    model.graph.value_info.extend([q_vi_add, k_vi_add, v_vi_add])
                    model.graph.value_info.remove(qkv_vi_add)

                    # add split nodes to graph
                    model.graph.node.extend([node_add_q, node_add_k, node_add_v])
                    model.graph.node.remove(next_node)

                    gqa_in_node_name = node_name_q_add
                    out_tensor_name = out_name_q_add
                    node_type_str = "Add"

            update_gqa_inputs(model, gqa_in_node_name, out_tensor_name, node_type_str)


def update_encodings(param_encodings: dict, num_layers: int, model_type: str):
    """
        Utility Function that updates the names of tensors in encodings.json file.
        The tensor names in encodings corresponding to GGUF are updated with ONNX model tensor names.
    """
    onnx_tensor_name_map = GGUF_TO_ONNX_TENSOR[model_type]

    for param_encoding in param_encodings:
        tensor_name = param_encoding["name"]

        for onnx_tensor_key in onnx_tensor_name_map:
            if onnx_tensor_key in tensor_name:
                onnx_tensor_value = onnx_tensor_name_map[onnx_tensor_key]
                if onnx_tensor_key == "output_norm":
                    onnx_tensor_value = onnx_tensor_value.format(max_block=num_layers)
                tensor_name = tensor_name.replace(onnx_tensor_key, onnx_tensor_value)
            param_encoding["name"] = tensor_name


def permute_weights(weights, num_heads: int, num_kv_heads: int):
    if num_kv_heads is not None and num_heads != num_kv_heads:
        num_heads = num_kv_heads
    return (weights.reshape(num_heads, 2, weights.shape[0] // num_heads // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))
