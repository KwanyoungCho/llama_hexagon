# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import copy

from qti.aisw.genai.qnn_genai_transformer_io import *

def permute_lazy(lazy_tensor: LazyTensor, n_head: int, n_head_kv: int) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().permute(n_head, n_head_kv)
    return LazyTensor(load, lazy_tensor.shape, lazy_tensor.data_type, f'permute({n_head}, {n_head_kv}) ' + lazy_tensor.description, lazy_tensor.scale, lazy_tensor.offset)

def part_lazy(lazy_tensor: LazyTensor, n_part: int, total_part: int = 3) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().part(n_part, total_part)
    s = lazy_tensor.shape.copy()
    s[0] = s[0] // total_part
    return LazyTensor(load, s, lazy_tensor.data_type, 'part ' + lazy_tensor.description, lazy_tensor.scale, lazy_tensor.offset)

def part_columns_lazy(lazy_tensor: LazyTensor, n_part: int, total_part: int = 3) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().part_columns(n_part, total_part)
    s = lazy_tensor.shape.copy()
    s[1] = s[1] // total_part
    return LazyTensor(load, s, lazy_tensor.data_type, 'part_columns ' + lazy_tensor.description, lazy_tensor.scale, lazy_tensor.offset)

def transpose2D_lazy(lazy_tensor: LazyTensor) -> LazyTensor:
    def load() -> Tensor:
        return lazy_tensor.load().transpose()
    s = lazy_tensor.shape.copy()
    s.reverse()
    return LazyTensor(load, s, lazy_tensor.data_type, 'transpose2D ' + lazy_tensor.description, lazy_tensor.scale, lazy_tensor.offset)

def combine_token_type_embd(token_embd: LazyTensor, token_type_embd: LazyTensor) -> LazyTensor:
    def load() -> Tensor:
        return UnquantizedTensor(token_embd.load().to_ggml().ndarray + token_type_embd.load().to_ggml().ndarray[0])
    s = token_embd.shape.copy()
    return LazyTensor(load, s, token_embd.data_type, 'combined token_embd + token_typ_embd[0]', lazy_tensor.scale, lazy_tensor.offset)

def split_QKV(lazy_tensor: LazyTensor, name: str, bid: int, transpose: bool, model: LazyModel, permute_rope: bool, params: Params):
    out_names = [
        TensorMap.get_tensor_type_name(MODEL_TENSOR.ATTN_Q, bid = bid),
        TensorMap.get_tensor_type_name(MODEL_TENSOR.ATTN_K, bid = bid),
        TensorMap.get_tensor_type_name(MODEL_TENSOR.ATTN_V, bid = bid)
    ]
    if transpose:
        outputs = [
            (out_names[0], part_columns_lazy(lazy_tensor, 0)),
            (out_names[1], part_columns_lazy(lazy_tensor, 1)),
            (out_names[2], part_columns_lazy(lazy_tensor, 2))
        ]
    else:
        outputs = [
            (out_names[0], part_lazy(lazy_tensor, 0)),
            (out_names[1], part_lazy(lazy_tensor, 1)),
            (out_names[2], part_lazy(lazy_tensor, 2))
        ]
    if permute_rope:
        outputs = [
            (out_names[0], permute_lazy(outputs[0][1], params.n_head, params.n_head)),
            (out_names[1], permute_lazy(outputs[1][1], params.n_head, params.n_head_kv)),
            (out_names[2], outputs[2][1])
        ]
    for out_name, out_tensor in outputs:
        model[out_name] = out_tensor
        print(f"{name:50s} -> {out_name:40s} | {out_tensor.data_type.name:6s} | {out_tensor.shape}")

def split_QKV_bias(lazy_tensor: LazyTensor, name: str, bid: int, model: LazyModel):
    outputs = [
        (TensorMap.get_tensor_type_name(MODEL_TENSOR.ATTN_Q_BIAS, bid = bid), part_lazy(lazy_tensor, 0)),
        (TensorMap.get_tensor_type_name(MODEL_TENSOR.ATTN_K_BIAS, bid = bid), part_lazy(lazy_tensor, 1)),
        (TensorMap.get_tensor_type_name(MODEL_TENSOR.ATTN_V_BIAS, bid = bid), part_lazy(lazy_tensor, 2))
    ]
    for out_name, out_tensor in outputs:
        model[out_name] = out_tensor
        print(f"{name:50s} -> {out_name:40s} | {out_tensor.data_type.name:6s} | {out_tensor.shape}")

def split_up_gate(lazy_tensor: LazyTensor, name: str, bid: int, transpose: bool, model: LazyModel):
    out_names = [
        TensorMap.get_tensor_type_name(MODEL_TENSOR.FFN_GATE, bid = bid),
        TensorMap.get_tensor_type_name(MODEL_TENSOR.FFN_UP, bid = bid)
    ]
    if transpose:
        outputs = [
            (out_names[0], part_columns_lazy(lazy_tensor, 0, 2)),
            (out_names[1], part_columns_lazy(lazy_tensor, 1, 2))
        ]
    else:
        outputs = [
            (out_names[0], part_lazy(lazy_tensor, 0, 2)),
            (out_names[1], part_lazy(lazy_tensor, 1, 2))
        ]
    for out_name, out_tensor in outputs:
        model[out_name] = out_tensor
        print(f"{name:50s} -> {out_name:40s} | {out_tensor.data_type.name:6s} | {out_tensor.shape}")

def validate_model_names(model: LazyModel, tensor_map: TensorMap):
    for name, _ in tensor_map.get_transpose_map().items():
        if name not in model:
            raise ValueError(f"{name} not found in model. Please check configuration.json")

def convert_decoder_model_names(model: LazyModel, params: Params, config_path: Path, permute_rope: bool) -> LazyModel:
    tensor_map = TensorMap(config_path, False)
    out: LazyModel = {}

    validate_model_names(model, tensor_map)

    for name, tensor in model.items():
        if name not in tensor_map.get_transpose_map():
            print(f"SKIPPING {name}")
            continue

        tensor.scale = tensor_map.get_tensor_scale(name)
        tensor.offset = tensor_map.get_tensor_offset(name)

        if len(re.findall(r'\d+', name)):
            bid = int(re.findall(r'\d+', name)[0])
            if tensor_map.get_tensor_type(name) == MODEL_TENSOR.ATTN_QKV:
                split_QKV(tensor, name, bid, tensor_map.get_tensor_transpose(name), out, permute_rope, params)
            elif tensor_map.get_tensor_type(name) == MODEL_TENSOR.ATTN_QKV_BIAS:
                split_QKV_bias(tensor, name, bid, out)
            elif tensor_map.get_tensor_type(name) == MODEL_TENSOR.FFN_UP_GATE:
                split_up_gate(tensor, name, bid, tensor_map.get_tensor_transpose(name), out)
            else:
                out_name = tensor_map.get_converted_name(name, bid = bid)
                tensor_out: LazyTensor = tensor
                if tensor_map.get_tensor_transpose(name):
                    tensor_out = transpose2D_lazy(tensor)
                if tensor_map.get_tensor_type(name) == MODEL_TENSOR.ATTN_Q and permute_rope:
                    tensor_out = permute_lazy(tensor_out, params.n_head, params.n_head)
                if tensor_map.get_tensor_type(name) == MODEL_TENSOR.ATTN_K and permute_rope:
                    tensor_out = permute_lazy(tensor_out, params.n_head, params.n_head_kv)
                out[out_name] = tensor_out
                print(f"{name:50s} -> {out_name:40s} | {out[out_name].data_type.name:6s} | {out[out_name].shape}")
        else:
            out_name = tensor_map.get_converted_name(name)
            if tensor_map.get_tensor_transpose(name):
                out[out_name] = transpose2D_lazy(tensor)
            else:
                out[out_name] = tensor
            print(f"{name:50s} -> {out_name:40s} | {out[out_name].data_type.name:6s} | {out[out_name].shape}")

    if out.get("output.weight", None) is None:
        out_name = "output.weight"
        out[out_name] = copy.deepcopy(out["token_embd.weight"])
        # TODO: Update for output.weight/lm_head.weight scales/offsets
        out[out_name].scale = 1.0
        out[out_name].offset = 0
        print(f"{out_name:94s} | {out[out_name].data_type.name:6s} | {out[out_name].shape}")

    return out

def convert_encoder_model_names(model: LazyModel, params: Params, config_path: Path) -> LazyModel:
    tensor_map = TensorMap(config_path, False)
    num_decoders = params.n_layer
    out: LazyModel = {}

    validate_model_names(model, tensor_map)

    token_embd: (str, LazyTensor) = None
    token_type_embd: (str, LazyTensor) = None
    for name, tensor in model.items():
        if name not in tensor_map.get_transpose_map():
            continue

        if tensor_map.get_tensor_type(name) == MODEL_TENSOR.TOKEN_EMBD:
            token_embd = (name, tensor)
        elif tensor_map.get_tensor_type(name) == MODEL_TENSOR.TOKEN_EMBD_TYPE:
            token_type_embd = (name, tensor)
    assert token_embd is not None, "Token Embedding tensor not found"
    assert token_type_embd is not None, "Token Type Embedding tensor not found"
    assert token_embd[1].shape[1] == token_type_embd[1].shape[1], "Token Embedding and Token Type Embedding weights mismatch"
    assert token_type_embd[1].shape[0] == 2, "Token Type Embedding has INVALID shape"
    model[token_embd[0]] = combine_token_type_embd(token_embd[1], token_type_embd[1])
    del model[token_type_embd[0]]

    for name, tensor in model.items():
        if name not in tensor_map.get_transpose_map():
            print(f"SKIPPING {name}")
            continue

        if len(re.findall(r'\d+', name)):
            bid = int(re.findall(r'\d+', name)[0])
            if bid == (num_decoders - 1):
                if tensor_map.get_tensor_type(name) == MODEL_TENSOR.FFN_OUT_NORM:
                    out_name = TensorMap.get_tensor_type_name(MODEL_TENSOR.OUTPUT_NORM)
                elif tensor_map.get_tensor_type(name) == MODEL_TENSOR.FFN_OUT_NORM_BIAS:
                    out_name = TensorMap.get_tensor_type_name(MODEL_TENSOR.OUTPUT_NORM_BIAS)
                else:
                    out_name = tensor_map.get_converted_name(name, bid)
            else:
                if tensor_map.get_tensor_type(name) == MODEL_TENSOR.FFN_OUT_NORM:
                    out_name = TensorMap.get_tensor_type_name(MODEL_TENSOR.ATTN_NORM, bid = (bid + 1))
                elif tensor_map.get_tensor_type(name) == MODEL_TENSOR.FFN_OUT_NORM_BIAS:
                    out_name = TensorMap.get_tensor_type_name(MODEL_TENSOR.ATTN_NORM_BIAS, bid = (bid + 1))
                else:
                    out_name = tensor_map.get_converted_name(name, bid)
            if tensor_map.get_tensor_transpose(name):
                out[out_name] = transpose2D_lazy(tensor)
            else:
                out[out_name] = tensor
            print(f"{name:50s} -> {out_name:40s} | {out[out_name].data_type.name:6s} | {out[out_name].shape}")
        else:
            if tensor_map.get_tensor_type(name) == MODEL_TENSOR.TOKEN_EMBD_NORM:
                out_name = TensorMap.get_tensor_type_name(MODEL_TENSOR.ATTN_NORM, bid = 0)
            elif tensor_map.get_tensor_type(name) == MODEL_TENSOR.TOKEN_EMBD_NORM_BIAS:
                out_name = TensorMap.get_tensor_type_name(MODEL_TENSOR.ATTN_NORM_BIAS, bid = 0)
            else:
                out_name = tensor_map.get_converted_name(name)
            if tensor_map.get_tensor_transpose(name):
                out[out_name] = transpose2D_lazy(tensor)
            else:
                out[out_name] = tensor
            print(f"{name:50s} -> {out_name:40s} | {out[out_name].data_type.name:6s} | {out[out_name].shape}")

    return out

def convert_model_names(model: LazyModel, params: Params, config_path: Path) -> LazyModel:
    if params.attention_mode == "causal":
        permute_rope = False
        if params.kq_comp_org is not None and params.kq_comp_org == "SoA":
            params.comp_org = "AoS"
            permute_rope = True
        return convert_decoder_model_names(model, params, config_path, permute_rope)
    elif params.attention_mode == "bidirectional":
        return convert_encoder_model_names(model, params, config_path)

def calculateLlama3ScalingFactors(params: Params):
    # RoPE Base theta
    base = params.f_rope_scale
    # Head dim = n_embd / n_heads
    dim = params.n_embd / params.n_head
    frequencies = base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    factor = params.rope_config.get('factor')
    low_freq_factor = params.rope_config.get('low_freq_factor')
    high_freq_factor = params.rope_config.get('high_freq_factor')
    original_max_position_embeddings = params.rope_config.get('original_max_position_embeddings')

    # Wavelength = num of tokens for RoPE to undergo full rotation i.e., (2 * pi) radian
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor

    rope_factors = []
    for frequency in frequencies:
        wavelen = 2 * math.pi * frequency
        if wavelen < high_freq_wavelen:
            rope_factors.append(1.0)
        elif wavelen > low_freq_wavelen:
            rope_factors.append(factor)
        else:
            smoothing_factor = (original_max_position_embeddings / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            rope_factors.append(1 / ((1 - smoothing_factor) / factor + smoothing_factor))
    return rope_factors

def calculateYarnScalingFactors(params: Params):
    # RoPE Base theta
    base = params.f_rope_scale
    # partial_rotary_factor from config
    partial_rotary_factor = 1.0
    # Head dim = n_embd / n_heads
    head_dim = params.n_embd / params.n_head
    dim = int(head_dim * partial_rotary_factor)
    factor = params.rope_config.get('factor')
    original_max_position_embeddings = params.rope_config.get('original_max_position_embeddings')

    # Optional YaRN beta_fast (β) / beta_slow (α): as per paper, default to 32 and 1
    if params.rope_config.get('beta_fast') is not None:
        beta_fast = params.rope_config.get('beta_fast')
    else:
        beta_fast = 32
    if params.rope_config.get('beta_slow') is not None:
        beta_fast = params.rope_config.get('beta_slow')
    else:
        beta_slow = 1

    # r(d) = L / 2 * pi * (base ** (2 * d / dim)); d in [0, dim-2]
    # Solve for d to get the below formula, substituting r as num_rotations (beta_fast / beta_slow)
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    # solve for d at beta_fast and beta_slow to get correction dims
    def find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings):
        low = math.floor(find_correction_dim(beta_fast, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(beta_slow, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    # From YaRN paper
    # γ(r) = 0, r < α
    # γ(r) = 1, r > β
    # γ(r) = ((r - α) / (β - α)), β <= r <= α
    # θ = 1 / base ** (2 * d / dim);  d in [0, dim-2]
    # s = factor
    # θ' = ((1 - γ(r)) * (θ / s)) + (γ(r) *  θ)
    def linear_ramp_factor(min, max, dim):
        # Prevent div by 0
        if min == max:
            max += 0.001

        linear_func = (np.arange(dim, dtype=np.float32) - min) / (max - min)
        ramp_func = np.clip(linear_func, 0, 1)
        return ramp_func

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
    extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2)

    # θ' = ((1 - γ(r)) * (θ / s)) + (γ(r) *  θ)
    # θ' = (((1 - γ(r)) / s) + γ(r)) * θ
    # rope_factors = 1 / (((1 - γ(r)) / s) + γ(r))
    # θ' = θ / rope_factors
    rope_factors = (1 / ((1 - extrapolation_factor) / factor + extrapolation_factor))
    return rope_factors

def calculateYarnAttnFactor(params: Params):
    if params.rope_config.get('attention_factor') is not None:
        attn_factor = params.rope_config.get('attention_factor')
    else:
        attn_factor = 0.1 * math.log(params.rope_config.get('factor')) + 1.0
    return attn_factor

def calculateLongRopeAttnFactor(params: Params):
    if params.rope_config.get("attention_factor") is not None:
        attn_factor = params.rope_config["attention_factor"]
    else:
        factor = params.n_ctx / params.rope_config.get('original_max_position_embeddings')
        attn_factor = math.sqrt(1 + (math.log(factor) / math.log(params.rope_config.get('original_max_position_embeddings'))))
    return attn_factor

#
# Model Params
#

@dataclass
class Params:
    n_align:             int
    n_vocab:             int
    n_ctx:               int
    n_embd:              int
    embd_per_head:       int
    n_ff:                int
    n_layer:             int
    n_head:              int
    n_head_kv:           int
    n_rot:               int
    f_norm_eps:          float | None = None
    f_rope_scale:        float | None = None
    f_rope_factor_short: list[float] | None = None
    f_rope_factor_long:  list[float] | None = None
    rope_attn_factor:    float | None = None
    name:                str | None = None
    arch:                str | None = None
    tokenizer:           str | None = None
    output:              str | None = None
    model_id:            str | None = None
    connector:           str | None = None
    gating:              str | None = None
    norm:                str | None = None
    activation:          str | None = None
    pos_embd:            str | None = None
    attention_mode:      str | None = None
    comp_org:            str | None = None
    kq_comp_org:         str | None = None
    ftype:               GGMLFileType | None = None
    rope_config:         dict | None = None

    # path to the directory containing the model files
    path_model:         Path | None = None

    @staticmethod
    def loadTransformerJson(model: LazyModel, config_path: Path) -> Params:
        global NAMES
        config = json.load(open(config_path))

        name                = config["general.name"]
        arch                = config["general.architecture"] if "general.architecture" in config else "generic"
        tokenizer           = config["general.tokenizer"] if "general.tokenizer" in config else "none"
        n_align             = config["general.alignment"] if "general.alignment" in config else 32
        model_id            = config["general.hf_hub_model_id"] if "general.hf_hub_model_id" in config else None
        output              = config["general.output"] if "general.output" in config else "logits"
        n_vocab             = config["size.vocabulary"]
        n_ctx               = config["size.context"]
        n_embd              = config["size.embedding"]
        n_head              = config["architecture.num_heads"]
        embd_per_head       = config["size.embedding_per_head"] if "size.embedding_per_head" in config else n_embd//n_head
        n_ff                = config["size.feedforward"]
        n_layer             = config["architecture.num_decoders"]
        n_head_kv           = config["architecture.num_kv_heads"] if "architecture.num_kv_heads" in config else n_head
        connector           = config["architecture.connector"]
        gating              = config["architecture.gating"]
        norm                = config["operation.normalization"]
        f_norm_eps          = config["operation.normalization_epsilon"] if "operation.normalization_epsilon" in config else 0.000001
        activation          = config["operation.activation"]
        pos_embd            = config["operation.positional_embedding"]
        attention_mode      = config["operation.attention_mode"] if "operation.attention_mode" in config else "causal"
        # if "operation.positional_embedding" is set to "RoPE", following params are in use
        n_rot               = config["operation.rope_num_rotations"] if "operation.rope_num_rotations" in config else embd_per_head
        comp_org            = config["operation.rope_complex_organization"] if "operation.rope_complex_organization" in config else None
        kq_comp_org         = config["tensor.kq_complex_organization"] if "tensor.kq_complex_organization" in config else None
        f_rope_scale        = config["operation.rope_scaling"] if "operation.rope_scaling" in config else 10000.0
        f_rope_factor_short = config["operation.rope.scaling.factor"] if "operation.rope.scaling.factor" in config else None
        rope_config         = config["operation.rope.scaling.config"] if "operation.rope.scaling.config" in config else None

        if comp_org is None:
            if kq_comp_org is not None and kq_comp_org == "SoA":
                comp_org = "AoS"

        return Params(
            n_align               =   n_align,
            n_vocab               =   n_vocab,
            n_ctx                 =   n_ctx,
            n_embd                =   n_embd,
            embd_per_head         =   embd_per_head,
            n_ff                  =   n_ff,
            n_layer               =   n_layer,
            n_head                =   n_head,
            n_head_kv             =   n_head_kv,
            n_rot                 =   n_rot,
            f_norm_eps            =   f_norm_eps,
            f_rope_scale          =   f_rope_scale,
            f_rope_factor_short   =   f_rope_factor_short,
            rope_config           =   rope_config,
            name                  =   name,
            arch                  =   arch,
            tokenizer             =   tokenizer,
            model_id              =   model_id,
            output                =   output,
            connector             =   connector,
            gating                =   gating,
            norm                  =   norm,
            activation            =   activation,
            pos_embd              =   pos_embd,
            attention_mode        =   attention_mode,
            comp_org              =   comp_org,
            kq_comp_org           =   kq_comp_org
        )

    @staticmethod
    def load(model_plus: ModelPlus, config_path: Path) -> Params:
        if config_path.exists():
            params = Params.loadTransformerJson(model_plus.model, config_path)
        else:
            raise ValueError('Cannot get params for model format')

        params.path_model = model_plus.paths[0].parent
        return params

def getConfigFromSDK(model_config_path: Path):
    model_config = json.load(open(model_config_path))
    sdk_path = os.environ.get('QNN_SDK_ROOT')
    if "architectures" in model_config:
        model_arch = model_config["architectures"][0]
        if model_arch == "QWenLMHeadModel":
            config_name = "qwen-7b-chat"
        elif model_arch == "BaiChuanForCausalLM":
            config_name = "baichuan1-7b"
        elif model_arch == "GPT2LMHeadModel":
            if "n_layer" in model_config:
                n_layer = model_config["n_layer"]
                if n_layer == 12:
                    config_name = "gpt2-124m"
                elif n_layer == 24:
                    config_name = "gpt2-335m"
                elif n_layer == 36:
                    config_name = "gpt2-774m"
                else:
                    raise Exception("Please provide configuration.json file for this model\n")
            else:
                raise Exception("Please provide configuration.json file for this model\n")
        elif model_arch == "LlamaForCausalLM" or model_arch == "LLaMAForCausalLM":
            if ("max_position_embeddings" in model_config or "max_sequence_length" in model_config) and "intermediate_size" in model_config:
                n_ctx  = model_config["max_position_embeddings"] if "max_position_embeddings" in model_config else model_config["max_sequence_length"]
                n_ff   = model_config["intermediate_size"]
                n_embd = model_config["hidden_size"]
                if n_ctx == 2048:
                    if n_ff == 11008:
                        config_name = "llama1-7b-hf"
                elif n_ctx == 4096:
                    if n_ff == 11008:
                        config_name = "llama2-7b"
                    elif n_ff == 13824:
                        config_name = "llama2-13b"
                elif n_ctx == 8192:
                    if n_ff == 14336:
                        config_name = "llama3-8b"
                elif n_ctx == 131072:
                    if n_ff == 14336:
                        config_name = "llama3.1-8b"
                    elif n_ff == 8192:
                        if n_embd == 2048:
                            config_name = "llama3.2-1b"
                        elif n_embd == 3072:
                            config_name = "llama3.2-3b"
                else:
                    raise Exception("Please provide configuration.json file for this model\n")
            else:
                raise Exception("Please provide configuration.json file for this model\n")
        elif model_arch == "GemmaForCausalLM":
            n_ff = model_config["intermediate_size"]
            if n_ff == 16384:
                config_name = "gemma-2b"
            elif n_ff == 24576:
                config_name = "gemma-7b"
        elif model_arch == "MistralForCausalLM":
            n_vocab = model_config["vocab_size"]
            if n_vocab == 32768:
                config_name = "mistral-7b-v0.3"
            elif n_vocab == 32000:
                config_name = "mistral-7b-v0.2"
        elif model_arch == "Phi3ForCausalLM":
            model_name = model_config["_name_or_path"]
            if "Phi-3.5" in model_name:
                config_name = "Phi-3.5-mini-instruct"
            elif "Phi-3" in model_name:
                config_name = "Phi-3-mini-128k-instruct"
        elif model_arch == "BertModel":
            config_name = "bge-large-en-v1_5"
        elif model_arch == "GPT2Model" and model_config["n_layer"] == 40:
            config_name = "cerebras-gpt-13b"
        else:
            raise Exception("Please provide configuration.json file for this model\n")
    elif "model_type" in model_config:
        model_type = model_config["model_type"]
        if model_type == "gpt2":
            n_embd = model_config["n_embd"] if "n_embd" in model_config else 0
            if n_embd == 768:
                config_name = "cerebras-gpt-111m"
            elif n_embd == 1088:
                config_name = "cerebras-gpt-256m"
            elif n_embd == 1536:
                config_name = "cerebras-gpt-590m"
            elif n_embd == 4096:
                config_name = "cerebras-gpt-6.7b"
            elif n_embd == 2560:
                config_name = "cerebras-gpt-2.7b"
            else:
                raise Exception("Please provide configuration.json file for this model\n")
        else:
            raise Exception("Please provide configuration.json file for this model\n")

    else:
        raise Exception("Please provide configuration.json file for this model\n")

    config_path = "/lib/python/qti/aisw/genai/configs/" + config_name + ".json"
    model_config_str = sdk_path + config_path
    model_config_path = Path(model_config_str)
    if model_config_path.exists():
        return model_config_path
    else:
        raise Exception("Please provide configuration.json file for this model\n")

def run_composer(model: Path, quantize: str = None, export_tokenizer_json: bool = False, outfile: Path = None, config_file: Path = None, lora: Path = None, lm_head_precision: str = None) -> None:
    if model:
        model_config_path = model / "config.json"
    else:
        raise Exception("Please provide base model path\n")

    if lora:
        if config_file:
            config_path = config_file
        elif model_config_path.exists():
            config_path = getConfigFromSDK(model_config_path)
        else:
            raise Exception("No configuration present for the base model. Please provide configuration file --config_file\n")
        from qti.aisw.genai.qnn_genai_transformer_lora import convert_lora_model
        convert_lora_model(lora, outfile, config_path)
        return

    if model:
        if config_file:
            config_path = config_file
        elif model_config_path.exists():
            config_path = getConfigFromSDK(model_config_path)
        else:
            raise Exception("No configuration present for the base model. Please provide configuration file --config_file\n")
    else:
        raise Exception("Please provide base model path\n")
    model_plus = load_some_model(model)

    params = Params.load(model_plus, config_path)

    if params.rope_config is not None:
        if (params.rope_config.get('type') == 'llama3'):
            rope_scaling_factors = calculateLlama3ScalingFactors(params)
            params.f_rope_factor_short = rope_scaling_factors
        elif (params.rope_config.get('type') == 'yarn'):
            rope_scaling_factors = calculateYarnScalingFactors(params)
            params.f_rope_factor_short = rope_scaling_factors
            params.rope_attn_factor = calculateYarnAttnFactor(params)
        elif (params.rope_config.get('type') == 'longrope'):
            params.f_rope_factor_short = params.rope_config["short_factor"]
            params.f_rope_factor_long = params.rope_config["long_factor"]
            params.rope_attn_factor = calculateLongRopeAttnFactor(params)
        else:
            raise Exception(f"ROPE type {params.rope_config.get('type')} is not yet supported\n")

    if params.n_ctx == -1:
        raise Exception("The model doesn't have a context size\n")

    if quantize:
        params.ftype = {
            "Z4":      GGMLFileType.MostlyZ4,
            "Z4_FP16": GGMLFileType.Z4_FP16,
            "Z4_BF16": GGMLFileType.Z4_BF16,
            "Q4":      GGMLFileType.MostlyQ4_0_32,
            "Z8":      GGMLFileType.MostlyZ8
        }[quantize]
    else:
        params.ftype = GGMLFileType.AllF32

    lm_quantize: str
    if lm_head_precision:
        lm_quantize = lm_head_precision
    else:
        if quantize:
            lm_quantize = quantize
        else:
            lm_quantize = "FP_32"

    print(f"params = {params}")

    model   = model_plus.model
    model   = convert_model_names(model, params, config_path)
    model   = convert_to_output_type(model, GGMLFileType.AllF32)
    outfile = outfile or default_outfile(model_plus.paths, params.ftype)

    print(f"Writing {outfile}, format {params.ftype}")
    OutputFile.write_all(outfile, params, model, lm_quantize)
    print(f"Wrote {outfile}")

    if export_tokenizer_json:
        print("Writing tokenizer.json")
        vocab_dir = model_plus.paths[0].parent
        if params.arch == "qwen":
            qnn_genai_transformer_tokenizer.QwenTokenizer(dir_model = vocab_dir, export_path = outfile.parent, export_tokenizer_json = True)._create_qwen_bpe(disable = False)
        elif params.name == "baichuan1-7b":
            qnn_genai_transformer_tokenizer.BaichuanTokenizer(dir_model = vocab_dir, export_path = outfile.parent, export_tokenizer_json = True)._create_baichuan_bpe()
        elif "cerebras" in (params.name).lower():
            qnn_genai_transformer_tokenizer.CerebrasTokenizer(dir_model = vocab_dir, export_path = outfile.parent, export_tokenizer_json = True)._create_cerebras_bpe(disable = False)
        elif "mistral" in (params.name).lower():
            qnn_genai_transformer_tokenizer.MistralTokenizer(dir_model = vocab_dir, export_path = outfile.parent, export_tokenizer_json = True)._create_mistral_tokenizer(disable = False)
        else:
            def user_warning_format(message, category, filename, lineno, file=None, line=None):
                return '%s: %s\n' % (category.__name__, message)
            warnings.formatwarning = user_warning_format
            warnings.warn("This option is only supported by QWen and Baichuan models.")
