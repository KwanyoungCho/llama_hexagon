from collections import defaultdict

 # MHA2SHA flags
mha2sha_base_arch_flags = defaultdict()
_mha2sha_llama2_flags = {
    "create_input_lists": False,
    "disable_auto_attn_finder": False,
    "is_gqa_model": False,
    "handle_rope_ops": True,
    "handle_past_key_value": True,
    "is_llm_model": True,
    "is_prepared_model": True,
    "replace_linear_with_conv": True,
    "strict_rope_pattern": True,
}
_mha2sha_llama3_flags = _mha2sha_llama2_flags | {"is_gqa_model": True}

mha2sha_base_arch_flags["llama2"] = _mha2sha_llama2_flags
mha2sha_base_arch_flags["llama3"] = _mha2sha_llama3_flags

# Splitter
splitter_base_arch_flags = defaultdict()
_llama2_flags = {
    "num_splits": 4,
    "split_embedding": False,
    "split_lmhead": False
}
_llama3_flags = {
    "num_splits": 5,
    "split_embedding": True,
    "split_lmhead": False
}

splitter_base_arch_flags["llama2"] = _llama2_flags
splitter_base_arch_flags["llama3"] = _llama3_flags
