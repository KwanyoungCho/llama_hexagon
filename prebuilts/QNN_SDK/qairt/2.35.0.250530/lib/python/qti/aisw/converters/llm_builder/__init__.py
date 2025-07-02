# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import sys
import onnxruntime_genai

from importlib import util

# Required for create_model method
ort_genai_lib_path = onnxruntime_genai.__path__[0]
sys.path.append(os.path.join(ort_genai_lib_path, "models"))

# Replace modules in transformers package
init_module_path = os.path.dirname(__file__)
ggml_module_path = os.path.join(init_module_path, "ggml.py")
ggml_spec = util.spec_from_file_location("transformers.integrations.ggml", ggml_module_path)
ggml_module_ref = util.module_from_spec(ggml_spec)
ggml_spec.loader.exec_module(ggml_module_ref)
sys.modules["transformers.integrations.ggml"] = ggml_module_ref

gguf_utils_module_path = os.path.join(init_module_path, "modeling_gguf_pytorch_utils.py")
gguf_utils_spec = util.spec_from_file_location("transformers.modeling_gguf_pytorch_utils", gguf_utils_module_path)
gguf_module_ref = util.module_from_spec(gguf_utils_spec)
gguf_utils_spec.loader.exec_module(gguf_module_ref)
sys.modules["transformers.modeling_gguf_pytorch_utils"] = gguf_module_ref

from .utils import (permute_weights, unpack_qkv, update_symbolic_shape_with_value,
                    decompose_gqa, decompose_layernorms, update_encodings,
                    MODEL_TYPE_TO_ARCH, MODEL_TYPE_TO_TOKENIZER)
from .gguf_parser import GGUFParser
from .graph_builder import GraphBuilder
