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

from qti.aisw.genai.qnn_genai_transformer_io import *

#
# Model Params
#


@dataclass
class Params:
    alpha:               int
    rank:                int
    n_layer:             int
    n_align:             int
    name:                str | None = None
    arch:                str | None = None
    ftype:               GGMLFileType | None = None

    # path to the directory containing the model files
    path_model:         Path | None = None

    @staticmethod
    def loadTransformerJson(config_path: Path, model_config_path: Path) -> Params:
        global NAMES
        config = json.load(open(config_path))

        name             = config["general.name"]
        arch             = config["general.architecture"] if "general.architecture" in config else "generic"
        n_layer          = config["architecture.num_decoders"]
        n_align          = 128

        config = json.load(open(model_config_path))
        alpha            = config["lora_alpha"] if "lora_alpha" in config else 16
        rank             = config["r"] if "r" in config else 64

        return Params(
            alpha          =   alpha,
            rank           =   rank,
            n_layer        =   n_layer,
            n_align        =   n_align,
            name           =   name,
            arch           =   arch,
        )

    @staticmethod
    def load(model_plus: ModelPlus, config_path: Path, model_config_path: Path) -> Params:
        if config_path.exists():
            params = Params.loadTransformerJson(config_path, model_config_path)
        else:
            raise ValueError('Cannot get params for model format')

        params.path_model = model_plus.paths[0].parent
        return params

def convert_lora_model_names(model: LazyModel, config_path: Path) -> LazyModel:
    tensor_map = TensorMap(config_path, lora = True)
    out: LazyModel = {}

    for name, tensor in model.items():
        if name not in tensor_map.get_transpose_map():
            print(f"SKIPPING {name}")
            continue

        if len(re.findall(r'\d+', name)):
            bid = int(re.findall(r'\d+', name)[0])
            out_name = tensor_map.get_converted_lora_name(name, bid = bid)
            if tensor_map.get_tensor_transpose(name):
                out[out_name] = transpose2D_lazy(tensor)
            else:
                out[out_name] = tensor
            print(f"{name:65s} -> {out_name:40s} | {out[out_name].data_type.name:6s} | {out[out_name].shape}")
        else:
            out_name = tensor_map.get_converted_lora_name(name)
            if tensor_map.get_tensor_transpose(name):
                out[out_name] = transpose2D_lazy(tensor)
            else:
                out[out_name] = tensor
            print(f"{name:65s} -> {out_name:40s} | {out[out_name].data_type.name:6s} | {out[out_name].shape}")

    return out

def convert_lora_model(lora: Path, outfile: Path, config_path: Path) -> None:
    model_config_path = lora / "adapter_config.json"
    model_plus = load_some_model(lora)

    params = Params.load(model_plus, config_path, model_config_path)
    params.ftype = GGMLFileType.AllF32

    print(f"params = {params}")

    model   = model_plus.model
    model   = convert_lora_model_names(model, config_path)
    ftype   = GGMLFileType.AllF32
    model   = convert_to_output_type(model, ftype)
    outfile = outfile or default_outfile(model_plus.paths, params.ftype)

    print(f"Writing {outfile}, format {params.ftype}")
    OutputFile.write_all(outfile, params, model)
    print(f"Wrote {outfile}")
