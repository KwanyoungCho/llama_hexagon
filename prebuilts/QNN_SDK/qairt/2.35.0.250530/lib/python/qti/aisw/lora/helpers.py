# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import re
from pathlib import Path
import onnx
from safetensors.numpy import load_file, save_file
from onnx.numpy_helper import to_array
from onnx import numpy_helper
import json


def is_attach_point(pytorch_module, adapter_target_modules, delimiter="_"):
    """
    Checks if a pytorch module is an attach point given the adapter config's "target_modules".

    Parameters:
        pytorch_module (str): Name of pytorch module.
        adapter_target_modules (list[str]): List of keywords from adapter config to determine if pytorch_module is an attach point.

    Returns:
        bool: True if pytorch_module is an attach point, False otherwise.
    """

    if pytorch_module in adapter_target_modules:
        # this module is specified directly in adapter_target_modules
        return True
    else:
        return any(pytorch_module.endswith(f"{delimiter}{target_key}") for target_key in adapter_target_modules)


def validate_file_path(file_path):
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("The path {} does not exist or is not a file.".format(file_path))


def create_base_graph_name(name):
    base_name = name
    if "_base_layer" in name:
        base_name = name.replace("_base_layer", "")
    elif ".base_layer" in name:
        base_name = name.replace(".base_layer", "")
    elif "/base_layer" in name:
        base_name = name.replace("/base_layer", "")

    return base_name


def create_encoding_map(encodings):
    encodings_map = {'param_encodings': {}, 'activation_encodings': {}}
    for enc in encodings['param_encodings']:
        name = enc['name']
        encodings_map['param_encodings'][name] = enc
    for enc in encodings['activation_encodings']:
        name = enc['name']
        encodings_map['activation_encodings'][name] = enc
    encodings_map['version'] = encodings['version']
    if "quantizer_args" in encodings:
        encodings_map["quantizer_args"] = encodings["quantizer_args"]

    return encodings_map


def get_encodings(encodings_map):
    encodings = {}
    encodings['param_encodings'] = list(encodings_map['param_encodings'].values())
    encodings['activation_encodings'] = list(encodings_map['activation_encodings'].values())
    encodings['version'] = encodings_map['version']
    if "quantizer_args" in encodings_map:
        encodings["quantizer_args"] = encodings_map["quantizer_args"]

    return encodings


def apply_safetensors_to_onnx(model, safetensors_dict):
    """
    Apply safetensors into an ONNX model and return the updated model.
    :param model: The original ONNX model to be patched.
    :param safetensors_dict: A dictionary containing safetensors to be applied to the model.
    :return: onnx.ModelProto: The patched ONNX model with safetensors integrated.
    """

    # Overwrite tensors in the ONNX model with those from the safetensors file
    for i, tensor in enumerate(model.graph.initializer):
        if tensor.name in safetensors_dict:
            original_tensor = to_array(tensor)
            # Validate Datatype and shape of the tensor
            if original_tensor.shape != safetensors_dict[tensor.name].shape:
                raise ValueError("Invalid Safetensors file {}: Unable to patched onnx model."
                                 "Shape of the tensor {} in the safetensors file is not matching with the "
                                 "onnx model.".format(safetensors_path, tensor.name))
            if original_tensor.dtype != safetensors_dict[tensor.name].dtype:
                raise ValueError("Invalid Safetensors file {}: Unable to patched onnx model."
                                 "Datatype of the tensor {} in the safetensors file is not matching with the "
                                 "onnx model.".format(safetensors_path, tensor.name))

            # Create a new tensor with the data from the safetensors file
            new_tensor = numpy_helper.from_array(safetensors_dict[tensor.name], name=tensor.name)

            # Replace the old tensor with the new tensor
            model.graph.initializer.remove(tensor)
            model.graph.initializer.insert(i, new_tensor)

    return model


def open_and_load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)