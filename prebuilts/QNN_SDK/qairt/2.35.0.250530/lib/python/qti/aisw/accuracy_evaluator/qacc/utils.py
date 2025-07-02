# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import yaml
import json
from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger


def convert_npi_to_json(npi_yaml_file, output_json):
    """Converts npi yaml file to output json in qnn quantization_overrides
    format and saves it at output_json location."""

    with open(npi_yaml_file) as F:
        data = yaml.safe_load(F)

    list_of_tensors_in_fp16 = []
    list_of_tensors_in_fp32 = []

    for key in data:
        if key == 'FP16NodeInstanceNames':
            list_of_tensors_in_fp16.extend(data[key])
        elif key == 'FP32NodeInstanceNames':
            list_of_tensors_in_fp32.extend(data[key])
        else:
            print('Incorrect entry in YAML file: ', key)
            exit(1)

    overrides_dict = {"activation_encodings": {}, "param_encodings": {}}

    activation_encodings_dict = {}
    for tensor in list_of_tensors_in_fp32:
        value = [{"bitwidth": 32, "dtype": "float"}]
        activation_encodings_dict[tensor] = value

    for tensor in list_of_tensors_in_fp16:
        value = [{"bitwidth": 16, "dtype": "float"}]
        activation_encodings_dict[tensor] = value

    overrides_dict["activation_encodings"] = activation_encodings_dict

    with open(output_json, 'w') as fp:
        json.dump(overrides_dict, fp)


def process_encoding(encoding_dict, output_name_to_elem_type):
    new_encodings = {}
    for name, enc in encoding_dict.items():
        name = Helper.sanitize_node_names(name)
        if (name in output_name_to_elem_type.keys()
                and output_name_to_elem_type[name] not in ["INT64", "INT32", "UINT64", "UINT32"]):
            new_encodings[name] = enc
        elif (name not in output_name_to_elem_type.keys()):
            qacc_file_logger.warning(f"Did not find tensor info for: {name} ")
            new_encodings[name] = enc
        else:
            print("Skipping: ", name)
    return new_encodings


def cleanup_quantization_overrides(quant_overrides_json_path: str, model_path: str,
                                   outpath: str) -> str:
    """Cleans up quantization_overrides json based on the input model supplied and saves it to outpath location.
    Using the given model, quantization_overrides encodings are filtered to retain only the valid nodes.
    Note: Clean up is applicable only for ONNX Models.
    For other model types, we return the same quantization_overrides json without any changes
    params:
    quant_overrides_json_path: Absolute path to the quantization_overrides json file
    model_path: Absolute path to the model file.
    outpath: Path to store the cleaned up json file.
    """
    onnx = Helper.safe_import_package("onnx")
    if Helper.get_model_type(model_path) == ModelType.ONNX:
        # Cleanup performed only for ONNX Models
        with open(quant_overrides_json_path, "r") as stream:
            try:
                raw_json_dict = json.load(stream)
                activation_encodings = raw_json_dict["activation_encodings"]
                param_encodings = raw_json_dict["param_encodings"]
            except Exception as e:
                qacc_file_logger.error(f"Error parsing quantization_overrides json. Reason: {e}")
        #Load Model
        original_model = onnx.load(model_path)
        inferred_model = onnx.shape_inference.infer_shapes(original_model)

        # dict to store node_name and node_type mappings
        output_name_to_elem_type = {}
        for elem in inferred_model.graph.value_info:
            # Node Name Sanitization logic applied.
            output_name = Helper.sanitize_node_names(elem.name)
            elem_type = onnx.TensorProto.DataType.Name(elem.type.tensor_type.elem_type)
            output_name_to_elem_type[output_name] = elem_type

        new_activation_encodings = process_encoding(activation_encodings, output_name_to_elem_type)
        new_param_encodings = process_encoding(param_encodings, output_name_to_elem_type)

        cleaned_json_dict = {}
        cleaned_json_dict["activation_encodings"] = new_activation_encodings
        cleaned_json_dict["param_encodings"] = new_param_encodings

        # Write the cleaned json into outpath file
        with open(f"{outpath}", "w") as out_file:
            json.dump(cleaned_json_dict, out_file, indent=4)
        qacc_file_logger.info(f"Cleaned Quantization Overrides JSON dumped at: {outpath}")
    else:
        qacc_file_logger.warning("Quantization Overrides JSON are cleaned only for ONNX Models")
        outpath = quant_overrides_json_path
    return outpath
