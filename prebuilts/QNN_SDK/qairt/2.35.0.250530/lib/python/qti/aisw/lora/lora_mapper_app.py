# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import yaml
import os
from qti.aisw.lora.helpers import is_attach_point


def get_pytorch_to_onnx_mapping(lora_config, lora_config_path):
    # validate each onnx name maps to one pytorch name
    def validate_config(pytorch_to_onnx, pytorch_to_onnx_file):
        seen_onnx_names = set()
        for onnx_names in pytorch_to_onnx.values():
            for onnx_name in onnx_names:
                if onnx_name in seen_onnx_names:
                    raise ValueError("{} has repeated onnx name, {}.".format(pytorch_to_onnx_file, onnx_name))
                seen_onnx_names.add(onnx_name)

    pytorch_to_onnx_file = lora_config['attach_point_onnx_mapping']
    if not os.path.isabs(pytorch_to_onnx_file):
        base_path = os.path.dirname(lora_config_path)
        pytorch_to_onnx_file = os.path.join(base_path, pytorch_to_onnx_file)

    with open(pytorch_to_onnx_file) as file:
        pytorch_to_onnx = json.load(file)

    validate_config(pytorch_to_onnx, pytorch_to_onnx_file)
    return pytorch_to_onnx

def get_lora_config(lora_config_path):
    with open(lora_config_path) as file:
        config = yaml.safe_load(file)

    return config

def get_adapter_configs(lora_config, lora_config_path):
    def get_adapter_config_paths(lora_config):
        adapter_names = set()
        absolute_paths = list()
        for adapter_info in lora_config['adapter']:
            if adapter_info['name'] in adapter_names:
                raise ValueError("Invalid LoRA YAML : adapter name, {}, is not unique".format(adapter_info['name']))
            adapter_names.add(adapter_info['name'])

            path = adapter_info["lora_config"]
            if os.path.isabs(path):
                absolute_paths.append(path)
            else:
                base_path = os.path.dirname(lora_config_path)
                absolute_path = os.path.join(base_path, path)
                absolute_paths.append(absolute_path)
        return absolute_paths

    def validate_adapter_configs(adapter_config):
        expected_keys = set(['name', 'rank', 'target_modules'])
        if not expected_keys.issubset(set(adapter_config.keys())):
            raise ValueError("Adapter config with name, {}, should have the keys: {}".format(adapter_config['name'], expected_keys))

    config_paths = get_adapter_config_paths(lora_config)
    adapter_configs = dict()

    for path in config_paths:
        with open(path) as file:
            adapter_config = json.load(file)
            validate_adapter_configs(adapter_config)
            adapter_configs[adapter_config['name']] = adapter_config

    return adapter_configs

def update_adapter_configs(adapter_configs, pytorch_to_onnx_map):
    for adapter_name, adapter_config in adapter_configs.items():
        adapter_target_modules = adapter_config.pop("target_modules")
        new_adapter_target_modules = list()

        for pytorch_name in pytorch_to_onnx_map:
            if is_attach_point(pytorch_name, adapter_target_modules):
                if len(pytorch_to_onnx_map[pytorch_name]) != 1:
                    raise ValueError(
                        "Each target module should map to 1 onnx name. {} maps to {} onnx names." \
                        .format(pytorch_name, len(pytorch_to_onnx_map[pytorch_name])))
                onnx_name = pytorch_to_onnx_map[pytorch_name][0]
                new_adapter_target_modules.append(onnx_name)
        adapter_config["target_operator_names"] = new_adapter_target_modules

    return adapter_configs

def save_output_files(updated_adapter_configs, lora_config_path, lora_config, output_dir):
    def get_new_file_name(path):
        file_name = os.path.basename(path)
        base_name, extension = os.path.splitext(file_name)
        new_file_name = base_name + "_updated" + extension
        return new_file_name

    def save_updated_adapter_config(adapter_info, updated_adapter_configs, output_dir, lora_config_path):
        adapter_name = adapter_info['name']
        adapter_path = adapter_info['lora_config']
        adapter_config = updated_adapter_configs[adapter_name]

        new_file_name = get_new_file_name(adapter_path)
        new_adapter_path = os.path.join(output_dir, new_file_name)

        with open(new_adapter_path, "w") as f:
            json.dump(adapter_config, f, indent=4)
        return new_adapter_path

    def save_updated_lora_config(lora_config, lora_config_path, output_dir):
        def make_all_paths_absolute(lora_config, lora_config_path):
            def make_path_absolute(path, lora_config_path):
                if path and not os.path.isabs(path):
                    lora_config_directory = os.path.dirname(os.path.abspath(lora_config_path))
                    path = os.path.join(lora_config_directory, path)
                return path

            lora_config['attach_point_onnx_mapping'] = make_path_absolute(lora_config['attach_point_onnx_mapping'], lora_config_path)
            use_case_names = set()
            for use_case_info in lora_config['use-case']:
                if use_case_info["name"] in use_case_names:
                    raise ValueError("Invalid LoRA YAML : use-case name, {}, is not unique".format(use_case_info["name"]))
                use_case_names.add(use_case_info["name"])

                use_case_info['model_name'] = make_path_absolute(use_case_info['model_name'], lora_config_path)

                # quant_overrides and quant_updatable_tensors are optional for the lora config yaml
                if 'quant_overrides' in use_case_info:
                    use_case_info['quant_overrides'] = make_path_absolute(use_case_info['quant_overrides'], lora_config_path)
                if 'quant_updatable_tensors' in use_case_info:
                    use_case_info['quant_updatable_tensors'] = make_path_absolute(use_case_info['quant_updatable_tensors'], lora_config_path)

        new_file_name = get_new_file_name(lora_config_path)
        new_lora_config_path = os.path.join(output_dir, new_file_name)
        make_all_paths_absolute(lora_config, lora_config_path)

        with open(new_lora_config_path, 'w') as f:
            yaml.dump(lora_config, f)
        return new_lora_config_path


    print("New files saved at:")
    for adapter_info in lora_config['adapter']:
        new_adapter_path = save_updated_adapter_config(adapter_info, updated_adapter_configs, output_dir, lora_config_path)
        adapter_info['lora_config'] = new_adapter_path
        print(new_adapter_path)

    new_lora_config_path = save_updated_lora_config(lora_config, lora_config_path, output_dir)
    print(new_lora_config_path)


def resolve_attach_point_name(lora_config_path, output_dir):
    """
    Creates new adapter config files with onnx operator names and new lora onnx
    config file with these new adapter config files.

    Parameters:
        lora_config_path (str):
            yaml file containing the paths of adapter configs and pytorch to
            onnx node mappings.

        output_dir (str):
            Directory to save the new files.

    Returns:
        None
    """

    lora_config = get_lora_config(lora_config_path)
    pytorch_to_onnx_map = get_pytorch_to_onnx_mapping(lora_config, lora_config_path)
    adapter_configs = get_adapter_configs(lora_config, lora_config_path)

    updated_adapter_configs = update_adapter_configs(adapter_configs, pytorch_to_onnx_map)

    save_output_files(updated_adapter_configs, lora_config_path, lora_config, output_dir)