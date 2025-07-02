# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import torch
import logging
import os
import importlib
import sys
from safetensors.numpy import load as load_safetensor
import json
from collections import defaultdict
from argparse import Namespace
import re
import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger("model preparer pro")

import qti.aisw.emitter.ir_graph_op_handler as op_handler
PEFT_AVAILABLE = True
try:
    import peft
except ImportError:
    PEFT_AVAILABLE = False

@dataclass
class TorchModelMetadata():
    block_name_mapping: Optional[Dict] = field(default=None)
    model_output: Optional[Any] = field(default=None)
    onnx_model_input_name: Optional[list[str]] = field(default=None)
    model_setting: Optional[dict] = field(default_factory=dict)
    module_name_list: Optional[List[str]] = field(default=None)
    state_saving_module_names: Optional[list[str]] = field(default=None)


class ChildModuleCounter:
    """
    Class for counting the number of child modules
    """
    def __init__(self, op_name_list: list[str]):
        self.counter = defaultdict(int)
        for op_name in op_name_list:
            if not op_name.startswith('/'):
                continue

            splitted_module_path = op_name.split('/')
            for module_path_idx in range(1, len(splitted_module_path)):
                current_module_path = '/'.join(splitted_module_path[:module_path_idx + 1])
                self.counter[current_module_path] += 1

    def get_children_count(self, module_path: str) -> int:
        """
        Get the number of child modules for given module path

        :param module_path:  The module path to get the number of child modules
        :return: The number of child modules for given module path
        """
        return self.counter[module_path]

def get_model_metadata(source_model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple],
                       model_info: Namespace)-> TorchModelMetadata:
    """
    Get the metadata for source model.

    :param source_model: Source Pytorch Model
    :param dummy_input: Dummy Input
    :param model_info: Information on the source model required for extracting metadata
    :return: DataClass with all the metadata extracted from source model.
    """

    model_metadata = TorchModelMetadata()
    #get block name to class name mapping
    if model_info.block_names is not None:
        block_name_mapping = {}
        for block_name in model_info.block_names:
            refined_name = re.sub(r"\[(\d+)]", r".\1", block_name)
            named_module = functools.reduce(getattr, refined_name.split("."), source_model)
            class_name = named_module.__class__.__name__
            block_name_mapping[block_name] = class_name
        model_metadata.block_name_mapping = block_name_mapping

    # get model output
    if model_info.order_outputs:
        with torch.inference_mode():
            if isinstance(dummy_input, (list, tuple)):
                model_output = source_model(*dummy_input)
            elif isinstance(dummy_input, dict):
                model_output = source_model(**dummy_input)
            else:
                model_output = source_model(dummy_input)
        model_metadata.model_output = model_output

    if model_info.return_prepare_model:
        # for model_setting
        param = next(source_model.parameters(), None)
        model_metadata.model_setting['device'] = 'cpu' if param is None else param.device.type
        model_metadata.model_setting['training'] = source_model.training

        # for module_setting
        group_norm_modules = []
        for name, module in source_model.named_modules():
            if isinstance(module, torch.nn.GroupNorm):
                # GroupNorm is broken down into InstanceNorm followed by Mul and Add, where the original GroupNorm weight
                # and bias now feed into the Mul and Add as standalone parameters.
                # To match the behavior of the original model, if the original GroupNorm was not affine, set
                # requires_grad = False for the prepared model's weight and bias feeding into Mul and Add.
                if not module.affine:
                    group_norm_modules.append(name)
        model_metadata.state_saving_module_names = group_norm_modules

        # for getting module names in source model
        module_names = [module_name for module_name, _ in source_model.named_modules()]
        if PEFT_AVAILABLE:
            if isinstance(source_model, peft.PeftModel):
                # Drop top level prefix in module name from Peft model that can cause module name mismatch
                module_names = [".".join(module_name.split(".")[1:]) for module_name in module_names]
        model_metadata.module_name_list = module_names

    return model_metadata

def load_torch_model_using_safetensors(model_name: str, path: str, filename: str) -> torch.nn.Module:
    """
    Load the pytorch model from the given path and filename.
    NOTE: The model can only be saved by saving the state dict. Attempting to serialize the entire model will result
    in a mismatch between class types of the model defined and the class type that is imported programatically.

    :param model_name: Name of model
    :param path: Path where the pytorch model definition file is saved
    :param filename: Filename of the pytorch model definition and the safetensors weight file

    :return: Imported pytorch model with embeded metadata
    """

    model_path = os.path.join(path, filename + '.py')
    if not os.path.exists(model_path):
        logger.error('Unable to find model file at path %s', model_path)
        raise AssertionError('Unable to find model file at path ' + model_path)

    # Import model's module and instantiate model
    spec = importlib.util.spec_from_file_location(filename, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[filename] = module
    spec.loader.exec_module(module)
    model = getattr(module, model_name)()

    # Load state dict using safetensors file
    state_dict_path = os.path.join(path, filename + '.safetensors')
    if not os.path.exists(state_dict_path):
        logger.error('Unable to find state dict file at path %s', state_dict_path)
        raise AssertionError('Unable to find state dict file at path ' + state_dict_path)
    state_dict, meta_data = _get_metadata_and_state_dict(state_dict_path)
    model.load_state_dict(state_dict, strict=False)

    # Sets the MPP meta data extracted from safetensors file into the model as an atribute
    # so that it can be extracted and saved at the time of weights export.
    model.__setattr__('mpp_meta', meta_data)
    return model


def _get_metadata_and_state_dict(safetensor_file_path: str) -> [dict, dict]:
    """
    Extracts the state dict from a numpy format safetensors as well as metadata.
    Converts the state_dict from numpy aray to torch tensors.

    :param safetensor_file_path: Path of the safetensor file.
    :return: state dict in torch.Tensor format and metadata
    """

    with open(safetensor_file_path, "rb") as f:
        data = f.read()

    # Get the header length to extract the metadata
    header_length = int.from_bytes(data[:8], "little", signed=False)
    meta_data = json.loads(data[8:8 + header_length].decode()).get('__metadata__', {})

    # Load the state dict and convert it to torch tensor
    state_dict = load_safetensor(data)
    state_dict = {k: torch.from_numpy(v) for k, v in state_dict.items()}

    return state_dict, meta_data


def get_device(model):
    """
    Function to find which device is model on
    Assumption : model is on single device
    :param model:
    :return: Device on which model is present
    """
    return next(model.parameters()).device


def match_model_settings(model_setting: dict, model_to_set: torch.nn.Module):
    """
    Match training and device settings of the model_to_match with those of source model .

    :param model_setting: Model setting of the source Model.
    :param model_to_set: Model to set
    """
    model_to_set.train(model_setting['training'])
    try:
        if get_device(model_to_set) != model_setting['device']:
            model_to_set.to(model_setting['device'])
    except StopIteration:
        # If there are no parameters in the model, get_device will have nothing to iterate over
        pass


def match_module_settings(state_saving_module_names: list,
                          prepared_model: torch.nn.Module):
    """
    Match settings for particular module types between the original model and the prepared model.

    :param state_saving_module_names: Name of the modules which requires to be set.
    :param prepared_model: Prepared PyTorch model to set module settings for
    """
    def disable_requires_grad(target_module: torch.nn.Module, attr_name: str):
        """
        Set requires_grad to False if attribute exists in target module

        :param target_module: Target torch.nn.Module
        :param attr_name: Attribute to set requires_grad to False
        """
        if hasattr(target_module, attr_name):
            getattr(target_module, attr_name).requires_grad = False

    for name in state_saving_module_names:
        disable_requires_grad(prepared_model, name + '_weight')
        disable_requires_grad(prepared_model, name + '_bias')


def is_leaf_module(module):

    """Utility function to determine if the given module is a leaf module - that is, does not have children modules
    :return:
        True if the module is a leaf, False otherwise
    """
    module_list = list(module.modules())

    # pylint: disable=unidiomatic-typecheck
    return bool(len(module_list) == 1)


def create_module_name_and_ir_op_dict(model: torch.nn.Module, ir_graph: Any) -> Dict[str,List[Any]]:
    """
    Create dictionaries mapping original PyTorch modules name to QNN-IR nodes.

    :param model: Original model
    :param ir_graph: QNN IR Graph
    :return: A dictionary mapping PyTorch module name to QNN-IR nodes
    """
    def is_trim_possible(left: str, right: str) -> bool:
        split_words = right.split('.')
        return len(split_words) > 1 and split_words[0] == left

    name_to_ir_op = defaultdict(list)
    for op_name, op in {op.name: op for op in ir_graph.get_ops()}.items():
        if '/' in op_name:
            # Node including / character is mainly from torch.onnx.export
            # IR nodes converted from ONNX can have the following names:
            # 1) /transformer/h.1/attn/attention/Softmax, 2) /lm_head/MatMul
            # Drop ONNX token type which is redundant when name matching
            # /transformer/h.1/attn/attention/Softmax => ['', 'transformer', 'h.1', 'attn', 'attention']
            # /lm_head/MatMul => ['', 'lm_head']
            tokens = op_name.split('/')[:-1]
            # Trimming redundant information from node's name containing scope
            # /layer2/layer2.0/downsample/downsample.0 => ['', 'layer2.0', 'downsample.0']
            # But, we won't trim if parent and child names are same and don't contain dot (.)
            # Because it's possible to have same module names from nested network structure
            # /linear/linear/Gemm => ['', 'linear', 'linear']
            tokens = [tokens[i] for i in range(len(tokens) - 1) if not is_trim_possible(tokens[i], tokens[i + 1])] + tokens[-1:]
            # Join tokens with dot character to match with PyTorch module name
            root_name = '.'.join([token for token in tokens if token])
            # Handle special cases:
            # 1) module corresponding to functional operator - In this case, do not drop ONNX token type instead treat
            # ONNX token type as name. /layer1.0/Add => /layer1.0.Add.
            # If there is another 'Add' in same namespace, it is guranteed to have different name (layer1.0.Add_1)
            # while exporting.
            # 2) module correpsonding to reused operator - In this case, AttributeError will be raised and no
            # extra steps required.
            try:
                parent_module = model.get_submodule(root_name)
                if not is_leaf_module(parent_module):
                    root_name = root_name + '.' + op_name.rsplit('/', maxsplit=1)[1]
            except AttributeError:
                pass
        elif '#' in op_name:
            # Node including # character is mainly from OnnxSaver
            # IR nodes converted from ONNX can have the following names: ada#1.end, relu#0-1.end
            # For the correct mapping, we need to extract the root node name first
            root_name, _, _ = op_name.partition('#')
        else:
            root_name = op_name
        name_to_ir_op[root_name].append(op)
    return name_to_ir_op


def get_named_module(model, name):
    """
    Given the name, get the target module in the model
    :param model: Model that contains the target module
    :param name: Name of the target module
    :return:
    """
    return functools.reduce(getattr, name.split("."), model)


def get_module_name_to_module_name_mapping(name_to_ir_op: Dict, module_name_list: List,
                                           ir_op_name_list: list,
                                           path: str, filename: str, prepared_model: torch.nn.Module,
                                           keep_original_model_structure: bool) -> \
        Dict[str, List[str]]:
    """
    Gets a dict attribute of prepared module which contains the mapping from original module names
    to the list of associated module names in prepared_module and store them in json format.

    :param name_to_ir_op: Source Modules name to corresponding IR op mapping.
    :param module_name_list: Source Model name list.
    :param ir_op_name_list: IR ops name list.
    :param path: Path to save the mapping
    :param filename: Filename to save the mapping
    :param prepared_model: Prepared PyTorch model
    :param keep_original_model_structure: Flag for keeping original model structure in prepared model
    :return: Mapping between original module names and prepared module names
    """
    op_handler.KEEP_ORIGINAL_MODEL_STRUCTURE = keep_original_model_structure
    original_module_name_to_prepared_module_name = {}
    child_module_counter = ChildModuleCounter(ir_op_name_list)
    for name, ops in name_to_ir_op.items():
        if name in module_name_list:
            if not ops:
                logger.warning('No IR Op found in the prepared model associated with the module name %s', name)
                continue
            prepared_module_name_list = []
            for op in ops:
                op_name = op.name
                module_path, _, op_type = op_name.rpartition('/')

                if keep_original_model_structure:
                    # For structure preserved model preparation,
                    # Prepared module name can be taken from module_path by replacing '/' with '.'
                    # e.g., '/top_level_modules.0/linears.0/fc' -> 'top_level_modules.0.linears.0.fc'
                    prepared_module_name = '.'.join(
                        [token for token in module_path.split('/') if token]
                    )

                    # In case of multiple ops has been generated by preparation such as Conv -> [Conv, Permute]
                    # This will not be one_to_one op and we need to add op_type to get prepared_module correctly
                    if not op_handler.is_one_to_one_op(op_name, child_module_counter):
                        prepared_module_name = f'{prepared_module_name}.{op_type}'
                else:
                    prepared_module_name = op_handler.get_op_name(op_name)

                try:
                    get_named_module(prepared_model, prepared_module_name)
                    prepared_module_name_list.append(prepared_module_name)
                except AttributeError:
                    logger.warning('No module found in the prepared model associated with the IR Op %s', op.name)
            if prepared_module_name_list:
                original_module_name_to_prepared_module_name[name] = prepared_module_name_list

    # Save mapping between original model and prepared model
    mapping_file = os.path.join(path, filename + '.json')
    with open(mapping_file, 'w') as f:
        json.dump(original_module_name_to_prepared_module_name, f, indent=4)

    return original_module_name_to_prepared_module_name


def set_module_name_to_module_mapping(name_mapping: Dict[str, List[str]], prepared_model: torch.nn.Module):
    """
    Sets a dict attribute of prepared module which contains the mapping from original module names
    to the list of associated modules in prepared_module.

    :param name_mapping: Mapping between original module names and prepared module names
    :param prepared_model: Prepared PyTorch model
    """
    original_module_to_prepared_module = {}
    for module_name, prepared_module_names in name_mapping.items():
        original_module_to_prepared_module[module_name] = [get_named_module(prepared_model, name) for name in
                                                           prepared_module_names]

    prepared_model.name_to_module_dict = original_module_to_prepared_module


def import_from_path(module_name, file_path):
    """
    Import a module from a given file path

    :param module_name: The name to assign to the module
    :param file_path: The path to the module file
    :return: The imported module
    """

    file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    # if the module is already present return it
    if module_name in sys.modules and sys.modules[module_name].__spec__ == spec:
        return sys.modules[module_name]

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

