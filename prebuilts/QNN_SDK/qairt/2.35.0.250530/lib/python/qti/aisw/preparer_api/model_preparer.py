# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Model Preparer Wrapper using QAIRT converter """
import os
import logging
import json
import numpy as np
from typing import Union, Tuple, Optional, List, Dict, Any
import torch

#import modules
from qti.aisw.tools.core.modules.converter.source_model_converter import (SourceModelConverterInputConfig,
                                                                 TorchModelInfo, SourceModelConverter)
from qti.aisw.tools.core.modules.converter.converter_module import (ConverterInputConfig,
                                                                  QAIRTConverter)
from qti.aisw.tools.core.modules.converter.optimizer_module import (OptimizerInputConfig,
                                                                    QAIRTOptimizer)
from qti.aisw.tools.core.modules.emitter.torch_emitter import (EmitterInputConfig,
                                                               TorchEmitterAndConfigGenerator)
from qti.aisw.emitter.utils.ir_graph_utils import get_ir_graph_from_dlc

import qti.aisw.emitter.utils.model_preparer_utils as mpp_utils
import qti.aisw.emitter.utils.torch_utils as torch_utils
import qti.aisw.emitter.utils.onnx_utils as onnx_utils
import qti.aisw.emitter.utils.config as CustomOpInfo
from qti.aisw.converters.common import ir_graph as ir_graph_lib

logger = logging.getLogger("TorchEmitter")

IrGraph = ir_graph_lib.IrGraph


try:
    import spconv.pytorch as spconv
except ImportError:
    def replace_spconv3d_modules(model): # pylint: disable=unused-argument
        """Dummy placeholder in case spconv doesn't exist"""
else:
    def replace_spconv3d_modules(model):
        """API to replace spconv.SparseConv3d with the Custom implementation of SparseConv3d

        Args:
            model: Pytorch model
        """

        for name, module in model.named_children():
            if isinstance(module, spconv.SparseConv3d):
                # Create the custom layer
                # pylint: disable=no-member
                custom_sp_conv3d_layer = qti.aisw.emitter.custom_ops.CustomSparseConv3DLayer(
                    in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, dilation=module.dilation, bias=(module.bias is not None)
                )
                # Copy the weights
                with torch.no_grad():
                    custom_sp_conv3d_layer.sp_conv_3d.weight.copy_(module.weight)
                    if module.bias is not None:
                        custom_sp_conv3d_layer.sp_conv_3d.bias.copy_(module.bias)
                # Set the custom layer in the model
                custom_sp_conv3d_layer.to(module.weight.device)
                setattr(model, name, custom_sp_conv3d_layer)
            else:
                replace_spconv3d_modules(module)


def convert_onnx_model_to_ir_graph(model_path: str, path: str, filename: str,
                                   skipped_optimizers: Optional[List[str]],
                                   converter_args: Optional[dict[str, Any]]) -> Tuple:
    """ Convert onnx model to Ir graph through QNN onnx converter frontend

    Args:
        model_path: Path of onnx model (on disk or temporary file) to be converted
        path: Path to save exported dlc model
        filename: Filename to save exported dlc model
        skipped_optimizers: optimizer names to disable during onnx simplification
        converter_args: Optional argument with QNN converter specific overrides

    Returns:
        Tuple of Ir Graph, Ir Graph output names and dlc reader obj
    """
    op_graph_output_names = None
    if converter_args is None:
        converter_input_arguments = ConverterInputConfig(input_network=model_path)
    else:
        assert isinstance(converter_args, dict)
        converter_input_arguments = ConverterInputConfig(input_network=model_path, **converter_args)

    with onnx_utils.disable_onnxsim_optimizers(skipped_optimizers):
        # Create converter module object and call API
        converter_obj = QAIRTConverter()
        convert_output = converter_obj.convert(converter_input_arguments)
        op_graph_input_names = [node.op.name for node in
                                convert_output.ir_graph.get_input_nodes_to_graph()]
        op_graph_output_names = convert_output.ir_graph.output_names
        dlc_path = os.path.join(path, filename + ".dlc")
        optimizer_input_arguments = OptimizerInputConfig(framework=convert_output.framework,
                                                         ir_graph=convert_output.ir_graph,
                                                         output_dlc= dlc_path)
        optimizer_obj = QAIRTOptimizer()
        optimizer_output = optimizer_obj.optimize(optimizer_input_arguments)

    ir_graph, dlc_reader_obj = get_ir_graph_from_dlc(optimizer_output.dlc_path)
    ir_graph_output_names = [output_tensor.name() for output_tensor in ir_graph.get_output_tensors_of_graph()]

    if op_graph_output_names is not None and set(ir_graph_output_names) == set(\
            op_graph_output_names):
        return ir_graph, op_graph_input_names, op_graph_output_names, dlc_reader_obj
    return ir_graph, op_graph_input_names, ir_graph_output_names, dlc_reader_obj


def is_prepared_model_input(model_input: Union[Any,]) -> bool:
    """returns True if input would be part of prepared model input """
    return model_input is not None and not isinstance(model_input, bool)


def get_model_inputs(dummy_input: Union[Tuple, List, Dict, torch.Tensor, np.ndarray],
                     input_names: Optional[List[str]]) -> Optional[Union[Tuple, List, Dict, torch.Tensor]]:
    """Get model inputs and outputs if applicable.

    Args:
        dummy_input: Inputs provided for model preparation
        input_names: Optional input names provided for model preparation
        model: Original model

    Returns:
        Tuple of model inputs and outputs to be used during model preparation
    """

    model_input = None
    if input_names is not None:
        if isinstance(dummy_input, (list, tuple)):
            model_input = dict(zip(input_names, dummy_input))
        elif isinstance(dummy_input, dict):
            model_input = dict(zip(input_names, dummy_input.values()))
        else:
            assert len(input_names) == 1
            model_input = {input_names[0]: dummy_input}
    else:
        model_input = dummy_input

    return model_input


def _prepare_model_from_ir_graph(ir_graph: IrGraph,
                                 path: str = './',
                                 filename: str = 'converted_model',
                                 model_name: str = 'ConvertedModel',
                                 keep_linear_without_bias: bool = False,
                                 ignore_encodings: bool = False,
                                 model_input_tensor: Any = None,
                                 model_output_tensor: Any = None,
                                 keep_original_model_structure: bool = False,
                                 block_names_mapping: Dict = None,
                                 ir_graph_input_names= None,
                                 ir_graph_output_names = None,
                                 custom_op_info: CustomOpInfo = None
                                 ) -> Tuple:
    """API to prepare model by taking in the IR Graph and rebuilding a new pytorch model.
    This API writes out the prepared model definition and weights to output files in the given
    path location.

    Two files will be output, all with the given filename:
        - a .py file containing a model definition of the prepared model
        - a .safetensor file containing the weights to load for the prepared model and other
        metadata required for LWQ.

    Args:
        ir_graph: IR Graph to prepare
        path: Path to save converted model definition
        filename: Filename to save converted model definition
        model_name: Name of the converted model
        keep_linear_without_bias: Flag variable whether to keep the original linear module after
            preparation QAIRT usually converts Linear(..., bias=False) to MatMul operation if this
            variable is set True, preparer pro will try to keep original Linear to Linear not MatMul
        ignore_encodings: Flag variable whether to extract encoding info already present in IR Graph.
        model_input: model input tensor or tuple of tensors.
        model_output: model output tensor or tuple of tensors.
        keep_original_model_structure:Flag for keeping original model structure in emitter model.
        block_names_mapping: Block name to class name mapping
        ir_graph_input_names: List of the ir_graph input names in order
        ir_graph_output_names: List of the ir_graph output names in order
        custom_op_info: DataClass consisting of op_type_to_module mapping and custom_module_paths

    Returns:
        File paths of model definition file and state dict file
    """

    emitter_input = EmitterInputConfig(input_graph=ir_graph,
                                       backend_name=None,
                                       path=path,
                                       filename=filename,
                                       model_name=model_name,
                                       dummy_model_input=model_input_tensor,
                                       dummy_model_output=model_output_tensor,
                                       keep_linear_without_bias=keep_linear_without_bias,
                                       keep_original_model_structure=keep_original_model_structure,
                                       block_names_mapping=block_names_mapping,
                                       ignore_encodings=ignore_encodings,
                                       ir_graph_input_names=ir_graph_input_names,
                                       ir_graph_output_names=ir_graph_output_names,
                                       custom_op_info=custom_op_info)
    emitter_obj = TorchEmitterAndConfigGenerator()
    emitter_output = emitter_obj.prepare_model(emitter_input)

    mpp_utils.validate_emitter_file_paths({'model_definition_path':emitter_output.model_definition_path,
                                            'state_dict_path': emitter_output.state_dict_path})

    return emitter_output.model_definition_path, emitter_output.state_dict_path


def _prepare_model_from_dlc(dlc_path: str,
                            path: str = './',
                            filename: str = 'converted_model',
                            model_name: str = 'ConvertedModel',
                            keep_linear_without_bias: bool = False,
                            ignore_encodings: bool = False,
                            model_input_tensor: Any = None,
                            model_output_tensor: Any = None,
                            keep_original_model_structure: bool = False,
                            block_names_mapping: Dict[str, str] = None,
                            custom_op_info: CustomOpInfo = None
                            ) -> Tuple[str]:
    """API to prepare model by taking in the DLC (Non quantized) and rebuilding a new pytorch model.
    This API writes out the prepared model definition and weights to output files in the given
    path location.

    Three files will be output, all with the given filename:
        - a .py file containing a model definition of the prepared model
        - a .safetensor file containing the weights to load for the prepared model and other
        metadata required for LWQ.

    Args:
        dlc_path: Path to the non quantized DLC.
        path: Path to save converted model definition
        filename: Filename to save converted model definition
        model_name: Name of the converted model
        keep_linear_without_bias: Flag variable whether to keep the original linear module after preparation
            QNN usually converts Linear(..., bias=False) to MatMul operation
            if this variable is set True, preparer pro will try to keep original Linear to Linear not MatMul
        ignore_encodings: Flag variable whether to extract encoding info already present in IR Graph.
        model_input: model input tensor or tuple of tensors.
        model_output: model output tensor or tuple of tensors.
        keep_original_model_structure:Flag for keeping original model structure in emitter model.
        block_names_mapping: Block name to class name mapping
        custom_op_info: DataClass consisting of op_type_to_module mapping and custom_module_paths

    Returns:
        File paths of model definition file, state dict file and backend aware base config
        file.
    """

    ir_graph, _ = get_ir_graph_from_dlc(dlc_path)
    ir_graph_output_names = [output_tensor.name() for output_tensor in ir_graph.get_output_tensors_of_graph()]

    if not ir_graph:
        error_msg = 'Failed to load IR Graph with ' + dlc_path
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    prepared_model_file_list = _prepare_model_from_ir_graph(ir_graph=ir_graph,
                                                            path=path,
                                                            filename=filename,
                                                            model_name=model_name,
                                                            keep_linear_without_bias=keep_linear_without_bias,
                                                            ignore_encodings=ignore_encodings,
                                                            model_input_tensor=model_input_tensor,
                                                            model_output_tensor=model_output_tensor,
                                                            keep_original_model_structure=keep_original_model_structure,
                                                            block_names_mapping=block_names_mapping,
                                                            ir_graph_output_names=ir_graph_output_names,
                                                            custom_op_info=custom_op_info
                                                            )

    return prepared_model_file_list


def prepare_model_from_onnx(model_path: str, path: str = './', filename: str = 'converted_model',
                            model_name: str ='ConvertedModel', skipped_optimizers: Optional[List[str]]
                            = None, keep_linear_without_bias: bool = False, converter_args: Optional[Dict] = None,
                            custom_op_info: CustomOpInfo = None, return_prepare_model: bool = False) -> Optional[Union[torch.nn.Module, Tuple]]:
    """API to prepare model by taking in the path to an onnx mode, converting to IR graph,
    and rebuilding a pytorch model aligned with QAIRT IR graph.
    This API writes out the prepared model definition and weights to output files in the given
    path location.

    Two files will be output, all with the given filename:
        - a .py file containing a model definition of the prepared model
        - a .safetensor file containing the weights to load for the prepared model and other
        metadata required for LWQ.

    Args:
        model_path: Path of the ONNX model
        path: Path to save converted model definition
        filename: Filename to save converted model definition
        model_name: Name of the converted model
        skipped_optimizers: Optimizer names to disable during onnx simplification
        keep_linear_without_bias: Flag variable whether to keep the original linear module
                                     after preparation QAIRT usually converts Linear(...,
                                     bias=False) to MatMul operation if this variable is set
                                     True, preparer pro will try to keep original Linear to
                                     Linear not MatMul
        converter_args: Optional argument with QAIRT converter specific overrides
        custom_op_info: DataClass consisting of op_type_to_module mapping and custom_module_paths
        return_prepare_model: Flag to return prepared model after loading or not.

    Returns:
        Converted pytorch model
    """

    ir_graph, ir_graph_input_names, ir_graph_output_names, _ = convert_onnx_model_to_ir_graph(
        model_path, path, filename, skipped_optimizers, converter_args=converter_args)
    prepared_model_file_list = _prepare_model_from_ir_graph(ir_graph=ir_graph,
                                                            path=path,
                                                            filename=filename,
                                                            model_name=model_name,
                                                            keep_linear_without_bias=keep_linear_without_bias,
                                                            ir_graph_input_names=ir_graph_input_names,
                                                            ir_graph_output_names=ir_graph_output_names,
                                                            custom_op_info=custom_op_info)
    if return_prepare_model:
        prepared_model = torch_utils.load_torch_model_using_safetensors(model_name, path, filename)
        return prepared_model

    return prepared_model_file_list


def prepare_model(model: torch.nn.Module,
                  dummy_input: Union[torch.Tensor, Tuple],
                  path: str = './',
                  filename: str = 'converted_model',
                  model_name: str = 'ConvertedModel',
                  input_names: Optional[List[str]] = None,
                  output_names: Optional[List[str]] = None,
                  skipped_optimizers: Optional[List[str]] = None,
                  keep_linear_without_bias: bool = False,
                  onnx_export_args: Optional[Dict] = None,
                  converter_args: Optional[Dict] = None,
                  block_names: Optional[List[str]] = None,
                  keep_original_model: bool = True,
                  order_inputs: bool = False,
                  order_outputs: bool = False,
                  keep_original_model_structure: bool = False,
                  custom_op_info: CustomOpInfo = None,
                  return_prepare_model: bool = False) -> Optional[Union[torch.nn.Module, Tuple]]:
    """API to prepare model by taking in the starting model, converting to IR graph, and rebuilding a
    new pytorch model. This API writes out the prepared model definition and weights to output
    files in the given path location. It can also returns a loaded prepared model if the
    flag return_prepare_model is passed as True.

    Two files will be output, all with the given filename:
        - a .py file containing a model definition of the prepared model
        - a .safetensor file containing the weights to load for the prepared model and other
        metadata required for LWQ.

    Args:
        model: Model to prepare
        dummy_input: Dummy input(s) to the model
        path: Path to save converted model definition
        filename: Filename to save converted model definition
        model_name: Name of the converted model
        input_names: Input names to use for the model which appear in the prepared model inputs
        output_names: Output names to use for the model which appear in the onnx model exported
                         during model prepare
        skipped_optimizers: Optimizer names to disable during onnx simplification
        keep_linear_without_bias: Flag variable whether to keep the original linear module
                                     after preparation QAIRT usually converts Linear(...,
                                     bias=False) to MatMul operation if this variable is set
                                     True, preparer pro will try to keep original Linear to
                                     Linear not MatMul
        onnx_export_args: Optional export arguments for torch onnx export. Can come in the
                             form of a dictionary.
        converter_args: Optional argument with QAIRT converter specific overrides
        block_names: Block names to extract separate torch.nn.Module
        keep_original_model: Flag to keep original model, and not move model to meta device. If
                                this flag is set to False, the original model is no longer
                                avalible for reference.
        order_inputs: Flag to order input of prepared model in accordance with source model.
        order_outputs: Flag to order output of prepared model in accordance with source model.
        keep_original_model_structure: Flag for keeping original model structure in prepared model
        custom_op_info: DataClass consisting of op_type_to_module mapping and custom_module_paths
        return_prepare_model: Flag to return prepared model after loading or not.

    Returns:
        Converted pytorch model
    """
    mpp_utils.validate_inputs(dummy_input, order_inputs, input_names, is_prepared_model_input)

    if not keep_original_model:
        info_msg = ('keep_original_model is set to False. Model will be moved to meta device and original model '
                    'will no longer be able for reference.')
        logger.info(info_msg)

    # Replace spconv.SparseConv3d module with our own custom implementation
    replace_spconv3d_modules(model)

    # Source Model Conversion : ConvertToONNX modular API
    onnx_input_names = None if order_inputs else input_names
    convert_to_onnx_model_info = TorchModelInfo(order_inputs=order_inputs,
                                                order_outputs=order_outputs,
                                                input_names=onnx_input_names,
                                                output_names=output_names,
                                                block_names=block_names,
                                                return_prepare_model=return_prepare_model)
    input_config = SourceModelConverterInputConfig(source_model=model,
                                                   dummy_input=dummy_input,
                                                   path=path,
                                                   filename=filename,
                                                   export_args=onnx_export_args,
                                                   skipped_optimizers=skipped_optimizers,
                                                   model_info=convert_to_onnx_model_info
                                                   )
    convert_to_onnx_obj = SourceModelConverter()
    convert_to_onnx_output = convert_to_onnx_obj.torch2onnx(input_config)
    source_model_metadata = convert_to_onnx_output.source_model_metadata

    # ONNX to IR Graph Conversion : QAIRTConverter/QAIRTOptimizer modular API
    if order_inputs:
        converter_args = mpp_utils.update_converter_args_input_names(converter_args, input_names,
                                                                     dummy_input,
                                                                     source_model_metadata.onnx_model_input_name)
    ir_graph, ir_graph_input_names, ir_graph_output_names, _ = convert_onnx_model_to_ir_graph(
        model_path=convert_to_onnx_output.onnx_path, path=path, filename=filename,
        skipped_optimizers=skipped_optimizers, converter_args=converter_args)

    mpp_utils.validate_inputs_for_ir_graph(dummy_input, ir_graph, order_inputs,
                                           is_prepared_model_input)

    module_name_to_ir_op = None
    if return_prepare_model:
        module_name_to_ir_op = torch_utils.create_module_name_and_ir_op_dict(model, ir_graph)

    if not keep_original_model:
        model.to("meta")

    # IR Graph to Emitter Pytorch Model Conversion : TorchEmitterAndConfigGenerator modular API
    model_input = get_model_inputs(dummy_input, input_names) if order_inputs else None
    emitter_files = _prepare_model_from_ir_graph(ir_graph=ir_graph,
                                                 path=path,
                                                 filename=filename,
                                                 model_name=model_name,
                                                 keep_linear_without_bias=keep_linear_without_bias,
                                                 model_input_tensor=model_input,
                                                 model_output_tensor=source_model_metadata.model_output,
                                                 keep_original_model_structure=keep_original_model_structure,
                                                 block_names_mapping=source_model_metadata.block_name_mapping,
                                                 ir_graph_input_names=ir_graph_input_names,
                                                 ir_graph_output_names=ir_graph_output_names,
                                                 custom_op_info=custom_op_info
                                                 )

    if return_prepare_model:
        # load emitter model in memory
        prepared_model = torch_utils.load_torch_model_using_safetensors(model_name, path, filename)

        metadata = prepared_model.mpp_meta.get('metadata', '')
        json_data = json.loads(metadata)
        ir_op_name_list = json_data.get('ir_op_name_list', None)
        torch_utils.match_model_settings(source_model_metadata.model_setting, prepared_model)
        torch_utils.match_module_settings(source_model_metadata.state_saving_module_names,
                                          prepared_model)
        module_name_mapping = torch_utils.get_module_name_to_module_name_mapping(module_name_to_ir_op,
                                                                                 source_model_metadata.module_name_list,
                                                                                 ir_op_name_list,
                                                                                 path, filename,
                                                                                 prepared_model,
                                                                                 keep_original_model_structure)
        torch_utils.set_module_name_to_module_mapping(module_name_mapping, prepared_model)
        return prepared_model

    return emitter_files


def update_state_dict_weights(source_state_dict_path: str, destination_state_dict_path: str,
                              mapping_file_path: str, updated_state_dict_path: Optional[str] =
                              None):
    """Given a source state dict, destination state dict, and name mapping information, update weights
    in the destination state dict with those in the source state dict.

    Args:
        source_state_dict_path: Path to source state dict
        destination_state_dict_path: Path to destination state dict
        mapping_file_path: Json file containing name mappings
        updated_state_dict_path: If not None, updated state dict is saved to this file. Otherwise, the existing
            destination state dict will be overwritten with the updated weights.
    """
    source_state_dict = torch.load(source_state_dict_path)
    destination_state_dict = torch.load(destination_state_dict_path)
    with open(mapping_file_path) as mapping_json:
        name_mappings = json.load(mapping_json)

    for name, weight in source_state_dict.items():
        rightmost_period_idx = name.rfind('.')
        module_name, weight_name = name[:rightmost_period_idx], name[rightmost_period_idx + 1:]
        if module_name not in name_mappings:
            warning_str = (f'Mapping for module with name {module_name} not found in prepared model.'
                           f' Skipping updating of weight {name}.')
            logger.warning(warning_str)
            continue

        destination_module_names = name_mappings[module_name]
        found_weight = False
        for destination_module_name in destination_module_names:
            destination_weight_name = destination_module_name + '.' + weight_name
            if destination_weight_name in destination_state_dict and \
                    destination_state_dict[destination_weight_name].shape == weight.shape:
                found_weight = True
                destination_state_dict[destination_weight_name] = weight
                break
        if not found_weight:
            # A bias with all zeros may not exist in the prepared model weights
            if weight_name == 'bias' and torch.equal(torch.zeros_like(weight), weight):
                continue

            warning_str = (f'Unable to find corresponding prepared model module with weight for '
                           f'original weight {name}. Skipping updating of weight.')
            logger.warning(warning_str)

    if updated_state_dict_path is not None:
        torch.save(destination_state_dict, updated_state_dict_path)
    else:
        torch.save(destination_state_dict, destination_state_dict_path)
