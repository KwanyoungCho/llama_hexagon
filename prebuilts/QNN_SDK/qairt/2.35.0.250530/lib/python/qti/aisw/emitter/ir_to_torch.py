# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


import os
import re
import logging
import json
import numpy as np
from typing import Union, Tuple, Optional, List, Dict, Set, Iterable, Any
from safetensors.numpy import save_file

import qti.aisw.emitter.ir_graph_op_handler as op_handler
from qti.aisw.emitter.op_handler_factory import OpHandlerState
from qti.aisw.emitter.utils.axis_tracking_utils import combine_transpose_order
import qti.aisw.emitter.utils.ir_graph_utils as ir_graph_utils
import qti.aisw.emitter.utils.model_preparer_utils as utils
from qti.aisw.emitter.utils.config import CustomOpInfo

from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common import ir_graph as ir_graph_lib


logger = logging.getLogger('Torch Emitter')

IrGraph = ir_graph_lib.IrGraph
IrOp = ir_graph_lib.IrOp

class TorchEmitter:
    '''
    Build a new pytorch model from IR graph.
    '''
    def __init__(self,
                 path: str,
                 filename: str,
                 model_name: str,
                 dummy_model_input: Union[Dict, tuple, np.array],
                 dummy_model_output: Union[Dict, tuple, np.array],
                 keep_linear_without_bias: bool,
                 keep_original_model_structure: bool,
                 block_names_mapping: dict = None,
                 ignore_encodings: bool = False,
                 ir_graph_output_names: List = None,
                 ir_graph_input_names: List = None,
                 custom_op_info: CustomOpInfo = None):
        '''
        :param path: Path to save newly built pytorch model definition
        :param filename: Filename to save newly built pytorch model definition
        :param model_name: Name of the converted model
        :param dummy_model_input: Dummy input of the Original Model
        :param dummy_model_output: Dummy output of the original Model,
        :param keep_linear_without_bias: Flag variable whether to keep the original linear module after preparation
                QNN usually converts Linear(..., bias=False) to MatMul operation
                if this variable is set True, preparer pro will try to keep original Linear to Linear not MatMul,
        :param block_names_mapping: Dict of block names to the container class.
        :param keep_original_model_structure: Flag for keeping original model structure in prepared model
        :param ir_graph_input_names: List of the ir_graph input names in order.
        :param ir_graph_output_names: List of the ir_graph output names in order.
        :param ignore_encodings: Flag variable whether to extract encoding info already present in IR Graph.
        '''
        self.path = path
        self.filename = filename
        self.model_name = model_name
        self.keep_linear_without_bias = keep_linear_without_bias
        self.dummy_model_input = dummy_model_input
        self.dummy_model_output = dummy_model_output
        self.block_names_mapping = block_names_mapping
        self.keep_original_model_structure = keep_original_model_structure
        self.ignore_encodings = ignore_encodings
        self.ir_graph_input_names = ir_graph_input_names
        self.ir_graph_output_names = ir_graph_output_names
        self.custom_op_info = custom_op_info

    def _handle_submodules(self, op: IrOp):
        '''
        Function for adding submodules found when initializing a given op.
        :param op: Op about to be initialized
        '''
        op_name = op.name
        child_module_counter = self.op_handler_state.child_module_counter
        module_path, _, _ = op_name.rpartition('/')

        # NOTE: Starting with `Reshape_post_` prefix is not correct module name path
        if module_path.startswith("Reshape_post_"):
            return

        if module_path and module_path not in self.op_handler_state.created_submodules:
            split_module_path = module_path.split('/')

            # For module name with /a/b/c, iterate in order of /a, /a/b, /a/b/c
            # so that upper modules should be initialized earlier than submodules.
            for module_path_idx in range(1, len(split_module_path)):
                current_module_path = '/'.join(split_module_path[:module_path_idx + 1])

                if current_module_path not in self.op_handler_state.created_submodules:
                    if module_path_idx + 1 <= len(split_module_path) or not op_handler.is_one_to_one_op(op_name, child_module_counter):

                        # 1) Check if leaf module is ModuleList pattern (e.g., /transformer/h.1/custom_module/ln.0)
                        has_module_list_pattern = re.findall(r"(\w+)\.(\d+)", split_module_path[module_path_idx])

                        # 2) If it has ModuleList pattern, we should track the number of child modules
                        if has_module_list_pattern:
                            # 2-1. [(ln, 0)] from /transformer/h.1/custom_module/ln.0 case
                            leaf_module, current_index = has_module_list_pattern[0]

                            # 2-2. Generate module_list_path that will be instantiated by nn.ModuleList()
                            # ['transformer', 'h.1', 'custom_module'] + ['ln']
                            module_list_path = "/".join(split_module_path[:module_path_idx] + [leaf_module])

                            # 2-3. Need to replace dot digit pattern in parent modules with bracket pattern
                            # 'transformer/h.1/custom_module/ln' -> 'transformer/h[1]/custom_module/ln'
                            module_list_path = re.sub(r"\.(\d+)", r"[\1]", module_list_path)

                            # 2-4. Update module_list_to_submodule_count dictionary that will be used instantiation
                            module_list_key = op_handler.get_op_name(module_list_path)
                            previous_index = self.op_handler_state.module_list_to_submodule_count[module_list_key]
                            self.op_handler_state.module_list_to_submodule_count[module_list_key] = max(int(current_index) + 1, previous_index)
                        else:
                            # Need to replace dot digit pattern in parent modules with bracket pattern in one to one op as well
                            # 'transformer/h.1/custom_module/ln.0/out_proj' -> 'transformer/h[1]/custom_module/ln[0]/out_proj'
                            module_path_with_bracket_pattern = re.sub(r"\.(\d+)", r"[\1]", current_module_path)
                            init_str = f"\t\tself.{op_handler.get_op_name(module_path_with_bracket_pattern)} = torch.nn.Module()"
                            self.op_handler_state.created_submodues_init.append(init_str)

                    self.op_handler_state.created_submodules.add(current_module_path)

    def _generate_axis_information(self):

        op_list = self.op_handler_state.ir_graph.get_ops()
        for op in op_list:
            op_type = op.type
            is_custom_op = ir_graph_utils.is_custom_ir_op(op)
            if ((not is_custom_op and  op_type in op_handler.ir_to_handler_dict.keys()) or
                (is_custom_op and op_type in self.custom_op_info.op_type_to_module)):
                if self.keep_original_model_structure:
                    self._handle_submodules(op)

                op_handler.generate_input_and_constant_axis_format(op, self.op_handler_state)
                if op_type == "Conv3d" and op.attrs_dict.get('reuse_sparse_indicies', False):
                    op_type = "SpConv3d"
                # pylint: disable=protected-access
                if is_custom_op:
                    op_handler.CustomOpHandler(op, self.op_handler_state)._generate_axis_information()
                else:
                    op_handler.ir_to_handler_dict.get(op_type)(op, self.op_handler_state)._generate_axis_information()
            else:
                error_msg = 'Encountered unknown op type ' + op_type + '. Unable to proceed with model preparation.'
                logger.error(error_msg)
                raise RuntimeError(error_msg)

    def _optimize_axis_info(self):
        '''
        Performs the following operation over the axis tracking information
        1. Optimizes out consecutive transposes into single transpose.
        2. Shifts input transpose for an op to the output of the producers of that tensor wherever possible.
        '''

        def _get_op_axis_info(op: IrOp):
            '''
            Get the OpAxisInfo for the IrOp.

            :param op: IrOp for which axis infor needs to be retrieved
            :return: OpAxisInfo  for the given op
            '''
            op_name = op_handler.ir_to_handler_dict.get(op_type, op_handler.CustomOpHandler)(op, self.op_handler_state).op_name
            return self.op_handler_state.op_axis_info[op_name]

        op_list = self.op_handler_state.ir_graph.get_ops()
        for op in op_list:
            op_type = op.type
            if op_type == "Conv3d" and op.attrs_dict.get('reuse_sparse_indicies', False):
                op_type = "SpConv3d"
            op_name = op_handler.ir_to_handler_dict.get(op_type, op_handler.CustomOpHandler)(op, self.op_handler_state).op_name
            for op_tensor in op.outputs():
                consumers = list(op_tensor.get_consumers())
                n_consumers = len(consumers)
                # If number of consumer is less than 1. It means it is output node so no optimization needed
                if n_consumers < 1:
                    continue

                # In case the consumer of the tensor is Transpose and the number of consumer is 1, we can skip
                # the optimization as this case will get handled in OpHandler for Transpose Op
                if n_consumers == 1 and consumers[0].type == 'Transpose':
                    continue

                # If consumer list is of length 1 ten we can optimize or in case all
                # the consumer has same input transpose requirement we can optimize
                first_op_axis_info = _get_op_axis_info(consumers[0])
                same_input_transpose_order = True
                first_op_input_transform = first_op_axis_info.input_transform.get(op_tensor.name(), None)
                for consumer in consumers[1:]:
                    op_axis_info = _get_op_axis_info(consumer)
                    if op_axis_info.input_transform.get(op_tensor.name(), None) != first_op_input_transform:
                        same_input_transpose_order = False
                        break

                if same_input_transpose_order and first_op_input_transform is not None:
                    # Pull out the transpose from input and add it to the output of the producer op.

                    # Step 1: Update the output transform order of the producer op
                    producer_op_axis_info = _get_op_axis_info(op)
                    current_op_transform = producer_op_axis_info.output_transform.get(op_tensor.name(), None)
                    transpose_to_add = first_op_axis_info.input_transform[op_tensor.name()]
                    effective_transform_order = combine_transpose_order(current_op_transform, transpose_to_add)
                    self.op_handler_state.op_axis_info[op_name].output_transform[op_tensor.name()] = effective_transform_order

                    # Step 2: Update TensorAxisInfo for the op_tensor
                    curr_transform_order = self.op_handler_state.tensor_axis_info[op_tensor.name()].transform_order
                    effective_transform_order = combine_transpose_order(curr_transform_order, transpose_to_add)
                    self.op_handler_state.tensor_axis_info[op_tensor.name()].transform_order = effective_transform_order

                    # Step 3: Update the input transform of the consumers
                    for consumer in consumers:
                        con_type = consumer.type
                        consumer_name = op_handler.ir_to_handler_dict.get(con_type, op_handler.CustomOpHandler)(consumer, self.op_handler_state).op_name
                        self.op_handler_state.op_axis_info[consumer_name].input_transform.pop(op_tensor.name())

    def _update_state_dict(self, state_dict: Dict[str, np.array]):
        '''
        If KEEP_ORIGINAL_MODEL_STRUCTURE is enabled
        parameter name with bracket pattern should be replaced with dot digit pattern
        For example, `encoder.layer[1].output.dense.weight` should be `encoder.layer.1.output.dense.weight`

        :param state_dict: state dictionary
        '''
        bracket_pattern_name_to_dot_pattern_name = {}
        for param_name in state_dict.keys():
            replaced_param_name = re.sub(r"\[(\d+)]", r".\1", param_name)
            if param_name != replaced_param_name:
                bracket_pattern_name_to_dot_pattern_name[param_name] = replaced_param_name

        for param_name, replaced_name in bracket_pattern_name_to_dot_pattern_name.items():
            param_tensor = state_dict[param_name]
            state_dict.pop(param_name)
            state_dict[replaced_name] = param_tensor

    def save_encodings(self, op_handler_state):
        '''
        Saves the encoding present in the graph to json file.
        '''
        encoding_file = os.path.join(self.path, self.filename + '_torch.encoding')
        if op_handler_state.encodings['param_encodings']:
            op_handler_state.encodings['param_encodings'] = {
                k: v if isinstance(v, list) else [v]
                for k, v in op_handler_state.encodings['param_encodings'].items()
            }
        with open(encoding_file, "w") as f:
            json.dump(op_handler_state.encodings, f, indent=4)

    def _write_safe_tensor_file(self, state_dict, metadata):
        '''
        Saves the state_dict and prepared model info in a safetensors file.
        :param state_dict: Python dictionary object that maps each layer to its parameter tensor.
        :param metadata: Dict that holds information of the prepared model
        :return: Path of the safetensors file.
        '''

        state_dict_path = os.path.join(self.path, self.filename + '.safetensors')
        save_file(state_dict, state_dict_path, metadata)
        return state_dict_path


    def generate_torch_artifacts(self, ir_graph: Union[IROpGraph, IrGraph]):
        '''
        This method prepare model by taking in the IR Graph and rebuilding a new pytorch model.
        It writes out the prepared model definition and weights to output
        files in the given path location.

        Two files will be output, all with the given filename:
        - a .py file containing a model definition of the prepared model
        - a .safetensors file containing the weights to load for the prepared model and
          prepared model mapping as metadata.

        :param ir_graph: IR graph to build pytorch model from
        :return: Converted pytorch model file path .safetensor file path.
        '''
        ir_graph_utils.validate_ir_graph(ir_graph, custom_op_type_to_module=self.custom_op_info.op_type_to_module)
        model_file = os.path.join(self.path, self.filename + '.py')
        with open(model_file, 'w') as f:
            if self.ir_graph_output_names is None:
                self.ir_graph_output_names = [output_tensor.name() for output_tensor in ir_graph.get_output_tensors_of_graph()]
            is_block_extraction_flow = self.keep_original_model_structure and bool(self.block_names_mapping)
            self.op_handler_state = OpHandlerState(ir_graph, f, self.keep_linear_without_bias, is_block_extraction_flow,
                                                   self.keep_original_model_structure, self.ignore_encodings, self.custom_op_info)
            self.op_handler_state.ir_graph_output_names = self.ir_graph_output_names
            self._generate_axis_information()
            self._optimize_axis_info()
            utils.write_model_initialization(self.model_name, self.op_handler_state)
            order_input = True if self.dummy_model_input is not None else False
            order_output = True if self.dummy_model_output is not None else False
            utils.write_model_forward_pass(self.op_handler_state, self.dummy_model_input, self.dummy_model_output,
                                           ir_graph_input_names=self.ir_graph_input_names,
                                           order_inputs=order_input,
                                           order_outputs=order_output)
            if is_block_extraction_flow:
                # pylint: disable=no-member
                model_definition = self.op_handler_state.model_def_mgr.get_structured_model_definition(self.model_name,
                                                                                                       self.block_names_mapping)
            else:
                model_definition = self.op_handler_state.model_def_mgr.get_model_definition()
            print(model_definition, file=f)

        if self.keep_original_model_structure:
            self._update_state_dict(self.op_handler_state.state_dict)

        if self.op_handler_state.encodings['param_encodings'] or self.op_handler_state.encodings['activation_encodings']:
            self.save_encodings(self.op_handler_state)

        ir_op_name_list = self.op_handler_state.ir_graph_constant_names + self.op_handler_state.ir_graph_ops_by_name
        prepared_model_info = utils.PreparedModelInfo(self.op_handler_state.prepared_param_name_map,
                                                      self.op_handler_state.additional_pad_info,
                                                      self.op_handler_state.additional_transpose_info,
                                                      ir_op_name_list)

        if len(self.op_handler_state.state_dict.items()) == 0:
            self.op_handler_state.state_dict['__no_data__'] = np.ndarray(shape=[1], dtype=np.bool_)
            self.op_handler_state.state_dict['__no_data__'][0] = True

        state_dict_file = self._write_safe_tensor_file(self.op_handler_state.state_dict, prepared_model_info.get_metadata_dump())

        return model_file, state_dict_file