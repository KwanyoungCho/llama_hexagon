# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" Op handlers for EMitter model using QNN IR converter """

import re
import sys
import logging
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from collections import defaultdict
from math import ceil
import numpy as np

from qti.aisw.emitter.op_handler_factory import OpHandlerState
from qti.aisw.emitter.utils.axis_tracking_utils import TensorAxisInfo, get_required_transpose, OpAxisInfo, WEIGHT_INDEX, \
    get_required_weight_transpose, get_output_tranapose_order, INPUT_FORMAT_ALIGNED_OPS, combine_transpose_order, \
    get_transpose_order, format_transpose
from qti.aisw.emitter.utils.axis_transformation import TRANSPOSE_ORDER, update_shape_using_transform_order

from qti.aisw.converters.common import ir_graph as ir_graph_lib

IrGraph, IrOp, IrTensor, IrStaticTensor, AxisFormat = (
    ir_graph_lib.IrGraph, ir_graph_lib.IrOp, ir_graph_lib.IrTensor,
    ir_graph_lib.IrStaticTensor, ir_graph_lib.AxisFormat)

logger = logging.getLogger('TorchEmitter')

# Flag for keeping original model structure in prepared model
KEEP_ORIGINAL_MODEL_STRUCTURE = False


class LSTMDirection(Enum):
    """
    Enum for LSTM direction
    """
    FORWARD = 0
    BACKWARD = 1


class PaddingType(Enum):
    """
    Enum for Padding Type for Conv/Conv Transpose Ops
    """
    SYMMETRIC = 0
    ASYMMETRIC = 1
    ASYMMETRIC_CONV_TRANSPOSE = 2


def _replace_invalid_chars_for_variable(name: str) -> str:
    """
    Replace invalid chars for Python variable such as dot, slash, ...

    :param name: name to replace invalid chars
    :return: cleansed string can be used Python variable name
    """
    # If KEEP_ORIGINAL_MODEL_STRUCTURE is enabled
    # Regex should capture string with bracket pattern first such as h[0], linear[1]
    # Otherwise, it will capture ordinal characters by \w+
    pattern = r"\w+\[\d+]|\w+" if KEEP_ORIGINAL_MODEL_STRUCTURE else r"\w+"
    found_list = re.findall(pattern, name)
    if found_list:
        op_name = "_".join(found_list)
    else:
        error_str = f"Unable to produce a valid op name from {name}"
        logger.error(error_str)
        raise RuntimeError(error_str)
    return op_name


def get_op_name(op_or_tensor_name: str, string_to_prepend_for_digit: str = "op",
                remove_dots: bool = False) -> str:
    """
    IR ops converted from ONNX can have the following names: ada#1.end, relu#0-1.end
    For the correct mapping, we need to extract the root op name first
    Also, dot character is replaced with an underscore because an error occurs
        if it is part of Python variable name

    :param op_or_tensor_name: Name of op or tensor
    :param string_to_prepend_for_digit: String to prepend in case that op name is a number
    :param remove_dots: Flag variable whether to remove dots in op or tensor name
    :return: Op name after preprocessing
    """
    if not KEEP_ORIGINAL_MODEL_STRUCTURE:
        remove_dots = True

    if "#" in op_or_tensor_name:
        op_or_tensor_name = op_or_tensor_name.replace(".end", "")

    splitted_op_or_tensor_name = [name for name in op_or_tensor_name.split("/") if name]

    for idx, name in enumerate(splitted_op_or_tensor_name):
        # Replace any non-alphanumeric or "_" characters with "_"
        name = _replace_invalid_chars_for_variable(name)

        # 243_fc, 50_post_reshape will be converted to op_243_fc, op_50_post_reshape respectively
        if name[0].isdigit():
            splitted_op_or_tensor_name[idx] = string_to_prepend_for_digit + "_" + name
        else:
            splitted_op_or_tensor_name[idx] = name

    if remove_dots:
        op_or_tensor_name = "_".join(splitted_op_or_tensor_name)
    else:
        op_or_tensor_name = ".".join(splitted_op_or_tensor_name)

        # NOTE: This is from QNN post-processing logic to add prefix, and it causes pollution of module name hierarchy
        # Thus, we move this prefix to the place of postfix to recover module name hierarchy
        prefix_to_move = "Reshape_post_."
        if op_or_tensor_name.startswith(prefix_to_move):
            op_or_tensor_name = f"{op_or_tensor_name[len(prefix_to_move):]}_{prefix_to_move[:-2]}"

    return op_or_tensor_name


def _create_init_args_from_kwargs_dict(kwargs_dict: Dict[str, Any], string_args: Optional[Set] = None) -> str:
    """
    Given a kwargs dict, generate a string containing the kwargs and values joined for use in initialization.

    :param kwargs_dict: Dictionary containing kwargs and values for initialization
    :param string_args: Set of arguments whose values should be kept as strings
    :return: String containing the kwargs and values joined for initialization
    """

    list_of_args_and_vals = []
    if string_args is None:
        string_args = set()
    for (arg, val) in kwargs_dict.items():
        if arg in string_args and isinstance(val, str):
            list_of_args_and_vals.append(f'{arg}={repr(val)}')
        else:
            list_of_args_and_vals.append(f'{arg}={val}')
    joined_list = ', '.join(list_of_args_and_vals)
    return joined_list


def _create_variable_name(output_op_name: str) -> str:
    """
    Given an output op name, generate a valid variable name for the op

    :param output_op_name: Name of output op
    :return: New variable name
    """
    return 't_' + _replace_invalid_chars_for_variable(output_op_name)


def _get_permute_order(rank: int) -> List[int]:
    """
    Computes the permute order which can be used to map the channel_last axis of IR to channel_first axis of PyTorch

    :param rank: rank of the input
    :return: Permute Order to use for remapping
    """
    x = range(rank)
    if rank > 2:
        x = [x[0], *x[2:], x[1]]
    return x


def qnn_numpy_type_to_actual_numpy_dtype(tensor: IrTensor) -> str:
    """
    Convert from QNN's getNumpyType() to actual Numpy type. QNN has functionality to get the Numpy's
    "type" from a QNN type. However, it is not in normal syntax. For example, 'bool' could be 'bool8' or
    'int8' could be 'b'. This function take the tensors QNN Numpy type and converts it to what the actual numpy
    type is.

    :param tensor: tensor to get type from
    :return: type in normal Numpy syntax
    """
    if tensor.data_type_string() == 'Float_16':
        return 'float16'

    dummy_np_arr_for_type = np.empty(shape=0, dtype=tensor.data_type_as_numpy_type())
    numpy_type = dummy_np_arr_for_type.dtype.__str__()

    return numpy_type


class ChildModuleCounter:
    """
    Class for counting the number of child modules
    """

    def __init__(self, op_name_list: List[str]):
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


def is_one_to_one_op(op_name: str, child_module_counter: ChildModuleCounter):
    """
    Check whether the original module is mapped to only the op

    :param op_name: Op name of the op to check one-to-one mapping
    :param op_name_list: List of all of op name in the ir graph
    :return: True if there is one to one mapping else False
    """
    module_path, _, _ = op_name.rpartition('/')
    if module_path and child_module_counter.get_children_count(module_path) == 1:
        return True

    return False


def extract_tensor_encoding(tensor: IrTensor) -> Union[Dict, None]:
    """
    Extract the encoding information present in the IrGraph for the given tensor.

    :param tensor: IrTensor of which encoding is to be extracted.
    :return: Encoding Dict is valid encoding is present
    """
    enc_info = tensor.get_encoding()

    def validate_symmetric_for_pcq(is_symmetric):
        assert is_symmetric, 'Per Channel Quantization requires is_symmetric to be set to True'
        return str(is_symmetric)

    if enc_info.type == ir_graph_lib.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED:
        return None

    if enc_info.type in [ir_graph_lib.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
                         ir_graph_lib.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET]:
        # Check for PCQ case
        enc = [
            {
                'bitwidth': channel_enc.bw,
                'scale': channel_enc.scale,
                'offset': channel_enc.offset - 2 ** (channel_enc.bw - 1),
                'min': channel_enc.min,
                'max': channel_enc.max,
                'is_symmetric': validate_symmetric_for_pcq(channel_enc.is_symmetric),
                'dtype': 'int'
            }
            for channel_enc in enc_info.axisEncInfo.encInfos
        ]
    else:
        enc = {'bitwidth': enc_info.encInfo.bw}
        if enc_info.encInfo.is_fixed_point:
            enc['dtype'] = 'int'
            enc['is_symmetric'] = str(enc_info.encInfo.is_symmetric)
            if enc_info.encInfo.scale != 0 and not (enc_info.encInfo.min == 0 and enc_info.encInfo.max == 0):
                enc.update({
                    'scale': enc_info.encInfo.scale,
                    'offset': enc_info.encInfo.offset,
                    'min': enc_info.encInfo.min,
                    'max': enc_info.encInfo.max,
                })
            elif enc_info.encInfo.bw != 0:
                # Partial encoding case, only either of scale/offset or min/max are provided.
                if enc_info.encInfo.min != 0 or enc_info.encInfo.max != 0:
                    enc['min'] = enc_info.encInfo.min
                    enc['max'] = enc_info.encInfo.max
                elif enc_info.encInfo.scale != 0:
                    enc['scale'] = enc_info.encInfo.scale
                    enc['offset'] = enc_info.encInfo.offset
                else:
                    raise ValueError("Invalid encoding info found.")
            else:
                raise ValueError("Invalid encoding info found.")
        else:
            enc['dtype'] = 'float'

    return enc


class OpHandler:
    """
    Base OpHandler class for performing Pytorch model definition actions depending on op type.
    """

    def __init__(self, op, op_handler_state, num_inputs: Optional[int] = 1):
        self._op = op
        self._op_handler_state = op_handler_state

        global KEEP_ORIGINAL_MODEL_STRUCTURE
        KEEP_ORIGINAL_MODEL_STRUCTURE = self._op_handler_state.keep_original_model_structure
        # Num inputs of None means that all inputs of the op are considered as inputs (instead of parameters)
        self._num_inputs = num_inputs

        # Extract op name
        op_name = op.name() if callable(op.name) else op.name
        module_path, _, _ = op_name.rpartition('/')
        if module_path in op_handler_state.created_submodules and is_one_to_one_op(op_name,
                                                                                   op_handler_state.child_module_counter):
            self._op_name = module_path
        else:
            self._op_name = op_name

        if KEEP_ORIGINAL_MODEL_STRUCTURE and module_path:
            # self._op_name is mainly used when generating initialization and execution statement
            # If op_name including pattern dot with digits such as transformer.h.0.ln_1 which implies ModuleList
            # op_name should be converted with using brackets, meaning that transformer.h[0].ln_1
            self._op_name = re.sub(r"\.(\d+)", r"[\1]", self._op_name)

        # To handle input/output transpose
        self.transposed_input_names = {}

    @property
    def op_name(self):
        """ Returns the op_name to be used in the prepared name """
        return self._op_name

    def get_attr_value(self, attr: str):
        """ Get the op's value of the given attribute """
        return self._op_handler_state.get_attr_value(self._op, attr)

    def get_input_op_names(self) -> List[str]:
        """
        Get a list of input op names for the op.

        :return: List of input op names
        """

        # num_inputs can be given as None. If so, treat all input names as inputs (instead of parameters)
        input_names = self._op.get_input_names
        if self._num_inputs is None:
            return input_names
        return input_names[:self._num_inputs]

    def get_weights(self):
        """ Method to fetch the static tensor data for an op """
        weights = []
        for input_tensor in self._op.inputs():
            if input_tensor.is_static_tensor():
                weights.append(input_tensor.get_data())
        return weights

    def get_ir_graph_output_op_names(self) -> List[str]:
        """
        Get a list of output op names for the op.

        :return: List of output op names
        """
        return [out_tensor.get_producer().name for out_tensor in
                self._op_handler_state.ir_graph.get_output_tensors_of_graph()]

    def get_ops_output_names(self) -> List[str]:
        """
        Helper function to get an ops output names

        :return: List of output op names for the op
        """
        return [out_tensor.name() for out_tensor in self._op.outputs()]

    def get_parameter_op_names(self) -> List[str]:
        """
        Get a list of parameter op names for the op.

        :return: List of parameter op names
        """

        # num_inputs can be given as None. If so, treat all input names as inputs (instead of parameters)
        if self._num_inputs is None:
            return []
        return self._op.get_input_names[self._num_inputs:]

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        # pylint: disable=no-self-use
        return

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in self.get_input_op_names()]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        string_inputs = ', '.join(string_inputs)

        output_tensors = []

        for output_op_name in self.get_ops_output_names():
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)

    def save_activation_encoding(self):
        """
        Saves the encoding into the op_handler_state for the activations of the op.
        """
        activation_encodings = {}

        def add_to_dict(key, counter, encoding):
            if key in activation_encodings:
                activation_encodings[key][counter] = encoding
            else:
                activation_encodings[key] = {counter: encoding}

        # Get the input activation encodings
        counter = 0
        for tensor in self._op.inputs():
            if not tensor.is_static_tensor():
                encoding = extract_tensor_encoding(tensor)
                if encoding is not None:
                    add_to_dict('input', str(counter), encoding)
            counter += 1

        # Get the output activation encodings
        counter = 0
        for tensor in self._op.outputs():
            encoding = extract_tensor_encoding(tensor)
            if encoding is not None:
                add_to_dict('output', str(counter), encoding)
            counter += 1

        if activation_encodings:
            self._op_handler_state.encodings['activation_encodings'][get_op_name(self._op_name)] = activation_encodings

    def generate_transpose_for_model_input(self):
        """
        Instantiate nn.Module for permute needed for model's input tensor
        """
        # In case of transpose op no need to handle it by any extra transpose.
        if self._op.type == 'Transpose':
            return

        for input_idx, tensor in enumerate(self._op.inputs()):
            # Additional handling for the app_write_tensor in case of transpose
            if tensor.is_app_write_tensor():
                tensor_name = tensor.name()
                op_axis_info = self._op_handler_state.op_axis_info[self._op_name]
                if op_axis_info.input_transform.get(tensor_name, None) is not None:
                    # Use same logic to generate the model input tensor variable name as during forward pass
                    additional_permute_op_name = 'temp_transpose_' + get_op_name(tensor_name, 'input', remove_dots=True) \
                                                 + '_' + get_op_name(self._op_name)
                    self._op_handler_state.additional_transpose_info[additional_permute_op_name] = (get_op_name(self._op_name), input_idx)
                    # Create instance of elementwise_ops.Permute()
                    op_str = f'\t\tself.{additional_permute_op_name} = elementwise_ops.Permute()'
                    self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op)

    def handle_input_transpose(self):
        """
        Adds the input transpose if required. And then aligns the input variables.
        """

        op_axis_info = self._op_handler_state.op_axis_info[self._op_name]
        for tensor in self._op.inputs()[:self._num_inputs]:
            input_op_name = tensor.name()
            if input_op_name in op_axis_info.input_transform and op_axis_info.input_transform[input_op_name] is not None:
                var_name = self._op_handler_state.op_to_tensors[input_op_name]

                # add transpose for this variable
                if var_name[:4]=='self':
                    temp_transpose_name = "temp_transpose_" + var_name[5:]
                else:
                    temp_transpose_name = "temp_transpose_" + var_name
                if tensor.is_app_write_tensor():
                    transpose_str = f"\t\t{temp_transpose_name} = self.{temp_transpose_name}_{get_op_name(self._op_name)}({var_name}, {op_axis_info.input_transform[input_op_name]})"
                else:
                    transpose_str = f"\t\t{temp_transpose_name} = {var_name}.permute({op_axis_info.input_transform[input_op_name]})"
                self._op_handler_state.model_def_mgr.add_execute_code(transpose_str, self._op, var_name if tensor.is_app_write_tensor() else None, temp_transpose_name)

                # add it to changed dictionary
                self.transposed_input_names[self._op_handler_state.op_to_tensors[input_op_name]] = temp_transpose_name

    def handle_output_transpose(self):
        """
        Adds the output transpose if required. Clears out the temporary variables.
        """
        for input_op_name in self.transposed_input_names:
            temp_transpose_name = self.transposed_input_names[input_op_name]
            transpose_str = f"\t\t{temp_transpose_name} = None"
            self._op_handler_state.model_def_mgr.add_execute_code(transpose_str, self._op)

        op_axis_info = self._op_handler_state.op_axis_info[self._op_name]
        for output_op_name in self.get_ops_output_names():
            if output_op_name in op_axis_info.output_transform and op_axis_info.output_transform[output_op_name] is not None:
                output_var_name = self._op_handler_state.op_to_tensors[output_op_name]
                transpose_str = f"\t\t{output_var_name} = {output_var_name}.permute({op_axis_info.output_transform[output_op_name]})"
                self._op_handler_state.model_def_mgr.add_execute_code(transpose_str, self._op, string_outputs=output_var_name)

    def _generate_axis_information(self):
        """
        Generates the axis information for the current op based on
        the IR axis format and the desired axis format for torch.
        """
        # Get the op_type
        # Actions will be based on the op type
        op_type = self._op.type

        if op_type in INPUT_FORMAT_ALIGNED_OPS:
            self._generate_axis_info_for_input_aligned_ops()
            return

        input_transpose = {}
        output_transpose = {}

        # Process weight transpose
        weight_index = WEIGHT_INDEX.get(op_type, None)
        if weight_index is not None:
            weight_tensor = self._op.inputs()[weight_index]
            tensor_axis_info = self._op_handler_state.tensor_axis_info[weight_tensor.name()]
            transpose_require = get_required_weight_transpose(tensor_axis_info)
            if transpose_require is not None:
                input_transpose[weight_tensor.name()] = transpose_require

        # Iterate through the input activation tensor and check if some processing is required
        for idx, ip_tensor in enumerate(self.get_input_op_names()):
            # Skip weight permute in case it is already transformed in the previous step
            if idx == weight_index:
                continue
            tensor_axis_info = self._op_handler_state.tensor_axis_info[ip_tensor]
            transpose_require = get_required_transpose(tensor_axis_info, op_type)
            if transpose_require is not None:
                input_transpose[ip_tensor] = transpose_require

        # Instantiate the TensorAxisInfo for output
        input_tensor_name = self._op.inputs()[0].name()
        input_tensor_axis_info = self._op_handler_state.tensor_axis_info[input_tensor_name]

        # Iterate through the output tensor
        if op_type in ['Reduce', 'Arg', 'Moments']:
            self._generate_output_order_for_ops_with_keepdims()
            op_tensor = self._op.outputs()[0]
            output_transpose_order = self._op_handler_state.tensor_axis_info[op_tensor.name()].transform_order
            transpose_required = self.get_output_tranpose_order_for_model_outputs(op_tensor, output_transpose_order)
            if transpose_required is not None:
                output_transpose[op_tensor.name()] = transpose_required
        else:
            for op_tensor in self._op.outputs():
                output_transpose_order = get_output_tranapose_order(op_type, op_tensor, input_tensor_axis_info)
                self._op_handler_state.tensor_axis_info[op_tensor.name()] = TensorAxisInfo(op_tensor.dims(),
                                                                                           output_transpose_order)

                transpose_required = self.get_output_tranpose_order_for_model_outputs(op_tensor, output_transpose_order)
                if transpose_required is not None:
                    output_transpose[op_tensor.name()] = transpose_required

        self._op_handler_state.op_axis_info[self._op_name] = OpAxisInfo(input_transpose, output_transpose)

    def get_output_tranpose_order_for_model_outputs(self, op_tensor, output_transpose_order) -> Union[Tuple, None]:
        """
        Special check if the tensor is a model output to align it to the src_model or ir_graph output shape.

        :param op_tensor: output tensor for which transpose requirement is to be checked.
        :param output_transpose_order: current transpose state of the tensor.
        :return: Required transpose order.
        """

        if op_tensor.is_app_read_tensor():
            # First align with the IrGraph
            output_state = output_transpose_order
            transpose_required = get_transpose_order(output_state, None)

            transpose_order = None
            # Align with the src_graph if src_model is available
            # if self._op_handler_state.model is not None:
            #     ir_axis_format = op_tensor.axis_format()
            #     src_axis_format = op_tensor.src_axis_format()
            #     try:
            #         transpose_order = TRANSPOSE_ORDER[ir_axis_format][src_axis_format] if ir_axis_format != src_axis_format else None
            #     except KeyError:
            #         log_str = f'No Transformation exists  form {ir_axis_format} to {src_axis_format}. Going with {ir_axis_format} as the output format'
            #         logger.error(log_str)
            #         transpose_order = None

            transpose_required = combine_transpose_order(transpose_required, transpose_order)
            # Note as the output transpose as been added to the tensor it's transform order needs to be updated
            # so as to indicate the correct transform order to the other consumer of this tensor.
            tensor_axis_info = self._op_handler_state.tensor_axis_info[op_tensor.name()]
            tensor_axis_info.transform_order = format_transpose(combine_transpose_order(tensor_axis_info.transform_order, transpose_required))
            self._op_handler_state.tensor_axis_info[op_tensor.name()] = tensor_axis_info

            return format_transpose(transpose_required)
        return None

    def _generate_output_order_for_ops_with_keepdims(self):
        input_tensor = self._op.inputs()[0]
        op_tensor = self._op.outputs()[0]
        input_tensor_axis_info = self._op_handler_state.tensor_axis_info[input_tensor.name()]
        if input_tensor_axis_info.transform_order is not None and len(input_tensor.dims()) != len(op_tensor.dims()):
            # Handle the variable length
            axes = self._op.attrs_dict.get("axes", None)
            if axes is None:
                axes = [self._op.attrs_dict.get("axis", None)]
            axes = list(axes)

            output_order = list(input_tensor_axis_info.transform_order)

            for axis in axes:
                output_order.remove(axis)

            output_order = tuple(np.argsort(output_order))
        else:
            output_order = get_output_tranapose_order(self._op.type, op_tensor, input_tensor_axis_info)
        self._op_handler_state.tensor_axis_info[op_tensor.name()] = TensorAxisInfo(op_tensor.dims(), output_order)

    def _generate_axis_info_for_input_aligned_ops(self):
        # Initialize with the transform_order of the first tensor
        input_tensor_names = [tensor.name() for tensor in self._op.inputs()]
        initial_transform_order = self._op_handler_state.tensor_axis_info[input_tensor_names[0]].transform_order
        input_aligned = True

        for index in range(1, len(input_tensor_names)):
            tensor_axis_info = self._op_handler_state.tensor_axis_info[input_tensor_names[index]]
            if initial_transform_order != tensor_axis_info.transform_order:
                input_aligned = False
                break

        input_transpose = {}
        output_transpose = {}

        pseudo_op_type = 'INDEX_TRANSPOSE_OP'
        # If input aligned behave as INDEXED BASED OPS # else works as SRC BASED OPS
        if not input_aligned:
            pseudo_op_type = 'SRC_AXIS_OP'
            for ip_tensor in self.get_input_op_names():
                tensor_axis_info = self._op_handler_state.tensor_axis_info[ip_tensor]
                transpose_require = get_required_transpose(tensor_axis_info, pseudo_op_type)
                if transpose_require is not None:
                    input_transpose[ip_tensor] = transpose_require

        # Instantiate the TensorAxisInfo for output
        input_tensor_name = self._op.inputs()[0].name()
        input_tensor_axis_info = self._op_handler_state.tensor_axis_info[input_tensor_name]

        # Iterate through the output tensor
        for op_tensor in self._op.outputs():
            output_transpose_order = get_output_tranapose_order(pseudo_op_type, op_tensor, input_tensor_axis_info)
            self._op_handler_state.tensor_axis_info[op_tensor.name()] = \
                TensorAxisInfo(op_tensor.dims(), output_transpose_order)

            output_transpose_order = self.get_output_tranpose_order_for_model_outputs(op_tensor, output_transpose_order)
            if output_transpose_order is not None:
                output_transpose[op_tensor.name()] = output_transpose_order

        self._op_handler_state.op_axis_info[self._op_name] = OpAxisInfo(input_transpose, output_transpose)

    def _update_axis_using_axis_information(self, axis):
        """
        Updates the axis value if needed from IR Graph Input Layout to Emitter Graph Input Layout

        :param axis: Axis value in the IR Graph
        :return: Updated axis value
        """
        axis_info = self._op_handler_state.tensor_axis_info[self._op.inputs()[0].name()]
        input_transpose = self._op_handler_state.op_axis_info[self._op_name].input_transform. \
            get(self._op.inputs()[0].name(), None)
        effective_transform_order = combine_transpose_order(axis_info.transform_order, input_transpose)
        if axis_info.transform_order is not None:
            for index, pos in enumerate(effective_transform_order):
                if pos == axis:
                    axis = index
                    break
        return axis

    def get_io_map(self):
        """
        Generates the input and output tensor name mapping for the layer.
        """
        op_name = get_op_name(self._op_name)
        io_dict = {'input': {}, 'output': {}}

        for index, name in enumerate(self.get_input_op_names()):
            io_dict['input'][str(index)] = name

        for index, name in enumerate(self.get_ops_output_names()):
            io_dict['output'][str(index)] = name

        io_dict['parameters'] = self.update_param_in_io_dict()
        self._op_handler_state.node_to_io_map['activation_encodings'][op_name] = io_dict

    def update_param_in_io_dict(self):
        """
        Update the param name mapping in the io_dict.
        """
        op_name = get_op_name(self._op_name)
        unmapped_params = []
        for name in self.get_parameter_op_names():
            splits = name.split('.')
            if len(splits) > 1:
                self._op_handler_state.node_to_io_map['param_encodings'][op_name + '.' + splits[-1]] = name
            else:
                # This  will not get mapped to any encodings. Only for analysis
                unmapped_params.append(name)
        return unmapped_params


def generate_input_and_constant_axis_format(op: IrOp, op_handler_state: OpHandlerState):
    """
    Populates the axis information for the inputs and constant tensors.

    :param op: op for which inputs and constant tensor information needs to be extracted.
    :param op_handler_state: OpHandlerState object.
    """
    for tensor in op.inputs():
        if tensor.name() not in op_handler_state.tensor_axis_info:
            if tensor.is_static_tensor() or tensor.is_app_write_tensor():
                transpose_order = None
                # Input to the model
                # Special Handling is required for the input in case this is invoked from src pytorch graph
                # as the input order can be different in QNN and in src Graph.
                # if op_handler_state.model is not None and tensor.is_app_write_tensor():
                #     ir_axis_format = tensor.axis_format()
                #     src_axis_format = tensor.src_axis_format()
                #     try:
                #         transpose_order = TRANSPOSE_ORDER[ir_axis_format][src_axis_format] if ir_axis_format != src_axis_format else None
                #     except KeyError:
                #         log_str = f'No Transformation exists  form {ir_axis_format} to {src_axis_format}. Going with {ir_axis_format} as the input format'
                #         logger.error(log_str)

                op_handler_state.tensor_axis_info[tensor.name()] = TensorAxisInfo(tensor.dims(), transpose_order)


def _add_input_pad_if_requried(op_handler: OpHandler, pad_value: int | float = 0):
    """
    Method adds forward pass for the additional padding layer introduced to handle  uneven padding.
    Requires for conv and pool layers.

    :param op_handler:  OpHandler object for which padding check is to be performed
    :param pad_value:   Fill value for constant padding, default is 0
    """
    # pylint: disable=protected-access
    op_name = get_op_name(op_handler._op_name)
    if op_name in op_handler._op_handler_state.additional_pad_info:
        pad_op_name = op_handler._op_handler_state.additional_pad_info[op_name]
        pad_amount = op_handler._op.attrs_dict["pad_amount"]
        padding = tuple(pad_amount[::-1].flatten())
        input_name = op_handler._op_handler_state.op_to_tensors[op_handler.get_input_op_names()[0]]
        input_name = op_handler.transposed_input_names.get(input_name, input_name)
        temp_input_tensor = input_name + "_temp_pad"
        padding_str = f"\t\t{temp_input_tensor} = self.{pad_op_name}({input_name}, {padding}, 'constant', {pad_value})"
        op_handler._op_handler_state.model_def_mgr.add_execute_code(padding_str, op_handler._op, input_name, temp_input_tensor)
        return temp_input_tensor
    return None


def _process_padding(op_handler: OpHandler) -> Tuple[int]:
    """
    Fetches the padding value from the ops attrs_dict and check if padding is uneven.
    In case of uneven padding, creates a pad op to handle the padding and set the padding
    value to zero for the current op.

    :param op_handler: OpHandler object for which padding check is to be performed
    :return: padding value to be used for the operation
    """
    # pylint: disable=protected-access
    op = op_handler._op
    op_handler_state = op_handler._op_handler_state
    padding_start, padding_end = [tuple(op.attrs_dict["pad_amount"][..., axis].flatten()) for axis in (0, 1)]

    if padding_start != padding_end:
        # As padding is un-even it will be handled by an additional pad op.

        if "TransposeConv" in op.type:
            padding_start = np.asarray(padding_start, dtype=np.int32)
            padding_end = np.asarray(padding_end, dtype=np.int32)
            padding_diff = list(ps - pe for ps, pe in zip(padding_start, padding_end))

            op_handler._padding_info['padding_type'] = PaddingType.ASYMMETRIC_CONV_TRANSPOSE
            op_handler._padding_info['padding_diff'] = padding_diff
            return tuple(padding_start)

        op_name = get_op_name(op_handler._op_name)
        pad_op_name = op_name + "_input_pad"
        op_handler_state.additional_pad_info[op_name] = pad_op_name
        op_str = f'\t\tself.{pad_op_name} = elementwise_ops.Pad()'
        op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, op)
        # As we are adding a separate pad operation, no need to the pad further.
        op_handler._padding_info['_padding_type'] = PaddingType.ASYMMETRIC
        return tuple([0] * len(padding_start))

    op_handler._padding_info['_padding_type'] = PaddingType.SYMMETRIC
    return padding_start


class ElementwiseUnaryOpHandler(OpHandler):
    """ Elementwise unary OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        op_name = get_op_name(self._op_name)
        attr_key = 'operation' if 'operation' in self._op.attrs_dict.keys() else 'eltwise_type'
        if attr_key == 'operation':
            op_type_to_torch_module_dict = {
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN: 'elementwise_ops.Sin()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS: 'elementwise_ops.Cos()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN: 'elementwise_ops.Asin()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN: 'elementwise_ops.Atan()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP: 'elementwise_ops.Exponential()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG: 'elementwise_ops.Log()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT: 'elementwise_ops.Sqrt()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS: 'elementwise_ops.Abs()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG: 'elementwise_ops.Neg()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND: 'elementwise_ops.Round()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL: 'elementwise_ops.ElementwiseCeil()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR: 'elementwise_ops.ElementwiseFloor()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN: 'elementwise_ops.ElementwiseUnarySign()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT: 'elementwise_ops.LogicalNot()',
            }
        else:
            op_type_to_torch_module_dict = {
                'ElementWiseSin': 'elementwise_ops.Sin()',
                'ElementWiseCos': 'elementwise_ops.Cos()',
                'ElementWiseAsin': 'elementwise_ops.Asin()',
                'ElementWiseAtan': 'elementwise_ops.Atan()',
                'ElementWiseExp': 'elementwise_ops.Exponential()',
                'ElementWiseLog': 'elementwise_ops.Log()',
                'ElementWiseSquareRoot': 'elementwise_ops.Sqrt()',
                'ElementWiseAbs': 'elementwise_ops.Abs()',
                'ElementWiseNeg': 'elementwise_ops.Neg()',
                'Erf': 'elementwise_ops.Erf()',
                'ElementWiseRound': 'elementwise_ops.Round()',
                'ElementWiseCeil': 'elementwise_ops.ElementwiseCeil()',
                'ElementWiseFloor': 'elementwise_ops.ElementwiseFloor()',
                'ElementWiseSign': 'elementwise_ops.ElementwiseUnarySign()',
                'ElementWiseNot': 'elementwise_ops.LogicalNot()',
            }

        eltwise_type = self._op.attrs_dict[attr_key]
        # TODO: Remove handling Softplus under ElementwiseUnaryOp once converter resolves it under ElementWiseNeuron Op
        if eltwise_type == 'ElementWiseSoftplus':
            beta = self.get_attr_value('beta')
            threshold = self.get_attr_value('threshold')
            torch_module = f'torch.nn.Softplus(beta={beta}, threshold={threshold})'
        elif eltwise_type in op_type_to_torch_module_dict:
            torch_module = op_type_to_torch_module_dict[eltwise_type]
        else:
            raise AssertionError(
                f"Unrecognized elementwise unary op {self._op.attrs_dict[attr_key]}")

        op_str = f'\t\tself.{op_name} = {torch_module}'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)


class ErfOpHandler(OpHandler):
    """ Erf Op Handler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        op_name = get_op_name(self._op_name)
        op_str = f'\t\tself.{op_name} = elementwise_ops.Erf()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

class ElementwiseBinaryOpHandler(OpHandler):
    """ Elementwise math OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=2)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        attr_key = 'operation' if 'operation' in self._op.attrs_dict else 'eltwise_type'
        if attr_key == 'operation':
            op_type_to_torch_module_dict = {
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD: 'elementwise_ops.Add()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT: 'elementwise_ops.Subtract()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY: 'elementwise_ops.Multiply()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE: 'elementwise_ops.Divide()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM: 'elementwise_ops.Minimum()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM: 'elementwise_ops.Maximum()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER: 'elementwise_ops.Pow()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MOD: 'elementwise_ops.Remainder()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FMOD: 'elementwise_ops.Fmod()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL: 'elementwise_ops.Equal()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER: 'elementwise_ops.Greater()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS: 'elementwise_ops.Less()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL: 'elementwise_ops.GreaterEqual()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL: 'elementwise_ops.LessEqual()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR: 'elementwise_ops.LogicalOr()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND: 'elementwise_ops.LogicalAnd()',
                ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FLOOR_DIV: 'elementwise_ops.FloorDivide()',
            }
        else:
            op_type_to_torch_module_dict = {
                'ElementWiseAdd': 'elementwise_ops.Add()',
                'ElementWiseSubtract': 'elementwise_ops.Subtract()',
                'ElementWiseMultiply': 'elementwise_ops.Multiply()',
                'ElementWiseDivide': 'elementwise_ops.Divide()',
                'ElementWiseMinimum': 'elementwise_ops.Minimum()',
                'ElementWiseMaximum': 'elementwise_ops.Maximum()',
                'ElementWisePower': 'elementwise_ops.Pow()',
                'ElementWiseMod': 'elementwise_ops.Remainder()',
                'ElementWiseFmod': 'elementwise_ops.Fmod()',
                'ElementWiseEqual': 'elementwise_ops.Equal()',
                'ElementWiseGreater': 'elementwise_ops.Greater()',
                'ElementWiseLess': 'elementwise_ops.Less()',
                'ElementWiseGreaterEqual': 'elementwise_ops.GreaterEqual()',
                'ElementWiseLessEqual': 'elementwise_ops.LessEqual()',
                'ElementWiseOr': 'elementwise_ops.LogicalOr()',
                'ElementWiseAnd': 'elementwise_ops.LogicalAnd()',
                'ElementWiseFloorDiv': 'elementwise_ops.FloorDivide()',
            }
        torch_module_name = op_type_to_torch_module_dict.get(self._op.attrs_dict[attr_key])
        if torch_module_name:
            op_str = f'\t\tself.{op_name} = {torch_module_name}'
        else:
            raise AssertionError(f"Unrecognized elementwise math op {self._op.attrs_dict[attr_key]}")
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """
        super().generate_execute_code()
        attr_key = 'operation' if 'operation' in self._op.attrs_dict else 'eltwise_type'
        if attr_key == 'operation':
            DIVIDE_OPERATION = ir_graph_lib.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE
        else:
            DIVIDE_OPERATION = 'ElementWiseDivide'
        if self._op.attrs_dict[attr_key] == DIVIDE_OPERATION:
            string_outputs = self._op_handler_state.op_to_tensors[self._op.outputs()[0].name()]
            output_tensor_dtype = self._op.outputs()[0].data_type_as_numpy_type()
            if output_tensor_dtype == 'i':
                dtype_conversion_str = f'\t\t{string_outputs} = {string_outputs}.type(torch.int32)'
                self._op_handler_state.model_def_mgr.add_execute_code(dtype_conversion_str, self._op, string_outputs=string_outputs)


class ElementwiseTernaryOpHandler(OpHandler):
    """ Elementwise Ternary OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=3)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.Where()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class ReduceOpHandler(OpHandler):
    """ Reduce OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_type_to_torch_module_dict = {
            'ReduceMean': 'elementwise_ops.Mean()',
            'ReduceSum': 'elementwise_ops.Sum()',
            'ReduceProd': 'elementwise_ops.Prod()',
            'ReduceMin': 'elementwise_ops.AMin()',
            'ReduceMax': 'elementwise_ops.AMax()'
        }

        reduce_type = self._op.attrs_dict['reduce_type']
        torch_module_name = op_type_to_torch_module_dict.get(reduce_type)
        if torch_module_name:
            op_str = f'\t\tself.{get_op_name(self._op_name)} = {torch_module_name}'
        else:
            raise AssertionError(f'Unrecognized reduce math op {reduce_type}')
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        input_op_names = self.get_input_op_names()
        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in input_op_names]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        string_inputs = ", ".join(string_inputs)

        output_tensors = []
        output_op_names = self.get_ops_output_names()
        for output_op_name in output_op_names:
            output_tensor = _create_variable_name(output_op_name)
            self._op_handler_state.op_to_tensors[output_op_name] = output_tensor
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ", ".join(output_tensors)

        # Get the values for the attributes axis and keepdims to be passed in the fwd pass
        kwargs_dict = dict()
        keepdim = self._op.attrs_dict["keep_dims"]
        dim = tuple(self._op.attrs_dict["axes"])

        # If number of axes equals rank of input tensor i.e., reduction is applied across all axes,
        # dim and keepdim are not required
        if len(dim) != len(self._op.inputs()[0].dims()):

            # pylint: disable=consider-using-generator
            dim = tuple([self._update_axis_using_axis_information(d) for d in dim])
            kwargs_dict["dim"] = dim

            if self._op.attrs_dict["reduce_type"] == "ReduceProd":
                kwargs_dict["dim"] = dim[0]

            kwargs_dict["keepdim"] = keepdim

        init_args = _create_init_args_from_kwargs_dict(kwargs_dict)

        execute_str = f"\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs}, {init_args})"
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)


class ConvHandler(OpHandler):
    """ Conv OpHandler """

    def __init__(self, op, op_handler_state):
        is_dynamic_conv = self._is_dynamic_conv(op)
        if is_dynamic_conv:
            num_inputs = len(op.inputs())  # [Input, Weight, Bias] or [Input, Weight]
        else:
            num_inputs = 1  # Input

        super().__init__(op, op_handler_state, num_inputs=num_inputs)
        self.is_dynamic_conv = is_dynamic_conv
        self._padding_info = {}
        if 'pad_amount' in op.attrs_dict:
            _process_padding(self)

    # pylint: disable=no-self-use
    def _is_dynamic_conv(self, op: IrOp) -> bool:
        """
        Check whether this op is dynamic conv or not

        :param op: Op
        :return: True if this node is dynamic conv else False
        """

        for input_ in op.inputs()[1:]:
            if not input_.is_static_tensor():
                return True
        return False

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        # pylint: disable=too-many-locals
        op_name = get_op_name(self._op_name)
        op_type_to_torch_module_dict = {
            'Conv1d': 'torch.nn.Conv1d',
            'Conv2d': 'torch.nn.Conv2d',
            'Conv3d': 'torch.nn.Conv3d',
            'TransposeConv1d': 'torch.nn.ConvTranspose1d',
            'TransposeConv2d': 'torch.nn.ConvTranspose2d',
            'TransposeConv3d': 'torch.nn.ConvTranspose3d',
        }
        torch_module_name = op_type_to_torch_module_dict.get(self._op.type)
        if not torch_module_name:
            raise AssertionError(f'Unrecognized convolution op {self._op.type}')

        op_attrs = self._op.attrs_dict
        kwargs_dict = dict()
        kwargs_dict["stride"] = tuple(op_attrs["stride"])
        kwargs_dict["padding"] = _process_padding(self)
        kwargs_dict["dilation"] = tuple(op_attrs.get("dilation", [1] * len(kwargs_dict["padding"])))
        kwargs_dict["groups"] = op_attrs["group"]

        if self.is_dynamic_conv:
            dynamic_conv_args = _create_init_args_from_kwargs_dict(kwargs_dict)
            op_str = f'\t\tself.{op_name} = elementwise_ops.DynamicConv2d({dynamic_conv_args})'
            self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)
            return

        weights = self.get_weights()
        weight, bias = weights if len(weights) == 2 else (weights[0], None)
        weight_tensor_name = self._op.inputs()[1].name()
        transform_order = None
        if weight_tensor_name in self._op_handler_state.op_axis_info[self._op_name].input_transform is not None:
            transform_order = list(
                self._op_handler_state.op_axis_info[self._op_name].input_transform[weight_tensor_name])
            if 'Transpose' in torch_module_name:
                transform_order[1], transform_order[0] = transform_order[0], transform_order[1]
            weight = weight.transpose(transform_order)
        in_channels = weight.shape[1] * op_attrs["group"]
        out_channels = weight.shape[0]

        if 'Transpose' in torch_module_name:
            in_channels, out_channels = out_channels, in_channels
            kwargs_dict['output_padding'] = tuple(op_attrs["output_padding"])

        kwargs_dict["in_channels"] = in_channels
        kwargs_dict["out_channels"] = out_channels
        kwargs_dict["kernel_size"] = weight.shape[2:]
        kwargs_dict["bias"] = np.count_nonzero(bias) != 0  # bias comes back as all zeros if no bias

        init_args = _create_init_args_from_kwargs_dict(kwargs_dict)
        op_str = f'\t\tself.{op_name} = {torch_module_name}({init_args})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

        self._op_handler_state.state_dict[op_name + '.weight'] = weight  # store params as np tensors
        self._op_handler_state.prepared_param_name_map[op_name + '.weight'] = (self._op.get_input_names[1], transform_order)
        if not self._op_handler_state.ignore_encodings:
            weight_enc = extract_tensor_encoding(self._op.inputs()[1])
            if weight_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.weight'] = weight_enc
        if kwargs_dict["bias"]:
            self._op_handler_state.state_dict[op_name + '.bias'] = bias  # store params as np tensors
            self._op_handler_state.prepared_param_name_map[op_name + '.bias'] = (self._op.get_input_names[2], None)
            if not self._op_handler_state.ignore_encodings:
                bias_enc = extract_tensor_encoding(self._op.inputs()[2])
                if bias_enc is not None:
                    self._op_handler_state.encodings['param_encodings'][op_name + '.bias'] = bias_enc

    def generate_execute_code(self):
        """
        Generate code for executing the node as a PyTorch module in file self._op_handler_state.f.
        """
        temp_input_tensor = _add_input_pad_if_requried(self)

        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in self.get_input_op_names()]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]

        # If additional padding layer is added,
        # then its output tensor name will be passed as input to current op.
        string_inputs[0] = string_inputs[0] if not temp_input_tensor else temp_input_tensor
        string_inputs = ', '.join(string_inputs)

        output_tensors = []
        for output_op_name in self.get_ops_output_names():
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs})'

        padding_type = self._padding_info.get('padding_type', None)
        if padding_type == PaddingType.ASYMMETRIC_CONV_TRANSPOSE:

            padding_diff = self._padding_info.get('padding_diff', None)
            op_type = self._op.type
            diff_dim_w = None if padding_diff[-1] == 0 else padding_diff[-1]
            # NCF/W
            out_tensor_slice_str = f'\t\t{string_outputs} = {string_outputs}[:, :, :{diff_dim_w}]'

            if op_type == 'TransposeConv2d':
                diff_dim_h = None if padding_diff[-2] == 0 else padding_diff[-2]
                # NCHW
                out_tensor_slice_str = f'\t\t{string_outputs} = {string_outputs}[:, :, :{diff_dim_h}, :{diff_dim_w}]'

            elif op_type == 'TransposeConv3d':
                diff_dim_h = None if padding_diff[-2] == 0 else padding_diff[-2]
                diff_dim_d = None if padding_diff[-3] == 0 else padding_diff[-3]
                # NCDHW
                out_tensor_slice_str = f'\t\t{string_outputs} = {string_outputs}[:, :, :{diff_dim_d}, :{diff_dim_h}, :{diff_dim_w}]'

            execute_str = execute_str + '\n' + out_tensor_slice_str

        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)

        # Free output of additional padding layer after it is used.
        if temp_input_tensor:
            free_str = f'\t\t{temp_input_tensor} = None'
            self._op_handler_state.model_def_mgr.add_execute_code(free_str, self._op)


class DepthWiseConv2dHandler(OpHandler):
    """ DepthWiseConv2d OpHandler """

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state)
        self._padding_info = {}

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        # pylint: disable=too-many-locals
        weights = self.get_weights()
        weight, bias = weights if len(weights) == 2 else (weights[0], None)
        weight_tensor_name = self._op.inputs()[1].name()
        transform_order = None
        if weight_tensor_name in self._op_handler_state.op_axis_info[self._op_name].input_transform is not None:
            transform_order = self._op_handler_state.op_axis_info[self._op_name].input_transform[weight_tensor_name]
            weight = weight.transpose(transform_order)

        in_channels = self._op.inputs()[0].dims()[3]
        out_channels = weight.shape[0]

        kernel_size = weight.shape[2:]
        stride = tuple(self._op.attrs_dict['stride'])
        padding = _process_padding(self)
        dilation = tuple(self._op.attrs_dict.get('dilation', [1] * len(padding)))
        groups = in_channels
        has_bias = np.count_nonzero(bias) != 0
        op_name = get_op_name(self._op_name)
        op_str = f'\t\tself.{op_name} = torch.nn.Conv2d(in_channels={in_channels},' \
                 f' out_channels={out_channels},' \
                 f' kernel_size={kernel_size},' \
                 f' stride={stride},' \
                 f' padding={padding},' \
                 f' dilation={dilation},' \
                 f' groups={groups},' \
                 f' bias={has_bias})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)
        self._op_handler_state.state_dict[op_name + '.weight'] = weight  # store params as np tensors
        self._op_handler_state.prepared_param_name_map[op_name + '.weight'] = (self._op.get_input_names[1], transform_order)
        if not self._op_handler_state.ignore_encodings:
            weight_enc = extract_tensor_encoding(self._op.inputs()[1])
            if weight_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.weight'] = weight_enc
        if has_bias:
            self._op_handler_state.state_dict[op_name + '.bias'] = bias  # store params as np tensors
            self._op_handler_state.prepared_param_name_map[op_name + '.bias'] = (self._op.get_input_names[2], None)
            if not self._op_handler_state.ignore_encodings:
                bias_enc = extract_tensor_encoding(self._op.inputs()[2])
                if bias_enc is not None:
                    self._op_handler_state.encodings['param_encodings'][op_name + '.bias'] = bias_enc

    def generate_execute_code(self):
        """
        Generate code for executing the op as a PyTorch module in file self._op_handler_state.f
        """
        temp_input_tensor = _add_input_pad_if_requried(self)

        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in self.get_input_op_names()]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]

        # If additional padding layer is added,
        # then its output tensor name will be passed as input to the current op.
        string_inputs[0] = string_inputs[0] if not temp_input_tensor else temp_input_tensor
        string_inputs = ", ".join(string_inputs)

        output_tensors = []
        for output_op_name in self.get_ops_output_names():
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ", ".join(output_tensors)

        execute_str = f"\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs})"
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)

        # Free output of additional padding layer after it is used.
        if temp_input_tensor:
            free_str = f"\t\t{temp_input_tensor} = None"
            self._op_handler_state.model_def_mgr.add_execute_code(free_str, self._op)


class FullyConnectedHandler(OpHandler):
    """ FullyConnected OpHandler """

    def __init__(self, op, op_handler_state):
        is_dynamic_fc = self._is_dynamic_fc(op)
        if is_dynamic_fc:
            num_inputs = len(op.inputs())

        else:
            num_inputs = 1
        super().__init__(op, op_handler_state, num_inputs=num_inputs)
        self.is_dynamic_fc = is_dynamic_fc

    # pylint: disable=no-self-use
    def _is_dynamic_fc(self, op: IrOp) -> bool:
        """
        Check whether given op is dynamic Linear/FC or not

        :param op: Op
        :return: True if this node is dynamic Linear else False
        """

        for input_ in op.inputs()[1:]:
            if not input_.is_static_tensor():
                return True
        return False

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        op_name = get_op_name(self._op_name)
        if self.is_dynamic_fc:
            op_str = f'\t\tself.{op_name} = elementwise_ops.DynamicLinear()'
            self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)
            return

        weights = self.get_weights()
        weight, bias = weights if len(weights) == 2 else (weights[0], None)
        weight = weight.transpose((1, 0))
        in_features = weight.shape[0]
        out_features = weight.shape[1]

        has_bias = np.count_nonzero(bias) != 0
        op_str = f'\t\tself.{op_name} = torch.nn.Linear(in_features={in_features},' \
                 f' out_features={out_features},' \
                 f' bias={has_bias})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)
        self._op_handler_state.state_dict[op_name + '.weight'] = weight.transpose((1, 0))  # store params as np tensors
        self._op_handler_state.prepared_param_name_map[op_name + '.weight'] = (self._op.get_input_names[1], None)
        if not self._op_handler_state.ignore_encodings:
            weight_enc = extract_tensor_encoding(self._op.inputs()[1])
            if weight_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.weight'] = weight_enc
        if has_bias:
            self._op_handler_state.state_dict[op_name + '.bias'] = bias  # store params as np tensors
            self._op_handler_state.prepared_param_name_map[op_name + '.bias'] = (self._op.get_input_names[2], None)
            if not self._op_handler_state.ignore_encodings:
                bias_enc = extract_tensor_encoding(self._op.inputs()[2])
                if bias_enc is not None:
                    self._op_handler_state.encodings['param_encodings'][op_name + '.bias'] = bias_enc

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """
        input_tensor = self._op.inputs()[0]
        if len(input_tensor.dims()) > 2:
            orig_input_name = self._op_handler_state.op_to_tensors[self.get_input_op_names()[0]]
            input_name = self.transposed_input_names.get(orig_input_name, orig_input_name)
            flatten_str = f'\t\t{input_name}_flattened = torch.flatten({input_name}, 1)'
            self.transposed_input_names[orig_input_name] = input_name + '_flattened'
            self._op_handler_state.model_def_mgr.add_execute_code(flatten_str, self._op, input_name, f"{input_name}_flattened")
            # in case the transpose is added to the original_input, clear the tmp input space.
            if orig_input_name != input_name:
                var_clear_str = f'\t\t{input_name} = None'
                self._op_handler_state.model_def_mgr.add_execute_code(var_clear_str, self._op)
        super().generate_execute_code()


class NeuronHandler(OpHandler):
    """ Neuron OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    @staticmethod
    def _get_neuron_translation_key_dict(neuron_type_key):
        if neuron_type_key == 'operation':
            neuron_translation_key_dict = {
                'RELU': ir_graph_lib.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU,
                'TANH': ir_graph_lib.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH,
                'SIGMOID': ir_graph_lib.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID,
                'ELU': ir_graph_lib.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU,
                'RELU_MIN_MAX': ir_graph_lib.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX,
                'HARD_SWISH': ir_graph_lib.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH,
                'HARD_SIGMOID': ir_graph_lib.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID,
                'SOFTPLUS': ir_graph_lib.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS,
                'GELU': ir_graph_lib.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU,
            }
        else:
            neuron_translation_key_dict = {
                'RELU': ir_graph_lib.QNN_OP_RELU,
                'TANH': ir_graph_lib.QNN_OP_TANH,
                'SIGMOID': ir_graph_lib.QNN_OP_SIGMOID,
                'ELU': ir_graph_lib.QNN_OP_ELU,
                'RELU_MIN_MAX': ir_graph_lib.QNN_OP_RELU_MIN_MAX,
                'HARD_SWISH': ir_graph_lib.QNN_OP_HARD_SWISH,
            }
        return neuron_translation_key_dict

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        # TODO: IR Graph NeuronOp is being modified to ElementwiseNeuronOp.
        # Modify the code to support only the later once the change is inside QNN
        neuron_type_key = 'operation' if 'operation' in self._op.attrs_dict else 'neuron_type'
        neuron_type = self._op.attrs_dict[neuron_type_key]

        neuron_translation_key_dict = self._get_neuron_translation_key_dict(neuron_type_key)

        # Softplus is resolved under ElementWiseNeuron from QNN version >= 2.21
        if 'SOFTPLUS' in neuron_translation_key_dict and neuron_type == neuron_translation_key_dict['SOFTPLUS']:
            beta = self.get_attr_value('beta')
            threshold = self.get_attr_value('threshold')
            module = f'torch.nn.Softplus(beta={beta}, threshold={threshold})'
        elif 'GELU' in neuron_translation_key_dict and neuron_type == neuron_translation_key_dict['GELU']:
            module = 'torch.nn.GELU()'
        elif neuron_type == neuron_translation_key_dict['RELU']:
            module = 'torch.nn.ReLU()'
        elif neuron_type == neuron_translation_key_dict['TANH']:
            module = 'torch.nn.Tanh()'
        elif neuron_type == neuron_translation_key_dict['SIGMOID']:
            module = 'torch.nn.Sigmoid()'
        elif neuron_type == neuron_translation_key_dict['ELU']:
            alpha = self._op.attrs_dict['alpha']
            module = f'torch.nn.ELU(alpha={alpha})'
        elif neuron_type == neuron_translation_key_dict['RELU_MIN_MAX']:
            clip_min = self._op.attrs_dict['min_value']
            clip_max = self._op.attrs_dict['max_value']
            if clip_min == 0 and clip_max == 6:
                module = 'torch.nn.ReLU6()'
            else:
                input_tensor_dtype = self._op.inputs()[0].data_type_as_numpy_type()
                if input_tensor_dtype == 'i':
                    clip_max = min(clip_max, sys.maxsize)
                module = f'torch.nn.Hardtanh(min_val={clip_min}, max_val={clip_max})'
        elif neuron_type == neuron_translation_key_dict['HARD_SWISH']:
            module = 'torch.nn.Hardswish()'
        elif neuron_type == neuron_translation_key_dict['HARD_SIGMOID']:
            module = 'torch.nn.Hardsigmoid()'
        else:
            raise RuntimeError(f'Unrecognized neuron op {neuron_type}')

        op_str = f'\t\tself.{get_op_name(self._op_name)} = {module}'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class ConcatHandler(OpHandler):
    """ Concat OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=None)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        axis = self._op.attrs_dict.get('axis', 0)
        axis = self._update_axis_using_axis_information(axis)
        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.Concat(axis={axis})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class PoolHandler(OpHandler):
    """ OpHandler for MaxPool, AveragePool """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)
        self._padding_info = {}

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        op_type_to_torch_module_dict = {
            'PoolMax2d': 'torch.nn.MaxPool2d',
            'PoolMax3d': 'torch.nn.MaxPool3d',
            'PoolAvg2d': 'torch.nn.AvgPool2d',
            'PoolAvg3d': 'torch.nn.AvgPool3d',
            'L2Pool2d': 'torch.nn.LPPool2d',
        }
        torch_module_name = op_type_to_torch_module_dict.get(self._op.attrs_dict['pool_type'])
        if not torch_module_name:
            raise AssertionError(f"Unrecognized pool op {self._op.attrs_dict['pool_type']}")

        kwargs_dict = dict()
        kwargs_dict['kernel_size'] = self.get_attr_value('filter_size')
        kwargs_dict['stride'] = self.get_attr_value('stride')
        kwargs_dict['ceil_mode'] = False
        padding = _process_padding(self)

        if self._op.attrs_dict['pool_type'] in ('PoolAvg2d', 'PoolAvg3d'):
            # Retrieve count_include_pad from original torch module
            # as QNN does not set it properly
            kwargs_dict['count_include_pad'] = self.get_attr_value('count_include_pad')
            kwargs_dict['padding'] = padding

        if self._op.attrs_dict['pool_type'] in ('PoolMax2d', 'PoolMax3d'):
            # Retrieve dilation from original torch module
            # as QNN does not set it properly
            kwargs_dict['dilation'] = self.get_attr_value('dilation')
            kwargs_dict['padding'] = padding

        if self._op.attrs_dict['pool_type'] == 'L2Pool2d':
            kwargs_dict['norm_type'] = 2  # QNN Converter fails if not 2

        init_args = _create_init_args_from_kwargs_dict(kwargs_dict)
        op_str = f'\t\tself.{op_name} = {torch_module_name}({init_args})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def generate_execute_code(self):
        """
        Generate code for executing the node as a PyTorch module in file self._op_handler_state.f
        """
        pad_value = "float('-inf')" if self._op.attrs_dict['pool_type'] in ('PoolMax2d', 'PoolMax3d') else 0
        temp_input_tensor = _add_input_pad_if_requried(self, pad_value=pad_value)

        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in self.get_input_op_names()]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]

        # If additional padding layer is added,
        # then its output tensor name will be passed as input to current op.
        string_inputs[0] = string_inputs[0] if not temp_input_tensor else temp_input_tensor
        string_inputs = ", ".join(string_inputs)

        output_tensors = []
        for output_op_name in self.get_ops_output_names():
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ", ".join(output_tensors)

        execute_str = f"\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs})"
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)

        # Free output of additional padding layer after it is used.
        if temp_input_tensor:
            free_str = f"\t\t{temp_input_tensor} = None"
            self._op_handler_state.model_def_mgr.add_execute_code(free_str, self._op)


class ReshapeHandler(OpHandler):
    """ Reshape OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        op_type_to_torch_module_dict = {
            'Reshape': 'elementwise_ops.Reshape()',
            'Transpose': 'elementwise_ops.Permute()',
        }
        torch_module_name = op_type_to_torch_module_dict.get(self._op.type)
        if torch_module_name:
            op_str = f'\t\tself.{op_name} = {torch_module_name}'
        else:
            raise AssertionError(f'Unrecognized reshape op {self._op.type}')
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    # pylint: disable=too-many-locals
    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        op_name = get_op_name(self._op_name)
        input_op_names = self.get_input_op_names()
        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name] for input_op_name in input_op_names]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        if len(string_inputs) != 1:
            logger.error('number of inputs for reshape should be 1')
            assert len(string_inputs) == 1
        string_inputs = ', '.join(string_inputs)

        output_tensors = []
        for output_op_name in self.get_ops_output_names():
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        if self._op.type == 'Reshape':
            shape = self._op.get_output_shapes()[0]
            input_tensor = self._op.inputs()[0]
            # ONNX reshape, and by extension QNN reshape, can have '0' in an index symbolizing for that dimension to
            # remain the same size as the input tensor. PyTorch does not have this convention, so we replace '0' with
            # the actual size of the input tensor at that dimension.
            for idx, size in enumerate(shape):
                if size == 0:
                    shape[idx] = input_tensor.dims()[idx]
                # We are keeping first dim(batch_dim) as inferred so any other inferred dims are changed to the
                # respective output tensor dim.

            execute_str = f'\t\t{string_outputs} = self.{op_name}({string_inputs}, {[-1] + list(shape)[1:]})'

        elif self._op.type == 'Transpose':
            perm = self.get_effective_permute_order()
            execute_str = f'\t\t{string_outputs} = self.{op_name}({string_inputs}, {list(perm)})'
        else:
            raise AssertionError('Unrecognized reshape op')
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)

    def get_effective_permute_order(self):
        """ Gets the effective permute order for the transpose op """
        perm = self._op.attrs_dict['perm']
        op_axis_info = self._op_handler_state.op_axis_info[self._op_name]
        # Handle perm at input
        input_tensor_name = self.get_input_op_names()[0]
        if op_axis_info.input_transform.get(input_tensor_name, None) is not None:
            transform_order = op_axis_info.input_transform.get(input_tensor_name, None)
            perm = np.array(perm).choose(transform_order).tolist()
        # Handle perm at output
        output_tensor_name = self.get_ops_output_names()[0]
        if op_axis_info.output_transform.get(output_tensor_name, None) is not None:
            transform_order = op_axis_info.output_transform.get(output_tensor_name, None)
            perm = np.array(transform_order).choose(perm).tolist()
        return perm

    def handle_input_transpose(self):
        """ Handle Input Transpose: Skip is op type is Transpose """
        if self._op.type == 'Reshape':
            super().handle_input_transpose()

    def handle_output_transpose(self):
        """ Handle Output Transpose: Skip is op type is Transpose """
        if self._op.type == 'Reshape':
            super().handle_output_transpose()


class ExpandHandler(OpHandler):
    """ Expand OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        op_str = f'\t\tself.{op_name} = elementwise_ops.Expand()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        input_op_names = self.get_input_op_names()
        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name] for input_op_name in input_op_names]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        if len(string_inputs) != 1:
            logger.error('number of inputs for expand should be 1')
            assert len(string_inputs) == 1
        string_input = string_inputs[0]

        output_tensors = []
        for output_op_name in self.get_ops_output_names():
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        shape = self._op.outputs()[0].dims()
        input_tensor = self._op.inputs()[0]

        if len(shape) == len(input_tensor.dims()):
            # PyTorch expand can have '-1' in an index symbolizing for that dimension to
            # remain the same size as the input tensor. IR converter change it to '1'
            # so we replace '1' with '-1' at that dimension.
            for idx, size in enumerate(shape):
                if size == 1 and shape[idx] != input_tensor.dims()[idx]:
                    shape[idx] = -1

            # If shape parameter has same dimension with input tensor
            # We check whether the two shapes have the same axis value
            # If both have same axis value fill them with -1, otherwise use the shape parameter value
            # This will let PyTorch do automatic shape inference in expand operation
            shape = [-1 if x == y else x for x, y in zip(shape, input_tensor.dims())]
        shape = update_shape_using_transform_order(
            self._op_handler_state.tensor_axis_info[self._op.outputs()[0].name()].transform_order, shape)
        execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_input}, torch.Size({list(shape)}))'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_input, string_outputs)


class ResizeHandler(OpHandler):
    """ Resize OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        kwargs_dict = dict()
        op_name = get_op_name(self._op_name)
        if self._op.type == 'Resize':
            interpolation_mode, transformation_mode, exclude_outside = \
                (self._op.attrs_dict.get(k, 0) for k in
                 ['interpolation_mode', 'transformation_mode', 'exclude_outside'])

            if exclude_outside:
                raise AssertionError(f'Unrecognized Resize op with exclude_outside: {exclude_outside}')

            input_shape = self._op.inputs()[0].dims()
            mode, align_corners = ResizeHandler._get_mode_and_align_corners(input_shape,
                                                                            interpolation_mode,
                                                                            transformation_mode)
            if align_corners:
                kwargs_dict['align_corners'] = align_corners

        elif self._op.type == 'ResizeNearestNeighbor':
            mode = 'nearest'

        elif self._op.type == 'ResizeBilinear':
            mode = 'bilinear'
            kwargs_dict['align_corners'] = self._op.attrs_dict['align_corners']

        else:
            raise TypeError(f"Resize type '{self._op.type}' is not supported.")

        # Output of Resize Ops is always in Channel last format
        # e.g., 2D output_shape => (minibatch, 16, 58, channels), 3D output_shape => (minibatch, 8, 25, 34, channels)
        #   We only need (16, 58) for 2D output (spatial), (8, 25, 34) for 3D output (volumetric)
        output_shape = self._op.outputs()[0].dims()
        kwargs_dict['size'] = output_shape[1:-1]

        kwargs_dict['mode'] = mode

        init_args = _create_init_args_from_kwargs_dict(kwargs_dict, string_args={'mode'})
        op_str = f'\t\tself.{op_name} = torch.nn.Upsample({init_args})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    @staticmethod
    def _get_mode_and_align_corners(input_shape: Tuple[int, int], interpolation_mode: int, transformation_mode: int) \
            -> Tuple[str, bool]:
        """
        Get the upsample mode and align corners values corresponding to the given resize parameters.

        :param input_shape: Input tensor shape
        :param interpolation_mode: QNN interpolation mode enum for the resize op
        :param transformation_mode: QNN transformation mode enum for the resize op
        :return: Tuple of upsample mode and align corners value
        """

        # Interpolation mode and transformation mode information can be found in QNN MasterOpDef xml
        mode_map = {
            (0, 3): ('nearest', False),
            (0, 0): ('nearest', False),
            (1, 0): (ResizeHandler._get_linearity(input_shape), False),
            (1, 1): (ResizeHandler._get_linearity(input_shape), False),
            (1, 2): (ResizeHandler._get_linearity(input_shape), True),
            (1, 3): (ResizeHandler._get_linearity(input_shape), False),
            (2, 0): ('bicubic', False),
            (2, 1): ('bicubic', False),
            (2, 2): ('bicubic', True),
        }
        if (interpolation_mode, transformation_mode) not in mode_map:
            raise AssertionError(f'Unrecognized Resize op with interpolation_mode: {interpolation_mode}, '
                                 f'transformation_mode: {transformation_mode}')
        return mode_map[(interpolation_mode, transformation_mode)]

    @staticmethod
    def _get_linearity(input_shape) -> str:
        """
        Get the linearity mode corresponding to the given input shape.

        :param input_shape: Input tensor shape
        :return: Linearity mode for the given input shape
        """

        if len(input_shape) == 4:
            return 'bilinear'
        if len(input_shape) == 5:
            return 'trilinear'
        raise AssertionError(f'Unrecognized Resize op with input shape: {input_shape}')


class IgnoreHandler(OpHandler):
    """ OpHandler for ops which can be ignored """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        return


class SoftmaxHandler(OpHandler):
    """ Softmax OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        axis = self._op.attrs_dict['axis']
        axis = self._update_axis_using_axis_information(axis)
        if self._op.type == 'Softmax':
            op_str = f'\t\tself.{get_op_name(self._op_name)} = torch.nn.Softmax(dim={axis})'
        elif self._op.type == 'LogSoftmax':
            op_str = f'\t\tself.{get_op_name(self._op_name)} = torch.nn.LogSoftmax(dim={axis})'
        else:
            raise RuntimeError('Unrecognized softmax op')

        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class InstanceNormHandler(OpHandler):
    """ InstanceNorm OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    # pylint: disable=too-many-statements
    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        num_dims_of_ops_input_tensor = len(self._op.inputs()[0].dims())
        if num_dims_of_ops_input_tensor < 4:
            module_type = 'torch.nn.InstanceNorm1d'
        elif num_dims_of_ops_input_tensor == 4:
            module_type = 'torch.nn.InstanceNorm2d'
        elif num_dims_of_ops_input_tensor == 5:
            module_type = 'torch.nn.InstanceNorm3d'
        else:
            raise AssertionError(f'Unrecognized InstanceNorm op with {num_dims_of_ops_input_tensor} '
                                 f'dimensional input.')

        op_name = get_op_name(self._op_name)

        kwargs_dict = dict()
        attributes = ['eps', 'affine', 'num_features']
        for attr in attributes:
            kwargs_dict[attr] = self.get_attr_value(attr)

        if momentum := self.get_attr_value('momentum'):
            kwargs_dict['momentum'] = momentum

        if track_running_stats := self.get_attr_value('track_running_stats'):
            kwargs_dict['track_running_stats'] = track_running_stats

        init_args = _create_init_args_from_kwargs_dict(kwargs_dict)

        op_str = f'\t\tself.{op_name} = {module_type}({init_args})'

        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

        if kwargs_dict['affine']:
            self._op_handler_state.state_dict[op_name + '.weight'] = self.get_attr_value('weight')  # store params as np tensors
            self._op_handler_state.prepared_param_name_map[op_name + '.weight'] = (self._op.get_input_names[1], None)
            self._op_handler_state.state_dict[op_name + '.bias'] = self.get_attr_value('bias')  # store params as np tensors
            self._op_handler_state.prepared_param_name_map[op_name + '.bias'] = (self._op.get_input_names[2], None)
            if not self._op_handler_state.ignore_encodings:
                weight_enc = extract_tensor_encoding(self._op.inputs()[1])
                if weight_enc is not None:
                    self._op_handler_state.encodings['param_encodings'][op_name + '.weight'] = weight_enc
                bias_enc = extract_tensor_encoding(self._op.inputs()[2])
                if bias_enc is not None:
                    self._op_handler_state.encodings['param_encodings'][op_name + '.bias'] = bias_enc
        if track_running_stats:
            self._op_handler_state.state_dict[op_name + '.running_mean'] = self.get_attr_value('running_mean')  # store params as np tensors
            self._op_handler_state.state_dict[op_name + '.running_var'] = self.get_attr_value('running_var')  # store params as np tensors


class LayerNormHandler(OpHandler):
    """ LayerNorm OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def _can_construct_pytorch_layernorm(self):
        """
        Check if PyTorch LayerNorm can be constructed from the QNN op without permuting the input/output tensors
        """
        input_dims = self._op.inputs()[0].dims()
        input_dims = update_shape_using_transform_order(
            self._op_handler_state.tensor_axis_info[self._op.inputs()[0].name()].transform_order, input_dims)
        axes = self._op.attrs_dict['axes']
        axes = [self._update_axis_using_axis_information(axis) for axis in axes]

        n_input = len(input_dims)
        n_axes = len(axes)

        valid_axes = list(range(n_input))

        for n in range(-1, -n_axes - 1, -1):
            if axes[n] != valid_axes[n]:
                return False

        return True

    # pylint: disable=too-many-locals
    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        op_name = get_op_name(self._op_name)
        gamma, beta = (tensor.get_data() for tensor in self._op.inputs()[1:])

        # For a normalized_shape with 3 dimensions, QNN seems to erroneously expand the param shapes by 1 dimension.
        # The below logic is for squeezing gamma and beta back to the correct dimensions, if required.
        # The assumption is that the extra dimensions added by QNN will be on the leftmost axes.
        # LayerNorm weight and bias dims are expected to be 1 less than dims in the LayerNorm input.
        dims_inferred_from_layernorm = len(self._op.inputs()[0].dims()) - 1
        dims_from_gamma = gamma.ndim
        if dims_inferred_from_layernorm != dims_from_gamma:
            axes_to_squeeze = tuple(range(dims_from_gamma - dims_inferred_from_layernorm))
            gamma = np.squeeze(gamma, axis=axes_to_squeeze)
            beta = np.squeeze(beta, axis=axes_to_squeeze)

        kwargs_dict = dict()

        if self._can_construct_pytorch_layernorm():
            kwargs_dict['normalized_shape'] = gamma.shape
            kwargs_dict['eps'] = self._op.attrs_dict['epsilon']
            kwargs_dict['elementwise_affine'] = True
            module_type = 'torch.nn.LayerNorm'

        else:
            # QNN is flexible about specifying axes, enabling LayerNorm to normalize only along the desired axes
            # Unlike PyTorch, which only performs normalization along the last 'D' axes
            # To align QNN and PyTorch for the purpose of LayerNorm, a permutation of the input tensor is necessary
            # This permutation reorders the tensor's axes, ensuring that the axes specified in the QNN operation
            # are positioned at the last dimensions
            # This enables PyTorch's LayerNorm to perform normalization along the last dimensions as expected
            # Additionally, this necessitates the calculation of the correct normalized shape
            # based on the input tensor shape and the specified axes, along with the reshaping beta and gamma parameters
            layernorm_op = self._op
            input_shape = layernorm_op.inputs()[0].dims()
            axes = self._op.attrs_dict['axes']
            input_shape = update_shape_using_transform_order(
                self._op_handler_state.tensor_axis_info[layernorm_op.inputs()[0].name()], input_shape)

            axes = [self._update_axis_using_axis_information(axis) for axis in axes]

            kwargs_dict['input_shape'] = input_shape
            kwargs_dict['axes'] = axes
            kwargs_dict['eps'] = self._op.attrs_dict['epsilon']

            normalized_shape = list(input_shape[a] for a in axes)
            gamma = gamma.reshape(normalized_shape)
            beta = beta.reshape(normalized_shape)

            module_type = 'emitter_ops.CustomLayerNorm'

        init_args = _create_init_args_from_kwargs_dict(kwargs_dict)

        op_str = f'\t\tself.{op_name} = {module_type}({init_args})'

        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)
        self._op_handler_state.state_dict[op_name + '.weight'] = gamma  # store params as np tensors
        self._op_handler_state.prepared_param_name_map[op_name + '.weight'] = (self._op.get_input_names[1], None)
        self._op_handler_state.state_dict[op_name + '.bias'] = beta   # store params as np tensors
        self._op_handler_state.prepared_param_name_map[op_name + '.bias'] = (self._op.get_input_names[2], None)
        if not self._op_handler_state.ignore_encodings:
            weight_enc = extract_tensor_encoding(self._op.inputs()[1])
            if weight_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.weight'] = weight_enc
            bias_enc = extract_tensor_encoding(self._op.inputs()[2])
            if bias_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.bias'] = bias_enc


class PreluHandler(OpHandler):
    """ PReLU OpHandler (PReLU, LeakyReLU) """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        weight = self.get_weights()[0]  # TODO: Fix
        op_name = get_op_name(self._op_name)

        # Prelu coming via TF can have weight (parameter) with ndims > 1, so we use a customPrelu
        # definition which allows this. In this case, we need to align weight axis layout in similar
        # way we do for the input (as we are performing Elementwise operation between input and weight)
        axis_info = self._op_handler_state.tensor_axis_info[self._op.get_input_names[1]]
        transform_order = get_required_transpose(axis_info, self._op.type)
        if transform_order is not None:
            weight = weight.transpose(transform_order)
        self._op_handler_state.op_axis_info[self._op_name].input_transform[self._op.get_input_names[1]] = transform_order

        op_str = f"\t\tself.{op_name} = emitter_ops.CustomPReLU({weight.shape})"

        self._op_handler_state.state_dict[f"{op_name}.weight"] = weight  # store params as np tensors
        self._op_handler_state.prepared_param_name_map[f"{op_name}.weight"] = (self._op.get_input_names[1], transform_order)
        if not self._op_handler_state.ignore_encodings:
            weight_enc = extract_tensor_encoding(self._op.inputs()[1])
            if weight_enc is not None:
                self._op_handler_state.encodings['param_encodings'][f"{op_name}.weight"] = weight_enc

        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)


class GatherHandler(OpHandler):
    """ Gather OpHandler (Embedding) """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)
        self._is_index_select = False
        if not self._op.inputs()[0].is_static_tensor() or len(self._op.inputs()[0].dims()) != 2:
            self._is_index_select = True
            self._num_inputs = 2

    def get_input_op_names(self) -> List[str]:
        """
        Get a list of input op names for the ops.

        :return: List of parameter ops names
        """
        if not self._is_index_select:
            # NOTE: Unlike Conv op ( input_names => [ops_output, conv.weight, conv.bias] )
            # Gather op has different input_names format ( input_names => [embedding.weight, ops_output] )
            return self._op.get_input_names[self._num_inputs:]
        return super().get_input_op_names()

    def get_parameter_op_names(self) -> List[str]:
        """
        Get a list of parameter ops names for the ops.

        :return: List of parameter ops names
        """

        if not self._is_index_select:
            return self._op.get_input_names[:self._num_inputs]
        return super().get_parameter_op_names()

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        if self._is_index_select:
            index_tensor = self._op.inputs()[1]
            if len(index_tensor.dims()) > 1:
                op_str = f"\t\tself.{op_name} = emitter_ops.IndexSelect()"
            else:
                op_str = f"\t\tself.{op_name} = elementwise_ops.IndexSelect()"

                axis = self._op.attrs_dict["axis"]
                dim_size = self._op.inputs()[0].dims()[axis]

                # If the indices are a constant tensor, they may need special handling as described below. In the case
                # the indices are a model input or intermediate tensor, no special handling is required.
                index_tensor = self._op.inputs()[1]
                # If index_tensor is a static tensor, Skip special handling.
                if index_tensor.is_static_tensor():
                    index_tensor_name = index_tensor.name()
                    indices = self._op_handler_state.state_dict.get(get_op_name(index_tensor_name))
                    assert indices is not None
                    # In the case of ONNX's Gather op, the indices can take values of [-dim_size, dim_size - 1].
                    # However, in torch's index_select, indices can only be within [0, dim_size - 1].
                    # Below logic replaces negative elements in the indices tensor with positive equivalents.
                    for idx, index in enumerate(indices):
                        if index < 0:
                            indices[idx] = dim_size + index
                else:
                    # In case of index value dynamic, if the values are negative it needs to be handled at run time.
                    op_str = f"\t\tself.{op_name} = emitter_ops.IndexSelect()"
        else:
            weight = self.get_weights()[0]
            num_embeddings, embedding_dim = weight.shape
            op_str = f"\t\tself.{op_name} = torch.nn.Embedding({num_embeddings}, {embedding_dim})"
            self._op_handler_state.state_dict[f"{op_name}.weight"] = weight     # store params as np tensors
            self._op_handler_state.prepared_param_name_map[op_name + '.weight'] = (self._op.get_input_names[0], None)
            if not self._op_handler_state.ignore_encodings:
                weight_enc = extract_tensor_encoding(self._op.inputs()[1])
                if weight_enc is not None:
                    self._op_handler_state.encodings['param_encodings'][op_name + '.weight'] = weight_enc
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        if self._is_index_select:
            input_op_names = self.get_input_op_names()

            output_tensors = []
            output_ops_names = self.get_ops_output_names()
            for output_ops_name in output_ops_names:
                self._op_handler_state.op_to_tensors[output_ops_name] = _create_variable_name(output_ops_name)
                output_tensors.append(self._op_handler_state.op_to_tensors[output_ops_name])
            string_outputs = ', '.join(output_tensors)

            input_tensor = self._op_handler_state.op_to_tensors[input_op_names[0]]
            input_tensor = self.transposed_input_names.get(input_tensor, input_tensor)

            dim = self._op.attrs_dict['axis']
            index = self._op_handler_state.op_to_tensors[input_op_names[1]]

            execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({input_tensor}, {dim}, {index})'
            self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, input_tensor, string_outputs)

        else:
            super().generate_execute_code()


class GeluHandler(OpHandler):
    """ Gelu OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_str = f'\t\tself.{get_op_name(self._op_name)} = torch.nn.GELU()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class MatMulHandler(OpHandler):
    """ MatMul OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=2)

    def _is_from_torch_linear_without_bias(self) -> bool:
        """
        Check original module was from Linear(..., bias=False) or not

        :return: True if original module was from Linear(..., bias=False) else False
        """
        return self._op_handler_state.is_from_linear_without_bias(self._op)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        attrs_dict = self._op.attrs_dict
        op_name = get_op_name(self._op_name)

        op_str = f'\t\tself.{op_name} = elementwise_ops.MatMul()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))

        attr_to_input_idx = {
            'transpose_in0': 0,
            'transpose_in1': 1
        }
        op_axis_info = self._op_handler_state.op_axis_info[self._op_name]
        # In case any input needs to be transposed before the matmul operation, merge
        # this transpose order with the input transpose requirement of the op and update op_axis_info.
        for attr_key, input_idx in attr_to_input_idx.items():
            if attrs_dict[attr_key]:
                in_tensor = self._op.inputs()[input_idx]
                ndim = len(in_tensor.dims())
                permute_order = tuple(range(ndim-2)) + (ndim-1, ndim-2)
                if in_tensor.name() in op_axis_info.input_transform:
                    existing_transform_order = op_axis_info.input_transform[in_tensor.name()]
                    permute_order = combine_transpose_order(existing_transform_order, permute_order)
                self._op_handler_state.op_axis_info[self._op_name].input_transform[in_tensor.name()] = permute_order

    def generate_execute_code(self):
        """
        Generate code for executing the ops as a Pytorch module in file self._op_handler_state.f.
        """
        input_op_names = self.get_input_op_names()
        if self._op_handler_state.keep_linear_without_bias and self._is_from_torch_linear_without_bias():
            string_inputs = [self._op_handler_state.node_to_tensors[input_op_names[0]]]
        else:
            string_inputs = [self._op_handler_state.op_to_tensors[input_node_name]
                             for input_node_name in input_op_names]

        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        string_inputs = ', '.join(string_inputs)

        output_tensors = []
        output_op_names = [output_tensor.name() for output_tensor in self._op.outputs()]
        for output_op_name in output_op_names:
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)


class CastHandler(OpHandler):
    """ Cast Op Handler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        # We get the conversion type by looking at the output tensors type of the cast op. To get the correct
        # PyTorch type, we get the output tensors data type in numpy format, then create a dummy array to get the actual
        # data type for PyTorch. This is because "data_type_as_numpy_type" could come back as 'b' for int8 which
        # Numpy is aware of but PyTorch is not. By doing dtype.__str__() we get the correct 'int8'.
        set_of_output_tensor_types = {output_tensor.data_type_as_numpy_type() for output_tensor in self._op.outputs()}
        assert len(set_of_output_tensor_types) == 1, \
            f"Creating Cast Op failed. Not all output tensor types are the same. Got: {', '.join(set_of_output_tensor_types)}"

        to_type = qnn_numpy_type_to_actual_numpy_dtype(self._op.outputs()[0])
        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.Cast(torch.{to_type})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class CumulativeOpHandler(OpHandler):
    """ Cumulative Op Handler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_type = self._op.type

        if op_type == 'CumulativeSum':
            op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.CumSum()'
        else:
            raise AssertionError(f'Unrecognized cumulative op {op_type}')

        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        input_op_names = self.get_input_op_names()
        string_input = self._op_handler_state.op_to_tensors[input_op_names[0]]
        string_input = self.transposed_input_names.get(string_input, string_input)

        output_op_name = self._op.outputs()[0].name()
        output_tensor = _create_variable_name(output_op_name)
        self._op_handler_state.op_to_tensors[output_op_name] = output_tensor

        dim = self._op.attrs_dict['axis']
        dim = self._update_axis_using_axis_information(dim)
        execute_str = f'\t\t{output_tensor} = self.{get_op_name(self._op_name)}({string_input}, dim={dim})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_input, output_tensor)


class SplitOpHandler(OpHandler):
    """ Split Op Handler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.Split()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))

    # pylint: disable=too-many-locals
    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        input_op_names = self.get_input_op_names()
        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name] for input_op_name in input_op_names]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        if len(string_inputs) != 1:
            logger.error('number of inputs for expand should be 1')
            assert len(string_inputs) == 1
        string_input = string_inputs[0]

        output_tensors = []
        for output_op_name in self.get_ops_output_names():
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        input_shape = self._op.inputs()[0].dims()
        split_indices = self._op.attrs_dict['split_index']
        axis = self._op.attrs_dict['axis']

        split_sizes = SplitOpHandler._get_split_size_or_sections(axis, input_shape, split_indices)

        axis = self._update_axis_using_axis_information(axis)

        kwargs_dict = dict()
        kwargs_dict['split_size_or_sections'] = split_sizes
        kwargs_dict['dim'] = axis

        init_args = _create_init_args_from_kwargs_dict(kwargs_dict)

        execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_input}, {init_args})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_input, string_outputs)

    @staticmethod
    def _get_split_size_or_sections(axis: int, input_shape: Tuple[int], split_indices: np.ndarray) -> \
            Union[int, List[int]]:
        """
        Given a list of split indices, generate a list of split sizes, or a single split size value if applicable.
        A single split size is possible if all split sizes are equal (also valid if all split sizes except the last are
        equal, and the last split size is less than or equal to all other split sizes).
        Example: Given a tensor of length 10, split indices of [3, 6, 9] will lead to split sizes of [3, 3, 3, 1]. This
        is equivalent to providing a single split size of 3.
        However, given the same tensor with split indices of [3, 6], this will lead to split sizes of [3, 3, 4]. In this
        case, the split sizes must be provided as is in order to produce the correct output tensors.

        :param axis: Axis to split
        :param input_shape: Input tensor shape
        :param split_indices: List of indices to split input tensor with
        :return: List of split sizes, or a single split size value
        """

        # Generate an array of split sizes by taking the differences between adjacent values in split_indices.
        # First, insert index 0 to the start and the total length of the tensor axis dimension to the end.
        split_indices = np.insert(split_indices, 0, 0)
        split_indices = np.append(split_indices, input_shape[axis])
        split_sizes = np.diff(split_indices)

        # Detect whether the split size array can equivalently be expressed as a single split size value. The condition
        # is that all split sizes except the last must be equal, and the last split size must be <= all other split
        # sizes.
        if np.all(split_sizes[:-1] == split_sizes[0]) and split_sizes[-1] <= split_sizes[0]:
            split_sizes = int(split_sizes[0])
        else:
            split_sizes = split_sizes.tolist()

        return split_sizes


class ArgOpHandler(OpHandler):
    """ ArgMin and ArgMax OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        arg_type = self._op.attrs_dict['arg_type']
        if arg_type == 'Argmin':
            op_str = f'\t\tself.{op_name} = elementwise_ops.Argmin()'
        elif arg_type == 'Argmax':
            op_str = f'\t\tself.{op_name} = elementwise_ops.Argmax()'
        else:
            raise AssertionError(f'Unrecognized elementwise arg op {arg_type}')
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """
        input_op_names = self.get_input_op_names()
        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in input_op_names]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        string_inputs = ', '.join(string_inputs)

        output_tensors = []
        output_op_names = self.get_ops_output_names()
        for output_op_name in output_op_names:
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        # Get the values for the attributes axis and keepdims to be passed in the fwd pass
        axis = self._op.attrs_dict['axis']
        axis = self._update_axis_using_axis_information(axis)
        keepdims = self._op.attrs_dict['keep_dims']

        execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs}, dim={axis}, keepdims={keepdims})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)


class L2NormHandler(OpHandler):
    """ L2Normalization OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        if self._op.type == 'L2Norm':
            op_str = f'\t\tself.{op_name} = elementwise_ops.Normalize()'
        else:
            raise AssertionError(f'Unrecognized op {self._op.type}')
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """
        input_op_names = self.get_input_op_names()
        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in input_op_names]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        string_inputs = ', '.join(string_inputs)

        output_tensors = []
        for output_op_name in self.get_ops_output_names():
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        if 'axes' in self._op.attrs_dict:
            ir_axis = self._op.attrs_dict['axes']
        else:
            ir_axis = self._op.attrs_dict['axis']
        ir_axis = self._update_axis_using_axis_information(ir_axis)
        execute_str = f"\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs}, dim={ir_axis})"
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)


class NonMaxSuppressionHandler(OpHandler):
    """ NonMaxSuppression OpHandler """

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state, num_inputs=2)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        if self._op.type == 'NonMaxSuppression':
            iou_threshold = self._op.attrs_dict["iou_threshold"]
            score_threshold = self._op.attrs_dict["score_threshold"]
            max_output_boxes_per_class = self._op.attrs_dict["max_boxes_selected"]
            op_str = f'\t\tself.{op_name} = elementwise_ops.NonMaxSuppression({iou_threshold}, {score_threshold}, {max_output_boxes_per_class})'
        else:
            raise AssertionError(f'Unrecognized op {self._op.type}')
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def generate_execute_code(self):
        """
        Generate code for executing the node as a Pytorch module in file self._op_handler_state.f.
        """
        input_node_names = self.get_input_op_names()
        string_inputs = [self._op_handler_state.op_to_tensors[input_node_name]
                         for input_node_name in input_node_names]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        string_inputs = ', '.join(string_inputs)

        output_tensors = []
        output_node_name = self.get_ops_output_names()[0]  # Ignoring the non-mandatory output of NMS

        self._op_handler_state.op_to_tensors[output_node_name] = _create_variable_name(output_node_name)
        output_tensors.append(self._op_handler_state.op_to_tensors[output_node_name])
        string_outputs = ', '.join(output_tensors)

        execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)


class StridedSliceOpHandler(OpHandler):
    """ StridedSlice Op Handler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.StridedSlice()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """
        input_op_names = self.get_input_op_names()
        string_input = self._op_handler_state.op_to_tensors[input_op_names[0]]
        string_input = self.transposed_input_names.get(string_input, string_input)

        output_op_name = self.get_ops_output_names()[0]
        self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
        output_tensor = self._op_handler_state.op_to_tensors[output_op_name]

        slice_ranges = self._op.attrs_dict['ranges'].tolist()
        # If explicit ranges are given per dimension, a separate ONNX op gets created from the resulting PyTorch module.
        # If the range covers the entire dimension, using 'None' instead optimizes the export so no additional slice op
        # is created for that particular dimension.
        for idx, slice_range in enumerate(slice_ranges):
            if slice_range[0] == 0 and self._op.inputs()[0].dims()[idx] == slice_range[1] and \
                    slice_range[2] == 1:
                slice_range[0] = None
                slice_range[1] = None

        input_axis_info = self._op_handler_state.tensor_axis_info[input_op_names[0]]

        if input_axis_info.transform_order is None:
            reordered_slice_ranges = slice_ranges
        else:
            reordered_slice_ranges = []
            transpose_order = input_axis_info.transform_order
            for idx in transpose_order:
                reordered_slice_ranges.append(slice_ranges[idx])

        # PyTorch does not support negative stride. So we need to perform te operation in reverse order
        # and then flip the result of that dimension.
        reversed_dimension = []
        for idx, slice_range in enumerate(reordered_slice_ranges):
            if slice_range[2] < 0:
                reversed_dimension.append(idx)
                slice_range[2] *= -1
                steps = ceil((slice_range[0] - slice_range[1]) / slice_range[2]) - 1

                slice_range[1] = slice_range[0]
                slice_range[0] = slice_range[1] - (steps * slice_range[2])
                slice_range[1] += 1

        execute_str = f'\t\t{output_tensor} = self.{get_op_name(self._op_name)}({string_input}, {reordered_slice_ranges})'
        if len(reversed_dimension) > 0:
            execute_str += f'\n\t\t{output_tensor} = {output_tensor}.flip({reversed_dimension})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_input, output_tensor)


class GatherElementsOpHandler(OpHandler):
    """ GatherElements Op Handler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=2)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        op_name = get_op_name(self._op_name)
        op_str = f'\t\tself.{op_name} = elementwise_ops.Gather()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """
        input_op_names = self.get_input_op_names()

        string_input = self._op_handler_state.op_to_tensors[input_op_names[0]]
        string_indices = self._op_handler_state.op_to_tensors[input_op_names[1]]
        string_input = self.transposed_input_names.get(string_input, string_input)
        string_indices = self.transposed_input_names.get(string_indices, string_indices)

        output_op_name = self.get_ops_output_names()[0]
        output_tensor = _create_variable_name(output_op_name)
        self._op_handler_state.op_to_tensors[output_op_name] = output_tensor

        dim = self._op.attrs_dict['axis']
        execute_str = f'\t\t{output_tensor} = self.{get_op_name(self._op_name)}({string_input},{dim},{string_indices}.to(dtype=torch.int64))'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_input, output_tensor)


class ChannelShuffleHandler(OpHandler):
    """ ChannelShuffle Op Handler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        groups = self._op.attrs_dict['num_groups']
        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.ChannelShuffle(groups={groups})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class PadHandler(OpHandler):
    """
    Op Handle for Pad Operation
    """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        if self._op.type == 'Pad':
            op_str = f'\t\tself.{op_name} = elementwise_ops.Pad()'
        else:
            raise AssertionError(f'Unrecognized op {self._op.type}')

        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def _get_padding_amount(self):
        pad_amount_unsigned = self._op.attrs_dict['pad_amount']
        pad_amount = pad_amount_unsigned.view(np.int32)
        input_tensor_name = self._op.inputs()[0].name()
        axis_info = self._op_handler_state.tensor_axis_info[input_tensor_name]
        input_permute_order = self._op_handler_state.op_axis_info[self._op_name].input_transform.get(input_tensor_name, None)
        effective_permute_order = combine_transpose_order(axis_info.transform_order, input_permute_order)
        pad_amount = update_shape_using_transform_order(effective_permute_order, pad_amount)
        return np.array(pad_amount)[::-1].reshape(-1)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        # For onnx only 3 values are supported by the converter now so skipped the conversion for mode value "1".
        pad_op_mode = {
            0: 'constant',
            2: 'reflect',
            3: 'replicate',
        }

        input_op_names = self.get_input_op_names()
        string_input = self._op_handler_state.op_to_tensors[input_op_names[0]]
        string_input = self.transposed_input_names.get(string_input, string_input)

        output_op_name = self.get_ops_output_names()[0]
        self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
        output_tensor = self._op_handler_state.op_to_tensors[output_op_name]

        pad_amount = self._get_padding_amount()
        scheme = pad_op_mode.get(self._op.attrs_dict['scheme'])
        if not scheme:
            raise AssertionError(f'Unrecognized pad schem: {self._op.scheme}')

        if scheme != 'constant':
            # Non-constant padding schemes are implemented for padding the last 3 dimensions of a 4D or 5D input tensor,
            # the last 2 dimensions of a 3D or 4D input tensor, or the last dimension of a 2D or 3D input tensor.
            input_len = len(self._op.inputs()[0].dims())
            allowed_len = 6 if input_len == 5 else (2 * (input_len - 1))
            pad_amount = pad_amount[:allowed_len]
        pad_amount = tuple(pad_amount)

        pad_value = self._op.attrs_dict.get('pad_constant_value', 0)

        if np.isinf(pad_value):
            pad_value = np.iinfo(np.int32).max

        elif np.isneginf(pad_value):
            pad_value = np.iinfo(np.int32).min

        execute_str = f'\t\t{output_tensor} = self.{get_op_name(self._op_name)}({string_input}, {pad_amount}, "{scheme}", {pad_value})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_input, output_tensor)


class RoIPoolingOpHandler(OpHandler):
    """ RoI Pooling Op Handler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=2)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        spatial_scale = self._op.attrs_dict['img_size_ratio'][0]
        output_dim = self._op.outputs()[0].dims()
        if self._op_handler_state.tensor_axis_info[self._op.outputs()[0].name()].transform_order is not None:
            output_size = tuple(output_dim[1:-1:])
        else:
            output_size = tuple(output_dim[2:])
        op_str = f'\t\tself.{get_op_name(self._op_name)} = torchvision.ops.RoIPool({output_size}, {spatial_scale})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class DepthToSpaceAndSpaceToDepthHandler(OpHandler):
    """ DepthToSpace and SpaceToDepth OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        op_name = get_op_name(self._op_name)
        block_size = self._op.attrs_dict['block_size']

        if self._op.type == 'DepthToSpace':
            mode = self._op.attrs_dict['mode']
            # CRD mode
            if mode == 1:
                if block_size.size > 1 and block_size[0] != block_size[1]:
                    op_str = f'\t\tself.{op_name} = elementwise_ops.DepthToSpaceCRDMode({list(block_size)})'
                else:
                    op_str = f'\t\tself.{op_name} = torch.nn.PixelShuffle({block_size[0]})'
            # DCR mode
            elif mode == 0:
                op_str = f'\t\tself.{op_name} = elementwise_ops.DepthToSpaceDCRMode({block_size[0]})'
                logger.warning('Mapping of DepthToSpace in DCR mode does not exist with PyTorch')
            else:
                raise AssertionError(f'Unrecognized mode {mode} for op {op_name}')

        elif self._op.type == 'SpaceToDepth':

            op_str = f'\t\tself.{op_name} = torch.nn.PixelUnshuffle({block_size[0]})'
        else:
            raise AssertionError(f'Unrecognized op {self._op.type}')
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)


class ScatterNDOpHandler(OpHandler):
    """ ScatterND OpHandler """

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state, num_inputs=3)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        #reduction_mapping from onnx to ir_graph = {1: "add", 2: "mul"}
        #Currently we are using ir_graph format for reduction value in prepared model file.
        op_name = get_op_name(self._op_name)
        if self._op.type == 'ScatterNd':
            reduction = self._op.attrs_dict['reduction']
            op_str = f'\t\tself.{op_name} = elementwise_ops.ScatterND(reduction={reduction})'
            logger.warning('Mapping of ScatterND does not exist with PyTorch')
        else:
            raise AssertionError(f'Unrecognized op {self._op.type}')
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)


class NonZeroOpHandler(OpHandler):
    """ NonZero Op Handler """

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """

        op_str = f'\t\tself.{get_op_name(self._op_name)} = emitter_ops.NonZero()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class TopKOpHandler(OpHandler):
    """ Top-K Op Handler"""

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """

        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.TopK()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))

    def generate_execute_code(self):
        """
        Generate code for executing the node as a Pytorch module in file self._op_handler_state.f.
        """

        input_node_names = self.get_input_op_names()
        string_input = self._op_handler_state.op_to_tensors[input_node_names[0]]
        string_input = self.transposed_input_names.get(string_input, string_input)

        # TopK has two outputs, values and indices
        values_output_node_name = self.get_ops_output_names()[0]
        values_tensor = _create_variable_name(values_output_node_name)
        self._op_handler_state.op_to_tensors[values_output_node_name] = values_tensor

        indices_output_node_name = self.get_ops_output_names()[1]
        indices_tensor = _create_variable_name(indices_output_node_name)
        self._op_handler_state.op_to_tensors[indices_output_node_name] = indices_tensor

        k = self._op.attrs_dict['k']
        execute_str = f'\t\t{values_tensor}, {indices_tensor} = self.{get_op_name(self._op_name)}({string_input}, k={k})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_input, f"{values_tensor}, {indices_tensor}")


class ShapeOpHandler(OpHandler):
    """ Shape Op Handler"""

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """

        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.Shape()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))

    def generate_execute_code(self):
        """
        Generate code for executing the node as a Pytorch module in file self._op_handler_state.f.
        """

        input_node_names = self.get_input_op_names()
        string_input = self._op_handler_state.node_to_tensors[input_node_names[0]]
        string_input = self.transposed_input_names.get(string_input, string_input)

        output_node_name = self._op.output_names[0]
        output_tensor = _create_variable_name(output_node_name)
        self._op_handler_state.node_to_tensors[output_node_name] = output_tensor

        start = self._op.list_params().get('start')  # Is None if 'start' is not defined
        end = self._op.list_params().get('end')  # Is None if 'end' is not defined

        execute_str = f'\t\t{output_tensor} = self.{get_op_name(self._op_name)}({string_input})[{start}:{end}]'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_input, output_tensor)


class TileOpHandler(OpHandler):
    """ Tile Op Handler"""

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """

        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.Tile()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))

    def generate_execute_code(self):
        """
        Generate code for executing the node as a Pytorch module in file self._op_handler_state.f.
        """

        input_node_names = self.get_input_op_names()
        string_input = self._op_handler_state.op_to_tensors[input_node_names[0]]
        string_input = self.transposed_input_names.get(string_input, string_input)

        output_node_name = self.get_ops_output_names()[0]
        output_tensor = _create_variable_name(output_node_name)
        self._op_handler_state.op_to_tensors[output_node_name] = output_tensor

        dims = self._op.attrs_dict["multiples"].tolist()

        input_tensor_name = self._op.inputs()[0].name()
        if self._op_handler_state.tensor_axis_info[input_tensor_name].transform_order is not None:
            dims = update_shape_using_transform_order(
                self._op_handler_state.tensor_axis_info[input_tensor_name].transform_order, dims)

        execute_str = f'\t\t{output_tensor} = self.{get_op_name(self._op_name)}({string_input}, dims={dims})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_input, output_tensor)


class LrnHandler(OpHandler):
    """ Lrn OpHandler """

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        # pylint: disable=too-many-locals
        attrs_dict = self._op.attrs_dict

        # In QNN, A square sum is computed over a region of size 2R + 1, where R is the radius.
        size = 2 * attrs_dict['radius'] + 1

        # In QNN definition, the alpha is descaled based on the given size, so it is scaled back in order to retrive the original alpha.
        alpha = attrs_dict['alpha'] * size
        beta = attrs_dict['beta']
        k = attrs_dict['bias']
        op_name = get_op_name(self._op_name)
        op_str = f'\t\tself.{op_name} = torch.nn.LocalResponseNorm(size={size},' \
                 f' alpha={alpha},' \
                 f' beta={beta},' \
                 f' k={k})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)


class RoIAlignOpHandler(OpHandler):
    """ RoI Align Op Handler """

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state, num_inputs=3)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        spatial_scale = 1.0 / self._op.attrs_dict["img_size_ratio"][0]
        sampling_ratio = self._op.attrs_dict["num_samples_x"]
        if sampling_ratio == 0:
            sampling_ratio = 1
        output_dim = self._op.outputs()[0].dims()
        if self._op_handler_state.tensor_axis_info[self._op.outputs()[0].name()].transform_order is not None:
            output_size = output_dim[1:-1:]
        else:
            output_size = output_dim[2:]
        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.RoiAlign({output_size}, {spatial_scale}, {sampling_ratio})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class LstmHandler(OpHandler):
    """ LSTM Op Handler"""

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state, num_inputs=3)

    def _transform_from_qnn_to_pytorch_format(self, tensor: np.ndarray) -> np.ndarray:
        """
        Transform weight tensor from QNN to PyTorch format
            QNN format: W_ii|W_if|W_io|W_ig
        PyTorch format: W_ii|W_if|W_ig|W_io
        Swap 3rd and 4th block (Each block has hidden_size length, total length = 4 * hidden_size)

        :param tensor: Weight tensor following QNN format
        :return: Weight tensor following PyTorch format
        """
        params = self._op.list_params()
        hidden_size = params['hidden_size']
        half_size = 2 * hidden_size
        return np.concatenate((tensor[:half_size, ...],
                               tensor[half_size + hidden_size:, ...],
                               tensor[half_size:half_size + hidden_size, ...]))

    # pylint: disable=too-many-locals
    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        op_name = get_op_name(self._op_name)
        op_type_to_torch_module_dict = {
            'RolledLstm': 'torch.nn.LSTM',
        }
        torch_module_name = op_type_to_torch_module_dict.get(self._op.type)
        if not torch_module_name:
            raise AssertionError(f'Unrecognized LSTM op {self._op.type}')

        all_nodes = self._op_handler_state.ir_graph.nodes_by_name
        weight_ih = all_nodes[self._op.input_names[3]].op.tensor
        weight_hh = all_nodes[self._op.input_names[4]].op.tensor
        bias_ih = all_nodes[self._op.input_names[5]].op.tensor

        _, input_size = weight_ih.shape
        _, hidden_size = weight_hh.shape
        has_bias = bias_ih.any()
        op_str = f'\t\tself.{op_name} = {torch_module_name}(input_size={input_size},' \
                 f' hidden_size={hidden_size},' \
                 f' bias={has_bias})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

        transformed_weight_ih = self._transform_from_qnn_to_pytorch_format(weight_ih)
        transformed_weight_hh = self._transform_from_qnn_to_pytorch_format(weight_hh)
        self._op_handler_state.state_dict[f'{op_name}.weight_ih_l0'] = transformed_weight_ih  # store params as np tensors
        self._op_handler_state.state_dict[f'{op_name}.weight_hh_l0'] = transformed_weight_hh  # store params as np tensors

        if has_bias:
            transformed_bias_ih = self._transform_from_qnn_to_pytorch_format(bias_ih)
            self._op_handler_state.state_dict[f'{op_name}.bias_ih_l0'] = transformed_bias_ih  # store params as np tensors
            self._op_handler_state.state_dict[f'{op_name}.bias_hh_l0'] = np.zeros(bias_ih.size, dtype=bias_ih.dtype)     # store params as np tensors

    def generate_execute_code(self):
        """
        Generate code for executing the node as a Pytorch module in file self._op_handler_state.f.
        """
        input_node_names = self.get_input_op_names()
        string_inputs = [self._op_handler_state.node_to_tensors[input_node_name]
                         for input_node_name in input_node_names]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]

        direction = self._op.list_params()['direction']
        if direction == LSTMDirection.FORWARD.value:
            input_tensor = string_inputs[0]

            # NOTE: Currently, it seems QNN LSTM has the initial hidden/cell state tensor on the contrary
            # For example,
            #   self.lstm_LSTM_forward(t_input, (t_initial_h_forward_split, t_initial_c_forward_split))
            #   self.lstm_LSTM_backward(t_input, (t_initial_h_backward_split, t_initial_c_backward_split))
            # To get correct output, it should be
            #   self.lstm_LSTM_forward(t_input, (t_initial_h_backward_split, t_initial_c_backward_split))
            #   self.lstm_LSTM_backward(t_input, (t_initial_h_forward_split, t_initial_c_forward_split))
            # Need to pass backward hidden/cell initial state tensor to forward LSTM and
            #   forward hidden/cell initial state tensor to backward LSTM
            # So this is a temporary workaround, it needs to be removed after QNN has been fixed.
            hidden_state_tensor = string_inputs[1].replace('forward', 'backward')
            cell_state_tensor = string_inputs[2].replace('forward', 'backward')
        elif direction == LSTMDirection.BACKWARD.value:
            # NOTE: Need to flip input tensor before passing backward LSTM
            input_tensor = f'torch.flip({string_inputs[0]}, [0])'
            hidden_state_tensor = string_inputs[1].replace('backward', 'forward')
            cell_state_tensor = string_inputs[2].replace('backward', 'forward')
        else:
            raise ValueError('direction should be forward or backward')
        string_inputs = f'{input_tensor}, ({hidden_state_tensor}, {cell_state_tensor})'

        output_tensors = []
        output_node_names = self._op.output_names
        for output_node_name in output_node_names:
            self._op_handler_state.node_to_tensors[output_node_name] = _create_variable_name(output_node_name)
            output_tensors.append(self._op_handler_state.node_to_tensors[output_node_name])
        string_outputs = f'{output_tensors[0]}, ({output_tensors[1]}, {output_tensors[2]})'

        execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs})'

        # NOTE: Need to flip output tensor before after backward LSTM
        if direction == LSTMDirection.BACKWARD.value:
            execute_str += f'\n\t\t{output_tensors[0]} = torch.flip({output_tensors[0]}, [0])'

        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)


class GatherNdOpHandler(OpHandler):
    """ GatherND OpHandler """

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state, num_inputs=2)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        batch_dim = self._op.attrs_dict['batch_dims']
        batch_dim = self._update_axis_using_axis_information(batch_dim)
        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.GatherNd(batch_dim={batch_dim})'
        logger.warning('Mapping of GatherNd does not exist with PyTorch')
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class BatchNormhandler(OpHandler):
    """ BatchNorm Op Handler"""

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        weights = self.get_weights()
        weight, bias = weights if len(weights) == 2 else (weights[0], None)
        num_features = weight.shape[0]

        op_name = get_op_name(self._op_name)
        input_rank = len(self._op.get_input_shapes()[0])

        batch_norms = {
            2: 'BatchNorm1d',
            3: 'BatchNorm1d',
            4: 'BatchNorm2d',
            5: 'BatchNorm3d'
        }
        bn_type = batch_norms.get(input_rank)
        if not bn_type:
            raise AssertionError(f'Unsupported input shape of (`{input_rank}D`) for BatchNorm')
        op_str = f'\t\tself.{op_name} = torch.nn.{bn_type}(num_features={num_features}, eps=0)'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)
        self._op_handler_state.state_dict[op_name + '.weight'] = weight     # store params as np tensors
        self._op_handler_state.prepared_param_name_map[op_name + '.weight'] = (self._op.get_input_names[1], None)
        self._op_handler_state.state_dict[op_name + '.bias'] = bias   # store params as np tensors
        self._op_handler_state.prepared_param_name_map[op_name + '.bias'] = (self._op.get_input_names[2], None)
        if not self._op_handler_state.ignore_encodings:
            weight_enc = extract_tensor_encoding(self._op.inputs()[1])
            if weight_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.weight'] = weight_enc
            bias_enc = extract_tensor_encoding(self._op.inputs()[2])
            if bias_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.bias'] = bias_enc

        # create tensors for running_mean and running_variance
        self._op_handler_state.state_dict[op_name + '.running_mean'] = np.zeros(weight.shape, dtype=weight.dtype)   # store params as np tensors
        self._op_handler_state.state_dict[op_name + '.running_var'] = np.ones(weight.shape, dtype=weight.dtype)     # store params as np tensors


class OneHotOpHandler(OpHandler):
    """ OneHot Op Handler """

    def __init__(self, node, op_handler_state):
        super().__init__(node, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        num_classes = self._op.attrs_dict['depth']
        on_value = self._op.attrs_dict['on_value']
        off_value = self._op.attrs_dict['off_value']
        axis = self._op.attrs_dict['axis']

        if axis == len(self._op.inputs()[0].dims()):
            op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.OneHot({num_classes}, {off_value}, {on_value})'
            self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))
        else:
            raise NotImplementedError(
                'OneHot not implemented when axis value does not represent the last dimension of input tensor')


class ScatterElementsOpHandler(OpHandler):
    """ ScatterElements Op Handler """

    def _get_string_name_or_tensor(self, tensor):
        """
        When scatter is used in a model, the arguments 'input', 'src' and 'index' can either be constants or tensors
        The appropriate value(tensor name or tensor value) need to be extracted depending on the type
        """
        if tensor.is_static_tensor():
            return np.array2string(tensor.get_data(), separator=", ").replace("\n", "")
        return self._op_handler_state.op_to_tensors[tensor.name()]

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        qnn_reduce_to_str = {
            0: None,
            1: 'add',
            2: 'multiply',
        }
        dim = self._op.attrs_dict['axis']

        reduction_val = self._op.attrs_dict["reduction"]
        reduce = qnn_reduce_to_str[reduction_val]

        kwargs_dict = dict()
        kwargs_dict['dim'] = dim
        kwargs_dict['reduce'] = reduce

        init_args = _create_init_args_from_kwargs_dict(kwargs_dict)
        op_str = f'\t\tself.{get_op_name(self._op_name)} = elementwise_ops.ScatterElements({init_args})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))

    def generate_execute_code(self):
        """
        Generate code for executing the node as a Pytorch module in file self._op_handler_state.f.
        """
        string_inputs = [self._get_string_name_or_tensor(tensor) for tensor in self._op.inputs()]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        init_args = ", ".join(string_inputs)
        output_node_name = self.get_ops_output_names()[0]
        output_tensor = _create_variable_name(output_node_name)
        self._op_handler_state.op_to_tensors[output_node_name] = output_tensor

        execute_str = f'\t\t{output_tensor} = self.{get_op_name(self._op_name)}({init_args})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, init_args, output_tensor)


class PackHandler(OpHandler):
    """ Pack OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=None)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        axis = self._op.attrs_dict['axis']
        axis = self._update_axis_using_axis_information(axis)

        op_str = f'\t\tself.{op_name} = emitter_ops.Stack(axis={axis})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)


class UnPackHandler(OpHandler):
    """ UnPack OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=1)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        axis = self._op.attrs_dict['axis']
        axis = self._update_axis_using_axis_information(axis)

        op_str = f'\t\tself.{op_name} = emitter_ops.UnBind(axis={axis})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)


class SpaceToBatchOpHandler(OpHandler):
    """
    Op Handler for TF/Keras based SpaceToBatch
    """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        block_size = self._op.attrs_dict['block_size'].tolist()
        pad_amount = self._op.attrs_dict['pad_amount'].tolist()
        op_str = f'\t\tself.{get_op_name(self._op_name)} = emitter_ops.SpaceToBatch({block_size}, {pad_amount})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class BatchToSpaceOpHandler(OpHandler):
    """
    Op Handler for TF/Keras based BatchToSpace
    """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        block_size = self._op.attrs_dict['block_size'].tolist()
        crops = self._op.attrs_dict['crops'].tolist()
        op_str = f'\t\tself.{get_op_name(self._op_name)} = emitter_ops.BatchToSpace({block_size}, {crops})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))


class MomentsHandler(OpHandler):
    """ Moments OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=1)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        op_name = get_op_name(self._op_name)
        axes = self._op.attrs_dict['axes']
        axes = [self._update_axis_using_axis_information(axis) for axis in axes]
        keep_dims = self._op.attrs_dict['keep_dims']

        op_str = f'\t\tself.{op_name} = emitter_ops.Moments({axes}, {keep_dims})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)


class CropAndResizeHandler(OpHandler):
    """
    Op Handler for CropAndResize
    """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=3)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        op_name = get_op_name(self._op_name)
        resize_dims = self._op.attrs_dict['resize_dims'].tolist()
        interpolation_mode = self._op.attrs_dict['interpolation_mode']
        extrapolation_value = self._op.attrs_dict['extrapolation_value']

        op_str = f'\t\tself.{op_name} = emitter_ops.CropAndResize({resize_dims}, {interpolation_mode}, {extrapolation_value})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)


class GroupNormHandler(OpHandler):
    """
    OpHandler for GroupNorm
    """
    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=1)

    def generate_create_code(self):
        """
        Generate code for creating the node as a Pytorch module in file self._op_handler_state.f
        """
        op_name = get_op_name(self._op_name)
        weights = self.get_weights()

        op_attrs = self._op.attrs_dict
        kwargs_dict = dict()
        kwargs_dict["eps"] = op_attrs["epsilon"]
        kwargs_dict["num_groups"] = op_attrs["group"]
        kwargs_dict["num_channels"] = self._op.inputs()[0].dims()[-1]
        init_args = _create_init_args_from_kwargs_dict(kwargs_dict)

        op_str = f'\t\tself.{op_name} = torch.nn.GroupNorm({init_args})'

        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)
        if len(weights) > 0:
            self._op_handler_state.state_dict[op_name + '.weight'] = weights[0]     # store params as np tensors
            self._op_handler_state.prepared_param_name_map[op_name + '.weight'] = (self._op.get_input_names[1], None)
            if not self._op_handler_state.ignore_encodings:
                weight_enc = extract_tensor_encoding(self._op.inputs()[1])
                if weight_enc is not None:
                    self._op_handler_state.encodings['param_encodings'][op_name + '.weight'] = weight_enc
            if len(weights) > 1:
                self._op_handler_state.state_dict[op_name + '.bias'] = weights[1]      # store params as np tensors
                self._op_handler_state.prepared_param_name_map[op_name + '.bias'] = (self._op.get_input_names[2], None)
                if not self._op_handler_state.ignore_encodings:
                    bias_enc = extract_tensor_encoding(self._op.inputs()[2])
                    if bias_enc is not None:
                        self._op_handler_state.encodings['param_encodings'][op_name + '.bias'] = bias_enc

    def get_weights(self):
        """ Method to fetch the static tensor data for an op """
        weights = []
        for input_tensor in self._op.inputs():
            if input_tensor.is_static_tensor():
                weights.append(input_tensor.get_data())
        return weights


class MultiClassNmsOpHandler(OpHandler):
    """
    OpHandler for Multiclass NMS
    """
    def __init__(self, op, op_handler_state):
        num_inputs = len(op.inputs())
        super().__init__(op, op_handler_state, num_inputs=num_inputs)

    def generate_create_code(self):
        op_name = get_op_name(self._op_name)
        iou_threshold = self._op.attrs_dict["iou_threshold"]
        score_threshold = self._op.attrs_dict["score_threshold"]
        # TODO: Currently QNN converter does not convert TF NonMaxSuppressionV5 (only op with soft_nms_sigma)
        if hasattr(self._op.attrs_dict, "soft_nms_sigma") and self._op.attrs_dict["soft_nms_sigma"] != 0.0:
            raise NotImplementedError("Model Preparer doesn't support soft_nms_sigma, as converter "
                                      "is yet to support non-zero soft_nms_sigma")
        max_selected_boxes_per_batch = self._op.outputs()[0].dims()[1]
        op_str = (f'\t\tself.{op_name} = emitter_ops.MultiClassNms({iou_threshold}, {score_threshold}, 0.0, '
                  f'{max_selected_boxes_per_batch})')
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in self.get_input_op_names()]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        string_inputs = ', '.join(string_inputs)

        output_tensors = []

        for idx, output_op_name in enumerate(self.get_ops_output_names()):
            if idx != 3:
                self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
                output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs})'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str)

class GridSamplehandler(OpHandler):
    """
    OpHandler for GridSample
    """
    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=2)

    # pylint: disable=too-many-locals
    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        op_name = get_op_name(self._op_name)
        module_type = 'elementwise_ops.GridSample'

        op_str = f'\t\tself.{op_name} = {module_type}()'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """
        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in self.get_input_op_names()]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        string_inputs = ', '.join(string_inputs)

        output_tensors = []

        for output_op_name in self.get_ops_output_names():
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        kwargs_dict = dict()

        kwargs_dict['align_corners'] = self._op.attrs_dict.get("align_corners", None)
        kwargs_dict['mode'] = self._op.attrs_dict.get("mode", 0)
        kwargs_dict['padding_mode'] = self._op.attrs_dict.get("padding_mode", 0)

        mode_mapper = {0: "bilinear", 1: "nearest", 2: "bicubic"}
        kwargs_dict['mode'] = mode_mapper[kwargs_dict['mode']]

        padding_mode_mapper = {0: "zeros", 1: "border", 2: "reflection"}
        kwargs_dict['padding_mode'] = padding_mode_mapper[kwargs_dict['padding_mode']]

        execute_str = f'\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs}, ' \
                      f'align_corners={kwargs_dict["align_corners"]}, ' \
                      f'mode="{kwargs_dict["mode"]}", ' \
                      f'padding_mode="{kwargs_dict["padding_mode"]}")'
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)

    def _generate_axis_information(self):
        """
        Updates the axis information for GridSample op.
        """
        super()._generate_axis_information()

        # Update the axis information for input index 1
        tensor_name = self._op.inputs()[1].name()
        tensor_axis_info = self._op_handler_state.tensor_axis_info[tensor_name]

        # Expected relative transpose order for the input tensor at index 1 is None for GridSample Op
        required_input_transpose = get_transpose_order(tensor_axis_info.transform_order, None)

        # Update the info in the OpAxisInfo
        self._op_handler_state.op_axis_info[self._op_name].input_transform[tensor_name] = required_input_transpose


class RmsNormOphandler(OpHandler):
    """
    Op Handler for RmsNorm
    """
    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=1)

    def generate_create_code(self):
        op_name = get_op_name(self._op_name)
        epsilon = self._op.attrs_dict['epsilon']
        axes = list(self._op.attrs_dict['axes'])
        inputs = self._op.inputs()
        input_shape = inputs[0].dims()
        op_str = (f'\t\tself.{op_name} = elementwise_ops.RmsNorm({input_shape}, {axes}, {epsilon})')
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str)
        self._op_handler_state.state_dict[op_name + '.weight'] = inputs[1].get_data()
        self._op_handler_state.prepared_param_name_map[op_name + '.weight'] = (self._op.get_input_names[1], None)
        self._op_handler_state.state_dict[op_name + '.bias'] = inputs[2].get_data()
        self._op_handler_state.prepared_param_name_map[op_name + '.bias'] = (self._op.get_input_names[2], None)
        if not self._op_handler_state.ignore_encodings:
            weight_enc = extract_tensor_encoding(inputs[1])
            if weight_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.weight'] = weight_enc
            bias_enc = extract_tensor_encoding(inputs[2])
            if bias_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.bias'] = bias_enc


class CreateSparseOpHandler(OpHandler):
    '''
    CreateSparse op handler
    '''

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=2)

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        input_op_names = self.get_input_op_names()
        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in input_op_names]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        string_inputs = ", ".join(string_inputs)

        output_tensors = []
        output_op_names = self.get_ops_output_names()
        for output_op_name in output_op_names:
            output_tensor = _create_variable_name(output_op_name)
            self._op_handler_state.op_to_tensors[output_op_name] = output_tensor
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ", ".join(output_tensors)

        execute_str = f"\t\t{string_outputs} = {string_inputs}"
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)

class CustomScatterDenseOpHandler(OpHandler):
    '''
    ScatterDense op handler
    '''

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=1)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        torch_module_name = "elementwise_ops.ScatterDense()"
        op_str = f'\t\tself.{get_op_name(self._op_name)} = {torch_module_name}'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, get_op_name(self._op_name))

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """

        input_op_names = self.get_input_op_names()
        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in input_op_names]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]
        string_inputs = ", ".join(string_inputs)

        output_tensors = []
        output_op_names = self.get_ops_output_names()
        for output_op_name in output_op_names:
            output_tensor = _create_variable_name(output_op_name)
            self._op_handler_state.op_to_tensors[output_op_name] = output_tensor
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ", ".join(output_tensors)

        execute_str = f"\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs})"
        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)

class SpConvHandler(OpHandler):
    """ Sparse Conv OpHandler """

    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=1)
        self._padding_info = {}
        if 'pad_amount' in op.attrs_dict:
            _process_padding(self)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """

        # pylint: disable=too-many-locals
        op_name = get_op_name(self._op_name)
        op_type_to_torch_module_dict = {
            'Conv3d': 'elementwise_ops.CustomSparseConv3DLayer',
        }
        custom_torch_module_name = op_type_to_torch_module_dict.get(self._op.type)
        op_attrs = self._op.attrs_dict
        kwargs_dict = dict()
        kwargs_dict["stride"] = tuple(op_attrs["stride"])
        kwargs_dict["padding"] = _process_padding(self)
        kwargs_dict["dilation"] = tuple(op_attrs.get("dilation", [1] * len(kwargs_dict["padding"])))
        kwargs_dict["groups"] = op_attrs["group"]

        weights = self.get_weights()
        weight, bias = weights if len(weights) == 2 else (weights[0], None)
        weight_tensor_name = self._op.inputs()[1].name()

        transform_order = list(
            self._op_handler_state.op_axis_info[self._op_name].input_transform[weight_tensor_name])
        if 'Transpose' in custom_torch_module_name:
            transform_order[1], transform_order[0] = transform_order[0], transform_order[1]
        weight = weight.transpose(transform_order) # O D H W I format

        in_channels = weight.shape[4]
        out_channels = weight.shape[0]

        kwargs_dict["in_channels"] = in_channels
        kwargs_dict["out_channels"] = out_channels
        kwargs_dict["kernel_size"] = weight.shape[1:-1]
        kwargs_dict["bias"] = np.count_nonzero(bias) != 0  # bias comes back as all zeros if no bias

        init_args = _create_init_args_from_kwargs_dict(kwargs_dict)
        op_str = f'\t\tself.{op_name} = {custom_torch_module_name}({init_args})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

        self._op_handler_state.state_dict[op_name + '.sp_conv_3d.weight'] = weight
        self._op_handler_state.prepared_param_name_map[op_name + '.sp_conv_3d.weight'] = (self._op.get_input_names[1],
                                                                                          transform_order)

        self._op_handler_state.state_dict[op_name + '.sp_conv_3d.bias'] = bias
        self._op_handler_state.prepared_param_name_map[op_name + '.sp_conv_3d.bias'] = (self._op.get_input_names[2], None)

        if not self._op_handler_state.ignore_encodings:
            weight_enc = extract_tensor_encoding(self._op.inputs()[1])
            if weight_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.sp_conv_3d.weight'] = weight_enc
            bias_enc = extract_tensor_encoding(self._op.inputs()[2])
            if bias_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.bias'] = bias_enc

    def generate_execute_code(self):
        """
        Generate code for executing the node as a PyTorch module in file self._op_handler_state.f.
        """
        temp_input_tensor = _add_input_pad_if_requried(self)

        string_inputs = [self._op_handler_state.op_to_tensors[input_op_name]
                         for input_op_name in self.get_input_op_names()]
        string_inputs = [self.transposed_input_names.get(inputs, inputs)
                         for inputs in string_inputs]

        string_inputs[0] = string_inputs[0] if not temp_input_tensor else temp_input_tensor
        string_inputs = ', '.join(string_inputs)

        output_tensors = []
        for output_op_name in self.get_ops_output_names():
            self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
            output_tensors.append(self._op_handler_state.op_to_tensors[output_op_name])
        string_outputs = ', '.join(output_tensors)

        if hasattr(self._op.inputs()[0].get_producer(), "type") and \
                self._op.inputs()[0].get_producer().type == "CreateSparse":
            # Input coming from CreateSparse op
            execute_str = f"\t\t{string_outputs} = self.{get_op_name(self._op_name)}(*{string_inputs})"
        else:
            # Input coming from spconv op
            execute_str = f"\t\t{string_outputs} = self.{get_op_name(self._op_name)}({string_inputs})"

        padding_type = self._padding_info.get('padding_type', None)
        if padding_type == PaddingType.ASYMMETRIC_CONV_TRANSPOSE:

            padding_diff = self._padding_info.get('padding_diff', None)
            diff_dim_w = None if padding_diff[-1] == 0 else padding_diff[-1]
            # NCF/W
            out_tensor_slice_str = f'\t\t{string_outputs} = {string_outputs}[:, :, :{diff_dim_w}]'
            execute_str = execute_str + '\n' + out_tensor_slice_str

        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, string_inputs, string_outputs)

        # Free output of additional padding layer after it is used.
        if temp_input_tensor:
            free_str = f'\t\t{temp_input_tensor} = None'
            self._op_handler_state.model_def_mgr.add_execute_code(free_str, self._op)

    def _generate_axis_information(self):
        """
        Generates the axis information for the current op based on
        the IR axis format and the desired axis format for torch.
        """
        # Get the op_type
        # Actions will be based on the op type
        op_type = "SpConv3d"

        input_transpose = {}
        output_transpose = {}

        # Iterate through the input activation tensor and check if some processing is required
        for ip_idx, ip_tensor in enumerate(self.get_input_op_names()):
            if hasattr(self._op.inputs()[ip_idx].get_producer(), "type") and \
                    self._op.inputs()[ip_idx].get_producer().type == "CreateSparse":
                transpose_require = None
            else:
                tensor_axis_info = self._op_handler_state.tensor_axis_info[ip_tensor]
                transpose_require = get_required_transpose(tensor_axis_info, op_type)
            if transpose_require is not None:
                input_transpose[ip_tensor] = transpose_require

        # Process weight transpose
        weight_index = 1
        weight_tensor = self._op.inputs()[weight_index]
        input_transpose[weight_tensor.name()] = (4, 0, 1, 2, 3) # O D H W I

        # Instantiate the TensorAxisInfo for output
        input_tensor_name = self._op.inputs()[0].name()
        input_tensor_axis_info = self._op_handler_state.tensor_axis_info[input_tensor_name]

        # Iterate through the output tensor
        for op_tensor in self._op.outputs():
            output_transpose_order = get_output_tranapose_order(op_type, op_tensor, input_tensor_axis_info)
            self._op_handler_state.tensor_axis_info[op_tensor.name()] = TensorAxisInfo(op_tensor.dims(),
                                                                                       output_transpose_order)

            transpose_required = self.get_output_tranpose_order_for_model_outputs(op_tensor, output_transpose_order)
            if transpose_required is not None:
                output_transpose[op_tensor.name()] = transpose_required

        self._op_handler_state.op_axis_info[self._op_name] = OpAxisInfo(input_transpose, output_transpose)


class CustomOpHandler(OpHandler):
    """Custom Op Handler"""

    def __init__(self, op, op_handler_state):
        num_inputs = len([x for x in op.inputs() if isinstance(x, IrTensor)])
        super().__init__(op, op_handler_state, num_inputs)

    def generate_create_code(self):
        """
        Generate code for creating the op as a Pytorch module in file self._op_handler_state.f
        """
        if self._op.type not in self._op_handler_state.custom_op_info.op_type_to_module:
            raise RuntimeError('Current IrOp type %s is not recognized in converter parameters', self._op.type)

        module_class = self._op_handler_state.custom_op_info.op_type_to_module[self._op.type]
        op_name = get_op_name(self._op_name)
        op_str = f'\t\tself.{op_name} = {module_class}({_create_init_args_from_kwargs_dict(self._op.attrs_dict)})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, self._op, op_name)

    def _generate_axis_information(self):
        """
        Generates the axis information for the CustomOps.
        All the custom ops are treated to behave as channel_last in IrOp and channel_first in Torch implementation.
        """
        op_type = 'CUSTOM_OP' # Special Entry to be used for axis tracking related to custom ops
        input_transpose = {}
        output_transpose = {}

        # Iterate through the input activation tensor and check if some processing is required
        for idx, ip_tensor in enumerate(self.get_input_op_names()):
            tensor_axis_info = self._op_handler_state.tensor_axis_info[ip_tensor]
            transpose_require = get_required_transpose(tensor_axis_info, op_type)
            if transpose_require is not None:
                input_transpose[ip_tensor] = transpose_require

        # Instantiate the TensorAxisInfo for output
        input_tensor_name = self._op.inputs()[0].name()
        input_tensor_axis_info = self._op_handler_state.tensor_axis_info[input_tensor_name]

        for op_tensor in self._op.outputs():
            output_transpose_order = get_output_tranapose_order(op_type, op_tensor, input_tensor_axis_info)
            self._op_handler_state.tensor_axis_info[op_tensor.name()] = TensorAxisInfo(op_tensor.dims(),
                                                                                       output_transpose_order)

            transpose_required = self.get_output_tranpose_order_for_model_outputs(op_tensor, output_transpose_order)
            if transpose_required is not None:
                output_transpose[op_tensor.name()] = transpose_required

        self._op_handler_state.op_axis_info[self._op_name] = OpAxisInfo(input_transpose, output_transpose)


# Dictionary mapping op types to OpHandlers
ir_to_handler_dict = {
    'Conv1d': ConvHandler,
    'Conv2d': ConvHandler,
    'Conv3d': ConvHandler,
    'SpConv3d': SpConvHandler,
    'TransposeConv1d': ConvHandler,
    'TransposeConv2d': ConvHandler,
    'TransposeConv3d': ConvHandler,
    'DepthWiseConv2d': DepthWiseConv2dHandler,
    'Neuron': NeuronHandler, # Supported only until QNN 2.20
    'ElementWiseNeuron': NeuronHandler, # Supported from QNN >= 2.21
    'Eltwise_Binary': ElementwiseBinaryOpHandler,
    'Eltwise_Ternary': ElementwiseTernaryOpHandler,
    'Reduce': ReduceOpHandler,
    'Split': SplitOpHandler,
    'Concat': ConcatHandler,
    'Pool': PoolHandler,
    'Pool3d': PoolHandler,
    'Reshape': ReshapeHandler,
    'Resize': ResizeHandler,
    'ResizeNearestNeighbor': ResizeHandler,
    'ResizeBilinear': ResizeHandler,
    'Transpose': ReshapeHandler,
    'Expand': ExpandHandler,
    'ElementWiseMultiply': ExpandHandler,  # NOTE: IrGraph has ElementWiseMultiply for Expand Op?
    'ElementWiseAnd': ExpandHandler,
    'FullyConnected': FullyConnectedHandler,
    'input': IgnoreHandler,
    'Softmax': SoftmaxHandler,
    'LayerNorm': LayerNormHandler,
    'LogSoftmax': SoftmaxHandler,
    'Prelu': PreluHandler,
    'Gather': GatherHandler,
    'InstanceNorm': InstanceNormHandler,
    'Gelu': GeluHandler, # TODO: Remove this once Gelu is resolved under Elementwise Binary
    'MatMul': MatMulHandler,
    'Cast': CastHandler,
    'CumulativeSum': CumulativeOpHandler,
    'Arg': ArgOpHandler,
    'Erf': ErfOpHandler,
    'Eltwise_Unary': ElementwiseUnaryOpHandler,
    'StridedSlice': StridedSliceOpHandler,
    'L2Norm': L2NormHandler,
    'GatherElements': GatherElementsOpHandler,
    'ChannelShuffle': ChannelShuffleHandler,
    'Pad': PadHandler,
    'RoiPooling': RoIPoolingOpHandler,
    'DepthToSpace': DepthToSpaceAndSpaceToDepthHandler,
    'SpaceToDepth': DepthToSpaceAndSpaceToDepthHandler,
    'NonZero': NonZeroOpHandler,
    'TopK': TopKOpHandler,
    'Shape': ShapeOpHandler,
    'Tile': TileOpHandler,
    'Lrn': LrnHandler,
    'RolledLstm': LstmHandler,
    'ScatterNd': ScatterNDOpHandler,
    'RoiAlign': RoIAlignOpHandler,
    'NonMaxSuppression': NonMaxSuppressionHandler,
    'GatherNd': GatherNdOpHandler,
    'Batchnorm': BatchNormhandler,
    'OneHot': OneHotOpHandler,
    'ScatterElements': ScatterElementsOpHandler,
    'Pack': PackHandler,
    'UnPack': UnPackHandler,
    'SpaceToBatch': SpaceToBatchOpHandler,
    'BatchToSpace': BatchToSpaceOpHandler,
    'Moments': MomentsHandler,
    'CropAndResize': CropAndResizeHandler,
    'GroupNorm': GroupNormHandler,
    'MultiClassNms': MultiClassNmsOpHandler,
    'GridSample': GridSamplehandler,
    'RmsNorm': RmsNormOphandler,
    'CreateSparse': CreateSparseOpHandler,
    'SparseToDense': CustomScatterDenseOpHandler,
    # 'gru': GruHandler,
}