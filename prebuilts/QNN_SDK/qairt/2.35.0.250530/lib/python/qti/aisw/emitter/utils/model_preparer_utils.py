# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" Utility functions for emitter model """
import collections
import logging
import re
import json
import os
from typing import  Tuple, Dict, List, Optional, Any, Callable
import numpy as np
import qti.aisw.emitter.ir_graph_op_handler as op_handler
import qti.aisw.emitter.op_handler_factory as op_handler_factory
from qti.aisw.emitter.utils.config import is_custom_ir_op
# pylint: disable=import-error
from qti.aisw.converters.common import ir_graph as ir_graph_lib

IrOp = ir_graph_lib.IrOp
IrGraph = ir_graph_lib.IrGraph

_logger = logging.getLogger('TorchEmitter')

class PreparedModelInfo:
    """
    Holds information of the prepared model to be pickled for later use.
    """

    def __init__(self,
                 param_name_mapping: Dict[str, str],
                 additional_pad_info: Optional[Dict[str, str]] = None,
                 additional_transpose_info: Optional[Dict[str, str]] = None,
                 ir_op_name_list: Optional[List[str]] = None):
        self.param_name_mapping = param_name_mapping
        if additional_pad_info:
            self.additional_pad_to_node_mapping = {
                v: k for k, v in additional_pad_info.items()
            }
        else:
            self.additional_pad_to_node_mapping = {}

        if additional_transpose_info:
            self.additional_transpose_info = additional_transpose_info
        else:
            self.additional_transpose_info = {}

        if ir_op_name_list:
            self.ir_op_name_list = ir_op_name_list
        else:
            self.ir_op_name_list = []

    def get_metadata_dump(self):
        '''
        :return: Emitter model info as a dict.
        '''
        prepared_model_info_dict = {
            'additional_pad_to_node_mapping': self.additional_pad_to_node_mapping,
            'additional_transpose_info': self.additional_transpose_info,
            'param_name_mapping': self.param_name_mapping,
            'ir_op_name_list': self.ir_op_name_list
        }
        metadata = {"metadata": json.dumps(prepared_model_info_dict)}
        return metadata

    @classmethod
    def load_from_metadata(cls, data: str) -> 'PreparedModelInfo':
        """
        Takes a dump generated from the `get_metadata_dump` method and creates a PreparedModelInfo object

        :param data: json dump of the PreparedModelInfo object
        :return: PreparedModelInfo Object containing info present in the metadata
        """
        data = data.get('metadata', '')
        json_data = json.loads(data)
        json_data['additional_pad_to_node_mapping'] = {
            v: k for k, v in json_data['additional_pad_to_node_mapping'].items()
        }
        return PreparedModelInfo(json_data.get('param_name_mapping', {}),
                                 json_data.get('additional_pad_to_node_mapping', None),
                                 json_data.get('additional_transpose_info', None),
                                 json_data.get('ir_op_name_list', None))

def check_input_for_dict(inp: Any, is_top_level=False) -> bool:
    """
    Returns True if inp contains a dictionary, False otherwise. Recursively checks within nested elements.
    :param inp: Input to check for dictionary
    :param is_top_level: Usage of dictionary for top level input kwargs is supported. If set to True, inp itself being
        a dictionary will not cause this function to return False.
    :return: True if inp contains a dictionary, False otherwise.
    """
    has_dict = False
    if isinstance(inp, dict):
        if is_top_level:
            for element in inp.values():
                has_dict = has_dict or check_input_for_dict(element)
            return has_dict
        return True

    if isinstance(inp, (list, tuple)):
        for element in inp:
            has_dict = has_dict or check_input_for_dict(element)

    return has_dict


def count_inputs(inputs: Any, flatten_inputs: bool, is_valid_input: Callable = None) -> int:
    """
    Returns the number of elements within inputs. If flatten_inputs is False, returns only the number of elements in the
    top level without recursively counting inner elements.
    :param inputs: Input to count number of elements
    :param flatten_inputs: If True, this will recursively parse nested elements to find the total number of items within
        inp. If False, only the number of elements in the top level will be counted.
    :param is_valid_input: evaluate to True if is valid type of input.
    :return: Number of elements within inp
    """
    if is_valid_input is None:
        is_valid_input = lambda x: True
    if not isinstance(inputs, (list, tuple, dict)):
        inputs = [inputs]

    if flatten_inputs:
        return count_flattened_elements(inputs, is_valid_input)
    return len(inputs)


def count_flattened_elements(inputs: Any, is_valid_input: Callable = None) -> int:
    """
    Counts the number of elements within inputs, recursively parsing nested elements to find the total count.
    :param inputs: Inputs to count number of elements of
    :param is_valid_input: evaluate to True if is valid type of input.
    :return: Total number of elements within inputs
    """
    count = 0
    if isinstance(inputs, (list, tuple)):
        for inp in inputs:
            count += count_flattened_elements(inp, is_valid_input)
        return count
    if isinstance(inputs, dict):
        for value in inputs.values():
            count += count_flattened_elements(value, is_valid_input)
        return count
    return 1 if is_valid_input is None or is_valid_input(inputs) else 0


def write_model_initialization(model_name: str, op_handler_state: op_handler_factory.OpHandlerState):
    """
    Write code for initializing the model. Also saves weight values from the ir_graph in the op_handler_state
    state_dict.

    :param model_name: Name of the model
    :param op_handler_state: Object for holding state
    """
    # Define converted model class and class variables
    model_def_mgr = op_handler_state.model_def_mgr
    model_def_mgr.add_module_definition_code(f'class {model_name}(torch.nn.Module):')
    model_def_mgr.add_module_definition_code('\tdef __init__(self):')
    model_def_mgr.add_module_definition_code(f'\t\tsuper({model_name}, self).__init__()')

    # Add nn.ModuleList initialization if present
    module_list_init_lines = []
    for module_list_path, num_of_submodules in op_handler_state.module_list_to_submodule_count.items():
        init_line = f"\t\tself.{module_list_path} = torch.nn.ModuleList([torch.nn.Module() for _ in range({num_of_submodules})])"
        model_def_mgr.add_complex_module_create_code(init_line)

    # Add submodules init lines if present
    # To avoid `no attribute` error when initializing modules, we need to sort initialization correctly
    # It can be achieved by alphabetical sort, for example
    # `self.transformer.h = torch.nn.ModuleList([torch.nn.Module() for _ in range(n)])` should be initialized
    # before initializing `self.transformer.h[0].ln_1 = torch.nn.Module()`
    # But, we can review and improve this kind of initialization in the future
    for init_line in sorted(op_handler_state.created_submodues_init + module_list_init_lines):
        model_def_mgr.add_complex_module_create_code(init_line)

    op_list = op_handler_state.ir_graph.get_ops()

    for op in op_list:
        op_type = op.type
        is_custom_op = is_custom_ir_op(op)
        # Detect whether it is a sparseConv3D module
        if not is_custom_op and op_type == "Conv3d" and op.attrs_dict.get('reuse_sparse_indicies', False):
            op_type = "SpConv3d"
        if (
            (not is_custom_op and op_type in op_handler.ir_to_handler_dict.keys())
            or (is_custom_op and  op_type in op_handler_state.custom_op_info.op_type_to_module)
        ):
            _handle_potential_constant_tensors(op, op_handler_state)
            if is_custom_op:
                op_handler_obj = op_handler.CustomOpHandler(op, op_handler_state)
            else:
                op_handler_obj = op_handler.ir_to_handler_dict.get(op_type)(op, op_handler_state)
            op_handler_obj.generate_create_code()
            op_handler_obj.generate_transpose_for_model_input()
            if not op_handler_state.ignore_encodings:
                op_handler_obj.save_activation_encoding()

        else:
            error_msg = 'Encountered unknown op type ' + op_type +'. Unable to proceed with model preparation.'
            _logger.error(error_msg)
            raise RuntimeError(error_msg)


# pylint: disable=too-many-locals, too-many-branches
def _handle_potential_constant_tensors(op: IrOp, op_handler_state: op_handler_factory.OpHandlerState):
    """
    Function for adding potential constant tensors found when initializing a given op.
    :param op: Op about to be initialized
    :param op_handler_state: Dict of the state of the build
    """
    potential_constant_tensors = [tensor for tensor in op.inputs() if tensor.is_static_tensor()]
    for tensor in potential_constant_tensors:
        tensor_name = tensor.name()
        if tensor_name not in op_handler_state.parameter_constant_names or len(tensor.get_consumers()) > 1:

            # Skip if the tensor is already computed
            if tensor.name() in op_handler_state.op_to_tensors:
                continue

            if op_handler.KEEP_ORIGINAL_MODEL_STRUCTURE and op_handler_state.is_parameter(tensor):
                # tensor_name is mainly used when generating initialization and execution statement
                # If tensor_name including pattern dot with digits such as transformer.h.0.ln_1.Constant_output_0 which implies ModuleList
                # tensor_name should be converted with using brackets, meaning that transformer.h[0].ln_1.Constant_output_0
                tensor_name = re.sub(r"\.(\d+)", r"[\1]", tensor_name)
                if op_handler.is_one_to_one_op(tensor_name, op_handler_state.child_module_counter):
                    tensor_name, _, _ = tensor_name.rpartition('/')

            tensor_name = op_handler.get_op_name(tensor_name)
            if tensor_name.isnumeric():
                tensor_name = f'const_{tensor_name}'

            transform_order = op_handler_state.tensor_axis_info[tensor.name()].transform_order
            const_data = tensor.get_data().transpose(transform_order) if transform_order else tensor.get_data()

            # Fetch the numpy datatype of the tensor
            dtype = op_handler.qnn_numpy_type_to_actual_numpy_dtype(tensor)
            dtype = dtype.lower().replace('_', '')
            if dtype == 'uint32':
                # torch does not support np.unit32, so need to convert it into np.int64 first
                _logger.warning('Updating datatype of tensor "%s" from np.uint32 to np.int64 as '
                                'PyTorch does not support conversion from np.uint32 to tensor.', {tensor.name()})
                const_data = const_data.astype(np.int64)
                dtype = 'int64'

            # Fetch the corresponding torch datatype
            dtype = _get_torch_dtype_from_numpy_dtype(dtype)
            shape = tensor.dims()
            if transform_order:
                shape = np.array(list(transform_order)).choose(shape).tolist()
            data = f'torch.zeros({shape}, dtype={dtype})'
            if op_handler_state.is_parameter(tensor):
                requires_grad = op_handler_state.is_trainable_parameter(tensor)
                op_str = f'\t\tself.{tensor_name} = torch.nn.Parameter({data}, requires_grad={requires_grad})'
            else:
                # A buffer name should not include dots, so we should find submodules to register buffer
                modules_to_register, _, buffer_name = tensor_name.rpartition('.')
                if modules_to_register == '':
                    modules_to_register = 'self'
                else:
                    modules_to_register = f'self.{modules_to_register}'
                op_str = f'\t\t{modules_to_register}.register_buffer({repr(buffer_name)}, {data})'
            op_handler_state.prepared_param_name_map[tensor_name] = (tensor.name(), transform_order)
            if not op_handler_state.ignore_encodings:
                encoding = op_handler.extract_tensor_encoding(tensor)
                if encoding is not None:
                    op_handler_state.encodings['param_encodings'][tensor_name] = encoding

            op_handler_state.state_dict[tensor_name] = const_data       # store constants as np tensors
            op_handler_state.op_to_tensors[tensor.name()] = f'self.{tensor_name}'
            op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str, op)


def _get_torch_dtype_from_numpy_dtype(dtype: str) -> str:
    """
    Convert numpy dtype into pytorch dtype
    :param dtype: numpy dtype to convert
    :return: equivalent pytorch dtype
    """
    dtype = dtype.lower().replace('_', '')
    if dtype == 'bool':
        return 'bool'
    if dtype in ('float16', 'float32', 'float64'):
        return f'torch.{dtype}'
    if dtype in ('uint8', 'int8', 'int16', 'int32', 'int64'):
        return f'torch.{dtype}'
    raise AssertionError(f'Unrecognized constant dtype {dtype}')


# pylint: disable=too-many-locals, too-many-branches
def write_model_forward_pass(op_handler_state: op_handler_factory.OpHandlerState,
                             model_input,
                             model_output,
                             ir_graph_input_names,
                             order_inputs: bool = False,
                             order_outputs: bool = False):
    """
    Write code for the model forward pass.

    :param op_handler_state: Object for holding state
    :param model_input: Input of the original model
    :param model_output: Output of the original model
    :param ir_graph_input_names: List of the ir_graph input names in order.
    :param order_inputs: Ordering of inputs in the same way original model inputs were returned
    :param order_outputs: Ordering of outputs in the same way original model outputs were returned
    """
    input_names = ['self']
    input_tensors_name = [input_tensor.name() for input_tensor in
                            op_handler_state.ir_graph.get_input_tensors_to_graph()]
    if ir_graph_input_names is None or set(ir_graph_input_names) != set(input_tensors_name):
        ir_graph_input_names = input_tensors_name
    for input_name in ir_graph_input_names:
        input_tensor_name = op_handler.get_op_name(input_name, 'input', remove_dots=True)
        op_handler_state.op_to_tensors[input_name] = input_tensor_name
        input_names.append(input_tensor_name)

    model_def_mgr = op_handler_state.model_def_mgr
    if order_inputs and model_input is not None:
        if isinstance(model_input, dict):
            model_def_mgr.add_model_input_code(f'\tdef forward(self, {", ".join(model_input.keys())}):')
            k = 0
            for key in model_input:
                m_inputs, k = create_nested_tuple_string_structure(input_names[1:], model_input[key], k)
                model_def_mgr.add_model_input_code(f'\t\t{m_inputs} = {key}')

        elif isinstance(model_input, tuple):
            model_def_mgr.add_model_input_code(f'\tdef forward(self, {", ".join(["input" + str(i) for i in range(len(model_input))])}):')
            k = 0
            for i, _ in enumerate(model_input):
                m_inputs, k = create_nested_tuple_string_structure(input_names[1:], model_input[i], k)
                model_def_mgr.add_model_input_code(f'\t\t{m_inputs} = input{i}')

        else:
            model_def_mgr.add_model_input_code(f'\tdef forward({", ".join(input_names)}):')
    else:
        model_def_mgr.add_model_input_code(f'\tdef forward({", ".join(input_names)}):')

    op_ref_count = {}
    for curr_op in op_handler_state.ir_graph.get_ops():  # get all gives ops is in execution order
        curr_op_type = curr_op.type
        is_custom_op = is_custom_ir_op(curr_op)
        if not is_custom_op and curr_op_type == "Conv3d" and curr_op.attrs_dict.get("reuse_sparse_indicies", False):
            curr_op_type = "SpConv3d"
        if is_custom_op:
            op_handler_obj = op_handler.CustomOpHandler(curr_op, op_handler_state)
        else:
            op_handler_obj = op_handler.ir_to_handler_dict.get(curr_op_type)(curr_op, op_handler_state)
        op_handler_obj.handle_input_transpose()
        op_handler_obj.generate_execute_code()
        op_handler_obj.handle_output_transpose()
        op_handler_obj.get_io_map()
        _delete_unused_variable(curr_op, op_ref_count, op_handler_state)

    # Write return line for model
    output_tensors = []
    output_names = op_handler_state.ir_graph_output_names

    op_to_tensors = op_handler_state.op_to_tensors

    for output_name in output_names:
        if output_name in op_to_tensors:
            output_tensors.append(op_to_tensors[output_name])
        else:
            _logger.warning('Output tensor %s not present in the graph', output_name)
    if order_outputs:
        assert model_output is not None
        output_string = f'\t\treturn {create_nested_tuple_string_structure(output_tensors, model_output)[0]}'
    else:
        output_string = f'\t\treturn {", ".join(output_tensors)}'
    op_handler_state.model_def_mgr.add_model_output_code(output_string)


def _delete_unused_variable(op: IrOp, op_ref_count: Dict[str, int], op_handler_state: op_handler_factory.OpHandlerState):
    """
    Write code for deleting unused local variables.

    :param op: Op to analyze input and output variables
    :param op_ref_count: The number of remaining references to a variable
    :param op_handler_state: Object for holding state
    """
    if is_custom_ir_op(op):
        input_op_names = op_handler.CustomOpHandler(op, op_handler_state).get_input_op_names()
    else:
        input_op_names = op_handler.ir_to_handler_dict[op.type](op, op_handler_state).get_input_op_names()

    # Remove duplicated input op names to not remove variables used later
    input_op_names = list(collections.OrderedDict.fromkeys(input_op_names))

    # Decrease reference count if a variable is used and frees it if it is no longer used
    for input_op_name in input_op_names:
        variable_name = op_handler_state.op_to_tensors[input_op_name]
        if variable_name in op_ref_count and input_op_name not in op_handler_state.ir_graph_output_tensors:
            op_ref_count[variable_name] -= 1
            if op_ref_count[variable_name] == 0:
                op_handler_state.model_def_mgr.add_execute_code(f'\t\t{variable_name} = None', op)

    op_output_names = [out_tensor.name() for out_tensor in op.outputs()]
    for output_op_name in op_output_names:
        if (output_op_name in op_handler_state.op_to_tensors) and (op.type != 'constant') \
                and (output_op_name not in op_handler_state.ir_graph_output_op_names):
            variable_name = op_handler_state.op_to_tensors[output_op_name]
            op_ref_count[variable_name] = len(op_handler_state.ir_graph.get_op_output_nodes(op)) # TODO: FIX


def create_nested_tuple_string_structure(src_list: List[str], target_structure,
                                         curr_index: int = 0) -> Tuple[str, int]:
    """
    Given a flattened list of original outputs, return a stringified version of the outputs arranged in the same format
    as model outputs, mimicking any tuple nesting.

    :param src_list: List of flattened names
    :param target_structure: Tensor/Tuple structure to which the src list need to be arranged.
    :param curr_index: Tracks the current index of src_list being processed
    :return: String of src names arranged in the same format as target_structure
    """
    assert curr_index < len(src_list)
    if isinstance(target_structure, (tuple, list)):
        partial_output_list = ['(']
        for output in target_structure:
            # For each output in the current tuple, recurse and obtain a stringified version of the output
            sub_output_list, curr_index = create_nested_tuple_string_structure(src_list, output, curr_index)
            partial_output_list.append(sub_output_list)
            partial_output_list.append(', ')
        partial_output_list.append(')')
        # Concatenate all stringified outputs for the tuple that finished processing, and return it as one string
        return ''.join(partial_output_list), curr_index

    # Base case, simply return current output as is
    return src_list[curr_index], curr_index + 1


def create_nested_structure(src_list: List[str], target_structure):
    """
    Given a flattened list of items, return a nested structure of the items arranged in the same format as
    target_structure.
    Dictionary entries will be converted to lists with entries in the same order as values in the dictionary.

    :param src_list: List of flattened items
    :param target_structure: Tensor/Tuple structure to which the src list needs to be arranged.
    :param curr_index: Tracks the current index of src_list being processed
    :return: String of src names arranged in the same format as target_structure
    """
    def create_nested_structure_helper(src_list: List[str], target_structure,
                                       curr_index: int):
        if isinstance(target_structure, dict):
            return_structure = []
            for value in target_structure.values():
                inner_structure, curr_index = create_nested_structure_helper(src_list, value, curr_index)
                return_structure.append(inner_structure)
            return return_structure, curr_index

        if isinstance(target_structure, (list, tuple)):
            return_structure = []
            for entry in target_structure:
                inner_structure, curr_index = create_nested_structure_helper(src_list, entry, curr_index)
                return_structure.append(inner_structure)
            return return_structure, curr_index

        return src_list[curr_index], curr_index + 1

    assert len(src_list) == count_inputs(target_structure, flatten_inputs=True)
    nested_structure, _ = create_nested_structure_helper(src_list, target_structure, curr_index=0)
    return nested_structure


def update_converter_args_input_names(converter_args: Optional[Dict[str, Any]], input_names:
                                      Optional[List[str]], dummy_input: Any,
                                      onnx_model_inputs: List[str]) -> Dict:
    """
    Converter args may contain input names given by the user, which do not necessarily correspond 
    to input names in the onnx graph. This can occur when model preparer's ORDER INPUTS is True 
    and input names are provided. This function updates entries in converter args to match what 
    is present in the onnx graph.

    :param converter_args: Converter args to update
    :param input_names: Input names if provided by the user
    :param dummy_input: Dummy input to the model
    :param onnx_model_inputs: Onnx model input names
    """
    if input_names is not None and converter_args is not None:
        if 'input_tensors' in converter_args:
            if not isinstance(dummy_input, (list, tuple, dict)):
                dummy_input = [dummy_input]

            nested_inputs = create_nested_structure(onnx_model_inputs, dummy_input)

            name_to_input_dict = dict(zip(input_names, nested_inputs))

            root_name_expression = r"^[a-zA-Z0-9_]+"
            indices_expression = r"\[[0-9]+\]"
            for input_tensor in converter_args['input_tensors']:
                corresponding_input = input_tensor.name
                if input_tensor.source_model_input_layout is not None or input_tensor.desired_input_shape:
                    match = re.match(root_name_expression, input_tensor.name)
                    assert match.group(0) is not None
                    assert match.group(0) in name_to_input_dict

                    indices = re.findall(indices_expression, input_tensor.name)
                    indices = [int(index[1:-1]) for index in indices]

                    corresponding_input = name_to_input_dict.get(match.group(0))
                    for i in indices:
                        corresponding_input = corresponding_input[i]

                input_tensor.name = corresponding_input

    return converter_args


def validate_inputs(dummy_input: Any, order_inputs: bool, input_names: Optional[List[str]] =
                    None, is_prepared_model_input: Callable = None):
    """Validate inputs to check the following:
        - Dicts are not used in the dummy input as individual arguments if order_inputs is True. A top level dict can be
        used as kwargs, but the order in which inputs are populated in the dict must align with the order in which they
        appear in the model's forward pass definition.
        - If input names are given, the length must match the length of dummy inputs given.

    Args:
        dummy_input: Inputs to validate
        order_inputs: Flag indicating if ordering of input is required.
        input_names: Input names to compare with dummy_input if present
        is_prepared_model_input: evaluate to True if is valid type of input.
    Raises:
        AssertionError: If Dict type dummy_input is provided along with order_inputs as True or
        if length of the input names provided does not match number of dummy
        inputs provided.
    """
    if check_input_for_dict(dummy_input, True):
        if order_inputs:
            error_msg = ('Dict found in dummy input. Dicts are not supported when ordering inputs. '
                         'Please redefine any modules with dict inputs to take tuple inputs '
                         'instead.')
            _logger.error(error_msg)
            raise AssertionError(error_msg)
        warning_msg = ('Dict found in dummy input. Flattened model arguments in the prepared model '
                       'will list dict arguments in the order in which they were inserted in the '
                       'dummy input dict.')
        _logger.warning(warning_msg)

    if input_names is not None:
        dummy_input_count = count_inputs(dummy_input, not order_inputs, is_prepared_model_input)
        if len(input_names) != dummy_input_count:
            error_msg = (f'Number of input names provided ({len(input_names)}) does not match number of dummy inputs '
                         f'provided ({dummy_input_count})')
            _logger.error(error_msg)
            raise AssertionError(error_msg)


def validate_inputs_for_ir_graph(dummy_input: Any, ir_graph: IrGraph, order_inputs: bool,
                                 is_prepared_model_input: Callable = None):
    """Validate dummy inputs to check that they match with the number of inputs the ir_graph expects.
    In the case that order_inputs is False, the number of dummy inputs may differ from the number of ir graph inputs
    if optional arguments are present in the original model definition.

    Args:
        dummy_input: Inputs to validate
        ir_graph: Ir graph to check inputs for
        order_inputs: Flag indicating if ordering of input is required.
        is_prepared_model_input: evaluate to True if is valid type of input.
    """
    dummy_input_count = count_inputs(dummy_input, True, is_prepared_model_input)
    if len(ir_graph.get_input_tensors_to_graph()) != dummy_input_count:
        if order_inputs:
            error_msg = (f'Number of ir graph inputs ({len(ir_graph.get_input_tensors_to_graph())}) does not match '
                         f'number of dummy inputs provided ({dummy_input_count}). '
                         f'Unable to align dummy inputs with ir graph inputs. '
                         f'This may be a result of optional arguments not specified in dummy '
                         f'inputs. Include any such inputs in dummy inputs.')
            _logger.error(error_msg)
            raise AssertionError(error_msg)

        _logger.warning('Number of ir graph inputs (%s) does not match number of flattened '
                        'dummy inputs provided (%s). This may be a result of optional arguments '
                        'not specified in dummy inputs. The prepared model will expect any such '
                        'optional arguments in the original model as required arguments.',
                        len(ir_graph.get_input_tensors_to_graph()), dummy_input_count)


def validate_emitter_file_paths(file_dict):
    """Validate if the paths in the dict exists or not.

    Args:
        file_dict: Dictionary with file paths as values.
    """
    for file_name, file_path in file_dict.items():
        if not os.path.exists(file_path):
            raise RuntimeError(f'Failed to generate {file_name}')
        else:
            _logger.info(f'{file_name} saved at location {file_path}')
