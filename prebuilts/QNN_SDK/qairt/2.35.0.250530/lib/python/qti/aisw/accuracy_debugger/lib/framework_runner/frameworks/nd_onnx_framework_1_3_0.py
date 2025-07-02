# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import re

from functools import reduce
from operator import mul

import logging
from collections import OrderedDict
import onnx
import numpy as np
import onnxruntime
from onnxsim import simplify
from onnx.onnx_ml_pb2 import ValueInfoProto

from qti.aisw.accuracy_debugger.lib.framework_runner.frameworks.nd_base_framework import BaseFramework
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_warning_message
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import ModelHelper, dump_intermediate_tensors,\
    dump_profile_json, MAX_RAM_LIMIT, GB
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.graph_op.framework_op import FrameworkOp


class OnnxFramework_1_3_0(BaseFramework):
    __VERSION__ = '1.3.0'
    FRAMEWORK_SUFFIX = '.onnx'

    def __init__(self, logger, custom_op_lib=None):
        super(OnnxFramework_1_3_0, self).__init__(logger)
        self._model = None
        self._graph = None
        self.ort_outs = None
        self.onnx_custom_op_lib = custom_op_lib
        self.is_custom_op_registered = False
        self.ort_custom_op_sess_options = None
        self.is_model_greater_than_two_gb = False
        self.model_path = None
        self._tmp_dir = None
        self._output_list = []
        self._model_size = None
        self._occupied_ram = None
        self._connected_graph = None

    @property
    def graph(self):
        return self._graph

    def check_model_size(self, model_path):
        model = onnx.load_model(model_path)
        model_size_in_bytes = model.ByteSize()
        model_size_in_gb = model_size_in_bytes / GB
        if model_size_in_gb > 2:
            self.is_model_greater_than_two_gb = True
            self.logger.debug(
                f"Model is greater than 2GB in size ({model_size_in_gb} GB), proceeding accordingly.")
        self._model_size = model_size_in_gb
        self._occupied_ram = 2 * model_size_in_bytes + 4 * GB
        self.logger.debug(f"Occupied RAM: {self._occupied_ram/GB} GB")

    def load_model(self, model_path):
        # Import graph and save to instance variable
        self._model = onnx.load_model(model_path)
        try:
            if self.is_model_greater_than_two_gb:
                onnx.checker.check_model(model_path)
                self.model_path = model_path
            else:
                onnx.checker.check_model(self._model)
        except ValueError as e:
            self.logger.warning(str(e))
        self._graph = self._model.graph

    def optimize(self, model_path, optimized_model_path, input_information):
        """
        Function to optimize the model using the Onnx Simplifier Tools
        Args :
            model_path              : path to the model
            optimized_model_path    : path to optimized model
        Returns:
            ret_status        : status as boolean type
            transformed_model : path to transformed model
        """
        # optimization pass by onnx simplifier
        self.load_model(model_path)
        simplified_model = None
        try:
            simplified_model, check = simplify(self._model,
                                               overwrite_input_shapes=input_information)
        except Exception as e:
            self.logger.warning(f"Onnx model simplification failed due to: {e}")
            check = False

        if check == False:
            self.logger.warning(
                "Simplified model validation failed. Continuing with unsimplified model")
            transformed_model = self._fix_old_ir_versions(model_path)
        else:
            self.logger.info("Simplified model validation is successful")

            if self.is_model_greater_than_two_gb:
                onnx.save(simplified_model, optimized_model_path, save_as_external_data=True)
            else:
                onnx.save(simplified_model, optimized_model_path)
            transformed_model = self._fix_old_ir_versions(optimized_model_path)
        return check, transformed_model

    def _fix_old_ir_versions(self, model_path):
        """
        Onnx Runtime doesn't handle the older ir_versions(<4) properly.
        This method adds the initializers to the inputs which is required by
        ir_version < 4.
        Args:
            model_path: path to the Onnx model
        Returns:
            model_path: path to the updated Onnx model.
        """
        model = onnx.load(model_path)
        if model.ir_version < 4:
            # Add initializers to the inputs.
            initializers = [i.name for i in model.graph.initializer]
            graphInputs = [i.name for i in model.graph.input]
            diff = np.setdiff1d(initializers, graphInputs, assume_unique=True).tolist()
            new_inputs = [
                onnx.helper.make_tensor_value_info(init.name, init.data_type, init.dims)
                for init in model.graph.initializer if init.name in diff
            ]
            model.graph.input.extend(new_inputs)
            if self.is_model_greater_than_two_gb:
                onnx.save(model, model_path, save_as_external_data=True)
            else:
                onnx.save(model, model_path)
        return model_path

    def add_outputs(self, output_dir, output_list=[]):
        # adds all intermediate nodes of model as output nodes
        if len(self.get_output_layers(names_only=True)) >= len(self.get_layer_identifiers()):
            # Do not modify the model if #outputnodes >= #modelnodes
            return
        for node in self._model.graph.node:
            for output in node.output:
                if len(output_list) == 0 or output in output_list:
                    self._output_list.extend([onnx.ValueInfoProto(name=output)])
        self._output_dir = output_dir
        self._tmp_dir = os.path.join(output_dir, 'tmp')
        os.makedirs(self._tmp_dir, exist_ok=True)

    def add_batch_output(self, output_list=[]):
        self._model.graph.output.extend(output_list)

    def remove_batch_output(self, output_list):
        for output in output_list:
            self._model.graph.output.remove(output)

    def register_custom_op(self, sess_opts=None):
        if sess_opts is None:
            sess_opts = onnxruntime.SessionOptions()

        if self.is_custom_op_registered:
            return sess_opts
        try:
            sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess_opts.register_custom_ops_library(self.onnx_custom_op_lib)
            self.is_custom_op_registered = True
            self.logger.info(f'Registered given custom op lib : {self.onnx_custom_op_lib}')
        except Exception as e:
            raise FrameworkError(
                f'Registration of custom op:{self.onnx_custom_op_lib} failed due to {e}.')
        return sess_opts

    def _extract_shape_info(self, tensor_info: ValueInfoProto) -> list:
        '''
        Given proto info for the tensor, return the dimension if exists

        :param tensor_info: Object of ValueInfoProto which contains the tensor information
        '''
        shape = []
        if hasattr(tensor_info.type.tensor_type, 'shape'):
            dim_info = tensor_info.type.tensor_type.shape.dim
            for dim in dim_info:
                if hasattr(dim, 'dim_param') and dim.dim_param:
                    shape.append(dim.dim_param)
                if hasattr(dim, 'dim_value') and dim.dim_value:
                    shape.append(dim.dim_value)
        return shape

    def _infer_shape(self, user_inputs: dict) -> dict:
        '''
        returns the shape for all intermediate tensors present in the model

        :param user_inputs: dict of model input names and corresponding shape provided by the user
        '''
        # We map the model input_shapes with input_shapes provided by the user to resolve any symbols
        # Then we use the same resolved symbols to resolve the shape for intermediate outputs of the model
        model_inputs = {}
        for model_input_info in self._graph.input:
            model_inputs[model_input_info.name] = self._extract_shape_info(model_input_info)

        symbols = {}
        for input_name, input_shape in user_inputs.items():
            for model_shape_dim, user_provided_inp_shape_dim in zip(model_inputs[input_name],
                                                                    input_shape):
                if isinstance(model_shape_dim, str):
                    symbols[model_shape_dim] = user_provided_inp_shape_dim

        tensor_shapes = {}
        shape_infer_model_path = os.path.join(self._tmp_dir, 'shape_infer_model.onnx')
        onnx.shape_inference.infer_shapes_path(self.model_path, shape_infer_model_path)
        infer_model = onnx.load(shape_infer_model_path, load_external_data=False)
        infer_graph = infer_model.graph
        for tensor_info in infer_graph.value_info:
            shape = self._extract_shape_info(tensor_info)
            tensor_shape = []
            for dim in shape:
                if isinstance(dim, str):
                    if dim in symbols:
                        tensor_shape.append(symbols[dim])
                    else:
                        tensor_shape = []
                        break
                else:
                    tensor_shape.append(dim)
            if tensor_shape:
                tensor_shapes[tensor_info.name] = tensor_shape

        return tensor_shapes

    def run_inference_batch(self, input_data: list, input_tensor_names: list,
                            output_tensor_names: list, model_path: str = None) -> dict:
        '''
        return dictionary of output tensor names as keys and corresponding output tensor as values

        :param input_data: list of model input tensors
        :param input_tensor_names: list of model input tensor names such that it corresponds to input_data one-to-one
        :param output_tensor_names: list of output tensor names for which outputs needs to be extracted from the model
        :param model_path: incase the model is > 2GB, model_path needs to be passed
        '''
        if len(input_data) != len(input_tensor_names):
            raise FrameworkError(get_message("ERROR_FRAMEWORK_ONNX_MISMATCH_INPUTS"))
        ort_inputs = {}

        if self.is_custom_op_registered and self.ort_custom_op_sess_options:
            ort_sess_opts = self.ort_custom_op_sess_options
        else:
            if self.onnx_custom_op_lib is not None:
                self.ort_custom_op_sess_options = self.register_custom_op()
                ort_sess_opts = self.ort_custom_op_sess_options
            else:
                ort_sess_opts = onnxruntime.SessionOptions()
        if self.is_model_greater_than_two_gb:
            if model_path:
                ort_session = onnxruntime.InferenceSession(model_path, ort_sess_opts)
            else:
                raise Exception(f"Model is of size > 2GB and model_path is None")
        else:
            ort_session = onnxruntime.InferenceSession(self._model.SerializeToString(),
                                                       ort_sess_opts)
        for data, name in zip(input_data, input_tensor_names):
            ort_inputs[name] = data

        outputs = [x.name for x in ort_session.get_outputs()]
        ort_outs = ort_session.run(outputs, ort_inputs)

        result = {}
        for output_tensor, data in zip(outputs, ort_outs):
            for output_tensor_name in output_tensor_names:
                if str(output_tensor_name) == str(output_tensor):
                    result.update({output_tensor: data})

        return result

    def _run_2gb_inference(self, input_data: list, input_tensor_names: list,
                           output_tensor_names: list, use_native_output_files: bool) -> None:
        '''
        executes the inference for models > 2GB.

        :param input_data: list of model input tensors
        :param input_tensor_names: list of model input tensor names such that it corresponds to input_data one-to-one
        :param output_tensor_names: list of output tensor names for which outputs needs to be extracted from the model
        :param use_native_output_files: dumps outputs as per framework model's actual data types
        '''

        def _execute_chunk(output_chunk, use_native_output_files):
            '''
            given the list of outputs, run the model and dump the outputs for the same
            '''
            self.add_batch_output(output_chunk)
            # Since model is greater than 2 GB, we need to save the model to retain the intermediate outputs
            model_path = os.path.join(self._tmp_dir, 'model_with_intermediate_outputs.onnx')
            self.logger.info("Saving model to: {}".format(model_path))
            onnx.save(self._model, model_path, save_as_external_data=True)
            chunk_result = self.run_inference_batch(input_data, input_tensor_names,
                                                    output_tensor_names, model_path)
            dump_intermediate_tensors(self._output_dir, chunk_result, self.logger, use_native_output_files)
            dump_profile_json(self._output_dir, chunk_result)
            self.remove_batch_output(output_chunk)

        # Get the intermediate tensor shapes
        user_inputs = {name: data.shape for data, name in zip(input_data, input_tensor_names)}
        tensor_shapes = self._infer_shape(user_inputs)

        # Now execute the model with one chunk output extraction at a time
        current_output_chunk = []
        current_output_chunk_size = 0

        # Run the model for list of outputs such that total size of outputs is less than MAX_RAM_LIMIT
        for output in self._output_list:
            if output.name in tensor_shapes:
                # 2 times of the tensor size because at one point of time, there are 2 copies are live
                # when we extract it from the inferenced model
                current_output_size = reduce(mul, tensor_shapes[output.name]) * 4 * 2
                # chunk output size must not exceed MAX_RAM_LIMIT - self._occupied_ram.
                # 4 GB is kept free so that other programs can function normally and there is minimum page faults.
                if current_output_size + current_output_chunk_size + self._occupied_ram <= MAX_RAM_LIMIT:
                    current_output_chunk.append(output)
                    current_output_chunk_size += current_output_size
                else:
                    _execute_chunk(current_output_chunk, use_native_output_files)
                    current_output_chunk = [output]
                    current_output_chunk_size = current_output_size

        if current_output_chunk:
            _execute_chunk(current_output_chunk, use_native_output_files)

    def run_inference(self, input_data: list, input_tensor_names: list, output_tensor_names: list, use_native_output_files: bool) -> dict:

        '''
        executes the inference for model

        :param input_data: list of model input tensors
        :param input_tensor_names: list of model input tensor names such that it corresponds to input_data one-to-one
        :param output_tensor_names: list of output tensor names for which outputs needs to be extracted from the model
        :param use_native_output_files: dumps outputs as per framework model's actual data types
        '''
        if len(input_data) != len(input_tensor_names):
            raise FrameworkError(get_message("ERROR_FRAMEWORK_ONNX_MISMATCH_INPUTS"))

        result = {}
        try:
            if self.is_model_greater_than_two_gb:
                self._run_2gb_inference(input_data, input_tensor_names, output_tensor_names, use_native_output_files)
            else:
                self.add_batch_output(self._output_list)
                result = self.run_inference_batch(input_data, input_tensor_names,
                                                  output_tensor_names)
        except Exception as e:
            raise Exception(f"Model execution failed with error: {e}")

        return result

    def get_intermediate_tensors(self, input_tensors, output_tensors):
        tensor_pairs = []
        input_initializer = [node.name for node in self._model.graph.initializer]
        for node in self._model.graph.node:
            inputs = []
            outputs = []
            for input in node.input:
                input_name = onnx.ValueInfoProto(name=input).name
                if input_name not in input_initializer:
                    inputs.append(input_name)
            for output in node.output:
                outputs.append(onnx.ValueInfoProto(name=output).name)
            tensor_pairs.append((inputs, outputs))

        return tensor_pairs

    def get_dimensions(self, tensor_name):
        pass

    def get_graph_structure(self):
        """Creates a detailed list of the network's operators.

        Iterates through the operators in the net, and retrieves every
        operator's index , as well as its type, inputs, and outputs

        :return: dictionary indexed by op index with values containing
            the index, tuple of list of inputs and list of outputs
        """
        op_dict = OrderedDict()
        i = 0
        input_initializer = [node.name for node in self._model.graph.initializer]
        for node in self._model.graph.node:
            inputs = []
            outputs = []
            for input in node.input:
                input_name = onnx.ValueInfoProto(name=input).name
                if input_name not in input_initializer:
                    inputs.append(input_name)
            for output in node.output:
                outputs.append(onnx.ValueInfoProto(name=output).name)
            op_dict[i] = (node.op_type, inputs, outputs)
            i += 1
        return op_dict

    def get_mapping_for_qnn_node(self, qnn_output):  # type: (str) -> str
        """Returns framework node name :return: the node name of qnn_output in
        the framework."""
        if qnn_output[1:].isdigit():
            qnn_output = qnn_output[1:]
        check_conv_batch_norm = False
        for node in self._model.graph.node:
            if not check_conv_batch_norm:
                for output in node.output:
                    tensor_name = onnx.ValueInfoProto(name=output).name
                    tensor_replace = re.sub(pattern='\\W', repl='_', string=tensor_name)
                    if qnn_output == tensor_replace:
                        if node.op_type == 'Conv':
                            check_conv_batch_norm = True
                            break
                        else:
                            return qnn_output
            else:
                check_conv_batch_norm = False
                if node.op_type == 'BatchNormalization':
                    return onnx.ValueInfoProto(name=node.output[0]).name  # node.output[0]
                else:
                    return qnn_output

        # if no matching, some warning will occur.
        logging.warning(get_warning_message("WARNING_FRAMEWORK_ONNX_MISMATCH_TENSOR")(qnn_output))
        return ""

    def get_mapping_for_snpe_node(self, snpe_output_tensor):  # type: (str) -> str
        """Returns framework node name :return: the node name of
        snpe_output_tensor in the framework."""
        check_conv_batch_norm = False
        for node in self._model.graph.node:
            if not check_conv_batch_norm:
                for output in node.output:
                    tensor_name = santize_node_name(onnx.ValueInfoProto(name=output).name)
                    if tensor_name == snpe_output_tensor or \
                            tensor_name == "_"+snpe_output_tensor:
                        if node.op_type == 'Conv':
                            check_conv_batch_norm = True
                            break
                        else:
                            return tensor_name
            else:
                check_conv_batch_norm = False
                if node.op_type == 'BatchNormalization':
                    # node.output[0]
                    return santize_node_name(onnx.ValueInfoProto(name=node.output[0]).name)
                else:
                    return tensor_name

        # if no matching, some warning will occur.
        logging.warning(
            get_warning_message("WARNING_FRAMEWORK_ONNX_MISMATCH_TENSOR")(snpe_output_tensor))
        return ""

    def get_version(self):
        return onnx.__version__

    def extract(self, start_layer_output_name, end_layer_output_name=None, out_model_path=None):
        raise NotImplementedError('Method extract is not implemented for onnx version < 1.8.0')

    ################################Layerwise_snooping utility methods ####################################
    def get_layer_identifiers(self, op_types_only=False):
        """This method returns list of layer name, output name and type in the
        onnx model.

        Returns:
            layers : list of tuples containing layer_name, output_name, layer_op_type.
        """
        layer_info = []
        model = self._model
        for node in model.graph.node:
            if op_types_only:
                if node.op_type not in layer_info:
                    layer_info.append(node.op_type)
            else:
                if node.op_type in ['Constant', 'Identity']:
                    continue
                layer_info.append((node.name, node.output[0], node.op_type))
        return layer_info

    def get_output_layers(self, names_only=False):
        """This method returns list of output layers and their datatype of
        provided onnx model.

        Args:
            names_only : boolean flag to return just list of output layer names
        Returns:
            output_layers_info : list of tuple containing output layer names and corresponding
            numpy datatype.
        """
        output_layers_info = []
        model = self._model

        layer_out_type_map = {}
        if not names_only:
            for node in model.graph.node:
                for idx in range(len(node.output)):
                    layer_out_type_map[node.output[idx]] = node.op_type

        # form list of tuple containing output layer names and corresponding numpy datatype
        for vi in model.graph.output:
            out_name = vi.name
            if names_only:
                output_layers_info.append(out_name)
            else:
                dim = []
                for i in range(len(vi.type.tensor_type.shape.dim)):
                    dim.append(vi.type.tensor_type.shape.dim[i].dim_value)
                try:
                    (out_dtype,
                     _) = ModelHelper.onnx_type_to_numpy(str(vi.type.tensor_type.elem_type))
                except Exception as e:
                    logging.error(e)
                output_layers_info.append((out_name, out_dtype, layer_out_type_map[out_name], dim))
        return output_layers_info

    def get_input_layers(self, names_only=False):
        """This method returns list of inputs in the onnx model.

        Args:
            names_only: only return list of names
        Returns:
            input_layers_info : list of tuple containing input layer names and corresponding
            numpy datatype.
        """
        input_layers_info = []
        model = self._model
        # form list of tuple containing input layer names and corresponding numpy datatype
        for vi in model.graph.input:
            inp_name = vi.name
            if names_only:
                input_layers_info.append(inp_name)
            else:
                (inp_dtype, _) = ModelHelper.onnx_type_to_numpy(str(vi.type.tensor_type.elem_type))
                dim = []
                for i in range(len(vi.type.tensor_type.shape.dim)):
                    if vi.type.tensor_type.shape.dim[i].dim_value:
                        dim.append(vi.type.tensor_type.shape.dim[i].dim_value)
                    elif vi.type.tensor_type.shape.dim[i].dim_param:
                        dim.append(vi.type.tensor_type.shape.dim[i].dim_param)
                input_layers_info.append((inp_name, inp_dtype, dim))
        return input_layers_info

    def get_node_activation_to_node_inputs_map(self):
        """
        Return a dictionary of node activation to node inputs map
        """
        node_activation_node_input_map = {}
        for node in self._graph.node:
            for out in node.output:
                node_activation_node_input_map[out] = node.input
        return node_activation_node_input_map

    def get_connected_graph(self):
        if self._connected_graph is None:
            self._create_connected_graph()
        return self._connected_graph

    def _create_connected_graph(self):
        '''
        create a map of node name to Op object
        '''
        connected_graph = {}
        for idx, inp in enumerate(self._graph.input):
            name = "input_{}".format(idx)
            op = FrameworkOp(name)
            op.set_inputs([])
            op.set_outputs([inp.name])
            op.set_op_type("input")
            connected_graph[name] = op

        for idx, node in enumerate(self._graph.node):
            name = node.name if node.name else "op_{}".format(idx)
            op = FrameworkOp(name)
            op.set_inputs(node.input)
            op.set_outputs(node.output)
            op.set_op_type(node.op_type)
            connected_graph[name] = op

        # Now set the children and parent ops for each op
        for _, node1 in connected_graph.items():
            for _, node2 in connected_graph.items():
                for output in node1.get_outputs():
                    if output in node2.get_inputs():
                        # node1 -> node2
                        node1.set_children_ops([node2])
                        node2.set_parent_ops([node1])

        self._connected_graph = connected_graph
