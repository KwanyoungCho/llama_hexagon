# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
from abc import abstractmethod
from logging import Logger

from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_runner import ModelTraverser
from qti.aisw.accuracy_debugger.lib.utils.encodings_utils import needs_encoding_update,\
     identify_inter_activations_path, is_convert_op_in_path, get_framework_type,\
         get_encodings_structure, add_encodings
from qti.aisw.accuracy_debugger.lib.utils.graph_utils import get_common_parent_activations


class EncodingsConverter:
    '''
    Base class for converting encodings file to AIMET format encodings file
    '''

    def __init__(self, framework_model_path: str, working_dir: str, logger: Logger) -> None:
        '''
        initializes the EncodingsConverter class

        :param framework_model_path: path to framework model.
        :param working_dir: path to working directory
        :param logger: object of Logger
        '''
        self._framework_model_path = framework_model_path
        self._working_dir = working_dir
        self._logger = logger
        self._framework_connected_graph = None
        self._target_connected_graph = None
        self._framework_activations = None
        self._framework_activation_op_map = None
        self._framework_input_op_map = None
        self._user_encodings = {}
        self._target_activation_op_map = None
        self._resolved_target_activations = {}
        self._version = None
        self._intialize()

    def _intialize(self) -> None:
        '''
        initializes the base class variables
        '''
        # Create working directory
        os.makedirs(self._working_dir, exist_ok=True)

        # create model_traverser class object
        framework_args = Namespace(framework=get_framework_type(self._framework_model_path),
                                   version=None, model_path=self._framework_model_path,
                                   output_dir=self._working_dir)
        model_traverser = ModelTraverser(logger=self._logger, args=framework_args, layer_list=False)

        # create framework_connected_graph
        framework_instance = model_traverser.framework_instance
        self._framework_connected_graph = framework_instance.get_connected_graph()

        # create framework_op_activation_to_framework_op_map with node activation as key
        # and framework_op object as value
        framework_activation_op_map = {}
        framework_input_op_map = {}
        for _, op in self._framework_connected_graph.items():
            for output in op.get_outputs():
                framework_activation_op_map[output] = op
            for input_name in op.get_inputs():
                framework_input_op_map[input_name] = op
        self._framework_activation_op_map = framework_activation_op_map
        self._framework_input_op_map = framework_input_op_map

        self._framework_activations = [
            output for output, op in framework_activation_op_map.items()
            if op.get_op_type() != "input"
        ]

    @abstractmethod
    def _create_target_connected_graph(self) -> None:
        '''
        creates target connected graph
        '''
        pass

    def get_framework_activations(self) -> list:
        '''
        :return: list of framework activations
        '''

        return self._framework_activations

    def get_framework_connected_graph(self) -> dict:
        '''
        :return: framework connected graph
        '''

        return self._framework_connected_graph

    def get_target_connected_graph(self) -> dict:
        '''
        :return: target connected graph
        '''

        return self._target_connected_graph

    def get_framework_activation_op_map(self) -> dict:
        '''
        :return: framework activation to framework op map
        '''

        return self._framework_activation_op_map

    def get_framework_input_op_map(self) -> dict:
        '''
        :return: framework input to framework op map
        '''

        return self._framework_input_op_map

    def get_target_activation_op_map(self) -> dict:
        '''
        :return: target activation to target op map
        '''

        return self._target_activation_op_map

    def get_resolved_target_activation(self) -> dict:
        '''
        :return: map of framework name to target activation name
        '''

        return self._resolved_target_activations

    def _set_output_encodings(self, encodings: dict, activation: str,
                              visited_activation_encodings: dict) -> tuple[dict, dict]:
        '''
        given the activation name, sets the activation encodings
        and returns the encodings dict

        :param encodings: dictionary of converted encodings
        :param activation: activation name in the framework graph
        :param visited_activation_encodings: set of tensor names for which encodings have already been added
        :return encodings: updated encodings with input's quant params
        :return visited_activation_encodings: updated dictionary of visited activation encodings,
            with activation names as keys and encodings as values
        '''
        if activation in self._user_encodings['activation_encodings'] and\
            (activation not in visited_activation_encodings or\
                needs_encoding_update(self._user_encodings["activation_encodings"][activation],
                                      visited_activation_encodings[activation], self._version)):
            tensor_info = {
                'name': activation,
                'type': 'activation_encodings',
                'encoding': self._user_encodings['activation_encodings'][activation]
            }
            encodings = add_encodings(encodings, tensor_info, self._version)
            visited_activation_encodings[activation] = self._user_encodings['activation_encodings'][
                activation]

        return encodings, visited_activation_encodings

    def _set_param_encodings(self, encodings: dict, param: str) -> dict:
        '''
        given the param name, sets the param encodings
        and returns the encodings dict

        :param encodings: dictionary of converted encodings
        :param param: param name of op in the framework graph
        '''
        # TODO: Once shared weights issue is resolved, address convert_op @ param level
        tensor_info = {
            'name': param,
            'type': 'param_encodings',
            'encoding': self._user_encodings["param_encodings"][param]
        }
        encodings = add_encodings(encodings, tensor_info, self._version)

        return encodings

    def _set_input_encodings(self, encodings: dict, input_name: str, output_name: str,
                             ignore_activation_encodings: set,
                             visited_activation_encodings: dict) -> tuple[dict, dict]:
        '''
        given the activation name, sets the activation encodings
        and returns the encodings dict

        :param encodings: dictionary of converted encodings
        :param input_name: one of the framework op's input name for which encodings has to be
            resolved and added
        :param output_name: one of the framework op's output name
        :param ignore_activation_encodings: list of target activations for which encodings has
            to be ignored in the new quantization overrides. for e.g. for conv, bn ,relu
            qnn_net_json and qairt_encodings_json gives out activation encodings for both bn and
            relu user may want to delete bn encodings for some optimizations.
        :param visited_activation_encodings: dictionary of visited activation encodings,
            with activation names as keys and encodings as values
        :return encodings: updated encodings with input's quant params
        :return visited_activation_encodings: updated dictionary of visited activation encodings,
            with activation names as keys and encodings as values
        '''

        common_parent_activations = get_common_parent_activations(input_name,
                                                                  self._target_activation_op_map,
                                                                  self._framework_activation_op_map,
                                                                  ignore_activation_encodings)

        self._logger.debug(
            f"Common parent activations for {input_name} are {str(common_parent_activations)}")
        for parent_activation in common_parent_activations:
            path = identify_inter_activations_path(output_name, parent_activation,
                                                   self._target_activation_op_map, 0)

            self._logger.debug(f"PATH {str(path)} BETWEEN: {output_name} and {parent_activation}")
            convert_op_in_between, convert_activation_name = is_convert_op_in_path(
                path, self._target_activation_op_map)
            input_tensor_enc = None
            if convert_op_in_between:
                input_tensor_enc = self._user_encodings["activation_encodings"][
                    convert_activation_name]
            else:
                if parent_activation in self._user_encodings["activation_encodings"]:
                    input_tensor_enc = self._user_encodings["activation_encodings"][
                        parent_activation]
                else:
                    self._logger.error(f"{parent_activation} not found in user encodings")
            # Now check whether encodings for the parent_activation is already present or not
            # If present, check precendece and update if requried
            if input_tensor_enc:
                tensor_info = {
                    'name': parent_activation,
                    'type': 'activation_encodings',
                    'encoding': input_tensor_enc
                }
                if parent_activation in visited_activation_encodings:
                    # Precendence: int16>int8>int4>fp32>fp16>fp8
                    if needs_encoding_update(input_tensor_enc,
                                             visited_activation_encodings[parent_activation],
                                             self._version):
                        encodings = add_encodings(encodings, tensor_info, self._version)
                        visited_activation_encodings[parent_activation] = input_tensor_enc
                else:
                    encodings = add_encodings(encodings, tensor_info, self._version)
                    visited_activation_encodings[parent_activation] = input_tensor_enc

        return encodings, visited_activation_encodings

    def create_subgraph_quantization_overrides(
        self, subgraph_target_activations: list = [],
        ignore_activation_encodings: set = set()) -> dict:
        '''
        creates override file for the subgraph. Eliminates convert ops.
        for e.g. for node "conv.1", it's input, param, and output encodings will be kept in lower
        precision.

        :param subgraph_target_activations: list of target activations for which subgraph
            quantization overrides has to be prepared. If empty, override will be created for full
            model graph
        :param ignore_activation_encodings: list of target activations for which encodings has
            to be ignored in the new quantization overrides. for e.g. for conv, bn ,relu
            qnn_net_json and qairt_encodings_json gives out activation encodings for both bn and
            relu user may want to delete bn encodings for some optimizations.
        :return encodings: override encodings for the given subgraph
        '''
        encodings = get_encodings_structure(self._version)
        visited_activation_encodings = dict()

        subgraph_activations = subgraph_target_activations or self._target_activation_op_map.keys()

        for output_name in subgraph_activations:
            self._logger.debug("-" * 75)
            self._logger.debug("ACTIVATION: {}".format(output_name))

            # Add output encodings
            if output_name in (set(self._framework_activations) - ignore_activation_encodings):
                encodings, visited_activation_encodings = self._set_output_encodings(
                    encodings, output_name, visited_activation_encodings)

            current_op = self._target_activation_op_map[output_name]
            for input_tensor_name in current_op.get_inputs():
                # Add param encodings
                # Overwrite the already present float static encodings for the params
                if input_tensor_name in self._user_encodings["param_encodings"]:
                    encodings = self._set_param_encodings(encodings, input_tensor_name)

                # Add input encodings if input is not in ignore_activation_encodings
                # Now there can be two cases:
                # 1. There is convert_op in between current_op and its parent op
                # 2. There is no convert_op in between
                # only for input_tensors which are actually activations
                # e.g. 316's inputs: ['315', 'features.0.1.weight', 'features.0.1.bias',
                # 'features.0.1.running_mean', 'features.0.1.running_var']
                # ['features.0.1.running_mean', 'features.0.1.running_var'] does not even
                # have param encodings
                elif input_tensor_name in self._target_activation_op_map.keys():
                    encodings, visited_activation_encodings = self._set_input_encodings(
                        encodings, input_tensor_name, output_name, ignore_activation_encodings,
                        visited_activation_encodings)

                # input_tensor_name is actually constant but not params with no encodings or
                # in ignore_activation_encodings
                else:
                    continue

        return encodings
