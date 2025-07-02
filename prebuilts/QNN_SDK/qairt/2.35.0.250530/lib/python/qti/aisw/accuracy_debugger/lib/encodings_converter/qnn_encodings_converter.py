# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os

from logging import Logger

from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, dump_json, santize_node_name
from qti.aisw.accuracy_debugger.lib.graph_op.target_op import TargetOp
from qti.aisw.accuracy_debugger.lib.encodings_converter.encodings_converter import EncodingsConverter
from qti.aisw.accuracy_debugger.lib.utils.encodings_utils import get_resolved_names,\
    convert_qnn_enc_to_qairt, QNN_PARAM_TYPE


class QnnEncodingsConverter(EncodingsConverter):
    '''
    class for converting qnn model_net.json file to AIMET format encodings file
    '''

    def __init__(self, framework_model_path: str, model_net_json_path: str, working_dir: str, logger: Logger) -> None:
        '''
        initializes the QairtEncodingsConverter class and its base class Encodings Converter

        :param framework_model_path: path to framework model.
        :param model_net_json_path: path to qnn model_net.json file
        :param working_dir: path to working directory
        :param logger: object of Logger
        '''
        super().__init__(framework_model_path, working_dir, logger)
        self._model_net_json_path = model_net_json_path
        self._qnn_framework_name_map = {}
        self._child_initialize()

    def _child_initialize(self):
        '''
        initializes the children class variables
        '''
        self._resolved_target_activations, self._qnn_framework_name_map = self._create_tensor_mapping()
        self._user_encodings, self._version = self._get_net_json_encodings()
        self._create_target_connected_graph()

        target_activation_op_map = {}
        for _, op in self._target_connected_graph.items():
            for output_name in op.get_outputs():
                target_activation_op_map[output_name] = op
        self._target_activation_op_map = target_activation_op_map

        tensor_mapping_path = os.path.join(self._working_dir, 'tensor_mapping.json')
        dump_json(self._resolved_target_activations, tensor_mapping_path)

    def _create_tensor_mapping(self):
        '''
        creates tensor mapping between framework tensors and qnn tensors
        :return framework_qnn_name_map: framework activation name as key and qnn name as value
        :return qnn_framework_name_map: qnn activation name as key and framework name as value
        '''
        framework_qnn_name_map = {}
        qnn_framework_name_map = {}
        net_json = read_json(self._model_net_json_path)
        qnn_tensors = net_json['graph']['tensors'].keys()
        # framework_activation_name -> sanitized_activation_name
        # Possible scenarios:
        # 1. sanitized_activation_name present in qnn_net_json
        # 2. sanitized_activation_name not present in qnn_net_json. Two reasons:
        #       2.a. Name has been changed: 491 -> _491_reshape (example)
        #       2.b. framework_activation is optimized and no longer alive
        for framework_activation_name in self._framework_activation_op_map:
            sanitized_activation_name = santize_node_name(framework_activation_name)
            # Case 1
            if sanitized_activation_name in qnn_tensors:
                framework_qnn_name_map[framework_activation_name] = sanitized_activation_name
                qnn_framework_name_map[sanitized_activation_name] = framework_activation_name
            # Case 2
            else:
                # Case 2.a, ignore case 2.b
                for qnn_tensor_name in qnn_tensors:
                    resolved_names = get_resolved_names(qnn_tensor_name)
                    if sanitized_activation_name in resolved_names:
                        framework_qnn_name_map[framework_activation_name] = qnn_tensor_name
                        qnn_framework_name_map[qnn_tensor_name] = framework_activation_name
                        break

        return framework_qnn_name_map, qnn_framework_name_map

    def _get_net_json_encodings(self) -> dict:
        '''
        Create qairt-quantizer format encodings from qnn model_net.json
        '''

        model_net_json = read_json(self._model_net_json_path)
        qairt_encodings = {'activation_encodings': {}, 'param_encodings': {}}
        net_json_encodings = model_net_json['graph']['tensors']
        visited_qnn_tensors = set()

        for framework_activation_name, framework_op in self._framework_activation_op_map.items():
            # Address all the framework tensors which has 1-1 encodings in net_json
            if framework_activation_name in self._resolved_target_activations:
                qnn_activation_name = self._resolved_target_activations[framework_activation_name]
                if net_json_encodings[qnn_activation_name]['type'] != QNN_PARAM_TYPE:
                    qairt_enc = convert_qnn_enc_to_qairt(net_json_encodings[qnn_activation_name])
                    if qairt_enc:
                        qairt_encodings['activation_encodings'][framework_activation_name] = qairt_enc
                    visited_qnn_tensors.update([qnn_activation_name])

            # Add the param tensor encodings if any
            for input_tensor_name in framework_op.get_inputs():
                sanitized_input_tensor = santize_node_name(input_tensor_name)
                if sanitized_input_tensor in net_json_encodings and net_json_encodings[sanitized_input_tensor]['type'] == QNN_PARAM_TYPE:
                    qairt_enc = convert_qnn_enc_to_qairt(net_json_encodings[sanitized_input_tensor])
                    if qairt_enc:
                        qairt_encodings['param_encodings'][input_tensor_name] = qairt_enc
                    visited_qnn_tensors.update([sanitized_input_tensor])

        # At this point following qnn tensor encodings has not been resolved:
        # 1. convert_ops
        # 2. some ops added by QNN which does not has any framework map, ignore them
        # Handle the convert_ops
        for qnn_tensor_name in (net_json_encodings.keys() - visited_qnn_tensors):
            if 'converted_QNN_DATATYPE' in qnn_tensor_name:
                visited_qnn_tensors.update([qnn_tensor_name])
                qairt_encodings['activation_encodings'][qnn_tensor_name] = convert_qnn_enc_to_qairt(net_json_encodings[qnn_tensor_name])
            else:
                # type 2
                continue

        return qairt_encodings

    def _create_target_connected_graph(self):
        '''
        creates target connected graph from dlc graph
        '''
        net_json = read_json(self._model_net_json_path)
        qnn_nodes = net_json['graph']['nodes']

        target_connected_graph = {}

        for op_name, qnn_op in qnn_nodes.items():
            target_op = TargetOp(op_name)
            target_op.set_op_type(qnn_op['type'].lower())
            op_activations = [self._qnn_framework_name_map.get(output_name, output_name) for output_name in qnn_op['output_names']]
            op_inputs = [self._qnn_framework_name_map.get(input_name, input_name) for input_name in qnn_op['input_names']]
            target_op.set_inputs(op_inputs)
            target_op.set_outputs(op_activations)
            target_connected_graph[op_name] = target_op

        # Make target_op for inputs
        for _, framework_op in self._framework_activation_op_map.items():
            if framework_op.get_op_type() == "input":
                name = framework_op.get_name()
                target_op = TargetOp(name)
                target_op.set_op_type("input")
                target_op.set_inputs([])
                target_op.set_outputs(framework_op.get_outputs())
                target_connected_graph[name] = target_op

        for _, node1 in target_connected_graph.items():
            for _, node2 in target_connected_graph.items():
                for output in node1.get_outputs():
                    if output in node2.get_inputs():
                        # node1 -> node2
                        node1.set_children_ops([node2])
                        node2.set_parent_ops([node1])

        self._target_connected_graph = target_connected_graph
