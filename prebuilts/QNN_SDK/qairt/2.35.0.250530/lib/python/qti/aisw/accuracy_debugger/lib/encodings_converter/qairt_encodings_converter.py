# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os

from logging import Logger

from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, dump_json
from qti.aisw.accuracy_debugger.lib.graph_op.target_op import TargetOp
from qti.aisw.accuracy_debugger.lib.encodings_converter.encodings_converter import EncodingsConverter
from qti.aisw.accuracy_debugger.lib.utils.encodings_utils import get_resolved_names


class QairtEncodingsConverter(EncodingsConverter):
    '''
    class for converting qairt-quantizer dumped encodings file to AIMET format encodings
    file
    '''

    def __init__(self, framework_model_path: str, dlc_path: str, qairt_encodings_file_path: str,
                 working_dir: str, logger: Logger) -> None:
        '''
        initializes the QairtEncodingsConverter class and its base class Encodings Converter

        :param framework_model_path: path to framework model.
        :param dlc_path: path to quantized dlc file
        :param qairt_encodings_file_path: path to qairt encodings file
        :param working_dir: path to working directory
        :param logger: object of Logger
        '''
        super().__init__(framework_model_path, working_dir, logger)
        self._dlc_path = dlc_path
        self._qairt_encodings_file_path = qairt_encodings_file_path
        self._child_initialize()

    def _get_user_encodings(self):
        qairt_encodings = read_json(self._qairt_encodings_file_path)
        if isinstance(qairt_encodings['activation_encodings'], dict):
            version = 'legacy'
        else:
            version = qairt_encodings['version']

        if version == 'legacy':
            return qairt_encodings, version
        else:
            user_encodings = {'activation_encodings': {}, 'param_encodings': {}}
            for tensor_type in qairt_encodings:
                if tensor_type in ['activation_encodings', 'param_encodings']:
                    for encoding in qairt_encodings[tensor_type]:
                        user_encodings[tensor_type][encoding['name']] = encoding

            return user_encodings, version

    def _child_initialize(self):
        '''
        initializes the children class variables
        '''
        self._create_target_connected_graph()
        target_activation_op_map = {}
        self._user_encodings, self._version = self._get_user_encodings()

        for _, op in self._target_connected_graph.items():
            for output_name in op.get_outputs():
                if output_name not in self._framework_activations and 'converted_QNN_DATATYPE' not in output_name:
                    # Resolve target activation and op incase of name change
                    # Do not do it for convert_QNN_DATATYPE activations
                    resolved_name, modified_op = self._resolve_target_name_change(output_name, op)
                else:
                    # output_name present in both framework and target graph
                    resolved_name, modified_op = output_name, op
                if output_name in self._user_encodings['activation_encodings']:
                    # sometimes encodings might not be present
                    # e.g. integer inputs
                    self._user_encodings['activation_encodings'][
                        resolved_name] = self._user_encodings['activation_encodings'][output_name]
                    # update the name of the in the encoding incase version is 1.0.0
                    if self._version == "1.0.0":
                        self._user_encodings['activation_encodings'][resolved_name][
                            'name'] = resolved_name
                # Prepare tensor mapping for the target activations
                # framework_name: target_name
                if resolved_name in self._framework_activation_op_map:
                    self._resolved_target_activations[resolved_name] = output_name
                target_activation_op_map[resolved_name] = modified_op
        self._target_activation_op_map = target_activation_op_map

        tensor_mapping_path = os.path.join(self._working_dir, 'tensor_mapping.json')
        dump_json(self._resolved_target_activations, tensor_mapping_path)

    def _modify_target_op(self, output_name: str, resolved_name: str, op: TargetOp) -> TargetOp:
        '''
        Incase the activation has changed in the target dlc, we modify the target_op
        accordingly.

        returns: modified op object
        :param output_name: current output name of the op
        :param resolved_name: name for the op output which is also present in the framework
        :param op: object of TargetOp class representing the current op
        '''
        op_activations = op.get_outputs()
        modified_op_activations = [
            resolved_name if activation == output_name else activation
            for activation in op_activations
        ]
        op.set_outputs(modified_op_activations)
        for children_op in op.get_children_ops():
            children_op_inputs = children_op.get_inputs()
            modified_op_inputs = [
                resolved_name if op_input == output_name else op_input
                for op_input in children_op_inputs
            ]
            children_op.set_inputs(modified_op_inputs)

        return op

    def _resolve_target_name_change(self, output_name: str, op: TargetOp) -> tuple:
        '''
        if the target activation name has been changed,
        resolve such names if possible and accordingly modify the target op object

        :return: resolved output name and modified target op object
        :param output_name: output_name of the target op
        :param op: object of current target op
        '''
        resolved_names = get_resolved_names(output_name)
        for resolved_name in resolved_names:
            if resolved_name in (self._framework_activations - self._target_connected_graph.keys()):
                # resolved name present in framework graph but not in target graph
                # 419(in framework) -> 419_reshpe(target)
                # 419(in framework) -> 419.nchw(target)
                # and there is no 419 activation in target
                modified_op = self._modify_target_op(output_name, resolved_name, op)
                return resolved_name, modified_op

        # resolved name not present in framework graph
        # this is new logical node added by target
        # Matmul_0_pre_reshape(target)
        # do nothing, return output_name and op
        # or any of the resolved_names in target graph
        # 419, 491.nchw, both in dlc, then do not resolve
        # the name for 419.nchw
        return output_name, op

    def _create_target_connected_graph(self):
        '''
        creates target connected graph from dlc graph
        '''
        from qti.aisw.dlc_utils import modeltools
        from qti.aisw.converters.common import ir_graph

        model_reader = modeltools.IrDlcReader()
        model_reader.open(self._dlc_path)
        ir_graph = model_reader.get_ir_graph()

        target_connected_graph = {}

        # Make target_op for inputs
        for idx, inp in enumerate(ir_graph.get_input_tensors_to_graph()):
            name = "input_{}".format(idx)
            target_op = TargetOp(name)
            target_op.set_op_type("input")
            target_op.set_data_type(inp.data_type().name)
            target_op.set_inputs([])
            target_op.set_outputs([inp.name()])
            target_connected_graph[name] = target_op

        for op in ir_graph.get_ops():
            static_tensors = []
            for input in op.inputs():
                if 'IrStaticTensor' in str(type(input)):
                    static_tensors.append(input.name())
            all_outputs_names = [output.name() for output in op.outputs()]
            for output in op.outputs():
                target_op = TargetOp(output.name())
                target_op.set_op_type(op.type)
                target_op.set_data_type(output.data_type().name)
                inp_names = [inp.name() for inp in op.inputs()]
                target_op.set_inputs(inp_names)
                target_op.set_outputs(all_outputs_names)
                target_op.set_static_tensors(static_tensors)
                target_connected_graph[output.name()] = target_op
        for _, node1 in target_connected_graph.items():
            for _, node2 in target_connected_graph.items():
                for output in node1.get_outputs():
                    if output in node2.get_inputs():
                        # node1 -> node2
                        node1.set_children_ops([node2])
                        node2.set_parent_ops([node1])

        self._target_connected_graph = target_connected_graph
