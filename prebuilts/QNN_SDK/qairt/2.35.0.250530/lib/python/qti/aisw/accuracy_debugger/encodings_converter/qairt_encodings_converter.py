# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
from logging import Logger

from qti.aisw.accuracy_debugger.encodings_converter.encodings_converter import EncodingsConverter
from qti.aisw.accuracy_debugger.encodings_converter.encodings_utils import (
    EncodingVersion,
    get_encodings_version,
    get_resolved_names,
    organize_qairt_encodings,
)
from qti.aisw.accuracy_debugger.graph_op.target_op import TargetOp
from qti.aisw.accuracy_debugger.utils.file_utils import dump_json, read_json
from qti.aisw.dlc_utils import modeltools  # type: ignore


class QairtEncodingsConverter(EncodingsConverter):
    """Class for converting encoding files dumped by the QAIRT Quantizer to the AIMET format."""

    def __init__(
        self,
        framework_model_path: str,
        dlc_path: str,
        qairt_encodings_file_path: str,
        working_dir: str,
        logger: Logger,
    ) -> None:
        """Initializes the QairtEncodingsConverter class and its base class EncodingsConverter.

        Args:
            framework_model_path: Path to the framework model.
            dlc_path: Path to the quantized DLC file.
            qairt_encodings_file_path: Path to the QAIRT encodings file.
            working_dir: Path to the working directory.
            logger: Logger object.
        """
        super().__init__(framework_model_path, working_dir, logger)
        self._dlc_path = dlc_path
        qairt_encodings = read_json(qairt_encodings_file_path)
        self._version = get_encodings_version(qairt_encodings)
        self._user_encodings = organize_qairt_encodings(qairt_encodings)
        self._child_initialize()

    def _child_initialize(self):
        """Initializes the children class variables"""
        self._target_connected_graph = self._create_target_connected_graph()
        target_activation_op_map = {}

        for _, op in self._target_connected_graph.items():
            for output_name in op.outputs:
                # Resolve target activation and op incase of name change
                # Skip for converted_QNN_DATATYPE activations
                if (
                    output_name not in self._framework_activations
                    and "converted_QNN_DATATYPE" not in output_name
                ):
                    resolved_name, modified_op = self._resolve_target_name_change(output_name, op)
                else:
                    # output_name present in both framework and target graph
                    resolved_name, modified_op = output_name, op

                # encodings for output_name may not be present in the user_encodings if it is
                # one of (integer tensor, constant tensor) hence resolve only for those
                # output_name which has encodings present in user_encodings
                if output_name in self._user_encodings["activation_encodings"]:
                    self._user_encodings["activation_encodings"][resolved_name] = (
                        self._user_encodings["activation_encodings"][output_name]
                    )
                    # update the name of the in the encoding incase version is 1.0.0
                    if self._version == EncodingVersion.V1:
                        self._user_encodings["activation_encodings"][resolved_name]["name"] = (
                            resolved_name
                        )

                # Prepare tensor mapping for the target activations
                # framework_name: target_name
                if resolved_name in self._framework_activation_op_map:
                    self._resolved_target_activations[resolved_name] = output_name
                target_activation_op_map[resolved_name] = modified_op

        self._target_activation_op_map = target_activation_op_map

        tensor_mapping_path = os.path.join(self._working_dir, "tensor_mapping.json")
        dump_json(self._resolved_target_activations, tensor_mapping_path)

    def _modify_target_op(self, output_name: str, resolved_name: str, op: TargetOp) -> TargetOp:
        """Modifies the TargetOp if the activation has changed in the target DLC.

        Args:
            output_name: The output name of the operator.
            resolved_name: The resolved name for the operator output, also present in the framework.
            op: The object of the TargetOp class representing the current operator.

        Returns:
            TargetOp: The modified TargetOp object.
        """
        op_activations = op.outputs
        modified_op_activations = [
            resolved_name if activation == output_name else activation
            for activation in op_activations
        ]
        op.outputs = modified_op_activations
        for children_op in op.children_ops:
            children_op_inputs = children_op.inputs
            modified_op_inputs = [
                resolved_name if op_input == output_name else op_input
                for op_input in children_op_inputs
            ]
            children_op.inputs = modified_op_inputs

        return op

    def _resolve_target_name_change(self, output_name: str, op: TargetOp) -> tuple:
        """If the target activation name has been changed,
        resolve such names if possible and accordingly modify the target op object

        Args:
            output_name: The output name of the target op
            op: Object of current target op

        Returns:
            tuple: Resolved output name and modified TargetOp object
        """
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
        # or any of the resolved_names not in target graph
        # 419, 491.nchw, both in dlc, then do not resolve
        # the name for 419.nchw
        return output_name, op

    def _create_target_connected_graph(self) -> None:
        """Creates target connected graph from DLC graph"""
        # TODO: Update the logic of target connected graph, bring parity between
        # framework and target connected graph. for e.g. keys should be op name
        # instead of activation name like we have in framework_connected_graph
        model_reader = modeltools.IrDlcReader()
        model_reader.open(self._dlc_path)
        ir_graph = model_reader.get_ir_graph()

        target_connected_graph = {}

        # Make target_op for inputs
        for idx, inp in enumerate(ir_graph.get_input_tensors_to_graph()):
            name = f"input_{idx}"
            target_op = TargetOp(name)
            target_op.op_type = "input"
            target_op.data_type = inp.data_type().name
            target_op.inputs = []
            target_op.outputs = [inp.name()]
            target_connected_graph[name] = target_op

        for op in ir_graph.get_ops():
            static_tensors = [
                op_input.name()
                for op_input in op.inputs()
                if "IrStaticTensor" in str(type(op_input))
            ]
            all_outputs_names = [output.name() for output in op.outputs()]

            for output in op.outputs():
                target_op = TargetOp(output.name())
                target_op.op_type = op.type
                target_op.data_type = output.data_type().name
                target_op.inputs = [inp.name() for inp in op.inputs()]
                target_op.outputs = all_outputs_names
                target_op.static_tensors = static_tensors
                target_connected_graph[output.name()] = target_op

        for _, node1 in target_connected_graph.items():
            for _, node2 in target_connected_graph.items():
                if any(output in node2.inputs for output in node1.outputs):
                    # node1 -> node2
                    node1.children_ops = [node2]
                    node2.parent_ops = [node1]

        return target_connected_graph
