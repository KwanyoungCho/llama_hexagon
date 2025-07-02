# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os

from qti.aisw.accuracy_debugger.lib.utils.snooper_utils import ActivationStatus, dump_csv
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import SnooperStage
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import dump_json
from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.qairt_snooper import QAIRTSnooper
from qti.aisw.accuracy_debugger.lib.utils.graph_utils import get_common_parent_activations,\
    get_subgraph


class QAIRTCumulativeLayerwiseSnooping(QAIRTSnooper):
    """Class that runs cumulative layer wise snooping."""

    def __init__(self, args, logger, verbose="info"):
        super().__init__(snooping_type="layerwise", args=args, logger=logger, verbose=verbose)

        self._csv_path = os.path.join(self._args.output_dir, 'cumulative-layerwise.csv')

    def _get_all_cumulative_layerwise_sub_graphs(self) -> None:
        '''
        Each op in the target graph represents a subgraph post optimizations.
        Those optimizations can be backend aware depends upon such implementations
        at the qairt-converter level.
        Maps each op in the target graph to the framework subgraph, subsequently
        finds out the subgraph quantization overrides.
        '''

        model_inputs = [item[0] for item in self._args.input_tensor]
        for activation_name in self._debug_graph_activations:
            input_tensor_names = set()
            target_op = self._target_activation_op_map[activation_name]
            self._activation_status[activation_name] = ActivationStatus(activation_name)

            for input_name in target_op.get_inputs():
                # some input_name can be param
                if input_name in self._target_activation_op_map:
                    common_parent_activations = get_common_parent_activations(
                        input_name, self._target_activation_op_map,
                        self._framework_activation_op_map, self._supergroup_activations)
                    input_tensor_names.update(common_parent_activations)

            if input_tensor_names:
                self._logger.debug("+" * 71)
                self._logger.debug(
                    f"Subgraph Inputs: {str(input_tensor_names)} \n Subgraph Outputs: {str(self._args.output_tensor)}"
                )

                subgraph_output_name = activation_name

                self._logger.debug("Getting target subgraph")
                target_parent_activations, _, _ = get_subgraph(model_inputs,
                                                                  target_op.get_outputs(),
                                                                  self._target_activation_op_map)
                target_subgraph_activations = self._target_activation_op_map.keys() - target_parent_activations
                target_subgraph_activations.update(target_op.get_outputs())

                self._logger.debug("Getting framework subgraph")
                framework_parent_activations, _, _ = get_subgraph(
                    model_inputs, target_op.get_outputs(), self._framework_activation_op_map)
                framework_subgraph_activations = self._framework_activation_op_map.keys() - framework_parent_activations
                framework_subgraph_activations.update(target_op.get_outputs())
                if framework_subgraph_activations is None:
                    framework_subgraph_activations = ['Due to converter optimizations, framework subgraph is not found']

                self._all_subgraphs['subgraphs'][subgraph_output_name] = {
                    "Inputs": ','.join(input_tensor_names),
                    "Outputs": ','.join(self._args.output_tensor),
                    "Target Tensors": ','.join(target_subgraph_activations),
                    "Framework Tensors": ','.join(framework_subgraph_activations),
                    "layer_type":
                    self._target_activation_op_map[subgraph_output_name].get_op_type()
                }

                status, msg = self._should_be_skipped(target_subgraph_activations,
                                                      subgraph_output_name)
                if status:
                    status_msg = f"Skipping target subgraph: {target_subgraph_activations} as it is of type {msg}"
                    self._activation_status[subgraph_output_name].set_status(
                        ActivationStatus.SKIP, status_msg)
                    self._logger.info(status_msg)
                    self._build_data_frame_for_subgraph(
                        framework_activation_name=subgraph_output_name)
                    target_subgraph_activations = set()
                    self._all_subgraphs['subgraphs'][subgraph_output_name][
                        'status'] = ActivationStatus.SKIP
                    self._all_subgraphs['subgraphs'][subgraph_output_name][
                        'status_msg'] = status_msg
                    self._all_subgraphs['subgraphs'][subgraph_output_name][
                        "override_file_path"] = ''

                if target_subgraph_activations:
                    # handle cases like split node
                    # subgraph_last_op ---> (out1, out2)
                    # add all subgraph outputs to the subgraph
                    target_subgraph_activations.update(self._args.output_tensor)

                    self._logger.debug(
                        f"subgraph {target_subgraph_activations}\n starts with {input_tensor_names} and ends with {self._args.output_tensor}")
                    try:
                        self._logger.debug("Creating subgraph override")
                        subgraph_override_file_path = self._create_subgraph_quantization_override(
                            target_subgraph_activations, target_op.get_outputs())
                        self._all_subgraphs['subgraphs'][subgraph_output_name][
                            "override_file_path"] = subgraph_override_file_path
                    except Exception as e:
                        status_msg = ""
                        self._all_subgraphs['subgraphs'][subgraph_output_name][
                            "override_file_path"] = ''
                        self._all_subgraphs['subgraphs'][subgraph_output_name][
                            'status'] = ActivationStatus.CUSTOM_OVERRIDE_GENERATION_FAILURE
                        self._all_subgraphs['subgraphs'][subgraph_output_name][
                            'status_msg'] = status_msg
                        self._activation_status[subgraph_output_name].set_status(
                            ActivationStatus.CUSTOM_OVERRIDE_GENERATION_FAILURE, status_msg)
                        self._build_data_frame_for_subgraph(
                            framework_activation_name=subgraph_output_name)

                self._logger.debug("+" * 70)

    def run(self) -> None:
        """
        Executes the layerwise snooping

        :raise Exception: if stage not in [source, verification]
        """

        self._logger.info('Started Cumulative layerwise snooping')

        self._initialize()

        if self._args.stage.lower() == SnooperStage.SOURCE.value:
            self._get_all_cumulative_layerwise_sub_graphs()
            all_subgraphs_path = os.path.join(self._args.output_dir, 'all_subgraphs.json')
            # Dump all_subgraphs for user to check the identified subgraphs
            dump_json(self._all_subgraphs, all_subgraphs_path)
            self._execute_all_sub_graphs()
            # Re-dump the all_subgraphs to update the 'status' field.
            dump_json(self._all_subgraphs, all_subgraphs_path)

        elif self._args.stage.lower() == SnooperStage.VERIFICATION.value:
            self._run_verification_for_all_subgraphs()
            self._build_data_frame_for_all_subgraphs()
            dump_csv(self._data_frame, self._columns, self._csv_path)
        else:
            raise Exception(f"Snooping does not support {self._args.stage} stage")

        self._plot_comparator_scores()

        self._logger.info("Cumulative Layerwise Snooping finished")
