# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import sys
import json
import shutil
import multiprocessing

import numpy as np

import signal
from abc import abstractmethod

from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.runner.qairt_ie_runner import RequiredArgs,\
    ConverterArgs, QuantizerArgs, NetrunArgs, OptionalArgs, execute_on_qairt
from qti.aisw.accuracy_debugger.lib.runner.exec_framework_runner import \
    trigger_framework_runner
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name, remove_file
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json
from qti.aisw.accuracy_debugger.lib.utils.snooper_utils import SnooperUtils, files_to_compareV2,\
    ActivationStatus, ActivationInfo, dump_csv
from qti.aisw.accuracy_debugger.lib.utils.graph_utils import get_common_parent_activations,\
    get_supergroup_activations, get_subgraph, validate_inputs_outputs, get_topological_order
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import MATH_INVARIANT_OPS, SnooperStage
from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import get_irgraph_tensors_info, \
    permute_tensor_data_axis_order
from qti.aisw.accuracy_debugger.lib.encodings_converter.qairt_encodings_converter import \
    QairtEncodingsConverter
from qti.aisw.accuracy_debugger.lib.visualizer.nd_visualizers import Visualizers


def signal_handler(sig, _):
    QAIRTSnooper.logger.info('Stopping snooping on user request.')
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


class QAIRTSnooper:
    '''
    Base class for qairt snooping algorithms
    '''

    logger = None

    def __init__(self, snooping_type, args, logger, verbose="info"):

        self._snooping_type = snooping_type
        self._logger = logger if logger else setup_logger(verbose, args.output_dir)
        QAIRTSnooper.logger = self._logger
        self._args = args

        self._framework_activation_info = None
        self._target_activation_info = None
        self._permute_data = None

        self._resolved_target_activations = None
        self._framework_activation_op_map = None
        self._target_activation_op_map = None

        self._activation_status = {}
        self._all_subgraphs = {}

        self._verifier_scores = {}
        self._columns = None
        self._data_frame = None
        self._comparators_obj = None
        self._comparators_names = None
        self._csv_path = None

        self._qairt_encodings_converter = None
        self._debug_graph_activations = None
        self._debug_graph_input_names = None
        self._debug_graph_output_names = None
        self._supergroup_activations = set()
        self._net_run_lock = multiprocessing.Lock()

    def _initialize(self) -> None:
        '''
        Initializes the snooping class variables

        :raise Exception: If snooping stage not in [source, verification]
        '''

        # Execute the framework diagnosis
        if self._args.golden_output_reference_directory is None:
            framework_results_dir = self._trigger_framework_runner()
            self._args.golden_output_reference_directory = framework_results_dir

        snooper_utility = SnooperUtils(self._args)
        self._comparators_obj = snooper_utility.getComparator()
        self._comparators_names = []
        for comparator in self._comparators_obj:
            if comparator.V_NAME == 'l1error' and comparator.mean:
                self._comparators_names.append('mae')
            else:
                self._comparators_names.append(comparator.V_NAME)

        self._data_frame, self._columns = self._data_frame_initialize()

        if self._args.stage.lower() == SnooperStage.SOURCE.value:
            self._create_qairt_override()
            self._qairt_encodings_converter = self._get_encodings_converter()
            self._framework_activation_op_map = self._qairt_encodings_converter.get_framework_activation_op_map(
            )
            self._resolved_target_activations = self._qairt_encodings_converter.get_resolved_target_activation(
            )
            self._target_activation_op_map = self._qairt_encodings_converter.get_target_activation_op_map(
            )
            # check whether user provided debug_subgraph_inputs and debug_subgraph_outputs are part of model or not
            self._logger.debug(f"Validating user provided debug subgraph inputs and outputs")
            validate_inputs_outputs(self._args.debug_subgraph_inputs,
                                    self._args.debug_subgraph_outputs,
                                    self._framework_activation_op_map)
            self._supergroup_activations = get_supergroup_activations(
                self._framework_activation_op_map, self._target_activation_op_map)
            self._all_subgraphs['ignore_activations'] = ','.join(self._supergroup_activations)
            self._logger.debug(f"Generating debug subgraph")
            self._debug_graph_activations, self._debug_graph_input_names, \
                self._debug_graph_output_names = self._get_debug_graph()
            self._all_subgraphs['debug_graph_input_tensors'] = ','.join(
                self._debug_graph_input_names)
            self._all_subgraphs['debug_graph_output_tensors'] = ','.join(
                self._debug_graph_output_names)
            self._all_subgraphs['debug_graph_activations'] = ','.join(self._debug_graph_activations)
            self._all_subgraphs['subgraphs'] = {}
            self._args.snooper_artifact_dir = self._args.output_dir

        elif self._args.stage.lower() == SnooperStage.VERIFICATION.value:
            tensor_mapping_path = os.path.join(self._args.snooper_artifact_dir,
                                               'encodings_converter', 'tensor_mapping.json')
            self._resolved_target_activations = read_json(tensor_mapping_path)
            all_subgraphs_path = os.path.join(self._args.snooper_artifact_dir, 'all_subgraphs.json')
            self._all_subgraphs = read_json(all_subgraphs_path)

        else:
            raise Exception(f"snooping stage: {self._args.stage} not supoorted.")

        self._framework_activation_info, self._target_activation_info, self._permute_data = self._get_profile_info(
        )

    def _data_frame_initialize(self) -> tuple[dict, list]:
        '''
        Initialize result csv related data structures.

        :return data_frame: empty data_frame to store the results
        :return columns: list of columns in data_frame
        '''

        data_frame = {}
        columns = [
            "Source Name", "Target Name", "STATUS", "Layer Type", "Framework Shape", "Target Shape",
            "Framework(Min, Max, Median)", "Target(Min, Max, Median)"
        ]

        for comp in self._comparators_names:
            columns.append(f"{comp}(current_layer)")
            for original_output in self._args.output_tensor:
                columns.append(f"{comp}({original_output})")
        columns.append("INFO")

        data_frame = {col: [] for col in columns}

        return data_frame, columns

    @abstractmethod
    def run(self) -> None:
        pass

    def _get_debug_graph(self) -> tuple[list, set, set]:
        '''
        creates target subgraph which will be used for debugging
        based on user provided debug_subgraph_inputs and
        debug_subgraph_outputs

        :return debug_graph_activations: list of topological sorted activations
            present in target debug graph such that each activation is also present
            in the framework graph. input_names are not part of debug_graph_activations
        :return debug_graph_inputs: set of debug graph input names
        :return debug_graph_outputs: set of debug graph output names
        :raises Exception: if debug subgraph turns out to be empty
        '''
        # TODO: find better name for debug_subgraph_inputs, debug_subgraph_outputs

        debug_framework_graph_inputs = set(self._args.debug_subgraph_inputs)
        debug_framework_graph_outputs = set(self._args.debug_subgraph_outputs)
        for activation_name, framework_op in self._framework_activation_op_map.items():
            if not self._args.debug_subgraph_inputs and framework_op.get_op_type() == "input":
                debug_framework_graph_inputs.update([activation_name])

            if not self._args.debug_subgraph_outputs and not framework_op.get_children_ops():
                debug_framework_graph_outputs.update([activation_name])

        debug_graph_input_names = set()
        for input_name in debug_framework_graph_inputs:
            partial_inputs = get_common_parent_activations(input_name,
                                                           self._framework_activation_op_map,
                                                           self._target_activation_op_map,
                                                           self._supergroup_activations)
            debug_graph_input_names.update(partial_inputs)

        debug_graph_output_names = set()
        for output_name in debug_framework_graph_outputs:
            partial_outputs = get_common_parent_activations(output_name,
                                                            self._framework_activation_op_map,
                                                            self._target_activation_op_map,
                                                            self._supergroup_activations)
            debug_graph_output_names.update(partial_outputs)

        # If user did not specified both debug subgraph inputs and outputs
        if not (self._args.debug_subgraph_inputs or self._args.debug_subgraph_outputs):
            # execute for full target graph exclusing inputs
            target_activations = set(
                self._target_activation_op_map.keys()) - debug_graph_input_names
            # filter out all the target activations which are not part of framework graph like
            # convert ops/ extra target ops added as no point debugging them bcz they do not
            # have corresponding framework ops.
            target_activations = target_activations.intersection(
                self._framework_activation_op_map.keys())
            # filter out all intermdiate supergroup activations as they should not be debugged
            debug_graph_activations = target_activations - self._supergroup_activations
            visited_debug_graph_inputs = debug_graph_input_names
            visited_debug_graph_outputs = debug_graph_output_names
        else:
            # user passed either of the debug subgraph inputs or outputs
            debug_graph_activations, visited_debug_graph_inputs, visited_debug_graph_outputs = get_subgraph(
                debug_graph_input_names, debug_graph_output_names, self._target_activation_op_map,
                self._framework_activation_op_map, self._supergroup_activations)

        # topological sort the debug_graph_activations
        target_topological_sort = get_topological_order(self._target_activation_op_map)
        debug_graph_topological_activations = [
            activation for activation in target_topological_sort
            if activation in debug_graph_activations
        ]

        if not debug_graph_activations:
            raise Exception(
                f"Please re-check debug_subgraph_inputs and debug_subgraph_outputs as debugging graph is empty"
            )

        self._logger.debug(
            f"Debugging Graph: {str(debug_graph_topological_activations)} with Inputs: {str(debug_graph_input_names)} and Outputs: {str(debug_graph_output_names)}"
        )

        return debug_graph_topological_activations, visited_debug_graph_inputs, visited_debug_graph_outputs

    def _create_subgraph_quantization_override(self, subgraph: set,
                                               subgraph_output_names: list) -> str:
        '''
        creates quantization overrides file for the given subgraph intermediate tensor names.

        :param subgraph: set of subgraph intermediate tensor names
        :param subgraph_output_names: list of subgraph otuput names
        :return subgraph_override_file_path: path to the subgraph override file
        '''
        subgraph_encodings = self._qairt_encodings_converter.create_subgraph_quantization_overrides(
            subgraph, self._supergroup_activations)

        subgraph_output_names = map(santize_node_name, subgraph_output_names)
        subgraph_override_dir = os.path.join(self._args.output_dir,
                                             'sub_graph_node_precision_files')

        if self._args.compulsory_override:
            compulsory_override  =read_json(self._args.compulsory_override)
            for key, value in compulsory_override.items():
                for tensor_name, enc in value.items():
                    subgraph_encodings[key][tensor_name] = enc

        os.makedirs(subgraph_override_dir, exist_ok=True)
        file_name = '#'.join(sorted(subgraph_output_names)) + ".json"
        subgraph_override_file_path = os.path.join(subgraph_override_dir, file_name)
        with open(subgraph_override_file_path, 'w') as file:
            json.dump(subgraph_encodings, file, indent=4)

        return subgraph_override_file_path

    def _handle_qairt_failure(self, subgraph_output_name: str,
                              subgraph_ie_artifact_dir: str) -> None:
        '''
        Determine the status of subgraph execution.

        :param subgraph_output_name: subgraph output name
        :param subgraph_ie_artifact_dir: path to the subgraph inference engine artifacts
        '''
        base_dlc_path = os.path.join(subgraph_ie_artifact_dir, "base.dlc")
        if not os.path.exists(base_dlc_path):
            self._all_subgraphs['subgraphs'][subgraph_output_name][
                'status'] = ActivationStatus.CONVERTER_FAILURE
            self._all_subgraphs['subgraphs'][subgraph_output_name]['status_msg'] = ""
            self._activation_status[subgraph_output_name].set_status(
                ActivationStatus.CONVERTER_FAILURE, "")
            return

        base_quantized_dlc_path = os.path.join(subgraph_ie_artifact_dir, "base_quantized.dlc")
        if not os.path.exists(base_quantized_dlc_path):
            self._all_subgraphs['subgraphs'][subgraph_output_name][
                'status'] = ActivationStatus.QUANTIZER_FAILURE
            self._all_subgraphs['subgraphs'][subgraph_output_name]['status_msg'] = ""
            self._activation_status[subgraph_output_name].set_status(
                ActivationStatus.QUANTIZER_FAILURE, "")
            return

        if self._args.executor_type.lower() == "snpe":
            base_quantized_offline_dlc_path = os.path.join(subgraph_ie_artifact_dir,
                                                           "base_quantized_offline.dlc")
            if not os.path.exists(base_quantized_offline_dlc_path):
                self._all_subgraphs['subgraphs'][subgraph_output_name][
                    'status'] = ActivationStatus.SNPE_DLC_GRAPH_PREPARE_FAILURE
                self._all_subgraphs['subgraphs'][subgraph_output_name]['status_msg'] = ""
                self._activation_status[subgraph_output_name].set_status(
                    ActivationStatus.SNPE_DLC_GRAPH_PREPARE_FAILURE, "")
                return
        else:
            qnn_model_bin_path = os.path.join(subgraph_ie_artifact_dir, "qnn_model_binaries",
                                              "qnn_model.bin")
            if not os.path.exists(qnn_model_bin_path):
                self._all_subgraphs['subgraphs'][subgraph_output_name][
                    'status'] = ActivationStatus.QNN_CONTEXT_BINARY_FAILURE
                self._all_subgraphs['subgraphs'][subgraph_output_name]['status_msg'] = ""
                self._activation_status[subgraph_output_name].set_status(
                    ActivationStatus.QNN_CONTEXT_BINARY_FAILURE, "")
                return

    def _handle_memory_efficient(self, subgraph_ie_artifact_dir: str) -> None:
        '''
        Delete the converter, quantizer, dlc-graph-prepare/context-bin-binaries
        present in the subgraph inference directory

        :param subgraph_ie_artifact_dir: path to the subgraph inference engine artifacts
        :raise Exception: If executor type not in [qnn, snpe]
        '''
        if self._args.memory_efficient:
            files_to_remove = ["base.dlc", "base_quantized.dlc"]

            if self._args.executor_type.lower() == "snpe":
                files_to_remove.append("base_quantized_offline.dlc")
            elif self._args.executor_type.lower() == "qnn":
                files_to_remove.append(os.path.join("qnn_model_binaries", "qnn_model.bin"))
            else:
                raise Exception(f'Executor type: {self._args.executor_type} not supported.')

            for file in files_to_remove:
                path = os.path.join(subgraph_ie_artifact_dir, file)
                remove_file(path)

    def _get_profile_info(self) -> tuple[dict, dict, dict]:
        """
        Reads profile_info.json from framework runner and
        permute_data to populate framework and target related
        tensor informations.

        :return framework_activation_info: dictionary of sanitized_framework_activations
            as key and object of ActivationInfo as value
        :return target_activation_info: dictionary of sanitized_target_activations
            as key and object of ActivationInfo as value
        :return permute_data: dictionary of target axis data
        """
        # for each framework activation, create ActivationInfo object
        profile_info_file_path = os.path.join(self._args.golden_output_reference_directory,
                                              'profile_info.json')
        profile_info = read_json(profile_info_file_path)
        framework_activation_info = {}
        for sanitized_activation_name, value in profile_info.items():
            activation_info = ActivationInfo(value[0], value[1], tuple(value[2:]))
            framework_activation_info[sanitized_activation_name] = activation_info

        # Create(if needed, incase of source stage) and read the permute_data file for target
        permute_file_path = os.path.join(self._args.snooper_artifact_dir, "permute_data.json")
        if not os.path.exists(permute_file_path):
            dlc_path = os.path.join(self._args.output_dir, "inference_engine", "initial_run",
                                    "base.dlc")
            permute_data = get_irgraph_tensors_info(dlc_path=dlc_path,
                                                    output_dir=self._args.output_dir)
            base_file_path = os.path.join(self._args.output_dir, 'base.json')
            shutil.copy(base_file_path, permute_file_path)
            remove_file(base_file_path)
        permute_data = read_json(permute_file_path)

        # for target activations, populate dimensions
        target_activation_info = {}
        for sanitized_activation_name, value in permute_data.items():
            activation_info = ActivationInfo(None, value['dims'], None)
            target_activation_info[sanitized_activation_name] = activation_info

        return framework_activation_info, target_activation_info, permute_data

    def _build_inference_engine_process(self, working_dir: str, output_dirname: str,
                                  calib_input_list: list = None, add_layer_outputs: list = None,
                                  intermediate_outputs: bool = False, float_fallback: bool = False,
                                  quantization_overrides: str = None, runtime: str = None,
                                  architecture: str = None, tensor_mapping: bool = False,
                                  converter_float_bitwidth: int = None,
                                  quantizer_float_bitwidth: int = 32,
                                  make_symlink: bool = False) -> None:
        """
        Executes the given model on qairt platform.

        :param calib_input_list: path to the calibration_input_list.txt
        :param add_layer_outputs: list of output tensor names which are to be
            dumped
        :param output_dirname: output directory name
        :param intermediate_outputs: whether to dump all the intermediate
            tensors or not
        :param float_fallback: boolean flag to enable the float_fallback
        :param quantization_overrides: external override file to the converter
        :param runtime: target device on which the execution will happen
        :param architecture: target device type on which execution will happen
        :param working_dir: path to working directory
        :param converter_float_bitwidth: converter float bitwidth for the converter
        :param quantizer_float_bitwidth: quantizer float bitwidth for the quantizer
        """

        required_class_args = {
            'runtime': runtime or self._args.runtime,
            'architecture': architecture or self._args.architecture,
            'input_list': self._args.input_list
        }
        required_args = RequiredArgs(**required_class_args)

        converter_class_args = {
            'model_path': self._args.model_path,
            'input_tensor': self._args.input_tensor,
            'output_tensor': self._args.output_tensor,
            'io_config': self._args.io_config,
            'quantization_overrides': quantization_overrides,
            'converter_float_bitwidth': converter_float_bitwidth,
            'extra_converter_args': self._args.extra_converter_args
        }
        converter_args = ConverterArgs(**converter_class_args)

        quantizer_class_args = {
            'calibration_input_list': calib_input_list,
            'bias_bitwidth': self._args.bias_bitwidth,
            'act_bitwidth': self._args.act_bitwidth,
            'weights_bitwidth': self._args.weights_bitwidth,
            'quantizer_float_bitwidth': quantizer_float_bitwidth,
            'act_quantizer_calibration': self._args.act_quantizer_calibration,
            'param_quantizer_calibration': self._args.param_quantizer_calibration,
            'act_quantizer_schema': self._args.act_quantizer_schema,
            'param_quantizer_schema': self._args.param_quantizer_schema,
            'percentile_calibration_value': self._args.percentile_calibration_value,
            'use_per_channel_quantization': self._args.use_per_channel_quantization,
            'use_per_row_quantization': self._args.use_per_row_quantization,
            'float_fallback': float_fallback,
            'extra_quantizer_args': self._args.extra_quantizer_args
        }
        quantizer_args = QuantizerArgs(**quantizer_class_args)

        netrun_class_args = {
            'perf_profile': self._args.perf_profile,
            'profiling_level': self._args.profiling_level,
            'userlogs': self._args.userlogs,
            'log_level': self._args.log_level,
            'use_native_output_files': self._args.use_native_output_files,
            'extra_runtime_args': self._args.extra_runtime_args
        }
        netrun_args = NetrunArgs(**netrun_class_args)

        optional_class_args = {
            'executor_type': self._args.executor_type,
            'engine_path': self._args.engine_path,
            'deviceId': self._args.deviceId,
            'verbose': self._args.verbose,
            'host_device': self._args.host_device,
            'working_dir': working_dir,
            'output_dirname': output_dirname,
            'debug_mode_off': not intermediate_outputs,
            'args_config': self._args.args_config,
            'remote_server': self._args.remote_server,
            'remote_username': self._args.remote_username,
            'remote_password': self._args.remote_password,
            'disable_offline_prepare': self._args.disable_offline_prepare,
            'backend_extension_config': self._args.backend_extension_config,
            'context_config_params': self._args.context_config_params,
            'graph_config_params': self._args.graph_config_params,
            'extra_contextbin_args': self._args.extra_contextbin_args,
            'add_layer_outputs': add_layer_outputs
        }
        optional_args = OptionalArgs(**optional_class_args)

        process = multiprocessing.Process(target=execute_on_qairt, args=(required_args, converter_args, quantizer_args, netrun_args, optional_args, self._logger, tensor_mapping, self._net_run_lock, make_symlink))

        return process

    def _get_encodings_converter(self) -> QairtEncodingsConverter:
        '''
        Creates object of QairtEncodingsConverter

        :return qairt_encodings_converter: object of QairtEncodingsConverter
        '''

        working_dir = os.path.join(self._args.output_dir, "encodings_converter")

        # create object of QairtEncodingsConverter and create AIMET enc file
        qairt_encodings_converter = QairtEncodingsConverter(self._args.model_path,
                                                            self._args.quantized_dlc_path,
                                                            self._args.quantization_overrides,
                                                            working_dir, self._logger)
        converted_encodings = qairt_encodings_converter.create_subgraph_quantization_overrides()
        converted_encodings_file_path = os.path.join(working_dir, "converted_encodings.json")
        with open(converted_encodings_file_path, 'w') as file:
            json.dump(converted_encodings, file, indent=4)

        return qairt_encodings_converter

    def _create_qairt_override(self) -> None:
        """
        This function is used to quantize the model in the beginning to generate the encoding file
        Once the encoding file is generated, it is filtered to convert into aimet style encoding.
        """

        if not self._args.quantized_dlc_path:
            # user have either override along with data
            # or only data. override can not be qairt-dumped.
            # run qairt-converter and quantizer to dump the end file
            float_fallback = False if self._args.calibration_input_list else True
            process = self._build_inference_engine_process(
                calib_input_list=self._args.calibration_input_list, output_dirname="initial_run",
                float_fallback=float_fallback, working_dir=self._args.output_dir,
                quantization_overrides=self._args.quantization_overrides,
                converter_float_bitwidth=self._args.converter_float_bitwidth,
                quantizer_float_bitwidth=self._args.quantizer_float_bitwidth)
            process.start()
            process.join()

            self._args.quantized_dlc_path = os.path.join(self._args.output_dir, "inference_engine",
                                                         "initial_run", "base_quantized.dlc")
            self._args.quantization_overrides = os.path.join(self._args.output_dir,
                                                             "inference_engine", "initial_run",
                                                             "base_quantized_encoding.json")

    def _trigger_framework_runner(self) -> str:
        '''
        Runs the framework runner for the given model and input

        :return framework_results_path: path to the framework results
        '''
        framework_args = {
            'model_path': self._args.model_path,
            'input_tensor': self._args.input_tensor,
            'output_tensor': self._args.output_tensor,
            'working_dir': self._args.working_dir,
            'verbose': self._args.verbose,
            'disable_graph_optimization': True,
            'onnx_custom_op_lib': self._args.onnx_custom_op_lib,
            'use_native_output_files': self._args.use_native_output_files
        }
        framework_results_path = trigger_framework_runner(**framework_args)

        return framework_results_path

    def _should_be_skipped(self, subgraph_activations: set,
                           subgraph_output_name: str) -> tuple[bool, str]:
        '''
        Finds out whether a subgraph should be skipped or not.

        :param subsubgraph_activationsgraph: set of activation in subgraph
        :param subgraph_output_name: subgraph final output name
        :return should_skip: True if current framework activation should be
            skipped from debugging
        :return op_type: framework op type associated with the activation
        '''
        target_op = self._target_activation_op_map[subgraph_output_name]
        target_op_type = target_op.get_op_type().lower()

        self._logger.debug(f"Activation: {subgraph_output_name},  Target op_type: {target_op_type}")

        # If all ops in the target subgraph is of type MATH_INVARIENT,
        # then skip
        subgraph_op_types = set()
        for activation_name in subgraph_activations:
            target_op = self._target_activation_op_map[activation_name]
            target_op_type = target_op.get_op_type().lower()
            subgraph_op_types.update([target_op_type])

        if not subgraph_op_types.difference(MATH_INVARIANT_OPS):
            return True, "MATH_INVARIENT"

        return False, ''

    def _build_data_frame_for_subgraph(self, framework_activation_name: str) -> None:
        '''
        Build the result dataframe for the given activation in the framework graph
        :param framework_activation_name: activation name in the framework graph
        '''

        sanitized_framework_activation = santize_node_name(framework_activation_name)
        framework_activation_info = self._framework_activation_info[sanitized_framework_activation]

        status = self._activation_status[framework_activation_name].get_status()
        info = self._activation_status[framework_activation_name].get_msg()
        self._data_frame['Source Name'].append(framework_activation_name)
        self._data_frame["STATUS"].append(status)
        self._data_frame["INFO"].append(info)
        self._data_frame["Layer Type"].append(
            self._all_subgraphs['subgraphs'][framework_activation_name]['layer_type'])
        self._data_frame["Framework Shape"].append(framework_activation_info.shape)
        self._data_frame["Framework(Min, Max, Median)"].append(
            framework_activation_info.distribution)

        if status == ActivationStatus.SUCCESS:
            resolved_target_activation = self._resolved_target_activations[
                framework_activation_name]
            sanitized_target_activation = santize_node_name(resolved_target_activation)
            target_activation_info = self._target_activation_info[sanitized_target_activation]
            self._data_frame['Target Name'].append(resolved_target_activation)
            self._data_frame["Target Shape"].append(target_activation_info.shape)
            self._data_frame["Target(Min, Max, Median)"].append(target_activation_info.distribution)

            for comp in self._comparators_names:
                self._data_frame[f"{comp}(current_layer)"].append(
                    self._verifier_scores[framework_activation_name]["self"][comp])
                for original_output in self._args.output_tensor:
                    self._data_frame[f"{comp}({original_output})"].append(
                        self._verifier_scores[framework_activation_name]["original_outputs"]
                        [original_output][comp])
        else:
            self._data_frame['Target Name'].append('-')
            self._data_frame["Target Shape"].append('-')
            self._data_frame["Target(Min, Max, Median)"].append('-')

            for comp in self._comparators_names:
                self._data_frame[f"{comp}(current_layer)"].append('nan')
                for original_output in self._args.output_tensor:
                    self._data_frame[f"{comp}({original_output})"].append('nan')

    def _build_data_frame_for_all_subgraphs(self) -> None:
        '''
        Builds the result data frame for each subgraph's final output tensor
        '''
        for framework_activation_name in self._activation_status:
            self._build_data_frame_for_subgraph(framework_activation_name)

    def _compute_verification_score(self, reference_raw_path: str, target_raw_path: str,
                                    framework_activation_name: str, permute_data: dict) -> dict:
        '''
        Computes the verifier score between two given tensors.

        :param reference_raw_path: path to framework activation's raw file
        :param target_raw_path: path to target activation's raw file
        :param framework_activation_name: framework activation name for which verifier score needs
            to be calculated
        :param permute_data: dictionary of permute order and dimension as key for the target tensor
        :return verifier_score: verifier score between two tensors
        '''
        resolved_target_activation = self._resolved_target_activations[framework_activation_name]
        sanitized_framework_activation = santize_node_name(framework_activation_name)
        sanitized_target_activation = santize_node_name(resolved_target_activation)
        data_type = self._framework_activation_info[sanitized_framework_activation].dtype
        target_raw, reference_raw = files_to_compareV2(reference_raw_path, target_raw_path,
                                                       data_type, self._logger)
        target_min = np.min(target_raw)
        target_max = np.max(target_raw)
        target_median = np.median(target_raw)
        self._target_activation_info[sanitized_target_activation].distribution = (target_min,
                                                                                  target_max,
                                                                                  target_median)

        verifier_scores = {comparator_name: 'nan' for comparator_name in self._comparators_names}

        # Permute target tensor to align with golden tensor
        if permute_data is not None and sanitized_target_activation in permute_data:
            try:
                target_raw, _ = permute_tensor_data_axis_order(
                    target_raw, permute_data[sanitized_target_activation])
            except Exception as e:
                self._logger.warning(f"Permute axis failed for tensor: {framework_activation_name} with error: {e}")
                return verifier_scores

        if (target_raw is not None) and (reference_raw is not None):

            for comparator_obj, comparator_name in zip(self._comparators_obj,
                                                       self._comparators_names):
                try:
                    _, verifier_score = comparator_obj.verify(None, None, [reference_raw],
                                                              [target_raw], False)
                except Exception as e:
                    self._logger.warning(f'Skipping comparision for node : {framework_activation_name} due to error: {e}')
                    verifier_score = 'nan'
                # store percentage match for each user supplied comparator
                if isinstance(verifier_score, str) and verifier_score == 'SAME':
                    verifier_score = 100.0
                verifier_scores[comparator_name] = str(verifier_score)

        return verifier_scores

    def _run_verification_for_subgraph(self, framework_activation_name: str,
                                       target_run_result_dir: str) -> None:
        '''
        Calculates the verifier score for the given subgraph's final output and model's final output between
        target and framework tensors.

        :param framework_activation_name: activation_name in the framework graph
        :param target_run_result_dir: path to the subgraph's inference run directory
        '''
        resolved_target_activation = self._resolved_target_activations[framework_activation_name]
        sanitized_framework_activation = santize_node_name(framework_activation_name)
        sanitized_target_activation = santize_node_name(resolved_target_activation)

        if self._activation_status[framework_activation_name].get_status(
        ) == ActivationStatus.SO_FAR_SO_GOOD:
            # Check if net-run has run properly and dumped the outputs as expected
            output_tensor_dir = os.path.join(target_run_result_dir, "output", "Result_0")
            if not os.path.exists(output_tensor_dir):
                status = ActivationStatus.QNN_NET_RUN_FAILURE if self._args.executor_type == "qnn" else ActivationStatus.SNPE_NET_RUN_FAILURE
                self._all_subgraphs['subgraphs'][framework_activation_name]['status'] = status
                self._all_subgraphs['subgraphs'][framework_activation_name]['status_msg'] = ""
                self._activation_status[framework_activation_name].set_status(status, "")
                return

            # First compute verification for the intermediate nodes
            reference_raw_path = os.path.join(self._args.golden_output_reference_directory,
                                              sanitized_framework_activation + ".raw")
            target_raw_path = os.path.join(target_run_result_dir, "output", "Result_0",
                                           sanitized_target_activation + ".raw")
            intermediate_node_verifier_scores = self._compute_verification_score(
                reference_raw_path, target_raw_path, framework_activation_name, self._permute_data)
            self._verifier_scores[framework_activation_name] = {}
            self._verifier_scores[framework_activation_name][
                "self"] = intermediate_node_verifier_scores

            # Now compute for its original model outputs
            original_outputs_verifier_scores = {}
            for original_output in self._args.output_tensor:
                resolved_target_original_output = self._resolved_target_activations[original_output]
                santized_original_framework_output = santize_node_name(original_output)
                santized_original_target_output = santize_node_name(resolved_target_original_output)
                reference_raw_path = os.path.join(self._args.golden_output_reference_directory,
                                                  santized_original_framework_output + ".raw")
                target_raw_path = os.path.join(target_run_result_dir, "output", "Result_0",
                                               santized_original_target_output + ".raw")
                original_output_verifier_scores = self._compute_verification_score(
                    reference_raw_path, target_raw_path, original_output, self._permute_data)
                original_outputs_verifier_scores[original_output] = original_output_verifier_scores
            self._verifier_scores[framework_activation_name][
                "original_outputs"] = original_outputs_verifier_scores

            self._activation_status[framework_activation_name].set_status(
                ActivationStatus.SUCCESS, "")
            self._all_subgraphs['subgraphs'][framework_activation_name][
                'status'] = ActivationStatus.SUCCESS
            self._all_subgraphs['subgraphs'][framework_activation_name]['status_msg'] = ""

        else:
            # subgraph debugging is failed either at stage [converter, quantizer,
            # snpe-dlc-graph-prepare/context-binary-generator]
            return

    def _run_verification_for_all_subgraphs(self) -> None:
        '''
        Runs verification for all the subgraphs under debugging
        '''
        for framework_activation_name, subgraph_info in self._all_subgraphs['subgraphs'].items():
            # Default status is SO_FAR_SO_GOOD, let run_verification_for_subgraph handle
            # the net_run and success stage update because we have those artifacts.
            # and run_verification_for_subgraph will only work for SO_FAR_SO_GOOD stage
            self._activation_status[framework_activation_name] = ActivationStatus(
                framework_activation_name, msg='verification_stage')

            if subgraph_info['status'] == ActivationStatus.SUCCESS:
                sanitized_activation_name = santize_node_name(framework_activation_name)
                target_run_result_dir = os.path.join(self._args.snooper_artifact_dir,
                                                     "inference_engine", sanitized_activation_name)
                self._run_verification_for_subgraph(
                    framework_activation_name=framework_activation_name,
                    target_run_result_dir=target_run_result_dir)
            else:
                self._activation_status[framework_activation_name].set_status(
                    subgraph_info['status'], subgraph_info['status_msg'])

    def _build_next_ie_process(self, curr_index):
        for i in range(curr_index, len(self._debug_graph_activations)):
            framework_activation_name = self._debug_graph_activations[i]
            if self._activation_status[framework_activation_name].get_status() == ActivationStatus.SO_FAR_SO_GOOD:
                subgraph_override_file_path = self._all_subgraphs['subgraphs'][framework_activation_name]['override_file_path']
                sanitized_framework_activation_name = santize_node_name(framework_activation_name)
                resolved_target_activation = self._resolved_target_activations[
                    framework_activation_name]
                output_tensors = [resolved_target_activation] + self._args.output_tensor

                process = self._build_inference_engine_process(
                    add_layer_outputs=output_tensors,
                    output_dirname=sanitized_framework_activation_name,
                    working_dir=self._args.output_dir, float_fallback=True,
                    quantization_overrides=subgraph_override_file_path,
                    converter_float_bitwidth=self._args.converter_float_bitwidth,
                    quantizer_float_bitwidth=self._args.quantizer_float_bitwidth)

                return process, framework_activation_name, i+1
        return None, None, len(self._debug_graph_activations)

    def _execute_all_sub_graphs(self):
        '''
        Executes all subgraph on qairt engine
        '''
        process_map = {}
        process_activations = []
        curr_index = 0

        # Initialize the process_map
        while True:
            process, framework_activation_name, curr_index = self._build_next_ie_process(curr_index)
            if process:
                process_map[framework_activation_name] = process
                process_activations.append(framework_activation_name)
                if len(process_activations) == self._args.max_parallel_subgraphs:
                    break
            else:
                break

        for framework_activation_name, process in process_map.items():
            self._logger.debug(f"Starting process for activation: {framework_activation_name}")
            process.start()

        self._logger.debug(f"Current Process Queue: {process_activations}")

        while process_activations:
            _process_activations = []
            new_process_activation_queue = []

            for framework_activation_name in process_activations:
                process = process_map[framework_activation_name]
                if not process.is_alive():
                    self._logger.debug(f"Joining process for activation: {framework_activation_name}")
                    # Join the current process as it is not alive
                    process.join()

                    sanitized_framework_activation_name = santize_node_name(framework_activation_name)
                    sub_graph_run_artifact_dir = os.path.join(self._args.output_dir, "inference_engine",
                                                            sanitized_framework_activation_name)
                    self._handle_qairt_failure(framework_activation_name, sub_graph_run_artifact_dir)
                    self._handle_memory_efficient(sub_graph_run_artifact_dir)
                    self._run_verification_for_subgraph(
                        framework_activation_name=framework_activation_name,
                        target_run_result_dir=sub_graph_run_artifact_dir)
                    self._build_data_frame_for_subgraph(
                        framework_activation_name=framework_activation_name)
                    dump_csv(self._data_frame, self._columns, self._csv_path)
                    self._logger.info(
                        f"STATUS for activation {framework_activation_name}: {self._activation_status[framework_activation_name].get_status()}"
                    )

                    # Start new process
                    if curr_index < len(self._debug_graph_activations):
                        try:
                            process, framework_activation_name, curr_index = self._build_next_ie_process(curr_index)
                            process_map[framework_activation_name] = process
                            new_process_activation_queue.append(framework_activation_name)
                            self._logger.debug(f"Starting process for activation: {framework_activation_name}")
                            process.start()
                        except Exception as e:
                            self._logger.debug(f"Process start error: {str(e)}")
                            curr_index += 1
                else:
                    _process_activations.append(framework_activation_name)

            _process_activations.extend(new_process_activation_queue)
            process_activations = _process_activations
            if new_process_activation_queue:
                self._logger.debug(f"New Addition to Process Queue: {new_process_activation_queue}, current index: {curr_index}")
                self._logger.debug(f"Current Process Queue: {process_activations} of length: {len(process_activations)}")

    def _plot_comparator_scores(self) -> None:
        '''
        Plots and dumps scores of all verifiers/comparators for both current layer of each subgraph
        and actual original outputs of the model.
        '''
        comparator_columns = []
        for comparator in self._comparators_names:
            comparator_columns.append(f"{comparator}(current_layer)")
            for output_name in self._args.output_tensor:
                comparator_columns.append(f"{comparator}({output_name})")

        plots_save_dir = os.path.join(self._args.output_dir, 'plots')
        os.makedirs(plots_save_dir, exist_ok=True)
        for column in comparator_columns:
            try:
                self._logger.debug(f'Plotting graph for {column} scores...')
                scores = [float(item) for item in self._data_frame[column]]
                Visualizers.line_plot(x=self._data_frame['Source Name'], y=scores, plot_name=column,
                                      save_dir=plots_save_dir)
            except Exception as e:
                self._logger.warning(f'Plotting graph failed with error: {e}')

        self._logger.info(f'Summary plots saved at {plots_save_dir}')
