# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import importlib
import multiprocessing.synchronize
import os
import json
import subprocess
import sys
import multiprocessing
from collections import OrderedDict
import json

from qti.aisw.accuracy_debugger.lib.inference_engine import inference_engine_repository
from qti.aisw.accuracy_debugger.lib.inference_engine.inference_engines.nd_inference_engine import InferenceEngine
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Engine, Runtime, SocTypes, MaxLimits
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import sanitize_output_tensor_files
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.inference_engine.converters.exec_conversion_quantization.nd_exec_qairt_conversion_quantization import ExecuteQAIRTConversion, ExecuteQAIRTQuantization


@inference_engine_repository.register(cls_type=ComponentType.inference_engine, framework=None,
                                      engine=Engine.QAIRT, engine_version="1.0.0")
class QAIRTInferenceEngine(InferenceEngine):

    class DLCData:

        def __init__(self, host_dlc_path, target_model_inputs):

            # dict of input tensor names to input data locations on target device
            self.target_model_inputs = target_model_inputs

            self.host_dlc_path = host_dlc_path
            self.target_dlc_path = None

            self.host_input_list_path = None
            self.target_input_list_path = None

    def __init__(self, context, converter, executor):
        super().__init__(context, converter, executor)
        # Fields from context
        self.engine_type = context.engine
        self.stage = context.stage
        self.engine_path = context.engine_path
        self.sdk_type = context.sdk_type
        self.executor_type = context.executor_type
        self.host_device = context.host_device
        self.target_device = context.target_device
        self.desired_input_shape = context.desired_input_shape
        self.out_tensor_node = context.out_tensor_node
        self.io_config = context.io_config
        self.input_network = context.input_network
        self.quantization_overrides = context.quantization_overrides
        self.converter_float_bitwidth = context.converter_float_bitwidth
        self.extra_converter_args = context.extra_converter_args
        self.extra_quantizer_args = context.extra_quantizer_args
        self.snpe_dlc_utils_package = context.snpe_dlc_utils_package
        self.snpe_dlc_utils = None
        self.host_output_dir = context.output_dir
        self.target_arch = self.executor.target_arch
        self.input_dlc = context.input_dlc
        self.runtime = context.runtime
        self.verbose = context.verbose
        self.input_list = context.input_list
        self.calibration_input_list = context.calibration_input_list
        self.weights_bitwidth = context.weights_bitwidth
        self.bias_bitwidth = context.bias_bitwidth
        self.act_bitwidth = context.act_bitwidth
        self.quantizer_float_bitwidth = context.quantizer_float_bitwidth
        self.act_quantizer_calibration = context.act_quantizer_calibration
        self.param_quantizer_calibration = context.param_quantizer_calibration
        self.act_quantizer_schema = context.act_quantizer_schema
        self.param_quantizer_schema = context.param_quantizer_schema
        self.percentile_calibration_value = context.percentile_calibration_value
        self.use_per_channel_quantization = context.use_per_channel_quantization
        self.use_per_row_quantization = context.use_per_row_quantization
        self.qairt_quantizer_config = context.qairt_quantizer_config
        self.extra_quantizer_args = context.extra_quantizer_args
        self.perf_profile = context.perf_profile
        self.profiling_level = context.profiling_level
        self.userlogs = context.userlogs
        self.log_level = context.log_level
        self.use_native_output_files = context.use_native_output_files
        self.extra_runtime_args = context.extra_runtime_args
        self.debug_mode = context.debug_mode
        self.offline_prepare = context.offline_prepare
        self.add_layer_outputs = context.add_layer_outputs
        self.executable_location = os.path.join(self.engine_path, 'bin', self.target_arch,
                                                self.executor.executable)
        self.float_fallback = context.float_fallback
        if self.float_fallback:
            # skip quantization with calibration input list when float_fallback is enabled
            self.calibration_input_list = None
        self.target_is_host = self.target_device.device in ['x86', 'x86_64-windows-msvc', 'wos']
        self.os_type = "linux"
        if (self.target_arch in ['wos-remote', 'x86_64-windows-msvc', 'wos']):
            self.os_type = "windows"

        # QNN executor specific
        self.context_binary_generator_config = context.context_binary_generator_config
        self.backend_extension_config = context.backend_extension_config
        self.extra_contextbin_args = context.extra_contextbin_args
        self.context_config_params = context.context_config_params
        self.graph_config_params = context.graph_config_params
        self.qnn_target_backend = None
        self.aic_backend_extension_shared_library_path = context.aic_backend_extension_shared_library_path
        self.htp_backend_extension_shared_library_path = context.htp_backend_extension_shared_library_path
        self.gpu_backend_extension_shared_library_path = context.gpu_backend_extension_shared_library_path
        self.qnn_model_dlc_name = 'QnnModelDlc.dll' if self.os_type == 'windows' else 'libQnnModelDlc.so'
        self.qnn_model_dlc_lib = os.path.join(
            self.engine_path, 'lib', 'aarch64-windows-msvc' if self.target_arch
            in ['wos-remote', 'wos'] else self.target_arch, self.qnn_model_dlc_name)
        if not os.path.exists(self.qnn_model_dlc_lib) and self.executor_type == Engine.QNN.value:
            raise InferenceEngineError(
                f'Unable to find {self.qnn_model_dlc_lib.replace(self.engine_path, "")} in the given SDK.'
            )

        config_runtime = 'dsp' if 'dsp' in self.runtime else self.runtime
        if self.target_arch in ["wos-remote", "wos"]:
            self.backend_paths = context.backend_locations[
                self.executor_type]["wos"][config_runtime]
        else:
            self.backend_paths = context.backend_locations[self.executor_type][
                self.target_arch][config_runtime]

        if not os.path.isdir(self.engine_path):
            raise InferenceEngineError('Please pass valid SDK path with --engine_path argument')

        # Working directory and file location constants
        self._INPUT_LIST_DIR = os.path.join(self.host_output_dir, 'input_list_files')
        self._BASE_DLC_PATH = os.path.join(self.host_output_dir, 'base.dlc')

        if self.host_device.device == "x86_64-windows-msvc":
            self.env_variables = context.x86_64_windows_msvc_environment_variables
        elif self.host_device.device == "wos":
            self.env_variables = context.wos_environment_variables
        else:
            self.env_variables = context.environment_variables

        # base dlc and associated info
        self.base_dlc_data = None
        self.base_dlc_info = None

        self.logger = context.logger
        self.host_env = {}

        self._input_paths = None
        self._offline_prepare_error = None

    # -------------------------------------------------- COMMON ------------------------------------------------------------

    def _common_setup(self):
        """Set the base dlc, and create directories for input lists on host and
        target."""
        if self.target_is_host:
            self._TARGET_INPUT_LIST_DIR = self._INPUT_LIST_DIR
            self._TARGET_EXECUTE_DIR = self.host_output_dir
        else:
            self._TARGET_INPUT_LIST_DIR = os.path.join(self.executor.target_path, 'data')
            self._TARGET_EXECUTE_DIR = self.executor.target_path

        # Fetch input file paths present in the input list,
        # these input paths will be used to push data to target before inference
        self._input_paths = self._get_input_paths()

        # TODO remove this call once modules(decomposed) are created for all QNN core tools executions
        # Currently only converter and quantizer modules are available, inside which set_host_environment() is defined already
        self._set_host_environment()

        # Create the base dlc and info
        self._set_base_dlc()

        # Create directory for input list files on host
        os.makedirs(self._INPUT_LIST_DIR, exist_ok=True)

        # Create directory for input list files on target
        code, _, err = self.target_device.make_directory(self._TARGET_INPUT_LIST_DIR)
        if code != 0:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_MKDIR_FAILED")(err))

    def _set_host_environment(self):
        sys.path.insert(0, os.path.join(self.engine_path, "lib", "python"))

        for var in self.env_variables:
            self.env_variables[var] = (self.env_variables[var]).format(
                sdk_tools_root=self.engine_path)

        # set environment variables depending on host device architecture
        if self.host_device.device in ["x86_64-windows-msvc", "wos"]:
            for var in self.env_variables:
                self.host_env[var] = self.env_variables[var] + os.pathsep
        else:
            for var in self.env_variables:
                self.host_env[var] = self.env_variables[var] + os.pathsep + '$' + var
        self.logger.info(f"Host environment: {self.host_env}")

    def _is_quantizer_required(self):
        """
        This function will determine if qairt-quantizer step is really required.
        This function returns False if:
        1. runtime is gpu (GPU does not support quantization)
        2. Converter_float_bitwidth is either 32 or 16
        Otherwise this function will return True
        """
        quantization_step = True
        if self.runtime == Runtime.gpu.value or self.converter_float_bitwidth in [16, 32]:
            quantization_step = False
        elif self.calibration_input_list is None and self.quantization_overrides is None:
            quantization_step = False

        return quantization_step

    def _set_base_dlc(self):
        """Convert the entire model to set the base dlc and extract model
        info."""
        if self.stage == 'source':
            # Execute qairt converter
            conversion_params = {
                "input_network": self.input_network,
                "output_path": self._BASE_DLC_PATH,
                "input_dims": self.desired_input_shape,
                "output_tensors": self.out_tensor_node,
                "quantization_overrides": self.quantization_overrides,
                "converter_float_bitwidth": self.converter_float_bitwidth,
                "io_config": self.io_config,
                "extra_converter_args": self.extra_converter_args
            }
            qairt_converter = ExecuteQAIRTConversion(host_device=self.host_device.device,
                                                     params=conversion_params,
                                                     output_directory=self.host_output_dir,
                                                     logger=self.logger, verbose=self.verbose,
                                                     engine_path=self.engine_path)
            qairt_converter.convert()
        elif self.stage == 'converted' or self.stage == 'quantized':
            self._BASE_DLC_PATH = self.input_dlc

        # quantize the model if it's float dlc ie. from stage source or converted.
        # Additionally check if quantization step is really required (for GPU or FP16/FP32 not required)
        if (self.stage == 'source' or self.stage == 'converted') and self._is_quantizer_required():
            # Execute qairt quantizer
            quantized_dlc_path = self._BASE_DLC_PATH.rsplit('.', 1)[0] + "_quantized.dlc"
            quantization_params = {
                "input_dlc_path": self._BASE_DLC_PATH,
                "output_path": quantized_dlc_path,
                "calibration_input_list": self.calibration_input_list,
                "weights_bitwidth": self.weights_bitwidth,
                "bias_bitwidth": self.bias_bitwidth,
                "act_bitwidth": self.act_bitwidth,
                "param_quantizer_schema": self.param_quantizer_schema,
                "act_quantizer_schema": self.act_quantizer_schema,
                "param_quantizer_calibration": self.param_quantizer_calibration,
                "act_quantizer_calibration": self.act_quantizer_calibration,
                "percentile_calibration_value": self.percentile_calibration_value,
                "use_per_row_quantization": self.use_per_row_quantization,
                "use_per_channel_quantization": self.use_per_channel_quantization,
                "float_fallback": self.float_fallback,
                "quantizer_float_bitwidth": self.quantizer_float_bitwidth,
                "extra_quantizer_args": self.extra_quantizer_args
            }
            qairt_quantizer = ExecuteQAIRTQuantization(host_device=self.host_device.device,
                                                       params=quantization_params,
                                                       output_directory=self.host_output_dir,
                                                       logger=self.logger, verbose=self.verbose,
                                                       engine_path=self.engine_path)
            qairt_quantizer.quantize()
            self._BASE_DLC_PATH = quantized_dlc_path

        # Set base dlc data object
        self.base_dlc_data = self.DLCData(host_dlc_path=self._BASE_DLC_PATH,
                                          target_model_inputs=None)

        # Extract info from initial dlc
        sys.path.insert(0, os.path.join(self.engine_path, self.snpe_dlc_utils_package))
        self.snpe_dlc_utils = importlib.import_module('snpe_dlc_utils')
        dlc_info = self.snpe_dlc_utils.ModelInfo(self.base_dlc_data.host_dlc_path)
        if len(dlc_info.graph_names) > 1:
            # If incase in future multiple graphs are created then raise an exception to catch this scenario
            raise InferenceEngineError(
                f"Multiple graphs found: {dlc_info.graph_names}, this scenario is not handled yet.")

        l_graph_names = list(dlc_info.graph_names)
        self.base_dlc_info, _, _, _, _, _, _ = dlc_info.extract_model_info(l_graph_names[0])


    def _get_input_paths(self):
        """
        Return file paths present in the given input_list
        """

        if not self.input_list: return

        # input_paths will store all input paths in form of list of lists;
        # ie. if a input list has 2 batch and each batch require 3 inputs then the input_paths would look like:
        # [[batch1_input1,batch1_input2,batch1_input3],[batch2_input1,batch2_input2,batch2_input3]]
        with open(self.input_list, "r") as input_list:
            input_paths = [line.strip(' \n').split(' ') for line in input_list.readlines()]

        return input_paths


    # -------------------------------------------------- COMMON HELPERS ----------------------------------------------------

    def _write_input_list(self, dlc_data, input_list_dir) -> str:
        """Create an input list on the host device.

        :param dlc_data: DLC data object
        :param input_list_dir: Directory to place input list files
        :return file_path: path to the input_list.txt on the host device
        """

        # Set input list file name
        file_path = os.path.join(input_list_dir, 'input_list.txt')

        string_to_write = ''

        for batch in dlc_data.target_model_inputs:
            for items in batch:
                string_to_write += items + ' '
            string_to_write += '\n'

        QAIRTInferenceEngine._write(file_path, string_to_write)

        return file_path

    @staticmethod
    def _write(host_input_list_path, string_to_write):
        with open(host_input_list_path, 'w+') as f:
            # print("####stringtowrite#####: {}".format(string_to_write))
            f.write(string_to_write)

    def _filter_nodes(self, dlc_path, nodes):
        """
        Returns filtered nodes list which are present in the irgraph
        """
        from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import get_tensor_names_from_dlc

        dlc_tensor_names = get_tensor_names_from_dlc(dlc_path, sanitize_names=False)
        filtered_nodes = []
        for node in nodes:
            if node in dlc_tensor_names:
                filtered_nodes.append(node)
            else:
                self.logger.warning(f"Skipping dumping of {node} as it is not present in the irgraph")
        self.logger.debug(f"Filtered list of nodes to dump: {filtered_nodes}")
        return filtered_nodes

    def snpe_offline_preparation(self, dlc_file):
        """
        Runs SNPE offline preparation for given dlc file and returns prepared dlc
        """
        output_dlc_file = dlc_file.replace('.dlc', '_offline.dlc')
        log_string = 'Starting offline graph preparation with: ' + \
                     'Input dlc: ' + str(dlc_file) + ' ' + \
                     'Output dlc: ' + str(output_dlc_file)
        self.logger.info(log_string)

        if self.runtime in SocTypes.RuntimeSocMap.value.keys():
            soc_id = SocTypes.RuntimeSocMap.value[self.runtime]
        else:
            raise InferenceEngineError(f'Unable to find SOC id for {self.runtime} runtime.')

        graph_prepare_command = f'snpe-dlc-graph-prepare --input_dlc {dlc_file} \
            --output_dlc {output_dlc_file} --htp_socs {soc_id}'

        if self.add_layer_outputs:
            add_layer_outputs = ','.join(self.add_layer_outputs)
            graph_prepare_command += f' --set_output_tensors {add_layer_outputs}'

        with open(os.path.join(self.host_output_dir, 'commands_executed.log'), 'a') as f:
            f.write(f'Offline graph preparation command: {graph_prepare_command}\n')

        try:
            code, _, err = self.host_device.execute(commands=[graph_prepare_command],
                                                    env=self.host_env)
            if code != 0:
                self._offline_prepare_error = str(err)
                raise InferenceEngineError('Offline graph preparation failed.')

            self.logger.info('Offline graph prepared successfully')
        except subprocess.CalledProcessError as exc:
            raise InferenceEngineError('Offline graph preparation failed.')

        return output_dlc_file

    def _execute_snpe_inference(self, netrun_lock: multiprocessing.synchronize.Lock = None):
        """Runs inference on SNPE executor.

        :param dlc_data: DLC data object
        """
        dlc_data = self.DLCData(self.base_dlc_data.host_dlc_path, None)

        if self.offline_prepare:
            host_dlc_path = self.snpe_offline_preparation(dlc_data.host_dlc_path)
        else:
            host_dlc_path = dlc_data.host_dlc_path

        # accquire netrun_lock, if available
        if netrun_lock:
            netrun_lock.acquire()
            self.logger.debug(f"Lock acquired for IE run {os.path.basename(self.host_output_dir)}")

        # Handle input data and files
        self._handle_model_inputs(dlc_data)

        # Push libs
        self._push_libs()

        dlc_data.target_dlc_path = self._push_file(host_dlc_path)
        self._push_file(self.executable_location)

        net_run_env = self.host_env.copy()
        # TODO: Fix the target arch name to arm64x-windows once libs and bins are shipped in this arch
        if self.target_device.device in ["x86", "x86_64-windows-msvc", "wos", "wos-remote"]:
            net_run_env['LD_LIBRARY_PATH'] = os.path.join(
                self.engine_path, "lib", 'aarch64-windows-msvc'
                if self.target_arch in ['wos-remote', 'wos'] else self.target_arch)
        elif self.target_arch == 'aarch64-android':
            for library_path_name, path in self.executor.get_execute_environment_variables():
                net_run_env[library_path_name] = path

        # add_layer_outputs takes precedence over debug mode, so disable debug mode when add_layer_outputs is used,
        # Unlike QNN context-bin, snpe graph prepare does not have --enable_intermediate outputs, so only option to
        # to dump all intermediate outputs is snpe-net-run
        debug_mode = False if self.add_layer_outputs else self.debug_mode
        add_layer_outputs = None

        # If set_output_tensors has been consumed at snpe-dlc-graph, no need to feed to net-run
        if self.add_layer_outputs and not self.offline_prepare:
            # Some of the nodes present in add_layer_outputs might not be present in the actual IR graph and
            # snpe-net-run errors out when non-existing nodes in irgraph are passed, this behavior is unlike
            # snpe-dlc-graph-prepare or qnn-net-run or qnn-context-bin which will just skip non-existing nodes
            # so for snp-net-run need to filter add_layer_outputs to remove nodes which won't be there in the graph
            add_layer_outputs = self._filter_nodes(host_dlc_path, self.add_layer_outputs)

            # create comma separted string and append graph name which is base
            add_layer_outputs = ','.join(add_layer_outputs)
            add_layer_outputs = f'"base {add_layer_outputs}"'

        execute_command = self.executor.build_execute_command(
            dlc_data.target_dlc_path, dlc_data.target_input_list_path, self.userlogs,
            perf_profile=self.perf_profile, profiling_level=self.profiling_level,
            extra_runtime_args=self.extra_runtime_args, debug_mode=debug_mode,
            add_layer_outputs=add_layer_outputs, use_native_output_files=self.use_native_output_files)

        log_string = 'Using inference command: ' + str(execute_command)
        self.logger.debug(log_string)

        with open(os.path.join(self.host_output_dir, 'commands_executed.log'), 'a') as f:
            f.write(f'Inference command: {execute_command}\n\n')

        try:
            code, _, err = self.target_device.execute(
                commands=[execute_command], cwd=self._TARGET_EXECUTE_DIR,
                env=None if self.target_arch == "wos-remote" else net_run_env)
            if code != 0:
                if netrun_lock:
                    netrun_lock.release()
                    self.logger.debug(f"Lock released for IE run {os.path.basename(self.host_output_dir)}")
                raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED'))
            self.logger.info(
                get_progress_message('PROGRESS_INFERENCE_ENGINE_GENERATED_INTERMEDIATE_TENSORS')(
                    self.engine_type))
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            # Release netrun_lock, if available
            if netrun_lock:
                netrun_lock.release()
                self.logger.debug(f"Lock released for IE run {os.path.basename(self.host_output_dir)}")
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED'))

        if not self.target_is_host:
            # Pull results
            self._pull_inference_results()

        # Release netrun_lock, if available
        if netrun_lock:
            netrun_lock.release()
            self.logger.debug(f"Lock released for IE run {os.path.basename(self.host_output_dir)}")

    def _pull_inference_results(self):
        """Pull inference results from target device to host."""
        code, out, err = self.target_device.pull(os.path.join(self._TARGET_EXECUTE_DIR, "output"),
                                                 self.host_output_dir)
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_PULL_RESULTS_FAILED")(err_msg))
        self.logger.debug('Pulled inference results successfully')

        if self.target_arch == "wos-remote":
            for remote_path in self.target_device.list_directory(
                    os.path.join(self._TARGET_EXECUTE_DIR, "bin/aarch64-windows-msvc")):
                if remote_path.endswith('schematic.bin'):
                    code, out, err = self.target_device.pull(remote_path, self.host_output_dir)
                    if code != 0:
                        err_msg = str(err) if err else str(out)
                        raise InferenceEngineError(
                            get_message("ERROR_INFERENCE_ENGINE_PULL_FILE_FAILED")(err_msg))
                    self.logger.debug('Pull schematic binary file successful')

        code, out, err = self.target_device.remove(target_path=self._TARGET_EXECUTE_DIR)
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(
                get_message('ERROR_INFERENCE_ENGINE_REMOVE_RESULTS_FAILED')(err_msg))

        self.logger.info('Removed inference results from device successfully')

    def get_binaries_to_push(self):
        """Get binaries used to convert/quantize and run a DLC.

        :return: List of binary paths
        """

        binaries_to_push = []
        runtime = 'dsp' if 'dsp' in self.runtime else self.runtime
        dsp_version = self.runtime.replace('dsp', '') if 'dsp' in self.runtime else ''
        dsp_version_upper = dsp_version.replace('v', 'V')
        for src_path in self.backend_paths:
            source_path = src_path.format(
                engine_path=self.engine_path, target_arch='aarch64-windows-msvc'
                if self.target_arch in ["wos-remote", "wos"] else self.target_arch,
                dsp_version=dsp_version, dsp_version_upper=dsp_version_upper)
            target_path = os.path.join(
                self._TARGET_EXECUTE_DIR,
                os.path.basename(src_path).format(dsp_version_upper=dsp_version_upper))
            binaries_to_push.append((source_path, target_path))

            if "hexagon" in source_path and self.target_device.device == "wos" and "dsp" in self.runtime:
                # For wos, hexagon libs are required to be copied to to {sdk_root}/lib/{target_arch}
                arch = 'aarch64-windows-msvc' if self.target_arch in ['wos-remote', 'wos'
                                                                      ] else self.target_arch
                destination_dir = os.path.join(self.engine_path, "lib", arch)
                destination = os.path.join(destination_dir, os.path.basename(source_path))
                code, _, err = self.target_device.push(source_path, destination)
                if code != 0:
                    raise InferenceEngineError(
                        get_message("ERROR_INFERENCE_ENGINE_PUSH_BINARIES_FAILED_DEVICE"))

        return binaries_to_push

    def _push_custom_op_packages(self):
        if self.extra_runtime_args and '--op_packages' in self.extra_runtime_args:
            op_packages_entries = self.extra_runtime_args.split('--op_packages')[-1].strip().split(
                ' ')[0].split(',')
            self.extra_runtime_args = self.extra_runtime_args.split('--op_packages')[0] + \
                ' '.join(self.extra_runtime_args.split('--op_packages')[-1].strip().split(' ')[1:])
            self.extra_runtime_args = self.extra_runtime_args.strip()

            op_package_path = []
            target_op_package_path = []
            interface_provider = []
            target = []
            for i, op_package_entry in enumerate(op_packages_entries):
                op_package_list = op_package_entry.split(':')
                op_package_path.append(op_package_list[0].strip())
                target_op_package_path.append(
                    os.path.join(self._TARGET_EXECUTE_DIR, os.path.basename(op_package_path[i])))
                #include push
                code, _, err = self.target_device.push(op_package_path[i],
                                                       target_op_package_path[i])
                if code != 0:
                    raise InferenceEngineError(
                        get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
                interface_provider.append(op_package_list[1].strip() if len(op_package_list) ==
                                          2 else '')
                target.append(op_package_list[2].strip() if len(op_package_list) == 3 else '')
            self.extra_runtime_args += ' --op_packages ' + ','.join([
                target_op_package_path[i] + ':' + interface_provider[i] + ':' + target[i]
                for i in range(len(target_op_package_path))
            ])

            if self.target_arch == 'wos-remote' and \
                self.extra_contextbin_args and '--op_packages' in self.extra_contextbin_args:
                self.extra_contextbin_args = self.extra_contextbin_args.split('--op_packages')[0] + \
                ' '.join(self.extra_contextbin_args.split('--op_packages')[-1].strip().split(' ')[1:])
                self.extra_contextbin_args = self.extra_contextbin_args.strip()
                self.extra_contextbin_args += ' --op_packages ' + ','.join([
                    target_op_package_path[i] + ':' + interface_provider[i] + ':' + target[i]
                    for i in range(len(target_op_package_path))
                ])

    def _create_remote_directory(self, base_path, dir_path):
        remote_shared_path = os.path.join(base_path, dir_path)
        dir_creator_cmd = f'mkdir -p {dir_path}'
        if not self.target_device.is_path(remote_shared_path):
            code, out, err = self.target_device.execute(commands=[dir_creator_cmd], cwd=base_path)
            if code != 0:
                err_msg = str(err) if err else str(out)
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_CREATE_DIR_FAILED')(err_msg))

    def _push_to_wos(self):
        """
        It pushes required files to remote windows device
        """
        arch = "aarch64-windows-msvc"
        # Make remote working directory
        code, _, err = self.target_device.make_directory(self._TARGET_EXECUTE_DIR)
        if code != 0:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_TARGET_DIR_CREATION_FAILED")(err))

        # Push bin and header folders
        folders_to_push = [
            f"bin/{arch}", "include/QNN", "share/QNN/converter", "share/QNN/converter/jni",
            "share/QNN/converter/jni/windows"
        ]
        for push_folder_name in folders_to_push:
            self._create_remote_directory(self._TARGET_EXECUTE_DIR, push_folder_name)
            local_dir = os.path.join(self.engine_path, push_folder_name)
            for generator_file in os.listdir(local_dir):
                src_path = os.path.join(local_dir, generator_file)
                dst_path = os.path.join(self._TARGET_EXECUTE_DIR, push_folder_name,
                                        os.path.basename(generator_file))
                code, _, err = self.target_device.push(src_path, dst_path)
                if code != 0:
                    raise InferenceEngineError(
                        get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))

        # Push remaining files
        env_setter_path = os.path.join(self.engine_path, 'bin', 'envsetup.ps1')
        dependency_checker_path = os.path.join(self.engine_path, 'bin',
                                               'check-windows-dependency.ps1')
        for t_file in [env_setter_path, dependency_checker_path]:
            # copy envsetup.ps1 and check-windows-dependency.ps1 to bin folder
            dst_path = os.path.join(self._TARGET_EXECUTE_DIR, 'bin', os.path.basename(t_file))
            code, _, err = self.target_device.push(t_file, dst_path)
            if code != 0:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))

    def _push_libs(self):
        """Push libraries to target device."""
        # If target is same as host, do not push anything
        if self.target_is_host:
            return

        # Push libraries to target device
        binary_paths = self.get_binaries_to_push()
        self.logger.info('Pushing libraries')
        for source, target in binary_paths:
            code, _, err = self.target_device.push(source, target)
            if code != 0:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_BINARIES_FAILED_DEVICE"))

        if self.executor_type == Engine.QNN.value:
            self._push_custom_op_packages()

        if self.target_arch == 'wos-remote':
            self._push_to_wos()

    def _push_file(self, data_path):
        """Push given file to target device."""
        if self.target_is_host:
            # Execution will be done on host machine for "x86", "x86_64-windows-msvc", "wos"
            return data_path

        try:
            self.logger.info(f'Pushing {data_path} to target')
            code, _, err = self.target_device.push(data_path, self._TARGET_EXECUTE_DIR)
            if code != 0:
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE"))
            return os.path.join(self._TARGET_EXECUTE_DIR, os.path.basename(data_path))
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE'))

    def _push_input_data(self, target_device):
        """Push user specified input data to target device.

        if target and host device are same then does not push, just need to set the correct path
        """
        # Push input data to target device
        self.logger.info('Pushing input data to target')
        target_model_inputs = []
        # get from input_list.
        # assumes the path is already full correct
        for batch in self._input_paths:
            per_batch_inputs = []
            for inputs in batch:
                if self.target_is_host:
                    device_model_input_path = inputs
                else:
                    if ":=" in inputs:
                        splited_inputs = inputs.split(":=")
                        tensor = splited_inputs[0]
                        data_path = splited_inputs[1]
                    else:
                        data_path = inputs

                    device_model_input_path = os.path.join(self._TARGET_EXECUTE_DIR,
                                                           os.path.basename(data_path))
                    code, _, err = self.target_device.push(data_path, device_model_input_path)
                    if code != 0:
                        raise InferenceEngineError(
                            get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
                    # Update the path after push to become tensor specific path
                    if ":=" in inputs:
                        device_model_input_path = tensor + ":=" + os.path.join(
                            self._TARGET_EXECUTE_DIR, os.path.basename(data_path))

                per_batch_inputs.append(device_model_input_path)
            target_model_inputs.append(per_batch_inputs)
        return target_model_inputs

    def _sanitize_result_folder(self):
        """
        Iteratively go through all result folders in output directory and sanitize
        output raw file names.
        """
        from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import get_tensor_names_from_dlc

        dlc_tensor_names = get_tensor_names_from_dlc(self.base_dlc_data.host_dlc_path,
                                                     sanitize_names=False)

        output_folder = os.path.join(self.host_output_dir, "output")
        for item in os.listdir(output_folder):
            item_full_path = os.path.join(output_folder, item)
            if os.path.isdir(item_full_path) and item.startswith("Result_"):
                status = sanitize_output_tensor_files(item_full_path, dlc_tensor_names)
                if status != 0:
                    raise InferenceEngineError(
                        "Failed to sanitize output folder {}".format(item_full_path))

    @staticmethod
    def build_backend_extension_config(output_file_path, config_file_path, shared_library_path,
                                       context_config_params=None, graph_config_params=None):
        """Utility method to help building the backend extension config json
        file by providing the .json file.

        :param output_file_path: path to backend extension file.
        :param config_file_path: path to backend specific compile/execute parameters .json
        :param shared_library_path: Backend extensions shared library path
        :param context_config_params: Optional params given in context_configs
        :param graph_config_params: Optional params given in graph_configs
        :return: If .json file not provided, returning None, else return
                 output_file_path
        """
        if not config_file_path:
            return None

        config = {
            "backend_extensions": {
                "shared_library_path": shared_library_path,
                "config_file_path": config_file_path
            }
        }

        if context_config_params:
            config["context_configs"] = context_config_params

        if graph_config_params:
            config["graph_configs"] = graph_config_params

        with open(output_file_path, "w") as file:
            json.dump(config, file, indent=4)
        return output_file_path

    def get_shared_lib_path(self, arch=None):
        """
        Returns shared library path based on runtime and architecture types
        """
        if arch is None:
            arch = 'aarch64-windows-msvc' if self.target_arch in ['wos-remote', 'wos'
                                                                ] else self.target_arch
        if self.runtime == Runtime.aic.value:
            shared_lib_path = self.aic_backend_extension_shared_library_path[self.os_type].format(
                engine_path=self.engine_path, target_arch=arch)
        elif self.runtime == Runtime.gpu.value:
            shared_lib_path = self.gpu_backend_extension_shared_library_path[self.os_type].format(
                engine_path=self.engine_path, target_arch=arch)
        elif "dsp" in self.runtime:
            shared_lib_path = self.htp_backend_extension_shared_library_path[self.os_type].format(
                engine_path=self.engine_path, target_arch=arch)
        else:
            self.logger.error(
                "--backend_extension_config is supported only for aic and dsp runtimes.")
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_BACKEND_CONFIG_PARSING_FAILED"))
        return shared_lib_path

    def qnn_offline_preparation(self, dlc_file):
        """
        Runs QNN offline preparation for given dlc file and returns context binary
        """
        try:
            backend_key = "backend_location"
            if self.runtime == 'aic':
                backend_key = "aic_backend_location"
            elif self.runtime == 'wos-remote':
                backend_key = "remote_backend_location"
            arch = 'aarch64-windows-msvc' if self.target_arch in ['wos-remote', 'wos'
                                                                  ] else self.target_arch
            backend_file = self.context_binary_generator_config[backend_key].format(
                engine_path=self.engine_path, target_arch=arch)

            context_binary_name = 'qnngraph.serialized' if self.target_arch in [
                "wos-remote", "wos"
            ] else 'qnn_model'
            context_binary_generate_command = [
                self.context_binary_generator_config["executable"],
                self.context_binary_generator_config["arguments"]["dlc_path"], dlc_file,
                self.context_binary_generator_config["arguments"]["backend"], backend_file,
                self.context_binary_generator_config["arguments"]["binary_file"],
                context_binary_name
            ]

            output_dir = os.path.join(self.host_output_dir, 'qnn_model_binaries')
            os.makedirs(output_dir,exist_ok=True)
            context_binary_generate_command.extend(
                [self.context_binary_generator_config["arguments"]["output_dir"], output_dir])

            qnn_model_dlc_lib_for_host = self.qnn_model_dlc_lib
            if self.os_type == 'linux':
                # For linux based os, use lib/x86_64-linux-clang/libQnnModelDlc.so with context-bin-generator
                qnn_model_dlc_lib_for_host = os.path.join(self.engine_path, 'lib',
                                                          'x86_64-linux-clang',
                                                          self.qnn_model_dlc_name)
                if not os.path.exists(qnn_model_dlc_lib_for_host):
                    raise InferenceEngineError(
                        f'Unable to find {qnn_model_dlc_lib_for_host.replace(self.engine_path, "")} in the given SDK.'
                    )

            # --dlc_path necessitates libQnnModelDlc.so as the --model argument.
            context_binary_generate_command.extend([
                self.context_binary_generator_config["arguments"]["model_path"],
                qnn_model_dlc_lib_for_host
            ])

            if self.profiling_level:
                context_binary_generate_command.extend([
                    self.context_binary_generator_config["arguments"]["profiling_level"],
                    self.profiling_level
                ])

            # if both debug_mode and add_layer_outputs are enabled, add_layer_outputs takes precedence
            if self.add_layer_outputs:
                add_layer_outputs = ','.join(self.add_layer_outputs)
                context_binary_generate_command.extend([
                    self.context_binary_generator_config["arguments"]["output_tensors"],
                    'base:' + add_layer_outputs
                ])
            elif self.debug_mode:
                context_binary_generate_command.append(
                    self.context_binary_generator_config["arguments"]
                    ["enable_intermediate_outputs"])

            if self.backend_extension_config:
                new_backend_config_path = os.path.join(self.host_output_dir,
                                                       "backend_extension_config.json")
                # For context-bin-generator, we will be using x86_64-linux-clang htp/dsp extension regardless of the given architecture
                shared_lib_path = self.get_shared_lib_path(arch='x86_64-linux-clang')
                self.build_backend_extension_config(new_backend_config_path,
                                                    self.backend_extension_config, shared_lib_path,
                                                    self.context_config_params,
                                                    self.graph_config_params)

                context_binary_generate_command += [
                    self.context_binary_generator_config["arguments"]["config_file"],
                    new_backend_config_path
                ]

            if self.extra_contextbin_args:
                context_binary_generate_command.append(self.extra_contextbin_args)

            context_binary_gen_command_str = ' '.join(context_binary_generate_command)
            self.logger.debug(
                'context bin generator command : {}'.format(context_binary_gen_command_str))

            with open(os.path.join(self.host_output_dir, 'commands_executed.log'), 'a') as f:
                f.write(f'Offline graph preparation command : {context_binary_gen_command_str}\n')

            code, out, err = self.host_device.execute(commands=[context_binary_gen_command_str],
                                                      cwd=output_dir, env=self.host_env)
            if code != 0:
                err_msg = str(err) if err else str(out)
                self._offline_prepare_error = str(err) + " " + str(out)
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED')(err_msg))
            self.logger.info(get_progress_message("PROGRESS_INFERENCE_ENGINE_MODEL_BINARIES"))
            return os.path.join(output_dir, context_binary_name + ".bin")
        except subprocess.CalledProcessError as exc:
            raise InferenceEngineError(
                get_message('ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED')(self.target_arch,
                                                                           str(exc)))

    def _execute_qnn_inference(self, netrun_lock: multiprocessing.synchronize.Lock = None):
        """Runs inference on QNN executor.

        :param dlc_data: DLC data object
        """

        dlc_data = self.DLCData(self.base_dlc_data.host_dlc_path, None)

        if self.offline_prepare:
            host_context_binary = self.qnn_offline_preparation(dlc_data.host_dlc_path)
            dlc_data.target_dlc_path = None
            target_qnn_model_dlc_lib = None

        # Accquire netrun_lock, if available
        if netrun_lock:
            netrun_lock.acquire()
            self.logger.debug(f"Lock aquired for IE run {os.path.basename(self.host_output_dir)}")

        # Handle model input and input files
        self._handle_model_inputs(dlc_data)

        # Push binaries
        self._push_libs()

        if self.offline_prepare:
            target_context_binary = self._push_file(host_context_binary)
        else:
            dlc_data.target_dlc_path = self._push_file(dlc_data.host_dlc_path)
            target_qnn_model_dlc_lib = self._push_file(self.qnn_model_dlc_lib)
            target_context_binary = None

        # Push qbb-net-run executable to target
        self._push_file(self.executable_location)

        # Set device environment
        if not self.target_is_host:
            self.device_env = {}
            for library_path_name, path in self.executor.get_execute_environment_variables():
                self.device_env[library_path_name] = path
        else:
            # For x86, x86_64_windows_msvc and WOS, target device and host device is currently same
            self.device_env = self.host_env

        # if offline_prepare or add_layer_outputs is provided, don't need to pass in debug option to execute command builder as
        # it's handled inside context binary generator
        debug_mode = False if self.add_layer_outputs or self.offline_prepare else self.debug_mode

        # For offline mode, use context binary generated
        # For online mode, use libQnnModelDlc.so along with --dlc_path argument
        model_binary = target_context_binary if self.offline_prepare else target_qnn_model_dlc_lib
        dlc_path = None if self.offline_prepare else dlc_data.target_dlc_path

        # backend path
        if self.target_device.device in ["android", "qnx", "wos-remote"]:
            # self.qnn_target_backend is already pushed to target device in _push_libs()
            self.qnn_target_backend = os.path.basename(self.backend_paths[0])
        else:
            self.qnn_target_backend = self.backend_paths[0].format(
                engine_path=self.engine_path, target_arch='aarch64-windows-msvc'
                if self.target_arch in ["wos-remote", "wos"] else self.target_arch)

        # native flags will be propagated through extra_runtime_args
        use_native_input_files = None

        target_output_dir = os.path.join(self._TARGET_EXECUTE_DIR, "output")
        print_version = None
        target_op_packages = []

        target_backend_extension_config = None
        if self.backend_extension_config:
            target_backend_extension_config = os.path.join(self.host_output_dir,
                                                        "target_backend_extension_config.json")
            shared_lib_path = self._push_file(self.get_shared_lib_path())
            relative_path_config = self._push_file(self.backend_extension_config)
            self.build_backend_extension_config(target_backend_extension_config,
                                                relative_path_config, shared_lib_path,
                                                self.context_config_params,
                                                self.graph_config_params)
            target_backend_extension_config = self._push_file(target_backend_extension_config)


        # Only not pass output_tensors to net-run if offline_prepare stage is False
        if self.add_layer_outputs and not self.offline_prepare:
            add_layer_outputs = ','.join(self.add_layer_outputs)
        else:
            add_layer_outputs = None

        execute_command = self.executor.build_execute_command(
            model_binary, self.qnn_target_backend, dlc_data.target_input_list_path,
            target_op_packages, target_output_dir, use_native_input_files, self.use_native_output_files,
            self.perf_profile, self.profiling_level, debug_mode, self.log_level, print_version,
            target_backend_extension_config, self.extra_runtime_args, add_layer_outputs,
            dlc_path=dlc_path)

        if self.target_arch == "aarch64-qnx":
            execute_command = './' + execute_command

        log_string = 'Using inference command: ' + str(execute_command)
        self.logger.debug(log_string)

        with open(os.path.join(self.host_output_dir, 'commands_executed.log'), 'a') as f:
            f.write(f'Inference command: {execute_command}\n\n')

        try:
            self.logger.info(
                get_progress_message('PROGRESS_INFERENCE_ENGINE_GENERATE_OUTPUTS')(
                    target_output_dir))
            if self.target_arch == 'wos-remote':
                cwd_path = os.path.join(self._TARGET_EXECUTE_DIR, 'bin', 'aarch64-windows-msvc')
                env_info = os.path.join(self._TARGET_EXECUTE_DIR, 'bin')
            else:
                cwd_path = self._TARGET_EXECUTE_DIR
                env_info = self.device_env

            code, out, err = self.target_device.execute(commands=[execute_command], cwd=cwd_path,
                                                        env=env_info)

            if code != 0:
                err_msg = str(err) if err else str(out)
                if netrun_lock:
                    netrun_lock.release()
                    self.logger.debug(f"Lock released for IE run {os.path.basename(self.host_output_dir)}")
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED')(err_msg))
            self.logger.info(
                get_progress_message('PROGRESS_INFERENCE_ENGINE_GENERATED_INTERMEDIATE_TENSORS')(
                    self.engine_type))
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            # Release netrun_lock, if available
            if netrun_lock:
                netrun_lock.release()
                self.logger.debug(f"Lock released for IE run {os.path.basename(self.host_output_dir)}")
            raise InferenceEngineError(
                get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED')(str(exc)))

        if not self.target_is_host:
            # Pull results
            self._pull_inference_results()

        # Release netrun_lock, if available
        if netrun_lock:
            netrun_lock.release()

    def _handle_model_inputs(self, dlc_data) -> None:
        '''
        creates the input_list.txt and pushes the list to the device if
        host_device is not same as target device. Modifies the dlc_data inplace for the following
        fields:
            1. target_dlc_path, if host_device is same as the target_device
            2. target_model_inputs: path to the input data on the target device
            3. host_input_list_path: Path to the input_list.txt on the host device
            4. target_input_list_path: Path to the input_list.txt on the target device
        '''
        # if target device is same as host, no need to push
        if self.target_is_host:
            dlc_data.target_dlc_path = self.base_dlc_data.host_dlc_path

        # Set inputs to the model on the target device
        dlc_data.target_model_inputs = self._push_input_data(self.target_device.device)

        # Create input list
        dlc_data.host_input_list_path = self._write_input_list(dlc_data, self._INPUT_LIST_DIR)

        # Push input list to target and set dlc_data's target input list
        self.logger.info('Pushing input list to target: ' + dlc_data.host_input_list_path)
        input_list_file_dest = os.path.join(self._TARGET_INPUT_LIST_DIR,
                                            os.path.basename(dlc_data.host_input_list_path))
        dlc_data.target_input_list_path = input_list_file_dest
        if not self.target_is_host:
            try:
                code, _, err = self.target_device.push(dlc_data.host_input_list_path,
                                                       input_list_file_dest)
                if code != 0:
                    raise InferenceEngineError(
                        get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
            except subprocess.CalledProcessError as exc:
                self.logger.error(
                    'Error pushing input list to target. Continuing with inference on next layer.')
                self.logger.error(str(exc))

    def _run_inference_engine(self, netrun_lock: multiprocessing.synchronize.Lock = None):
        """
        Executes base dlc container on the target device and pulls results
        """
        self.logger.debug(f'Dumping outputs with use_native_output_files={self.use_native_output_files}')

        def execute_inference(subset):
            self.add_layer_outputs = subset
            # Execute inference
            try:
                # set the _offline_prepare_error as None
                self._offline_prepare_error = None
                if self.executor_type == Engine.SNPE.value:
                    self._execute_snpe_inference(netrun_lock)
                elif self.executor_type == Engine.QNN.value:
                    self._execute_qnn_inference(netrun_lock)
            except InferenceEngineError as exc:
                self.logger.error(
                    'Error executing inference. Continuing with inference on next layer.')
                self.logger.error(str(exc))
                # Fallback, if offline prepare failed due to memory issue
                if self._offline_prepare_error and 'Serialization error memory usage too large'\
                    in self._offline_prepare_error:
                    self.logger.debug('Falling back to smaller subset of --set_output_tensor')
                    execute_inference(subset[:len(subset)//2])
                    execute_inference(subset[len(subset)//2:])
                else:
                    self._offline_prepare_error = None

        # If runtime is cpu, we can directly pass --debug to net-run as memory not a conern for cpu
        if self.runtime == 'cpu':
            execute_inference(self.add_layer_outputs)
        else:
            from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import get_dlc_size, \
            get_intermediate_tensors_size_from_dlc, divide_output_tensors

            # Get the size of the DLC file
            dlc_size = get_dlc_size(self.base_dlc_data.host_dlc_path)

            # Get the size of intermediate tensors from the DLC file
            intermediate_tensors_size_dict = get_intermediate_tensors_size_from_dlc(
                self.base_dlc_data.host_dlc_path)

            # Filter intermediate tensors to include only those specified in add_layer_outputs
            intermediate_tensors_size_dict = {
                output_tensor: intermediate_tensors_size_dict.get(output_tensor, 0)
                for output_tensor in self.add_layer_outputs
            } if self.add_layer_outputs else intermediate_tensors_size_dict

            # Calculate the total size of intermediate tensors
            intermediate_tensors_size = sum(intermediate_tensors_size_dict.values())

            # Check if the total size (DLC size + intermediate tensors size) exceeds the maximum allowed model size
            is_size_greater_than_max = dlc_size + intermediate_tensors_size > MaxLimits.max_model_size_with_intermediates.value

            # Calculate the allowed size for outputs (remaining space after considering DLC size)
            allowed_size_for_outputs = MaxLimits.max_model_size_with_intermediates.value - dlc_size

            # If add_layer_outputs or debug_mode is enabled and the total size exceeds the maximum allowed size,
            # divide the output tensors into subsets to fit within the allowed size and run execute inference for each subset
            if (self.add_layer_outputs
                    or self.debug_mode) and is_size_greater_than_max and allowed_size_for_outputs > 0:
                subsets_output_tensors = divide_output_tensors(intermediate_tensors_size_dict,
                                                            allowed_size_for_outputs)
                for subset in subsets_output_tensors:
                    execute_inference(subset)

            else:
                # Otherwise, execute inference as usual
                execute_inference(self.add_layer_outputs)

        if self.executor_type == Engine.SNPE.value:
            self._sanitize_result_folder()

        if self.target_arch == 'wos-remote' or self.target_arch == 'aarch64-qnx':
            self.target_device.close()

    def run(self, netrun_lock: multiprocessing.synchronize.Lock = None):
        self._common_setup()
        self._run_inference_engine(netrun_lock)

    def get_graph_structure(self):
        if self.base_dlc_info is None:
            self.logger.info('Converting model to obtain graph structure.')
            self._set_base_dlc()
        graph_list_structure = [(layer.name, [
            layer.type,
            dict(
                zip([santize_node_name(input_name)
                     for input_name in layer.input_names], layer.input_dims)),
            dict(
                zip([santize_node_name(output_name)
                     for output_name in layer.output_names], layer.output_dims_list))
        ]) for layer in self.base_dlc_info]
        return OrderedDict(graph_list_structure)
