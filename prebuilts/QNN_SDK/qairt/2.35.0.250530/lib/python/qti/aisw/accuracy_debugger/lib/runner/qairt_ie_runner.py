# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import multiprocessing
from logging import Logger
import multiprocessing.synchronize

from qti.aisw.accuracy_debugger.lib.runner.component_runner import exec_inference_engine


class RequiredArgs:

    def __init__(self, runtime: str, architecture: str, input_list: list) -> None:
        '''
        Initializes RequiredArgs class

        :param runtime: runtime on which model will be executed
        :param architecture: Device architecture for inference
        :param input_list: List of input raw files for inference.
        '''
        self.runtime = runtime
        self.architecture = architecture
        self.input_list = input_list

    def get_command(self) -> list:
        '''
        Builds the command with requried_args.

        :return required_command: list of args and subsequent value.
        '''
        required_command = [
            '--runtime', self.runtime, '--architecture', self.architecture, '--input_list',
            self.input_list
        ]

        return required_command


class ConverterArgs:

    def __init__(self, model_path: str = None, input_tensor: list = [], output_tensor: list = [],
                 io_config: str = None, quantization_overrides: str = None,
                 converter_float_bitwidth: int = None, extra_converter_args: str = None) -> None:
        '''
        Initializes the ConverterArgs class

        :param model_path: path to the framework model
        :param input_tensor: list of model input tensor [['input.1', shape, raw/file/path, dtype],
            ..., [...]]
        :param output_tensor: list of model output tensors
        :param io_config: model input output config file, if user wants to pass input info
            through a file.
        :param quantization_overrides: path to external encodings override file
        :param converter_float_bitwidth: converter float bitwidth
        :param extra_converter_args: extra converter arguments: "args:=value;..."
        '''
        self.model_path = model_path
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.io_config = io_config
        self.quantization_overrides = quantization_overrides
        self.converter_float_bitwidth = converter_float_bitwidth
        self.extra_converter_args = extra_converter_args

    def get_command(self) -> list:
        '''
        Builds converter command list

        :return converter_command: list of converter args and corresponding value
        '''
        converter_command = []

        if self.model_path:
            converter_command.extend(['--model_path', self.model_path])
        for item in self.input_tensor:
            converter_command.extend(['--input_tensor', *item])
        for item in self.output_tensor:
            converter_command.extend(['--output_tensor', item])
        if self.io_config:
            converter_command.extend(['--io_config', self.io_config])
        if self.quantization_overrides:
            converter_command.extend(['--quantization_overrides', self.quantization_overrides])
        if self.converter_float_bitwidth:
            converter_command.extend(
                ['--converter_float_bitwidth',
                 str(self.converter_float_bitwidth)])
        if self.extra_converter_args:
            converter_command.extend(['--extra_converter_args', self.extra_converter_args])

        return converter_command


class QuantizerArgs:

    def __init__(self, input_dlc: str = None, calibration_input_list: str = None,
                 bias_bitwidth: int = None, act_bitwidth: int = None, weights_bitwidth: int = None,
                 quantizer_float_bitwidth: int = None, act_quantizer_calibration: str = None,
                 param_quantizer_calibration: str = None, act_quantizer_schema: str = None,
                 param_quantizer_schema: str = None, percentile_calibration_value: float = None,
                 use_per_channel_quantization: bool = False, use_per_row_quantization: bool = False,
                 float_fallback: bool = True, extra_quantizer_args: str = None) -> None:
        '''
        Initializes the QuantizerArgs class

        :param input_dlc: if stage is converted, pass path to input_dlc
        :param calibration_input_list: path to calibation input list for encodings generation
        :param bias_bitwidth: bias bitwidth
        :param act_bitwidth: activation bitwidth
        :param weights_bitwidth: weights bitwidth
        :param quantizer_float_bitwidth: quantizer float bitwidth
        :param act_quantizer_calibration: activation quantizer calibration
        :param param_quantizer_calibration: parameter quantizer calibration
        :param act_quantizer_schema: activation quantizer schema
        :param param_quantizer_schema: parameter quantizer schema
        :param percentile_calibration_value: If schema is percentile, then percentile
            calibration value
        :param use_per_channel_quantization: True, if conv weights have to be quantized per
            channel
        :param use_per_row_quantization: True, if matmul weights have to be quantized per
            row
        :param float_fallback: True, if float fallback is required.
        :param extra_quantizer_args: extra quantizer args-> "args1=value1; ..."
        '''
        self.input_dlc = input_dlc
        self.calibration_input_list = calibration_input_list
        self.bias_bitwidth = bias_bitwidth
        self.act_bitwidth = act_bitwidth
        self.weights_bitwidth = weights_bitwidth
        self.quantizer_float_bitwidth = quantizer_float_bitwidth
        self.act_quantizer_calibration = act_quantizer_calibration
        self.param_quantizer_calibration = param_quantizer_calibration
        self.act_quantizer_schema = act_quantizer_schema
        self.param_quantizer_schema = param_quantizer_schema
        self.percentile_calibration_value = percentile_calibration_value
        self.use_per_channel_quantization = use_per_channel_quantization
        self.use_per_row_quantization = use_per_row_quantization
        self.float_fallback = float_fallback
        self.extra_quantizer_args = extra_quantizer_args

    def get_command(self) -> list:
        '''
        Builds quantizer related command list

        :return quantizer_command: list of quantizer agrs and corresponding value
        '''
        quantizer_command = []

        if self.input_dlc:
            quantizer_command.extend(['--input_dlc', self.input_dlc])
        if self.calibration_input_list:
            quantizer_command.extend(['--calibration_input_list', self.calibration_input_list])
        if self.bias_bitwidth:
            quantizer_command.extend(['--bias_bitwidth', str(self.bias_bitwidth)])
        if self.act_bitwidth:
            quantizer_command.extend(['--act_bitwidth', str(self.act_bitwidth)])
        if self.weights_bitwidth:
            quantizer_command.extend(['--weights_bitwidth', str(self.weights_bitwidth)])
        if self.quantizer_float_bitwidth:
            quantizer_command.extend(
                ['--quantizer_float_bitwidth',
                 str(self.quantizer_float_bitwidth)])
        if self.act_quantizer_calibration:
            quantizer_command.extend(
                ['--act_quantizer_calibration', self.act_quantizer_calibration])
        if self.param_quantizer_calibration:
            quantizer_command.extend(
                ['--param_quantizer_calibration', self.param_quantizer_calibration])
        if self.act_quantizer_schema:
            quantizer_command.extend(['--act_quantizer_schema', self.act_quantizer_schema])
        if self.param_quantizer_schema:
            quantizer_command.extend(['--param_quantizer_schema', self.param_quantizer_schema])
        if self.percentile_calibration_value:
            quantizer_command.extend(
                ['--percentile_calibration_value',
                 str(self.percentile_calibration_value)])
        if self.use_per_channel_quantization:
            quantizer_command.extend(['--use_per_channel_quantization'])
        if self.use_per_row_quantization:
            quantizer_command.extend(['--use_per_row_quantization'])
        if self.float_fallback:
            quantizer_command.extend(['--float_fallback'])
        if self.extra_quantizer_args:
            quantizer_command.extend(['--extra_quantizer_args', self.extra_quantizer_args])

        return quantizer_command


class NetrunArgs:

    def __init__(self, perf_profile: str = None, profiling_level: str = None, userlogs: str = None,
                 log_level: str = None, use_native_output_files=False, extra_runtime_args: str = None) -> None:
        '''
        Intializes the NetrunArgs class

        :param perf_profile: performance profile for inference
        :param profiling_level: profiling level for inference
        :param userlogs: userlogs for snpe-net-run
        :param log_level: log_level for qnn-net-run
        :param use_native_output_files: use_native_output_files for qnn-net-run
        :param extra_runtime_args: extra runtime args -> "args1:=value1; ..."
        '''
        self.perf_profile = perf_profile
        self.profiling_level = profiling_level
        self.userlogs = userlogs
        self.log_level = log_level
        self.use_native_output_files = use_native_output_files
        self.extra_runtime_args = extra_runtime_args

    def get_command(self) -> list:
        '''
        Builds list of netrun command

        :return netrun_command: list of netrun args and corresponding value
        '''
        netrun_command = []

        if self.perf_profile:
            netrun_command.extend(['--perf_profile', self.perf_profile])
        if self.profiling_level:
            netrun_command.extend(['--profiling_level', self.profiling_level])
        if self.userlogs:
            netrun_command.extend(['--userlogs', self.userlogs])
        if self.log_level:
            netrun_command.extend(['--log_level', self.log_level])
        if self.use_native_output_files:
            netrun_command.extend(['--use_native_output_files'])
        if self.extra_runtime_args:
            netrun_command.extend(['--extra_runtime_args', self.extra_runtime_args])

        return netrun_command


class OptionalArgs:

    def __init__(self, executor_type: str = None, stage: str = None, engine_path: str = None,
                 deviceId: str = None, verbose: bool = False, host_device: str = None,
                 working_dir: str = None, output_dirname: str = None, debug_mode_off: bool = False,
                 args_config: str = None, remote_server: str = None, remote_username: str = None,
                 remote_password: str = None, golden_output_reference_directory: str = None,
                 disable_offline_prepare: bool = False, backend_extension_config: str = None,
                 context_config_params: str = None, graph_config_params: str = None,
                 extra_contextbin_args: str = None, start_layer: str = None, end_layer: str = None,
                 add_layer_outputs: list = None, add_layer_types: list = None,
                 skip_layer_types: list = None, skip_layer_outputs: list = None) -> None:
        '''
        Initializes the OptionalArgs class

        :param executor_type: backend executor type: {snpe, qnn}
        :param stage: stage from which inference has to continue
        :param engine_path: SDK path
        :param deviceId: device id of the target device
        :param verbose: True if logs required
        :param host_device: host device type
        :param working_dir: working directory path
        :param output_dirname: output directory name for the inference
        :param debug_mode_off: If True, intermediate outputs will not be dumped
        :param args_config: If user wants to pass the inference engine args via
            a file.
        :param remote_server: remote server ip address/link
        :param reote_username: remote server login name
        :param remote_password: password for logging into remote server
        :param golden_output_reference_directory: fp32 reference directory path
        :param disable_offline_prepare: If True, offline prepare will be disabled
        :para backend_extension_config: path to backend extension config file
        :param context_config_params: string of context config parameters
        :param graph_config_params: string of graph config parameters
        :param extra_contextbin_args: extra context binary args if any -> "args1:=value1, ..."
        :param start_layer: start_layer from which the intermediate outputs will be dumped
        :param end_layer: end_layer to which the intermediate outputs will be dumped
        :param add_layer_outputs: list of activation names for which intermediate outputs
            has to be dumped
        :param add_layer_types: list of layer types for which intermediate outputs
            has to be dumped
        :param skip_layer_types: list of layer types for which intermedoate outputs
            will not be dumped
        :param skip_layer_outputs: list of layer activations for which intermediate
            outputs will not be dumped
        '''
        self.executor_type = executor_type
        self.stage = stage
        self.engine_path = engine_path
        self.deviceId = deviceId
        self.verbose = verbose
        self.host_device = host_device
        self.working_dir = working_dir
        self.output_dirname = output_dirname
        self.debug_mode_off = debug_mode_off
        self.args_config = args_config
        self.remote_server = remote_server
        self.remote_username = remote_username
        self.remote_password = remote_password
        self.golden_output_reference_directory = golden_output_reference_directory
        self.disable_offline_prepare = disable_offline_prepare
        self.backend_extension_config = backend_extension_config
        self.context_config_params = context_config_params
        self.graph_config_params = graph_config_params
        self.extra_contextbin_args = extra_contextbin_args
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.add_layer_outputs = add_layer_outputs
        self.add_layer_types = add_layer_types
        self.skip_layer_types = skip_layer_types
        self.skip_layer_outputs = skip_layer_outputs

    def get_command(self) -> list:
        '''
        Build list of optional command

        :return optional_command: list of optinal args and corresponding value
        '''
        optional_command = []

        if self.executor_type:
            optional_command.extend(['--executor_type', self.executor_type])
        if self.stage:
            optional_command.extend(['--stage', self.stage])
        if self.engine_path:
            optional_command.extend(['--engine_path', self.engine_path])
        if self.deviceId:
            optional_command.extend(['--deviceId', self.deviceId])
        if self.verbose:
            optional_command.extend(['--verbose'])
        if self.host_device:
            optional_command.extend(['--host_device', self.host_device])
        if self.working_dir:
            optional_command.extend(['--working_dir', self.working_dir])
        if self.output_dirname:
            optional_command.extend(['--output_dirname', self.output_dirname])
        if self.debug_mode_off:
            optional_command.extend(['--debug_mode_off'])
            if self.add_layer_outputs:
                optional_command.extend(['--add_layer_outputs', ','.join(self.add_layer_outputs)])
            if self.add_layer_types:
                optional_command.extend(['--add_layer_types', ','.join(self.add_layer_types)])
        if self.args_config:
            optional_command.extend(['--args_config', self.args_config])
        if self.remote_server:
            optional_command.extend(['--remote_server', self.remote_server])
        if self.remote_username:
            optional_command.extend(['--remote_username', self.remote_username])
        if self.remote_password:
            optional_command.extend(['--remote_password', self.remote_password])
        if self.golden_output_reference_directory:
            optional_command.extend(
                ['--golden_output_reference_directory', self.golden_output_reference_directory])
        if self.disable_offline_prepare:
            optional_command.extend(['--disable_offline_prepare'])
        if self.backend_extension_config:
            optional_command.extend(['--backend_extension_config', self.backend_extension_config])
        if self.context_config_params:
            optional_command.extend(['--context_config_params', self.context_config_params])
        if self.graph_config_params:
            optional_command.extend(['--graph_config_params', self.graph_config_params])
        if self.extra_contextbin_args:
            optional_command.extend(['--extra_contextbin_args', self.extra_contextbin_args])
        if self.start_layer:
            optional_command.extend(['--start_layer', self.start_layer])
        if self.end_layer:
            optional_command.extend(['--end_layer', self.end_layer])
        if self.skip_layer_types:
            optional_command.extend(['--skip_layer_types', ','.join(self.skip_layer_types)])
        if self.skip_layer_outputs:
            optional_command.extend(['--skip_layer_outputs', ','.join(self.skip_layer_outputs)])

        return optional_command


def execute_on_qairt(required_args: RequiredArgs, converter_args: ConverterArgs,
                     quantizer_args: QuantizerArgs, netrun_args: NetrunArgs,
                     optional_args: OptionalArgs, logger: Logger = None,
                     tensor_mapping: bool = False,
                     netrun_lock: multiprocessing.synchronize.Lock = None,
                     make_symlink: bool = True) -> None:
    '''
    executes the qairt inference engine.

    :param required_args: Object of RequiredArgs
    :param converter_args: Object of ConverterArgs
    :param quantizer_args: Object of QuantizerArgs
    :param netrun_args: Object of NetrunArgs
    :param optional_args: Object of OptionalArgs
    :param logger: Object of logging.Logger
    :param tensor_mapping: True, if tensor mapping needs to be generated.
    '''

    inference_engine_command = ['--inference_engine']
    required_command = required_args.get_command()
    converter_command = converter_args.get_command()
    quantizer_command = quantizer_args.get_command()
    netrun_command = netrun_args.get_command()
    optinal_command = optional_args.get_command()

    inference_engine_command.extend((required_command + converter_command + quantizer_command +
                                     netrun_command + optinal_command))

    logger.debug(f"Running exec_inference_engine with command: {str(inference_engine_command)}")

    exec_inference_engine(inference_engine_command,'QAIRT', None, tensor_mapping=tensor_mapping,
                          netrun_lock=netrun_lock, make_symlink=make_symlink)
    #TODO: Need to use Python API call once available

    # exec_inference_engine(args=all_args, engine_type="QAIRT", logger=self._logger,
    #                       validate_args=True)
