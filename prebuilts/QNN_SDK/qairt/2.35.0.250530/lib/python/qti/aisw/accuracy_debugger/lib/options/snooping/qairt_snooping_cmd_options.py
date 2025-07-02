# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Architecture_Target_Types, Runtime
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError, UnsupportedError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import get_framework_info
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Framework


class QAIRTSnoopingCmdOptions(CmdOptions):

    def __init__(self, args, type, validate_args=True):
        super().__init__(component="snooping", args=args, engine="QAIRT",
                         validate_args=validate_args)
        self.type = type

    def _base_initialize(self):

        self._required_args = self.parser.add_argument_group('Required Arguments')
        self._converter_args = self.parser.add_argument_group('QAIRT Converter Arguments')
        self._quantizer_args = self.parser.add_argument_group('QAIRT Quantizer Arguments')
        self._net_run_args = self.parser.add_argument_group('Net-run Arguments')
        self._optional_args = self.parser.add_argument_group('Other optional Arguments')
        self._verifier_args = self.parser.add_argument_group('Verifier Arguments')

        self._required_args.add_argument(
            '-r', '--runtime', type=str.lower, required=True,
            choices=[r.value for r in Runtime if r.value not in ['htp']], help='Runtime to be used.\
                Note: In case of SNPE execution(--executor_type snpe), aic runtime is not supported.'
        )

        self._required_args.add_argument(
            '-a', '--architecture', type=str, required=True,
            choices=Architecture_Target_Types.target_types.value,
            help='Name of the architecture to use for inference engine.Note: In case of SNPE\
                execution(--executor_type snpe), aarch64-qnx architecture is not supported.')

        self._required_args.add_argument(
            '-l', '--input_list', type=str, required=True,
            help="Path to the input list text file to run inference(used with net-run). \
            Note: When having multiple entries in text file, in order to save memory and time, \
                you can pass --debug_mode_off to skip intermediate outputs dump.")

        self._converter_args.add_argument('--input_network', '--model_path', dest='model_path',
                                          type=str, default=None, required=False,
                                          help='Path to the model file(s).')

        self._converter_args.add_argument(
            '--input_tensor', nargs='+', action='append', required=False,
            help='The name and dimension of all the input buffers to the network \
            specified in the format [input_name comma-separated-dimensions sample-data data-type] \
            Note: sample-data and data-type are optional \
            for example: \'data\' 1,224,224,3. Note that the quotes should always be included in \
            order to handle special characters, spaces, etc. \
            For multiple inputs, specify multiple --input_tensor on the command line like:  \
            --input_tensor "data1" 1,224,224,3 sample1.raw float32 \
            --input_tensor "data2" 1,50,100,3 sample2.raw int64 \
            NOTE: Required for TensorFlow and PyTorch. Optional for Onnx and Tflite. \
            In case of Onnx, this feature works only with Onnx 1.6.0 and above.')

        self._converter_args.add_argument(
            '--out_tensor_node', '--output_tensor', dest='output_tensor', type=str, required=False,
            action='append', help='Name of the graph\'s output Tensor Names. \
            Multiple output names should be provided separately like: \
            --out_tensor_node out_1 --out_tensor_node out_2 \
            NOTE: Required for TensorFlow. Optional for Onnx, Tflite and PyTorch')

        self._converter_args.add_argument(
            '--io_config', type=str, required=False, default=None,
            help="Use this option to specify a yaml file for input and output options.")

        self._converter_args.add_argument('-qo', '--quantization_overrides', type=str,
                                          required=False, default=None,
                                          help="Path to quantization overrides json file.")

        self._converter_args.add_argument(
            '--converter_float_bitwidth', type=int, required=False, default=None, choices=[32, 16],
            help='Use this option to convert the graph to the specified float \
                    bitwidth, either 32 (default) or 16. \
                    Note: Cannot be used with --calibration_input_list and --quantization_overrides'
        )

        self._converter_args.add_argument(
            '--extra_converter_args', type=str, required=False, default=None,
            help="additional converter arguments in a quoted string. \
                 example: --extra_converter_args 'arg1=value1;arg2=value2'")

        self._quantizer_args.add_argument(
            '--calibration_input_list', type=str, required=False, default=None,
            help='Path to the inputs list text file to run quantization(used with qairt-quantizer).'
        )

        self._quantizer_args.add_argument(
            '-bbw', '--bias_bitwidth', type=int, required=False, default=8, choices=[8, 32],
            help="option to select the bitwidth to use when quantizing the bias. default 8")
        self._quantizer_args.add_argument(
            '-abw', '--act_bitwidth', type=int, required=False, default=8, choices=[8, 16],
            help="option to select the bitwidth to use when quantizing the activations. default 8")
        self._quantizer_args.add_argument(
            '-wbw', '--weights_bitwidth', type=int, required=False, default=8, choices=[8, 4],
            help="option to select the bitwidth to use when quantizing the weights. default 8")

        self._quantizer_args.add_argument(
            '--quantizer_float_bitwidth', type=int, required=False, default=32, choices=[32, 16],
            help='Use this option to select the bitwidth to use for float tensors, \
                either 32 (default) or 16.')

        self._quantizer_args.add_argument(
            '--act_quantizer_calibration', type=str.lower, required=False, default="min-max",
            choices=['min-max', 'sqnr', 'entropy', 'mse', 'percentile'],
            help="Specify which quantization calibration method to use for activations. \
                                    This option has to be paired with --act_quantizer_schema.")

        self._quantizer_args.add_argument(
            '--param_quantizer_calibration', type=str.lower, required=False, default="min-max",
            choices=['min-max', 'sqnr', 'entropy', 'mse', 'percentile'],
            help="Specify which quantization calibration method to use for parameters.\
                                    This option has to be paired with --param_quantizer_schema.")

        self._quantizer_args.add_argument(
            '--act_quantizer_schema', type=str.lower, required=False, default='asymmetric',
            choices=['asymmetric', 'symmetric', 'unsignedsymmetric'],
            help="Specify which quantization schema to use for activations. \
                                    Note: Default is asymmetric.")

        self._quantizer_args.add_argument(
            '--param_quantizer_schema', type=str.lower, required=False, default='asymmetric',
            choices=['asymmetric', 'symmetric', 'unsignedsymmetric'],
            help="Specify which quantization schema to use for parameters. \
                                    Note: Default is asymmetric.")

        self._quantizer_args.add_argument(
            '--percentile_calibration_value', type=float, required=False, default=99.99,
            help="Value must lie between 90 and 100. Default is 99.99")

        self._quantizer_args.add_argument(
            '--use_per_channel_quantization', action="store_true", default=False,
            help="Use per-channel quantization for convolution-based op weights.\
            Note: This will replace built-in model QAT encodings when used for a given weight.")

        self._quantizer_args.add_argument(
            '--use_per_row_quantization', action="store_true", default=False,
            help="Use this option to enable rowwise quantization of Matmul and FullyConnected ops.")

        self._quantizer_args.add_argument(
            '--extra_quantizer_args', type=str, required=False, default=None,
            help="additional quantizer arguments in a quoted string. \
                    example: --extra_quantizer_args 'arg1=value1;arg2=value2'")

        self._net_run_args.add_argument(
            '--perf_profile', type=str.lower, required=False, default="balanced", choices=[
                'low_balanced', 'balanced', 'default', 'high_performance',
                'sustained_high_performance', 'burst', 'low_power_saver', 'power_saver',
                'high_power_saver', 'extreme_power_saver', 'system_settings'
            ],
            help='Specifies perf profile to set. Valid settings are "low_balanced" , "balanced" , \
                "default", "high_performance" ,"sustained_high_performance", "burst", \
                "low_power_saver", "power_saver", "high_power_saver", "extreme_power_saver", \
                and "system_settings". Note: perf_profile argument is now deprecated for \
                HTP backend, user can specify performance profile \
                through backend extension config now.')

        self._net_run_args.add_argument(
            '--profiling_level', type=str.lower, required=False, default=None,
            help='Enables profiling and sets its level. \
            For QNN executor, valid settings are "basic", "detailed" and "client" \
            For SNPE executor, valid settings are "off", "basic", "moderate", "detailed", \
            and "linting". Default is detailed.')

        self._net_run_args.add_argument(
            '--userlogs', type=str.lower, required=False, default=None,
            choices=["warn", "verbose", "info", "error", "fatal"], help="Enable verbose logging. \
            Note: This argument is applicable only when --executor_type snpe")

        self._net_run_args.add_argument(
            '--log_level', type=str.lower, required=False, default=None,
            choices=['error', 'warn', 'info', 'debug', 'verbose'], help="Enable verbose logging. \
            Note: This argument is applicable only when --executor_type qnn")

        self._net_run_args.add_argument(
            '--use_native_output_files', action="store_true", default=False, required=False,
            help="Specifies that the output files will be generated in the data \
                                    type native to the graph. If not specified, output files will \
                                    be generated in floating point.")

        self._net_run_args.add_argument(
            '--extra_runtime_args', type=str, required=False, default=None,
            help="additional net runner arguments in a quoted string. \
                example: --extra_runtime_args 'arg1=value1;arg2=value2'")

        self._optional_args.add_argument(
            '--executor_type', type=str.lower, required=False, default=None,
            choices=['qnn', 'snpe'],
            help='Choose between qnn(qnn-net-run) and snpe(snpe-net-run) execution. \
                If not provided, qnn-net-run will be executed for QAIRT or QNN SDK, \
                or else snpe-net-run will be executed for SNPE SDK.')

        self._optional_args.add_argument('-p', '--engine_path', type=str, required=False,
                                         help="Path to SDK folder.")

        self._optional_args.add_argument(
            '--deviceId', required=False, default=None,
            help='The serial number of the device to use. If not passed, '
            'the first in a list of queried devices will be used for validation.')

        self._optional_args.add_argument('-v', '--verbose', action="store_true", default=False,
                                         help="Set verbose logging at debugger tool level")

        self._optional_args.add_argument(
            '--host_device', type=str, required=False, default='x86',
            choices=['x86', 'x86_64-windows-msvc', 'wos'],
            help='The device that will be running conversion. Set to x86 by default.')

        self._optional_args.add_argument('-w', '--working_dir', type=str, required=False,
            default='working_directory',
            help='Working directory for the {} to store temporary files. '.format(self.component) + \
                'Creates a new directory if the specified working directory does not exist')
        self._optional_args.add_argument(
            '--output_dirname', type=str, required=False, default='<curr_date_time>',
            help=f'output directory name for the {self.component} to store temporary files under \
                <working_dir>/{self.component} Creates a new directory if the specified working \
                directory does not exist')

        self._optional_args.add_argument(
            '--args_config', type=str, required=False,
            help="Path to a config file with arguments. This can be used to feed arguments to "
            "the AccuracyDebugger as an alternative to supplying them on the command line.")

        self._optional_args.add_argument('--remote_server', type=str, required=False, default=None,
                                         help="ip address of remote machine")
        self._optional_args.add_argument('--remote_username', type=str, required=False,
                                         default=None, help="username of remote machine")
        self._optional_args.add_argument('--remote_password', type=str, required=False,
                                         default=None, help="password of remote machine")

        self._optional_args.add_argument(
            '--golden_output_reference_directory', '--golden_dir_for_mapping',
            dest='golden_output_reference_directory', type=str, required=False, default=None,
            help="Optional parameter to indicate the directory of the goldens, \
                it's used for tensor mapping without running model with framework runtime.")

        self._optional_args.add_argument(
            '--disable_offline_prepare', action="store_true", default=False,
            help=f"Use this option to disable offline preparation. \
                                  Note: By default offline preparation will be done for DSP/HTP runtimes."
        )

        self._optional_args.add_argument(
            '--backend_extension_config', type=str, required=False, default=None,
            help="Path to config to be used with qnn-context-binary-generator. \
                Note: This argument is applicable only when --executor_type qnn")

        self._optional_args.add_argument(
            '--context_config_params', type=str, default=None, required=False,
            help="optional context config params in a quoted string. \
                example: --context_config_params 'context_priority=high; cache_compatibility_mode=strict' \
                    Note: This argument is applicable only when --executor_type qnn")

        self._optional_args.add_argument(
            '--graph_config_params', type=str, default=None, required=False,
            help="optional graph config params in a quoted string. \
                example: --graph_config_params 'graph_priority=low; graph_profiling_num_executions=10'"
        )

        self._optional_args.add_argument(
            '--extra_contextbin_args', type=str, required=False, default=None, help=
            "Additional context binary generator arguments in a quoted string(applicable only when \
                --executor_type qnn). example: --extra_contextbin_args 'arg1=value1;arg2=value2'")

        self._optional_args.add_argument('--onnx_custom_op_lib', default=None,
                                         help="path to onnx custom operator library")

        self._optional_args.add_argument(
            '-f', '--framework', nargs='+', type=str.lower, required=False,
            help='Framework type and version, version is optional. '
            'Currently supported frameworks are [' + ', '.join([f.value for f in Framework]) + ']. '
            'For example, tensorflow 2.10.1 ')

    def _verify_update_base_parsed_args(self, parsed_args):

        if parsed_args.golden_output_reference_directory:
            parsed_args.golden_output_reference_directory = get_absolute_path(
                parsed_args.golden_output_reference_directory)
        parsed_args.engine_path = get_absolute_path(parsed_args.engine_path)

        if parsed_args.framework is None:
            framework_name = get_framework_info(parsed_args.model_path)
            if framework_name is None:
                raise ParameterError(
                    "Unable to detect framework type of the given model, please pass --framework option"
                )
            parsed_args.framework = [framework_name]

        parsed_args.framework_version = None
        if len(parsed_args.framework) > 2:
            raise ParameterError("Maximum two arguments required for framework.")
        elif len(parsed_args.framework) == 2:
            parsed_args.framework_version = parsed_args.framework[1]
        parsed_args.framework = parsed_args.framework[0]

        if parsed_args.framework not in ['onnx', 'tensorflow'] and self.type != "oneshot-layerwise":
            raise UnsupportedError(
                "Layerwise snooping supports only onnx and tensorflow frameworks")

        if parsed_args.runtime == Runtime.htp.value and parsed_args.architecture != 'x86_64-linux-clang':
            raise ParameterError("Runtime htp supports only x86_64-linux-clang architecture")

        if parsed_args.input_tensor is not None:
            # get proper input_tensor format
            for tensor in parsed_args.input_tensor:
                if len(tensor) < 3:
                    raise argparse.ArgumentTypeError(
                        "Invalid format for input_tensor, format as "
                        "--input_tensor \"INPUT_NAME\" INPUT_DIM INPUT_DATA.")
                tensor[2] = get_absolute_path(tensor[2])

        return parsed_args

    def get_all_associated_parsers(self):
        parsers_to_be_validated = [self.parser]

        return parsers_to_be_validated
