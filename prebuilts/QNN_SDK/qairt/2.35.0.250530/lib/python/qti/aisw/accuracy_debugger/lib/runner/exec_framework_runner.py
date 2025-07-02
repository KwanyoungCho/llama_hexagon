# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
from logging import Logger

from qti.aisw.accuracy_debugger.lib.runner.component_runner import exec_framework_runner


def trigger_framework_runner(model_path: str, input_tensor: list, output_tensor: list,
                             working_dir: str | None = None, output_dirname: str | None = None,
                             args_config: str | None = None, verbose: bool = False,
                             disable_graph_optimization: bool = False,
                             onnx_custom_op_lib: str | None = None, use_native_output_files=False,
                             add_layer_outputs: list = [],
                             add_layer_types: list = [], skip_layer_types: list = [],
                             skip_layer_outputs: list = [], start_layer: str | None = None,
                             end_layer: str | None = None, framework: str | None = None,
                             logger: Logger = None) -> str:
    '''
    Runs the framework runner for the given model and input

    :param model_path: path to the framework model
    :param input_tensor: list of input tensors. for e.g.
        [[input1, input1_shape, path/to/input1.raw, input1_dtype], [...], ...]
    :param output_tensor: list of output_tensors. for e.g. [out1, out2, ...]
    :param working_dir: path/to/working_directory
    :param output_dirname: name of the output directory, else name would be
        timestamp.
    :param args_config: Path to a config file with arguments.
    :param verbose: True if verbose printing is required
    :param disable_graph_optimization: True if onnx graph has to be optimized, else False
    :param onnx_custom_op_lib: path to onnx custom op lib
    :param use_native_output_files: Dump outputs in native format
    :param add_layer_outputs: list of layer activations for which intermediate outputs has
        to be dumped.
    :param add_layer_types: list of layer types for which intermediate outputs has to be
        dumped
    :param skip_layer_types: list of layer types for which intermediate outputs are not
        requried
    :param skip_layer_outputs: list of layer activations for which intermediate outputs
        are not required
    :param start_layer: save all intermediate layer outputs from provided start layer to
        bottom layer of model
    :param end_layer: save all intermediate layer outputs from top layer to  provided end
        layer of model
    :param framework: framework type: [onnx, tflite, tf]
    :param logger: object of logger.Logger

    :return framework_result_dir: path to the framework results
    '''
    required_args = ['--framework_runner', "--model_path", model_path]
    optional_args = []

    for item in input_tensor:
        required_args += ['--input_tensor', *item]
    for item in output_tensor:
        required_args += ['--output_tensor', item]

    if working_dir:
        optional_args += ['--working_dir', working_dir]
    if output_dirname:
        optional_args += ['--output_dirname', output_dirname]
    if args_config:
        optional_args += ['--args_config', args_config]
    if verbose:
        optional_args += ['--verbose']
    if disable_graph_optimization:
        optional_args += ['--disable_graph_optimization']
    if onnx_custom_op_lib:
        optional_args += ['--onnx_custom_op_lib', onnx_custom_op_lib]
    if use_native_output_files:
        optional_args += ['--use_native_output_files']
    if add_layer_outputs:
        optional_args += ['--add_layer_outputs', ','.join(add_layer_outputs)]
    if add_layer_types:
        optional_args += ['--add_layer_types', ','.join(add_layer_types)]
    if skip_layer_types:
        optional_args += ['--skip_layer_types', ','.join(skip_layer_types)]
    if skip_layer_outputs:
        optional_args += ['--skip_layer_outputs', ','.join(skip_layer_outputs)]
    if start_layer:
        optional_args += ['--start_layer', start_layer]
    if end_layer:
        optional_args += ['--end_layer', end_layer]
    if framework:
        optional_args += ['--framework', framework]

    framework_args = required_args + optional_args
    exec_framework_runner(args=framework_args, logger=logger, validate_args=True)
    framework_result_dir = os.path.join(working_dir, 'framework_runner', 'latest')

    return framework_result_dir
