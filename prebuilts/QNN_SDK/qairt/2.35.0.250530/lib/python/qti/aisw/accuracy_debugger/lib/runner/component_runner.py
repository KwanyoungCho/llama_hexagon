# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import traceback
import multiprocessing

import pandas as pd
import numpy as np
import csv
import onnx

from logging import Logger

from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_runner import FrameworkRunner
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_get_tensor_mapping import TensorMapper
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_inference_engine_manager import InferenceEngineManager
from qti.aisw.accuracy_debugger.lib.options.acc_debugger_cmd_options import AccDebuggerCmdOptions
from qti.aisw.accuracy_debugger.lib.options.framework_runner_cmd_options import FrameworkRunnerCmdOptions
from qti.aisw.accuracy_debugger.lib.options.inference_engine_cmd_options import InferenceEngineCmdOptions
from qti.aisw.accuracy_debugger.lib.options.qairt_inference_engine_cmd_options import QAIRTInferenceEngineCmdOptions
from qti.aisw.accuracy_debugger.lib.options.verification_cmd_options import VerificationCmdOptions
from qti.aisw.accuracy_debugger.lib.options.compare_encodings_cmd_options import CompareEncodingsCmdOptions
from qti.aisw.accuracy_debugger.lib.options.quant_checker_cmd_options import QuantCheckerCmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name

from qti.aisw.accuracy_debugger.lib.options.tensor_inspection_cmd_options import TensorInspectionCmdOptions

from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine, DebuggingAlgorithm, FrameworkExtension, Framework, ComponentLogCodes
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError, InferenceEngineError, VerifierError
from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace, remove_layer_options
from qti.aisw.accuracy_debugger.lib.utils.nd_symlink import symlink
from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import save_to_file, filter_summary_report
from qti.aisw.accuracy_debugger.lib.verifier.nd_verification import Verification
from qti.aisw.accuracy_debugger.lib.compare_encodings.compare_encodings_runner import CompareEncodingsRunner
from qti.aisw.accuracy_debugger.lib.tensor_inspection.tensor_inspection_runner import TensorInspectionRunner
from qti.aisw.accuracy_debugger.lib.verifier.nd_tensor_inspector import TensorInspector
from qti.aisw.accuracy_debugger.lib.quant_checker import get_generator_cls
from qti.aisw.accuracy_debugger.lib.quant_checker.nd_quant_checker import QuantChecker
from qti.aisw.accuracy_debugger.lib.utils.common import update_model_path
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import simplify_onnx_model
from qti.aisw.accuracy_debugger.lib.visualizer.nd_visualizers import Visualizers


def exec_framework_runner(args, logger=None, validate_args=True):
    framework_args = FrameworkRunnerCmdOptions(args, validate_args).parse()
    if (logger is None):
        logger = setup_logger(framework_args.verbose, framework_args.output_dir,
                              component=ComponentLogCodes.framework_runner.value)

    logger.info(get_progress_message('PROGRESS_FRAMEWORK_STARTING'))

    symlink('latest', framework_args.output_dir, logger)

    try:
        framework_runner = FrameworkRunner(logger, framework_args)
        framework_runner.run()
        logger.info(get_progress_message('PROGRESS_FRAMEWORK_FINISHED'))
    except FrameworkError as e:
        raise FrameworkError("Conversion failed: {}".format(str(e)))
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))


def exec_inference_engine(args, engine_type, logger=None, validate_args=True, tensor_mapping=True, 
                          netrun_lock: multiprocessing.synchronize.Lock = None,
                          make_symlink: bool = True):
    if engine_type in [Engine.QNN.value, Engine.SNPE.value]:
        inference_engine_args = InferenceEngineCmdOptions(engine_type, args, validate_args).parse()
    elif engine_type == Engine.QAIRT.value:
        inference_engine_args = QAIRTInferenceEngineCmdOptions(engine_type, args,
                                                               validate_args).parse()
    else:
        raise InferenceEngineError(
            get_message("ERROR_INFERENCE_ENGINE_ENGINE_NOT_FOUND")(engine_type))

    if (logger is None):
        logger = setup_logger(inference_engine_args.verbose, inference_engine_args.output_dir,
                              component=ComponentLogCodes.inference_engine.value)

    logger.info(get_progress_message('PROGRESS_INFERENCE_ENGINE_STARTING'))

    if make_symlink:
        symlink('latest', inference_engine_args.output_dir, logger)

    try:
        inference_engine_manager = InferenceEngineManager(inference_engine_args, logger=logger)
        inference_engine_manager.run_inference_engine(netrun_lock)

        # Run tensor mapping
        if tensor_mapping:
            get_mapping_arg = Namespace(
                None, golden_outputs_dir=inference_engine_args.golden_output_reference_directory,
                target_outputs_dir=inference_engine_args.output_dir,
                work_dir=inference_engine_args.output_dir,
                engine=inference_engine_args.executor_type if hasattr(
                    inference_engine_args, 'executor_type') else inference_engine_args.engine,
                framework=inference_engine_args.framework, version=None,
                model_path=inference_engine_args.model_path)
            TensorMapper(get_mapping_arg, logger).run()

        logger.info(get_progress_message('PROGRESS_INFERENCE_ENGINE_FINISHED'))
    except InferenceEngineError as e:
        raise InferenceEngineError("Inference failed: {}".format(str(e)))
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))


def exec_verification(args, logger=None, run_tensor_inspection=False, validate_args=True):
    verification_args = VerificationCmdOptions(args, validate_args).parse()
    if (logger is None):
        logger = setup_logger(verification_args.verbose, verification_args.output_dir,
                              component=ComponentLogCodes.verification.value)
    symlink('latest', verification_args.output_dir, logger)

    try:
        logger.info(get_progress_message("PROGRESS_VERIFICATION_STARTING"))
        if not verification_args.tensor_mapping:
            logger.warn(
                "--tensor_mapping is not set, a tensor_mapping will be generated based on user input."
            )
            get_mapping_arg = Namespace(
                None, golden_outputs_dir=verification_args.golden_output_reference_directory,
                target_outputs_dir=verification_args.inference_results,
                work_dir=verification_args.inference_results, engine=verification_args.engine,
                framework=None, version=None, model_path=None)
            verification_args.tensor_mapping = TensorMapper(get_mapping_arg, logger).run()

        verify_results = []
        for verifier in verification_args.verify_types:
            # Splitting with comma to handle cases where verifiers have parameters(Ex: --default_verifier rtolatol,rtolmargin,0.1,atolmargin,0.2)
            verifier = verifier[0].split(',')
            verify_type = verifier[0]
            verifier_configs = verifier[1:]
            verification = Verification(verify_type, logger, verification_args, verifier_configs)
            if verification.has_specific_verifier() and len(verification_args.verify_types) > 1:
                raise VerifierError(get_message('ERROR_VERIFIER_USE_MULTI_VERIFY_AND_CONFIG'))
            verify_result = verification.verify_tensors()
            verify_result = verify_result.drop(columns=['Units', 'Verifier'])
            verify_result = verify_result.rename(columns={'Metric': verify_type})
            verify_results.append(verify_result)

        if run_tensor_inspection:
            # run tensor inspector which plots analysis graphs between golden and target data
            logger.info(get_progress_message('PROGRESS_TENSOR_INSPECTION_STARTING'))
            inspection_results = TensorInspector(logger, verification_args).run()
            logger.info(get_progress_message('PROGRESS_TENSOR_INSPECTION_FINISHED'))

        # if verification_args.verifier_config is None, all tensors use the same verifer. So we can export Summary
        if verification_args.verifier_config == None:
            summary_df = None
            for verify_result in verify_results:
                if summary_df is None:
                    summary_df = verify_result
                    continue

                summary_df = pd.merge(summary_df, verify_result,
                                      on=['Name', 'LayerType', 'Size', 'Tensor_dims'])

            if run_tensor_inspection:
                summary_df = pd.merge(summary_df, inspection_results, on=['Name'])

            try:
                logger.debug(f'Filtering verification summary report...')
                summary_df = filter_summary_report(summary_df, verification_args.inference_results)
            except Exception as e:
                logger.warning(f'Filtering failed with error: {e}')

            # Append Framework names column (if model path is available)
            model_arg, model_path = None, None
            if '--model_path' in list(args):
                model_arg = '--model_path'
            elif '-m' in list(args):
                model_arg = '-m'

            if model_arg:
                model_path = args[args.index(model_arg) + 1]

            if model_path and model_path.endswith('.onnx'):
                model = onnx.load(model_path)
                source_names = []
                for node in model.graph.node:
                    source_names.extend(node.output)
                target_names = summary_df['Name'].to_list()

                source_target_map = {}
                for source_name in source_names:
                    sanitized_source_name = santize_node_name(source_name)
                    if sanitized_source_name in target_names:
                        source_target_map[sanitized_source_name] = source_name

                source_names_ordered = []
                for target_name in target_names:
                    if target_name in source_target_map:
                        source_names_ordered.append(source_target_map[target_name])
                    else:
                        source_names_ordered.append('')

                summary_df.insert(0, 'Source Name', source_names_ordered)
                summary_df = summary_df.rename(columns={'Name': 'Target Name'})

            # Plot verifier scores
            plots_save_dir = os.path.join(verification_args.output_dir, 'plots')
            os.makedirs(plots_save_dir, exist_ok=True)
            verifier_names = [verifier[0] for verifier in verification_args.default_verifier]
            for verifier_name in verifier_names:
                try:
                    logger.debug(f'Plotting graph for {verifier_name} scores...')
                    Visualizers.line_plot(x=summary_df['Name'].values,
                                          y=summary_df[verifier_name].values,
                                          plot_name=verifier_name, save_dir=plots_save_dir)
                except Exception as e:
                    logger.warning(f'Plotting graph failed with error: {e}')

            filename = os.path.join(verification_args.output_dir, Verification.SUMMARY_NAME)
            save_to_file(summary_df, filename)
            logger.info(f'Summary report saved at {filename}.csv')
            logger.info(f'Summary plots saved at {plots_save_dir}')

        logger.info(get_progress_message("PROGRESS_VERIFICATION_FINISHED"))
    except VerifierError as excinfo:
        raise Exception("Verification failed: {}".format(str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


def exec_compare_encodings(args, engine_type, logger=None, validate_args=True):
    compare_encodings_args = CompareEncodingsCmdOptions(args, validate_args).parse()
    if (logger is None):
        logger = setup_logger(compare_encodings_args.verbose, compare_encodings_args.output_dir,
                              component=ComponentLogCodes.compare_encodings.value)

    logger.info('Starting Compare encodings feature...')

    symlink('latest', compare_encodings_args.output_dir, logger)

    try:
        compare_encodings = CompareEncodingsRunner(logger, compare_encodings_args)
        compare_encodings.run(engine_type)
        logger.info('Successfully ran Compare encodings feature!')
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))


def exec_tensor_inspection(args, logger=None, validate_args=True):
    tensor_inspection_args = TensorInspectionCmdOptions(args, validate_args).parse()
    if (logger is None):
        logger = setup_logger(tensor_inspection_args.verbose, tensor_inspection_args.output_dir,
                              component=ComponentLogCodes.tensor_inspection.value)

    logger.info('Starting Tensor Inspection...')

    symlink('latest', tensor_inspection_args.output_dir, logger)

    dtype_map = {
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "uint16": np.uint16,
        "float32": np.float32,
    }

    try:
        # initialize TensorInspectionRunner
        tensor_inspector = TensorInspectionRunner(logger)

        summary = []
        golden_files = os.listdir(tensor_inspection_args.golden_data)
        for file in os.listdir(tensor_inspection_args.target_data):

            if file not in golden_files:
                logger.warning(f"{file} present only in target data path, skipping this file.")
                continue

            if not file.endswith('.raw'):
                logger.warning(f"{file} is not a raw file, skipping this file.")
                continue

            golden_path = os.path.join(tensor_inspection_args.golden_data, file)
            target_path = os.path.join(tensor_inspection_args.target_data, file)
            golden = np.fromfile(golden_path, dtype=dtype_map[tensor_inspection_args.data_type])
            target = np.fromfile(target_path, dtype=dtype_map[tensor_inspection_args.data_type])

            # trigger TensorInspectionRunner on current golden and target tensors
            result = tensor_inspector.run(file, golden, target, tensor_inspection_args.output_dir,
                                          target_encodings=tensor_inspection_args.target_encodings,
                                          verifiers=tensor_inspection_args.verifier)
            summary.append(result)

        # dump summary results to csv file
        csv_path = os.path.join(tensor_inspection_args.output_dir, 'summary.csv')
        verifier_names = [verifier[0].split(',')[0] for verifier in tensor_inspection_args.verifier]
        fields = ['Name'
                  ] + verifier_names + ['golden_min', 'golden_max', 'target_min', 'target_max']
        if tensor_inspection_args.target_encodings:
            fields.extend([
                'calibrated_min', 'calibrated_max', '(target_min-calibrated_min)',
                '(target_max-calibrated_max)'
            ])

        with open(csv_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)

        logger.info('Successfully ran Tensor Inspection feature!')
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))


def exec_wrapper(args, engine_type, logger=None, validate_args=True):
    wrapper_args = AccDebuggerCmdOptions(engine_type, args, validate_args).parse()
    subcomponent = ComponentLogCodes.oneshot_layerwise.value
    if engine_type == Engine.QNN.value:
        if wrapper_args.debugging_algorithm == DebuggingAlgorithm.layerwise.value:
            subcomponent = ComponentLogCodes.layerwise.value
        elif wrapper_args.debugging_algorithm == DebuggingAlgorithm.cumulative_layerwise.value:
            subcomponent = ComponentLogCodes.cumulative_layerwise.value

    if (logger is None):
        logger = setup_logger(wrapper_args.verbose, wrapper_args.output_dir, component=subcomponent)
    symlink('latest', wrapper_args.output_dir, logger)

    # run framework runner if required
    framework_results = None
    if wrapper_args.golden_output_reference_directory:
        # skip framework runner if golden_output_reference_directory is passed
        framework_results = wrapper_args.golden_output_reference_directory
        logger.info("Golden reference directory supplied. Skipping framework runner.")
    elif wrapper_args.framework == Framework.pytorch.value:
        # collect golden reference outputs from inference engine on cpu runtime
        reference_args = list(args)
        # Filter out layer options like '--start_layer', '--end_layer', '--add_layer_outputs', '--add_layer_types',
        # '--skip_layer_outputs', '--skip_layer_types' for framework_runner component
        if engine_type == Engine.QNN.value and \
            (wrapper_args.debugging_algorithm == DebuggingAlgorithm.layerwise.value or \
            wrapper_args.debugging_algorithm == DebuggingAlgorithm.cumulative_layerwise.value):
            reference_args = remove_layer_options(reference_args)
        model_name = "model"
        reference_args.extend(['--model_name', model_name])
        reference_args.extend(['--output_dirname', 'reference_outputs'])
        # replace --runtime arg to avoid ambiguity error
        if '--runtime' in reference_args:
            reference_args[reference_args.index('--runtime')] = '-r'
        if '-r' in reference_args:
            reference_args[reference_args.index('-r') + 1] = 'cpu'
        # replace --engine args to avoid ambiguity error
        if '--engine' in reference_args:
            reference_args[reference_args.index('--engine')] = '-e'
        if '--offline_prepare' in reference_args:
            del reference_args[reference_args.index('--offline_prepare')]
        if '--architecture' in reference_args:
            reference_args[reference_args.index('--architecture') + 1] = "x86_64-linux-clang"
        # runs inference engine
        exec_inference_engine(reference_args, engine_type, logger=logger, validate_args=False)
        framework_results = os.path.join(wrapper_args.working_dir, 'inference_engine',
                                         'reference_outputs', 'output', 'Result_0')
    else:
        framework_args = list(args)
        # Filter out layer options like '--start_layer', '--end_layer', '--add_layer_outputs', '--add_layer_types',
        # '--skip_layer_outputs', '--skip_layer_types' for framework_runner component
        if engine_type == Engine.QNN.value and \
            (wrapper_args.debugging_algorithm == DebuggingAlgorithm.layerwise.value or \
            wrapper_args.debugging_algorithm == DebuggingAlgorithm.cumulative_layerwise.value):
            framework_args = remove_layer_options(framework_args)
        exec_framework_runner(framework_args, logger=logger, validate_args=False)
        framework_results = os.path.join(wrapper_args.working_dir, 'framework_runner', 'latest')

        if "--disable_graph_optimization" not in args and wrapper_args.framework == "onnx":
            optimized_model_path = os.path.join(
                framework_results, "optimized_model" +
                FrameworkExtension.framework_extension_mapping[wrapper_args.framework])
            if os.path.exists(optimized_model_path):
                # Replace given model with simplified/optimized model
                args = update_model_path(args, optimized_model_path)

    if engine_type == Engine.QNN.value and \
            (wrapper_args.debugging_algorithm == DebuggingAlgorithm.layerwise.value or \
            wrapper_args.debugging_algorithm == DebuggingAlgorithm.cumulative_layerwise.value):

        # When golden_output_reference_directory flag is used framework runner will be skipped
        # So, simplify model to avoid extraction failures in LW and CLW when golden_output_reference_directory is passed
        if wrapper_args.golden_output_reference_directory and wrapper_args.framework == 'onnx' and\
                                                        "--quantization_overrides" not in args:
            # Apply onnx simplification to the given model
            custom_op_lib = wrapper_args.onnx_custom_op_lib if hasattr(
                wrapper_args, 'onnx_custom_op_lib') else None
            optimized_model_path = simplify_onnx_model(logger, model_path=wrapper_args.model_path,
                                                       input_tensor=wrapper_args.input_tensor,
                                                       output_dir=wrapper_args.output_dir,
                                                       custom_op_lib=custom_op_lib)

            # Replace given model with simplified model
            args = update_model_path(args, optimized_model_path)
            logger.debug(f"Simplified model located at {optimized_model_path}")

        layerwise_args = list(args)
        layerwise_args.extend(['--golden_output_reference_directory', framework_results])

        # run layerwise snooping
        if wrapper_args.debugging_algorithm == DebuggingAlgorithm.layerwise.value:
            exec_layerwise_snooping(layerwise_args, logger, validate_args=False)
        # run cumulative layerwise snooping
        elif wrapper_args.debugging_algorithm == DebuggingAlgorithm.cumulative_layerwise.value:
            exec_cumulative_layerwise_snooping(layerwise_args, logger, validate_args=False)
        return

    # inference engine args pre-processing
    inference_args = list(args)
    model_name = "model"
    inference_args.extend(
        ['--model_name', model_name, '--golden_output_reference_directory', framework_results])
    # replace --engine args to avoid ambiguity error
    if '--engine' in inference_args:
        inference_args[inference_args.index('--engine')] = '-e'
    # runs inference engine
    exec_inference_engine(inference_args, engine_type, logger=logger, validate_args=False)

    # verification args pre-processing
    verification_args = list(args)
    graph_structure = model_name + '_graph_struct.json'
    graph_structure_path = os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest',
                                        graph_structure)
    verification_args.extend(['--graph_struct', graph_structure_path])

    verification_args.extend([
        '--inference_results',
        os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest', 'output/Result_0')
    ])

    verification_args.extend([
        '--golden_output_reference_directory', framework_results, '--tensor_mapping',
        os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest', 'tensor_mapping.json')
    ])

    if engine_type == Engine.QNN.value:
        qnn_model_net_json = model_name + '_net.json'
        qnn_model_net_json_path = os.path.join(wrapper_args.working_dir, 'inference_engine',
                                               'latest', qnn_model_net_json)
        verification_args.extend(['--qnn_model_json_path', qnn_model_net_json_path])
    elif engine_type in [Engine.QAIRT.value, Engine.SNPE.value]:
        dlc_path = os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest', 'base.dlc')
        verification_args.extend(['--dlc_path', dlc_path])

    # runs verification
    exec_verification(verification_args, logger=logger,
                      run_tensor_inspection=wrapper_args.enable_tensor_inspection,
                      validate_args=False)

    if engine_type == Engine.QNN.value and wrapper_args.debugging_algorithm == DebuggingAlgorithm.modeldissection.value:
        # deep analyzer args pre-processing
        da_param_index = args.index('--deep_analyzer')
        deep_analyzers = args[da_param_index + 1].split(',')
        del args[da_param_index:da_param_index + 2]
        deep_analyzer_args = list(args)
        deep_analyzer_args.extend([
            '--tensor_mapping',
            os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest',
                         'tensor_mapping.json'), '--inference_results',
            os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest',
                         'output/Result_0'), '--graph_struct', graph_structure_path,
            '--framework_results', framework_results, '--result_csv',
            os.path.join(wrapper_args.working_dir, 'verification', 'latest', 'summary.csv')
        ])
        # runs deep analyzers
        for d_analyzer in deep_analyzers:
            exec_deep_analyzer(deep_analyzer_args + ['--deep_analyzer', d_analyzer], logger=logger,
                               validate_args=False)


def exec_deep_analyzer(args, logger=None, validate_args=True):
    da_args = AccuracyDeepAnalyzerCmdOptions(args, validate_args).parse()
    if not os.path.isdir(da_args.output_dir):
        os.makedirs(da_args.output_dir)
    if not logger:
        logger = setup_logger(da_args.verbose, da_args.output_dir)

    symlink('latest', da_args.output_dir, logger)

    try:
        from qti.aisw.accuracy_debugger.lib.deep_analyzer.nd_deep_analyzer import DeepAnalyzer
        from qti.aisw.accuracy_debugger.lib.options.accuracy_deep_analyzer_cmd_options import AccuracyDeepAnalyzerCmdOptions
        from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import DeepAnalyzerError

        if not da_args.tensor_mapping:
            logger.warn(
                "--tensor_mapping is not set, a tensor_mapping will be generated based on user input."
            )
            get_mapping_arg = Namespace(None, framework=da_args.framework,
                                        version=da_args.framework_version,
                                        model_path=da_args.model_path,
                                        output_dir=da_args.inference_results, engine=da_args.engine,
                                        golden_dir_for_mapping=da_args.framework_results)
            da_args.tensor_mapping = TensorMapper(get_mapping_arg, logger).run()
        deep_analyzer = DeepAnalyzer(da_args, logger)
        deep_analyzer.analyze()
        logger.info("Successfully ran deep_analyzer!")
    except DeepAnalyzerError as excinfo:
        raise DeepAnalyzerError("deep analyzer failed: {}".format(str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


def exec_cumulative_layerwise_snooping(args, logger=None, validate_args=True):
    try:
        from qti.aisw.accuracy_debugger.lib.snooping.nd_cumulative_layerwise_snooper import CumulativeLayerwiseSnooping
        from qti.aisw.accuracy_debugger.lib.options.layerwise_snooping_cmd_options import LayerwiseSnoopingCmdOptions
        from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import LayerwiseSnoopingError

        args = LayerwiseSnoopingCmdOptions(args, snooper='cumulative_layerwise',
                                           validate_args=validate_args).parse()
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        if not logger:
            logger = setup_logger(args.verbose, args.output_dir)

        symlink('latest', args.output_dir, logger)

        snooper = CumulativeLayerwiseSnooping(args, logger)
        snooper.run()
        logger.info("Successfully ran cumulative layerwise snooping!")
    except LayerwiseSnoopingError as excinfo:
        raise LayerwiseSnoopingError("Cumulative layerwise snooping failed: {}".format(
            str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


def exec_layerwise_snooping(args, logger=None, validate_args=True):
    try:
        from qti.aisw.accuracy_debugger.lib.snooping.nd_layerwise_snooper import LayerwiseSnooping
        from qti.aisw.accuracy_debugger.lib.options.layerwise_snooping_cmd_options import LayerwiseSnoopingCmdOptions
        from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import LayerwiseSnoopingError

        args = LayerwiseSnoopingCmdOptions(args, snooper='layerwise',
                                           validate_args=validate_args).parse()
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        if not logger:
            logger = setup_logger(args.verbose, args.output_dir)

        symlink('latest', args.output_dir, logger)

        snooper = LayerwiseSnooping(args, logger)
        snooper.run()
        logger.info("Successfully ran layerwise snooping!")
    except LayerwiseSnoopingError as excinfo:
        raise LayerwiseSnoopingError("Layerwise snooping failed: {}".format(str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


def exec_quant_checker(args, engine, logger=None, validate_args=True):
    # runs the quant checker (parsed_args)
    quant_checker_args = QuantCheckerCmdOptions(args, engine, validate_args).parse()
    if (logger is None):
        logger = setup_logger(quant_checker_args.verbose, quant_checker_args.output_dir,
                              component=ComponentLogCodes.quant_checker.value)
    # Create latest symlink
    symlink('latest', quant_checker_args.output_dir, logger)
    if quant_checker_args.golden_output_reference_directory is None:
        #STEP1: Run and dump the intermediate outputs
        # Since framework runner does not support input list text file and it only
        # takes single input, so we need to iteratively run for all inputs in the text file
        if quant_checker_args.input_list:
            if '--output_dirname' not in args:
                args.extend(['--output_dirname', 'abc'])
                output_dirname_index = -1
                original_dirname = ''
            else:
                output_dirname_index = args.index('--output_dirname') + 1
                original_dirname = args[output_dirname_index]

            input_tensor_flag_indices = [
                idx for idx in range(len(args)) if args[idx] in ('-i', '--input_tensor')
            ]
            input_tensor_file_path_indices = [idx + 3 for idx in input_tensor_flag_indices]
            original_input_tensor_file_paths = [args[idx] for idx in input_tensor_file_path_indices]

            with open(quant_checker_args.input_list) as file:
                for line in file.readlines():
                    filenames = line.rstrip().split('\n')[0]
                    if filenames == "":
                        continue
                    file_name = []
                    for idx, file in enumerate(filenames.split(' ')):
                        #input_list in case of multi input nodes contain ":=" string
                        #while single input model may not contain them
                        file = file.split(":=")[1] if ":=" in file else file
                        base_name = os.path.basename(file)
                        name, _ = os.path.splitext(base_name)
                        file_name.append(name)
                        args[input_tensor_file_path_indices[idx]] = file
                    args[output_dirname_index] = "_".join(file_name)
                    exec_framework_runner(args, logger=logger, validate_args=False)
            #Restore original file path
            for idx, file_path in zip(input_tensor_file_path_indices,
                                      original_input_tensor_file_paths):
                args[idx] = file_path
            #Restore original working dirname
            if original_dirname == '':
                args = args[:-2]
            else:
                args[output_dirname_index] = original_dirname

        #run framework runner
        else:
            exec_framework_runner(args, logger=logger, validate_args=False)

        quant_checker_args.golden_output_reference_directory = os.path.join(
            quant_checker_args.working_dir, 'framework_runner')
        if '--disable_graph_optimization' not in args:
            optimized_model_path = os.path.join(
                quant_checker_args.golden_output_reference_directory, "optimized_model" +
                FrameworkExtension.framework_extension_mapping[quant_checker_args.framework])
            if os.path.exists(optimized_model_path):
                quant_checker_args.model_path = optimized_model_path
    else:
        if '--disable_graph_optimization' not in args:
            optimized_model_path = os.path.join(
                quant_checker_args.golden_output_reference_directory, "optimized_model" +
                FrameworkExtension.framework_extension_mapping[quant_checker_args.framework])
            if os.path.exists(optimized_model_path):
                quant_checker_args.model_path = optimized_model_path
            else:
                logger.info(
                    "Please make sure model passed to QuantChecker is same as on which fp32 outputs are dumped."
                )

    if engine == "SNPE":
        quant_checker_args.input_list = quant_checker_args.snpe_input_list

    #generate model for each quantization scheme, and perform verirfication
    quant_checker = QuantChecker(quant_checker_args, logger)
    quant_checker.run()


def exec_binary_snooping(args, engine, logger=None, validate_args=False):
    from qti.aisw.accuracy_debugger.lib.options.binary_snooping_cmd_options import BinarySnoopingCmdOptions
    from qti.aisw.accuracy_debugger.lib.snooping.nd_binary_snooper import BinarySnooping
    binary_snooping_args = BinarySnoopingCmdOptions(args, engine, validate_args).parse()
    if (logger is None):
        logger = setup_logger(binary_snooping_args.verbose, binary_snooping_args.output_dir,
                              component=ComponentLogCodes.binary_snooping.value)

    if '--output_dirname' not in args:
        args.extend(['--output_dirname', 'abc'])
        output_dirname_index = -1
        original_dirname = ''
    else:
        output_dirname_index = args.index('--output_dirname') + 1
        original_dirname = args[output_dirname_index]

    input_tensor_flag_indices = [
        idx for idx in range(len(args)) if args[idx] in ('-i', '--input_tensor')
    ]
    input_tensor_file_path_indices = [idx + 3 for idx in input_tensor_flag_indices]
    original_input_tensor_file_paths = [args[idx] for idx in input_tensor_file_path_indices]

    with open(binary_snooping_args.input_list) as file:
        for idx, line in enumerate(file.readlines()):
            if line:
                filenames = line.rstrip().split('\n')[0]
                for file_idx, file in enumerate(filenames.split(' ')):
                    #input_list in case of multi input nodes contain ":=" string
                    #while single input model may not contain them
                    file = file.split(":=")[1] if ":=" in file else file
                    args[input_tensor_file_path_indices[file_idx]] = file
                args[output_dirname_index] = "Result_" + str(idx)
                framework_args = [
                    *(args), "--add_layer_outputs", ",".join(binary_snooping_args.output_tensor)
                ]
                exec_framework_runner(framework_args, logger=logger, validate_args=False)
    #Restore original file path
    for idx, file_path in zip(input_tensor_file_path_indices, original_input_tensor_file_paths):
        args[idx] = file_path
    #Restore original working dirname
    if original_dirname == '':
        args = args[:-2]
    else:
        args[output_dirname_index] = original_dirname

    binary_snooper = BinarySnooping(binary_snooping_args, logger)
    binary_snooper.run()


def exec_snooper(args: list, type: str, logger: Logger | None = None,
                 validate_args: bool = True) -> None:
    '''
    Executes the qairt snooping algorithms.

    :param args: list of snooping arguments
    :param type: snooping algorithm type
    :param logger: object of logging.Logger
    :param validate_args: True, if argument validation is required.

    :raise Exception: If snooping algorithm fails.
    '''
    from qti.aisw.accuracy_debugger.lib.utils.snooper_utils import \
        get_qairt_snooping_cmd_option_class
    from qti.aisw.accuracy_debugger.lib.utils.snooper_utils import get_qairt_snooper_class

    try:
        snooper_cmd_class = get_qairt_snooping_cmd_option_class(type)
        subcomponent_log_code = ComponentLogCodes.get_component_code(type)

        snooper_args = snooper_cmd_class(args, validate_args=validate_args).parse()
        if not logger:
            logger = setup_logger(snooper_args.verbose, snooper_args.output_dir,
                                  component=subcomponent_log_code)
        symlink('latest', snooper_args.output_dir, logger)
        snooper_class = get_qairt_snooper_class(type)
        snooper = snooper_class(args=snooper_args, logger=logger)
        snooper.run()
    except Exception as e:
        raise Exception("Encountered Error: {}".format(str(e)))
