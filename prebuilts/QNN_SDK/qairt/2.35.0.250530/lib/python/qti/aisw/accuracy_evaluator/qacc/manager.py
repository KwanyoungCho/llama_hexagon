# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import builtins
import copy
import csv
import datetime
import glob
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import wait
from enum import Enum
from itertools import chain
from typing import Optional

import qti.aisw.accuracy_evaluator.qacc.dataset as ds
import qti.aisw.accuracy_evaluator.qacc.inference as infer
import qti.aisw.accuracy_evaluator.qacc.plugin as pl
from qti.aisw.accuracy_evaluator.common.compiler import CompilationEngine
from qti.aisw.accuracy_evaluator.common.transforms import ModelHelper
from qti.aisw.accuracy_evaluator.common.utilities import ComparatorHelper, Helper, ModelType
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.qacc.config_definitions import *
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.tools.core.utilities.data_processing.core.transformations import (
    ComponentRegistry,
    DatasetConfig,
    MetricConfig,
    PostProcessorConfig,
    PreProcessorConfig,
    ProcessorChainExecutor,
)
from qti.aisw.tools.core.utilities.data_processing.datasets.base import RawInputListDataset
from tabulate import tabulate
from tqdm import tqdm


class EvaluatorStatus(Enum):
    """Enum to represent the error status of evaluator"""
    SUCCESS = 0
    PIPELINE_FAILURE = 1
    PIPELINE_EXCEPTION = 2


console = builtins.print

# stage completion status
STAGE_PREPROC_PASS = False
STAGE_INFER_PASS = False

pipeline_cache = pl.PipelineCache.getInstance()


def format_params(
        params_dict: dict,
        prefix: str = '',
        separator: str = ': ',
        newline: str = '\n'
    ) -> str:
    r"""Formats a dictionary of parameters into a string representation.

    Args:
        params_dict: A dictionary containing parameter names and their values.
        prefix: A string to prepend to the formatted parameters.
        separator: A string to separate parameter names and values.
        newline: A string to represent a new line.

    Returns:
        A string representation of the parameters in the format:
        "key: value\nkey: value\n..."
    """
    params_str = prefix
    for k, v in params_dict.items():
        if v is not None:
            params_str += f'{k}{separator}{v}{newline}'
    return params_str


class QACCManager:
    """Manages the inference and evaluation steps for a model."""

    def __init__(self, config: EvaluatorPipelineConfig, work_dir: str,
                 use_memory_plugins: Optional[bool] = False):
        """Initializes the QACCManager.

        Args:
            config (EvaluatorPipelineConfig): The configuration object.
            work_dir (str): The working directory for the inference process.
            use_memory_plugins (bool, Optional): Whether to use memory plugins.
        """
        # Stores the runtime info for each inference schema.
        self.inference_schema_run_status = {}
        # available plugins must be loaded before loading configuration.
        self.use_memory_plugins = use_memory_plugins
        if self.use_memory_plugins:
            ComponentRegistry()
        else:
            pl.PluginManager.findAvailablePlugins()

        qacc_logger.info('Loading model config')
        try:
            # Copy the config pydantic object to local member
            self.config = config
            # TBD: Loading the model config from the yaml file, if config is a path.
        except Exception as e:
            qacc_logger.error('qacc failed to load config file. check log for more details.')
            qacc_file_logger.exception(e)
            sys.exit(1)

        self.work_dir = work_dir
        self.input_info = self.config.inference_config.inputs_info
        self.input_names = list(self.input_info.keys())
        self.output_info = self.config.inference_config.outputs_info

    def process_dataset(self, dataset_config):
        qacc_logger.info('Executing dataset plugins')
        out_dir = self.get_output_path(self.work_dir, qcc.DATASET_DIR)
        plugin_manager = pl.PluginManager()
        return plugin_manager.execute_dataset_transformations(dataset_config, out_dir)

    def prepare_dataset(self, dataset_config: DatasetConfig, use_calibration=False, max_samples=None):
        # qacc_logger.info('Preparing dataset')
        dataset_config_local = copy.deepcopy(dataset_config)
        dataset = ComponentRegistry.get_components_from_configs([dataset_config_local],
                 use_calibration=use_calibration, max_samples=max_samples)[0]
        return dataset

    def preprocess(self, dataset, is_calibration=False):
        # Execute preprocessing.
        if is_calibration:
            qacc_logger.info('Executing Preprocessors for calibration inputs')
            out_dir = self.get_output_path(self.work_dir, qcc.STAGE_PREPROC_CALIB)
            pipeline_cache.set_val(qcc.PIPELINE_CALIB_DIR, out_dir)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_IS_CALIB, is_calibration)
        else:
            qacc_logger.info('Executing Preprocessors')
            out_dir = self.get_output_path(self.work_dir, qcc.STAGE_PREPROC)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_DIR, out_dir)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_IS_CALIB, is_calibration)
        preprocessing_config = self.config.preprocessing_config
        transformations = pl.Transformations(
            plugin_config_list=preprocessing_config.preprocessing_plugin_list,
            max_input_len=dataset.get_total_entries())
        plugin_manager = pl.PluginManager(dataset)

        ret_status = plugin_manager.execute_transformations(transformations=transformations,
                                                            output_dir=out_dir, batch_offset=0,
                                                            input_names=self.input_names)
        return ret_status, self.get_output_path(out_dir, qcc.QNN_PROCESSED_OUTFILE)

    def preprocess_memory(self, dataset, is_calibration=False):
        """Preprocess the configured dataset. All the items are processed at a
        time and a processed outlist file is returned which contains path to
        the preprocessed files.

        Args:
            dataset: Dataset object containing information on the source data to be processed
            is_calibration : Flag to indicate whether the supplied dataset is Calibration set
        Returns:
            status: 0 if success otherwise 1
            processed_file_list_path: path to the file list containing paths of preprocessed input raw files.
        """
        if is_calibration:
            qacc_logger.info('Executing Preprocessors for calibration inputs')
            out_dir = self.get_output_path(self.work_dir, qcc.STAGE_PREPROC_CALIB)
            pipeline_cache.set_val(qcc.PIPELINE_CALIB_DIR, out_dir)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_IS_CALIB, is_calibration)
        else:
            qacc_logger.info('Executing Preprocessors')
            out_dir = self.get_output_path(self.work_dir, qcc.STAGE_PREPROC)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_DIR, out_dir)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_IS_CALIB, is_calibration)

        if self.config.preprocessing_config:
            # Do Preprocessing only when it is configured within config file.
            processor_configs = [PreProcessorConfig(name=b.name, params=b.params) for b in self.config.preprocessing_config.preprocessing_plugin_list]
            processors_objs = ComponentRegistry.get_components_from_configs(processor_configs)
            executor = ProcessorChainExecutor(processors_objs, dump_outputs=True, output_dir=out_dir, node_names=self.input_names)
            processed_file_path = []
            for data in tqdm(dataset):
                _, output_paths_per_sample = executor.process(data)
                processed_file_path.append(output_paths_per_sample)
        else:
            # Pick items from the preprocessed dataset and write it out into processed-outputs.txt
            executor = ProcessorChainExecutor([], dump_outputs=True, output_dir=out_dir, node_names=self.input_names)
            os.makedirs(os.path.join(out_dir), exist_ok=True)
            processed_file_path = [','.join(executor.save_outputs(sample, sub_folder_name='preproc')) + '\n' for sample in tqdm(dataset)]
        ret_status = 0
        preprocessed_out_file = os.path.join(out_dir, qcc.PROCESSED_OUTFILE)
        with open(preprocessed_out_file, 'w') as file:
            file.writelines(processed_file_path)

        preprocessed_out_file_qnn_format = os.path.join(out_dir, qcc.QNN_PROCESSED_OUTFILE)

        # Write the processed inputs in a different format.
        if self.input_names is not None:
            with open(preprocessed_out_file_qnn_format, 'w') as fl:
                for input_paths in processed_file_path:
                    for i, input_path in enumerate(input_paths.split(',')):
                        if input_path is None:
                            qacc_file_logger.error('Null input found at index {} while creating'
                                                   ' qnn-processed-outputs.txt.\n Record {}'.format(
                                                       i, input_path))
                            raise RuntimeError('Some inputs were not processed!')
                        inp_abspath = os.path.abspath(input_path)
                        if i:
                            fl.write(" " + f"{self.input_names[i]}:={inp_abspath}")
                        else:
                            fl.write(f"{self.input_names[i]}:={inp_abspath}")
        return ret_status, self.get_output_path(out_dir, qcc.QNN_PROCESSED_OUTFILE)

    def infer(self, model_path, processed_input_file, inference_schema, dataset, device_id,
              inference_schema_name, compile_only=False, load_binary_from_dir=False):

        # Execute Inference.
        qacc_logger.info('({}) Starting inference engine'.format(inference_schema_name))
        dir_name = self.get_output_path(dir=self.work_dir, type=qcc.STAGE_INFER,
                                        inference_schema_name=inference_schema_name)
        infer_ds_path = self.get_output_path(dir=dir_name, type=qcc.INFER_OUTFILE)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_DIR, dir_name, inference_schema_name)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_FILE, infer_ds_path, inference_schema_name)
        binary_path = self.get_output_path(dir=self.work_dir, type=qcc.BINARY_PATH,
                                           inference_schema_name=inference_schema_name)

        # set network binary directory
        network_bin_dir = inference_schema.precompiled_path \
            if inference_schema.precompiled_path is not None else binary_path

        # store values in pipeline cache
        pipeline_cache.set_val(qcc.PIPELINE_NETWORK_BIN_DIR, network_bin_dir, inference_schema_name)
        pipeline_cache.set_val(qcc.PIPELINE_NETWORK_DESC,
                               os.path.join(network_bin_dir, qcc.NETWORK_DESC_FILE),
                               inference_schema_name)
        pipeline_cache.set_val(qcc.PIPELINE_PROGRAM_QPC,
                               os.path.join(network_bin_dir, qcc.PROGRAM_QPC_FILE),
                               inference_schema_name)

        # Set Precompiled path to the corresponding binary path
        qnn_sdk_dir = pipeline_cache.get_val(qcc.QNN_SDK_DIR)
        precompiled_path = binary_path if load_binary_from_dir else None
        infer_mgr = infer.InferenceManager(inference_schema, self.config.inference_config,
                                           binary_path)
        if self.use_memory_plugins:
            calibration_file = (CalibrationType.RAW, self.calibration_file)
        else:
            if self.config.dataset_config is not None:
                calibration_file = dataset.get_dataset_calibration()
            else:
                calibration_file = (CalibrationType.RAW, processed_input_file)
        err_status, infer_fail_stage, execution_time = infer_mgr.execute(
            model_path=model_path, output_dir=dir_name, input_file=processed_input_file,
            output_file=infer_ds_path, calibration=calibration_file, device_id=device_id,
            precompiled_path=precompiled_path, console_tag=inference_schema_name,
            compile_only=compile_only, qnn_sdk_dir=qnn_sdk_dir)

        return err_status, infer_fail_stage, infer_ds_path, execution_time

    def post_inference(self, inference_schema, dataset, infer_ds_path, inference_schema_name,
                       pipeline_stages, adapter_objs, postprocessor_objs, metric_objs=[]):
        """Performs Postprocessing and metric computation on the entire
        inference outputs.

        Args:
            inference_schema: Schema object to be executed
            dataset: Dataset object containing information on the source data to be processed
            infer_ds_path: Path to the inference output file list.
            inference_schema_name: Name of the inference schema being executed.
            pipeline_stages : List of stages configured for execution
            adapter_objs: List containing adapter objects that are to be executed.
            postprocessor_objs: List containing postprocessing objects that are to be executed.
            metric_objs: List containing metric plugin objects that are to be executed.

        Returns:
            metric_result: dictionary containing the metrics
        """
        dtypes = [o[0] for o in self.output_info.values()]
        shapes = [o[1] for o in self.output_info.values()]
        ground_truth_from_orig_dataset = [dataset[i].annotation for i in range(len(dataset))]
        metadata_from_orig_dataset = [dataset[i].metadata for i in range(len(dataset))]
        model_output_raw_dataset = RawInputListDataset(inputlist_file=infer_ds_path, dtypes=dtypes,
                                        shapes=shapes, absolute_path_list=True)
        model_output_raw_dataset.annotation = ground_truth_from_orig_dataset
        qacc_logger.info(f"Postprocessing and Computing Metrics for : {inference_schema_name}")
        for idx, output in enumerate(tqdm(model_output_raw_dataset)):
            output.metadata = metadata_from_orig_dataset[idx]
            for adapter_obj in adapter_objs:
                output = adapter_obj.transform(output)
            for postprocessor_obj in postprocessor_objs:
                output = postprocessor_obj(output)
            for metric in metric_objs:
                metric(output)
        metric_result = {}
        for metric in metric_objs:
            qacc_logger.debug(
                f"Invoking Metric.finalize() for {inference_schema_name} --> {metric}")
            metric_result.update(metric.finalize())
        qacc_logger.info(f"Metrics for {inference_schema_name} : {metric_result}")

        return 0, metric_result

    def postprocess(self, idx, dataset, infer_ds_path, inference_schema_name):
        # Execute post processing for this inference results.
        # Get inference output dataset
        if self.config.postprocessing_config:
            qacc_logger.info('({}) Executing Postprocessors'.format(inference_schema_name))
            infer_dataset = ds.DataSet(input_list_file=infer_ds_path)
            squash_results = self.config.postprocessing_config.squash_results
            postprocessing_config = self.config.postprocessing_config
            transformations = pl.Transformations(
                plugin_config_list=postprocessing_config.postprocessing_plugin_list,
                max_input_len=dataset.get_total_entries())
            plugin_manager = pl.PluginManager(infer_dataset, orig_dataset=dataset)
            dir_name = self.get_output_path(dir=self.work_dir, type=qcc.STAGE_POSTPROC,
                                            inference_schema_name=inference_schema_name)
            pipeline_cache.set_val(qcc.PIPELINE_POSTPROC_DIR, dir_name, inference_schema_name)
            err_status = plugin_manager.execute_transformations(transformations=transformations,
                                                                output_dir=dir_name, batch_offset=0,
                                                                squash_results=squash_results)
            if err_status:
                return 1, None

            metrics_input_file = self.get_output_path(dir=dir_name, type=qcc.PROCESSED_OUTFILE)
        else:
            metrics_input_file = infer_ds_path
        pipeline_cache.set_val(qcc.PIPELINE_POSTPROC_FILE, metrics_input_file,
                               inference_schema_name)

        return 0, metrics_input_file

    def evaluate_metrics(self, idx, dataset, postproc_file, inference_schema):
        """Evaluate the given metrics on the inferred data."""
        inference_schema_name = inference_schema.get_inference_schema_name()
        if self.config.metrics_config.metrics_plugin_list:
            qacc_logger.info('({}) Evaluating metrics'.format(inference_schema_name))
            processed_dataset = ds.DataSet(input_list_file=postproc_file)
            plugin_manager = pl.PluginManager(processed_dataset, orig_dataset=dataset)
            metrics_pl_cfg = self.config.metrics_config.metrics_plugin_list
            metrics_results = []
            metrics_results_dict = {}
            dir_name = self.get_output_path(self.work_dir, qcc.STAGE_METRIC, inference_schema_name)
            err_status = plugin_manager.execute_metrics(metrics_plugin_config=metrics_pl_cfg,
                                                        output_dir=dir_name,
                                                        results_str_list=metrics_results,
                                                        results_dict=metrics_results_dict)
            if err_status:
                self.inference_schema_run_status[inference_schema_name]['metrics'] = {}
                self.inference_schema_run_status[inference_schema_name][
                    'status'] = qcc.SCHEMA_METRIC_FAIL
                return 1
            metrics_info = ''
            for res in metrics_results:
                qacc_logger.info('({}) metric: {}'.format(inference_schema_name,
                                                          res.replace('\n', ' ')))
                if len(metrics_info) > 0:
                    metrics_info += '\n' + res
                else:
                    metrics_info = res
            self.inference_schema_run_status[inference_schema_name][
                'metrics'] = metrics_results_dict
        else:
            self.inference_schema_run_status[inference_schema_name]['metrics'] = {}
            return 0

    def compare_infer_results(self, preproc_file):
        """Compare inference outputs with configured comparator.

        Comparison can be done if there are more than 1 inference
        schemas. User can configure a reference inference schema by
        is_ref=True in inference_schema section in yaml. In absence of
        is_ref, the first defined inference schema is considered as
        reference and the outputs of other inference schemas are
        compared against those of the reference schema.
        """

        def get_filelist_from_file(filepath: str) -> list[list[str]]:
            """Read the given file to return a list of filepaths. If the file is empty,
            an empty list would be returned.

            Args:
                filepath: Path to file containing file paths
            Returns:
                List of file paths list present in the file
            """
            path_list = []
            if not os.path.exists(filepath):
                raise FileNotFoundError(f'File {filepath} does not exist.')
            with open(filepath) as files:
                for line in files:
                    path_list.append(re.split(r'[ ,]', line.strip()))
            return path_list

        inference_schemas = self.config.inference_config.inference_schemas
        if inference_schemas and len(inference_schemas) < 2:
            qacc_logger.info('Not enough inference schemas to compare inference outputs')
            return 0

        ref_inference_schema = self.config.get_ref_inference_schema()

        ref_out_dir = self.get_output_path(self.work_dir, qcc.STAGE_INFER,
                                           ref_inference_schema.get_inference_schema_name())
        ref_out_file = self.get_output_path(ref_out_dir, qcc.INFER_OUTFILE)
        ref_schema_name = ref_inference_schema.get_inference_schema_name()
        if not os.path.exists(ref_out_file):
            qacc_file_logger.error(f'Reference inference out file {ref_out_file} does not exist')
            return 1

        outputs_ref = get_filelist_from_file(ref_out_file)
        preproc_inputs = get_filelist_from_file(preproc_file)

        out_names, comparator_list, comp_dtypes, comp_names = ComparatorHelper.get_comparators(
            comp_type=self.config.verifier_config.type, tolerance=self.config.verifier_config.tol,
            out_info=self.output_info, ref_out_file=ref_out_file)
        qacc_file_logger.info(f'Comparators: {comparator_list}')
        qacc_file_logger.info(f'Comparator dtypes: {comp_dtypes}')

        # compare outputs for all inference schemas with reference.
        top = min(int(self.config.verifier_config.fetch_top), len(outputs_ref))

        qacc_file_logger.info('Comparing inference output files. This may take some time..')
        qacc_file_logger.info('================ Inference output comparisons ====================')
        qacc_file_logger.info('Comparing all files ...')
        for inference_schema in inference_schemas:
            if inference_schema.idx == ref_inference_schema.idx:
                continue

            inference_schema_name = inference_schema.get_inference_schema_name()

            try:
                out_file = self.get_output_path(
                    self.get_output_path(self.work_dir, qcc.STAGE_INFER, inference_schema_name),
                    qcc.INFER_OUTFILE)

                if not os.path.exists(out_file):
                    qacc_file_logger.error(
                        f'Inference schema infer out file {out_file} does not exist')
                    continue

                outputs_inference_schema = get_filelist_from_file(out_file)

                if len(outputs_ref) != len(outputs_inference_schema):
                    qacc_file_logger.error(
                        f'Infer output files count for {ref_schema_name}:{len(outputs_ref)} '
                        f'does not match for {inference_schema_name}:{len(outputs_inference_schema)}'
                    )
                    return 1

                # compare each output of each inference schema and reference.
                output_results_per_output = {}
                for i, ref_inps in enumerate(outputs_ref):
                    inference_schema_inps = outputs_inference_schema[i]
                    if len(ref_inps) != len(inference_schema_inps):
                        qacc_file_logger.error(
                            f'Record {i}: Number of reference inputs {len(ref_inps)} must match '
                            f'number of inference schema {inference_schema_name} '
                            f'inputs {len(inference_schema_inps)}')
                        return 1

                    for out_i, (a_path, r_path) in enumerate(zip(inference_schema_inps, ref_inps)):
                        a_tensor = Helper.get_tensor_from_file(a_path.strip(), comp_dtypes[out_i])
                        r_tensor = Helper.get_tensor_from_file(r_path.strip(), comp_dtypes[out_i])
                        result = comparator_list[out_i].compare([a_tensor], [r_tensor])
                        if out_i in output_results_per_output:
                            output_results_per_output[out_i].append(result[0])
                        else:
                            output_results_per_output[out_i] = [result[0]]

                self.inference_schema_run_status[inference_schema_name]['comparator'] = {}
                deviating_filename = os.path.join(self.work_dir,
                                                  inference_schema_name + '_deviating.txt')
                for out_i, oname in enumerate(out_names):
                    _mean = round(
                        sum(output_results_per_output[out_i]) /
                        len(output_results_per_output[out_i]), 3)
                    self.inference_schema_run_status[inference_schema_name]['comparator'].update(
                        {f'({comp_names[out_i].value}) {oname}': _mean})
                    qacc_file_logger.info(f'\t({comp_names[out_i].value}) {oname} => {_mean}')

                deviating_file_list = ComparatorHelper.get_top_deviating_samples(
                    output_results_per_output, comp_names[0], preproc_inputs, top)
                if deviating_file_list:
                    with open(deviating_filename, 'w') as f:
                        f.write('\n'.join(deviating_file_list))

            except Exception as e:
                qacc_file_logger.error(e)
                return 1

        return 0

    def parse_pipeline_stages(self) -> tuple[int, tuple[list[str], Optional[str], Optional[str]]]:
        """Parses pipeline stages from the configuration.

        Returns:
            Tuple[int, Tuple[List[str], Optional[str], Optional[str]]]:
                - 0 if stages are successfully parsed.
                - 1 if there are no pipeline stages configured.
                - The tuple contains (pipeline_stages, pipeline_start, pipeline_end).
        """
        pipeline_stages, pipeline_start, pipeline_end = self.get_pipeline_stages_from_config(
            self.config)
        if len(pipeline_stages):
            qacc_file_logger.info('Configured stages: {}'.format(pipeline_stages))
        else:
            qacc_logger.error('Invalid pipeline start and end stages')
            return 1, (None, None, None)
        return 0, (pipeline_stages, pipeline_start, pipeline_end)

    def parse_and_validate_device_ids(self,
              device_id: int | str | list) -> tuple[bool, list[int | str]]:
        """Parse and validate device IDs based on their type.

        Args:
            device_id: Can be an integer, string, or list containing device IDs.
                - Integer: Represents a single AIC device ID.
                - String: Can represent a single HTP device ID (hex format) or a comma-separated list.
                - List: List of device IDs (integers or hex strings).

        Returns:
            tuple: (status, device_ids)
                - status: Boolean indicating if validation was successful.
                - device_ids: List of validated device IDs.
        """
        def is_hexadecimal(s: str) -> bool:
            """Check if a string is a valid hexadecimal number.

            Args:
                s: String to check.

            Returns:
                bool: True if the string is a valid hexadecimal number.
            """
            try:
                int(s, 16)
                return True
            except ValueError:
                return False

        if isinstance(device_id, int):
            # Single AIC device ID i.e. device_id=0 format
            device_ids = [device_id]
        elif isinstance(device_id, (str, list)):
            """
            device_id can be any of the following:
            - list of AIC devices: '0,1'
            - an HTP device: fb01855e
            - list of HTP devices: 'fb01855e,90438458'
            """
            device_id_list = device_id.strip().split(',') if isinstance(device_id, str) else device_id
            device_ids = []
            for device in device_id_list:
                if isinstance(device, str) and len(device) == 8 and is_hexadecimal(device):
                    # HTP device case
                    device_ids.append(device)
                else:
                    device_ids.append(int(device))
        status = True
        if device_ids and isinstance(device_ids[0], int):
            status = Helper.validate_aic_device_id(device_ids)
        return status, device_ids

    def parse_inference_schemas(self, inference_schema_name: Optional[str] = None,
                        inference_schema_tag: Optional[str] = None):
        """Parses and filters inference schemas based on name and tag, then creates an
        inference schema manager.

        Args:
            inference_schema_name (str, optional): The name of the inference schema to filter by.
            inference_schema_tag (str, optional): The tag of the inference schema to filter by.

        Returns:
            tuple[list[InferenceSchema], InferenceSchemaManager]:
                - A list of filtered inference schemas.
                - An inference schema manager instance.
        """
        # list of inference schemas from config
        inference_schemas = self.config.inference_config.inference_schemas
        # Filter inference schemas using supplied cli inference-schema and inference-schema-tag
        inference_schemas = self.filter_inference_schemas(
            inference_schemas, inference_schema_name=inference_schema_name,
            inference_schema_tag=inference_schema_tag)
        # once inference schema(s) is selected perform further actions
        # create inference schema manager
        inference_schema_manager = infer.InferenceSchemaManager(inference_schemas, self.config)

        # search space scan and adding inference schema combination
        # TODO: Refactor is_calib_req logic from the below into scan_and_add_inference_schema_permutations()
        inference_schemas, is_calib_req = \
                        inference_schema_manager.scan_and_add_inference_schema_permutations()
        self.config.inference_config._is_calib_req = is_calib_req
        # update the config object with all inference schema permutation
        self.config.inference_config.inference_schemas = inference_schemas
        self.config.inference_config.set_inference_schema_names()

        # create schedule for different inference schemas
        inference_schema_manager.create_schedule()

        return inference_schemas, inference_schema_manager

    def clean_and_simplify_model(self,
            inference_schemas: list[InferenceSchemaConfiguration],
            onnx_symbol:  Optional[str] = None,
            is_custom_op_model: bool = False,
            has_quantization_overrides_flag: bool = False
            ) -> tuple[str, list[InferenceSchemaConfiguration]]:
        """Cleans and simplifies the model based on configuration and inference schema parameters.

        This method handles the cleaning and simplification of the model, taking into account
        whether the model is a custom op model, whether quantization overrides are present,
        and whether simplification is requested. It updates the inference schemas with the
        appropriate model paths and converter parameters.

        Args:
            inference_schemas: List of inference schemas to be updated with model paths and converter parameters.
            onnx_symbol: Optional list of ONNX symbols for model cleaning.
            is_custom_op_model: Boolean indicating if the model is a custom op model.
            has_quantization_overrides_flag: Boolean indicating if quantization overrides are present.

        Returns:
            A tuple containing the cleaned/simplified model path and the updated list of inference schemas.
        """
        # self.config.inference_config._model_object is True for tf session and pytorch module. False otherwise
        # clean only if the model is not a tf session or pytorch module or if clean_model is set to True.
        if self.config.inference_config._model_object or not self.config.inference_config.clean_model:
            return self.config.inference_config.model_path
        # clean_model is set to True
        if self.config.inference_config.clean_model:
            qacc_logger.info('Cleaning up model..')
            symbols = {}
            if self.config.inference_config.onnx_define_symbol:
                sym_from_config = self.config.inference_config.onnx_define_symbol.split(' ')
                for sym in sym_from_config:
                    elems = sym.split('=')
                    symbols[elems[0]] = int(elems[1])
            if onnx_symbol:
                for sym in onnx_symbol:
                    elems = sym[0].split(':')
                    symbols[elems[0]] = int(elems[1])
            if is_custom_op_model:
                # For custom op model: No clean up, simplification & checm model to be performed
                # if user specifies simplify_model=True and provides custom_op_model
                if self.config.inference_config.simplify_model:
                    self.config.inference_config.simplify_model = False
                    qacc_file_logger.warning(
                        "Can't simplify the model when custom ops is specified, continuing without simplification and cleanup."
                    )
                model_path = ModelHelper.clean_model_for_qairt(
                    self.config.inference_config.model_path, out_dir=self.work_dir,
                    symbols=symbols, check_model=False, simplify_model=False)

            # if any of the inference schemas contain "quantization_overrides", Generate model without simplification
            elif has_quantization_overrides_flag or not self.config.inference_config.simplify_model:
                # if user specifies simplify_model=True and provides quantization_overrides ignore the provided value and use simplify_model=False
                if self.config.inference_config.simplify_model:
                    qacc_file_logger.warning(
                        "Can't simplify the model when quantization overrides is specified, continuing only with cleanup."
                    )
                cleaned_model_path = ModelHelper.clean_model_for_qairt(
                    self.config.inference_config.model_path, out_dir=self.work_dir,
                    symbols=symbols, check_model=self.config.inference_config.check_model,
                    simplify_model=False)
                self.config.inference_config._cleaned_only_model_path = cleaned_model_path

            # Always prepare a simplified cleaned model with user provided similpy action  as some inference schemas could still not have quantization_overrides
            model_path = ModelHelper.clean_model_for_qairt(
                self.config.inference_config.model_path, out_dir=self.work_dir,
                symbols=symbols, check_model=self.config.inference_config.check_model,
                simplify_model=self.config.inference_config.simplify_model)

            # update the  inference schemas with relevant converter params based on simplification flags etc
            if Helper.get_model_type(self.config.inference_config.model_path) == ModelType.ONNX:
                for inference_schema in inference_schemas:
                    if is_custom_op_model:
                        # if the model has custom op disable onnx_simplification flag in converter params
                        # if the model contains custom op use source model. [No Cleaning and Simplification]
                        inference_schema._model_path = self.config.inference_config.model_path
                        if inference_schema.converter_params:
                            inference_schema.converter_params.onnx_simplification = False
                        qacc_file_logger.info(
                            f"Disabling onnx_simplification in converter args for {inference_schema.get_inference_schema_name()}"
                            " as model has custom op"
                        )
                    elif inference_schema.converter_params and inference_schema.converter_params.quantization_overrides:
                        # if the inference_schema has quantization_overrides, add onnx_simplification flag to converter params
                        # if quantization_overrides present in inference_schema --> Use cleaned model [Not simplified]
                        inference_schema._model_path = cleaned_model_path
                        inference_schema.converter_params.onnx_simplification = False
                        qacc_file_logger.info(
                            f"Disabling onnx_simplification in converter args for inference_schema={inference_schema.get_inference_schema_name()}"
                            "as quantization_overrides is configured")
                    elif not self.config.inference_config.simplify_model:
                        # if simplify_model is set to False present in inference_schema --> Use cleaned model [Not simplified]
                        inference_schema._model_path = cleaned_model_path
                        if inference_schema.converter_params:
                            inference_schema.converter_params.onnx_simplification = False
                        qacc_file_logger.info(
                        f"Disabling onnx_simplification in converter args for {inference_schema.get_inference_schema_name()}"
                            " as simplify model flag is set to False in inference config")
                    else:
                        # In other cases use the simplified  + cleaned model
                        inference_schema._model_path = model_path
            else:
                # For Non-ONNX models use model_path after cleanup
                for inference_schema in inference_schemas:
                    inference_schema._model_path = model_path
            return model_path, inference_schemas

    def run_memory_pipeline(self,
            work_dir: str = 'qacc_temp',
            inference_schema_name: Optional[str] = None,
            inference_schema_tag: Optional[str] = None,
            cleanup: str = '',
            onnx_symbol: Optional[str] = None,
            device_id: Optional[str] = None,
            silent: bool = False) -> tuple[int, list[list]]:
        """Executes the memory pipeline for given EvaluatorConfiguration
        This function manages the execution of the inference pipeline by:
            - Parsing pipeline stages.
            - Filtering and validating inference schemas.
            - Preparing calibration data if required.
            - Compiling the models.
            - Executing the compiled models on specified devices.
            - Collecting and summarizing the results.

        Args:
            work_dir (str): Directory to store the results of the memory pipeline.
            inference_schema_name (Optional[str]): Name of the InferenceSchema to be used
                for execution.
            inference_schema_tag (Optional[str]): Tag of the InferenceSchema to be used
                for execution.
            cleanup (str): Cleanup option to be passed to qacc_cleanup.py.
            onnx_symbol (Optional[str]): ONNX symbol to be used in ONNX models.
            device_id (Optional[str]): Device ID to be used during execution.
            silent (bool): Flag to disable logging.

        Returns:
            Tuple[int, List[List]]
            A tuple containing the overall status code and a list of results for
                each inference schema.
        """
        ret_status = 0
        self.results = []
        inference_schema_manager = None
        model_path = None
        inference_schemas = []

        # Parse pipeline stages to determine the execution flow
        ret_status, (pipeline_stages, _, _) = self.parse_pipeline_stages()

        # Filter inference_schemas from config on name and tag, setup the device_ids and
        # create execution schedule
        if qcc.STAGE_INFER in pipeline_stages:
            if device_id:
                status, device_ids = self.parse_and_validate_device_ids(device_id)
                # If device_ids supplied are valid, then override the device ids in config
                if status:
                    self.config.inference_config.device_ids = device_ids

            inference_schemas, inference_schema_manager = self.parse_inference_schemas(
                                                    inference_schema_name=inference_schema_name,
                                                    inference_schema_tag=inference_schema_tag)

        if inference_schema_manager._is_calib_req:
            # Check if Calibration preprocessing is to be done and create the calibration_file
            calib_dataset = self.prepare_dataset(self.config.dataset_config, use_calibration=True,
                                        max_samples=self.config.info_config.max_calibration)
            err_status, calibration_file = self.preprocess_memory(calib_dataset, True)
            if err_status != 0:
                ret_status = EvaluatorStatus.PIPELINE_FAILURE
                return ret_status, [[]]
        else:
            calibration_file = None

        # Check if the model is a custom op model and if quantization overrides are enabled
        is_custom_op_model, has_quantization_overrides_flag = QACCManager.check_model_for_simplification(inference_schemas)

        # Clean and Simply the source model provided
        model_path, inference_schemas = self.clean_and_simplify_model(inference_schemas=inference_schemas,
                                onnx_symbol=onnx_symbol, is_custom_op_model=is_custom_op_model,
                                has_quantization_overrides_flag=has_quantization_overrides_flag)

        # Flatten the schedules
        schedule = [schedule for schedule_block in inference_schema_manager.get_schedule()
                                for schedule in schedule_block]

        # Compilation Block.
        # Compile all the inference schemas using the QAIRTCompiler clas
        if self.config.info_config.max_parallel_compilation:
            max_parallel_compilation = self.config.info_config.max_parallel_compilation
        else:
            max_parallel_compilation = int(os.cpu_count() / 2)

        # Note: Currently AIC Compilation is failing with multiple workers so setting
        # max_parallel_compilation to 1 for AIC.
        if any([inference_schema.backend == BackendType.AIC for inference_schema in inference_schemas]):
            max_parallel_compilation = 1
        compilation_engine = CompilationEngine(calibration_file=calibration_file, work_dir=work_dir,
                                    inputs_info=self.input_info, outputs_info=self.output_info,
                                    num_workers=max_parallel_compilation)
        # Run the compilation
        compilation_futures = compilation_engine.start_compile(inference_schemas=inference_schemas)

        pipeline_futures = {}
        pending_schemas = {}  # key: inference_schema_name value: compilation_future
        completed_schemas = []
        total_schema_compilations = len(compilation_futures)
        failed_compilations = 0

        evalaution_timing_dict = {}
        global_results_dict = {inference_schema.get_inference_schema_name(): {"_schema": inference_schema}
                                         for inference_schema in inference_schemas}
        is_comparator_enabled = self.config.verifier_config.enabled if len(inference_schemas) > 1 else False
        dump_stages = self.config.info_config.dump_stages
        ref_inference_schema_name = self.config.get_ref_inference_schema().get_inference_schema_name()

        if not self.config.info_config.max_parallel_evaluations:
            # Default Case: A maximum of half the number of CPU cores or the number of devices
            num_devices = len(inference_schema_manager.device_locks) if inference_schema_manager else 0
            max_parallel_evaluations = max(int(os.cpu_count() / 2), num_devices)
        else:
            max_parallel_evaluations = self.config.info_config.max_parallel_evaluations

        has_remote_device_evaluation = any([inference_schema.target_arch == TargetArchType.ANDROID
                                for inference_schema in inference_schemas])

        if has_remote_device_evaluation:
            dataset = self.prepare_dataset(self.config.dataset_config)
            err_status, preproc_file = self.preprocess_memory(dataset)
            if dump_stages and qcc.STAGE_PREPROC in dump_stages:
                dump_stages.remove(qcc.STAGE_PREPROC)
            if err_status != 0:
                ret_status = EvaluatorStatus.PIPELINE_FAILURE
                return ret_status, [[]]
        else:
            preproc_file = None

        inference_schema_executor = infer.InferenceSchemaExecutor(config=self.config,
                                         comparator_enabled=is_comparator_enabled,
                                         dump_stages=dump_stages,
                                         num_workers=max_parallel_evaluations,
                                         has_remote_device_evaluation=has_remote_device_evaluation,
                                         preprocessed_file_list=preproc_file)
        qacc_logger.info(f"Running Maximum of {max_parallel_evaluations} evaluation in parallel")
        # Processes compilations, schedules evaluations, and manages device locks until all schemas
        # are processed.
        while len(pipeline_futures) != (total_schema_compilations - failed_compilations):
            for inference_schema_name, (future, inference_schema) in compilation_futures.items():
                device_id = schedule[inference_schema.idx][1]
                device_lock = inference_schema_manager.device_locks[device_id] if device_id != -1 else None

                # Skip the completed schemas from further processing
                if inference_schema_name in completed_schemas:
                    continue

                # Futures in pending_schemas have completed compilation and awaiting evaluation.
                elif inference_schema_name in pending_schemas:
                    future = pending_schemas[inference_schema_name]
                    if not device_lock.locked() and device_lock.acquire():
                        total_compilation_time, (err_code, compiled_model, compilation_time_profile) = future.result()
                        plat_future = inference_schema_executor.execute(model=compiled_model,
                            work_dir=self.work_dir, inference_schema=inference_schema,
                            pipeline_stages=pipeline_stages, device_id=device_id,
                            device_lock=device_lock)
                        pipeline_futures[inference_schema_name] = plat_future
                        completed_schemas.append(inference_schema_name)
                        evalaution_timing_dict[inference_schema_name] = {'total_compilation_time': total_compilation_time,
                                                                        **compilation_time_profile}
                # Entry point: Start evaluation of schemas whose compilation is complete.
                elif future.done():
                    total_compilation_time, (err_code, compiled_model, compilation_time_profile) = future.result()
                    evalaution_timing_dict[inference_schema_name] = {
                                                'total_compilation_time': total_compilation_time,
                                                **compilation_time_profile
                                                }
                    if err_code == qcc.SCHEMA_COMPILE_SUCCESS:
                        if ref_inference_schema_name != inference_schema_name and \
                            ref_inference_schema_name not in pipeline_futures and \
                            ref_inference_schema_name not in completed_schemas:
                            # Skip if the reference inference schema name does not match the current
                            # one and is not in the pipeline futures
                            continue
                        if inference_schema.name != InferenceEngineType.QNN or device_id == -1:
                            # Schedule Non QNN * reference schemas to run on X86 Host CPU: ONNXRT
                            # /TORCHSCRIPT and CPU & HTP Backend (simulation)
                            plat_future = inference_schema_executor.execute(model=compiled_model,
                                            work_dir=self.work_dir, inference_schema=inference_schema,
                                            pipeline_stages=pipeline_stages, device_id=device_id)
                            completed_schemas.append(inference_schema_name)
                            # If comparator enabled then wait for onnxrt/ref plat to complete.
                            if is_comparator_enabled and inference_schema.is_ref:
                                wait([plat_future])
                            pipeline_futures[inference_schema_name] = plat_future
                            completed_schemas.append(inference_schema_name)
                        else:
                            # Handle case when: Compilation complete but Target not available
                            if device_lock and device_lock.locked():
                                pending_schemas[inference_schema_name] = future
                            elif device_lock and not device_lock.locked() and device_lock.acquire():
                                plat_future = inference_schema_executor.execute(model=compiled_model,
                                    work_dir=self.work_dir, inference_schema=inference_schema,
                                    pipeline_stages=pipeline_stages,
                                     device_id=device_id, device_lock=device_lock)
                                pipeline_futures[inference_schema_name] = plat_future
                                completed_schemas.append(inference_schema_name)
                        global_results_dict[inference_schema_name]['status'] = err_code
                    else:
                        # Case to handle failure of compilation
                        completed_schemas.append(inference_schema_name)
                        failed_compilations += 1
                        global_results_dict[inference_schema_name]['status'] = err_code
            time.sleep(5)

        # Add code to shutdown compilation_engine workers once all the futures are complete
        if len(pipeline_futures) == total_schema_compilations - failed_compilations:
            compilation_engine.workers.shutdown()
        wait([f[0] for f in compilation_futures.values()])  # ensure compilation to complete
        total_schema_executions = len(pipeline_futures)
        # # If there are any running schemas, wait for them to complete
        if len([f for f in pipeline_futures.values() if f.running()]) > 0:
            qacc_logger.info(
                f"Currently Running [{len([f for f in pipeline_futures.values() if f.running()])}] "
                f"Platform Executions in parallel")
            while True:
                if random.random() > 0.8:
                    qacc_logger.info(
                        f"Waiting for remaining schemas to complete Execution. Completed: ["
                        f"{len([f for f in pipeline_futures.values() if f.done()])}/"
                        f"{total_schema_executions}].")
                if len([f for f in pipeline_futures.values() if f.done()]) == total_schema_executions:
                    break
                time.sleep(15)
        # # Additional Check to ensure all Workergroup executions are complete
        wait(list(pipeline_futures.values()))

        # Deviating samples to be prepared
        deviating_samples_unique_indices = set()
        # on Completion of all the schema execution: Collect all the results and status from
        # the futures object
        for inference_schema_name, pipeline_future in pipeline_futures.items():
            ret_status, metrics_scores_dict, comparator_scores_dict, inference_time = (
                pipeline_future.result()
            )
            evalaution_timing_dict[inference_schema_name]['inference_time'] = inference_time
            global_results_dict[inference_schema_name]["metrics"] = metrics_scores_dict
            if 'most_deviating_indices' in comparator_scores_dict and not has_remote_device_evaluation:
                if len(comparator_scores_dict['most_deviating_indices']) > 0:
                    deviating_samples_unique_indices.add(*comparator_scores_dict['most_deviating_indices'])
                del comparator_scores_dict['most_deviating_indices']
            global_results_dict[inference_schema_name]["comparator"] = comparator_scores_dict

            # Update only inference_output based status only when Compilation status is Success
            if (qcc.get_inference_schema_status(
                global_results_dict[inference_schema_name]['status']
            ) == 'Success'):
                global_results_dict[inference_schema_name]['status'] = ret_status

        dataset = self.prepare_dataset(self.config.dataset_config)
        total_inputs = len(dataset)
        # When remote device evaluation is enabled, preprocessed files are saved to disk,
        # so the most deviating samples will also be included in the preprocessed data
        if not has_remote_device_evaluation:
            # Dump the most deviating sample inputs to preproc folder
            preproc_dir = os.path.join(self.work_dir, 'preproc')
            os.makedirs(os.path.join(self.work_dir, 'preproc'), exist_ok=True)
            if self.config.preprocessing_config:
                processor_configs = [PreProcessorConfig(name=b.name, params=b.params)
                                for b in self.config.preprocessing_config.preprocessing_plugin_list]
                processors_objs = ComponentRegistry.get_components_from_configs(processor_configs)
                executor = ProcessorChainExecutor(processors_objs, dump_outputs=True,
                                        output_dir=preproc_dir, node_names=self.input_names)
            else:
                executor = ProcessorChainExecutor([], dump_outputs=True, output_dir=preproc_dir,
                                        node_names=self.input_names)

            for sample in dataset:
                if sample.idx in deviating_samples_unique_indices:
                    executor.process(sample)

        # print Results table
        qacc_file_logger.info(f"Evaluated on {total_inputs} number of samples.")
        summary = []
        ret_status = EvaluatorStatus.SUCCESS
        for schema_name, results in global_results_dict.items():
            inference_schema = results["_schema"]
            backend_str = ''
            schema_name_str = schema_name + "\n (Reference)" if inference_schema.is_ref else schema_name
            entry = [schema_name_str]
            status_code = results['status']
            inference_schema_status_str = qcc.get_inference_schema_status(status_code)
            entry.append(inference_schema_status_str)
            if qcc.get_inference_schema_status(status_code) != 'Success':
                ret_status = EvaluatorStatus.PIPELINE_FAILURE

            entry.append(inference_schema.precision.value)  # Precision
            if inference_schema.backend:
                backend_str = f"{inference_schema.backend.value}\n{inference_schema.target_arch.value}"
            entry.append(backend_str)

            backed_extension_str, sub_modules_str = self.format_inference_schema_params(inference_schema)
            entry.append(backed_extension_str)
            entry.append(sub_modules_str)

            if 'metrics' in global_results_dict[schema_name] and \
                    global_results_dict[schema_name]['metrics']:
                metric_str = ''
                metrics_dict = global_results_dict[schema_name][
                    'metrics']
                for k, v in metrics_dict.items():
                    metric_str += '{}: {} \n'.format(k, v)
                entry.append(metric_str)
            else:
                metrics_dict = {}
                entry.append('-')

            if 'comparator' in global_results_dict[schema_name] and \
             len(global_results_dict[schema_name]['comparator']) >   0 and not inference_schema.is_ref:
                comparator_str = ''
                comparator_dict = global_results_dict[schema_name]['comparator']
                for k, v in comparator_dict.items():
                    comparator_str += f'{k}: {v} \n'
                entry.append(comparator_str)
            else:
                comparator_dict = {}
                entry.append('-')
            summary.append(entry)
            self.results.append([
                inference_schema.idx, inference_schema.tag, inference_schema_name,
                qcc.get_inference_schema_status(results['status']), inference_schema.precision.value,
                inference_schema.quantizer_params, metrics_dict, comparator_dict
            ])  # appending metric results

        summary.sort(reverse=True, key=lambda x: x[-1])
        # summary = [i[:-1] for i in summary]
        qacc_logger.info('Execution Summary:')
        headers = [
            'Inference schema', 'Status', 'Precision', 'Backend', 'Backend Extensions',
            'Sub Modules', 'Metrics', 'Comparator'
        ]
        result_csv_path = self.get_output_path(self.work_dir, qcc.RESULTS_TABLE_CSV)
        self.write2csv(result_csv_path, summary, header=headers)
        qacc_logger.info(f"\n{tabulate(summary, headers=headers)}")

        # print the timing statistics table
        timing_table_keys = ['Schema Name', 'convert', 'optimize', 'quantize',
                    'generate_binary', 'total_compilation_time',
                    'inference_time']

        timing_data = []
        for schema_name, timing_values in evalaution_timing_dict.items():
            row = [schema_name]
            for k in timing_table_keys[1:]:
                if k in timing_values:
                    row.append(timing_values[k])
                else:
                    row.append(0)
            timing_data.append(row)
        timing_csv = os.path.join(self.work_dir, 'stages_timing.csv')
        self.write2csv(timing_csv, timing_data, header=timing_table_keys)
        qacc_logger.info(f"\n{tabulate(timing_data, headers=timing_table_keys)}")
        return ret_status, self.results

    @staticmethod
    def format_inference_schema_params(inference_schema: InferenceSchemaConfiguration) -> tuple[str, str]:
        """Formats the inference schema parameters into a string representation.

        Args:
            inference_schema: The inference schema object containing parameters.

        Returns:
            A tuple containing:
            - backed_extension_str: Formatted string of backend extensions parameters.
            - sub_modules_str: Combined formatted string of quantizer, contextbin, and netrun parameters.
        """
        backed_extension_str = ''
        quantizer_params_str = ''
        contextbin_params_str = ''
        netrun_params_str = ''

        if inference_schema.backend_extensions:
            backed_extension_str = format_params(inference_schema.backend_extensions.dict(), prefix='\n')

        if inference_schema.quantizer_params:
            quantizer_params_str = 'Quantizer params:\n'
            quantizer_params_str += format_params(inference_schema.quantizer_params.dict())

        if inference_schema.contextbin_params:
            contextbin_params_str = '\nContext binary params:\n'
            contextbin_params_str += format_params(inference_schema.contextbin_params.dict())

        if inference_schema.netrun_params:
            netrun_params_str = '\nNetrun params:\n'
            netrun_params_str += format_params(inference_schema.netrun_params.dict())

        sub_modules_str = quantizer_params_str + contextbin_params_str + netrun_params_str
        return backed_extension_str, sub_modules_str

    def print_and_write_results(self,
                inference_schemas: list[InferenceSchemaConfiguration]) -> EvaluatorStatus:
        """Prints and writes the results of inference schema evaluations to both console and CSV file.

        Args:
            inference_schemas: List of inference schema objects to evaluate.

        Returns:
            EvaluatorStatus: The overall status of the evaluation process.
        """
        summary = []
        ret_status = EvaluatorStatus.SUCCESS
        # print the results in the same order as config
        for inference_schema_idx, inference_schema in enumerate(inference_schemas):
            entry = []
            inference_schema_name = inference_schema.get_inference_schema_name()
            status_code = self.inference_schema_run_status[inference_schema_name]['status']
            entry.append(inference_schema_name)
            if self.inference_schema_run_status[inference_schema_name]['infer_stage_status']:
                inference_schema_status_str = f"{qcc.get_inference_schema_status(status_code)} \n\
                    in {self.inference_schema_run_status[inference_schema_name]['infer_stage_status']}"
            else:
                inference_schema_status_str = qcc.get_inference_schema_status(status_code)
            entry.append(inference_schema_status_str)
            if qcc.get_inference_schema_status(status_code) != 'Success':
                ret_status = EvaluatorStatus.PIPELINE_FAILURE

            entry.append(inference_schema.precision.value)
            backend_str = ''
            if inference_schema.backend:
                backend_str = f"{inference_schema.backend.value}\n{inference_schema.target_arch.value}"
            entry.append(backend_str)

            backed_extension_str, sub_modules_str = self.format_inference_schema_params(inference_schema)
            entry.append(backed_extension_str)
            entry.append(sub_modules_str)

            if 'metrics' in self.inference_schema_run_status[inference_schema_name] and \
                    self.inference_schema_run_status[inference_schema_name]['metrics']:
                metric_str = ''
                metrics_dict = self.inference_schema_run_status[inference_schema_name][
                    'metrics']
                for k, v in metrics_dict.items():
                    metric_str += '{}: {} \n'.format(k, v)
                entry.append(metric_str)
            else:
                metrics_dict = {}
                entry.append('-')

            if 'comparator' in self.inference_schema_run_status[inference_schema_name]:
                comparator_dict = self.inference_schema_run_status[inference_schema_name][
                    'comparator']
                comparator_str = ''
                compare_value = float(list(comparator_dict.values())[0])
                for k, v in comparator_dict.items():
                    comparator_str += '{}: {} \n'.format(k, v)
                entry.append(comparator_str)
                entry.append(compare_value)
            else:
                comparator_dict = {}
                entry.append('-')
                entry.append(float("-inf"))
            summary.append(entry)
            self.results.append([
                inference_schema.idx, inference_schema.tag, inference_schema_name,
                qcc.get_inference_schema_status(status_code), inference_schema.precision.value,
                inference_schema.quantizer_params, metrics_dict, comparator_dict
            ])  # appending metric results
        summary.sort(reverse=True, key=lambda x: x[-1])
        summary = [i[:-1] for i in summary]
        qacc_logger.info('Execution Summary:')
        headers = [
            'Inference schema', 'Status', 'Precision', 'Backend', 'Backend extensions',
            'Sub Modules', 'Metrics', 'Comparator'
        ]
        result_csv_path = self.get_output_path(self.work_dir, qcc.RESULTS_TABLE_CSV)
        self.write2csv(result_csv_path, summary, header=headers)
        qacc_logger.info(f"\n{tabulate(summary, headers=headers)}")
        return ret_status

    def run_pipeline(self, work_dir='qacc_temp', inference_schema_name=None,
                     inference_schema_tag=None, cleanup='', onnx_symbol=None, device_id=None,
                     qnn_sdk_dir="", silent=False):
        """Executes the E2E pipeline based on the args and model configuration.

        Args:
            Arguments passed from cmd line are supplied to respective variables
        work_dir: path to directory to store the evaluation results and associated artifacts
        inference_schema_name: run only on this inference schema type Allowed values ['qnn','aic','onnxrt',
        'tensorflow','torchscript']
        inference_schema_tag: run only this inference schema tag
        cleanup:'cleanup preprocessing, inference and postprocessing output files.
            cleanup = 'end': deletes the files after all stages are completed.
            cleanup = 'intermediate' : deletes the intermediate inference and postprocessing
            output files. Selecting intermediate option saves space but disables comparator option'
        onnx_symbol: Replace onnx symbols in input/output shapes. Can be passed as list of
        multiple items. Default replaced by 1. e.g __unk_200:1
        device_id: Target Device to be used for accuracy evaluation
        Returns:
            status: 0 if success otherwise 1
        """
        ret_status = 0
        ret_status, (pipeline_stages, pipeline_start, pipeline_end) = self.parse_pipeline_stages()

        # execute dataset plugins
        if isinstance(self.config.dataset_config, DatasetConfiguration):
            self.config.dataset_config = self.process_dataset(self.config.dataset_config)
            # handle max_inputs and max_calib for backward compatibility
            # override max_calib from dataset plugin to dataset
            if pipeline_cache.get_val(qcc.PIPELINE_MAX_INPUTS):
                # Update max_inputs post process_dataset()
                self.config.dataset_config.update_max_inputs()
            # Set max_inputs in pipeline cache with updated values
            pipeline_cache.set_val(qcc.PIPELINE_MAX_INPUTS, self.config.dataset_config.max_inputs)
            # override max_calib from dataset plugin to inference section for backward compatibility
            if pipeline_cache.get_val(qcc.PIPELINE_MAX_CALIB):
                self.config.inference_config._max_calib = pipeline_cache.get_val(
                    qcc.PIPELINE_MAX_CALIB)
            else:
                pipeline_cache.set_val(qcc.PIPELINE_MAX_CALIB,
                                       self.config.inference_config._max_calib)
            # create dataset with modified dataset config
            dataset = ds.DataSet(dataset_config=self.config.dataset_config, caching=True)
        else:
            dataset = self.prepare_dataset(self.config.dataset_config)

        preproc_file = None
        inference_schema_manager = None
        model_path = None
        inference_schemas = []
        self.results = []

        if qcc.STAGE_INFER in pipeline_stages:
            if device_id:
                status, device_ids = self.parse_and_validate_device_ids(device_id)
                # If device_ids supplied are valid, then override the device ids in config
                if status:
                    self.config.inference_config.device_ids = device_ids

            inference_schemas, inference_schema_manager = self.parse_inference_schemas(
                                                    inference_schema_name=inference_schema_name,
                                                    inference_schema_tag=inference_schema_tag)

        # get the pipeline_inputs
        if isinstance(self.config.dataset_config, DatasetConfiguration):
            total_inputs = dataset.get_total_entries()
        else:
            total_inputs = len(dataset)

        # clean model if configured.
        if qcc.STAGE_INFER in pipeline_stages:
            # confirm for inference schemas and estimate space
            if self.config.dataset_config:
                self.confirmation_prompt(inference_schemas, self.config, total_inputs,
                                         inference_schema_manager, cleanup, silent,
                                         use_memory_plugins=self.use_memory_plugins)
            is_custom_op_model, has_quantization_overrides_flag = QACCManager.check_model_for_simplification(
                    inference_schemas)
            model_path, inference_schemas = self.clean_and_simplify_model(inference_schemas=inference_schemas,
                                             onnx_symbol=onnx_symbol, is_custom_op_model=is_custom_op_model,
                                             has_quantization_overrides_flag=has_quantization_overrides_flag)
            if self.config.inference_config:
                # check batchsize to be passed to inference engine
                if self.config.inference_config.inputs_info:
                    inp_dims = self.config.inference_config.inputs_info
                    key_list = list(inp_dims.keys())
                    if len(key_list) == 1:
                        in_node = key_list[0]
                        bs = ModelHelper.get_model_batch_size(model_path, in_node)
                        qacc_file_logger.info(f'Batchsize from Model graph: {bs}')
                        if bs != self.config.info_config.batchsize:
                            # When Model bs != input_bs (supplied) override  input_bs for execution
                            pipeline_cache.set_val(qcc.INTERNAL_EXEC_BATCH_SIZE,
                                                   self.config.info_config.batchsize)
                    else:
                        qacc_file_logger.warning(
                            'Setting batchsize for multiple inputs is currently unsupported')

        # set values in pipeline pipeline_cache
        pipeline_cache.set_val(qcc.PIPELINE_BATCH_SIZE, self.config.info_config.batchsize)
        pipeline_cache.set_val(qcc.PIPELINE_WORK_DIR, self.work_dir)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_INPUT_INFO,
                               self.config.inference_config.inputs_info)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_OUTPUT_INFO,
                               self.config.inference_config.outputs_info)
        pipeline_cache.set_val(qcc.QNN_SDK_DIR, qnn_sdk_dir)

        ret_status, preproc_file = self.execute_pipeline_stages(
            pipeline_stages, inference_schema_manager, model_path, cli_work_dir=work_dir,
            pipeline_start=pipeline_start, pipeline_end=pipeline_end,
            cleanup=cleanup, is_custom_op_model=is_custom_op_model)

        if ret_status:
            qacc_file_logger.info('Pipeline execution interrupted')

        # do comparison of infer outputs across inference schemas if configured.
        if qcc.STAGE_INFER in pipeline_stages and self.config.verifier_config and self.config.verifier_config.enabled \
                and len(inference_schemas) > 1 and preproc_file is not None and (STAGE_INFER_PASS):
            ret_status = self.compare_infer_results(preproc_file)
            if ret_status:
                qacc_logger.error('qacc comparator failed.')

        ret_status = EvaluatorStatus.SUCCESS
        # Constant for all inference schemas
        if self.config.dataset_config:
            ds_name = self.config.dataset_config.name
            batch_size = self.config.info_config.batchsize
            max_inputs = pipeline_cache.get_val(qcc.PIPELINE_MAX_INPUTS)
            qacc_file_logger.info(
                f"Using Dataset: {ds_name} Batch Size: {batch_size} Max Inputs : {max_inputs}")
        if pipeline_end != qcc.STAGE_PREPROC:  # Print Summary
            self.results = []  # Used for Api
            ret_status = self.print_and_write_results(inference_schemas=inference_schemas)
        # delete output files of all stages.
        if qcc.CLEANUP_AT_END == cleanup:
            self.cleanup_files(self.work_dir, stage='all')
        qacc_file_logger.debug(pipeline_cache._pipeline_cache)
        return ret_status, self.results

    def execute_pipeline_stages(self, pipeline_stages, inference_schema_manager, model_path,
                                cli_work_dir, pipeline_start, pipeline_end, cleanup=None,
                                is_custom_op_model=False):
        """Execute pipeline stages."""
        # using global stage variables
        global STAGE_PREPROC_PASS
        global STAGE_INFER_PASS

        compile_only = pipeline_end == qcc.STAGE_COMPILE
        load_compiled_binary_from_dir = pipeline_start == qcc.STAGE_COMPILE

        batchsize = self.config.info_config.batchsize
        # capturing start time of calibration
        start_time = time.time()

        # perform preprocessing for calibration inputs
        # this adds support to supply calibration file with inputs files.
        # These inputs can be filenames other than what is mentioned in inputlist.
        # To use this add calibration section in the dataset.yaml as below:
        # calibration:
        #             type: filename
        #             file: calibration_file.txt
        if isinstance(self.config.dataset_config, DatasetConfig) and (pipeline_end == qcc.STAGE_PREPROC
                     or self.config.inference_config._is_calib_req) and (qcc.STAGE_PREPROC in pipeline_stages):
            calib_dataset_config = copy.deepcopy(self.config.dataset_config)
            calib_dataset = self.prepare_dataset(calib_dataset_config, use_calibration=True,
                             max_samples=self.config.info_config.max_calibration)
            err_status, calib_file = self.preprocess_memory(calib_dataset, True) # TODO: Check this path
            self.calibration_file = calib_file
            if err_status:
                qacc_file_logger.info('Calibration preprocessing failed')
                return 1
            pipeline_cache.set_val(qcc.PIPELINE_CALIB_FILE, calib_file)
            qacc_file_logger.info(
                'Calibration preprocessing complete. calibration file: {}'.format(calib_file))

        elif isinstance(self.config.dataset_config, DatasetConfiguration) and \
                self.config.dataset_config is not None and \
                self.config.dataset_config.calibration_file \
                and self.config.dataset_config.calibration_type == CalibrationType.DATASET \
                and (pipeline_end == qcc.STAGE_PREPROC or
                self.config.inference_config._is_calib_req) \
                and (qcc.STAGE_PREPROC in pipeline_stages):

            # modify the inputlist file to calibration file
            # this is done to execute all the preprocessing plugins
            # using files in calibration file
            calib_dataset_config = copy.deepcopy(self.config.dataset_config)
            calib_dataset_config.inputlist_file = self.config.dataset_config.calibration_file
            calib_dataset_config.max_inputs = self.config.inference_config._max_calib

            # create dataset object with inputlist as calibration file
            calib_dataset = ds.DataSet(dataset_config=calib_dataset_config, caching=True)

            # using batch index 0
            err_status, calib_file = self.preprocess(calib_dataset, True)
            if err_status:
                qacc_file_logger.info('Calibration preprocessing failed')
                return 1
            else:
                # Setting it to RAW as these inputs are already preprocessed
                self.config.dataset_config.calibration_type = CalibrationType.RAW
                self.config.dataset_config.calibration_file = calib_file

                # updating the max calib
                # This is added as in certain scenarios the number of processed outputs
                # could increase or decrease based on processing technique used like
                # in the case of BERT model.
                self.config.inference_config._max_calib = len(open(calib_file).readlines())
                pipeline_cache.set_val(qcc.PIPELINE_CALIB_FILE, calib_file)
                qacc_file_logger.info(
                    'Calibration preprocessing complete. calibration file: {}'.format(calib_file))
        else:
            self.calibration_file = None

        # set calibration time
        self.capture_time(qcc.INTERNAL_CALIB_TIME, start_time)
        start_time = None  # reset start time

        # Preprocessing
        # capturing start time of preprocessing
        start_time = time.time()
        if (qcc.STAGE_PREPROC in pipeline_stages):
            if self.use_memory_plugins:
                dataset = self.prepare_dataset(self.config.dataset_config)
                err_status, preproc_file = self.preprocess_memory(dataset)
            else:
                # create new dataset object
                dataset = ds.DataSet(dataset_config=self.config.dataset_config, caching=True)
                err_status, preproc_file = self.preprocess(dataset)
            if err_status:
                STAGE_PREPROC_PASS = False
                return 1
            else:
                # calibration_file = dataset.get_dataset_calibration()
                STAGE_PREPROC_PASS = self.validate_pipeline_stage(qcc.STAGE_PREPROC, self.work_dir)
                if not STAGE_PREPROC_PASS:
                    qacc_file_logger.info('{} stage validation failed'.format(qcc.STAGE_PREPROC))
                else:
                    pipeline_cache.set_val(qcc.PIPELINE_PREPROC_FILE, preproc_file)
        else:
            # required for idx based dataset access while
            # post processing
            if not self.use_memory_plugins and self.config.dataset_config:
                qacc_file_logger.info('Loading dataset')
                dataset = ds.DataSet(dataset_config=self.config.dataset_config, caching=True)
                dataset.load_dataset()

            # if cli preproc not supplied and preprocessing stage is skipped then treat
            # input list as preproc. This is used for supporting scenarios where only
            # preprocessed data is available.
            qacc_logger.info('Loading preprocessed data')
            # To support AUTO team where generally only preprocessed data is available
            if self.use_memory_plugins:
                dataset = self.prepare_dataset(self.config.dataset_config)
                err_status, preproc_file = self.preprocess_memory(dataset)
            else:
                preproc_file = dataset.get_input_list_file()
            STAGE_PREPROC_PASS = True
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_FILE, preproc_file)

        # set preprocessing time
        self.capture_time(qcc.INTERNAL_PREPROC_TIME, start_time)
        start_time = None  # reset start time
        if qcc.STAGE_INFER in pipeline_stages:

            if self.config.inference_config is None:
                qacc_logger.error('No inference section found in model config.'
                                  'Use -pipeline-start and -pipeline-end flag to skip inference')
                return 1

            # get all the inference schemas
            inference_schemas = self.config.inference_config.inference_schemas

            # run a schedule in distributed manner
            for schedule in inference_schema_manager.get_schedule():

                # get the scheduled inference schemas
                # schedule format: [(inference_schema_idx, device_id), ... , (inference_schema_idx, device_id)]
                # example: [[(0,-1), (1,0), (2,1)], [(3,0), (4,1)]]
                schd_inference_schemas = []

                for schd_inference_schema in schedule:
                    inference_schema_idx = schd_inference_schema[0]
                    inference_schema = inference_schemas[inference_schema_idx]
                    device_id = schd_inference_schema[1]

                    # store in schd_inference_schemas
                    inference_schema_tuple = (inference_schema_idx, inference_schema, device_id)
                    schd_inference_schemas.append(inference_schema_tuple)

                # run inference sequentially for QNN
                # TODO: Parallelize the inference for QNN backends.
                for inference_schema_idx, inference_schema, device_id in schd_inference_schemas:
                    self.run_schedule_in_parallel(
                        preproc_file, dataset, inference_schema_idx, inference_schema,
                        device_id, pipeline_stages,
                        model_path=inference_schema._model_path,
                        compile_only_flag=compile_only,
                        load_compiled_binary_from_dir_flag=load_compiled_binary_from_dir,
                        cleanup=cleanup)

            # marking infer stage passed
            STAGE_INFER_PASS = self.validate_pipeline_stage(qcc.STAGE_INFER, self.work_dir)
            if not STAGE_INFER_PASS:
                qacc_file_logger.info('{} stage validation failed'.format(qcc.STAGE_INFER))

        # delete preprocessed outputs
        if STAGE_PREPROC_PASS and (qcc.CLEANUP_INTERMEDIATE == cleanup):
            self.cleanup_files(self.work_dir, qcc.STAGE_PREPROC)

        # terminate pipeline if only preprocessing is configured
        if STAGE_PREPROC_PASS and pipeline_end == qcc.STAGE_PREPROC:
            # squash preproc files
            preproc_dir = self.get_output_path(self.work_dir, qcc.STAGE_PREPROC)
            if os.path.exists(preproc_dir):
                preproc_file = self.get_output_path(preproc_dir, qcc.QNN_PROCESSED_OUTFILE)
                pipeline_cache.set_val(qcc.PIPELINE_PREPROC_DIR, preproc_dir)
                pipeline_cache.set_val(qcc.PIPELINE_PREPROC_FILE, preproc_file)

            if not STAGE_INFER_PASS:
                return 0, preproc_file

        # setting paths and starting metric evaluation
        for inference_schema_idx, inference_schema in enumerate(inference_schemas):
            inference_schema_name = inference_schema.get_inference_schema_name()
            if self.inference_schema_run_status[inference_schema_name]['status'] in [
                    qcc.SCHEMA_POSTPROC_FAIL, qcc.SCHEMA_INFER_FAIL
            ]:
                continue

            # setting postprocessing file
            if qcc.STAGE_POSTPROC in pipeline_stages and self.config.postprocessing_config \
                    and self.inference_schema_run_status[inference_schema_name][
                'status'] == qcc.SCHEMA_POSTPROC_SUCCESS:
                # validate postproc stage
                if not self.validate_pipeline_stage(qcc.STAGE_POSTPROC, self.work_dir):
                    qacc_file_logger.info('{} stage validation failed'.format(qcc.STAGE_POSTPROC))
                else:
                    postproc_file = pipeline_cache.get_val(qcc.PIPELINE_POSTPROC_FILE,
                                                           inference_schema_name)
            else:
                postproc_file = pipeline_cache.get_val(qcc.PIPELINE_INFER_FILE,
                                                       inference_schema_name)

        return 0, preproc_file

    @staticmethod
    def confirmation_prompt(inference_schemas, config, num_samples,
                            inference_schema_manager, cleanup, silent, use_memory_plugins=False):
        """Prompts the user with.

        - number of inference_schemas
        - total space required in Distributed Strategies
        - disabling of comparator
        """

        # disable comparator for intermediate delete
        # calculate based on delete option
        def log_disk_usage(size, msg):
            if size >= 1024:
                qacc_logger.info(msg + '  - {} GB'.format(round(size / 1024, 2)))
            else:
                qacc_logger.info(msg + '  - {} MB'.format(round(size, 2)))

        cleanup_inter = False
        if qcc.CLEANUP_INTERMEDIATE == cleanup:
            qacc_logger.info('Disabling comparator as -cleanup intermediate is selected')
            if config.verifier_config:
                config.verifier_config.enabled = False
            cleanup_inter = True

        num_inference_schemas = len(inference_schemas)
        total_req_sizes = QACCManager.get_estimated_req_size(num_inference_schemas, config, num_samples,
                                                             cleanup_inter, use_memory_plugins=use_memory_plugins)
        qacc_logger.info('Total inference schemas : {}'.format(num_inference_schemas))
        if not use_memory_plugins:
            qacc_logger.info('Total inputs for execution: {} and calibration: {}'.format(
                config.dataset_config.max_inputs, config.inference_config._max_calib))
        preproc_size, calib_size, infer_size = total_req_sizes[0], total_req_sizes[1], \
                                               total_req_sizes[2]
        log_disk_usage(preproc_size + calib_size + infer_size, 'Approximate disk usage')
        if not cleanup_inter and inference_schema_manager.get_schedule() is not None:
            inference_schemas = len(
                inference_schema_manager.get_schedule()[0])  # get len of first schedule
            size = ((infer_size / num_inference_schemas) *
                    inference_schemas) + calib_size + preproc_size
            log_disk_usage(size, 'Approximate disk usage if -cleanup intermediate option is used')

        user_input = input('Do you want to continue execution? (yes/no) :').lower() \
            if not silent else 'y'
        if user_input not in ['yes', 'y']:
            qacc_logger.info('User terminated execution')
            sys.exit(1)

    @staticmethod
    def get_estimated_req_size(num_inference_schemas, config, num_samples, cleanup_inter=False, use_memory_plugins=False):
        """Estimate the required size for Distributed strategy.

        Returns:
            total_req_sizes: [preproc, calib, infer]
        """

        def _parse_range(index_str):
            if len(index_str) == 0:
                return []
            nums = index_str.split("-")
            assert len(nums) <= 2, 'Invalid range in calibration file '
            start = int(nums[0])
            end = int(nums[-1]) + 1
            return range(start, end)

        if not hasattr(config, 'inference_config'):
            return [0, 0]  # inference section not available for calculation

        size_dict = {
            'bool': 1,
            'float': 4,
            'float32': 4,
            'float16': 2,
            'float64': 8,
            'int8': 1,
            'int16': 2,
            'int32': 4,
            'int64': 8
        }

        # calculate preproc output size
        input_dims = config.inference_config.inputs_info
        qacc_file_logger.debug('input_dims type{} value {}'.format(type(input_dims), input_dims))
        preproc_size = 0
        batch_size = 1
        for in_node, val in input_dims.items():
            qacc_file_logger.debug('val {} for node {}'.format(val, in_node))
            if val[0] not in size_dict:  # datatype
                qacc_file_logger.error('input type {} not supported in input_info '
                                       'in config'.format(val[0]))
            preproc_size_per_out = 1
            tensor_shape = val[1]
            batch_dim = val[2]
            if batch_dim:
                batch_size = tensor_shape[batch_dim]
            for idx, v in enumerate(val[1]):  # tensor shape
                preproc_size_per_out *= v
            preproc_size += preproc_size_per_out * size_dict[val[0]]
        # (inputs/batch_size) --> num of preproc files
        total_preproc_size = preproc_size * (num_samples / batch_size)
        qacc_file_logger.info(f'preproc size: {round(total_preproc_size / (1024 * 1024), 3)} MB')
        # calculate calibration output size
        calib_size = 0
        if not use_memory_plugins:
            if config.dataset_config.calibration_file and config.inference_config._is_calib_req:
                calib_file = config.dataset_config.calibration_file
                if config.dataset_config.calibration_type == CalibrationType.DATASET \
                        or config.dataset_config.calibration_type == CalibrationType.RAW:
                    calib_inputs = sum(1 for input in open(calib_file))
                else:
                    cf = open(calib_file, 'r')
                    indexes_str = cf.read().replace('\n', ',').strip()
                    indexes = sorted(set(chain.from_iterable(map(_parse_range,
                                                                indexes_str.split(",")))))
                    cf.close()
                    calib_inputs = len(indexes)

                if -1 != config.inference_config._max_calib:
                    calib_inputs = min(calib_inputs, config.inference_config._max_calib)
                else:
                    config.inference_config._max_calib = calib_inputs
                calib_size = (calib_inputs / batch_size) * preproc_size
                qacc_file_logger.info('calib_inputs {} preproc_size {} batch_size {}'.format(
                    calib_inputs, preproc_size, batch_size))
            else:
                config.inference_config._max_calib = 0
        # Update the Pipeline cache with New Values after pre-processing
        pipeline_cache.set_val(qcc.PIPELINE_MAX_CALIB, config.inference_config._max_calib)
        # calculating infer output size
        output_dims = config.inference_config.outputs_info
        qacc_file_logger.debug('output_dims type{} value {}'.format(type(output_dims), output_dims))
        infer_size = 0
        batch_size = 1
        for out_node, val in output_dims.items():
            qacc_file_logger.debug('val {} for node {}'.format(val, out_node))
            if val[0] not in size_dict:
                qacc_file_logger.error('output type {} not supported in outputs_info '
                                       'in config'.format(val[0]))
            infer_size_per_out = 1
            for idx, v in enumerate(val[1]):
                if 0 == idx:
                    batch_size = v
                infer_size_per_out *= v
            infer_size += infer_size_per_out * size_dict[val[0]]

        infer_size = infer_size * num_inference_schemas * (
            num_samples / batch_size)  # (inputs/batch_size) --> num of infer files

        MB_divider = (1024 * 1024)
        total_req_sizes = [
            total_preproc_size / MB_divider, calib_size / MB_divider, infer_size / MB_divider
        ]
        return total_req_sizes

    def run_schedule_in_parallel(self, preproc_file, dataset, inference_schema_idx,
                                 inference_schema, device_id, pipeline_stages, model_path,
                                 compile_only_flag=False, load_compiled_binary_from_dir_flag=False,
                                 cleanup=''):
        """Run in parallel."""
        inference_schema_name = inference_schema.get_inference_schema_name()

        if self.use_memory_plugins:
            # Catch failures during to plugin creation at earlier stage before inference
            metric_objs = []
            postprocessor_objs = []
            adapter_objs = []
            if self.config.adapter_config is not None:
                adapter_objs = ComponentRegistry.get_components_from_configs([self.config.adapter_config])
            if qcc.STAGE_POSTPROC in pipeline_stages and self.config.postprocessing_config \
                and self.config.postprocessing_config.postprocessing_plugin_list:
                # Do Postprocessing only when it is configured within config file.
                postprocessor_configs = [PostProcessorConfig(name=b.name, params=b.params)
                            for b in self.config.postprocessing_config.postprocessing_plugin_list]
                postprocessor_objs = ComponentRegistry.get_components_from_configs(postprocessor_configs)
            # Setup metric plugins if configured
            if qcc.STAGE_METRIC in pipeline_stages and self.config.metrics_config \
                    and self.config.metrics_config.metrics_plugin_list:
                metric_configs = [MetricConfig(name=b.name, params=b.params)
                            for b in self.config.metrics_config.metrics_plugin_list]
                metric_objs = ComponentRegistry.get_components_from_configs(metric_configs)
        qacc_file_logger.info(
            'Pipeline Execution - Inference schema: {} running on device-id: {}'.format(
                inference_schema_name, device_id if not device_id == -1 else 'Not AIC'))

        err_status, infer_fail_stage, infer_file, execution_time = self.infer(
            model_path, preproc_file, inference_schema, dataset, device_id, inference_schema_name,
            compile_only=compile_only_flag, load_binary_from_dir=load_compiled_binary_from_dir_flag)

        if err_status:
            qacc_logger.error('({}) inference failed'.format(inference_schema_name))
            self.inference_schema_run_status[inference_schema_name] = {
                'status': qcc.SCHEMA_INFER_FAIL
            }
            self.inference_schema_run_status[inference_schema_name][
                'infer_stage_status'] = infer_fail_stage
            # exit the  thread
            return 1
        else:
            self.inference_schema_run_status[inference_schema_name] = {
                'status': qcc.SCHEMA_INFER_SUCCESS
            }
            self.inference_schema_run_status[inference_schema_name][
                'infer_stage_status'] = infer_fail_stage

        # set quantization, compilation and infer time
        pipeline_cache.set_val(qcc.INTERNAL_QUANTIZATION_TIME, execution_time[0],
                               inference_schema_name)
        pipeline_cache.set_val(qcc.INTERNAL_COMPILATION_TIME, execution_time[1],
                               inference_schema_name)
        pipeline_cache.set_val(qcc.INTERNAL_INFER_TIME, execution_time[2], inference_schema_name)

        # Post processing
        # capturing start time of post processing
        start_time = time.time()
        metrics_result = {}
        if self.use_memory_plugins and any(
            [True for stage in [qcc.STAGE_POSTPROC, qcc.STAGE_METRIC] if stage in pipeline_stages]):
            dir_name = self.get_output_path(dir=self.work_dir, type=qcc.STAGE_INFER,
                                            inference_schema_name=inference_schema_name)
            infer_ds_path = self.get_output_path(dir=dir_name, type=qcc.INFER_OUTFILE)
            err_status, metrics_result = self.post_inference(
                inference_schema, dataset, infer_ds_path, inference_schema_name, pipeline_stages,
                adapter_objs=adapter_objs, postprocessor_objs=postprocessor_objs, metric_objs=metric_objs)
            self.inference_schema_run_status[inference_schema_name]['metrics'] = metrics_result
        else:
            if qcc.STAGE_POSTPROC in pipeline_stages:
                err_status, postproc_file = self.postprocess(inference_schema_idx, dataset,
                                                             infer_file, inference_schema_name)
                if err_status:
                    qacc_logger.error('({}) post processing failed'.format(inference_schema_name))
                    self.inference_schema_run_status[inference_schema_name][
                        'status'] = qcc.SCHEMA_POSTPROC_FAIL
                    return 1
                else:
                    self.inference_schema_run_status[inference_schema_name][
                        'status'] = qcc.SCHEMA_POSTPROC_SUCCESS

                # set post processing time
                self.capture_time(qcc.INTERNAL_POSTPROC_TIME, start_time, inference_schema_name)
                start_time = None  # reset start time

                # delete intermediate inference output files if configured.
                if qcc.CLEANUP_INTERMEDIATE == cleanup and self.config.postprocessing_config:
                    self.cleanup_files(self.work_dir, qcc.STAGE_INFER, inference_schema_name)

            # Metrics
            # capturing start time of infer
            start_time = time.time()
            if qcc.STAGE_METRIC in pipeline_stages:
                ret_status = self.evaluate_metrics(inference_schema_idx, dataset, postproc_file,
                                                   inference_schema)
                if ret_status:
                    qacc_logger.error(
                        '({}) Metrics evaluation failed. See qacc.log for more details.'.format(
                            inference_schema_name))

                # delete postprocessed output files if configured.
                if qcc.CLEANUP_INTERMEDIATE == cleanup:
                    if self.config.postprocessing_config:
                        self.cleanup_files(self.work_dir, qcc.STAGE_POSTPROC, inference_schema_name)
                    else:
                        self.cleanup_files(self.work_dir, qcc.STAGE_INFER, inference_schema_name)

            # set metric time
            self.capture_time(qcc.INTERNAL_METRIC_TIME, start_time, inference_schema_name)
            start_time = None  # reset start time

    def cleanup_files(self, work_dir, stage, inference_schema_name=None):
        """Cleanup output files generated during various stages of the
        pipeline.
        """
        # check if cleaning all stages
        cleanup_all = ('all' == stage)

        # cleanup preproc outputs
        if qcc.STAGE_PREPROC == stage or cleanup_all:
            qacc_logger.info('Cleaning up pre-processed outputs')
            shutil.rmtree(self.get_output_path(work_dir, qcc.STAGE_PREPROC), ignore_errors=True)
            shutil.rmtree(self.get_output_path(work_dir, qcc.STAGE_PREPROC_CALIB),
                          ignore_errors=True)

        # cleanup infer outputs
        if qcc.STAGE_INFER == stage or cleanup_all:
            qacc_logger.info('Cleaning up inference outputs')
            dir = self.get_output_path(work_dir, qcc.STAGE_INFER, inference_schema_name)

            infer_files = []
            file_types = ['bin', 'raw']
            for file_type in file_types:
                infer_files.extend(glob.glob(dir + '/**/*.' + file_type, recursive=True))
            for file in infer_files:
                if qcc.INFER_SKIP_CLEANUP in file:
                    continue
                os.remove(file)

        # cleanup postproc outputs
        if qcc.STAGE_POSTPROC == stage or cleanup_all:
            qacc_logger.info('Cleaning up post-processed outputs')
            shutil.rmtree(self.get_output_path(work_dir, qcc.STAGE_POSTPROC, inference_schema_name),
                          ignore_errors=True)

    def validate_pipeline_stage(self, stage, work_dir):
        """Performs validation on the pipeline stage results.

        Returns:
             True: if the results are valid, False otherwise
        """
        exit_execution = False
        if stage == qcc.STAGE_PREPROC:
            file_types = ['raw']
        elif stage == qcc.STAGE_POSTPROC:
            file_types = ['txt']
        elif stage == qcc.STAGE_INFER:
            file_types = ['bin', 'raw']

        dir = os.path.join(work_dir, stage)
        if os.path.exists(dir):
            files = []

            # fetch all files based on extension
            for file_type in file_types:
                files.extend(glob.glob(dir + '/**/' + '*.' + file_type, recursive=True))

            # if no files generated mark validation failed
            if 0 == len(files):
                qacc_file_logger.warning('No files found to validate')
                if exit_execution:
                    return False

            # check all files
            for file in files:
                # if file size zero mark validation failed
                if os.path.getsize(file) == 0:
                    qacc_file_logger.warning('File size zero: {}'.format(file))
                    if exit_execution:
                        return False

        # if didn't return False till this point means validation passed
        return True

    def capture_time(self, key, start_time, nested_key=None):
        pipeline_cache.set_val(key, time.time() - start_time, nested_key)

    def copy_pipeline_stage_execution_time(self, inference_schemas, pipeline_stages):

        def get_time_from_dict(key, nested_key=None):
            if pipeline_cache.get_val(key, nested_key) is None:
                return 0
            else:
                return pipeline_cache.get_val(key, nested_key)

        # common execution time
        qacc_file_logger.info('Preprocessing Time Summary:')
        preproc_time = get_time_from_dict(qcc.INTERNAL_CALIB_TIME) + get_time_from_dict(
            qcc.INTERNAL_PREPROC_TIME)
        summary = [['Preprocessing', str(datetime.timedelta(seconds=preproc_time))]]

        table = tabulate(summary, headers=['Preprocessing', 'Time (hh:mm:ss)'])
        console(table)
        qacc_file_logger.info(table)

        if qcc.STAGE_INFER in pipeline_stages:
            qacc_file_logger.info('Inference schema Wise Time Summary (hh:mm:ss):')
            summary = []
            for inference_schema_idx, inference_schema in enumerate(inference_schemas):
                entry = []
                total_time = 0
                inference_schema_name = inference_schema.get_inference_schema_name()
                entry.append(inference_schema_name)
                entry.append(
                    str(
                        datetime.timedelta(seconds=get_time_from_dict(
                            qcc.INTERNAL_QUANTIZATION_TIME, inference_schema_name))))
                entry.append(
                    str(
                        datetime.timedelta(seconds=get_time_from_dict(qcc.INTERNAL_COMPILATION_TIME,
                                                                      inference_schema_name))))
                entry.append(
                    str(
                        datetime.timedelta(seconds=get_time_from_dict(qcc.INTERNAL_INFER_TIME,
                                                                      inference_schema_name))))
                entry.append(
                    str(
                        datetime.timedelta(seconds=get_time_from_dict(qcc.INTERNAL_POSTPROC_TIME,
                                                                      inference_schema_name))))
                entry.append(
                    str(
                        datetime.timedelta(seconds=get_time_from_dict(qcc.INTERNAL_METRIC_TIME,
                                                                      inference_schema_name))))
                phases = [
                    qcc.INTERNAL_QUANTIZATION_TIME, qcc.INTERNAL_COMPILATION_TIME,
                    qcc.INTERNAL_INFER_TIME, qcc.INTERNAL_POSTPROC_TIME, qcc.INTERNAL_METRIC_TIME
                ]
                for phase in phases:
                    total_time += get_time_from_dict(phase, inference_schema_name)
                entry.append(str(datetime.timedelta(seconds=total_time)))
                summary.append(entry)
            headers = [
                'Inference schema', 'Quantization', 'Compilation', 'Inference', 'Postprocessing',
                'Metrics', 'Total'
            ]
            table = tabulate(summary, headers=headers)
            profile_csv_path = self.get_output_path(self.work_dir, qcc.PROFILING_TABLE_CSV)
            self.write2csv(profile_csv_path, summary, header=headers)
            console(table)
            qacc_file_logger.info(table)

    def get_output_path(self, dir, type, inference_schema_name=None):
        """Returns the output directory for various stages of the pipeline."""
        # preprocessing or infer file or metric file
        if type in [
                qcc.STAGE_PREPROC, qcc.INFER_OUTFILE, qcc.PROCESSED_OUTFILE,
                qcc.QNN_PROCESSED_OUTFILE, qcc.STAGE_PREPROC_CALIB, qcc.STAGE_INFER,
                qcc.STAGE_POSTPROC, qcc.STAGE_METRIC, qcc.PROFILING_TABLE_CSV,
                qcc.RESULTS_TABLE_CSV, qcc.INFER_RESULTS_FILE, qcc.DATASET_DIR, qcc.INPUT_LIST_FILE,
                qcc.CALIB_FILE
        ] and inference_schema_name is None:
            return os.path.join(dir, type)

        # inference or postprocessing
        elif type in [qcc.STAGE_INFER, qcc.STAGE_POSTPROC, qcc.STAGE_METRIC]:
            return os.path.join(dir, type, inference_schema_name)

        # binary
        elif type == qcc.BINARY_PATH:
            return os.path.join(dir, qcc.STAGE_INFER, inference_schema_name, 'temp')

    def filter_inference_schemas(self,
            inference_schemas: list[InferenceSchemaConfiguration],
            inference_schema_name: Optional[str] = None,
            inference_schema_tag: Optional[str] = None) -> list[InferenceSchemaConfiguration]:
        """Filter inference schemas based on provided name or tag.

        Args:
            inference_schemas: List of inference schemas to filter.
            inference_schema_name: Optional name to filter schemas by.
            inference_schema_tag: Optional tag to filter schemas by.

        Returns:
            List of filtered inference schemas.
        """
        # select inference schema based on supplied args
        if inference_schema_name:
            inference_schemas = [p for p in inference_schemas if p.name == inference_schema_name]
            if len(inference_schemas) == 0:
                qacc_logger.error('Invalid inference schema name in -inference_schema option')
                sys.exit(1)
        if inference_schema_tag:
            inference_schemas = [
                p for p in inference_schemas if p.tag is not None and inference_schema_tag in p.tag
            ]
            if len(inference_schemas) == 0:
                qacc_logger.error('Invalid inference schema tag in -inference_schema_tag option')
                sys.exit(1)
        return inference_schemas

    def write2csv(self, fname, rows, header):
        # check all rows have same length
        assert len(header) == len(rows[0])
        with open(fname, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(rows)

    def update_relative_paths(self, preproc_file, work_dir):
        """Create a new preproc file and modify the relative paths to absolute
        paths.
        """
        updated_preproc_file = os.path.join(work_dir, "updated_input_list.txt")
        original_list_dir = os.path.dirname(os.path.abspath(preproc_file))
        with open(updated_preproc_file, "w") as write_file, \
             open(preproc_file, "r") as read_file:

            for line in read_file:
                write_file.write(os.path.join(original_list_dir, line))

        return updated_preproc_file

    def get_pipeline_stages_from_config(self, config):
        pipeline_stages = [
            qcc.STAGE_PREPROC, qcc.STAGE_COMPILE, qcc.STAGE_INFER, qcc.STAGE_POSTPROC,
            qcc.STAGE_METRIC
        ]
        pipeline_start = 'infer'
        pipeline_end = 'infer'
        if config.preprocessing_config:
            pipeline_start = qcc.STAGE_PREPROC
            pipeline_end = qcc.STAGE_PREPROC
        if config.inference_config:
            pipeline_end = qcc.STAGE_INFER
        if config.postprocessing_config:
            pipeline_end = qcc.STAGE_POSTPROC
        if config.metrics_config and config.metrics_config.metrics_plugin_list:
            pipeline_end = qcc.STAGE_METRIC
        pipeline_stages = pipeline_stages[pipeline_stages.index(pipeline_start):pipeline_stages.
                                          index(pipeline_end) + 1]

        return pipeline_stages, pipeline_start, pipeline_end

    @classmethod
    def check_model_for_simplification(cls, inference_schemas):
        """Skip model simplification when the model contains a custom op."""
        # Assume custom op and quantization_overrides not present
        is_custom_op_model = False
        has_quantization_overrides_flag = False

        for inference_schema in inference_schemas:
            converter_parameters = inference_schema.converter_params.model_fields_set \
                if inference_schema.converter_params else {}
            # Any of the inference schema contains custom op (Ideally all inference
            # schemas should contain custom op field if model contains custom op)
            if converter_parameters:
                is_custom_op_model = is_custom_op_model or len(
                    set(qcc.CUSTOM_OP_FLAGS).intersection(set(converter_parameters))) > 0
            if is_custom_op_model:
                # can do early exit with single occurrence
                break

        for inference_schema in inference_schemas:
            converter_parameters = inference_schema.converter_params.model_fields_set \
                if inference_schema.converter_params else {}
            has_quantization_overrides_flag = has_quantization_overrides_flag or True \
                    if 'quantization_overrides' in converter_parameters else False
            if has_quantization_overrides_flag:
                # can do early exit with single occurrence
                break
        return is_custom_op_model, has_quantization_overrides_flag
