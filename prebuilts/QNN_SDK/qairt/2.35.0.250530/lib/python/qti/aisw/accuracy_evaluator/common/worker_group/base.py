# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import csv
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from qti.aisw.accuracy_evaluator.common.utilities import TopKTracker, timer_decorator
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.qacc.config_definitions import (
    InferenceEngineType,
    InferenceSchemaConfiguration,
)
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.tools.core.modules.api.definitions.common import Model
from qti.aisw.tools.core.utilities.comparators.factory import Comparator
from qti.aisw.tools.core.utilities.data_processing import (
    Metric,
    OutputAdapter,
    PostProcessor,
    PreProcessor,
    Representation,
)
from qti.aisw.tools.core.utilities.data_processing.core.transformations import (
    ProcessorChainExecutor,
)
from qti.aisw.tools.core.utilities.data_processing.datasets.base import IndexableDataset


EVALUATOR_IO_NODE_INFO = Dict[str, List[Any]]
DEVICE_ID_TYPING = List[str | int] | int | str


class WorkerGroup:
    """A base class for worker groups that handle the inference and evaluation of models.
    This class provides a framework for preprocessing, inference, postprocessing,
    metric calculation, and comparator operations for model evaluation.
    """
    def __init__(
        self,
        model: Model | str,
        work_dir: str | os.PathLike,
        inference_schema: InferenceSchemaConfiguration,
        dataset: IndexableDataset,
        preprocessors_objs: Optional[list[PreProcessor]] = None,
        postprocessor_objs: Optional[list[PostProcessor]] = None,
        adapter_objs: Optional[list[OutputAdapter]] = None,
        metric_objs: Optional[list[Metric]] = None,
        comparator_objs: Optional[Comparator] = None,
        device_id: DEVICE_ID_TYPING = None,
        reference_platform: Optional[InferenceSchemaConfiguration] = None,
        input_info: Optional[EVALUATOR_IO_NODE_INFO] = None,
        output_info: Optional[EVALUATOR_IO_NODE_INFO] = None,
        dump_stages: Optional[list] = None,
        comparator_enabled: bool = False,
        num_most_deviating_samples: Optional[int] = 1):
        """Initialize the WorkerGroup with necessary configurations.

        Args:
            model: The model to be evaluated.
            work_dir: The working directory for output files.
            inference_schema: The inference schema configuration.
            dataset: The dataset for evaluation.
            preprocessors_objs: List of preprocessors to be used.
            postprocessor_objs: List of postprocessors to be used.
            adapter_objs: List of adapters to be used.
            metric_objs: List of metrics to be used for evaluation.
            comparator_objs: Comparator objects for comparing outputs.
            device_id: Device ID for inference (ignored for CPU and reference schemas).
            reference_platform: Reference platform configuration for comparison.
            input_info: Input information for the model.
            output_info: Output information for the model.
            dump_stages: List of stages to dump outputs.
            comparator_enabled: Whether comparator is enabled for comparison.
            num_most_deviating_samples: Number of most deviating samples to track.
        """
        self.model = model
        self.inference_schema = inference_schema
        self.inference_schema_name = inference_schema.get_inference_schema_name()
        self.dataset = dataset
        self.preprocessors_objs = preprocessors_objs
        self.postprocessor_objs = postprocessor_objs
        self.adapter_objs = adapter_objs
        self.metric_objs = metric_objs
        self.comparator_objs = comparator_objs

        self.work_dir = work_dir
        self.infer_dir = os.path.join(
            self.work_dir, "infer", self.inference_schema_name
        )
        self.device_id = device_id  # Ignored for CPU backend and Reference schemas
        self.input_info = input_info
        self.output_info = output_info
        self.reference_platform = reference_platform
        self.comparator_enabled = comparator_enabled
        self.dump_stages = dump_stages if dump_stages else []
        self.comparator_score_dict = {}
        self.metrics_score_dict = {}
        self.output_names = list(self.output_info.keys())
        self.input_names = list(self.input_info.keys())
        self.num_most_deviating_samples = num_most_deviating_samples
        self._setup()
        self.validate()
        if self.comparator_enabled and self.comparator_objs:
            keep_largest = (
                True
                if self.comparator_objs[0]._interpretation_strategy == "max"
                else False
            )
            self.most_deviating_input_tracker = TopKTracker(
                k=self.num_most_deviating_samples, keep_largest=keep_largest
            )
            self.comparator_score_dict = {
                output_name: [] for output_name in self.output_names
            }
            self.comparator_score_dict['most_deviating_indices'] = []
            if self.reference_platform.name == InferenceEngineType.QNN:
                self.comparator_out_frmt = "Result_{}/{}.raw"
            else:
                self.comparator_out_frmt = "{}_{}.raw"
            self.ref_out_dir = os.path.join(
                self.work_dir,
                "infer",
                self.reference_platform.get_inference_schema_name(),
            )

    def validate(self):
        """Validate logic for workergroup"""
        #  Validate that the number of comparators matches the number of outputs.
        if len(self.comparator_objs) != len(self.output_names):
            raise ValueError("Number of comparators and outputs do not match.")

    def _setup(self):
        """Set up preprocessors and postprocessors based on the provided configurations."""
        if self.preprocessors_objs:
            preproc_out_dir = os.path.join(self.work_dir, qcc.STAGE_PREPROC)
            dump_filelist = dump_outputs = (
                True if qcc.STAGE_PREPROC in self.dump_stages else False
            )
            self.preprocessor = ProcessorChainExecutor(
                self.preprocessors_objs,
                dump_outputs=dump_outputs,
                output_dir=preproc_out_dir,
                node_names=self.input_names,
                dump_filelist=dump_filelist,
            )
        else:
            self.preprocessor = None

        if self.postprocessor_objs:
            postproc_out_dir = os.path.join(
                self.work_dir, qcc.STAGE_POSTPROC, self.inference_schema_name
            )
            dump_outputs = True if qcc.STAGE_POSTPROC in self.dump_stages else False
            self.postprocessor = ProcessorChainExecutor(
                self.postprocessor_objs,
                dump_outputs=dump_outputs,
                output_dir=postproc_out_dir,
            )
        else:
            self.postprocessor = None
        self.setup_inference_engine()

    def setup_inference_engine(self):
        """Set up the inference engine. This method is intended to be overridden by subclasses."""
        pass

    def teardown(self):
        """Teardown resources used by the worker group."""
        pass

    @abstractmethod
    def infer(input_sample: Representation) -> Representation:
        """Perform inference on the input sample.

        Args:
            input_sample: The input sample to be processed.

        Returns:
            The output of the inference.
        """
        pass

    def get_deviant_indices_with_scores(self) -> list[(float, int)]:
        """Get the top K most deviating indices with their corresponding scores.

        Returns:
            The top K most deviating indices and their scores.
        """
        return self.most_deviating_input_tracker.get_top_k()

    def get_deviant_indices(self) -> list[int]:
        """Get the indices of the most deviating samples.

        Returns:
            The indices of the most deviating samples.
        """
        return self.most_deviating_input_tracker.get_top_k_indices()

    def compare_infer_results(self, model_output: Representation):
        """Compare the inference results with the reference outputs.

        Args:
            model_output: The output from the model inference.
        """
        idx = model_output._idx
        comp_score = []
        for out_i, output_name in enumerate(self.output_names):
            # Get the reference output from disk to memory.
            if self.reference_platform.name == InferenceEngineType.QNN:
                ref_out_file_path = os.path.join(
                    self.ref_out_dir, self.comparator_out_frmt.format(idx, output_name)
                )
            else:
                ref_out_file_path = os.path.join(
                    self.ref_out_dir, self.comparator_out_frmt.format(output_name, idx)
                )
            ref_output = np.fromfile(
                ref_out_file_path, dtype=self.output_info[output_name][0]
            ).reshape(self.output_info[output_name][1])
            result = self.comparator_objs[out_i].compare([model_output.data[out_i]], [ref_output])
            comp_score.append(result)
            self.comparator_score_dict[output_name].append((idx, result))
        tracker_score = round(np.array(comp_score).mean(), 4)
        self.most_deviating_input_tracker.add((tracker_score, idx))

    @timer_decorator
    def execute_pipeline_stages(self,
            pipeline_stages: list[str]) -> tuple[int, dict[str, float], dict[str, float]]:
        """Execute the pipeline stages for preprocessing, inference, postprocessing,
        and metric calculation.

        Args:
            pipeline_stages: List of stages to execute.

        Returns:
            A tuple containing the status of the execution, metrics scores, and comparator scores.
        """
        ret_status = qcc.SCHEMA_INFER_SUCCESS
        total_sample_count = len(self.dataset)
        log_frequency = max(int(total_sample_count / 10), 1)
        sample_completed = 0

        try:
            for ret_status, data in self.preprocessed_dataset():
                if qcc.STAGE_INFER in pipeline_stages:
                    try:
                        inference_output = self.infer(data)
                        inference_output.data = [
                            np.asarray(inference_output.data[out_idx].reshape(out_info[1]),
                                            dtype=out_info[0], order="C")
                            for out_idx, out_info in enumerate(self.output_info.values())
                            ]
                    except Exception as e:
                        qacc_file_logger.error(f"Error in Inference: {e}")
                        ret_status = qcc.SCHEMA_INFER_FAIL
                        break

                ret_status = self.post_inference(pipeline_stages=pipeline_stages,
                                    data=inference_output, ret_status=ret_status)
                sample_completed += 1
                if sample_completed % log_frequency == 0:
                    qacc_logger.info(
                        f"Evaluation Status {self.inference_schema_name} -> "
                        f"{sample_completed}/{total_sample_count} samples completed.")
                if qcc.get_inference_schema_status(ret_status) != 'Success':
                    break

            execution_status = qcc.get_inference_schema_status(ret_status) == 'Success'
            if execution_status:
                self._finalize_metrics_and_comparator(pipeline_stages)
        except Exception as e:
            qacc_file_logger.error(
                f"Failed to execute_pipeline_stages for {self.inference_schema_name}. Reason: {e}"
            )
        finally:
            self.teardown()

        qacc_file_logger.info(
            f"Completed Evaluation of inference_schema={self.inference_schema_name}"
        )
        return ret_status, self.metrics_score_dict, self.comparator_score_dict

    def _dump_comparator_scores_sample_wise(self):
        """Dumps comparator scores for each sample into a CSV file, with one row per sample.
        Each row includes the sample index and the corresponding score for each output.
        """
        csv_path = os.path.join(self.infer_dir, 'comparator_scores_sample_wise.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header row
            writer.writerow(['idx'] + [f'score ({name})' for name in self.output_names])

            # Get the list of indices from the first output (assuming all have the same indices)
            sample_indices = [idx for idx, _ in self.comparator_score_dict[self.output_names[0]]]

            # Iterate through each index
            for sample_idx in sample_indices:
                row = [sample_idx]

                # For each output, find the corresponding score
                for name in self.output_names:
                    scores = [score[0]
                        for i, score in self.comparator_score_dict[name] if i == sample_idx]
                    row.append(scores[0] if scores else None)  # Use the first (and only) match

                # Write the row to the CSV
                writer.writerow(row)

    def preprocessed_dataset(self):
        """Preprocess the dataset and yield the preprocessed data.

        Yields:
            A tuple of status and preprocessed data.
        """
        ret_status = qcc.SCHEMA_INFER_SUCCESS
        for data in self.dataset:
            if self.preprocessor:
                try:
                    data, _ = self.preprocessor.process(data)
                except Exception as e:
                    qacc_file_logger.error(f"Error in preprocessing data: {e}")
                    ret_status = qcc.SCHEMA_PREPROC_FAIL
                    break

            data.data = [
                np.asarray(
                    data.data[inp_idx].reshape(inp_info[1]),
                    dtype=inp_info[0],
                    order="C",
                )
                for inp_idx, inp_info in enumerate(self.input_info.values())
            ]
            yield ret_status, data

    def post_inference(self, pipeline_stages, data, ret_status):
        """Post-inference processing steps including dumping, comparator, adapter,
        and metric execution.

        Args:
            pipeline_stages: List of pipeline stages to execute.
            data: The inference output data.
            ret_status: Current status of the pipeline execution.

        Returns:
            Updated status after post-inference processing.
        """
        # Inference output dumping logic for reference platforms
        if self.inference_schema.is_ref and self.comparator_enabled:
            if self.inference_schema.name == InferenceEngineType.QNN:
                os.makedirs(
                    os.path.join(self.ref_out_dir, f"Result_{data.idx}"),
                    exist_ok=True
                )
                inference_output_paths = [
                    os.path.join(
                        self.ref_out_dir,
                        self.comparator_out_frmt.format(data.idx, out_name),
                    )
                    for out_name in self.output_names
                ]
            else:
                inference_output_paths = [
                    os.path.join(
                        self.ref_out_dir,
                        self.comparator_out_frmt.format(out_name, data.idx),
                    )
                    for out_name in self.output_names
                ]
            qacc_file_logger.debug(f"{inference_output_paths=}")
            data.save(inference_output_paths)

        if not self.inference_schema.is_ref and qcc.STAGE_INFER in self.dump_stages:
            # logic to dump the inference output to the disk.
            if self.inference_schema.name == InferenceEngineType.QNN:
                os.makedirs(
                    os.path.join(self.infer_dir, f"Result_{data.idx}"),
                    exist_ok=True
                )
                infer_fn_format = "Result_{}/{}.raw"
                output_paths = [
                    os.path.join(
                        self.infer_dir,
                        infer_fn_format.format(data.idx, output_name),
                    )
                    for output_name in self.output_names
                ]
            else:
                infer_fn_format = "{}_{}.raw"
                output_paths = [
                    os.path.join(
                        self.infer_dir,
                        infer_fn_format.format(output_name, data.idx),
                    )
                    for output_name in self.output_names
                ]
            data.save(output_paths)

        # Comparator Step: Applicable only when comparator is enabled and for non Reference Schemas
        if not self.inference_schema.is_ref and self.comparator_enabled:
            try:
                self.compare_infer_results(data)
            except Exception as e:
                qacc_file_logger.error(
                    f"Failed to execute comparator. Reason: {e}"
                )
                ret_status = qcc.SCHEMA_COMPARATOR_FAIL

        # Adapter Execution
        if self.adapter_objs:
            try:
                for adapter_obj in self.adapter_objs:
                    data = adapter_obj.transform(data)
            except Exception as e:
                qacc_file_logger.error(
                    f"Failed to execute Adapter. Reason: {e}"
                )
                ret_status = qcc.SCHEMA_INFER_FAIL
        # Postprocessor Execution
        if self.postprocessor:
            try:
                data, _ = self.postprocessor.process(data)
            except Exception as e:
                qacc_file_logger.error(
                    f"Failed to execute postprocessor. Reason: {e}"
                )
                ret_status = qcc.SCHEMA_POSTPROC_FAIL

        # Metric Execution
        if qcc.STAGE_METRIC in pipeline_stages and self.metric_objs:
            # metric calculate based on objects
            for metric in self.metric_objs:
                metric(data)
        return ret_status

    def _finalize_metrics_and_comparator(self, pipeline_stages: list[str]):
        """Finalizes metrics and comparator scores based on pipeline stages.

        Args:
            pipeline_stages: List of pipeline stages to process.
        """
        result = {}
        if qcc.STAGE_METRIC in pipeline_stages and self.metric_objs:
            # Metric Finalize based on metric objects
            for metric in self.metric_objs:
                qacc_file_logger.debug(
                    f"Invoking Metric.finalize() for {self.inference_schema_name}"
                    f" --> {metric}"
                )
                result.update(metric.finalize())
        self.metrics_score_dict = result

        # Finalize on Comparator Scores:
        if not self.inference_schema.is_ref and self.comparator_enabled:
            comparator_result = {}
            for idx, output_name in enumerate(self.output_names):
                comparator_score_per_sample = [
                    i[1]
                    for i in self.comparator_score_dict[
                        output_name
                    ]
                ]
                if len(comparator_score_per_sample) > 0:
                    mean_score = round(
                        np.array(comparator_score_per_sample).mean(), 4
                    )
                else:
                    mean_score = 0
                comparator_result_key = f"({self.comparator_objs[idx].name}) {output_name}"
                comparator_result[comparator_result_key] = mean_score
            # Dump the Comparator Results as CSV
            self._dump_comparator_scores_sample_wise()
            self.comparator_score_dict = comparator_result
        else:
            self.comparator_score_dict = {}

    @timer_decorator
    def execute_pipeline_stages_non_host(self,
                preprocessed_file_lists: List[str | Path],
                pipeline_stages: list[str],
                data_chunk_size: int = None,
                ) -> tuple[int, dict[str, float], dict[str, float]]:
        """Execute the specified pipeline stages for non-host processing.

        This method processes the preprocessed dataset in batches, performs inference
        if required, and executes post-inference operations. It also handles metric
        calculation and comparator score processing if enabled.

        Args:
            pipeline_stages: List of pipeline stages to execute.
            preprocessed_file_lists: List of preprocessed file lists to process.
            data_chunk_size: Number of samples to use at a time from the dataset
                into for processing.

        Returns:
            A tuple containing:
            - ret_status: The status of the inference process.
            - metrics_score_dict: Dictionary of metric scores.
            - comparator_score_dict: Dictionary of comparator scores.
        """
        ret_status = qcc.SCHEMA_INFER_SUCCESS
        total_sample_count = len(self.dataset)
        log_frequency = max(int(total_sample_count / 10), 1)
        sample_completed = 0
        if data_chunk_size is None:
            # if data_chunk_size is None then use the size of dataset as data chunk size
            data_chunk_size = total_sample_count

        try:
            for file_list in preprocessed_file_lists:
                if qcc.STAGE_INFER in pipeline_stages:
                    try:
                        output_data = self.infer_on_filelist(file_list)
                    except Exception as e:
                        qacc_file_logger.error(f"Error in Inference: {e}")
                        ret_status = qcc.SCHEMA_INFER_FAIL

                for idx, infer_output in enumerate(output_data):
                    data = self.dataset[idx + sample_completed]
                    data.data = infer_output
                    ret_status = self.post_inference(pipeline_stages=pipeline_stages,
                                        data=data, ret_status=ret_status)
                sample_completed += data_chunk_size
                if sample_completed % log_frequency == 0:
                    qacc_logger.info(
                        f"Evaluation Status {self.inference_schema_name} -> "
                        f"{sample_completed}/{total_sample_count} samples completed.")
                if qcc.get_inference_schema_status(ret_status) != 'Success':
                    break

            execution_status = qcc.get_inference_schema_status(ret_status) == 'Success'
            if execution_status:
                self._finalize_metrics_and_comparator(pipeline_stages)
        except Exception as e:
            qacc_file_logger.error(
                f"Failed to execute_pipeline_stages for {self.inference_schema_name}. Reason: {e}"
            )
        finally:
            self.teardown()
        qacc_file_logger.info(
            f"Completed Evaluation of inference_schema={self.inference_schema_name}"
        )
        return ret_status, self.metrics_score_dict, self.comparator_score_dict
