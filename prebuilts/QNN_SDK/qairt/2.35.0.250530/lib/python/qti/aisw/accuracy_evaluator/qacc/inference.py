# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import copy
import os
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import chain, product
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from qti.aisw.accuracy_evaluator.common.infer_engines.QAIRTInferenceEngine import (
    QAIRTInferenceEngine,
)
from qti.aisw.accuracy_evaluator.common.utilities import *
from qti.aisw.accuracy_evaluator.common.worker_group import QAIRTWorkerGroup
from qti.aisw.accuracy_evaluator.qacc import *
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.qacc.config_definitions import *
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.tools.core.modules.api.definitions.common import Model
from qti.aisw.tools.core.utilities.data_processing.core.transformations import (
    ComponentRegistry,
    MetricConfig,
    PostProcessorConfig,
    PreProcessorConfig,
)


class InferenceManager:

    def __init__(self, inference_schema_config, infer_config, binary_path):
        self.inference_schema_config = inference_schema_config
        self.binary_path = binary_path
        self.infer_config = infer_config
        # capture execution time
        # (quantization time, compilation time, inference time)
        self.execution_time = [0, 0, 0]

    def execute(self, model_path, output_dir, input_file, output_file, calibration, device_id,
                precompiled_path, console_tag, compile_only, qnn_sdk_dir=""):
        if self.inference_schema_config.name == InferenceEngineType.QNN:
            return self.execute_qairt(model_path, output_dir, input_file, output_file, device_id,
                                      precompiled_path, console_tag, calibration=calibration,
                                      compile_only=compile_only, qnn_sdk_dir=qnn_sdk_dir)
        elif self.inference_schema_config.name == InferenceEngineType.ONNXRT:
            return self.execute_onnxrt(model_path, output_dir, input_file, output_file)
        elif self.inference_schema_config.name == InferenceEngineType.TFRT:
            return self.execute_tfrt(model_path, output_dir, input_file, output_file)
        elif self.inference_schema_config.name == InferenceEngineType.TORCHSCRIPTRT:
            return self.execute_torchscriptrt(model_path, output_dir, input_file, output_file)

    def execute_qairt(self, model_path, output_dir, input_file, output_file, device_id,
                      precompiled_path, console_tag, calibration=None, compile_only=False,
                      qnn_sdk_dir=""):

        backend = self.inference_schema_config.backend
        precision = self.inference_schema_config.precision
        target_arch = self.inference_schema_config.target_arch
        backend_extensions = self.inference_schema_config.backend_extensions
        netrun_params = self.inference_schema_config.netrun_params
        quantizer_params = self.inference_schema_config.quantizer_params
        converter_params = self.inference_schema_config.converter_params
        contextbin_params = self.inference_schema_config.contextbin_params

        calibration_file = None
        if quantizer_params and quantizer_params.float_fallback:
            # when float_fallback is set to True, calibration_file should be None
            calibration_file = None
        elif calibration and self.inference_schema_config.precision == PrecisionType.QUANT:
            calibration_file = self.parse_generate_calibration(calibration, input_file,
                                                               os.path.dirname(input_file))

        os.makedirs(output_dir, exist_ok=True)
        engine = QAIRTInferenceEngine(
            model_path=model_path, inputlistfile=input_file, calibration_file=calibration_file,
            output_path=output_dir, inputs_info=self.infer_config.inputs_info,
            outputs_info=self.infer_config.outputs_info, gen_out_file=output_file,
            backend_extensions=backend_extensions, netrun_params=netrun_params,
            quantizer_params=quantizer_params, converter_params=converter_params,
            contextbin_params=contextbin_params, backend=backend, precision=precision,
            target_arch=target_arch, device_id=device_id)
        if converter_params:
            # dumping all converter params before execution
            outfile = os.path.join(output_dir, 'converter_params_list.json')
            data = (
                f'converter_params: {self.inference_schema_config.converter_params.model_dump_json(exclude_unset=True)}'
            )
            with open(outfile, 'w', encoding='utf-8') as f:
                f.write(data)
        if quantizer_params:
            # dumping all quantizer params before execution
            outfile = os.path.join(output_dir, 'quantizer_params_list.json')
            data = f'quantizer_params: {self.inference_schema_config.quantizer_params.model_dump_json(exclude_unset=True)}'
            with open(outfile, 'w', encoding='utf-8') as f:
                f.write(data)
        try:
            engine.execute()
            ret_status = True
            qacc_file_logger.info('Inference success on QNN in execution stage.')
        except Exception as e:
            qacc_logger.info(e)
            qacc_file_logger.error('Inference failed on QNN in execution stage.')
            ret_status = False
        finally:
            infer_stages_status = engine.stage_status

        infer_fail_stage = self._get_first_fail_stage(infer_stages_status)
        return not ret_status, infer_fail_stage, [0, 0, 0]

    def execute_onnxrt(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.OnnxRTEngine import OnnxInferenceEngine  # noqa: I001
        engine = OnnxInferenceEngine(
            model=model_path, inputlistfile=input_file,
            multithread=self.inference_schema_config.multithreaded, output_path=output_dir,
            input_info=self.infer_config.inputs_info, output_info=self.infer_config.outputs_info,
            gen_out_file=output_file, convert_nchw=self.inference_schema_config.convert_nchw)

        try:
            status, self.execution_time[2] = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            qacc_logger.error('(onnxrt) Inference failed. See qacc.log for more details.')
            qacc_file_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'onnx-inference'

        return not status, infer_fail_stage, self.execution_time

    def execute_tfrt(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.TensorflowRTEngine import TensorflowInferenceEngine  # noqa: I001
        engine = TensorflowInferenceEngine(model=model_path, inputlistfile=input_file,
                                           multithread=self.inference_schema_config.multithreaded,
                                           output_path=output_dir,
                                           input_info=self.infer_config.inputs_info,
                                           output_info=self.infer_config.outputs_info,
                                           gen_out_file=output_file)
        try:
            status, _, self.execution_time[2], _ = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            qacc_logger.error('tensorflow runtime inference failed. See qacc.log for more details.')
            qacc_file_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'tensorflow-inference'

        return not status, infer_fail_stage, self.execution_time

    def execute_torchscriptrt(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.TorchScriptRTEngine import TorchScriptInferenceEngine  # noqa: I001
        engine = TorchScriptInferenceEngine(model=model_path, inputlistfile=input_file,
                                            multithread=self.inference_schema_config.multithreaded,
                                            output_path=output_dir,
                                            input_info=self.infer_config.inputs_info,
                                            output_info=self.infer_config.outputs_info,
                                            gen_out_file=output_file)
        try:
            status, self.execution_time[2] = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            qacc_logger.error(
                'torchscript runtime inference failed. See qacc.log for more details.')
            qacc_file_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'torchscript-inference'

        return not status, infer_fail_stage, self.execution_time

    def execute_tfrt_session(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.TensorflowSessionRTEngine import TensorflowSessionInferenceEngine  # noqa: I001
        engine = TensorflowSessionInferenceEngine(
            model=model_path, inputlistfile=input_file,
            multithread=self.inference_schema_config.multithreaded, output_path=output_dir,
            input_info=self.infer_config.inputs_info, output_info=self.infer_config.outputs_info,
            gen_out_file=output_file)
        try:
            status, _, self.execution_time[2], _ = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            qacc_logger.error('tensorflow runtime inference failed. See qacc.log for more details.')
            qacc_file_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'tensorflow-session-inference'

        return not status, infer_fail_stage, self.execution_time

    def _parse_range(self, index_str):
        if len(index_str) == 0:
            return []
        nums = index_str.split("-")
        assert len(nums) <= 2, 'Invalid range in calibration file '
        start = int(nums[0])
        end = int(nums[-1]) + 1
        return range(start, end)

    def parse_generate_calibration(self, calibration, input_file, output_dir):
        if calibration is None or input_file is None:
            return None
        (calib_type, calib_file) = calibration

        if calib_type == CalibrationType.RAW:
            return calib_file
        elif calib_type == CalibrationType.INDEX:
            cf = open(calib_file, 'r')
            indexes_str = cf.read().replace('\n', ',').strip()
            indexes = sorted(
                set(chain.from_iterable(map(self._parse_range, indexes_str.split(",")))))
            cf.close()
            _path = os.path.join(output_dir, 'calibration.txt')
            qacc_file_logger.info('Generating calibration file')
            with open(input_file) as f, open(_path, 'w') as f2:
                for index, line in enumerate(f):
                    if index in indexes:
                        f2.write(line)
            return _path
        else:
            raise RuntimeError('Invalid calibration type {}'.format(calib_type))

    def _get_first_fail_stage(self, stage_status):
        for stage in stage_status:
            if stage_status[stage] == False:  # noqa: E712
                return stage
        return ""


class InferenceSchemaManager:

    def __init__(self, inference_schemas, config):
        self.inference_schemas = inference_schemas
        self.device_ids = config.inference_config.device_ids
        self.schedule = None
        if self.device_ids:
            self.device_locks = {device_id: Lock() for device_id in self.device_ids}
        else:
            self.device_locks = {}
        # used to perform calibration if int8 inference schema available
        self._is_calib_req = False

    def scan_and_add_inference_schema_permutations(self):
        """Scans the inference_schema section and finds all the possible
        inference schema permutations. Once the scan is complete, these
        possible inference schema permutations are added to the existing
        inference schema list.

        Example:
        Given an inference schema
            inference_schema:
                name: qnn
                precision: <value>
                quantizer_params:
                    param1: input1 | input2
                    param2: input3 | input4
                    param3: range(2.0, 4.0, 1.0) # all values from 2 to 4 with step-size 1

        will create following inference schemas
            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input1
                    param2: input3
                    param3: 2.0
            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input1
                    param2: input3
                    param3: 3.0
            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input1
                    param2: input3
                    param3: 4.0

            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input1
                    param2: input4
                    param3: 2.0
            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input1
                    param2: input4
                    param3: 3.0
            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input1
                    param2: input4
                    param3: 4.0

            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input2
                    param2: input3
                    param3: 2.0
            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input2
                    param2: input3
                    param3: 3.0
            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input2
                    param2: input3
                    param3: 4.0

            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input2
                    param2: input4
                    param3: 2.0
            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input2
                    param2: input4
                    param3: 3.0
            inference_schema:
                name: qnn
                precision: <value>
                params:
                    param1: input2
                    param2: input4
                    param3: 4.0
        """
        # updated inference schemas consisting of original plus newly
        # generated inference schemas
        updated_inference_schemas = []

        for inference_schema in self.inference_schemas:

            if (inference_schema.name != InferenceEngineType.QNN):
                qacc_file_logger.debug('scan_and_add: Non QNN inference schema {} added'.format(
                    inference_schema.name.value))
                updated_inference_schemas.append(inference_schema)
                continue

            # get nested list of values
            param_values = []
            param_keys = []

            quant_params_dict = {}
            dummy_inp_list = None
            if inference_schema.quantizer_params:
                # create a dictionary of quantizer params and their corresponding values
                # {param1 : [val1, val2], param2 : [val3, val4], param3: [val5, val6, val7]}
                for k, v in inference_schema.quantizer_params:
                    if k in inference_schema.quantizer_params.model_fields_set:
                        if str(k) == 'input_list':
                            dummy_inp_list = v
                            continue
                        # convert val to list to enable product of values
                        if not isinstance(v, list) or str(k) in [
                                "preserve_io_datatype", "algorithms"
                        ]:
                            v = [v]
                        quant_params_dict[k] = v
                # Product of the dict values will create a list of tuples. Each tuple will have
                # one val corresponding to each param
                # param_values = [(val1, val3, val5), (val1, val3, val6), (val1, val3, val7),
                # (val1, val4, val5), (val1, val4, val6), (val1, val4, val7), (val2, val3, val5),
                # (val2, val3, val6), (val2, val3, val7), (val2, val4, val5), (val2, val4, val6),
                # (val2, val4, val7)]
                param_values = list(product(*quant_params_dict.values()))
                param_keys = quant_params_dict.keys()
                qacc_file_logger.debug('scan_and_add: Options for keys-{} values-{} added'.format(
                    param_keys, param_values))
                for param in param_values:
                    # Each 'param' contains values for given quantizer params
                    # param1 -> val1
                    # param2 -> val3
                    # param3 -> val5

                    # param1 -> val1
                    # param2 -> val3
                    # param3 -> val6
                    # and so on, which will be zipped together to create a dictionary, to
                    # be used to create the QuantizerParams object
                    new_inference_schema = copy.deepcopy(inference_schema)
                    new_quant_params_dict = dict(zip(param_keys, param))
                    new_quant_params_dict['input_list'] = dummy_inp_list
                    new_inference_schema.quantizer_params = QuantizerParams(**new_quant_params_dict)
                    updated_inference_schemas.append(new_inference_schema)
                    qacc_file_logger.debug(updated_inference_schemas)

            else:
                updated_inference_schemas.append(inference_schema)

            # check whether for current inference schema calibration is needed.
            # The key is needed in estimating disk space and performing
            # preprocessing for calibration inputs.
            if not self._is_calib_req:
                # check only if is_calib_req is False
                # if even inference schema needs calibration then this field will be True
                self._is_calib_req = (inference_schema.precision == PrecisionType.QUANT
                                and inference_schema.precompiled_path is None)

        for up_inference_schema in updated_inference_schemas:
            qacc_file_logger.info('Inference schema: {} - params: {}'.format(
                up_inference_schema.name.value, up_inference_schema.quantizer_params))

        # updating inference schema list
        self.inference_schemas = updated_inference_schemas
        return updated_inference_schemas, self._is_calib_req

    def create_schedule(self):
        """Creates a schedule based on distributed inference strategy.

        A schedule has following format:
            [parallel_chuck_1, parallel_chuck_2, ... , parallel_chuck_n]

        Each parallel chunk has following format:
            [(inference_schema_idx, device_id), ... , (inference_schema_idx, device_id)]

        Note: device_id for inference_schemas other than aic is -1

        Example:
            case1:
                device_ids = [0,1]
                inference_schemas = [onnx, aic, aic, aic, aic]
                schedule = [[(0,-1), (1,0), (2,1)], [(3,0), (4,1)]]
        """
        self.schedule = []
        slots = len(self.device_ids)
        distributed_inference_schemas = []
        used_slots = 0

        for idx, inference_schema in enumerate(self.inference_schemas):
            if inference_schema.backend == BackendType.AIC:
                # if all slots filled
                if used_slots == slots:
                    self.schedule.append(copy.deepcopy(distributed_inference_schemas))
                    distributed_inference_schemas = []
                    used_slots = 0

                distributed_inference_schemas.append((idx, int(self.device_ids[used_slots])))

                # inc used slots
                used_slots += 1

            else:
                # device id for reference framework inference: Device id is set to -1.
                # device id for cpu and htp backend evaluated on host (sim configurations)
                if inference_schema.name in [InferenceEngineType.ONNXRT,
                            InferenceEngineType.TFRT, InferenceEngineType.TORCHSCRIPTRT] or \
                            inference_schema.target_arch == TargetArchType.X86:
                    distributed_inference_schemas.append((idx, -1))
                else:
                    distributed_inference_schemas.append((idx, self.device_ids[0]))

        # copy the last chuck
        self.schedule.append(copy.deepcopy(distributed_inference_schemas))
        qacc_file_logger.info('Distributed schedule: {}'.format(self.schedule))

    def get_schedule(self):
        return self.schedule


class InferenceSchemaExecutor:
    """A class for executing inference schemas for model evaluation.
    This class is responsible for setting up and running inference pipelines
    with various inference engines (like QNN(i.e. QAIRT), ONNXRT, TorchScriptRT).
    """
    def __init__(self, config: EvaluatorPipelineConfig,
                dump_stages: list[str] = None,
                comparator_enabled: bool = False,
                num_workers: Optional[int] = None,
                has_remote_device_evaluation: bool = False,
                preprocessed_file_list: Optional[str | Path] = None
                ):
        """Initialize the InferenceSchemaExecutor.

        Args:
            config (EvaluatorPipelineConfig): Configuration for the evaluator pipeline.
            dump_stages (list[str], optional): List of stages to dump. Defaults to None.
            comparator_enabled (bool, optional): Whether comparator is enabled. Defaults to False.
            has_remote_device_evaluation (bool, optional): Whether remote device evaluation
                is enabled. Defaults to False.
            preprocessed_file_list (list[str], optional): List of preprocessed files.
                Defaults to None.
            num_workers (int, optional): Number of worker threads/processes to use.
                Defaults to None.
        """
        self.num_workers = num_workers
        self.config = config
        self.dump_stages = dump_stages
        self.comparator_enabled = comparator_enabled
        if self.comparator_enabled:
            self.reference_platform = self.config.get_ref_inference_schema()
        else:
            self.reference_platform = None
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers,
                                         thread_name_prefix='Inference_')
        self.preprocessed_file_list = preprocessed_file_list
        self.has_remote_device_evaluation = has_remote_device_evaluation
        self.data_chunk_size = self.config.info_config.data_chunk_size
        self.preprocessors_objs = []
        self.postprocessor_objs = []
        self.adapter_objs = None
        self.validate()

    def validate(self):
        """Validates the configuration settings before running the evaluation.

        Raises:
            ValueError: If required parameters are missing for remote device evaluation
                        or when comparator is enabled but reference platform is not set.
        """
        if self.has_remote_device_evaluation and self.preprocessed_file_list is None:
            raise ValueError("Preprocessed file list is required to run remote device evaluation")
        if self.comparator_enabled and not self.reference_platform:
            raise ValueError(
                "Reference platform must be set for comparator enabled executions.")

    def execute_inference_schema(self,
        model: Model | str,
        work_dir: str | os.PathLike,
        inference_schema: InferenceSchemaConfiguration,
        pipeline_stages: list[str],
        device_id: Optional[str | int] = None,
        device_lock: Optional[Lock] = None) -> tuple[bool, dict[str, Any], dict[str, Any], float]:
        """Execute the inference schema for a given model and pipeline stages.

        Args:
            model: The model to be evaluated. Can be a string path or a Model object.
            work_dir: Working directory for intermediate files.
            inference_schema: The schema defining how inference should be executed.
            pipeline_stages: List of stages to execute in the pipeline.
            device_id: Optional device identifier for hardware-specific execution.
            device_lock: Optional lock for device access.

        Returns:
            A tuple containing:
            - ret_status: Boolean indicating if the execution was successful.
            - metrics_scores_dict: Dictionary of metric scores.
            - comparator_scores_dict: Dictionary of comparator scores.
            - inference_time: The time taken for inference in seconds.
        """
        metrics_scores_dict = {}
        comparator_scores_dict = {}
        ret_status = False
        inference_time = 0
        inference_schema_name = inference_schema.get_inference_schema_name()

        try:
            dataset, metric_objs, output_comparators = self.setup_pipeline_objects(self.config)

            if inference_schema.name == InferenceEngineType.QNN:
                worker_group = QAIRTWorkerGroup(model=model, work_dir=work_dir,
                                    inference_schema=inference_schema, dataset=dataset,
                                    preprocessors_objs=self.preprocessors_objs,
                                    postprocessor_objs=self.postprocessor_objs,
                                    adapter_objs=self.adapter_objs,
                                    metric_objs=metric_objs,
                                    comparator_objs=output_comparators,
                                    device_id=device_id,
                                    reference_platform=self.reference_platform,
                                    dump_stages=self.dump_stages,
                                    comparator_enabled=self.comparator_enabled,
                                    input_info=self.input_info,
                                    output_info=self.output_info)

            elif inference_schema.name == InferenceEngineType.ONNXRT:
                from qti.aisw.accuracy_evaluator.common.worker_group import OnnxRTWorkerGroup
                worker_group = OnnxRTWorkerGroup(model, work_dir=work_dir,
                                    inference_schema=inference_schema, dataset=dataset,
                                    preprocessors_objs=self.preprocessors_objs,
                                    postprocessor_objs=self.postprocessor_objs,
                                    adapter_objs=self.adapter_objs,
                                    metric_objs=metric_objs,
                                    comparator_objs=output_comparators,
                                    device_id=device_id,
                                    reference_platform=self.reference_platform,
                                    dump_stages=self.dump_stages,
                                    comparator_enabled=self.comparator_enabled,
                                    input_info=self.input_info,
                                    output_info=self.output_info)

            elif inference_schema.name == InferenceEngineType.TORCHSCRIPTRT:
                from qti.aisw.accuracy_evaluator.common.worker_group import TorchScriptWorkerGroup
                worker_group = TorchScriptWorkerGroup(model, work_dir=work_dir,
                                    inference_schema=inference_schema, dataset=dataset,
                                    preprocessors_objs=self.preprocessors_objs,
                                    postprocessor_objs=self.postprocessor_objs,
                                    adapter_objs=self.adapter_objs,
                                    metric_objs=metric_objs,
                                    comparator_objs=output_comparators,
                                    device_id=device_id,
                                    reference_platform=self.reference_platform,
                                    dump_stages=self.dump_stages,
                                    comparator_enabled=self.comparator_enabled,
                                    input_info=self.input_info,
                                    output_info=self.output_info)

            elif inference_schema.name == InferenceEngineType.TFRT:
                from qti.aisw.accuracy_evaluator.common.worker_group import TensorflowRTWorkerGroup
                worker_group = TensorflowRTWorkerGroup(model, work_dir=work_dir,
                                    inference_schema=inference_schema, dataset=dataset,
                                    preprocessors_objs=self.preprocessors_objs,
                                    postprocessor_objs=self.postprocessor_objs,
                                    adapter_objs=self.adapter_objs,
                                    metric_objs=metric_objs,
                                    comparator_objs=output_comparators,
                                    device_id=device_id,
                                    reference_platform=self.reference_platform,
                                    dump_stages=self.dump_stages,
                                    comparator_enabled=self.comparator_enabled,
                                    input_info=self.input_info,
                                    output_info=self.output_info)
            else:
                raise ValueError(f"Inference engine type: {inference_schema.name} is not valid")

            qacc_file_logger.info(
                        f"Starting evaluation for {inference_schema_name} using device: {device_id}")

            if inference_schema.target_arch != TargetArchType.X86:
                inference_time, (ret_status, metrics_scores_dict, comparator_scores_dict) = worker_group.execute_pipeline_stages_non_host(
                                                                                    preprocessed_file_lists=self.preprocessed_file_lists,
                                                                                    pipeline_stages=pipeline_stages,
                                                                                    )
            else:
                inference_time, (ret_status, metrics_scores_dict, comparator_scores_dict) = worker_group.execute_pipeline_stages(pipeline_stages=pipeline_stages)

            # Create the deviating inputlist file if comparator scores are available
            if len(comparator_scores_dict) > 0:
                comparator_scores_dict['most_deviating_indices'] = worker_group.get_deviant_indices()
                with open(os.path.join(work_dir, f"{inference_schema_name}_deviating.txt"), 'w') as f:
                    for idx in worker_group.get_deviant_indices():
                        processed_input_paths = [os.path.abspath(os.path.join(work_dir, 'preproc',
                                                                f"{input_name}_{idx}.raw"))
                                                                for input_name in self.input_info]
                        f.writelines(",".join(processed_input_paths) + '\n')

        except Exception as e:
            qacc_file_logger.error(f"Failed to execute pipeline stages for {inference_schema_name}."
                        f" Reason: {e}")
            ret_status = qcc.SCHEMA_INFER_FAIL
        finally:
            if device_lock:
                device_lock.release()

        qacc_logger.info(
            f"Completed Evaluation of {inference_schema_name}. "
            f"Metrics: {metrics_scores_dict}. "
            f"Comparator Scores: {comparator_scores_dict}"
        )
        return ret_status, metrics_scores_dict, comparator_scores_dict, inference_time

    def setup_pipeline_objects(self,
                config: EvaluatorPipelineConfig)-> tuple[Any, list[Any], list[Any]]:
        """Set up the pipeline objects (dataset, preprocessors, adapters,
        postprocessors, metrics, and comparators).

        Args:
            config: EvaluatorPipelineConfig object containing all necessary information.

        Returns:
            A tuple containing:
            - dataset: The dataset object.
            - metric_objs: List of metric objects.
            - output_comparators: List of output comparator objects.
        """
        metric_objs = []
        output_comparators = []
        self.input_info = config.inference_config.inputs_info
        self.output_info = config.inference_config.outputs_info

        # Dataset Setup
        dataset_config_local = copy.deepcopy(config.dataset_config)
        # Note: result from get_components_from_configs is indexed at 0 because
        # it returns a list containing the dataset.
        dataset = ComponentRegistry.get_components_from_configs([dataset_config_local])[0]

        # Preproc Setup
        if config.preprocessing_config:
            preprocessor_configs = [PreProcessorConfig(name=processor.name, params=processor.params)
                        for processor in config.preprocessing_config.preprocessing_plugin_list]
            self.preprocessors_objs = ComponentRegistry.get_components_from_configs(preprocessor_configs)

        if self.has_remote_device_evaluation:
            # when has_remote_device_evaluation=True, preprocessed_file_list to be used for evaluation.
            # preprocessing steps would be skipped.
            self.preprocessed_file_lists = [self.preprocessed_file_list]
            if self.data_chunk_size is not None:
                # Split the preprocessed file list into smaller chunks
                self.preprocessed_file_lists = chunked_file_list_generator(
                        input_file_list=self.preprocessed_file_list,
                        lines_per_file=self.data_chunk_size)

        # Adapters Setup
        if config.adapter_config:
            self.adapter_objs = ComponentRegistry.get_components_from_configs([self.config.adapter_config])

        # Postproc Setup
        if config.postprocessing_config:
            postprocessor_configs = [PostProcessorConfig(name=b.name, params=b.params)
                        for b in config.postprocessing_config.postprocessing_plugin_list]
            self.postprocessor_objs = ComponentRegistry.get_components_from_configs(postprocessor_configs)

        # Metric Setup
        if config.metrics_config:
            metric_configs = [MetricConfig(name=b.name, params=b.params)
                        for b in config.metrics_config.metrics_plugin_list]
            metric_objs = ComponentRegistry.get_components_from_configs(metric_configs)

        # Comparators Setup
        if config.verifier_config:
            _, output_comparators, _, _ = ComparatorHelper.get_comparators(
                                comp_type=config.verifier_config.type,
                                tolerance=config.verifier_config.tol,
                                interpretation_strategy=config.verifier_config._interpretation_strategy,
                                out_info=self.output_info)

        return dataset, metric_objs, output_comparators

    def execute(self, model: Model | str, work_dir: str,
                    inference_schema: InferenceSchemaConfiguration,
                    pipeline_stages: list[str],
                    device_id: Optional[str | int] = None,
                    device_lock: Optional[Lock]=None) -> Future:
        """Execute the inference schema asynchronously using a thread pool.

        Args:
            model: The model to be evaluated.
            work_dir: Working directory for intermediate files.
            inference_schema: The schema defining how inference should be executed.
            pipeline_stages: List of stages to execute in the pipeline.
            device_id: Optional device identifier for hardware-specific execution.
            device_lock: Optional lock for device access.

        Returns:
            A future object representing the asynchronous execution.
        """
        future = self.executor.submit(self.execute_inference_schema, model=model,
                            work_dir=work_dir, inference_schema=inference_schema,
                            pipeline_stages=pipeline_stages,
                            device_id=device_id, device_lock=device_lock)

        return future
