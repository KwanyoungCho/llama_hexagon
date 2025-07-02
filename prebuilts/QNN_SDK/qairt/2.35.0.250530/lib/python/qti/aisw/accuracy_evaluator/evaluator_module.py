# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import inspect
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Literal, Optional

import qti.aisw.accuracy_evaluator.common.exceptions as ce
import qti.aisw.accuracy_evaluator.qacc.manager as manager
import yaml
from pydantic import Field, FilePath, NewPath, ValidationInfo, field_validator
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.qacc.config_definitions import EvaluatorPipelineConfig
from qti.aisw.accuracy_evaluator.qacc.logger import QaccLogger
from qti.aisw.accuracy_evaluator.qacc.manager import EvaluatorStatus
from qti.aisw.tools.core.modules.api.compliance.function_signature_compliance import (
    expect_module_compliance,
)
from qti.aisw.tools.core.modules.api.definitions.common import AISWBaseModel
from qti.aisw.tools.core.modules.api.definitions.interface import Module
from qti.aisw.tools.core.modules.api.definitions.schema import ModuleSchema, ModuleSchemaVersion


"""
This file defines the Accuracy Evaluator Module.
"""


# Dump the pydantic config to a yaml file.
def dump_config_yaml(config, yaml_filepath):
    data = config.model_dump_json(exclude_unset=True)
    json_filepath = yaml_filepath.rsplit('.', 1)[0] + '.json'
    with open(json_filepath, 'w') as o:
        o.write(data)
    with open(json_filepath, 'r') as o:
        load_json = json.load(o)
    with open(yaml_filepath, 'w') as o:
        yaml.dump(load_json, o, default_flow_style=False, sort_keys=False, indent=4)


# class EvaluatorInputs(AISWBaseModel, arbitrary_types_allowed=True):
class EvaluatorInputs(AISWBaseModel):
    config: EvaluatorPipelineConfig = Field(
        description="Config object for Evaluator Pipeline Config.")
    work_dir: NewPath = Field(default=os.path.join(os.getcwd(), "./qacc_temp"),
                              description="Path to the working directory.")
    onnx_symbol: Optional[str] = Field(
        default=None, alias="onnx_define_symbol",
        description="Replace onnx symbols in input/output shapes. " +
        "Can be passed as list of multiple items. " + "Default replaced by 1. Example: __unk_200:1")
    device_id: Optional[str | int] = Field(
        default=None, description=
        "List of target device IDs separated by comma, to be used for accuracy evaluation")
    inference_schema_type: Optional[str] = Field(
        default=None,
        description="run only the inference schemas with this name. Example: qnn, onnxrt")
    inference_schema_tag: Optional[str] = Field(
        default=None, description="run only inference schemas having this tag")
    cleanup: Optional[Literal['', 'end', 'intermediate']] = Field(
        default='', description="end: deletes the files after all stages are completed. " +
        "intermediate: deletes after previous stage outputs are used.")
    use_memory_plugins: Optional[bool] = Field(default=False,
                         description="Flag to enable memory plugins.")
    use_memory_pipeline: Optional[bool] = Field(default=False,
                         description="Flag to enable memory pipeline. use_memory_plugins is ignored.")
    silent: bool = Field(default=False,
                         description="Run in silent mode. Do not expect any CLI input from user.")


class EvaluatorOutputs(AISWBaseModel):
    metric_results: List[List[Any]] = Field(
        description="List containing the accuracy metric results. " +
        "One entry for each of the inference schema executed" +
        "Each entry has (index, tags, schema name, result, precision, metric)")
    qacc_log: FilePath = Field(description="Log file containing the evaluator logs")
    config_yaml: FilePath = Field(description="Model config used by this run dumped in yaml format")
    deviating_input_samples: Optional[List[FilePath]] = Field(
        default=[], description="Lists of Most Deviating samples filelist")


class EvaluatorModuleSchemaV1(ModuleSchema):
    """A dataclass that defines the properties of a module. It is semi-closed to modification but is
    freely extensible.
    """
    _VERSION = ModuleSchemaVersion(major=0, minor=2, patch=0)
    _BACKENDS = None
    name: Literal["EvaluatorModule"] = "EvaluatorModule"
    path: Literal[str(inspect.getfile(os))] = str(inspect.getfile(os))
    arguments: EvaluatorInputs
    outputs: Optional[EvaluatorOutputs] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION

    @field_validator('backends')
    @classmethod
    def reject_non_default_value_backends(cls, v: Any, info: ValidationInfo):
        return cls._check_for_non_default_value(v, cls._BACKENDS, info)


@expect_module_compliance
class EvaluatorModule(Module):
    """This module is a concrete implementation of a module interface using a single schema: EvaluatorModuleSchemaV1
    """
    _SCHEMA = EvaluatorModuleSchemaV1
    _PREVIOUS_SCHEMAS = []
    _LOGGER = logging.getLogger("EvaluatorLogger")
    _init_workdir = True

    def __init__(self, logger=None):
        super().__init__(logger)
        self.logger = logger

    @property
    def _schema(self):
        return self._SCHEMA

    def properties(self) -> Dict[str, Any]:
        return self._schema.model_json_schema()

    def get_logger(self) -> Any:
        # add logic to use evaluator's existing logging infra
        return self._logger

    def _set_log_level(self, log_level: int):
        QaccLogger.set_log_level(log_level)

    def enable_debug(self):
        self._set_log_level(logging.DEBUG)

    def _init_workdir_logger(self, work_dir):
        # add logic to use evaluator's existing logging infra
        try:
            if os.path.exists(work_dir):
                qacc_logger.warning(f'Work directory {work_dir} already exists. Deleting it.')
                shutil.rmtree(work_dir)
            # create empty working directory
            os.makedirs(work_dir)
        except Exception as e:
            raise ce.FailedToCreateWorkDir(
                "Failed to create work dir at {} due to exception {}".format(work_dir, e))

        self.log_file = os.path.join(work_dir, 'qacc.log')
        try:
            log_level = logging.INFO
            QaccLogger.setup_logger(log_file=self.log_file, log_level=log_level)
        except Exception as e:
            raise ce.FailedToSetupLogger("Failed to setup logger at {} due to exception {}".format(
                self.log_file, e))

    # This is the entry point to accuracy evaluator
    def evaluate(self, args: EvaluatorInputs) -> EvaluatorOutputs:

        # Create a new work directory and initialize the logger for evaluator.
        # If the work directory exists, it would be deleted.
        # The boolean init_workdir is to ensure the method is called only once.
        if self._init_workdir:
            self._init_workdir_logger(args.work_dir)
            self._init_workdir = False

        # Get the qnn-sdk-dir from env
        qnn_sdk_dir = os.environ.get('QNN_SDK_ROOT')
        if not qnn_sdk_dir:
            raise ce.QnnSdkRootNotSet("QNN_SDK_ROOT variable is not set.")
        elif not os.path.isdir(qnn_sdk_dir):
            raise ce.QnnSdkRootNotValid("QNN_SDK_ROOT is not directory.")

        # dump the config in the working directory
        config_file = "model_config.yaml"
        config_yaml = os.path.join(args.work_dir, config_file)
        dump_config_yaml(args.config, config_yaml)

        qacc_file_logger.info(f'Running evaluator with config: {args.config!r}')
        metric_results = []
        error_status = EvaluatorStatus.SUCCESS

        # Call Run Pipeline
        # use memory plugins when user sets use_memory_plugins or use_memory_pipeline flag
        if args.config.info_config.memory_pipeline:
            args.use_memory_pipeline = True
        use_memory_plugins = args.use_memory_plugins or args.use_memory_pipeline
        try:
            mgr = manager.QACCManager(config=args.config, work_dir=str(args.work_dir),
                                      use_memory_plugins=use_memory_plugins)
            if not args.use_memory_pipeline:
                error_status, metric_results = mgr.run_pipeline(
                    inference_schema_name=args.inference_schema_type, work_dir=str(args.work_dir),
                    inference_schema_tag=args.inference_schema_tag, cleanup=args.cleanup,
                    onnx_symbol=args.onnx_symbol, device_id=args.device_id, silent=args.silent,
                    qnn_sdk_dir=qnn_sdk_dir)
            else:
                error_status, metric_results = mgr.run_memory_pipeline(
                    work_dir=str(args.work_dir), inference_schema_name=args.inference_schema_type,
                    inference_schema_tag=args.inference_schema_tag, cleanup=args.cleanup,
                    onnx_symbol=args.onnx_symbol, device_id=args.device_id, silent=args.silent)

        except Exception as e:
            qacc_logger.error(
                f'Accuracy evaluator pipeline. See {os.path.abspath(args.work_dir)}/qacc.log for more details'
            )
            error_status = EvaluatorStatus.PIPELINE_EXCEPTION
            qacc_file_logger.exception(e)

        if error_status != EvaluatorStatus.SUCCESS:
            qacc_logger.error(
                f' Accuracy evaluator pipeline failed. See {os.path.abspath(args.work_dir)}/qacc.log for more details'
            )
            raise ce.EvaluatorRunPipelineFailed(
                "Accuracy evaluator pipeline failed with error status {}".format(error_status))
        else:
            qacc_logger.info('Accuracy evaluator pipeline completed successfully')

        return EvaluatorOutputs(metric_results=metric_results, qacc_log=self.log_file,
                                 config_yaml=config_yaml)
