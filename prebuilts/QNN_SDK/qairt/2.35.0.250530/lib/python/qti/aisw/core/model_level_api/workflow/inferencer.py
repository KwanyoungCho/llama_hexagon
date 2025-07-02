#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================
from qti.aisw.core.model_level_api.workflow.workflow import Workflow, WorkflowMode


class Inferencer(Workflow):
    def __init__(self, backend, model, executor=None, sdk_path=None):
        super().__init__(backend, model, executor, sdk_path)
        self._workflow_mode = WorkflowMode.INFERENCE
        self._backend.workflow_mode = self._workflow_mode
        self._latest_run_config = None

        if self._executor is None:
            target_default_executor_cls = self._backend.target.get_default_executor_cls()
            self._executor = target_default_executor_cls()


    def setup(self, config=None, output_dir='./output/'):
        self._latest_run_config = config
        profiling_data = self._executor.setup(self._workflow_mode,
                                              self._backend,
                                              self._model,
                                              self._sdk_path,
                                              config,
                                              output_dir)

        if profiling_data is not None:
            self._profiling_data.append(profiling_data)

    def run(self, input_data, config=None, output_dir='./output/'):
        if config is not None:
            self._latest_run_config = config
        output_data, profiling_data = self._executor.run_inference(config,
                                                                   self._backend,
                                                                   self._model,
                                                                   self._sdk_path,
                                                                   input_data,
                                                                   output_dir)

        if profiling_data is not None:
            self._profiling_data.append(profiling_data)
        return output_data


    def teardown(self, output_dir='./output/'):
        profiling_data = self._executor.teardown(self._backend,
                                                 self._sdk_path,
                                                 self._latest_run_config,
                                                 output_dir)
        if profiling_data is not None:
            self._profiling_data.append(profiling_data)