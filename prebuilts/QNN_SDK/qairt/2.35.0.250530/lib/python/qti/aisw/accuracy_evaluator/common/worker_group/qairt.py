# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import qti.aisw.tools.core.modules.net_runner.net_runner_module as net_runner
from qti.aisw.accuracy_evaluator.common.worker_group.base import WorkerGroup
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger
from qti.aisw.accuracy_evaluator.qacc.config_definitions import AICBackendExtensions, TargetArchType
from qti.aisw.tools.core.modules.api.definitions.common import BackendType, Target
from qti.aisw.tools.core.utilities.data_processing import (
    NDArrayRepresentation,
)
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DevicePlatformType,
    RemoteDeviceIdentifier,
)


EVALUATOR_IO_NODE_INFO = Dict[str, List[Any]]


class QAIRTWorkerGroup(WorkerGroup):
    """Class representing a worker group for QAIRT inference.
    This class extends the WorkerGroup base class and provides specific implementations
    for QAIRT Runtime inference, validation, and teardown.
    """

    def _setup(self):
        self.backend = self.inference_schema.backend
        self.target_arch = self.inference_schema.target_arch
        self.netrun_params = self.inference_schema.netrun_params
        self.backend_extensions = self.inference_schema.backend_extensions
        self.backend_config_file = None
        self.backend_config_dict = None
        if self.netrun_params and self.netrun_params.backend_extensions:
            self.backend_config_file = self.netrun_params.backend_extensions
        elif self.backend_extensions:
            self.backend_config_dict = self.backend_extensions.get_netrun_config_dict()
        eval_on_android_device = (self.backend
                                       in [BackendType.CPU, BackendType.GPU, BackendType.HTP]
                                       and self.target_arch == TargetArchType.ANDROID)
        device_identifier = None
        if eval_on_android_device:
            # if the user has provided device id (hexa decimal format) create a Remote Device with
            # serial id provided
            if self.device_id:
                device_identifier = RemoteDeviceIdentifier(serial_id=self.device_id)
            self.target = Target(type=DevicePlatformType.ANDROID, identifier=device_identifier)
        else:
            # Note: Currently supports only Android and X68 targets.
            self.target = Target(type=DevicePlatformType.X86_64_LINUX, identifier=device_identifier)

        # Set given AIC device ID with backend extension param
        if self.backend == BackendType.AIC and self.device_id is not None:
            if self.backend_extensions is not None:
                # if the user has provided runtime_device_ids in inference schema
                if self.backend_extensions.runtime_device_ids is not None:
                    qacc_file_logger.warning(
                        "Using runtime_device_ids=[{self.device_id}] instead"
                        " of the user provided value in the Backend extensions."
                    )
                # Expected format of providing device ids for AIC backend extensions:
                # "runtime_device_ids": [0,1]
                self.backend_extensions.runtime_device_ids = [self.device_id]
            else:
                # When the user doesn't specify backend extension in Inference Schema
                self.backend_extensions = AICBackendExtensions(runtime_device_ids=[self.device_id])
        super()._setup()

    def setup_inference_engine(self):
        """Method for initializing the worker group setup"""
        qacc_file_logger.info(f"Initializing {self.inference_schema_name} worker group")
        # Initialize any necessary resources or configurations here

        self.net_runner_module_obj = net_runner.NetRunner()  # Need to persist this as well\
        if self.netrun_params:
            # netrun_params from model config is supplied via InferenceConfig
            netrun_params_dict = self.netrun_params.model_dump(exclude_unset=True,
                                                            exclude=['backend_extensions'])
            infer_config = net_runner.InferenceConfig(**netrun_params_dict)
        else:
            infer_config = net_runner.InferenceConfig(log_level='error')

        inference_identifier = net_runner.InferenceIdentifier(model=self.model, target=self.target,
                                                                backend=self.backend)
        net_runner_load_arg_config = net_runner.NetRunnerLoadArgConfig(
                                            identifier=inference_identifier,
                                            inference_config=infer_config,
                                            backend_config_file=self.backend_config_file,
                                            backend_config_dict=self.backend_config_dict)
        self.net_runner_load_output_config = self.net_runner_module_obj.load(
                                                            net_runner_load_arg_config)

    def teardown(self):
        """Method for cleaning up the worker group"""
        qacc_file_logger.info(f"Teardown for {self.inference_schema_name} worker group")
        try:
            unload_arg_config = net_runner.NetRunnerUnloadArgConfig(
                                    handle=self.net_runner_load_output_config.handle)
            self.net_runner_module_obj.unload(unload_arg_config)
        except Exception as e:
            raise Exception(f"Failed to unload the artifacts from device. Reason: {e}")
        return True

    def infer(self, data: NDArrayRepresentation) -> NDArrayRepresentation:
        """Method to perform inference"""
        self.net_runner_run_arg_config = net_runner.NetRunnerRunArgConfig(
                                identifier=self.net_runner_load_output_config.handle,
                                input_data=data.data)
        inference_outputs = self.net_runner_module_obj.run(self.net_runner_run_arg_config)
        data.data = list(inference_outputs.output_data[0].values())
        qacc_file_logger.debug(f"{data.idx=} output_data={data.data}")
        return data

    def infer_on_filelist(self, input_list: list[str | Path]) -> list[list[np.ndarray]]:
        """Perform inference on a list of input files.

        Args:
            input_list: A list of file paths or strings representing the input data.

        Returns:
            A list of lists containing numpy arrays representing the inference outputs.
        """
        inference_outputs = []
        self.net_runner_run_arg_config = net_runner.NetRunnerRunArgConfig(
                            identifier=self.net_runner_load_output_config.handle,
                            input_data=input_list)
        netrunner_output = self.net_runner_module_obj.run(self.net_runner_run_arg_config)
        for idx, output_data in enumerate(netrunner_output.output_data):
            output_data = list(output_data.values())
            output_data = [np.asarray(output_data[out_idx].reshape(out_info[1]),
                                            dtype=out_info[0], order="C")
                            for out_idx, out_info in enumerate(self.output_info.values())
                            ]
            inference_outputs.append(output_data)
        return inference_outputs
