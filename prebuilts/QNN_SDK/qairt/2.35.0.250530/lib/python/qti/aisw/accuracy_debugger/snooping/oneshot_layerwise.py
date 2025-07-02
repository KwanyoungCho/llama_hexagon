# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    InputSample,
    QuantizerInputArguments,
    RemoteHostDetails,
)
from qti.aisw.accuracy_debugger.snooping.snooper import Snooper
from qti.aisw.accuracy_debugger.snooping.snooper_utils import filter_snooping_report
from qti.aisw.accuracy_debugger.utils.constants import Algorithm
from qti.aisw.accuracy_debugger.utils.exceptions import VerificationError
from qti.aisw.tools.core.modules.api.definitions.common import BackendType
from qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module import GenerateConfig
from qti.aisw.tools.core.modules.net_runner.net_runner_module import InferenceConfig
from qti.aisw.tools.core.utilities.comparators.comparator import Comparator
from qti.aisw.tools.core.utilities.comparators.mse import MSEComparator
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType


class OneshotLayerwiseSnooper(Snooper):
    """Subclass for oneshot layerwise algorithm."""

    def __init__(self, model: Path, logger: logging.Logger):
        """Initializes the OneshotLayerwiseSnooper.

        Args:
            model (Path): Path to framework model.
            logger (logging.Logger): A python logger instance
        """
        super().__init__(model=model, name=Algorithm.ONESHOT, logger=logger)

    def run(
        self,
        input_tensors: list[InputSample] | dict[str, NDArray],
        backend: BackendType,
        platform: DevicePlatformType,
        converter_args: Optional[ConverterInputArguments] = None,
        quantizer_args: Optional[QuantizerInputArguments] = None,
        context_bin_args: Optional[GenerateConfig] = None,
        context_bin_backend_extension: Optional[Path | dict] = None,
        offline_prepare: Optional[bool] = None,
        net_runner_args: Optional[InferenceConfig] = None,
        net_run_backend_extension: Optional[Path | dict] = None,
        remote_host_details: Optional[RemoteHostDetails] = None,
        soc_model: str = "",
        comparators: list[Comparator] = [MSEComparator()],
        working_directory: Optional[Path] = None,
        golden_reference_path: Optional[Path] = None,
        retain_compilation_artifacts: bool = False,
        dump_output_tensors: bool = False,
    ) -> Path:
        """This is entry point method of algorithm. It accepts data necessary to debug the model and
        generates report.

        Args:
           input_tensors: List of InputSample or dictionary of tensor name to numpy array.
           backend: Backend type.
           platform: Target platform.
           converter_args: Conversion arguments.
           quantizer_args: Quantization arguments.
           context_bin_args: Context binary generation arguments.
           context_bin_backend_extension: Backend extension config for context binary generator.
           offline_prepare: Boolean to indicate offline prepare of graph.
           net_runner_args: Net runner arguments.
           net_run_backend_extension: Backend extension config for net-runner.
           remote_host_details: Details of remote host.
           soc_model : Name of SOC model on target device.
           comparators: List of comparators to use in verification stage
           working_directory: Path to directory to store artifacts.
           golden_reference_path: Path to directory containing reference outputs.
           retain_compilation_artifacts: Flag to retain compilation artifacts. Default is
                             set to False.
           dump_output_tensors: Boolean to indicate whether to dump output tensors.

        Returns:
           Path: File path to snooping report.
        """
        if not working_directory:
            working_directory = Path.cwd() / "working_directory"
            working_directory.mkdir(exist_ok=True)

        working_directory = (
            working_directory
            / f"{self._name}_snooping"
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        working_directory.mkdir(parents=True)

        input_sample = self._load_input_tensors(input_tensors)

        if golden_reference_path:
            # Load raw files into memory
            reference_outputs = self._load_data_from_directory(golden_reference_path)
        else:
            # Generate reference outputs using framework manager.
            reference_outputs = self._generate_reference_data(
                self.model, input_sample, dump_output_tensors, working_directory
            )

        # Enable intermediate output dumping.
        if offline_prepare is False or backend not in BackendType.offline_preparable_backends():
            if net_runner_args:
                net_runner_args.debug = True
            else:
                net_runner_args = InferenceConfig(debug=True)
        else:
            if context_bin_args:
                context_bin_args.enable_intermediate_outputs = True
            else:
                context_bin_args = GenerateConfig(enable_intermediate_outputs=True)

        # Enable framework traces in converter args.
        if converter_args:
            converter_args.enable_framework_trace = True
        else:
            converter_args = ConverterInputArguments(enable_framework_trace=True)

        inference_output_directory = working_directory / "inference_engine"
        inference_output_directory.mkdir(exist_ok=True)
        inference_output_config = self._execute_inference(
            model=self.model,
            converter_args=converter_args,
            quantizer_args=quantizer_args,
            context_bin_args=context_bin_args,
            context_bin_backend_extension=context_bin_backend_extension,
            offline_prepare=offline_prepare,
            net_runner_args=net_runner_args,
            net_run_backend_extension=net_run_backend_extension,
            input_sample=input_sample,
            backend=backend,
            platform=platform,
            remote_host_details=remote_host_details,
            working_directory=inference_output_directory,
            soc_model=soc_model,
            dump_output_tensors=dump_output_tensors,
        )

        inference_outputs = inference_output_config.output_data
        converted_dlc = inference_output_config.converter_dlc
        # Compare framework outputs with inference output and generate snooping report.
        verifier_output = self._verify(
            reference_outputs, inference_outputs[0], converted_dlc, comparators
        )

        if not verifier_output:
            raise VerificationError(
                "Verification of tensors failed. Please check the logs for more details."
            )

        if not retain_compilation_artifacts:
            self._logger.info(
                f"Cleaning up compilation artifacts:\n"
                f"Converter DLC: {inference_output_config.converter_dlc}\n"
                f"Quantizer DLC: {inference_output_config.quantizer_dlc}\n"
                f"Offline Graph: {inference_output_config.offline_graph}"
            )
            inference_output_config.cleanup_artifacts()

        snooping_report_path = self._generate_snooping_report(
            verifier_output, comparators, working_directory, inference_outputs[0]
        )

        self._plot_graphs(
            csv_path=snooping_report_path,
            layer_names_column="op name",
            comparator_columns=[comparator.name for comparator in comparators],
            output_dir=working_directory,
        )

        return snooping_report_path

    @staticmethod
    def _generate_snooping_report(
        verifier_output: dict[tuple[str, str], dict],
        comparators: list[Comparator],
        working_directory: Path,
        inference_data: dict[str, NDArray],
    ) -> Path:
        """Generate a CSV snooping report from verification output data.

        Args:
            verifier_output: Output from verifier module.
            comparators: List of comparator used for verification.
            working_directory: Directory to store report.
            inference_data: Dictionary containing outputs of the model

        Returns:
            Path: Path to the generated snooping report.
        """
        snooping_report = []
        for key, value in verifier_output.items():
            temp = {
                "op name": key[1],
                "op type": value["op_type"],
                "size": np.prod(value["dimensions"]),
                "dimensions": value["dimensions"],
            }
            for comparator in comparators:
                temp[comparator.name] = value[comparator.name]
            snooping_report.append(temp)

        snooping_report = pd.DataFrame(snooping_report)
        snooping_report = filter_snooping_report(snooping_report, inference_data)

        snooper_report_file = working_directory / "oneshot_layerwise.csv"

        snooping_report.to_csv(snooper_report_file, index=False)

        return snooper_report_file
