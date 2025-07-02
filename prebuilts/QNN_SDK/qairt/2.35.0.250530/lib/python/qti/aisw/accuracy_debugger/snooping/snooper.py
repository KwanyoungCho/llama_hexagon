# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike, NDArray
from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    InputSample,
    QuantizerInputArguments,
    RemoteHostDetails,
)
from qti.aisw.accuracy_debugger.framework_runner.framework_factory import get_framework_type
from qti.aisw.accuracy_debugger.inference_engine.qairt_inference_engine import (
    InferenceEngine,
    InferenceEngineInputConfig,
    InferenceEngineOutputConfig,
)
from qti.aisw.accuracy_debugger.tensor_visualizer.visualizers import line_plot
from qti.aisw.tools.core.modules.api.definitions.common import BackendType
from qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module import (
    GenerateConfig,
)
from qti.aisw.tools.core.modules.net_runner.net_runner_module import InferenceConfig
from qti.aisw.tools.core.utilities.comparators.comparator import Comparator
from qti.aisw.tools.core.utilities.comparators.mse import MSEComparator
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DevicePlatformType,
)
from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.tensor_mapping.tensor_mapping import (
    TensorMapper,
    TensorMapperInputConfig,
)
from qti.aisw.tools.core.utilities.verifier.verifier import Verifier


class Snooper(ABC):
    """Abstract class for all the snooper algorithms."""

    def __init__(self, model: Path, name: str, logger: logging.Logger):
        """Initializes the Snooper.

        Args:
            model (Path): Path to framework model.
            name (str): The name of the snooper algorithm
            logger (logging.Logger): A python logger instance
        """
        self.model = Path(model)
        self.framework_type = get_framework_type(self.model)
        self._name = name
        self._logger = logger

    @abstractmethod
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
        comparators: list[Comparator] = [MSEComparator],
        working_directory: Optional[Path] = None,
        golden_reference_path: Optional[Path] = None,
        dump_output_tensors: bool = False,
    ) -> Path:
        """This is abstract method, and it should be implemented by the child class algorithms.

        Args:
           input_tensors : List of InputSample or dictionary of tensor name to numpy array.
           backend: Backend type.
           platform: Target platform.
           converter_args: Input arguments required by the converter module.
           quantizer_args: Input arguments required by the quantizer module.
           context_bin_args: Input arguments required by context_bin_gen module.
           context_bin_backend_extension: Backend extension config file for context binary generator.
           offline_prepare: Boolean to indicate offline prepare of graph.
           net_runner_args: Input arguments required by the netrunner module.
           net_run_backend_extension: Backend extension config file for net-runner.
           remote_host_details: Details of remote host.
           soc_model : Name of SOC model on target device.
           comparators: List of comparators to use in verification stage.
           working_directory: Path to directory to store artifacts.
           golden_reference_path: Path to directory containing reference outputs.
           dump_output_tensors: Boolean to indicate whether to dump output tensors.

        Returns:
           Path: File path to snooping report.
        """

    @staticmethod
    def _load_input_tensors(
        input_tensors: list[InputSample] | dict[str, NDArray],
    ) -> dict[str, NDArray] | None:
        """Load input tensors from raw files to numpy arrays.

        Args:
            input_tensors: List of input tensors.

        Returns:
            dict[str, NDArray]: Dictionary of numpy arrays containing input tensor data.
        """
        if isinstance(input_tensors, list):
            input_tensors_dict = {}
            for tensor in input_tensors:
                dtype = tensor.data_type if tensor.data_type else np.float32
                input_tensors_dict[tensor.name] = np.resize(
                    np.fromfile(tensor.raw_file, dtype), tensor.dimensions
                )
            return input_tensors_dict
        else:
            return input_tensors

    def _generate_reference_data(
        self,
        model: Path,
        inputs: dict[str, NDArray],
        dump_output_tensors: bool = False,
        output_dir: Optional[Path] = None,
    ) -> dict[str, NDArray]:
        """Generate golden reference outputs for the model using framework manager utility.

        Args:
            model: Path to model file.
            inputs: Dictionary of input tensors.
            dump_output_tensors: Boolean to indicate whether to dump output tensors.
            output_dir: Path to directory to store output tensors

        Returns:
            Dictionary containing reference outputs.
        """
        framework_manager = FrameworkManager(self._logger)
        framework_model = framework_manager.load(model)
        reference_outputs = framework_manager.generate_intermediate_outputs(framework_model, inputs)
        if dump_output_tensors:
            if output_dir is None:
                raise ValueError(
                    "Output directory must be specified when dump_output_tensors is True."
                )
            output_dir = output_dir / "reference_output"
            Helper.save_output_to_file(reference_outputs, output_dir)

        return reference_outputs

    def _execute_inference(
        self,
        model: Path,
        converter_args: Optional[ConverterInputArguments] = None,
        quantizer_args: Optional[QuantizerInputArguments] = None,
        context_bin_args: Optional[GenerateConfig] = None,
        context_bin_backend_extension: Optional[Path | dict] = None,
        offline_prepare: Optional[bool] = None,
        net_runner_args: Optional[InferenceConfig] = None,
        net_run_backend_extension: Optional[Path | dict] = None,
        input_sample: Optional[dict[str, NDArray]] = None,
        backend: Optional[BackendType] = None,
        platform: Optional[DevicePlatformType] = None,
        remote_host_details: Optional[RemoteHostDetails] = None,
        working_directory: Optional[Path] = None,
        soc_model: str = "",
        dump_output_tensors: bool = False,
    ) -> InferenceEngineOutputConfig:
        """Compile and execute model using inference engine.

        Args:
            model: Path to model
            converter_args: Input arguments required by the converter module
            quantizer_args: Input arguments required by the quantizer module
            context_bin_args: Input arguments required by context_bin_gen module
            context_bin_backend_extension: Backend extension config file or dictionary for
                                           context binary generator.
            offline_prepare: Boolean to indicate offline prepare of graph
            net_runner_args: Input arguments required by the netrunner module
            net_run_backend_extension: Backend extension config file or dictionary for net-runner
            input_sample: Input to netrunner module
            backend: Backed type
            platform: Target platform
            remote_host_details: Details of remote host
            working_directory: Path to directory to store artifacts.
            soc_model : Name of SOC model on target device.
            dump_output_tensors: Boolean to indicate whether to dump output tensors.

        Returns:
            InferenceEngineOutputConfig: Inference engine output config containing inference data
                                         and artifacts like, converted dlc, quantized dlc and,
                                         context binary.
        """
        inf_engine_input_config = InferenceEngineInputConfig(
            input_model=model,
            converter_arguments=converter_args,
            quantizer_arguments=quantizer_args,
            context_bin_gen_arguments=context_bin_args,
            context_bin_backend_extension=context_bin_backend_extension,
            offline_prepare=offline_prepare,
            net_run_arguments=net_runner_args,
            net_run_input_data=input_sample,
            net_run_backend_extension=net_run_backend_extension,
            backend=backend,
            platform=platform,
            remote_host_details=remote_host_details,
            working_directory=working_directory,
            soc_model=soc_model,
            dump_output=dump_output_tensors,
        )
        inf_engine = InferenceEngine(self._logger)
        inf_output_config = inf_engine.run_inference_engine(inf_engine_input_config)

        return inf_output_config

    def _verify(
        self,
        reference_output: dict[str, NDArray],
        inference_output: dict[str, NDArray],
        dlc_file: Path,
        comparators: list[Comparator],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """This method verifies the inference output with reference output using given comparators.

        Args:
            reference_output: Reference output.
            inference_output: Inference output.
            dlc_file: Path to dlc file.
            comparators: List of comparators.

        Returns:
            dict: A dictionary containing the verification results for each comparator.
        """
        verifier = Verifier(comparators, logger=self._logger)
        verifier_output = verifier.verify_dictionary_of_tensors(
            reference_output, inference_output, dlc_file=dlc_file
        )
        return verifier_output

    @staticmethod
    def _get_tensor_mapping(quantized_dlc: Path) -> dict[str, str]:
        """This method converters and quantizes the model using inference engine and extract tensor
        mapping file from quantized DLC using tensor mapping utility.

        Args:
           quantized_dlc: Path to quantized DLC file.

        Returns:
            dict: A dictionary mapping from QNN tensor names to framework names.
        """
        tensor_mapper_input = TensorMapperInputConfig(dlc_path=quantized_dlc)
        tensor_mapper = TensorMapper()
        tensor_mapper_output = tensor_mapper.run(tensor_mapper_input)
        return tensor_mapper_output.tensor_mapping_output

    @staticmethod
    def _load_data_from_directory(
        directory: Path, dtype: DTypeLike = np.float32
    ) -> dict[str, NDArray]:
        """This method recursively visits the directory and loads raw files into numpy array.

        Args:
            directory: Path to directory containing raw files.
            dtype: Data type of raw files. Default is np.float32.

        Returns:
            dict: A dictionary mapping from file name to numpy array.
        """
        tensor_data = {}
        for file in directory.rglob("*.raw"):
            tensor_name = file.stem
            tensor_name = Helper.transform_node_names(tensor_name)
            tensor_data[tensor_name] = np.fromfile(file, dtype)
        return tensor_data

    def _plot_graphs(
        self, csv_path: Path, layer_names_column: str, comparator_columns: list, output_dir: Path
    ) -> None:
        """Plots graphs for verifiers/comparators scores present in the given CSV data.

        Args:
            csv_path: Path to the CSV snooper
            layer_names_column: The name of the column in the CSV that lists the names of the
                                layers in the model
            comparator_columns: List of comparator columns names in the given CSV
            output_dir: Output directory path
        """
        plots_save_dir = output_dir / "plots"
        plots_save_dir.mkdir(exist_ok=True)
        snooping_report_df = pd.read_csv(csv_path)

        for column in comparator_columns:
            try:
                self._logger.debug(f"Plotting graph for {column} scores...")
                line_plot(
                    x=snooping_report_df[layer_names_column],
                    y=[float(item) for item in snooping_report_df[column]],
                    plot_name=column,
                    save_dir=plots_save_dir,
                )
            except Exception as e:
                self._logger.warning(f"Plotting graph failed with error: {e}")

        self._logger.info(f"{self._name} report plots saved at {plots_save_dir}")
