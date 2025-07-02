# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
import os
import tempfile
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.common.utilities import timer_decorator
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger
from qti.aisw.accuracy_evaluator.qacc.config_definitions import (
    CompilationParams,
    ConverterParams,
    InferenceEngineType,
    InferenceSchemaConfiguration,
    QuantizerParams,
)
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.tools.core.modules.api.definitions.common import BackendType, Model
from qti.aisw.tools.core.modules.context_bin_gen import context_bin_gen_module
from qti.aisw.tools.core.modules.converter import (
    BackendInfoConfig,
    converter_module,
    optimizer_module,
    quantizer_module,
)
from qti.aisw.tools.core.modules.converter.common import DLCBackendConfig
from qti.aisw.tools.core.modules.converter.constants import (
    PytorchFrameworkInfo,
    TensorflowFrameworkInfo,
)
from qti.aisw.tools.core.modules.converter.utils import get_framework


class QAIRTCompiler:
    """QAIRTCompiler class for performing model compilation using QAIRT SDK"""

    def __init__(
        self,
        calibration_file: Optional[str | Path] = None,
        inputs_info: Optional[Dict[str, List[Any]]] = None,
        outputs_info: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """Initialize QAIRTCompiler with parameters

        Args:
            calibration_file (Optional[str | Path]): Path to calibration file for quantization
            inputs_info (Optional[Dict[str, List[Any]]]): Information about input tensors
            outputs_info (Optional[Dict[str, List[Any]]]): Information about output tensors
        """
        self.inputs_info = inputs_info
        self.outputs_info = outputs_info
        self.calibration_file = calibration_file

    def compile_process(self, *args, **kwargs):
        """Entry function invoked by the processpool executor.
        This is required for the timer_decorator to work correctly.
        """
        return self.compile(*args, **kwargs)

    @timer_decorator
    def compile(
        self,
        model_path: str | Path,
        inference_schema_name: str,
        inference_schema_type: InferenceEngineType,
        precompiled_path: Optional[str] = None,
        compilation_params: Optional[CompilationParams] = None,
        out_dir: Optional[str | Path] = None,
    ) -> tuple[int, Model | str, dict]:
        """Compile the given source model using any given compilation parameters.

        Args:
            model_path (str | Path): Path to the model file
            inference_schema_name (str): Name of the inference schema to use
            inference_schema_type (InferenceEngineType): Type of inference engine
            precompiled_path (str, optional): Path to a precompiled model.
                Defaults to None.
            compilation_params (CompilationParams, optional): Parameters for compiling the model.
                Defaults to None.
            out_dir (str | Path, optional): Path to a directory for storing compiled artifacts.
                Defaults to None.

        Returns:
            tuple[int, Model | str, dict]: A tuple containing:
                - int: Compilation status (success or failure)
                - Model or str: The compiled model or path
                - dict: Timing information for each compilation stage
        Raises:
            TypeError: If compilation_params is not of type CompilationParams

        """
        compiled_model = None
        compilation_timing = {}
        if precompiled_path is not None and compilation_params:
            # Case when precompiled path & compilation_params is provided
            qacc_file_logger.warning(
                f"Compilation parameters for {inference_schema_name} will be ignored "
                "because a precompiled model path is specified."
            )
        elif not compilation_params:
            # Case when compilation_params is not given. Default values to be used.
            compilation_params = CompilationParams()
        if not isinstance(compilation_params, CompilationParams):
            raise TypeError("compilation_params must be of type CompilationParams")

        try:
            if inference_schema_type != InferenceEngineType.QNN:
                compiled_model = model_path
            elif precompiled_path:
                # if precompiled_path is present assume it as precompiled context_binary and use it
                precompiled_path = Path(precompiled_path)
                if precompiled_path.suffix == ".dlc":
                    compiled_model = Model(dlc_path=precompiled_path)
                elif precompiled_path.suffix == ".bin":
                    compiled_model = Model(context_binary_path=precompiled_path)
                else:
                    raise ValueError(
                        f"Unsupported file format for precompiled model: {precompiled_path}."
                        " Only .dlc and .bin formats are supported."
                    )
                qacc_file_logger.info(
                    f"Using the provided precompiled binary {precompiled_path} for {inference_schema_name}."
                )
            else:
                model_file_name = Path(model_path).stem
                binary = None
                # Unpack fields from compilation params
                converter_params = compilation_params.converter_params
                quantizer_params = compilation_params.quantizer_params
                backend = compilation_params.backend
                backend_config_dict = compilation_params.context_backend_extensions_dict
                backend_config_file = compilation_params.context_backend_extensions_json
                offline_prepare = compilation_params.offline_prepare
                soc_model = compilation_params.soc_model

                if backend:
                    qacc_file_logger.info(
                        f"Performing model compilation for given inference_schema = {inference_schema_name}"
                    )
                    backend_info = BackendInfoConfig(
                        backend=backend.value, soc_model=soc_model
                    )
                    compiled_model_suffix = f"_{backend}"
                else:
                    qacc_file_logger.info("Performing generic model compilation")
                    backend_info = None
                    compiled_model_suffix = ""

                if out_dir:
                    out_dir = Path(out_dir)
                    if not out_dir.exists():
                        out_dir.mkdir(parents=True, exist_ok=True)
                else:
                    out_dir = Path(tempfile.mkdtemp(prefix="qairt_compile_"))

                # Converter Step
                convert_time, (ir_graph, framework, dlc_backend_config) = self.convert(
                    model_path,
                    converter_params,
                    inference_schema_name=inference_schema_name,
                )
                compilation_timing["convert"] = convert_time
                dlc_path = out_dir / f"{model_file_name}{compiled_model_suffix}.dlc"

                # Optimizer Step
                optimize_time, dlc = self.optimize(
                    ir_graph=ir_graph,
                    framework=framework,
                    output_path=dlc_path,
                    dlc_backend_config=dlc_backend_config,
                    backend_info=backend_info,
                    inference_schema_name=inference_schema_name,
                )
                compilation_timing["optimize"] = optimize_time

                # Quantizer Step if applicable
                if quantizer_params:
                    if len(quantizer_params.output_dlc) == 0:
                        compiled_model_suffix = f"_quantized{compiled_model_suffix}"
                        quantized_dlc_path = (
                            out_dir / f"{model_file_name}{compiled_model_suffix}.dlc"
                        )
                    else:
                        quantized_dlc_path = quantizer_params.output_dlc

                    quantize_time, dlc = self.quantize(
                        input_dlc=dlc,
                        output_path=quantized_dlc_path,
                        quantizer_params=quantizer_params,
                        backend_info=backend_info,
                        inference_schema_name=inference_schema_name,
                    )
                    compilation_timing["quantize"] = quantize_time
                compiled_model = Model(dlc_path=dlc)

                # Context Binary Generation Step: If offline_prepare is True
                if offline_prepare:
                    generate_binary_time, binary = self.generate_binary(
                        dlc_file=dlc,
                        backend_config_dict=backend_config_dict,
                        backend_config_file=backend_config_file,
                        backend=backend,
                        inference_schema_name=inference_schema_name,
                    )
                    compilation_timing["generate_binary"] = generate_binary_time
                    compiled_model = Model(context_binary_path=binary)
                qacc_file_logger.info(
                    f"Completed model compilation for inference_schema = {inference_schema_name}."
                )

            status = qcc.SCHEMA_COMPILE_SUCCESS
        except Exception as e:
            status = qcc.SCHEMA_COMPILE_FAIL
            qacc_file_logger.error(f"Failed to Compile {inference_schema_name}. {e}")
        return status, compiled_model, compilation_timing

    @timer_decorator
    def convert(
        self,
        model_path: str | Path,
        converter_params: ConverterParams = None,
        inference_schema_name: str = "",
    ) -> tuple[IROpGraph, str, dict]:
        """Convert source model to IR graph using QAIRT converter

        Args:
            model_path (str | Path): Path to the source model file
            converter_params (ConverterParams, optional): Additional conversion parameters
                Defaults to None.
            inference_schema_name (str, optional): Name of the inference schema
                Defaults to empty string.

        Returns:
            tuple[IROpGraph, str, dict]: A tuple containing:
                - IROpGraph: The converted IR graph object
                - str: The framework name of the original model
                - dict: DLC backend configuration generated during conversion

        Raises:
            QAIRTConverterException: If the model conversion fails
        """
        qacc_file_logger.debug("Preparing converter module config")
        _, model_ext = os.path.splitext(model_path)
        model_framework_type = get_framework(model_ext)

        # ONNX & TFlite Case: Input and output info are optional
        input_tensor_config = [
            converter_module.InputTensorConfig(
                name=in_name, source_model_input_datatype=in_info[0]
            )
            for in_name, in_info in self.inputs_info.items()
        ]
        output_tensor_config = None

        if model_framework_type in [
            TensorflowFrameworkInfo.name,
            PytorchFrameworkInfo.name,
        ]:
            # TF Case: input with shape info and output names required
            input_tensor_config = [
                converter_module.InputTensorConfig(
                    name=in_name,
                    source_model_input_datatype=in_info[0],
                    desired_input_shape=",".join(str(dim) for dim in in_info[1]),
                )
                for in_name, in_info in self.inputs_info.items()
            ]
            output_tensor_config = [
                converter_module.OutputTensorConfig(name=out_name)
                for out_name in self.outputs_info.keys()
            ]

        if converter_params is None:
            converter_args = converter_module.ConverterInputConfig(
                input_network=str(model_path),
                input_tensors=input_tensor_config,
                output_tensors=output_tensor_config,
            )
        else:
            qacc_file_logger.debug(
                f"{inference_schema_name} \
                Conversion parameters: {converter_params.model_dump(exclude_unset=True)}"
            )
            converter_args = converter_module.ConverterInputConfig(
                input_network=str(model_path),
                **converter_params.model_dump(exclude_unset=True),
                input_tensors=input_tensor_config,
                output_tensors=output_tensor_config,
            )

        try:
            qacc_file_logger.debug("Initializing Converter module")
            converter = converter_module.QAIRTConverter()
            qacc_file_logger.info(
                f"[{inference_schema_name}] Converting source model to IR"
            )
            converter_output = converter.convert(converter_args)
            ir_graph = converter_output.ir_graph
            framework = converter_output.framework
            dlc_backend_config = converter_output.dlc_backend_config
        except Exception as exception:
            raise ce.QAIRTConverterException(
                "Failed to convert the model!"
            ) from exception
        qacc_file_logger.info(f"[{inference_schema_name}] Completed converting to IR")
        return ir_graph, framework, dlc_backend_config

    @timer_decorator
    def optimize(
        self,
        ir_graph: IROpGraph,
        framework: str,
        output_path: str | Path,
        dlc_backend_config: DLCBackendConfig,
        backend_info: Optional[BackendInfoConfig] = None,
        inference_schema_name: Optional[str] = "",
    ) -> str:
        """Optimize the converted IR graph using the IROptimizer.

        This method performs the optimization of the input IR graph to generate
        an optimized DLC file. It uses the configured optimizer module to
        perform the optimization and stores the output at the specified path.

        Args:
            ir_graph (IROpGraph): The IR graph to be optimized.
            framework (str): The framework of the source model.
            output_path (str | Path): The output path where the optimized DLC
                will be stored.
            dlc_backend_config (DLCBackendConfig): Configuration for the DLC backend.
            backend_info (BackendInfoConfig, optional): Additional backend information.
                Defaults to None.
            inference_schema_name (str, optional): Name of the inference schema.
                Defaults to an empty string.

        Returns:
            str: Path to the DLC file generated by serializing the optimized IR graph.
        """
        qacc_file_logger.debug("Preparing optimizer module config")
        optimizer_args = optimizer_module.OptimizerInputConfig(
            ir_graph=ir_graph,
            framework=framework,
            output_dlc=str(output_path),
            dlc_backend_config=dlc_backend_config,
            backend_info=backend_info,
        )

        try:
            qacc_file_logger.debug("Initializing Optimizer module")
            optimizer = optimizer_module.QAIRTOptimizer()
            qacc_file_logger.info(f"[{inference_schema_name}] Optimizing IR graph")
            optimizer_output = optimizer.optimize(optimizer_args)
            optimized_dlc = optimizer_output.dlc_path
        except Exception as exception:
            raise ce.QAIRTOptimizerException(
                "Failed to optimize the model!"
            ) from exception
        qacc_file_logger.info(
            f"[{inference_schema_name}] Completed optimization of IR graph."
        )
        return optimized_dlc

    @timer_decorator
    def quantize(
        self,
        input_dlc: str,
        output_path: str,
        quantizer_params: Optional[QuantizerParams] = None,
        backend_info: Optional[BackendInfoConfig] = None,
        inference_schema_name: str = "",
    ) -> str:
        """Quantize the given DLC file.

        This method quantizes a DLC file using the specified quantizer parameters
        and backend information. The quantized model is saved to the provided
        output path.

        Args:
            input_dlc (str): Path to the DLC file that needs to be quantized.
            output_path (str): File path to be used for saving the quantized DLC.
            quantizer_params (QuantizerParams, optional):
                Quantization parameters. Defaults to None.
            backend_info (BackendInfoConfig, optional):
                Backend information for quantization. Defaults to None.
            inference_schema_name (str, optional):
                Name of the inference schema. Defaults to empty string.

        Returns:
            str: Path to the quantized DLC file.
        """
        qacc_file_logger.debug("Preparing quantizer module config")
        if quantizer_params is None:
            quant_args = quantizer_module.QuantizerInputConfig(
                input_dlc=str(input_dlc), input_list=self.calibration_file
            )
        else:
            qacc_file_logger.debug(
                f"{inference_schema_name} \
                Quantization parameters: {quantizer_params.model_dump(exclude_unset=True)}"
            )
            quant_args = quantizer_module.QuantizerInputConfig(
                input_dlc=str(input_dlc),
                input_list=self.calibration_file,
                **quantizer_params.model_dump(exclude_unset=True),
            )

        quant_args.output_dlc = str(output_path)

        if backend_info:
            quant_args.backend_info = backend_info

        try:
            qacc_file_logger.debug("Initializing Quantizer module")
            quantizer = quantizer_module.QAIRTQuantizer()
            qacc_file_logger.info(f"[{inference_schema_name}] Performing quantization")
            quantizer_output = quantizer.quantize(quant_args)
            quantized_dlc = quantizer_output.dlc_output
        except Exception as exception:
            raise ce.QAIRTQuantizerException(
                "Failed to quantize the model!"
            ) from exception

        qacc_file_logger.info(f"[{inference_schema_name}] Completed quantization")
        qacc_file_logger.debug(f"Quantized model saved at: {quantized_dlc}")
        return quantized_dlc

    @timer_decorator
    def generate_binary(
        self,
        dlc_file: str | Path,
        backend: BackendType,
        backend_config_file: Optional[os.PathLike] = None,
        backend_config_dict: Optional[dict] = None,
        inference_schema_name: str = "",
    ) -> str:
        """Generate binary from DLC using context-binary-generator module.

        Args:
            dlc_file (str | Path): Path to the DLC file.
            backend (BackendType): The target backend for binary generation.
            backend_config_file (Optional[os.PathLike]): Path to the backend
                configuration file.
            backend_config_dict (Optional[dict]): Dictionary containing backend
                configuration parameters.
            inference_schema_name (str): Name of the inference schema for logging purposes.

        Returns:
            str: Path to the generated binary.
        """
        out_dir = Path(dlc_file).parent
        name = Path(dlc_file).stem
        qacc_file_logger.debug("Preparing arg config of context-bin-gen module")
        generate_config = context_bin_gen_module.GenerateConfig(log_level="error")

        context_bin_gen_config = context_bin_gen_module.ContextBinGenArgConfig(
            backend=backend,
            backend_config_dict=backend_config_dict,
            backend_config_file=backend_config_file,
            model=Model(dlc_path=dlc_file),
            output_dir=out_dir,
            output_filename=name,
            generate_config=generate_config,
        )

        try:
            qacc_file_logger.debug("Initializing context-bin-gen module")
            context_bin_gen = context_bin_gen_module.ContextBinGen()
            qacc_file_logger.info(
                f"[{inference_schema_name}] Starting offline graph preparation"
            )
            output_config = context_bin_gen.generate(context_bin_gen_config)
            binary_path = output_config.context_binary.context_binary_path
            qacc_file_logger.info(
                f"[{inference_schema_name}] Completed offline graph preparation"
            )
            qacc_file_logger.debug(
                f"[{inference_schema_name}] Binary file saved at %s", binary_path
            )
        except Exception as exception:
            raise ce.QnnContextBinaryGeneratorException(
                f"[{inference_schema_name}] Failed to prepare graph!"
            ) from exception

        return binary_path


class CompilationEngine:
    """A class to manage the compilation process for accuracy evaluator.

    This class handles the parallel compilation using worker processes,
    and offers methods to prepare compilation arguments based on the given inference schema.
    """

    def __init__(
        self,
        work_dir: os.PathLike,
        inputs_info: Dict[str, List[Any]],
        outputs_info: Dict[str, List[Any]],
        num_workers: Optional[int] = None,
        calibration_file: Optional[os.PathLike] = None,
    ):
        """Initialize the CompilationEngine.

        Args:
            work_dir: The working directory for compilation tasks.
            inputs_info: Dictionary containing input tensor information.
            outputs_info: Dictionary containing output tensor information.
            num_workers: Number of worker processes for parallel compilation.
            calibration_file: Path to the calibration file for quantization.
        """
        self.inputs_info = inputs_info
        self.outputs_info = outputs_info
        self.calibration_file = calibration_file
        self.work_dir = work_dir
        self.workers = ProcessPoolExecutor(max_workers=num_workers)
        self.compilation_futures = {}

    def prepare_compile_args_for_schema(
        self, inference_schema: InferenceSchemaConfiguration
    ):
        """Prepare compile arguments for the conversion process.

        This method updates the input/output tensor config in converter params based on the
        model framework, and prepares the quantizer parameters by adding the calibration file
        to the input list.

        Args:
            inference_schema: The inference schema containing various parameters.

        Returns:
            compiler_params: An instance of CompilationParams with updated values.
        """
        # Initialize variables to store user provided backend extensions in
        # JSON & dictionary formats
        context_backend_extensions_json, context_backend_extensions_dict = None, None
        if (
            inference_schema.contextbin_params
            and inference_schema.contextbin_params.backend_extensions
        ):
            context_backend_extensions_json = (
                inference_schema.contextbin_params.backend_extensions
            )
        elif inference_schema.backend_extensions:
            context_backend_extensions_dict = (
                inference_schema.backend_extensions.get_context_bin_config_dict()
            )

        # Create CompilationParams instance and populate it with updated values
        compiler_params = CompilationParams(
            backend=inference_schema.backend,
            quantizer_params=inference_schema.quantizer_params,
            converter_params=inference_schema.converter_params,
            context_backend_extensions_json=context_backend_extensions_json,
            context_backend_extensions_dict=context_backend_extensions_dict,
        )
        return compiler_params

    def start_compile(
        self, inference_schemas: List[InferenceSchemaConfiguration]
    ) -> Dict[str, Tuple[Future, InferenceSchemaConfiguration]]:
        """Start the compilation process for all provided inference schemas.

        This method iterates through each inference schema, prepares the
        compilation arguments, and submits a compilation task to the workers.

        Args:
            inference_schemas: A list of InferenceSchemaConfiguration objects
                              representing the schemas to compile.

        Returns:
            A dictionary mapping inference schema names to tuples of:
            - A Future object representing the compilation task
            - The corresponding InferenceSchemaConfiguration object
        """
        # Iterate over each inference schema and prepare compilation arguments
        for inference_schema in inference_schemas:
            inference_schema_name = inference_schema.get_inference_schema_name()
            compiler_params = self.prepare_compile_args_for_schema(inference_schema)

            if (
                inference_schema.quantizer_params
                and inference_schema.quantizer_params.float_fallback
            ):
                # when float_fallback is set to True, calibration_file should be None
                calibration_file = None
            else:
                calibration_file = self.calibration_file

            compiler = QAIRTCompiler(
                calibration_file=calibration_file,
                inputs_info=self.inputs_info,
                outputs_info=self.outputs_info,
            )

            # Get the name of the current inference schema
            # output directory for compilation results
            compilation_out_dir = Path(self.work_dir) / "infer" / inference_schema_name
            if not compilation_out_dir.exists():
                compilation_out_dir.mkdir(parents=True, exist_ok=True)

            # Submit a compilation task to the workers and store the resulting future
            compiled_model_path_future = self.workers.submit(
                compiler.compile_process,
                compilation_params=compiler_params,
                inference_schema_name=inference_schema_name,
                inference_schema_type=inference_schema.name,
                model_path=inference_schema._model_path,
                precompiled_path=inference_schema.precompiled_path,
                out_dir=compilation_out_dir,
            )
            # Dump Converter, Quantizer params to the out_dir
            if compiler_params.converter_params:
                self._dump_params_to_json(
                    compiler_params.converter_params,
                    params_type="converter",
                    output_dir=compilation_out_dir,
                )
            if compiler_params.quantizer_params:
                self._dump_params_to_json(
                    compiler_params.quantizer_params,
                    params_type="quantizer",
                    output_dir=compilation_out_dir,
                )

            self.compilation_futures[inference_schema_name] = (
                compiled_model_path_future,
                inference_schema,
            )
        return self.compilation_futures

    def _dump_params_to_json(
        self,
        params: QuantizerParams | ConverterParams,
        params_type: str,
        output_dir: str | Path,
    ) -> None:
        """Dump the given parameters to a JSON file in the specified output directory.

        Args:
            params: The Quantizer/Coverter parameters to dump.
            params_type: The type of parameters (e.g., "converter", "quantizer").
            output_dir: The directory where the JSON file will be saved.
        """
        outfile = Path(output_dir) / f"{params_type}_params_list.json"
        data = {
            f"{params_type}_params": params.model_dump(exclude_unset=True)
            }
        with open(outfile, "w", encoding="utf-8") as f:
          json.dump(data, f, indent=4, ensure_ascii=False)
