# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import (
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    field_validator,
    model_validator,
)
from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    QuantizerInputArguments,
    RemoteHostDetails,
)
from qti.aisw.accuracy_debugger.utils.constants import (
    supported_backends,
    supported_platforms,
)
from qti.aisw.accuracy_debugger.utils.exceptions import (
    ConversionFailure,
    ExecutionFailure,
    GenerateBinaryFailure,
    OptimizationFailure,
    ParameterError,
    QuantizationFailure,
)
from qti.aisw.tools.core.modules.api.definitions.common import (
    AISWBaseModel,
    BackendType,
    Model,
    Target,
)
from qti.aisw.tools.core.modules.context_bin_gen import (
    GenerateConfig,
    context_bin_gen_module,
)
from qti.aisw.tools.core.modules.converter import (
    BackendInfoConfig,
    converter_module,
    optimizer_module,
    quantizer_module,
)
from qti.aisw.tools.core.modules.net_runner import InferenceConfig, net_runner_module
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DevicePlatformType,
)
from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager
from qti.aisw.tools.core.utilities.framework.utils.constants import (
    OnnxFrameworkInfo,
    PytorchFrameworkInfo,
    TensorflowFrameworkInfo,
    TFLiteFrameworkInfo,
)
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


def validate_backend_platform(backend, platform, offline_prepare) -> None:
    """Validations for combination of backend and platform."""
    if platform and platform not in supported_platforms:
        raise ValueError(
            f"Platform type {platform} is not supported."
            f"Supported platforms are {supported_platforms}"
        )
    if backend:
        if backend not in supported_backends:
            raise ValueError(
                f"Backend type {backend} is not supported."
                f"Supported backends are {supported_backends}"
            )
        if offline_prepare and backend not in BackendType.offline_preparable_backends():
            raise ValueError(
                f"Offline graph preparation is unsupported for {backend} backend."
                f"Supported backends are {BackendType.offline_preparable_backends()}"
            )

    if backend and platform:
        if backend == BackendType.AIC and platform != DevicePlatformType.X86_64_LINUX:
            raise ValueError(f"AIC backend is unsupported for {platform} platform.")

        if backend == BackendType.HTP and (
            platform != DevicePlatformType.ANDROID and platform != DevicePlatformType.X86_64_LINUX
        ):
            raise ValueError(f"HTP backend is unsupported for {platform} platform.")

        if backend == BackendType.GPU and platform != DevicePlatformType.ANDROID:
            raise ValueError(
                f"GPU backend is supported only for android platform but {platform} platform given."
            )


def validate_float_fallback(
    converter_arguments: ConverterInputArguments, quantizer_arguments: QuantizerInputArguments
):
    """Validations for converter and quantizer arguments
    Args:
        converter_arguments: ConverterInputArguments object containing args for converter
        quantizer_arguments: QuantizerInputArguments object containing args for quantizer

    Raises:
        ValueError: If float_fallback is True and quantization_overrides is None
    """
    quantization_overrides = (
        converter_arguments.quantization_overrides if converter_arguments else None
    )
    if quantizer_arguments:
        if quantizer_arguments.float_fallback and not quantization_overrides:
            raise ValueError(
                "External quantization overrides must be provided when using 'float_fallback'."
            )


class InferenceEngineConfig(AISWBaseModel):
    """Base pydantic class for InferenceEngine"""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )


class InferenceEngineInputConfig(InferenceEngineConfig):
    """Input configuration class for Inference Engine

    Attributes:
        input_model: Path to the source model/dlc/bin file
        converter_arguments: Input arguments required by the converter module
        quantizer_arguments: Input arguments required by quantizer module
        backend: Backend type for inference to be run
        platform: The type of device platform to be used for inference
        context_bin_gen_arguments: Input arguments required by the context_bin_gen_module
        context_bin_backend_extension: Backend extension config for context_bin_gen_module
        offline_prepare: Boolean to indicate offline prepare of graph
        net_run_arguments: Input arguments required by the net_run module
        net_run_input_data: Input data to net-runner
        net_run_backend_extension: Backend extension config for net-runner
        dump_output: Enable to dump the results of the netrun into a raw file
        remote_host_details: Details and credentials of the remote host
        working_directory: Path to the directory to store the output result
        soc_model : Name of SOC model on target device.
    """

    _source_model: bool
    input_model: FilePath
    converter_arguments: Optional[ConverterInputArguments] = None
    quantizer_arguments: Optional[QuantizerInputArguments] = None
    backend: Optional[BackendType] = None
    platform: Optional[DevicePlatformType] = None
    context_bin_gen_arguments: Optional[GenerateConfig] = None
    context_bin_backend_extension: Optional[FilePath | dict] = None
    offline_prepare: Optional[bool] = False
    net_run_arguments: Optional[InferenceConfig] = None
    net_run_input_data: Optional[net_runner_module.NetRunnerInputData] = None
    net_run_backend_extension: Optional[FilePath | dict] = None
    dump_output: Optional[bool] = False
    remote_host_details: Optional[RemoteHostDetails] = None
    working_directory: Optional[DirectoryPath] = Field(default=None, validate_default=True)
    soc_model: Optional[str] = ""

    @model_validator(mode="after")
    def validate_input_model(self):
        """Validation for the type of input_model provided based on arguments in input config"""
        self._source_model = False
        try:
            FrameworkManager.infer_framework_type(self.input_model)
            self._source_model = True
        except Exception:
            if self.converter_arguments:
                raise ValueError(
                    "Invalid source model for converter. Support model format are "
                    f"{OnnxFrameworkInfo.name, TensorflowFrameworkInfo.name},"
                    f"{TFLiteFrameworkInfo.name} and {PytorchFrameworkInfo.name}"
                )

        if not self._source_model:
            input_model_suffix = self.input_model.suffix
            if self.quantizer_arguments or self.offline_prepare:
                if input_model_suffix != ".dlc":
                    raise ValueError(
                        "DLC file should be given for quantization or offline prepare but, "
                        f"'{input_model_suffix}' file given."
                    )
            else:
                if input_model_suffix not in [".bin", ".dlc"]:
                    raise ValueError(
                        "'.bin' or '.dlc' file is expected for net-run but, "
                        f"'{input_model_suffix}' file given."
                    )

        return self

    @field_validator("working_directory")
    @classmethod
    def validate_directory(cls, v):
        """Validation and Initialization for the working directory"""
        if v is None:
            v = Path.cwd() / "working_directory"
            v.mkdir(exist_ok=True)
        return v

    @model_validator(mode="after")
    def validate_arguments(self):
        """Validation for the backend and platform provided"""
        validate_backend_platform(self.backend, self.platform, self.offline_prepare)
        if self.offline_prepare:
            # backend is need to prepare offline graph
            if not self.backend:
                raise ValueError("Backend is required to prepare offline graph")
            if self.context_bin_gen_arguments:
                if (
                    self.context_bin_gen_arguments.enable_intermediate_outputs
                    and self.context_bin_gen_arguments.set_output_tensors
                ):
                    raise ValueError(
                        "Either enable_intermediate_outputs or set_output_tensors must be set at a "
                        "time in contex binary module"
                    )
        else:
            if self.context_bin_gen_arguments:
                raise ValueError(
                    "Context binary generation arguments should supplied only when offline prepare "
                    "is enabled"
                )

        if self.net_run_input_data:
            # backend and platform are needed for
            if not self.backend:
                raise ValueError("Backend is required to execute graph")
            if not self.platform:
                raise ValueError("Platform is required to execute graph")
            if self.net_run_arguments:
                if self.net_run_arguments.debug and self.net_run_arguments.set_output_tensors:
                    raise ParameterError(
                        "Either debug or set_output_tensors parameter should be set at a time in "
                        "net-runner module."
                    )
                if self.net_run_arguments.debug or self.net_run_arguments.set_output_tensors:
                    if self.offline_prepare or self.input_model.suffix == ".bin":
                        raise ParameterError(
                            "In offline prepare, the debug or set_output_tensors parameters for "
                            "netrunner should not be set"
                        )
        else:
            if self.net_run_arguments:
                raise ValueError(
                    "Net run input data should be supplied when net_run_arguments are supplied"
                )
        return self

    @model_validator(mode="after")
    def validate_float_fallback(self):
        """Validation for the float fallback"""
        validate_float_fallback(self.converter_arguments, self.quantizer_arguments)
        return self


class InferenceEngineOutputConfig(InferenceEngineConfig):
    """Output configuration class for Inference Engine

    Attributes:
        output_data: Inference output data.
        output_dir: Path to the dumped raw files when dump_output is set.
        converter_dlc: Path to the generated DLC after conversion
        quantizer_dlc: Path to the quantized DLC
        offline_graph: Path to the generated context binary
    """

    output_data: Optional[list[dict[str, np.ndarray]]] = None
    output_dir: Optional[DirectoryPath] = None
    converter_dlc: Optional[FilePath] = None
    quantizer_dlc: Optional[FilePath] = None
    offline_graph: Optional[FilePath] = None

    def cleanup_artifacts(self) -> None:
        """Delete the artifacts generated by the inference engine."""
        attr = self.model_dump()
        exclude_keys = ["output_data", "output_dir"]
        for k, v in attr.items():
            if k not in exclude_keys and v:
                v.unlink(missing_ok=True)


class InferenceEngine:
    """User interface class for model inference.
    Contains methods to convert, quantize, generate_binary and execute the model,
    based on the backend and platform provided in the InferenceEngineInputConfig.
    """

    def __init__(self, logger: Any = None) -> None:
        """Initialize InferenceEngine
        Args:
            logger (Any): Desired python logger
        """
        if logger:
            self.logger = logger
        else:
            self.log_area = LogAreas.register_log_area("Inference")
            self.logger = QAIRTLogger.register_area_logger(area=self.log_area, level="INFO")

    def run_inference_engine(
        self, config: InferenceEngineInputConfig
    ) -> InferenceEngineOutputConfig:
        """Execute Inference Engine
        Args:
            config: InferenceEngineInputConfig object containing args for inference

        Returns:
            InferenceEngineOutputConfig: Compilation artifacts and inference results
        """
        try:
            input_model = config.input_model
            result = InferenceEngineOutputConfig()

            backend_info = None
            if config.backend:
                backend_info = BackendInfoConfig(
                    backend=config.backend.value, soc_model=config.soc_model
                )

            if config._source_model:
                output_dlc_path = config.working_directory / "base.dlc"
                converter_output = self._convert(
                    input_model, output_dlc_path, config.converter_arguments
                )
                converter_dlc = self._optimize(converter_output, output_dlc_path, backend_info)
                input_model = converter_dlc
                result.converter_dlc = converter_dlc

            if config.quantizer_arguments:
                quantized_dlc_path = config.working_directory / "base_quantized.dlc"
                if config.backend and config.backend not in BackendType.quantizable_backends():
                    backend_info = None
                quantized_dlc = self._quantize(
                    input_model,
                    quantized_dlc_path,
                    config.quantizer_arguments,
                    backend_info,
                )
                input_model = quantized_dlc
                result.quantizer_dlc = quantized_dlc

            if config.offline_prepare:
                model_obj = Model(dlc_path=input_model)
                model_obj = self._generate_binary(
                    model_obj,
                    config.working_directory,
                    config.backend,
                    config.context_bin_gen_arguments,
                    config.context_bin_backend_extension,
                )
                result.offline_graph = model_obj.context_binary_path
                input_model = model_obj.context_binary_path

            if config.net_run_input_data:
                if config.platform is None:
                    raise ValueError("platform is mandatory for inference")

                if str(input_model).endswith(".bin"):
                    model_obj = Model(context_binary_path=input_model)
                else:
                    model_obj = Model(dlc_path=input_model)

                target = self._create_target(config.platform, config.remote_host_details)
                netrun_output = self._execute(
                    model_obj,
                    config.working_directory,
                    config.backend,
                    target,
                    config.net_run_input_data,
                    config.net_run_arguments,
                    config.net_run_backend_extension,
                )
                if config.dump_output:
                    output_dir = self._dump_inference_outputs(
                        netrun_output, config.working_directory
                    )
                    result.output_dir = output_dir

                result.output_data = netrun_output

            self.logger.debug("Inference engine completed successfully!")
            return result
        except Exception as e:
            self.logger.error("Inference engine failed.")
            raise e

    def _create_target(
        self, platform_type: DevicePlatformType, remote_host_details: RemoteHostDetails
    ) -> Target:
        """Get module variant of Target class
        Args:
            platform (Target): Target parameter of type DevicePlatformType
        Returns:
            module_target: Module defined Target object
        """
        if remote_host_details:
            identifier = remote_host_details.identifier
            credentials = remote_host_details.credentials
            return Target(type=platform_type, identifier=identifier, credentials=credentials)
        return Target(type=platform_type)

    def _dump_inference_outputs(
        self,
        inference_outputs: list[net_runner_module.NamedTensorMapping],
        output_path: Path,
    ) -> str:
        """Dump the generated outputs into raw file
        Args:
            inference_outputs: Netrunner output of type list[net_runner_module.NamedTensorMapping]
            output_path: Path to dump the raw outputs
        Returns:
            str: Path to output directory
        """
        output_path = output_path / "Output"
        output_path.mkdir(parents=True, exist_ok=True)
        for idx, output_dict in enumerate(inference_outputs):
            base_dir = output_path / f"Result_{idx}"
            base_dir.mkdir(parents=True, exist_ok=True)
            for output_name, out_tensor in output_dict.items():
                out_tensor.tofile(base_dir / f"{output_name}.raw")
        return output_path

    def _convert(
        self, model: Path, output_path: Path, converter_args: ConverterInputArguments
    ) -> converter_module.ConverterOutputConfig:
        """Perform model conversion
        Args:
            model: Path to the source framework model
            output_path: Path where the converted output model should be saved
            converter_args: ConverterInputArguments object containing arguments for conversion

        Returns:
            ConverterOutputConfig: Object with IR graph and framework of source model

        Raises:
            ConversionFailure: If conversion model fails
        """
        try:
            self.logger.debug("Converting source model to IR")
            if converter_args is None:
                converter_args = converter_module.ConverterInputConfig(input_network=model)
            else:
                converter_args = converter_module.ConverterInputConfig(
                    input_network=model,
                    **converter_args.model_dump(exclude_unset=True),
                )
            self.logger.debug(f"Conversion parameters: {converter_args.model_dump()}")
            converter_args.output_path = str(output_path)
            converter = converter_module.QAIRTConverter()
            converter_output = converter.convert(converter_args)
            self.logger.debug("Completed converting to IR")
        except Exception as exception:
            raise ConversionFailure("Failed to convert the model!") from exception
        return converter_output

    def _optimize(
        self,
        converter_output: converter_module.ConverterOutputConfig,
        output_dlc: Path,
        backend_info: Optional[BackendInfoConfig] = None,
    ) -> str:
        """Perform model optimization
        Args:
            converter_output : ConverterOutputConfig object containing irgraph and framework
            output_dlc : Output path to save the dlc after optimization.
            backend_info: backend specific information required for backend aware optimization.

        Returns:
            Path to the optimized dlc

        Raises:
            OptimizationFailure: If optimization of IRgraph fails
        """
        try:
            self.logger.debug("Optimizing IR graph")
            ir_graph = converter_output.ir_graph
            framework = converter_output.framework
            optimizer_args = optimizer_module.OptimizerInputConfig(
                ir_graph=ir_graph,
                framework=framework,
                dlc_backend_config=converter_output.dlc_backend_config,
                output_dlc=str(output_dlc),
                backend_info=backend_info,
            )
            self.logger.debug(f"Optimization parameters: {optimizer_args.model_dump()}")
            optimizer = optimizer_module.QAIRTOptimizer()
            optimizer_output = optimizer.optimize(optimizer_args)
            optimized_dlc = optimizer_output.dlc_path
            self.logger.debug(f"Optimized graph is saved at {optimized_dlc}")
            self.logger.debug("Completed optimization of IR graph")
        except Exception as exception:
            raise OptimizationFailure("Failed to optimize the model!") from exception
        return optimized_dlc

    def _quantize(
        self,
        input_dlc: Path,
        output_dlc_path: Path,
        quantizer_args: QuantizerInputArguments,
        backend_info: Optional[BackendInfoConfig] = None,
    ) -> str:
        """Perform model quantization
        Args:
            output_dlc_path : File path to be used for saving the Quantized DLC
            input_dlc : Path to the DLC file that needs to be quantized
            quantizer_args: Arguments required for quantization
            backend_info: backend specific information required for backend aware quantization.

        Returns:
            str: Path to the quantized DLC

        Raises:
            QuantizationFailure: If quantization of DLC fails
        """
        try:
            self.logger.debug("Performing quantization")
            quant_args = quantizer_module.QuantizerInputConfig(
                input_dlc=input_dlc,
                **quantizer_args.model_dump(exclude_unset=True),
                output_dlc=str(output_dlc_path),
                backend_info=backend_info,
            )
            self.logger.debug(f"Quanization parameters: {quant_args.model_dump()}")
            quantizer = quantizer_module.QAIRTQuantizer()
            quantizer_output = quantizer.quantize(quant_args)
            quantized_dlc = quantizer_output.dlc_output
            self.logger.debug(f"Quantized graph is saved at {quantized_dlc}")
            self.logger.debug("Completed quantization")
        except Exception as exception:
            raise QuantizationFailure("Failed to quantize the model!") from exception
        return quantized_dlc

    def _generate_binary(
        self,
        model_obj: Model,
        output_dir: Path,
        backend: BackendType,
        context_bin_args: Optional[GenerateConfig] = None,
        context_bin_backend_extension: Optional[Path | dict] = None,
    ) -> Model:
        """Perform offline graph preparation to generate binary

        Args:
            model_obj: Module defined Model object having dlc_path set
            output_dir: output path to export the generated context_bin file
            backend: The backend to use for execution (e.g., CPU, GPU, etc.).
            context_bin_backend_extension: Backend extension configuration file or dictionary.
            context_bin_args: Arguments for context_bin_generation
        Returns:
            Model: Model object with the path to the generated context binary

        Raises:
            ContextBinGenerationFailure: If offline preparation graph fails
        """
        try:
            self.logger.debug("Preparing offline graph")
            input_config = context_bin_gen_module.ContextBinGenArgConfig(
                backend=backend,
                model=model_obj,
                output_dir=output_dir,
                generate_config=context_bin_args,
            )
            self.logger.debug(f"Offline graph prepare parameters: {input_config.model_dump()}")
            if context_bin_backend_extension:
                if isinstance(context_bin_backend_extension, dict):
                    input_config.backend_config_dict = context_bin_backend_extension
                else:
                    input_config.backend_config_file = context_bin_backend_extension

            context_bin_gen = context_bin_gen_module.ContextBinGen(logger=self.logger)
            output_config = context_bin_gen.generate(input_config)
            offline_graph = output_config.context_binary
            self.logger.debug(f"Offline graph saved at {offline_graph}")
            self.logger.debug("Completed offline graph preparation")
        except Exception as exception:
            raise GenerateBinaryFailure("Failed to generate Binaries") from exception
        return offline_graph

    def _execute(
        self,
        model_obj: Model,
        output_dir: Path,
        backend: BackendType,
        target: Target,
        net_run_input_data: net_runner_module.NetRunnerInputData,
        net_run_args: Optional[InferenceConfig] = None,
        net_run_backend_extension: Optional[Path | dict] = None,
    ) -> list[dict[str, np.ndarray]]:
        """Perform model inference
        Args:
            model_obj: Module defined Model object having either context_binary_path or dlc_path
            output_dir: Output path to dump artifacts like, profiling logs, generated during
                        inference
            backend: The backend to use for execution (e.g., CPU, GPU, etc.)
            target: The target platform for execution (e.g.,android)
            net_run_args: Arguments for net-runner
            net_run_input_data: Inputs for inference
            net_run_backend_config_file: Backend extension config for net-runner

        Returns:
            list[dict[str, np.ndarray]]: Inference outputs

        Raises:
            InferenceFailure: if inference fails
        """
        try:
            self.logger.debug(
                f"Running inference for {backend} backend on {target.type.value} target"
            )
            identifier = net_runner_module.InferenceIdentifier(
                model=model_obj, target=target, backend=backend
            )
            net_runner_arg_config = net_runner_module.NetRunnerRunArgConfig(
                identifier=identifier,
                output_dir=str(output_dir),
                backend_config_file=net_run_backend_extension,
                inference_config=net_run_args,
                input_data=net_run_input_data,
            )
            self.logger.debug(
                f"Net-run graph prepare parameters: {net_runner_arg_config.model_dump()}"
            )
            if net_run_backend_extension is not None:
                if isinstance(net_run_backend_extension, dict):
                    net_runner_arg_config.backend_config_dict = net_run_backend_extension
                else:
                    net_runner_arg_config.backend_config_file = net_run_backend_extension

            net_runner = net_runner_module.NetRunner(logger=self.logger)
            output_config = net_runner.run(net_runner_arg_config)
            self.logger.debug("Completed inference")
        except Exception as exception:
            raise ExecutionFailure("Failed to execute the model!") from exception
        return output_config.output_data
