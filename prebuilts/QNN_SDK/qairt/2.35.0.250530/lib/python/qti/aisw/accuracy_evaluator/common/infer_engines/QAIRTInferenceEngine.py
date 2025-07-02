# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import qti.aisw.accuracy_evaluator.common.exceptions as ce

# Imports related to Inference and Context Binary Modules
import qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module as context_bin_gen
import qti.aisw.tools.core.modules.converter.converter_module as converter
import qti.aisw.tools.core.modules.converter.optimizer_module as optimizer
import qti.aisw.tools.core.modules.converter.quantizer_module as quantizer
import qti.aisw.tools.core.modules.net_runner.net_runner_module as net_runner
from pydantic import FilePath
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.qacc.config_definitions import (
    AICBackendExtensions,
    ContextBinParams,
    ConverterParams,
    HTPBackendExtensions,
    HTPMCPBackendExtensions,
    NetRunParams,
    PrecisionType,
    QuantizerParams,
    TargetArchType,
)
from qti.aisw.accuracy_evaluator.qacc.logger import QaccLogger
from qti.aisw.tools.core.modules.api.definitions.common import (
    BackendType,
    Model,
    Target,
)
from qti.aisw.tools.core.modules.converter.common import BackendInfoConfig
from qti.aisw.tools.core.modules.converter.constants import (
    PytorchFrameworkInfo,
    TensorflowFrameworkInfo,
)
from qti.aisw.tools.core.modules.converter.utils import get_framework
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DevicePlatformType,
    RemoteDeviceIdentifier,
)


SUPPORTED_BACKEND_EXTENSIONS = Union[HTPBackendExtensions, HTPMCPBackendExtensions,
                                     AICBackendExtensions]

EVALUATOR_IO_NODE_INFO = Dict[str, List[Any]]


def prepare_logger(name: str, log_file: str, log_level=logging.INFO) -> logging.Logger:
    sub_module_logger = logging.getLogger(name)
    sub_module_logger.handlers = []
    sub_module_logger.parent = []
    fh_file = logging.FileHandler(log_file, mode='w+')
    fh_file.setFormatter(logging.Formatter(QaccLogger.file_format))
    fh_file.setLevel(log_level)
    sub_module_logger.addHandler(fh_file)
    return sub_module_logger


class QAIRTInferenceEngine:

    def __init__(self, model_path: FilePath, inputlistfile: FilePath,
                 inputs_info: EVALUATOR_IO_NODE_INFO, outputs_info: EVALUATOR_IO_NODE_INFO,
                 output_path: str, gen_out_file: str, backend: BackendType = BackendType.CPU,
                 precision: PrecisionType = PrecisionType.FP32,
                 target_arch: TargetArchType = TargetArchType.X86,
                 calibration_file: Optional[FilePath] = None,
                 converter_params: Optional[ConverterParams] = None,
                 quantizer_params: Optional[QuantizerParams] = None,
                 contextbin_params: Optional[ContextBinParams] = None,
                 netrun_params: Optional[NetRunParams] = None,
                 backend_extensions: Optional[SUPPORTED_BACKEND_EXTENSIONS] = None,
                 device_id: Optional[Any] = None):

        self.model_path = model_path
        self.inputlistfile = inputlistfile
        self.inputs_info = inputs_info
        self.outputs_info = outputs_info
        self.inference_schema_work_dir = output_path
        self.gen_out_file = gen_out_file
        self.backend = backend
        self.precision = precision
        self.target_arch = target_arch
        self.calibration_file = calibration_file
        self.converter_params = converter_params
        self.quantizer_params = quantizer_params
        self.contextbin_params = contextbin_params
        self.netrun_params = netrun_params
        self.backend_extensions = backend_extensions
        self.device_id = device_id

        device_identifier = None
        # Android device supported backends: CPU, GPU, HTP
        self.eval_on_android_device = (self.backend
                                       in [BackendType.CPU, BackendType.GPU, BackendType.HTP]
                                       and self.target_arch == TargetArchType.ANDROID)
        if self.eval_on_android_device:
            # if the user has provided device id (hexa decimal format) create a Remote Device with serial id provided
            if self.device_id:
                device_identifier = RemoteDeviceIdentifier(serial_id=self.device_id)
            self.target = Target(type=DevicePlatformType.ANDROID, identifier=device_identifier)
        else:
            # Note: Currently evaluator supports only Android and X68 targets.
            self.target = Target(type=DevicePlatformType.X86_64_LINUX, identifier=device_identifier)

        self.validate()
        self.stage_status = OrderedDict([("converter", None), ("optimizer", None),
                                         ("quantizer", None), ("context-binary-generator", None),
                                         ("qnn-net-run", None)])

    def validate(self):
        # TODO: Add more validation checks
        if (self.precision == PrecisionType.FP32 and self.converter_params
                and self.converter_params.float_bitwidth == 16):
            raise ce.UnsupportedException(
                "The float_bitwidth cannot be set to 16 bits for FP32 PrecisionType")
        if self.backend not in BackendType.offline_preparable_backends() and self.target is None:
            raise ce.UnsupportedException(
                f"Target must be provided if backend provided is not one of {BackendType.offline_preparable_backends()}"
            )

    def convert(
        self, model_path: FilePath, inputs_info: EVALUATOR_IO_NODE_INFO,
        outputs_info: EVALUATOR_IO_NODE_INFO, inference_schema_work_dir: FilePath,
        converter_params: Optional[ConverterParams] = None
    ) -> (bool, converter.ConverterOutputConfig):
        """Performs conversion followed by optimization on the framework
        model using QAIRTConverter &  QAIRTOptimizer Module.
        """
        try:
            _, model_ext = os.path.splitext(model_path)
            model_framework_type = get_framework(model_ext)
            # ONNX & TFlite Case: Input and output info are optional
            input_tensor_config = [
                converter.InputTensorConfig(name=in_name, source_model_input_datatype=in_info[0])
                for in_name, in_info in inputs_info.items()
            ]
            output_tensor_config = None
            if model_framework_type in [
                TensorflowFrameworkInfo.name,
                PytorchFrameworkInfo.name,
            ]:
                # TF Case: input with shape info and output names required
                input_tensor_config = [
                    converter.InputTensorConfig(
                        name=in_name,
                        source_model_input_datatype=in_info[0],
                        desired_input_shape=",".join(str(dim) for dim in in_info[1]),
                    ) for in_name, in_info in inputs_info.items()
                ]
                output_tensor_config = [
                    converter.OutputTensorConfig(name=out_name) for out_name in outputs_info.keys()
                ]

            if converter_params:
                # Note: When using quantization overrides yaml was returned instead of json due to model_serializer
                converter_params_dict = {k: v for k, v in converter_params if k in converter_params.model_fields_set}
                converter_input_arguments = converter.ConverterInputConfig(
                    **converter_params_dict, input_network=model_path,
                    input_tensors=input_tensor_config, output_tensors=output_tensor_config)
            else:
                converter_input_arguments = converter.ConverterInputConfig(
                    input_network=model_path, input_tensors=input_tensor_config,
                    output_tensors=output_tensor_config)

            qacc_file_logger.debug(
                f'Conversion parameters: {converter_input_arguments.model_dump(exclude_unset=True)}'
            )
            log_file = os.path.join(inference_schema_work_dir, 'converter.log')
            converter_logger = prepare_logger(name='converter', log_file=log_file)
            converter_obj = converter.QAIRTConverter(logger=converter_logger)
            converted_output = converter_obj.convert(converter_input_arguments)
        except Exception as e:
            self.stage_status["converter"] = False
            raise ce.QAIRTConverterException(f"failed to convert the given model. Reason: {e}")
        return True, converted_output

    def optimize(self, converted_output: converter.ConverterOutputConfig, output_dlc_path: str,
                 inference_schema_work_dir: Union[str, FilePath]) -> (bool, str):
        try:
            # Create the folder to store the convert + optimized DLC:
            # work_dir/infer/<schema_name>/qnn_ir/
            optimizer_input_arguments = optimizer.OptimizerInputConfig(
                framework=converted_output.framework,
                ir_graph=converted_output.ir_graph,
                output_dlc=output_dlc_path,
                dlc_backend_config=converted_output.dlc_backend_config,
                backend_info=BackendInfoConfig(backend=self.backend.value),
            )
            log_file = os.path.join(inference_schema_work_dir, 'optimizer.log')
            optimizer_logger = prepare_logger(name='optimizer', log_file=log_file)
            optimizer_obj = optimizer.QAIRTOptimizer(logger=optimizer_logger)
            optimizer_output = optimizer_obj.optimize(optimizer_input_arguments)
            # Handle the flow when Quantizer is not done. i.e Converted--> Optimized Model (return)
            # set the converter output to optimizer output
        except Exception as e:
            self.stage_status["optimizer"] = False
            raise ce.QAIRTOptimizerException(f"failed to optimize the given model. Reason: {e}")
        return True, optimizer_output.dlc_path

    def quantize(self, input_dlc: str, calibration_file: FilePath,
                 quantizer_params: QuantizerParams, output_dlc_path: str,
                 inference_schema_work_dir: Union[str, FilePath]) -> (bool, str):
        """Performs quantization on the converted model using QAIRTQuantizer Module."""
        try:
            if quantizer_params:
                # get actual quantizer_params dict instead of dict from model_serializer
                quantizer_params_dict = {}
                for k, v in quantizer_params:
                    if k in quantizer_params.model_fields_set:
                        if str(k) != "input_list":
                            quantizer_params_dict[k] = v
                quantizer_inputs = quantizer.QuantizerInputConfig(
                    **quantizer_params_dict,
                    input_list=calibration_file,
                    input_dlc=input_dlc,
                    output_dlc=output_dlc_path,
                )
            else:
                quantizer_inputs = quantizer.QuantizerInputConfig(
                    input_list=calibration_file,
                    input_dlc=input_dlc,
                    output_dlc=output_dlc_path,
                )
            log_file = os.path.join(inference_schema_work_dir, 'quantizer.log')
            qacc_file_logger.debug(
                f'Quantization parameters: {quantizer_inputs.model_dump(exclude_unset=True)}')
            quantizer_logger = prepare_logger(name='quantizer', log_file=log_file)
            quantizer_obj = quantizer.QAIRTQuantizer(logger=quantizer_logger)
            quantizer_output_config = quantizer_obj.quantize(quantizer_inputs)
        except Exception as e:
            self.stage_status["quantizer"] = False
            raise ce.QAIRTQuantizerException(f"failed to quantize the given model. Reason: {e}")
        return True, quantizer_output_config.dlc_output

    def generate_context(self, model: Model, backend: BackendType, output_dir: str,
                         contextbin_params: Optional[ContextBinParams] = None,
                         target: Target = None, backend_config_file: Optional[FilePath] = None,
                         backend_config_dict: Optional[Dict] = None) -> (bool, Model):
        """Generate Context Binary."""
        try:
            if contextbin_params:
                # contextbin_params from model config is supplied via GenerateConfig
                contextbin_params_dict = contextbin_params.model_dump(
                    exclude_unset=True, exclude=['backend_extensions'])
                generate_config = context_bin_gen.GenerateConfig(**contextbin_params_dict)
            else:
                generate_config = context_bin_gen.GenerateConfig(log_level='error')
            context_bin_generator = context_bin_gen.ContextBinGen()
            qacc_file_logger.debug(
                f'ContextBinGen Config parameters: {generate_config.model_dump(exclude_unset=True)}'
            )

            context_binary_model = context_bin_generator.generate(
                context_bin_gen.ContextBinGenArgConfig(
                    model=model,
                    backend_config_file=backend_config_file,
                    backend_config_dict=backend_config_dict,
                    generate_config=generate_config,
                    backend=backend,
                    target=target,
                    output_dir=output_dir,
                ))
            qacc_file_logger.info("context-binary-generator ran successfully")
        except Exception as e:
            self.stage_status["context-binary-generator"] = False
            raise ce.QnnContextBinaryGeneratorException(
                f"context-binary-generator failed to run successfully. Reason: {e}")
        return True, context_binary_model.context_binary

    def setup_inference(
            self, model: Model, target: Target, backend: BackendType,
            backend_config_file: FilePath = None, backend_config_dict: Optional[Dict] = None,
            netrun_params: Optional[NetRunParams] = None) -> (net_runner.NetRunnerLoadArgConfig):
        # set up the inference identifier along with backend extensions
        try:
            if netrun_params:
                # netrun_params from model config is supplied via InferenceConfig
                netrun_params_dict = netrun_params.model_dump(exclude_unset=True,
                                                              exclude=['backend_extensions'])
                infer_config = net_runner.InferenceConfig(**netrun_params_dict)
            else:
                infer_config = net_runner.InferenceConfig(log_level='error')
                qacc_file_logger.debug(
                    f'InferenceConfig parameters: {infer_config.model_dump(exclude_unset=True)}')

            inference_identifier = net_runner.InferenceIdentifier(model=model, target=target,
                                                                  backend=backend)
            net_runner_load_arg_config = net_runner.NetRunnerLoadArgConfig(
                identifier=inference_identifier, backend_config_file=backend_config_file,
                backend_config_dict=backend_config_dict, inference_config=infer_config)
        except Exception as e:
            self.stage_status['qnn-net-run'] = False
            raise ce.QnnNetRunException(f"Failed to setup the artifacts for inference. Reason: {e}")
        return net_runner_load_arg_config

    def get_inference_handle(
        self, net_runner_module_obj: net_runner.NetRunner,
        net_runner_load_arg_config: net_runner.NetRunnerLoadArgConfig
    ) -> (net_runner.NetRunner, net_runner.NetRunnerLoadArgConfig):
        try:
            net_runner_load_output_config = net_runner_module_obj.load(net_runner_load_arg_config)
        except Exception as e:
            self.stage_status['qnn-net-run'] = False
            raise ce.QnnNetRunException(
                f"Failed to get inference handle for inference. Reason: {e}")

        return net_runner_load_output_config

    def teardown_inference(self, net_runner_module_obj: net_runner.NetRunner,
                           net_runner_unload_arg_config: net_runner.NetRunnerLoadArgConfig) -> bool:
        # unload artifacts from the device
        try:
            net_runner_module_obj.unload(net_runner_unload_arg_config)
        except Exception as e:
            self.stage_status['qnn-net-run'] = False
            raise ce.QnnNetRunException(f"Failed to unload the artifacts from device. Reason: {e}")
        return True

    def infer(
        self,
        net_runner_module_obj: net_runner.NetRunner,
        inference_handle: net_runner.InferencerHandle,
        input_data: net_runner.NetRunnerInputData,
    ) -> (bool, Union[None, List[net_runner.NamedTensorMapping]]):

        try:

            net_runner_run_arg_config = net_runner.NetRunnerRunArgConfig(
                identifier=inference_handle, input_data=input_data)
            inference_outputs = net_runner_module_obj.run(net_runner_run_arg_config)
            # qacc_file_logger.info("qnn-net-run ran successfully")
        except Exception as e:
            self.stage_status["qnn-net-run"] = False
            raise ce.QnnNetRunException(f"qnn-net-run failed to run successfully. Reason: {e}")
        return True, inference_outputs.output_data

    def _dump_inference_outputs(self,
                                inference_outputs: List[net_runner.NamedTensorMapping]) -> None:
        for idx, output_dict in enumerate(inference_outputs):
            base_dir = f"{self.inference_schema_work_dir}/Result_{idx}/"
            os.makedirs(base_dir, exist_ok=True)
            for output_name, out_tensor in output_dict.items():
                out_tensor.tofile(f"{base_dir}/{output_name}.raw")

    def get_output_names(self) -> List[str]:
        """Function to determine the output names based on the outputs_info provided"""
        output_names = []
        # Add logic to use outputs_info from config in the same order
        for output_name, output_node_info in self.outputs_info.items():
            output_names.append(output_name)
        # handle output name change introduced by tf converter
        _, model_ext = os.path.splitext(self.model_path)
        model_framework_type = get_framework(model_ext)
        if model_framework_type == TensorflowFrameworkInfo.name:
            output_names = [f"{output_name}_{idx}" for idx, output_name in enumerate(output_names)]
        return output_names

    def gen_output_file(self) -> None:
        """Function to create the output file containing the paths of the generated outputs"""
        # Create the output file if requested.
        qacc_file_logger.debug(f"Generating output file {self.gen_out_file}")
        # Output file names
        output_names = self.get_output_names()

        num_inputs = sum(1 for line in open(self.inputlistfile))
        with open(self.gen_out_file, "w") as F:
            for i in range(num_inputs):
                paths = []
                for out_name in output_names:
                    _path = os.path.join(self.inference_schema_work_dir,
                                         f"Result_{i}/{out_name}.raw")
                    paths.append(_path)
                F.write(",".join(paths) + "\n")

    def handleNonFloatDatatype(self, node_type: Literal["input", "output"],
                               isInt64to32: bool = True) -> None:
        """Function to convert int64 datatypes to int32 or vice versa
        Args:
            node_type: str:  Node type can be either of 'input' or 'output'
            isInt64to32: bool: True means convert int64 to int32, False means convert int32 to int64
        """
        valid_node_types = ["input", "output"]
        if node_type not in valid_node_types:
            raise ValueError(
                f"node_type supplied is not supported. node_type can be one of {valid_node_types}")
        # Convert int64 inputs to lower (int32) precision
        toConvertInx = []
        if node_type == "input":
            node_info = self.inputs_info
            file_list = [self.inputlistfile]
            if self.calibration_file and (os.path.dirname(self.inputlistfile) != os.path.dirname(
                    self.calibration_file)):
                file_list.append(self.calibration_file)
        elif node_type == "output":
            node_info = self.outputs_info
            file_list = [self.gen_out_file]

        # Check which inputs are int64 and store their indices
        if node_info:
            inx = 0
            isInt64Inp = False
            for in_name, in_info in node_info.items():
                if in_info[0] == "int64":
                    toConvertInx.append(inx)
                    isInt64Inp = True
                inx += 1
            if not isInt64Inp:
                return  # Early exit if no int64 inputs found
        # read the inputlist file and calibration file, select the int64 inputs and cast them
        # to int32
        for file_path in file_list:
            with open(file_path, "r") as F:
                paths = F.readlines()
            src_dt = np.int64
            dst_dt = np.int32
            if not isInt64to32:
                if node_type == "output":
                    src_dt = np.float32  # default netrun output datatype
                else:
                    src_dt = np.int32
                dst_dt = np.int64
            for path_per_line in paths:
                # There could be more than one input
                if node_type == "input":
                    # The delimiter for multi input list is ' '
                    input_paths = path_per_line.split()
                elif node_type == "output":
                    # The delimiter for multi output list is ','
                    input_paths = path_per_line.split(",")
                for i, path in enumerate(input_paths):
                    if i in toConvertInx:
                        input_path = (path.strip().split(":=")[1] if ":=" in path else path.strip())
                        inputSrc = np.fromfile(input_path, src_dt)
                        inputSrc = inputSrc.flatten()
                        inputDst = inputSrc.astype(dst_dt)
                        inputDst.tofile(input_path)

    def execute(self) -> None:
        """Executes the QNN workflow in sequence
        TODO: Separate the compile and execution stages.
        """
        # convert inputs from int64 to int32
        # Note: This is due to limitation from Converter as it converts int64 to int32 by default

        # inference_schema_work_dir = os.path.join(os.getcwd(), self.output_path)
        os.makedirs(self.inference_schema_work_dir, exist_ok=True)

        # Result of Converter + Optimizer:
        # converted dlc model output file path: i.e  <<work_dir>>/infer/<<schema_folder>>/model.dlc
        converter_output_dlc_path = os.path.join(self.inference_schema_work_dir, "model.dlc")

        # Set given AIC device ID with backend extension param
        if self.backend == BackendType.AIC and self.device_id is not None:
            if self.backend_extensions is not None:
                # if the user has provided runtime_device_ids in inference schema
                if self.backend_extensions.runtime_device_ids is not None:
                    qacc_logger.warning(
                        "Evaluator would use runtime_device_ids=[{self.device_id}] instead of the user provided value in the Backend extensions."
                    )
                # Expected format of providing device ids for AIC backend extensions: "runtime_device_ids": [0,1]
                self.backend_extensions.runtime_device_ids = [self.device_id]
            else:
                # When the user doesn't specify backend extension in Inference Schema
                self.backend_extensions = AICBackendExtensions(runtime_device_ids=[self.device_id])

        context_backend_extensions_json, netrun_backend_extensions_json = None, None
        context_backend_extensions_dict, netrun_backend_extensions_dict = None, None
        # Create config JSON files for relevant backends
        if (self.backend in [BackendType.AIC, BackendType.HTP, BackendType.HTP_MCP]
                or self.eval_on_android_device):
            if self.contextbin_params and self.contextbin_params.backend_extensions:
                context_backend_extensions_json = self.contextbin_params.backend_extensions
            elif self.backend_extensions:
                context_backend_extensions_dict = self.backend_extensions.get_context_bin_config_dict()

            if self.netrun_params and self.netrun_params.backend_extensions:
                netrun_backend_extensions_json = self.netrun_params.backend_extensions
            elif self.backend_extensions:
                netrun_backend_extensions_dict = self.backend_extensions.get_netrun_config_dict()

        """Conversion of int64 inputs to int32 is not required for AIC backend as the same can be
        be achieved by Cast operation in Converter. So, disabling it for AIC.
        """
        if self.backend != BackendType.AIC:
            self.handleNonFloatDatatype(node_type="input", isInt64to32=True)

        try:
            # Convert the Model from framework to IR Graph
            self.stage_status["converter"], converter_output = self.convert(
                model_path=self.model_path, inputs_info=self.inputs_info,
                outputs_info=self.outputs_info,
                inference_schema_work_dir=self.inference_schema_work_dir,
                converter_params=self.converter_params)

            # Run the Optimizer Step: IR Graph to DLC (Serialization to DLC)
            self.stage_status["optimizer"], dlc = self.optimize(
                converted_output=converter_output, output_dlc_path=converter_output_dlc_path,
                inference_schema_work_dir=self.inference_schema_work_dir)

            # Perform Quantization only for "quant" Precision and when optimizer stage was successful.
            if self.stage_status["optimizer"] and self.precision in [
                    PrecisionType.QUANT
            ] and self.backend in BackendType.quantizable_backends():
                self.stage_status["quantizer"], dlc = self.quantize(
                    input_dlc=dlc, calibration_file=self.calibration_file,
                    quantizer_params=self.quantizer_params,
                    output_dlc_path=converter_output_dlc_path,
                    inference_schema_work_dir=self.inference_schema_work_dir)

            model_obj = Model(dlc_path=converter_output_dlc_path)
            # Invoke context binary generator only when offline prepare is possible
            if (self.stage_status["optimizer"] and self.stage_status["quantizer"] in [None, True]
                    and self.backend in BackendType.offline_preparable_backends()):
                self.stage_status["context-binary-generator"], model_obj = self.generate_context(
                    model_obj, self.backend, self.inference_schema_work_dir, self.contextbin_params,
                    target=self.target, backend_config_file=context_backend_extensions_json,
                    backend_config_dict=context_backend_extensions_dict)

            if self.stage_status["context-binary-generator"] in [None, True]:
                net_runner_module_obj = net_runner.NetRunner()
                # Setup the inference
                net_runner_load_arg_config = self.setup_inference(
                    model=model_obj, target=self.target, backend=self.backend,
                    backend_config_file=netrun_backend_extensions_json,
                    backend_config_dict=netrun_backend_extensions_dict,
                    netrun_params=self.netrun_params)

                # Get the inference Handle
                net_runner_load_output_config = self.get_inference_handle(
                    net_runner_module_obj, net_runner_load_arg_config)

                # Perform Inference on the model
                self.stage_status["qnn-net-run"], inference_outputs = self.infer(
                    net_runner_module_obj=net_runner_module_obj,
                    inference_handle=net_runner_load_output_config.handle,
                    input_data=self.inputlistfile)

                # unload artifacts from the device
                unload_arg_config = net_runner.NetRunnerUnloadArgConfig(handle=net_runner_load_output_config.handle)
                self.teardown_inference(net_runner_module_obj, unload_arg_config)
                # Dump the outputs for the next stages of evaluation (postproc & metric)
                if self.stage_status["qnn-net-run"]:
                    self._dump_inference_outputs(inference_outputs)
        except Exception as e:
            raise e

        """Revert back the converted inputs from int32 to int64.
        However, this is not required for AIC backend as the same can be
        be achieved by Cast operation in Converter. So, disabling it for AIC.
        """
        if self.backend != BackendType.AIC:
            self.handleNonFloatDatatype(node_type="input", isInt64to32=False)

        # Generate the infer output file for verifiers and post inference evaluation
        self.gen_output_file()
        # Handle Outputs which are non float datatype
        self.handleNonFloatDatatype(node_type="output", isInt64to32=False)
