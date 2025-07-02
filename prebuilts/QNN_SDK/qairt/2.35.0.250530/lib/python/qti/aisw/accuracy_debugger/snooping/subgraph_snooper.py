# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import logging
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    InputSample,
    QuantizerInputArguments,
    RemoteHostDetails,
)
from qti.aisw.accuracy_debugger.encodings_converter.qairt_encodings_converter import (
    QairtEncodingsConverter,
)
from qti.aisw.accuracy_debugger.framework_runner.framework_factory import get_framework_instance
from qti.aisw.accuracy_debugger.snooping.snooper import Snooper
from qti.aisw.accuracy_debugger.snooping.snooper_utils import ActivationInfo, ActivationStatus
from qti.aisw.accuracy_debugger.utils.constants import MATH_INVARIANT_OPS
from qti.aisw.accuracy_debugger.utils.exceptions import (
    ConversionFailure,
    ExecutionFailure,
    GenerateBinaryFailure,
    OptimizationFailure,
    QuantizationFailure,
    VerificationError,
)
from qti.aisw.accuracy_debugger.utils.file_utils import dump_csv, dump_json
from qti.aisw.accuracy_debugger.utils.graph_utils import (
    get_common_parent_activations,
    get_subgraph,
    get_supergroup_activations,
    get_topological_order,
)
from qti.aisw.tools.core.modules.api.definitions.common import BackendType
from qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module import GenerateConfig
from qti.aisw.tools.core.modules.net_runner.net_runner_module import InferenceConfig
from qti.aisw.tools.core.utilities.comparators.comparator import Comparator
from qti.aisw.tools.core.utilities.comparators.mse import MSEComparator
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.verifier.layout import TensorLayout
from qti.aisw.tools.core.utilities.verifier.verifier import Verifier


class SubgraphSnooper(Snooper, ABC):
    """SubgraphSnooper class for snooping subgraphs."""

    def __init__(self, model: Path, name: str, logger: logging.Logger, is_cumulative: bool):
        """Initializes the SubgraphSnooper.

        Args:
            model (Path): Path to framework model.
            name (str): The name of the snooper algorithm
            logger (logging.Logger): A python logger instance
            is_cumulative (bool): Specifies whether the algorithm is layerwise or cumulative.
        """
        super().__init__(model, name, logger)
        self.is_cumulative = is_cumulative

    def _get_all_subgraphs(
        self,
        framework_activation_op_map: dict,
        debug_graph_activations: list,
        target_activation_op_map: dict,
        supergroup_activations: set,
        resolved_target_activations: dict,
        target_activation_info: dict[str, ActivationInfo],
        framework_activation_info: dict[str, ActivationInfo],
        verifier_scores: dict,
        comparators: list,
        data_frame: dict,
        output_tensor: list,
        qairt_encodings_converter: QairtEncodingsConverter,
        output_dir: Path,
    ) -> tuple[dict, dict]:
        """Each op in the target graph represents a subgraph post optimizations.
        Those optimizations can be backend aware depends upon such implementations
        at the qairt-converter level.
        Maps each op in the target graph to the framework subgraph, subsequently
        finds out the subgraph quantization overrides.

        Args:
            framework_activation_op_map: Framework activation to framework op map.
            debug_graph_activations: Activations from the debug graph.
            target_activation_op_map: Target activations to Target op map.
            supergroup_activations: Activations of supergroups.
            resolved_target_activations: Framework name to target activation name.
            target_activation_info: Target Activation information.
            verifier_scores: Comparator scores from verifier.
            comparators: List of comparators.
            framework_activation_info: Framework activation info.
            data_frame: csv data frame.
            output_tensor: Output tensor name.
            qairt_encodings_converter: QairtEncodingsConverter object.
            output_dir: Path to output directory.

        Returns:
            tuple[dict, dict]: All Subgraph, Data going into the csv report
        """
        subgraphs = {}
        for activation in debug_graph_activations:
            target_op = target_activation_op_map[activation]
            self._activation_status[activation] = ActivationStatus(activation)

            # Set subgraph outputs based on snooper type
            subgraph_outputs = output_tensor if self.is_cumulative else target_op.outputs

            input_tensor_names = set()
            for input_name in target_op.inputs:
                # some input_name can be param
                if input_name in target_activation_op_map:
                    common_parent_activations = get_common_parent_activations(
                        input_name,
                        target_activation_op_map,
                        framework_activation_op_map,
                        supergroup_activations,
                    )
                    input_tensor_names.update(common_parent_activations)

            if input_tensor_names:
                self._logger.debug("+" * 71)
                self._logger.debug(f"Subgraph Inputs: {input_tensor_names}")
                self._logger.debug(f"Subgraph Outputs: {subgraph_outputs}")

                input_tensor_names = set(list(input_tensor_names))

                self._logger.debug("Getting target subgraph")
                target_subgraph_activations, _, _ = get_subgraph(
                    input_tensor_names, subgraph_outputs, target_activation_op_map
                )

                self._logger.debug("Getting framework subgraph")
                framework_subgraph_activations, _, _ = get_subgraph(
                    input_tensor_names, subgraph_outputs, framework_activation_op_map
                )
                if framework_subgraph_activations is None:
                    framework_subgraph_activations = [
                        "Due to converter optimizations, framework subgraph is not found"
                    ]

                subgraphs[activation] = {
                    "Inputs": ",".join(input_tensor_names),
                    "Outputs": ",".join(subgraph_outputs),
                    "Target Tensors": ",".join(target_subgraph_activations),
                    "Framework Tensors": ",".join(framework_subgraph_activations),
                    "layer_type": target_op.op_type,
                }

                skip, reason = self._should_be_skipped(
                    target_subgraph_activations, activation, target_activation_op_map
                )

                if skip:
                    status_msg = (
                        f"Skipping target subgraph {target_subgraph_activations} "
                        f"as it is of type {reason}"
                    )
                    self._activation_status[activation].set_status(
                        ActivationStatus.SKIP, status_msg
                    )
                    self._logger.info(status_msg)
                    data_frame = self._build_data_frame_for_subgraph(
                        activation,
                        resolved_target_activations,
                        target_activation_info,
                        verifier_scores,
                        subgraphs[activation]["layer_type"],
                        comparators,
                        framework_activation_info,
                        data_frame,
                        output_tensor,
                    )
                    subgraphs[activation]["status"] = ActivationStatus.SKIP
                    subgraphs[activation]["status_msg"] = status_msg
                    subgraphs[activation]["override_file_path"] = ""
                else:
                    # handle cases like split node
                    # subgraph_last_op ---> (out1, out2)
                    # add all subgraph outputs to the subgraph
                    target_subgraph_activations.update(subgraph_outputs)

                    self._logger.debug(
                        f"Subgraph {target_subgraph_activations} starts "
                        f"with {input_tensor_names} and ends with {subgraph_outputs}"
                    )
                    try:
                        self._logger.debug("Creating subgraph override")
                        subgraph_override_file_path = self._create_subgraph_quantization_override(
                            target_subgraph_activations,
                            target_op.outputs,
                            supergroup_activations,
                            qairt_encodings_converter,
                            output_dir,
                        )
                        subgraphs[activation]["override_file_path"] = str(
                            subgraph_override_file_path
                        )
                    except Exception:
                        self._logger.error(
                            f"Failed to create quantization overrides for "
                            f"{target_subgraph_activations}"
                        )
                        status_msg = ""
                        subgraphs[activation]["override_file_path"] = ""
                        subgraphs[activation]["status"] = (
                            ActivationStatus.CUSTOM_OVERRIDE_GENERATION_FAILURE
                        )
                        subgraphs[activation]["status_msg"] = status_msg
                        self._activation_status[activation].set_status(
                            ActivationStatus.CUSTOM_OVERRIDE_GENERATION_FAILURE, status_msg
                        )
                        data_frame = self._build_data_frame_for_subgraph(
                            activation,
                            resolved_target_activations,
                            target_activation_info,
                            verifier_scores,
                            subgraphs[activation]["layer_type"],
                            comparators,
                            framework_activation_info,
                            data_frame,
                            output_tensor,
                        )

                self._logger.debug("+" * 70)

        return subgraphs, data_frame

    def run(
        self,
        input_tensors: list[InputSample] | dict[str, NDArray],
        backend: BackendType,
        platform: DevicePlatformType,
        converter_args: ConverterInputArguments,
        quantizer_args: QuantizerInputArguments,
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
        """Run method used by layerwise and cumulative layerwise snooping classes.

        Args:
            input_tensors: List of InputSample or dictionary of tensor name to numpy array.
            backend: Type of backend.
            platform: Platform of target device (android, wos, x86_64_linux, etc.)
            converter_args: Input arguments required by the converter module.
            quantizer_args: Input arguments required by the quantizer module.
            context_bin_args: Input arguments required by the context_bin_gen module.
            context_bin_backend_extension: Backend extension config file for context binary generator.
            offline_prepare: Boolean to indicate offline prepare of graph.
            net_runner_args: Input arguments required by the netrunner module.
            net_run_backend_extension: Backend extension config for net-runner.
            remote_host_details: Details of remote host.
            soc_model : Name of SOC model on target device.
            comparators: List of comparators to use in verification stage.
            working_directory: Path to the directory to store the results.
            golden_reference_path: Path to directory containing golden reference outputs.
                                   Defaults to None.
            retain_compilation_artifacts: Boolean flag to retain convert, quantize and context-bin-gen
                              artifacts. Default is set to False.
            dump_output_tensors: Boolean to indicate whether to dump output tensors.

        Returns:
            Path: Path to CSV report file.

        Raises:
            Exception: If inference engine fails to generate qairt encodings (or)
                Fails to create QairtEncodingsConverter Object
        """
        self._logger.info(f"Started {self._name} snooping")

        if not working_directory:
            working_directory = Path.cwd() / "working_directory"
            working_directory.mkdir(exist_ok=True)

        output_dir = (
            Path(working_directory)
            / f"{self._name}_snooping"
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_filename = "cumulative.csv" if self.is_cumulative else "layerwise.csv"
        csv_path = output_dir / csv_filename

        verifier_scores = {}
        if converter_args and converter_args.output_tensors:
            output_tensor = [output_tensor.name for output_tensor in converter_args.output_tensors]
        else:
            output_tensor = self._get_output_tensor_names()

        input_sample = self._load_input_tensors(input_tensors)

        all_subgraphs = {}

        # Execute the framework diagnosis
        if golden_reference_path:
            self._logger.info(f"Loading reference data from given {golden_reference_path}")
            golden_reference_output = self._load_data_from_directory(golden_reference_path)
        else:
            self._logger.info(f"Generating reference data for {self.model}")
            golden_reference_output = self._generate_reference_data(
                self.model, input_sample, dump_output_tensors, output_dir
            )

        comparator_names = [comparator.name for comparator in comparators]
        data_frame = self._initialize_data_frame(comparator_names, output_tensor)

        # Generate quantization encodings
        quantizer_args.dump_encoding_json = True
        inference_output_directory = output_dir / "inference_engine"
        inference_output_directory.mkdir(exist_ok=True)
        try:
            self._logger.info("Generate quantization encodings")
            inference_output_config = self._execute_inference(
                model=self.model,
                converter_args=converter_args,
                quantizer_args=quantizer_args,
                backend=backend,
                working_directory=inference_output_directory,
                soc_model=soc_model,
            )
        except Exception as exception:
            self._logger.error(f"Failed to generate encodings. Reason: {exception}")
            raise exception

        quantized_dlc_path = inference_output_config.quantizer_dlc
        quantization_overrides = str(quantized_dlc_path.with_suffix("")) + "_encoding.json"
        qairt_encodings_converter = self._get_encodings_converter(
            output_dir, self.model, quantized_dlc_path, quantization_overrides
        )
        framework_activation_op_map = qairt_encodings_converter.get_framework_activation_op_map()
        target_activation_op_map = qairt_encodings_converter.get_target_activation_op_map()
        resolved_target_activations = qairt_encodings_converter.get_resolved_target_activation()

        supergroup_activations = get_supergroup_activations(
            framework_activation_op_map, target_activation_op_map
        )
        all_subgraphs["ignore_activations"] = ",".join(supergroup_activations)

        self._logger.info("Generating debug subgraph")
        debug_graph_activations, debug_graph_input_names, debug_graph_output_names = (
            self._get_debug_graph(
                framework_activation_op_map, target_activation_op_map, supergroup_activations
            )
        )

        all_subgraphs["debug_graph_input_tensors"] = ",".join(debug_graph_input_names)
        all_subgraphs["debug_graph_output_tensors"] = ",".join(debug_graph_output_names)
        all_subgraphs["debug_graph_activations"] = ",".join(debug_graph_activations)
        all_subgraphs["subgraphs"] = {}

        # Get Activation info for framework and target as well as layout data
        framework_activation_info, target_activation_info, layout_info = self._get_profile_info(
            golden_reference_output, quantized_dlc_path, output_dir
        )

        # Generate subgraphs
        self._logger.info("Generating subgraphs from debug graph")
        all_subgraphs["subgraphs"], data_frame = self._get_all_subgraphs(
            framework_activation_op_map,
            debug_graph_activations,
            target_activation_op_map,
            supergroup_activations,
            resolved_target_activations,
            target_activation_info,
            framework_activation_info,
            verifier_scores,
            comparators,
            data_frame,
            output_tensor,
            qairt_encodings_converter,
            output_dir,
        )

        # Dump all_subgraphs for user to check the identified subgraphs
        all_subgraphs_path = output_dir / "all_subgraphs.json"
        dump_json(all_subgraphs, all_subgraphs_path)

        # Execute the generated subgraphs
        self._logger.info("Executing generated subgraphs")
        csv_path = self._execute_all_sub_graphs(
            output_dir=inference_output_directory,
            all_subgraphs=all_subgraphs,
            resolved_target_activations=resolved_target_activations,
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
            soc_model=soc_model,
            comparators=comparators,
            layout_info=layout_info,
            golden_reference_output=golden_reference_output,
            data_frame=data_frame,
            output_tensor=output_tensor,
            target_activation_info=target_activation_info,
            framework_activation_info=framework_activation_info,
            verifier_scores=verifier_scores,
            csv_path=csv_path,
            retain_compilation_artifacts=retain_compilation_artifacts,
            dump_output_tensors=dump_output_tensors,
        )

        # Re-dump the all_subgraphs to update the 'status' field.
        dump_json(all_subgraphs, all_subgraphs_path)

        self._plot_comparator_scores(csv_path, comparators, output_tensor, output_dir)

        return csv_path

    def _plot_comparator_scores(
        self,
        csv_path: Path,
        comparators: list[Comparator],
        output_tensor: list[str],
        output_dir: Path,
    ) -> None:
        """Plots and dumps scores of all verifiers/comparators for both current layer of each
        subgraph and actual original outputs of the model.

        Args:
            csv_path: Path to the CSV snooper report
            comparators: List of comparators to use in verification stage.,
            output_tensor: Output tensors of the framework model
            output_dir: Output directory path
        """
        comparator_columns = []
        for comparator in comparators:
            comparator_columns.append(f"{comparator.name}(current_layer)")
            for output_name in output_tensor:
                comparator_columns.append(f"{comparator.name}({output_name})")

        self._plot_graphs(
            csv_path=csv_path,
            layer_names_column="Source Name",
            comparator_columns=comparator_columns,
            output_dir=output_dir,
        )

    def _get_encodings_converter(
        self,
        output_dir: Path,
        model_path: Path,
        quantized_dlc_path: Path,
        quantization_overrides: str,
    ) -> QairtEncodingsConverter:
        """Method consumes the model, quantized dlc and quantization overrides
        to return the qairt encodings converter object.

        Args:
            output_dir (Path): Output directory.
            model_path (Path): Path to framework model.
            quantized_dlc_path (Path): Path to quantized DLC.
            quantization_overrides (str): Path to quantization override.

        Returns:
            QairtEncodingsConverter: QAIRT Encoding converter object

        Raises:
            Exception: If QairtEncodingsConverter object creation fails
        """
        working_dir = output_dir / "encodings_converter"
        try:
            qairt_encodings_converter = QairtEncodingsConverter(
                str(model_path),
                str(quantized_dlc_path),
                quantization_overrides,
                str(working_dir),
                self._logger,
            )
            converted_encodings = qairt_encodings_converter.create_subgraph_encodings()
        except Exception as exception:
            raise Exception(
                f"QairtEncodingsConverter object creation failed with error: {exception}"
            )
        converted_encodings_file_path = working_dir / "converted_encodings.json"
        with open(converted_encodings_file_path, "w") as file:
            json.dump(converted_encodings, file, indent=4)

        return qairt_encodings_converter

    def _get_debug_graph(
        self,
        framework_activation_op_map: dict,
        target_activation_op_map: dict,
        supergroup_activations: set,
    ) -> tuple[list, set, set]:
        """Generate subgraph for debugging

        Args:
            framework_activation_op_map: Framework activation to framework op map.
            target_activation_op_map: Target activations to Target op map.
            supergroup_activations: Activations of supergroups.

        Returns:
            tuple[list, set, set]: Activations, input names and output names of debug graph

        Raises:
            Exception: if debug subgraph turns out to be empty
        """
        debug_framework_graph_inputs = set()
        debug_framework_graph_outputs = set()
        for activation_name, framework_op in framework_activation_op_map.items():
            if framework_op.op_type == "input":
                debug_framework_graph_inputs.update([activation_name])
            if not framework_op.children_ops:
                debug_framework_graph_outputs.update([activation_name])

        debug_graph_input_names = set()
        for input_name in debug_framework_graph_inputs:
            partial_inputs = get_common_parent_activations(
                input_name,
                framework_activation_op_map,
                target_activation_op_map,
                supergroup_activations,
            )
            debug_graph_input_names.update(partial_inputs)

        debug_graph_output_names = set()
        for output_name in debug_framework_graph_outputs:
            partial_outputs = get_common_parent_activations(
                output_name,
                framework_activation_op_map,
                target_activation_op_map,
                supergroup_activations,
            )
            debug_graph_output_names.update(partial_outputs)

        target_activations = set(target_activation_op_map.keys()) - debug_graph_input_names
        # filter out all the target activations which are not part of framework graph like
        # convert ops/ extra target ops added as no point debugging them bcz they do not
        # have corresponding framework ops.
        target_activations = target_activations.intersection(framework_activation_op_map.keys())
        # filter out all intermediate supergroup activations as they should not be debugged
        debug_graph_activations = target_activations - supergroup_activations

        # Topological sort the target activations
        target_topological_sort = get_topological_order(target_activation_op_map)
        debug_graph_topological_activations = [
            activation
            for activation in target_topological_sort
            if activation in debug_graph_activations
        ]

        if not debug_graph_activations:
            raise RuntimeError("Failed to generate debug graph activations")

        self._logger.debug(
            f"Debug Graph: {debug_graph_topological_activations} with Inputs: "
            f"{debug_graph_input_names} and Outputs: {debug_graph_output_names}"
        )

        return (
            debug_graph_topological_activations,
            debug_graph_input_names,
            debug_graph_output_names,
        )

    def _should_be_skipped(
        self, subgraph_activations: set, subgraph_output_name: str, target_activation_op_map: dict
    ) -> tuple[bool, str]:
        """Determines whether a subgraph should be skipped.

        Args:
            subgraph_activations (set): Set of subgraph activations
            subgraph_output_name (str): Last output of the subgraph
            target_activation_op_map (dict): Mapping of target activation names to op.

        Returns:
            tuple[bool, str]: Indicates whether the node should be skipped and the reason.
        """
        target_op = target_activation_op_map[subgraph_output_name]
        target_op_type = target_op.op_type.lower()

        self._logger.debug(f"Activation: {subgraph_output_name},  Target op_type: {target_op_type}")

        # Skip if all ops in the target subgraph are MATH_INVARIANT ops
        subgraph_op_types = set()
        for activation_name in subgraph_activations:
            target_op = target_activation_op_map[activation_name]
            target_op_type = target_op.op_type.lower()
            subgraph_op_types.update([target_op_type])

        # Skip if all ops in the subgraph are classified as MATH_INVARIANT ops
        if subgraph_op_types.issubset(MATH_INVARIANT_OPS):
            return True, "MATH_INVARIANT"

        return False, ""

    def _build_data_frame_for_subgraph(
        self,
        framework_activation_name: str,
        resolved_target_activations: dict,
        target_activation_info: dict[str, ActivationInfo],
        verifier_scores: dict,
        layer_type: str,
        comparators: list[Comparator],
        framework_activation_info: dict[str, ActivationInfo],
        data_frame: dict,
        output_tensor: list[str],
    ) -> dict:
        """Build the result dataframe for the given activation in the framework graph

        Args:
            framework_activation_name: activation to build the dataframe
            resolved_target_activations: mapping between framework and target activations
            target_activation_info: Activation information of target output.
            verifier_scores: Comparator scores from verifier.
            layer_type: Layer type of the activation
            comparators: list of comparators to be used in verification stage
            framework_activation_info: Framework activations information
            data_frame: dataframe to be dumped in csv report
            output_tensor: output tensors of the framework model

        Returns:
            dict: dataframe to be dumped in the csv report
        """
        comparator_names = [comparator.name for comparator in comparators]
        sanitized_framework_activation = Helper.transform_node_names(framework_activation_name)
        framework_activation_info = framework_activation_info[sanitized_framework_activation]

        status = self._activation_status[framework_activation_name].get_status()
        info = self._activation_status[framework_activation_name].get_msg()

        data_frame["Source Name"].append(framework_activation_name)
        data_frame["STATUS"].append(status)
        data_frame["INFO"].append(info)
        data_frame["Layer Type"].append(layer_type)
        data_frame["Framework Shape"].append(framework_activation_info.shape)
        data_frame["Framework(Min, Max, Median)"].append(framework_activation_info.distribution)

        if status == ActivationStatus.SUCCESS:
            resolved_target_activation = resolved_target_activations[framework_activation_name]
            sanitized_target_activation = Helper.transform_node_names(resolved_target_activation)
            target_activation_info = target_activation_info[sanitized_target_activation]
            data_frame["Target Name"].append(resolved_target_activation)
            data_frame["Target Shape"].append(target_activation_info.shape)
            data_frame["Target(Min, Max, Median)"].append(target_activation_info.distribution)

            for comp in comparator_names:
                data_frame[f"{comp}(current_layer)"].append(
                    verifier_scores[framework_activation_name]["self"][comp]
                )
                for original_output in output_tensor:
                    data_frame[f"{comp}({original_output})"].append(
                        verifier_scores[framework_activation_name]["original_outputs"][
                            original_output
                        ][comp]
                    )
        else:
            data_frame["Target Name"].append("-")
            data_frame["Target Shape"].append("-")
            data_frame["Target(Min, Max, Median)"].append("-")

            for comp in comparator_names:
                data_frame[f"{comp}(current_layer)"].append("nan")
                for original_output in output_tensor:
                    data_frame[f"{comp}({original_output})"].append("nan")

        return data_frame

    def _execute_all_sub_graphs(
        self,
        output_dir: Path,
        all_subgraphs: dict,
        resolved_target_activations: dict,
        converter_args: ConverterInputArguments,
        quantizer_args: QuantizerInputArguments,
        context_bin_args: GenerateConfig,
        context_bin_backend_extension: Path | dict,
        offline_prepare: bool,
        net_runner_args: InferenceConfig,
        net_run_backend_extension: Path | dict,
        input_sample: dict,
        backend: BackendType,
        platform: DevicePlatformType,
        remote_host_details: RemoteHostDetails,
        soc_model: str,
        comparators: list[Comparator],
        layout_info: dict,
        golden_reference_output: dict,
        data_frame: dict,
        output_tensor: list[str],
        target_activation_info: dict[str, ActivationInfo],
        framework_activation_info: dict[str, ActivationInfo],
        verifier_scores: dict,
        csv_path: Path,
        retain_compilation_artifacts: bool = False,
        dump_output_tensors: bool = False,
    ) -> Path:
        """Executes all subgraph on QAIRT inference engine

        Args:
            output_dir: Output directory path
            all_subgraphs: Information of all subgraphs.
            resolved_target_activations: Output target activation.
            converter_args: Input arguments required by the converter module.
            quantizer_args: Input arguments required by the quantizer module.
            context_bin_args: Input arguments required by the context_bin_gen module.
            context_bin_backend_extension: Backend extension config file or dictionary for context
                                           binary generator.
            offline_prepare: Boolean to indicate offline prepare of graph.
            net_runner_args: Input arguments required by the netrunner module.
            net_run_backend_extension: Backend extension config file or dictionary for net-runner
            input_sample: Input to netrunner module
            backend: Type of backend
            platform: Platform of target device (android, wos, x86_64_linux, etc.)
            remote_host_details: Details of remote host
            soc_model: Name of SOC model on target device.
            comparators: List of comparators to use in verification stage.
            layout_info: Layout information used for verification
            golden_reference_output: Golden reference output data
            data_frame: Dataframe with information on subgraphs
            output_tensor: Output tensors of the framework model
            target_activation_info: Target Activation information.
            framework_activation_info: Framework activation info.
            verifier_scores: Comparator scores from verifier.
            csv_path: Path to CSV snooper report
            retain_compilation_artifacts: Flag to retain compilation artifacts. Default is set to
                                          False
            dump_output_tensors: Boolean to indicate whether to dump output tensors.

        Returns:
            Path: Path to the given CSV snooper report
        """
        for framework_activation_name in all_subgraphs["subgraphs"]:
            subgraph_override_file_path = all_subgraphs["subgraphs"][framework_activation_name][
                "override_file_path"
            ]

            if (
                self._activation_status[framework_activation_name].get_status()
                == ActivationStatus.INITIALIZED
            ):
                sanitized_framework_activation_name = Helper.transform_node_names(
                    framework_activation_name
                )
                working_dir = output_dir / sanitized_framework_activation_name
                working_dir.mkdir(exist_ok=True)

                resolved_target_activation = resolved_target_activations[framework_activation_name]
                output_tensors = [resolved_target_activation] + output_tensor
                sanitized_output_tensors = list(map(Helper.transform_node_names, output_tensors))

                converter_args.quantization_overrides = subgraph_override_file_path

                quantizer_args_dict = quantizer_args.model_dump()
                quantizer_args_dict["input_list"] = None
                quantizer_args_dict["float_fallback"] = True
                quantizer_args = QuantizerInputArguments(**quantizer_args_dict)

                # Set output tensors for offline preparation
                if (
                    offline_prepare is False
                    or backend not in BackendType.offline_preparable_backends()
                ):
                    # TODO use set_output_tensors option instead of debug when set_output_tensors
                    # option enabled.
                    if net_runner_args:
                        net_runner_args.debug = True
                    else:
                        net_runner_args = InferenceConfig(
                            debug=True,
                        )

                else:
                    if context_bin_args:
                        context_bin_args.set_output_tensors = sanitized_output_tensors
                    else:
                        context_bin_args = GenerateConfig(
                            set_output_tensors=sanitized_output_tensors
                        )

                # Execute inference
                exception_message = None
                inference_output_directory = working_dir / "inference_engine"
                inference_output_directory.mkdir(exist_ok=True)
                try:
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
                    target_output_data = inference_output_config.output_data[0]
                except ConversionFailure as exception:
                    exception_message = f"Conversion failure. {exception}"
                    status = ActivationStatus.CONVERTER_FAILURE
                except OptimizationFailure as exception:
                    exception_message = f"Optimization failure. {exception}"
                    status = ActivationStatus.OPTIMIZER_FAILURE
                except QuantizationFailure as exception:
                    exception_message = f"Quantization failure. {exception}"
                    status = ActivationStatus.QUANTIZER_FAILURE
                except GenerateBinaryFailure as exception:
                    exception_message = f"Offline graph preparation failure. {exception}"
                    status = ActivationStatus.OFFLINE_PREPARE_FAILURE
                except ExecutionFailure as exception:
                    exception_message = f"Execution failure. {exception}"
                    status = ActivationStatus.NET_RUN_FAILURE
                except Exception as exception:
                    exception_message = f"Inference Engine failure. {exception}"
                    status = ActivationStatus.INFERENCE_FAILURE
                else:
                    status = ActivationStatus.INFERENCE_DONE
                    try:
                        verifier_scores, target_activation_info = (
                            self._run_verification_for_subgraph(
                                framework_activation_name,
                                golden_reference_output,
                                target_output_data,
                                verifier_scores,
                                resolved_target_activations,
                                comparators,
                                layout_info,
                                target_activation_info,
                                output_tensor,
                            )
                        )
                        if not retain_compilation_artifacts:
                            self._logger.info(
                                f"Cleaning up artifacts for subgraph {framework_activation_name}:\n"
                                f"Converter DLC: {inference_output_config.converter_dlc}\n"
                                f"Quantizer DLC: {inference_output_config.quantizer_dlc}\n"
                                f"Offline Graph: {inference_output_config.offline_graph}"
                            )
                            inference_output_config.cleanup_artifacts()
                        status = ActivationStatus.SUCCESS
                    except VerificationError as exception:
                        exception_message = f"Verification failure. {exception}"
                        status = ActivationStatus.VERIFICATION_FAILURE
                finally:
                    if exception_message:
                        self._logger.error(
                            f"Subgraph execution failed. Reason: {exception_message}"
                        )

                    all_subgraphs["subgraphs"][framework_activation_name]["status"] = status
                    all_subgraphs["subgraphs"][framework_activation_name]["status_msg"] = ""
                    self._activation_status[framework_activation_name].set_status(status, "")

                    data_frame = self._build_data_frame_for_subgraph(
                        framework_activation_name,
                        resolved_target_activations,
                        target_activation_info,
                        verifier_scores,
                        all_subgraphs["subgraphs"][framework_activation_name]["layer_type"],
                        comparators,
                        framework_activation_info,
                        data_frame,
                        output_tensor,
                    )

                    # Dump snooper data into CSV
                    dump_csv(data_frame, csv_path)

                    self._logger.info(
                        f"STATUS for activation {framework_activation_name}: "
                        f"{self._activation_status[framework_activation_name].get_status()}"
                    )
            else:
                # subgraph is either SKIP or OVERRIDE_FAILURE, we have already logged the info
                continue

        return csv_path

    # TODO: Handle memory efficient code (file deletion)

    def _initialize_data_frame(self, comparator_names: list, output_tensor: list) -> dict:
        """Initialize result csv related data structures.

        Args:
            comparator_names (list): List of comparator names
            output_tensor (list[str]): List of output tensor names.

        Returns:
            dict: Dataframe for storing data per activation.
        """
        columns = [
            "Source Name",
            "Target Name",
            "STATUS",
            "Layer Type",
            "Framework Shape",
            "Target Shape",
            "Framework(Min, Max, Median)",
            "Target(Min, Max, Median)",
        ]

        for comp in comparator_names:
            columns.append(f"{comp}(current_layer)")
            for original_output in output_tensor:
                columns.append(f"{comp}({original_output})")
        columns.append("INFO")

        data_frame = {col: [] for col in columns}

        return data_frame

    def _get_profile_info(
        self, golden_output: dict[str, NDArray], dlc_path: Path, output_dir: Path
    ) -> tuple[dict[str, ActivationInfo], dict[str, ActivationInfo], dict]:
        """Reads profile_info.json from framework runner and layout_data to
        populate framework and target related tensor informations.

        Args:
            golden_output (dict[str, NDArray]): Dictionary of output name to numpy array
            dlc_path (Path): Path to DLC file.
            output_dir (Path): Path to output directory.

        Returns:
            tuple[dict[str, ActivationInfo], dict[str, ActivationInfo], dict]: Returns
                the framework activation, target activation and layout info.
        """
        # Create Profile Info
        profile_info = {}
        for output_tensor_name, data in golden_output.items():
            santized_tensor_name = Helper.transform_node_names(output_tensor_name)

            if not data.size or data.dtype == bool:
                if data.size == 0:
                    profile_info[santized_tensor_name] = (
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    )
                else:
                    profile_info[santized_tensor_name] = (
                        str(data.dtype),
                        data.shape,
                        data.tolist(),
                        data.tolist(),
                        data.tolist(),
                    )
            else:
                profile_info[santized_tensor_name] = (
                    str(data.dtype),
                    data.shape,
                    str(round(np.min(data), 3)),
                    str(round(np.max(data), 3)),
                    str(round(np.median(data), 3)),
                )

        framework_activation_info = {}
        for sanitized_activation_name, value in profile_info.items():
            activation_info = ActivationInfo(
                dtype=value[0], shape=value[1], distribution=tuple(value[2:])
            )
            framework_activation_info[sanitized_activation_name] = activation_info

        tensor_layout = TensorLayout()
        layout_info = tensor_layout.get_layout_info_from_dlc(dlc_path)

        target_activation_info = {}
        for sanitized_activation_name, value in layout_info.items():
            activation_info = ActivationInfo(dtype=None, shape=value["dims"], distribution=None)
            target_activation_info[sanitized_activation_name] = activation_info

        return framework_activation_info, target_activation_info, layout_info

    def _run_verification_for_subgraph(
        self,
        framework_activation_name: str,
        golden_reference_output: dict,
        target_output: dict,
        verifier_scores: dict,
        resolved_target_activations: dict,
        comparators: list[Comparator],
        layout_info: dict,
        target_activation_info: dict[str, ActivationInfo],
        output_tensor: list[str],
    ) -> tuple[dict, dict]:
        """Calculates the verifier score for the given subgraph's final output and model's
        final output between target and framework tensors.

        Args:
            framework_activation_name (str): Name of the activation as per framework model
            golden_reference_output (dict): Dictionary of output names to tensor value
            target_output (dict): Final output of the subgraph.
            verifier_scores (dict): Verifier scores based on comparator.
            resolved_target_activations (dict): Output target activation.
            comparators (list[Comparator]): List of comparators for verification.
            layout_info (dict): Dictionary of layout data for each tensor.
            target_activation_info (dict[str, ActivationInfo]): Target activation information.
            output_tensor (list[str]): List of output tensor names

        Returns:
            tuple[dict, dict]: verifier_scores, target_activation_info
        """
        resolved_target_activation = resolved_target_activations[framework_activation_name]
        sanitized_target_activation = Helper.transform_node_names(resolved_target_activation)

        # First compute verification for the intermediate nodes
        reference_activation_output = {}
        target_activation_output = {}
        reference_activation_output[framework_activation_name] = golden_reference_output[
            framework_activation_name
        ]
        target_activation_output[sanitized_target_activation] = target_output[
            sanitized_target_activation
        ]
        intermediate_node_verifier_scores, target_activation_info = (
            self._compute_verification_score(
                reference_activation_output,
                target_activation_output,
                framework_activation_name,
                layout_info,
                resolved_target_activations,
                target_activation_info,
                comparators,
            )
        )

        verifier_scores[framework_activation_name] = {"self": intermediate_node_verifier_scores}

        # Now compute for its original model outputs
        original_outputs_verifier_scores = {}
        for original_output in output_tensor:
            resolved_target_original_output = resolved_target_activations[original_output]
            santized_original_target_output = Helper.transform_node_names(
                resolved_target_original_output
            )

            reference_activation_output = {}
            reference_activation_output[original_output] = golden_reference_output[original_output]

            target_activation_output = {}
            target_activation_output[santized_original_target_output] = target_output[
                santized_original_target_output
            ]

            original_output_verifier_score, target_activation_info = (
                self._compute_verification_score(
                    reference_activation_output,
                    target_activation_output,
                    original_output,
                    layout_info,
                    resolved_target_activations,
                    target_activation_info,
                    comparators,
                )
            )
            original_outputs_verifier_scores[original_output] = original_output_verifier_score

        verifier_scores[framework_activation_name]["original_outputs"] = (
            original_outputs_verifier_scores
        )

        return verifier_scores, target_activation_info

    def _compute_verification_score(
        self,
        reference_output: dict,
        target_output: dict,
        framework_activation_name: str,
        layout_info: dict,
        resolved_target_activations: dict,
        target_activation_info: dict[str, ActivationInfo],
        comparators: list[Comparator],
    ) -> tuple[dict, dict]:
        """Computes the verifier score between two given tensor outputs.

        Args:
            reference_output (dict): Reference output tensors from framework.
            target_output (dict): Target output
            framework_activation_name (str): Framework Activation name
            layout_info (dict): dictionary of layout information.
            resolved_target_activations (dict): dictionary of Target activation after resolution.
            target_activation_info (dict[str, ActivationInfo]): Target activation information.
            comparators (list[Comparator]): List of selected comparators

        Raises:
            VerificationError: If verification fails.

        Returns:
            tuple[dict, dict]: Verification scores and Target activation info
        """
        resolved_target_activation = resolved_target_activations[framework_activation_name]
        sanitized_target_activation = Helper.transform_node_names(resolved_target_activation)

        target_min = np.min(target_output[sanitized_target_activation])
        target_max = np.max(target_output[sanitized_target_activation])
        target_median = np.median(target_output[sanitized_target_activation])

        target_activation_info[sanitized_target_activation].distribution = (
            target_min,
            target_max,
            target_median,
        )

        verifier_scores = {}
        if target_output is not None and reference_output is not None:
            verifier = Verifier(comparators, logger=self._logger)
            tensor_mapping_dict = {sanitized_target_activation: framework_activation_name}
            graph_info = {
                "layout_info": layout_info,
                "tensor_mapping": tensor_mapping_dict,
            }

            try:
                verifier_score_dict = verifier.verify_dictionary_of_tensors(
                    reference_output, target_output, graph_info=graph_info
                )
            except Exception as exception:
                raise VerificationError(
                    f"Verification failed for reference activation: {framework_activation_name} "
                    f"and target_activation: {sanitized_target_activation}. Reason: {exception}"
                ) from exception

            verifier_score_dict = list(verifier_score_dict.values())[0]

            for comparator in comparators:
                verifier_score = verifier_score_dict[comparator.name]
                verifier_scores[comparator.name] = str(verifier_score)

        return verifier_scores, target_activation_info

    def _create_subgraph_quantization_override(
        self,
        subgraph: set,
        subgraph_output_names: list,
        supergroup_activations: set,
        qairt_encodings_converter: QairtEncodingsConverter,
        output_dir: Path,
    ) -> Path:
        """Creates quantization overrides file for the given subgraph intermediate tensor names.

        Args:
            subgraph (set): Subgraph under execution
            subgraph_output_names (list): Outputs of the subgraph under execution.
            supergroup_activations (set): Activations that are a supergroup.
            qairt_encodings_converter (QairtEncodingsConverter): QAIRT encodings converter object.
            output_dir (Path): Path to output directory.

        Returns:
            Path: Path to quantization override for given subgraph
        """
        subgraph_encodings = qairt_encodings_converter.create_subgraph_encodings(
            subgraph, supergroup_activations
        )

        subgraph_output_names = list(map(Helper.transform_node_names, subgraph_output_names))
        subgraph_override_dir = output_dir / "sub_graph_node_precision_files"
        subgraph_override_dir.mkdir(parents=True, exist_ok=True)
        file_name = "#".join(sorted(subgraph_output_names)) + ".json"
        subgraph_override_file_path = subgraph_override_dir / file_name
        with open(subgraph_override_file_path, "w") as file:
            json.dump(subgraph_encodings, file, indent=4)

        return subgraph_override_file_path

    def _get_output_tensor_names(self) -> list[str]:
        """Get output tensor names from model

        Returns:
            list: A list of output tensor names
        """
        framework_instance = get_framework_instance(
            framework=self.framework_type, logger=self._logger
        )
        model_proto = framework_instance.load_model(self.model)
        return framework_instance.get_output_tensor_names(model_proto)
