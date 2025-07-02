# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
from abc import abstractmethod
from logging import Logger

from qti.aisw.accuracy_debugger.encodings_converter.encodings_utils import (
    add_encodings,
    get_encodings_structure,
    identify_inter_activations_path,
    is_convert_op_in_path,
    needs_encoding_update,
)
from qti.aisw.accuracy_debugger.framework_runner.framework_factory import (
    get_framework_instance,
    get_framework_type,
)
from qti.aisw.accuracy_debugger.utils.graph_utils import get_common_parent_activations


class EncodingsConverter:
    """Base class for converting encodings file to AIMET format encodings file"""

    def __init__(self, framework_model_path: str, working_dir: str, logger: Logger) -> None:
        """Initializes the EncodingsConverter class

        Args:
            framework_model_path: Path to the framework model.
            working_dir: Path to the working directory.
            logger: Logger instance for logging.
        """
        self._framework_model_path = framework_model_path
        self._working_dir = working_dir
        self._logger = logger
        self._framework_connected_graph = None
        self._target_connected_graph = None
        self._framework_activations = None
        self._framework_activation_op_map = None
        self._framework_input_op_map = None
        self._user_encodings = {}
        self._target_activation_op_map = None
        self._resolved_target_activations = {}
        self._intialize()

    def _intialize(self) -> None:
        """Initializes the internal state of the EncodingsConverter."""
        # Create working directory
        os.makedirs(self._working_dir, exist_ok=True)

        # Get the Framework instance object
        framework_instance = get_framework_instance(
            framework=get_framework_type(self._framework_model_path), logger=self._logger
        )
        model = framework_instance.load_model(self._framework_model_path)
        # create framework_connected_graph
        self._framework_connected_graph = framework_instance.create_connected_graph(model)

        # create framework_op_activation_to_framework_op_map with node activation as key
        # and framework_op object as value
        framework_activation_op_map = {}
        framework_input_op_map = {}
        for _, op in self._framework_connected_graph.items():
            for output in op.outputs:
                framework_activation_op_map[output] = op
            for input_name in op.inputs:
                framework_input_op_map[input_name] = op
        self._framework_activation_op_map = framework_activation_op_map
        self._framework_input_op_map = framework_input_op_map

        self._framework_activations = [
            output for output, op in framework_activation_op_map.items() if op.op_type != "input"
        ]

    @abstractmethod
    def _create_target_connected_graph(self) -> None:
        """Creates target connected graph"""

    def get_framework_activations(self) -> list:
        """Return list of framework activations"""
        return self._framework_activations

    def get_framework_connected_graph(self) -> dict:
        """Return framework connected graph"""
        return self._framework_connected_graph

    def get_target_connected_graph(self) -> dict:
        """Return target connected graph"""
        return self._target_connected_graph

    def get_framework_activation_op_map(self) -> dict:
        """Return framework activation to framework op map"""
        return self._framework_activation_op_map

    def get_framework_input_op_map(self) -> dict:
        """Return framework input to framework op map"""
        return self._framework_input_op_map

    def get_target_activation_op_map(self) -> dict:
        """Return target activation to target op map"""
        return self._target_activation_op_map

    def get_resolved_target_activation(self) -> dict:
        """Return map of framework name to target activation name"""
        return self._resolved_target_activations

    def _set_output_encodings(
        self, encodings: dict, activation: str, visited_activation_encodings: dict
    ) -> tuple[dict, dict]:
        """Given the activation name, sets the activation encodings
        and returns tuple of updated encodings dict and visited activation encodings dict

        Args:
            encodings (dict): dictionary of converted encodings
            activation (str): activation name in the framework graph
            visited_activation_encodings (dict): dictionary with activation names as keys and
                encodings as values. This is used when the encoding for the activation is already
                present in the so-far-prepared encodings dictionary,
                but needs to be updated with some other convert op encoding according to precedence
        Returns:
            (dict, dict): tuple of updated encodings with encodings for given activation and
                updated visited_activation_encodings with activation.

        Raises:
            Exception: If fails to add the activation encoding to the encodings dictionary
        """
        if activation in self._user_encodings["activation_encodings"] and (
            activation not in visited_activation_encodings
            or needs_encoding_update(
                self._user_encodings["activation_encodings"][activation],
                visited_activation_encodings[activation],
                self._version,
            )
        ):
            try:
                encodings = add_encodings(
                    encodings=encodings,
                    tensor_name=activation,
                    tensor_encoding=self._user_encodings["activation_encodings"][activation],
                    encoding_type="activation_encodings",
                )
            except Exception as exception:
                self._logger.error(
                    f"Failed to add encodings for tensor {activation} with the error: {exception}"
                )
                raise exception
            visited_activation_encodings[activation] = self._user_encodings["activation_encodings"][
                activation
            ]

        return encodings, visited_activation_encodings

    def _set_param_encodings(self, encodings: dict, param: str) -> dict:
        """Given the param name, sets the param encodings
        and returns the encodings dict

        Args:
            encodings: dictionary of converted encodings
            param: param name in the framework graph

        Returns:
            dict: updated encodings dict with param encodings

        Raises:
            Exception: If fails to add the param encoding to the encodings dictionary
        """
        # TODO: Once shared weights issue is resolved, address convert_op @ param level
        try:
            encodings = add_encodings(
                encodings=encodings,
                tensor_name=param,
                tensor_encoding=self._user_encodings["param_encodings"][param],
                encoding_type="param_encodings",
            )
        except Exception as exception:
            self._logger.error(
                f"Failed to add encodings for tensor {param} with the error: {exception}"
            )
            raise exception

        return encodings

    def _set_input_encodings(
        self,
        encodings: dict,
        input_name: str,
        output_name: str,
        ignore_activation_encodings: set,
        visited_activation_encodings: dict,
    ) -> tuple[dict, dict]:
        """Given the activation name, sets the activation encodings
        and returns the encodings dict

        Args:
            encodings (dict): Dictionary of converted encodings.
            input_name (str): One of the framework op's input name for which encodings has to be
                resolved and added.
            output_name (str): One of the framework op's output name.
            ignore_activation_encodings (set): List of framework activations for which encodings
                has to be ignored in the new quantization overrides. For example, for conv, bn, relu
                qnn_net_json and qairt_encodings_json gives out activation encodings for both bn and
                relu. The user may want to delete bn encodings for some optimizations.
            visited_activation_encodings (dict): dictionary with activation names as keys and
                encodings as values. This is used when the encoding for the activation is already
                present in the so-far-prepared encodings dictionary,
                but needs to be updated with some other convert op encoding according to precedence

        Returns:
            (dict, dict): tuple of updated encodings dictionary and updated
                visited_activation_encodings with activation

        Raises:
            Exception: If fails to add the activation encoding to the encodings dictionary
        """
        common_parent_activations = get_common_parent_activations(
            input_name,
            self._target_activation_op_map,
            self._framework_activation_op_map,
            ignore_activation_encodings,
        )

        self._logger.debug(
            f"Common parent activations for {input_name} are {str(common_parent_activations)}"
        )
        for parent_activation in common_parent_activations:
            path = identify_inter_activations_path(
                output_name, parent_activation, self._target_activation_op_map, 0
            )

            self._logger.debug(f"PATH {str(path)} BETWEEN: {output_name} and {parent_activation}")
            convert_op_in_between, convert_activation_name = is_convert_op_in_path(
                path, self._target_activation_op_map
            )

            # Two cases:
            # 1. There is convert_op in between current_op and its parent op
            # 2. There is no convert_op in between
            input_tensor_enc = None
            if convert_op_in_between:
                input_tensor_enc = self._user_encodings["activation_encodings"][
                    convert_activation_name
                ]
            elif parent_activation in self._user_encodings["activation_encodings"]:
                input_tensor_enc = self._user_encodings["activation_encodings"][parent_activation]
            else:
                self._logger.error(f"{parent_activation} not found in user encodings")

            # Now check whether encodings for the parent_activation is already present or not
            # If present, check precendece and update if requried
            if input_tensor_enc and (
                parent_activation not in visited_activation_encodings
                or needs_encoding_update(
                    input_tensor_enc,
                    visited_activation_encodings[parent_activation],
                    self._version,
                )
            ):
                # Precendence: int16>int8>int4>fp32>fp16>fp8
                try:
                    encodings = add_encodings(
                        encodings=encodings,
                        tensor_name=parent_activation,
                        tensor_encoding=input_tensor_enc,
                        encoding_type="activation_encodings",
                    )
                except Exception as exception:
                    self._logger.error(
                        f"Failed to add encodings for tensor {parent_activation} "
                        f"with the error: {exception}"
                    )
                    raise exception
                visited_activation_encodings[parent_activation] = input_tensor_enc

        return encodings, visited_activation_encodings

    def create_subgraph_encodings(
        self, subgraph_target_activations: list = [], ignore_activation_encodings: set = set()
    ) -> dict:
        """Creates quantization encodings for the subgraph. Eliminates convert ops.
        for e.g. for node "conv.1", it's input, param, and output encodings will be kept in lower
        precision.

        Args:
            subgraph_target_activations: List of target activations for which subgraph
                quantization encodings has to be prepared. If empty, encodings will be created for
                full model graph.
            ignore_activation_encodings: List of framework activations for which encodings have
                to be ignored in the new quantization encodings. For example, for conv, bn, relu
                QAIRT encodings gives out activation encodings for both bn and
                relu. The user may want to delete bn encodings for some optimizations.

        Returns:
            dict: Quantization encodings for the given subgraph.

        Raises:
            Exception: If fails to generate subgraph encodings
        """
        encodings = get_encodings_structure(self._version)
        visited_activation_encodings = dict()

        subgraph_activations = subgraph_target_activations or self._target_activation_op_map.keys()

        for output_name in subgraph_activations:
            self._logger.debug("-" * 75)
            self._logger.debug(f"ACTIVATION: {output_name}")

            # Add output encodings
            if output_name in (set(self._framework_activations) - ignore_activation_encodings):
                try:
                    encodings, visited_activation_encodings = self._set_output_encodings(
                        encodings, output_name, visited_activation_encodings
                    )
                except Exception as exception:
                    raise exception

            current_op = self._target_activation_op_map[output_name]
            for input_tensor_name in current_op.inputs:
                # Add param encodings
                # Overwrite the already present float static encodings for the params
                if input_tensor_name in self._user_encodings["param_encodings"]:
                    try:
                        encodings = self._set_param_encodings(encodings, input_tensor_name)
                    except Exception as exception:
                        raise exception

                # Add input encodings
                # only for input_tensors which are actually activations
                # e.g. 316's inputs: ['315', 'features.0.1.weight', 'features.0.1.bias',
                # 'features.0.1.running_mean', 'features.0.1.running_var']
                # ['features.0.1.running_mean', 'features.0.1.running_var'] does not even
                # have param encodings
                elif input_tensor_name in self._target_activation_op_map.keys():
                    try:
                        encodings, visited_activation_encodings = self._set_input_encodings(
                            encodings,
                            input_tensor_name,
                            output_name,
                            ignore_activation_encodings,
                            visited_activation_encodings,
                        )
                    except Exception as exception:
                        raise exception

                # input_tensor_name is actually constant but not params with no encodings or
                # in ignore_activation_encodings
                else:
                    self._logger.debug(
                        f"Tensor: {input_tensor_name} is a constant in the target graph"
                    )

        return encodings
