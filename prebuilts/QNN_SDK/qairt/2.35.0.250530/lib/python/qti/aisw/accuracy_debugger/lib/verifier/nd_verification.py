# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_warning_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import VerifierError, VerifierInputError
from qti.aisw.accuracy_debugger.lib.utils.nd_graph_structure import GraphStructure
from qti.aisw.accuracy_debugger.lib.utils.nd_logger import get_logger_log_file_path
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_tensor_paths
from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import permute_tensor_data_axis_order,\
    save_to_file, get_irgraph_tensors_info, get_irgraph_dtypes, load_data
from qti.aisw.accuracy_debugger.lib.verifier.models import VerifyResult
from qti.aisw.accuracy_debugger.lib.verifier.nd_verifier_factory import VerifierFactory
from qti.aisw.accuracy_debugger.lib.verifier.verifiers import ScaledDiffVerifier
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.common import truncate_native_tag


class Verification:
    SUMMARY_NAME = "summary"

    def __init__(self, default_verifier, logger, args, default_verifier_config=None):

        def generate_inference_to_golden_map(inference_tensors, mapping):
            return {
                inference:
                mapping[inference]
                if inference in mapping and mapping[inference] is not None else inference
                for inference in inference_tensors
            }

        def load_verifiers(verifier_configs):
            # type: (dict) -> dict
            verifier_objects = {}
            for verifier, config in verifier_configs.items():
                verifier_object = VerifierFactory().factory(verifier, config.get("parameters", {}))
                if verifier_config is None:
                    raise VerifierError(
                        get_message('ERROR_VERIFIER_INVALID_VERIFIER_NAME')(verifier))
                verifier_objects[verifier] = verifier_object, config
            return verifier_objects

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self._missing_verification_data_warning_message = None

        self.verifier_configs = {}
        if args.verifier_config and os.path.exists(args.verifier_config):
            with open(args.verifier_config) as verifier_config:
                self.verifier_configs = json.load(verifier_config)

        self.tensor_mapping = {}
        if args.tensor_mapping and os.path.exists(args.tensor_mapping):
            with open(args.tensor_mapping) as tensor_mapping:
                self.tensor_mapping = json.load(tensor_mapping)

        self.golden_tensor_paths = get_tensor_paths(args.golden_output_reference_directory)
        self.inference_tensor_paths = get_tensor_paths(args.inference_results)
        self.inference_tensors = tuple(self.inference_tensor_paths.keys())
        self.inference_types = None
        self.tensor_dimensions = None
        self.output_tensor_encodings = None

        if args.graph_struct and os.path.exists(args.graph_struct):
            graph_struct = GraphStructure.load_graph_structure(args.graph_struct)
            # To maintain topological order in summary, will use tensor names from graph
            self.inference_tensors = graph_struct.get_all_tensors()
            self.inference_types = graph_struct.get_all_types()
            self.tensor_dimensions = graph_struct.get_tensor_dimension_dict()
            self.output_tensor_encodings = graph_struct.get_all_output_tensor_encodings_dict()

        if ScaledDiffVerifier.NAME in self.verifier_configs or default_verifier == ScaledDiffVerifier.NAME:
            if not self.output_tensor_encodings:
                raise VerifierError(
                    get_message('ERROR_VERIFIER_SCALED_DIFF_MISSING_OUTPUT_ENCODING'))
            o_encodings = list(self.output_tensor_encodings.items())
            if o_encodings[0][1]['min'] == o_encodings[0][1]['max'] == o_encodings[0][1]['scale'] \
                 == o_encodings[0][1]['offset'] == 0.0:
                raise VerifierError(get_message('ERROR_VERIFIER_CANNOT_USE_SCALEDDIFF_VERIFIR'))

        self.inference_to_golden_tensor_map = generate_inference_to_golden_map(
            self.inference_tensors, self.tensor_mapping)

        self.specific_verifiers = load_verifiers(self.verifier_configs)

        self.default_verifier = default_verifier
        default_config = {}
        if default_verifier in self.verifier_configs:
            default_config = self.verifier_configs[default_verifier]["parameters"]
        elif default_verifier_config:
            ret, default_config = VerifierFactory().validate_configs(default_verifier,
                                                                     default_verifier_config)
            if not ret:
                errormsg = str(default_config['error']) if 'error' in default_config else ''
                raise VerifierError("VerifierFactory config_verify error: " + errormsg)
        self.default_verifier_obj = VerifierFactory().factory(default_verifier, default_config)
        if self.default_verifier_obj is None:
            raise VerifierError(
                get_message('ERROR_VERIFIER_INVALID_VERIFIER_NAME')(default_verifier))

        self.output_dir = args.output_dir
        self.tensors_info = get_irgraph_tensors_info(qnn_model_json_path=args.qnn_model_json_path,
                                                     dlc_path=args.dlc_path,
                                                     output_dir=self.output_dir)
        self.irgraph_dtypes = None
        if args.use_native_output_files:
            self.logger.debug(f'Loading outputs with use_native_output_files={args.use_native_output_files}')
            self.irgraph_dtypes = get_irgraph_dtypes(qnn_model_json_path=args.qnn_model_json_path,
                                                        dlc_path=args.dlc_path)

    def has_specific_verifier(self):
        for _, verifier_data in self.specific_verifiers.items():
            _, config = verifier_data
            tensors = config.get("tensors", list())
            for tensor in tensors:
                if tensor in self.inference_tensors:
                    return True
        return False

    def verify_tensors(self):
        """Runs the verifiers on their corresponding tensors :return: Pandas
        dataframe of the verifier return data."""

        def get_tensor_data(inference_tensor_names):
            # type: (list[str]) -> (list[np.array], list[np.array])
            """Given a list inference tensors, find the corresponding golden
            tensors and then retrieve the data for both :param
            inference_tensor_names: a list of tensor names :return: pair of
            lists containing tensor data."""

            missing = [
                tensor for tensor in inference_tensor_names
                if tensor not in self.inference_tensor_paths.keys()
            ]

            if missing:
                if self._missing_verification_data_warning_message is None:
                    log_file_path = get_logger_log_file_path(self.logger)
                    self._missing_verification_data_warning_message = get_warning_message(
                        "WARNING_VERIFIER_MISSING_TENSOR_DATA")(log_file_path)
                    self.logger.warning(self._missing_verification_data_warning_message)

                self.logger.debug(
                    get_warning_message("WARNING_VERIFIER_MISSING_INFERENCE_TENSOR_DATA")(
                        str(missing)))
                return None, None

            golden_tensors = [
                self.inference_to_golden_tensor_map[tensor] for tensor in inference_tensor_names
            ]
            missing = [
                tensor for tensor in golden_tensors
                if tensor not in self.golden_tensor_paths.keys()
            ]
            if missing:
                if self._missing_verification_data_warning_message is None:
                    log_file_path = get_logger_log_file_path(self.logger)
                    self._missing_verification_data_warning_message = get_warning_message(
                        "WARNING_VERIFIER_MISSING_TENSOR_DATA")(log_file_path)
                    self.logger.warning(self._missing_verification_data_warning_message)

                self.logger.debug(
                    get_warning_message("WARNING_VERIFIER_MISSING_GOLDEN_TENSOR_DATA")(
                        str(missing)))
                return None, None

            golden_data = load_data(golden_tensors, self.golden_tensor_paths, self.irgraph_dtypes)
            inference_data = load_data(inference_tensor_names, self.inference_tensor_paths, self.irgraph_dtypes)

            if self.tensors_info:
                # Permute target tensor to align with golden tensor
                for i in range(len(inference_tensor_names)):
                    tensor_name = os.path.basename(inference_tensor_names[i])
                    tensor_name = santize_node_name(tensor_name)
                    if tensor_name in self.tensors_info:
                        inference_data[i], _ = permute_tensor_data_axis_order(
                            inference_data[i], self.tensors_info[tensor_name])

            return golden_data, inference_data

        def generate_tensor_save_path(tensor_dir, tensor_name):
            save_path = os.path.join(tensor_dir, tensor_name)
            save_path_dir = os.path.dirname(save_path)

            # create the folder to save the tensor in if it doesn't already exist
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir, exist_ok=True)
            return save_path

        def verify(name, layer_type, tensor_dimensions, inference_tensors, verifier):
            """Find corresponding golden tensors for inference tensors, loads
            their data, and then performs verification :param name: Name for
            this verification result :param inference_tensors: list of tensors
            :param verifier: verifier to perform verification with :return:
            dictionary of verification result with a 'Name' field."""
            golden_data, inference_data = get_tensor_data(inference_tensors)

            if golden_data is None or inference_data is None:
                return None

            if type(verifier) == ScaledDiffVerifier:
                verifier.set_output_encoding(self.output_tensor_encodings[inference_tensors[0]])

            result = verifier.verify(layer_type, tensor_dimensions, golden_data, inference_data)
            return dict(result._asdict(), Name=name)

        summary_df = pd.DataFrame(columns=["Name"] + list(VerifyResult._fields))

        verifier_config_tensors = []
        for verifier_config in self.verifier_configs.values():
            verifier_config_tensors.extend(verifier_config["tensors"])
        # collect all the tensors used by specific verifiers
        verifier_config_tensors = set(
            [item for sublist in verifier_config_tensors for item in sublist])

        # ignore cases where the tensor will be verifier by a specific verifier
        default_verifiable_tensors = list(
            filter(lambda t: t not in verifier_config_tensors, self.inference_tensors))

        default_verifier_path = os.path.join(self.output_dir, self.default_verifier)

        if not os.path.isdir(default_verifier_path):
            os.makedirs(default_verifier_path)

        # We run our default verifier on all remaining tensors that have not been verified yet by any specified
        # verifier, we may consider adding support for more default verifiers in the future
        for inference_tensor_name in default_verifiable_tensors:
            try:
                layer_type, layer_dims = ' ', ' '
                if self.inference_types and inference_tensor_name in self.inference_types:
                    layer_type = self.inference_types[inference_tensor_name]
                if self.tensor_dimensions and inference_tensor_name in self.tensor_dimensions:
                    layer_dims = str(self.tensor_dimensions[inference_tensor_name])

                # verify one tensor at a time
                verification_data = verify(
                    inference_tensor_name, layer_type, layer_dims,
                    [inference_tensor_name], self.default_verifier_obj)
            except VerifierInputError as e:
                self.logger.warning(str(e))
                continue

            if verification_data is None:
                continue

            summary_df = pd.concat([summary_df, pd.DataFrame([verification_data])],
                                   ignore_index=True, sort=False)

            # tensor_path = generate_tensor_save_path(default_verifier_path, inference_tensor_name)
            # self.default_verifier_obj.save_data(tensor_path)

        for verifier_name, verifier_data in self.specific_verifiers.items():
            specific_verifier, config = verifier_data
            verifier_path = os.path.join(self.output_dir, verifier_name)

            for tensors in config.get("tensors", list()):
                try:
                    layer_type, layer_dims = ' ', ' '
                    if self.inference_types and tensors in self.inference_types:
                        layer_type = self.inference_types[tensors]
                    if self.tensor_dimensions and tensors in self.tensor_dimensions:
                        layer_dims = str(self.tensor_dimensions[tensors])

                    verification_data = verify(
                        str(tensors), layer_type, layer_dims,
                        [tensors], specific_verifier)
                except VerifierInputError as e:
                    self.logger.warning(str(e))
                    continue

                if verification_data is None:
                    if self._missing_verification_data_warning_message is None:
                        log_file_path = get_logger_log_file_path(self.logger)
                        self._missing_verification_data_warning_message = get_warning_message(
                            "WARNING_VERIFIER_MISSING_TENSOR_DATA")(log_file_path)
                        self.logger.warning(self._missing_verification_data_warning_message)
                    continue

                summary_df = pd.concat([summary_df, pd.DataFrame([verification_data])],
                                       ignore_index=True, sort=False)

                # tensor_path = generate_tensor_save_path(verifier_path, tensors[0])
                # specific_verifier.save_data(tensor_path)

        # Don't truncate columns
        max_colwidth = None
        if sys.version_info < (3, 8):
            # display.max_colwidth only uses an int argument prior to 3.8
            max_colwidth = -1
        with pd.option_context('display.max_colwidth', max_colwidth):
            filename = os.path.join(default_verifier_path, self.SUMMARY_NAME)
            save_to_file(summary_df, filename)

        return summary_df
