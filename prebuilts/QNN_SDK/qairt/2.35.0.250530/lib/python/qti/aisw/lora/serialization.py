# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
import os
import json
import copy
import yaml
from enum import Enum
from safetensors.numpy import load_file, save_file
from .helpers import (open_and_load_json, create_encoding_map, get_encodings,
                      create_base_graph_name, validate_file_path, apply_safetensors_to_onnx)
from qti.aisw.converters.common.converter_ir.op_graph import QuantUpdatableMode
from qti.aisw.converters.common.utils.converter_utils import *
from enum import Enum
import onnx


class EncodingVersion(Enum):
    ENCODING_VERSION_V1 = "0.6.1"
    ENCODING_VERSION_V2 = "1.0.0"


class DefaultWeightEncoding(object):
    """
    This class contains the default values for all fields related to weight encoding.
    These encodings values will be used as encodings of weights tensors(zero tensor) of the inactive lora branches.
    """
    BITWIDTH = 8
    DTYPE = "int"
    IS_SYMMETRIC = "True"
    MAX = 0.0039061307907104492
    MIN = -0.00390625
    OFFSET = -128
    SCALE = 3.0636787414550781176470588235294e-5


class DefaultActivationEncoding(object):
    """
    This class contains the default values for all fields related to activation encoding
    These encodings values will be used as encodings of the activation tensors of the inactive lora branches.
    """
    BITWIDTH = 16
    DTYPE = "int"
    IS_SYMMETRIC = "False"
    MAX = 0.0039061307907104492
    MIN = -0.00390625
    OFFSET = -32768
    SCALE = 1.1920928955078125e-07


class EncodingGenerator(object):

    def __init__(self, attach_point_info_map, max_rank_attach_point_map):
        """
        This class handles the generation of the updated use case encoding
        for concurrences (including base concurrency).
        :param attach_point_info_map: A dictionary mapping attach point names to AttachPointInfo objects
        :param max_rank_attach_point_map: A dictionary mapping attach point names to MaxRankAttachPoint objects
        """
        self.attach_point_info_map = attach_point_info_map
        self.max_rank_attach_point_map = max_rank_attach_point_map

    @staticmethod
    def _get_default_weight_encoding_dict_v1(num_channels):
        """
        Function to create the default per channel weight encodings for version 1.
        :param num_channels: number of channels
        :return: a list containing encoding for every channel.
        """
        default_weight_encoding_dict = {
            "bitwidth": DefaultWeightEncoding.BITWIDTH,
            "dtype": DefaultWeightEncoding.DTYPE,
            "is_symmetric": DefaultWeightEncoding.IS_SYMMETRIC,
            "max": DefaultWeightEncoding.MAX,
            "min": DefaultWeightEncoding.MIN,
            "offset": DefaultWeightEncoding.OFFSET,
            "scale": DefaultWeightEncoding.SCALE
        }
        per_channel_encoding = [default_weight_encoding_dict for _ in range(num_channels)]
        return per_channel_encoding

    @staticmethod
    def _get_default_weight_encoding_dict_v2(name, num_channels):
        """
        Function to create the default per channel weight encodings for version 2.
        :param name: name of the tensor
        :param num_channels: number of channels
        :return: a list containing encoding for every channel.
        """
        per_channel_encoding = {
            "bw": DefaultWeightEncoding.BITWIDTH,
            "dtype": DefaultWeightEncoding.DTYPE.upper(),
            "enc_type": "PER_CHANNEL",
            "is_sym": DefaultWeightEncoding.IS_SYMMETRIC,
            "name": name,
            "offset": [DefaultWeightEncoding.OFFSET]*num_channels,
            "scale": [DefaultWeightEncoding.SCALE]*num_channels
        }
        return per_channel_encoding


    @staticmethod
    def _get_default_weight_encoding(name, num_channels, version):
        """
        Function to create the default per channel weight encodings.
        :param name: name of the tensor
        :param num_channels: number of channels
        :param version: encoding version
        :return: Default Encoding Value
        """
        if EncodingGenerator.is_version_v2(version):
            per_channel_encoding = EncodingGenerator._get_default_weight_encoding_dict_v2(name, num_channels)
        else:
            per_channel_encoding = EncodingGenerator._get_default_weight_encoding_dict_v1(num_channels)
        return per_channel_encoding

    @staticmethod
    def _get_default_activations_encoding(name, version):
        """
        Function to create the default activation encodings.
        :param name: name of the tensor
        :param version: encoding version
        :return:  a list containing one dict which represents activation encoding.
        """
        if EncodingGenerator.is_version_v2(version):
            activation_encoding = {
                "bw": DefaultActivationEncoding.BITWIDTH,
                "dtype": DefaultActivationEncoding.DTYPE.upper(),
                "enc_type": "PER_TENSOR",
                "is_sym": DefaultActivationEncoding.IS_SYMMETRIC,
                "name": name,
                "offset": [
                    DefaultActivationEncoding.OFFSET
                ],
                "scale": [
                    DefaultActivationEncoding.SCALE
                ]
            }
        else:
            activation_encoding = [{
                "bitwidth": DefaultActivationEncoding.BITWIDTH,
                "dtype": DefaultActivationEncoding.DTYPE,
                "is_symmetric": DefaultActivationEncoding.IS_SYMMETRIC,
                "max": DefaultActivationEncoding.MAX,
                "min": DefaultActivationEncoding.MIN,
                "offset": DefaultActivationEncoding.OFFSET,
                "scale": DefaultActivationEncoding.SCALE
            }]
        return activation_encoding

    @staticmethod
    def get_version(encodings):
        """
        Get the encoding version from the encoding dictionary.
        :param encodings: encoding dictionary
        :return: version of the encoding
        """

        if "version" in encodings:
            if encodings["version"] == "1.0.0":
                version = EncodingVersion.ENCODING_VERSION_V2
            elif encodings["version"] == "0.6.1":
                version = EncodingVersion.ENCODING_VERSION_V1
            else:
                log_error("Invalid Encoding Version: Got {}, Supported Version {}".
                          format(encodings["version"], ["1.0.0", "0.6.1"]))
        else:
            version = EncodingVersion.ENCODING_VERSION_V1
        return version

    @staticmethod
    def is_version_v2(version):
        """
        Check whether the specified encoding version is version 1 or version 2.
        :param version: String specifying version of the encoding
        :return: True if the version is version 1 or version 2
        """
        if version == EncodingVersion.ENCODING_VERSION_V2:
            return True
        return False

    def _generate_tensor_mapping(self, concurrency_info):
        """
        Generate mapping of the lora tensor for this concurrency where key is name of the lora tensor in the
        concurrency graph and the value is the name of the tensor in the max concatenated graph.
        :param concurrency_info: A concurrency info object
        :return: A dict mapping original tensor name to its corresponding name in the max concatenated graph.
        """

        tensor_mapping = {}
        concurrency_name = concurrency_info.name
        for attach_point_name in concurrency_info.attach_point_names:
            attach_point_info = self.attach_point_info_map[attach_point_name]
            max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
            original_tensor_names = attach_point_info.tensor_names[concurrency_name]
            updated_tensor_names = max_rank_attach_point.lora_tensor_names
            tensor_mapping[original_tensor_names.lora_a_weight_name] = updated_tensor_names.lora_a_weight_name
            tensor_mapping[original_tensor_names.lora_a_act] = updated_tensor_names.lora_a_act
            tensor_mapping[original_tensor_names.mul_scale] = updated_tensor_names.mul_scale
            tensor_mapping[original_tensor_names.mul] = updated_tensor_names.mul
            tensor_mapping[original_tensor_names.lora_b_weight_name] = updated_tensor_names.lora_b_weight_name
            tensor_mapping[original_tensor_names.lora_b_act] = updated_tensor_names.lora_b_act
            tensor_mapping[original_tensor_names.add] = updated_tensor_names.add

        return tensor_mapping

    def is_attach_point_tensor(self, tensor_name):
        """
        Return whether the tensor is a base graph tensor or not
        :param tensor_name: name of the tensor
        :return: True if the tensor is a base graph tensor, otherwise false
        """
        # Check whether the tensor contains base_layer substring.
        # In concurrency graph, base_layer is added in the name of the attach point nodes and tensor.
        if tensor_name != create_base_graph_name(tensor_name):
            return True
        return False


    def _update_names(self, encodings, tensor_mapping):
        """
        Update names of the lora branches in the encoding dict based on the provided mapping.
        :param encodings: Encoding Dictionary
        :param tensor_mapping: A dictionary mapping tensor name to the tensor name in the max concatenated graph.
        :return: Updated Encoding dictionary.
        """

        def update_encoding_dict(encoding_dict, original_encoding_dict):
            # update param encodings dictionary
            for tensor_name in original_encoding_dict:
                updated_tensor_name = tensor_name
                # Check whether the tensor is a lora branch tensor or attach point tensor.
                # If it is a lora branch tensor, then it should be present in the tensor mapping.
                if tensor_name in tensor_mapping:
                    updated_tensor_name = tensor_mapping[tensor_name]
                # Check whether the tensor contains attach point tensor.
                elif self.is_attach_point_tensor(tensor_name):
                    # update the name with the name in the base_graph.
                    # In concurrency graph, base_layer is added in the name of the attach point nodes and tensors.
                    # We need to update the name in the encoding after removing base_layer substring from the
                    # tensor name since in the max graph names are according to the base graph.
                    updated_tensor_name = create_base_graph_name(tensor_name)
                if tensor_name != updated_tensor_name:
                    encoding_dict[updated_tensor_name] = encoding_dict[tensor_name]
                    # this check is for version 2 encoding dictionary where "name" field is present inside dictionary
                    if "name" in encoding_dict[updated_tensor_name]:
                        encoding_dict[updated_tensor_name]["name"] = updated_tensor_name
                    # Delete the older tensor name entry from the encoding dictionary
                    del encoding_dict[tensor_name]

        # create a copy of the original encoding.
        updated_encodings = copy.deepcopy(encodings)

        # Update weight encoding names
        update_encoding_dict(updated_encodings["param_encodings"], encodings["param_encodings"])

        # Update weight encoding names
        update_encoding_dict(updated_encodings["activation_encodings"], encodings["activation_encodings"])

        return updated_encodings

    def _update_lora_a_weight_encoding(self, updated_encodings, concurrency_info):
        """
        Update the encoding of the Lora A conv weight in the encoding dictionary to
        align it to the max concatenated graph.
        For example, Support the number of channels in the encoding is m and
        number of channels in the graph is n.
        If these are not equal (m != n), then pad or trim the encoding to make it equal to n channels.
        Note: This function updates the encoding in place. It does not return anything.
        :param updated_encodings: Encoding dictionary
        :param concurrency_info: A Concurrency info object.
        :return: None
        """

        def update_weight_encoding_value_v1(encoding, max_rank):
            # For lora A weight, num of channels in the concurrency graph may not be equal to the max rank.
            # If the max rank is greater than the number of channel, then pad the per channel encoding
            # to make it equal to the max rank.
            # If the max rank is less than the rank, then take the initial max rank
            if len(encoding) > 1:
                if max_rank > len(encoding):
                    remaining_channels = max_rank - len(encoding)
                    pad_encoding_values = self._get_default_weight_encoding_dict_v1(remaining_channels)
                    encoding = encoding + pad_encoding_values
                else:
                    encoding = encoding[:max_rank]
            return encoding

        def update_weight_encoding_value_v2(encoding, max_rank):
            enc_type = encoding['enc_type']
            if enc_type == 'PER_CHANNEL':
                # For lora A weight, num of channels in the concurrency graph may not be equal to the max rank.
                # If the max rank is greater than the number of channel, then pad scale and offset with default values
                # If the max rank is less than the rank, then take the initial max rank values from
                # both scale and offset
                if max_rank > len(encoding['scale']):
                    remaining_channels = max_rank - len(encoding['scale'])
                    encoding['scale'] = encoding['scale'] + [DefaultWeightEncoding.SCALE]*remaining_channels
                    encoding['offset'] = encoding['offset'] + [DefaultWeightEncoding.OFFSET]*remaining_channels
                else:
                    encoding['scale'] = encoding['scale'][:max_rank]
                    encoding['offset'] = encoding['offset'][:max_rank]
            elif enc_type == "PER_TENSOR":
                pass
            else:
                raise ValueError("Unsupported enc_type {}".format(enc_type))

            return encoding

        version = self.get_version(updated_encodings)
        for attach_point_name in concurrency_info.attach_point_names:
            attach_point_info = self.attach_point_info_map[attach_point_name]
            max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
            updated_tensor_names = max_rank_attach_point.lora_tensor_names
            lora_a_weight = updated_tensor_names.lora_a_weight_name
            max_rank = attach_point_info.max_rank
            lora_weight_encoding = updated_encodings['param_encodings'][lora_a_weight]
            # Update the Lora A weight encoding in the encoding dict.
            if lora_a_weight in updated_encodings['param_encodings']:
                if self.is_version_v2(version):
                    updated_lora_weight_encoding = update_weight_encoding_value_v2(lora_weight_encoding, max_rank)
                else:
                    updated_lora_weight_encoding = update_weight_encoding_value_v1(lora_weight_encoding, max_rank)

            updated_encodings['param_encodings'][lora_a_weight] = updated_lora_weight_encoding

    def _update_encodings_with_default_values(self, encodings, max_rank_attach_point, version):
        """
        Update the encoding dict with default weight and activation encoding for all
        the lora branch tensors (activation and weights).
        For add node, copy the encoding from the preceding conv node in the main branch.
        Note: This function updates the encoding in place. It does not return anything.
        :param encodings: Encodings dictionary
        :param max_rank_attach_point: A MaxRankAttachPoint object.
        :param version: encoding version (Encoding_VERSION_V1 or ENCODING_VERSION_V2)
        :return: None.
        """
        # Num of output channels. It is required for per channel quantization
        if max_rank_attach_point.op.op_type == 'Conv':
            num_channels_lora_a = max_rank_attach_point.lora_a_weight_shape[0]
            num_channels_lora_b = max_rank_attach_point.lora_b_weight_shape[0]
        elif max_rank_attach_point.op.op_type == "MatMul":
            num_channels_lora_a = max_rank_attach_point.lora_a_weight_shape[1]
            num_channels_lora_b = max_rank_attach_point.lora_b_weight_shape[1]
        else:
            raise ValueError("Invalid Attach Point {} : "
                             "op_type for attach_point should be Conv or MatMul but got {}".
                             format(max_rank_attach_point.op.name, max_rank_attach_point.op.op_type))

        # The output of the attach point is one of the input of the add node
        attach_point_output_name = max_rank_attach_point.op.output[0]
        lora_tensor_names = max_rank_attach_point.lora_tensor_names

        lora_a_weight_name = lora_tensor_names.lora_a_weight_name
        encodings["param_encodings"][lora_a_weight_name] = self._get_default_weight_encoding(lora_a_weight_name,
                                                                                             num_channels_lora_a,
                                                                                             version)
        lora_b_weight_name = lora_tensor_names.lora_b_weight_name
        encodings["param_encodings"][lora_b_weight_name] = self._get_default_weight_encoding(lora_b_weight_name,
                                                                                             num_channels_lora_b,
                                                                                             version)
        lora_a_act = lora_tensor_names.lora_a_act
        encodings["activation_encodings"][lora_a_act] = self._get_default_activations_encoding(lora_a_act, version)
        lora_b_act = lora_tensor_names.lora_b_act
        encodings["activation_encodings"][lora_b_act] = self._get_default_activations_encoding(lora_b_act, version)
        mul_output = lora_tensor_names.mul
        encodings["activation_encodings"][mul_output] = self._get_default_activations_encoding(mul_output, version)

        # Copy the encoding of the output of the conv node (or input of the add node) to the output of the add node
        if attach_point_output_name in encodings["activation_encodings"]:
            encodings["activation_encodings"][lora_tensor_names.add] = copy.deepcopy(encodings["activation_encodings"][attach_point_output_name])
            if "name" in encodings["activation_encodings"][lora_tensor_names.add]:
                encodings["activation_encodings"][lora_tensor_names.add]["name"] = lora_tensor_names.add

    def _add_default_values(self, encodings, concurrency_info):
        """
        Add default encodings for the Lora branches which are not the part of the specified concurrency.
        Note: This function updates the encoding in place. It does not return anything.
        :param encodings: Encoding Dictionary
        :param concurrency_info: A Concurrency Info object.
        :return: None
        """
        version = self.get_version(encodings)
        for attach_point_name in self.max_rank_attach_point_map:
            # Only add the default values if the attach-point is not the part of the concurrency
            if attach_point_name not in concurrency_info.attach_point_names:
                max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
                self._update_encodings_with_default_values(encodings, max_rank_attach_point, version)

    def generate_encodings_for_concurrency(self, encodings, concurrency_info):
        """
        Generate new encoding dict for the given concurrency
        :param encodings: Use case encoding
        :param concurrency_info: A concurrency info object
        :return: use-case encodings for max-rank graph
        """

        # change the encoding v2 dictionary into a map to easy look up and make it aligned to v1.
        version = self.get_version(encodings)
        if self.is_version_v2(version):
            encodings = create_encoding_map(encodings)

        # Step 1: Create tensor mapping of the lora branch tensors for the given concurrency
        tensor_mapping = self._generate_tensor_mapping(concurrency_info)

        # Step 2: Update lora tensor names in the concurrency encoding using the tensor mapping
        # create in the first step.
        updated_encodings = self._update_names(encodings, tensor_mapping)

        # Step 3: Update Lora A weight encoding in the concurrency encoding
        self._update_lora_a_weight_encoding(updated_encodings, concurrency_info)

        # Step 4: Add default encoding for the lora branches which are not the part of the given concurrency
        self._add_default_values(updated_encodings, concurrency_info)

        # change the encoding dict in original v2 format
        if self.is_version_v2(version):
            updated_encodings = get_encodings(updated_encodings)

        return updated_encodings


class TensorNamesSerializer(object):
    def __init__(self, max_rank_attach_point_map, quant_updatable_mode):
        self.max_rank_attach_point_map = max_rank_attach_point_map
        self.quant_updatable_mode = quant_updatable_mode

    def generate_tensor_names(self):
        def get_indices():
            indices = []
            for attach_point in self.max_rank_attach_point_map.values():
                indices_tensor_name = attach_point.alpha_separation_gather_indices_tensor_name
                indices.append(indices_tensor_name)
            return indices

        def get_weights():
            weights = []
            for attach_point in self.max_rank_attach_point_map.values():
                attach_point_tensors_names = attach_point.lora_tensor_names
                weights.append(attach_point_tensors_names.lora_a_weight_name)
                weights.append(attach_point_tensors_names.lora_b_weight_name)
            return weights

        def get_activations():
            activations = []
            for attach_point in self.max_rank_attach_point_map.values():
                attach_point_tensors_names = attach_point.lora_tensor_names
                activations.append(attach_point_tensors_names.mul)
                activations.append(attach_point_tensors_names.lora_a_act)
                activations.append(attach_point_tensors_names.lora_b_act)
            return activations

        if self.quant_updatable_mode in [QuantUpdatableMode.NONE]:
            return get_weights() + get_indices()
        elif self.quant_updatable_mode == QuantUpdatableMode.ADAPTER_ONLY:
            return get_weights() + get_indices() + get_activations()
        else:
            raise RuntimeError("Invalid quant_updatable_mode: {}".format(self.quant_updatable_mode))

    def save_tensor_names(self, path):
        tensor_names = self.generate_tensor_names()
        with open(path, 'w') as tensor_name_file:
            for name in tensor_names:
                tensor_name_file.write(f"{name}\n")
        log_info("Lora tensor name list saved at {}".format(path))


class EncodingSerializer(object):
    def __init__(self,
                 concurrency_infos,
                 attach_point_info_map,
                 max_rank_attach_point_map,
                 quant_updatable_mode,
                 output_dir):
        self.concurrency_infos = concurrency_infos
        self.attach_point_info_map = attach_point_info_map
        self.max_rank_attach_point_map = max_rank_attach_point_map
        self.output_dir = output_dir
        if quant_updatable_mode not in [QuantUpdatableMode.NONE, QuantUpdatableMode.ADAPTER_ONLY]:
            raise ValueError("Unsupported Mode: {}".format(mode))
        self.quant_updatable_mode = quant_updatable_mode
        self.concurrency_encodings_file_path = {}
        self.encoding_generator = EncodingGenerator(self.attach_point_info_map, self.max_rank_attach_point_map)

    def get_base_concurrency(self):
        """
        Get the concurrency info for the base use-case.
        :return: A ConcurrencyInfo object.
        """
        for concurrency_info in self.concurrency_infos:
            if concurrency_info.is_base():
                base_concurrency = concurrency_info
        return base_concurrency

    def get_concurrency_info(self, concurrency_name):
        """
        Get the Concurrency info for the given concurrency name
        :param concurrency_name: name of the concurrency
        :return: ConcurrencyInfo object
        """
        for concurrency_info in self.concurrency_infos:
            if concurrency_info.name == concurrency_name:
                return concurrency_info


    def _add_tensor_encodings(self, destination_encodings, source_encodings, weight_tensors=[], activation_tensors=[]):
        """
        Copies the encoding for the specified weight and activation tensors from encoding2 to encoding1.

        :param destination_encodings: The target encoding dictionary where the specified tensors' encodings
        will be copied to.
        :param source_encodings: The source encoding dictionary from which the specified tensors' encodings
        will be copied.
        :param weight_tensors: A list of weight tensor names whose encodings will be copied from
        source_encodings to destination_encodings.
        :param activation_tensors: A list of activation tensor names whose encodings will be copied from
        source_encodings to destination_encodings.
        :return: None
        """
        for tensor_name in activation_tensors:
            destination_encodings['activation_encodings'][tensor_name] = copy.deepcopy(source_encodings['activation_encodings'][tensor_name])
            if "name" in destination_encodings['activation_encodings'][tensor_name]:
                destination_encodings['activation_encodings'][tensor_name]["name"] = tensor_name

        for tensor_name in weight_tensors:
            destination_encodings['param_encodings'][tensor_name] = copy.deepcopy(source_encodings['param_encodings'][tensor_name])
            if "name" in destination_encodings['param_encodings'][tensor_name]:
                destination_encodings['param_encodings'][tensor_name]["name"] = tensor_name

    def generate_none_mode_encoding(self):
        """
        Generate the encoding dict for none quant mode.
        :return: Encoding dict
        """
        log_debug("Generating Encodings for the none quant-updateable mode")
        base_concurrency = self.get_base_concurrency()

        base_encodings = open_and_load_json(base_concurrency.quant_overrides)

        if base_encodings["version"] == "1.0.0":
            base_encodings = create_encoding_map(base_encodings)

        # Create a map where the key is the concurrency name and the value is a list of attach points.
        # For each attach point in the values, extract the encoding from the concurrency encoding and
        # add it to the base encoding.
        concurrency_to_attach_point = {}
        for attach_point_name in self.attach_point_info_map:
            attach_point_info = self.attach_point_info_map[attach_point_name]
            if attach_point_info.max_concurrency not in concurrency_to_attach_point:
                concurrency_to_attach_point[attach_point_info.max_concurrency] = [attach_point_name]
            else:
                concurrency_to_attach_point[attach_point_info.max_concurrency].append(attach_point_name)

        for concurrency_name in concurrency_to_attach_point:
            concurrency_info = self.get_concurrency_info(concurrency_name)

            # load usecase encodings
            usecase_encodings = open_and_load_json(concurrency_info.quant_overrides)

            # generate updated encodings for this concurrency
            usecase_encodings = self.encoding_generator.generate_encodings_for_concurrency(
                usecase_encodings,
                concurrency_info
            )

            if base_encodings["version"] == "1.0.0":
                usecase_encodings = create_encoding_map(usecase_encodings)

            for attach_point_name in concurrency_to_attach_point[concurrency_name]:
                max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
                lora_tensor_names = max_rank_attach_point.lora_tensor_names
                # Add the encoding of Lora branch tensors from the use-case encodings to the base encodings.
                weight_tensors = [lora_tensor_names.lora_a_weight_name, lora_tensor_names.lora_b_weight_name]
                activation_tensors = [lora_tensor_names.lora_a_act, lora_tensor_names.lora_b_act,
                                      lora_tensor_names.mul, lora_tensor_names.add]
                self._add_tensor_encodings(base_encodings, usecase_encodings, weight_tensors, activation_tensors)

            log_debug("Encodings for the following LoRA branches corresponding are extracted from the use-case {} : "
                      "{}".format(concurrency_name, concurrency_to_attach_point[concurrency_name]))

        if base_encodings["version"] == "1.0.0":
            base_encodings = get_encodings(base_encodings)

        return base_encodings

    def _add_lora_branch_encodings(self, base_encodings, usecase_encodings, concurrency_info):
        """
        Copy the lora branch from updated use case encoding to the base encodings.
        Use case encodings is properly aligner as per the max rank graph and also contains default values.
        :param base_encodings: A dictionary containing base graph encodings
        :param usecase_encodings: A dictionary containing updated use-case encodings
        :param concurrency_info: A ConcurrencyInfo object
        :return: A dict containing final encoding for this concurrency
        """
        if base_encodings["version"] == "1.0.0":
            base_encodings = create_encoding_map(base_encodings)
            usecase_encodings = create_encoding_map(usecase_encodings)

        for attach_point_name in self.attach_point_info_map:
            max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
            lora_tensor_names = max_rank_attach_point.lora_tensor_names
            # Add the encoding of Lora branch tensors (lora_a_weight_name, lora_b_weight_name,
            # lora_a_act, lora_b_act, mul_output from the use-case encodings to the base encodings.
            weight_tensors = [lora_tensor_names.lora_a_weight_name, lora_tensor_names.lora_b_weight_name]
            activation_tensors = [lora_tensor_names.lora_a_act, lora_tensor_names.lora_b_act,
                                  lora_tensor_names.mul, lora_tensor_names.add]
            self._add_tensor_encodings(base_encodings, usecase_encodings, weight_tensors, activation_tensors)

            # For default case, use the conv output encoding from the base encodings for lora-add.
            if attach_point_name not in concurrency_info.attach_point_names:
                attach_point_output_name = max_rank_attach_point.op.output[0]
                if attach_point_output_name in base_encodings["activation_encodings"]:
                    base_encodings["activation_encodings"][lora_tensor_names.add] = copy.deepcopy(base_encodings["activation_encodings"][attach_point_output_name])
                    if "name" in base_encodings["activation_encodings"][lora_tensor_names.add]:
                        base_encodings["activation_encodings"][lora_tensor_names.add]["name"] = lora_tensor_names.add

        if base_encodings["version"] == "1.0.0":
            base_encodings = get_encodings(base_encodings)

        return base_encodings

    def generate_encoding_for_concurrency(self, concurrency_info):
        """
        Generate Encodings for the specified Concurrency based on the mode.
        Note: Concurrency encodings can only be generated for the following modes: ADAPTER_ONLY.
        Therefore, this method is supported only for these modes."
        :param concurrency_info: A concurrency info object
        :return: Encoding dict for this use-case.
        """
        base_concurrency = self.get_base_concurrency()
        # load usecase encodings
        validate_file_path(concurrency_info.quant_overrides)
        f = open(concurrency_info.quant_overrides)
        usecase_encodings = json.load(f)

        # generate new encodings for this concurrency. This will return a new dictionary
        updated_encodings = self.encoding_generator.generate_encodings_for_concurrency(
            usecase_encodings,
            concurrency_info
        )

        # Take the base encoding and add the LoRa branch tensors from the
        # use-case encodings. This ensures that the encodings for the base graph tensors remain
        # consistent across all use cases.
        # This will be the final encoding for this concurrency.
        f = open(base_concurrency.quant_overrides)
        base_encodings = json.load(f)
        updated_encodings = self._add_lora_branch_encodings(base_encodings, updated_encodings, concurrency_info)

        return updated_encodings

    def serialize(self):
        """
        This method generates and serializes the encoding for the mode specified during instantiation.
        :return: None
        """
        def save_encodings(encodings, concurrency_info):
            if concurrency_info.is_base():
                path = os.path.join(self.output_dir, "base_encodings.json".format(concurrency_info.name))
            else:
                path = os.path.join(self.output_dir, "{}_encodings.json".format(concurrency_info.name))

            with open(path, 'w') as json_file:
                json.dump(encodings, json_file, indent=4)
            self.concurrency_encodings_file_path[concurrency_info.name] = path
            log_info("Encoding file for {} concurrency saved at {}".format(concurrency_info.name, path))

        base_concurrency = self.get_base_concurrency()
        # If the encodings is not present for the base graph then skip do not run encodings serializer
        if not base_concurrency.quant_overrides:
            log_info("Quantization overrides have not been provided for the base graph. "
                     "Consequently, encodings will not be generated for any use cases")
            return None

        if self.quant_updatable_mode == QuantUpdatableMode.NONE:
            base_encodings = self.generate_none_mode_encoding()
            # Encoding will be same across all the use-case in None Quant mode.
            # serialize the generated base encodings for all the use-cases.
            for concurrency_info in self.concurrency_infos:
                save_encodings(base_encodings, concurrency_info)
        else:
            for concurrency_info in self.concurrency_infos:
                encodings = self.generate_encoding_for_concurrency(concurrency_info)
                save_encodings(encodings, concurrency_info)


class LoraSerializer(object):
    def __init__(
            self,
            concurrency_infos,
            safe_tensor_path,
            indices_map,
            attach_point_info_map,
            max_rank_attach_point_map,
            output_dir,
            max_graph_path,
            quant_updatable_mode,
            dump_onnx=False
    ):
        self.concurrency_infos = concurrency_infos
        self.safe_tensor_path = safe_tensor_path
        self.indices_map = indices_map
        self.attach_point_info_map = attach_point_info_map
        self.max_rank_attach_point_map = max_rank_attach_point_map
        self.output_dir = output_dir
        self.concurrency_weight_file_path = {}
        self.concurrency_encodings_file_path = {}
        self.max_graph_path = max_graph_path
        self.quant_updatable_mode = quant_updatable_mode
        self.encoding_serializer = EncodingSerializer(
            self.concurrency_infos,
            self.attach_point_info_map,
            self.max_rank_attach_point_map,
            quant_updatable_mode,
            self.output_dir,
        )
        self.dump_onnx = dump_onnx

    @staticmethod
    def __concatenate_weights(weight_list, axis, max_rank):
        """
        Concatenates a list of n-dimensional array along the specified axis and
        adds padding to the concatenated vector till specified max rank
        :param weight_list: a list of n-d arrays
        :param axis: axis along with array will be concatenated
        :param max_rank: the required rank for the given axis
        :return: A concatenated array
        """
        # concatenate all the weights along the axis
        concatenated_weight = np.concatenate(weight_list, axis=axis)

        # calculate the required padding
        padding_needed = max_rank - concatenated_weight.shape[axis]
        if padding_needed > 0:
            padding_shape = list(concatenated_weight.shape)
            padding_shape[axis] = padding_needed
            padding = np.full(padding_shape, 0, dtype=np.float32)
            # concatenate the padding tensor to the weights
            padded_concatenated_weight = np.concatenate([concatenated_weight, padding], axis=axis)
        else:
            padded_concatenated_weight = concatenated_weight

        return padded_concatenated_weight

    def generate_safetensors_for_concurrency(self, concurrency_info, adapter_weights):
        """
        Generate concatenated weights for max rank graph for the given concurrency(or use case)
        and save it in the safetensor format.
        Along with lora A and lora B node weights, it also includes the indices tensor required for this concurrency
        :param concurrency_info: ConcurrencyInfo object
        :param adapter_weights: A dict containing all the adapter weights.
        :return: A dict containing all the weights for this concurrency
        """
        concurrency_safetensors = {}
        for attach_point_name in self.attach_point_info_map:
            attach_point_info = self.attach_point_info_map[attach_point_name]
            max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
            # Initialize lora A and lora B weight with default values
            lora_a_weight = np.zeros(max_rank_attach_point.lora_a_weight_shape, dtype=np.float32)
            lora_b_weight = np.zeros(max_rank_attach_point.lora_b_weight_shape, dtype=np.float32)
            # update lora A and lora B weight if this attach point exists in this concurrency
            if attach_point_name in concurrency_info.attach_point_names:
                lora_a_weights = []
                lora_b_weights = []
                # Extract all the adapter weights for this attach point required for this concurrency
                for adapter_name in concurrency_info.adapter_names:
                    if adapter_name in attach_point_info.adapter_names:
                        lora_a_name = attach_point_info.weight_info[adapter_name].lora_weightA_name
                        lora_b_name = attach_point_info.weight_info[adapter_name].lora_weightB_name
                        lora_a_weights.append(adapter_weights[lora_a_name])
                        lora_b_weights.append(adapter_weights[lora_b_name])

                # Max rank for this attach point
                max_rank = attach_point_info.max_rank

                # Concatenate all the extracted weights along the required dimension.
                # ------------------------------------------------------------------------------|
                # |   op-type    |   lora A concatenated axis   |    lora B Concatenated axis   |
                # |--------------|--------------------------------------------------------------|
                # |    Conv      |           0                  |              1                |
                # |    MatMul    |           1                  |              0                |
                # -------------------------------------------------------------------------------

                if attach_point_info.op_type == "Conv":
                    lora_a_weight = self.__concatenate_weights(lora_a_weights, 0, max_rank)
                    lora_b_weight = self.__concatenate_weights(lora_b_weights, 1, max_rank)
                elif attach_point_info.op_type == "MatMul":
                    lora_a_weight = self.__concatenate_weights(lora_a_weights, 1, max_rank)
                    lora_b_weight = self.__concatenate_weights(lora_b_weights, 0, max_rank)

            # Extract the new name of Lora A and B node for this attach point in the max graph.
            lora_a_weight_name = max_rank_attach_point.lora_tensor_names.lora_a_weight_name
            lora_b_weight_name = max_rank_attach_point.lora_tensor_names.lora_b_weight_name

            concurrency_safetensors[lora_a_weight_name] = lora_a_weight
            concurrency_safetensors[lora_b_weight_name] = lora_b_weight

            # Get the indices tensor for this attach point
            indices_tensor = self.indices_map[concurrency_info.name].attach_pt_indices[attach_point_name].alpha_indices
            indices_tensor = np.array(indices_tensor)
            indices_tensor_name = max_rank_attach_point.alpha_separation_gather_indices_tensor_name
            concurrency_safetensors[indices_tensor_name] = indices_tensor

        return concurrency_safetensors

    def generate_and_save_safetensors(self):
        """
        Function to generate safe tensor for each concurrency.
        return: None
        """
        adapter_weights = load_file(self.safe_tensor_path)
        for concurrency_info in self.concurrency_infos:
            if concurrency_info.is_base():
                continue
            concurrency_safetensors = self.generate_safetensors_for_concurrency(concurrency_info, adapter_weights)
            # save the weights in the safe tensor file
            path = os.path.join(self.output_dir, "{}.safetensors".format(concurrency_info.name))
            save_file(concurrency_safetensors, path)
            self.concurrency_weight_file_path[concurrency_info.name] = path
            log_info("Safetensors file for {} concurrency saved at {}".format(concurrency_info.name, path))

    def dump_patched_onnx_model(self):
        """
        Dump the patched Onnx file for all the concurrences.
        :return: None
        """
        for concurrency_info in self.concurrency_infos:
            if concurrency_info.is_base():
                continue
            # Load the ONNX model
            model = onnx.load(self.max_graph_path, load_external_data=True)
            concurrency_name = concurrency_info.name
            # load the safetensors file for this concurrency
            safetensor_file_path = self.concurrency_weight_file_path[concurrency_name]
            safetensors_dict = load_file(safetensor_file_path)

            # Apply the safetensors file and generate the patched Onnx model
            patched_model = apply_safetensors_to_onnx(model, safetensors_dict)

            # Save the patched Onnx model
            path = os.path.join(self.output_dir, "{}.onnx".format(concurrency_name))
            onnx.save(patched_model, path, save_as_external_data=True,
                      all_tensors_to_one_file=True, location=concurrency_name+".data")
            log_info("Patched ONNX model for use-case {} saved at {}".format(concurrency_name, path))

    def generate_and_save_importer_config(self):
        """
        Generate lora config for qairt-lora-importer tool. It will contain the information about each use case.
        :return: None
        """
        importer_config_data = {"use_case": []}
        for concurrency_name in self.concurrency_weight_file_path:
            use_case_dict = dict()
            use_case_dict['model_name'] = self.max_graph_path
            use_case_dict['name'] = concurrency_name
            use_case_dict['lora_weights'] = self.concurrency_weight_file_path[concurrency_name]
            if concurrency_name in self.concurrency_encodings_file_path:
                use_case_dict['quant_overrides'] = self.concurrency_encodings_file_path[concurrency_name]
            importer_config_data["use_case"].append(use_case_dict)

        save_path = os.path.join(self.output_dir, "lora_importer_config.yaml")
        log_info("Importer Config saved at {}".format(save_path))

        with open(save_path, "w") as f:
            yaml.dump(importer_config_data, f)

    def serialize(self):
        """
        Generate the following output files of the qairt creator tool.
          - Safetensors file and encodings file per concurrency
          - Encodings file and encodings file per concurrency
          - A txt file containing lora tensor names
          - Lora Importer YAML config file
        :return: None
        """
        # generate concatenated weight per concurrency and save in safetensor file
        log_debug("Generating safetensors file and encoding file for each concurrency...")
        self.generate_and_save_safetensors()

        if self.dump_onnx:
            # Dump the patched ONNX model
            self.dump_patched_onnx_model()

        # Generate encodings per concurrency and dump the encodings in a json file.
        # This steps also generates encoding for the base model.
        self.encoding_serializer.serialize()
        self.concurrency_encodings_file_path = self.encoding_serializer.concurrency_encodings_file_path
        log_debug("Successfully generated safetensors file and encoding file for each concurrency.")

        # Generate and save all the updateable tensor in the max rank graph
        log_debug("Generating lora tensor names list...")
        tensor_names_serializer = TensorNamesSerializer(self.max_rank_attach_point_map, self.quant_updatable_mode)
        tensor_names_serializer.save_tensor_names(path=os.path.join(self.output_dir, "lora_tensor_names.txt"))
        log_debug("Successfully generated lora tensor names list.")

        # Create Lora importer yaml config
        log_debug("Generating yaml config for lora importer tool...")
        self.generate_and_save_importer_config()
        log_debug("Successfully generated yaml config for lora importer tool.")

