# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import yaml
from .helpers import *
from dataclasses import dataclass, field
from safetensors.numpy import load_file, save_file
import onnx
from onnx.numpy_helper import to_array
from qti.aisw.converters.common.utils.converter_utils import *


@dataclass
class AdapterInfo:
    name: str
    rank: int
    attach_points: dict


@dataclass
class AttachPtIndices:
    def __init__(self):
        self.alpha_indices = []


@dataclass
class ConcurrencyIndices:
    def __init__(self, name):
        self.name = name
        self.attach_pt_indices = {}


@dataclass
class ConcurrencyInfo:
    name: str
    adapter_names: list
    model: str
    quant_overrides: str
    attach_point_names: list = field(default_factory=list)
    adapter_infos: dict = field(default_factory=dict)

    def is_base(self):
        """
        Check whether the concurrency is the base concurrency or not
        :return: True if the concurrency is base concurrency, otherwise False
        """
        if self.name == 'Base' or self.name == 'base':
            return True
        return False


@dataclass
class LoraTensorNames:
    lora_a_weight_name: str = ""
    lora_a_act: str = ""
    mul_scale: str = ""
    mul: str = ""
    lora_b_weight_name: str = ""
    lora_b_act: str = ""
    add: str = ""


@dataclass
class LoraNodeNames:
    lora_a: str = ""
    lora_b: str = ""
    mul: str = ""
    add: str = ""


@dataclass
class LoraWeightInfo:
    lora_weightA_name: str
    lora_weightB_name: str


@dataclass
class AttachPointInfo(object):
    name: str  # onnx op-name in the base graph
    max_rank: int  # max-rank computed for this attach-point
    max_concurrency: str = None  # concurrency name which has the maximum rank for this attach point
    adapter_names: list = field(default_factory=list)  # Note*: ordered list of adapters that have this attach-point
    op_type: str = ""  # operation type for the attach point
    weight_info: dict = field(default_factory=dict)  # for adapters
    tensor_names: dict = field(default_factory=dict)  # for encoding mapping


class LoraConfigParser(object):
    """
    This class handles parsing and validation of the Lora yaml config file
    and create list of concurrency info object.
    """

    @staticmethod
    def validate(config_path):
        """
        Function to validate lora YAML config.
        It checks all the paths and other required information in the lora config.
        :param config_path: ath of the lora YAML config file
        :return:
        """
        with open(config_path) as fp:
            lora_config = yaml.safe_load(fp)

        if 'adapter' not in lora_config:
            raise ValueError("Invalid LoRA YAML : adapter field is required")

        adapter_names = set()
        for adapter in lora_config['adapter']:
            if "name" not in adapter:
                raise ValueError("Invalid LoRA YAML : name field is required for each adapter")

            if "lora_config" not in adapter:
                raise ValueError("Invalid LoRA YAML : lora_config field is required for each adapter")

            if adapter['name'] in adapter_names:
                raise ValueError("Invalid LoRA YAML : adapter name, {}, is not unique".format(adapter['name']))

            adapter_names.add(adapter['name'])

            config = adapter['lora_config']
            validate_file_path(config)

        if 'use-case' not in lora_config:
            raise ValueError("Invalid LoRA YAML : use-case field is required")

        has_quant_updatable_tensors = False
        use_case_names = set()
        base_exists = False
        for concurrency in lora_config['use-case']:
            if "name" not in concurrency:
                raise ValueError("Invalid LoRA YAML : name field is required for each use-cases")

            if "adapter_names" not in concurrency:
                raise ValueError("Invalid LoRA YAML : adapter_names field is required for each use-cases")

            if "model_name" not in concurrency:
                raise ValueError("Invalid LoRA YAML : model_name field is required for each use-cases")

            if (concurrency["name"] == "Base" or concurrency["name"] == "base") and base_exists:
                raise ValueError("Invalid LoRA YAML : Multiple Base Concurrency exists in the file")
            elif concurrency["name"] == "Base" or concurrency["name"] == "base":
                base_exists = True

            if (concurrency["name"] == "Base" or concurrency["name"] == "base") and concurrency["adapter_names"]:
                raise ValueError("Invalid LoRA YAML : Base Concurrency must not have any adapter")

            if "quant_updatable_tensors" in concurrency:
                has_quant_updatable_tensors = True

            if concurrency["name"] in use_case_names:
                raise ValueError("Invalid LoRA YAML : use-case name, {}, is not unique".format(concurrency["name"]))
            use_case_names.add(concurrency["name"])

        if has_quant_updatable_tensors:
            log_warning("LoRA YAML contains quant_updatable_tensors field which is deprecated and will be ignored")

        if not base_exists:
            raise ValueError("Invalid LoRA YAML : Base Concurrency must exist in the concurrences/use-cases list")



    @staticmethod
    def parse_config(config_path):
        """
        Parses a lora config and create list of concurrency info object. The config is expected to contain
        information about adapters and concurrences.
        :param config_path: path of the lora config file
        :return: list of concurrency info objects
        """
        log_debug("Parsing Lora Config file {}".format(config_path))

        # validate the LoRA config
        LoraConfigParser.validate(config_path)

        # Load LoRA yaml file
        with open(config_path) as fp:
            lora_config = yaml.safe_load(fp)

        # parse Adapters information from the lora config file
        adapter_info_map = {}
        adapters = lora_config['adapter']
        for adapter in adapters:
            name = adapter['name']
            config = adapter['lora_config']
            with open(config) as fp:
                adapter_config = yaml.safe_load(fp)
            adapter_info = AdapterInfo(
                name=name,
                rank=adapter_config['rank'],
                attach_points=adapter_config['target_operator_names']
            )
            adapter_info_map[name] = adapter_info

        # parse concurrences information from the lora config file
        concurrences = lora_config['use-case']

        concurrency_infos = []
        for concurrency in concurrences:
            name = concurrency['name']
            quant_overrides = None
            if 'quant_overrides' in concurrency:
                quant_overrides = concurrency['quant_overrides']
            if not quant_overrides:
                log_debug("No quant_overrides provided for this concurrency {}".format(name))
            concurrency_info = ConcurrencyInfo(
                name=name,
                adapter_names=concurrency['adapter_names'],
                model=concurrency['model_name'],
                quant_overrides=quant_overrides
            )

            for adapter_name in concurrency_info.adapter_names:
                concurrency_info.adapter_infos[adapter_name] = adapter_info_map[adapter_name]

            concurrency_infos.append(concurrency_info)

        log_debug("Lora Config parsing completed successfully")

        return concurrency_infos


class LoraExplorer(object):
    """
    This is a helper class for parsing ONNX model and finding lora branches in the Graph.
    """

    def __init__(self, onnx_path, load_weights=False):
        # load onnx model for this concurrency
        self.load_weights = load_weights
        model = onnx.load(onnx_path, load_external_data=self.load_weights)
        graph = model.graph

        self.consumers_map = {}
        self.producer_map = {}
        self.nodes_map = {}
        for node in graph.node:
            self.nodes_map[node.name] = node
            inputs = node.input
            for inp in inputs:
                if inp in self.consumers_map:
                    self.consumers_map[inp].append(node)
                else:
                    self.consumers_map[inp] = [node]
            for out in node.output:
                self.producer_map[out] = node

        self.weight_map = {}
        if self.load_weights:
            for weight in model.graph.initializer:
                self.weight_map[weight.name] = to_array(weight)

    def _find_first_add_node(self, node_name):
        """
        Find the first add node after the given node name in the graph
        This Add node is also the last node of the lora branch.
        :param node_name: Name of the Node
        :return Add node and the input of the add node coming from the lora branch.
        """
        curr_node = self.nodes_map[node_name]
        input_name = curr_node.input[0]
        # Iterates and find the first add node in the forward direction.
        while curr_node.op_type != "Add":
            input_name = curr_node.output[0]
            if curr_node.output[0] not in self.consumers_map:
                raise ValueError("Add operation not found in the Lora Branch")
            curr_node = self.consumers_map[curr_node.output[0]][0]

        return curr_node, input_name

    def _find_first_node_backwards(self, tensor_name, type):
        """
        Find the node with the specified op type in the backward direction starting from the specified tensor name.
        :param tensor_name: Name of the tensor
        :param type: Op type of the node
        :return: A NodeProto
        """
        # backtrack and find two conv nodes in the branch
        curr_node = self.producer_map[tensor_name]
        while curr_node.op_type != type:
            if curr_node.input[0] not in self.producer_map:
                raise ValueError("{} operation not found in the Lora Branch".format(type))
            curr_node = self.producer_map[curr_node.input[0]]
        return curr_node

    def get_lora_branch_operations(self, node_name):
        """
        Find the lora branch for the given attach point name
        :param node_name: Attach point node name
        :return: 4 NodeProtos which represents lora A conv node, Mul node, Lora B conv node and Add node
        """
        # Find add node of the Lora branch
        add_node, input_name_attach_branch = self._find_first_add_node(node_name)
        add_node_other_input = add_node.input[0]
        # find the input of the add node which is coming from the LoRA branch.
        # Note: Add node has two inputs. One of the input is from main branch of the graph
        # and other is from the lora branch.
        if add_node_other_input == input_name_attach_branch:
            add_node_other_input = add_node.input[1]

        attach_point_node = self.nodes_map[node_name]

        # TODO: Remove this once converter support matmul op type for lora branch
        if attach_point_node.op_type == "MatMul":
            raise ValueError("MatMul op type is not supported for the Lora Branch {}".format(node_name))

        # Find Mul node in the lora branch
        mul_node = self._find_first_node_backwards(add_node_other_input, "Mul")
        # Find LoRA A and LoRA B node in the LoRA branch
        lora_b_node = self._find_first_node_backwards(add_node_other_input, attach_point_node.op_type)
        lora_a_node = self._find_first_node_backwards(lora_b_node.input[0], attach_point_node.op_type)

        # Check for Bias tensor. Lora branch does not support bias input in conv operations.
        if len(lora_a_node.input) == 3 or len(lora_b_node.input) == 3:
            raise ValueError("Conv Operations in LoRA branch does not support bias inputs")

        return lora_a_node, mul_node, lora_b_node, add_node

    def get_weights(self, name):
        if name in self.weight_map:
            return self.weight_map[name]

    def get_node_type(self, node_name):
        return self.nodes_map[node_name].op_type

    def __del__(self):
        del self.weight_map
        del self.producer_map
        del self.consumers_map
        del self.nodes_map


def find_max_rank(concurrency_infos):
    """
    Calculate max rank per attach point and create AttachPointInfo for each adapter.

    :param concurrency_infos: List of ConcurrencyInfo
    :return: A dict containing AttachPointInfo for each attach point
    """
    log_debug("Calculating Max Rank for each attach points.")
    # max_rank_attach_point stores max rank for each attach point.
    # It also stores the use_case name which has the maximum rank for this attach point.
    # { attach_point_name : (max_rank, use_case)}
    max_rank_attach_point = {}
    # adapter_names_per_attach_point dict store list of adapter names per attach point
    adapter_names_per_attach_point = {}

    # calculate max rank per attach point. Iterate over all the concurrency and
    # calculate per adapter rank for the concurrency. After that update the max rank dictionary.
    # Store adapter names for each attach points
    for concurrency_info in concurrency_infos:
        # rank_attach_point dict store the rank for each attach point in this concurrency
        rank_attach_point = {}
        adapters = concurrency_info.adapter_names
        for adapter in adapters:
            adapter_info = concurrency_info.adapter_infos[adapter]
            for attach_point in adapter_info.attach_points:
                if attach_point not in rank_attach_point:
                    rank_attach_point[attach_point] = adapter_info.rank
                else:
                    rank_attach_point[attach_point] += adapter_info.rank

                if attach_point in adapter_names_per_attach_point:
                    adapter_names_per_attach_point[attach_point].add(adapter)
                else:
                    adapter_names_per_attach_point[attach_point] = set([adapter])

        concurrency_info.attach_point_names = list(rank_attach_point.keys())

        # update max_rank_attach_point dict
        for attach_point in rank_attach_point:
            curr_max_rank = 0
            if attach_point in max_rank_attach_point:
                curr_max_rank = max_rank_attach_point[attach_point][0]
            if rank_attach_point[attach_point] > curr_max_rank:
                max_rank_attach_point[attach_point] = (rank_attach_point[attach_point], concurrency_info.name)

    # create attach point information using the max rank and adapter names per attach point
    attach_point_info = {}

    # Create adapter info for each attach point
    for attach_point_name in max_rank_attach_point:
        # sort the adapter names based on the lexicographical order
        adapter_names = list(adapter_names_per_attach_point[attach_point_name])
        adapter_names = sorted(adapter_names)

        attach_point_info[attach_point_name] = AttachPointInfo(
            attach_point_name,
            max_rank_attach_point[attach_point_name][0],
            max_rank_attach_point[attach_point_name][1],
            adapter_names
        )

    log_debug("Max Rank calculation completed Successfully.")

    return attach_point_info


def find_max_adapters_in_concurrency(concurrency_infos):
    """
    Calculate maximum number of adapters required in the concurrences
    :param concurrency_infos: List of ConcurrencyInfo
    :return: a Integer (maximum number of adapter required)
    """
    max_adapters = 0
    for concurrency_info in concurrency_infos:
        num_adapters = len(concurrency_info.adapter_names)
        max_adapters = max(max_adapters, num_adapters)

    return max_adapters


def extract_lora_tensor_names(concurrency_infos, attach_point_info_map, skip_validation):
    """
    Extra Lora branch tensor names for each concurrency from its corresponding concurrency graphs.
    This function does not returne anything.
    It will update the tensor name in the Attach point info object.
    :param concurrency_infos: list of concurrency info
    :param attach_point_info_map: A dict containing AttachPointInfo for each attach point
    :return: None
    """
    log_debug("Extracting Lora tensor names from the concurrency graphs.")
    for concurrency_info in concurrency_infos:
        concurrency_name = concurrency_info.name
        if concurrency_info.is_base():
            continue
        model_path = concurrency_info.model
        graph_helper = LoraExplorer(model_path)
        attach_point_found = set()

        for node_name in graph_helper.nodes_map.keys():
            base_name = create_base_graph_name(node_name)

            if base_name in concurrency_info.attach_point_names:
                # Extra lora branch node of this attach point
                try:
                    lora_a_node, mul_node, lora_b_node, add_node = (
                        graph_helper.get_lora_branch_operations(node_name))
                except Exception as e:
                    log_error("Invalid Lora Branch: unable to find lora branch operations for the "
                              "attach point {} in use-case {}. Got error {}".format(node_name, concurrency_name, e))
                    raise e

                attach_point_info_map[base_name].op_type = graph_helper.get_node_type(node_name)
                # add extracted tensor names for this concurrency in the attach_point_info
                attach_point_info_map[base_name].tensor_names[concurrency_name] = LoraTensorNames(
                    lora_a_node.input[1],
                    lora_a_node.output[0],
                    mul_node.input[1],
                    mul_node.output[0],
                    lora_b_node.input[1],
                    lora_b_node.output[0],
                    add_node.output[0]
                )
                attach_point_found.add(base_name)

        # Check whether all the attach points required for this concurrency are covered or not.
        remaining_attach_points = set(concurrency_info.attach_point_names) - attach_point_found
        if not skip_validation:
            log_assert(len(remaining_attach_points) == 0,
                   "Following attach points are not present in the {} concurrency graph : {}"
                   .format(concurrency_name, list(remaining_attach_points))
            )
        else:
            # fail if no attach point was found for this concurrency
            log_assert(len(attach_point_found) != 0,
                "Following attach points are not present in the {} concurrency graph : {}"
                .format(concurrency_name, list(remaining_attach_points))
            )
            # if some attach-points were found proceed
            log_warning("Following attach points are not present in the {} concurrency graph : {}"
                    .format(concurrency_name, list(remaining_attach_points)))
            log_warning("Continuing to create max-rank lora model with the attach-points that were found")
            for missing_ap in remaining_attach_points:
                concurrency_info.attach_point_names.remove(missing_ap)
                if missing_ap in attach_point_info_map:
                    del attach_point_info_map[missing_ap]

    log_debug("Lora tensor names extracted successfully.")


def extract_lora_weights(concurrency_infos, attach_point_info_map, output_dir):
    """
    Extra Lora adapter weight from the concurrency graphs
    :param concurrency_infos: list of concurrency info
    :param attach_point_info_map: A dict containing AttachPointInfo for each attach point
    :param output_dir: Directory to save the temporary safetensors file
    :return: Safe tensor file path
    """
    log_debug("Extracting Adapter weights from the Graphs.")

    lora_weights = {}
    adapter_covered = set()
    for concurrency_info in concurrency_infos:
        concurrency_name = concurrency_info.name
        if concurrency_info.is_base():
            continue

        # check whether all the adapter that are present in this concurrency are
        # already extracted or not. If yes, no need to extract the weight again.
        weights_already_extracted = True
        for adapter_name in concurrency_info.adapter_names:
            if adapter_name not in adapter_covered:
                weights_already_extracted = False

        if weights_already_extracted:
            continue

        model_path = concurrency_info.model
        graph_helper = LoraExplorer(model_path, load_weights=True)

        for attach_point_name in concurrency_info.attach_point_names:
            attach_point_info = attach_point_info_map[attach_point_name]
            lora_a_weight_name = attach_point_info.tensor_names[concurrency_name].lora_a_weight_name
            lora_b_weight_name = attach_point_info.tensor_names[concurrency_name].lora_b_weight_name
            lora_a_weight = graph_helper.get_weights(lora_a_weight_name)
            lora_b_weight = graph_helper.get_weights(lora_b_weight_name)
            start = 0
            for adapter_name in concurrency_info.adapter_names:
                if attach_point_name not in concurrency_info.adapter_infos[adapter_name].attach_points:
                    continue
                end = start + concurrency_info.adapter_infos[adapter_name].rank
                new_lora_a_name = attach_point_name + "_lora_A_weight_" + adapter_name
                new_lora_b_name = attach_point_name + "_lora_B_weight_" + adapter_name

                # extract weights for this adapter from the lora A and B weights
                if attach_point_info.op_type == 'Conv':
                    lora_weights[new_lora_a_name] = lora_a_weight[start:end]
                    lora_weights[new_lora_b_name] = lora_b_weight[:, start:end]
                elif attach_point_info.op_type == 'MatMul':
                    lora_weights[new_lora_a_name] = lora_a_weight[:, start:end]
                    lora_weights[new_lora_b_name] = lora_b_weight[start:end]
                else:
                    raise ValueError("Invalid Attach Point {} : "
                                     "op_type for attach_point should be Conv or MatMul but got {}".
                                     format(attach_point_name, attach_point_info.op_type))
                start = end
                # add lora weights names for this adapter in the attach_point_info map.
                # These will be used later to recover adapter weights from the safetensor file.
                attach_point_info.weight_info[adapter_name] = LoraWeightInfo(new_lora_a_name, new_lora_b_name)

        adapter_covered.update(concurrency_info.adapter_names)

        del graph_helper

    # save the weights in the safe tensor file. This file will be deleted later after serialization.
    path = os.path.join(output_dir, "temp_lora.safetensors")
    save_file(lora_weights, path)

    log_debug("Adapter Weights Extracted Successfully.")

    return path


def compute_alpha_scattering_indices(concurrency_infos, attach_point_info_map, max_rank_concurrency):
    """Computes the alpha scattering indices for each concurrency scenario and attach point.
    Args:
        concurrency_infos (list): A list of ConcurrencyInfo objects, each representing a concurrency scenario
        attach_point_info_map (dict): A dictionary mapping attach point names to AttachPointInfo objects
        max_rank_concurrency (int): The maximum rank of the concurrency scenarios
    Returns:
        dict: A dictionary where each key is a concurrency scenario name and the value is a ConcurrencyIndices object
    """
    log_debug("Calculating indices values for the alpha vector.")

    concurrency_indices_map = {}
    for concurrency_info in concurrency_infos:
        concurrency_name = concurrency_info.name
        concurrency_indices = ConcurrencyIndices(concurrency_name)
        for attach_point_name, attach_point_info in attach_point_info_map.items():
            attach_pt_indices = AttachPtIndices()
            max_rank = attach_point_info.max_rank
            adapter_names = concurrency_info.adapter_names
            total_adapter_rank = sum(concurrency_info.adapter_infos[adapter_name].rank for adapter_name in adapter_names if adapter_name in attach_point_info.adapter_names)
            if total_adapter_rank > max_rank:
                raise ValueError(f"Adapter ranks exceed max rank for attach point {attach_point_name}")
            indices = []
            for adapter_name in adapter_names:
                if adapter_name in attach_point_info.adapter_names:
                    adapter_rank = concurrency_info.adapter_infos[adapter_name].rank
                    indices.extend([adapter_names.index(adapter_name)] * adapter_rank)
            padding_index = max_rank_concurrency
            indices.extend([padding_index] * (max_rank - len(indices)))
            attach_pt_indices.alpha_indices = [indices]
            concurrency_indices.attach_pt_indices[attach_point_name] = attach_pt_indices
        concurrency_indices_map[concurrency_name] = concurrency_indices

    log_debug("Indices values calculated successfully.")

    return concurrency_indices_map


def get_base_concurrency_indices(indices_map):
    for key, value in indices_map.items():
        if ConcurrencyInfo.is_base(value):
            return indices_map[key]
