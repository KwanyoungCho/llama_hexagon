# -----------------------------------------------------------------------------
#
# Qualcomm Technologies, Inc. Proprietary
# (c) 2024 Qualcomm Technologies, Inc. All rights reserved.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
# -----------------------------------------------------------------------------

"""Model adaptions for LoRA."""

import copy
from collections import defaultdict

from onnx.helper import make_tensor_value_info
from onnx.onnx_pb import TensorProto

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.optimizer_extension.lora_extension import (
    LoraExtension,
    LoraNode,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.encoding_mapper_utils import (
    AimetEncodingVersion,
    create_tensor_name_to_encodings,
    handle_aimet_v1_activ_encodings,
    update_encodings_to_v1_format,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.logger import (
    log_info,
    log_warning,
)

LORA_MLP_PATTERN_WILDCARD = ["Transpose", "Reshape"]


class LoraModelAdpation:
    """Lora model adaption for non-mha2sha adaption."""

    def __init__(self, prequant_adaption) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            prequant_adaption:
                PreQuantAdaption instance holding the model loader and model info.
        """
        self.prequant_adaption = prequant_adaption
        self.confirmed_lora_alpha_name_list = []
        self.lora_extension = LoraExtension(prequant_adaption)
        self.encodings_map = {"activation_encodings": defaultdict(list), "param_encodings": defaultdict(list)}

        self.lora_alpha_inputs = []

    def search_lora_pattern(self, candidate_add_node):
        # quick check (since lora alpha is also an add node)
        if candidate_add_node.name in self.confirmed_lora_alpha_name_list:
            return None
        lora_nodes = self.lora_extension.verify_and_capture_lora_structure(candidate_add_node)
        if not lora_nodes:
            return None
        for lora_node in lora_nodes:
            if lora_node.lora_alpha is not None:
                self.confirmed_lora_alpha_name_list.append(lora_node.lora_alpha.name)

        return lora_nodes

    def _add_lora_alpha_from_input(self, lora_node: LoraNode):
        """Re-connect lora tensors to remove redundent nodes"""
        lora_alpha_input_name = f"lora_alpha"
        if lora_alpha_input_name not in self.lora_alpha_inputs:
            lora_alpha_input = make_tensor_value_info(lora_alpha_input_name, TensorProto.FLOAT, [1])
            self.prequant_adaption.model.graph.input.append(lora_alpha_input)
            self.lora_alpha_inputs.append(lora_alpha_input_name)

        idx = lora_node.lora_alpha_index
        const_tensor = lora_node.lora_alpha.input[idx]

        lora_node.lora_alpha.input[idx] = lora_alpha_input_name

        # Remove the Mul constant from graph
        try:
            init = self.prequant_adaption.get_initializer_by_name[const_tensor]
            self.prequant_adaption.model.graph.initializer.remove(init)
        except Exception:
            try:
                const_node = self.prequant_adaption.get_node_by_output_name[const_tensor]
                assert const_node.op_type in ["Constant", "Identity"]
                self.prequant_adaption.model.graph.node.remove(const_node)
            except Exception:
                log_warning(f"Unble to remove constant tensor: {const_tensor} from the graph")

    def verify_lora_mul_const_input(self, mul_node):
        mul_scale_tensor = mul_node.input[1]

        if mul_scale_tensor in self.prequant_adaption.get_node_by_output_name.keys():
            scale = self.prequant_adaption.get_node_by_output_name[mul_node.input[1]]
        elif mul_scale_tensor in self.prequant_adaption.get_initializer_by_name.keys():
            scale = self.prequant_adaption.get_initializer_by_name[mul_node.input[1]]
        elif mul_scale_tensor in self.prequant_adaption.mha_model_input_names:
            return False
        else:
            raise ValueError(f"expect mul input have type node or initializer.")

        return True

    def add_lora_alpha_from_input(self):
        add_list = [node for node in self.prequant_adaption.model.graph.node if node.op_type == "Add"]
        visited_add = set()
        all_lora_nodes = []

        for _add in add_list:
            if _add.name in visited_add:
                continue
            visited_add.add(_add.name)
            # Search down stream for LORA_MLP_PATTERN
            lora_nodes = self.search_lora_pattern(_add)
            if lora_nodes:
                all_lora_nodes.extend(lora_nodes)
                for lora_node in lora_nodes:
                    visited_add.add(lora_node.lora_add.name)

                    origin_lora_alpha_name = lora_node.lora_alpha.input[1]
                    self._add_lora_alpha_from_input(lora_node)

                    # record encodings map when lora_alpha is a initializer
                    if origin_lora_alpha_name in self.prequant_adaption.get_initializer_by_name:
                        new_lora_alpha_name = lora_node.lora_alpha.input[1]
                        self.encodings_map["activation_encodings"][origin_lora_alpha_name].append(
                            new_lora_alpha_name
                        )

        log_info(f"Updated {len(all_lora_nodes)} lora to use alpha from model input.")
        return all_lora_nodes

    def get_lora_nodes(self):
        add_list = [node for node in self.prequant_adaption.model.graph.node if node.op_type == "Add"]
        visited_add = set()
        all_lora_nodes = []

        for _add in add_list:
            if _add.name in visited_add:
                continue
            visited_add.add(_add.name)
            # Search down stream for LORA_MLP_PATTERN
            lora_nodes = self.search_lora_pattern(_add)
            if lora_nodes:
                all_lora_nodes.extend(lora_nodes)
                for lora_node in lora_nodes:
                    visited_add.add(lora_node.lora_add.name)

        return all_lora_nodes

    def transform_encodings(self, origin_encodings):
        encoding_version = AimetEncodingVersion(origin_encodings["version"])

        origin_name_to_encodings = origin_encodings
        if encoding_version == AimetEncodingVersion.V1:
            origin_name_to_encodings = create_tensor_name_to_encodings(origin_name_to_encodings)

        new_name_to_encodings = copy.deepcopy(origin_name_to_encodings)

        # add new encodings / modify encodings
        for enc_type in ("activation_encodings", "param_encodings"):
            for origin_name, new_names in self.encodings_map[enc_type].items():
                if origin_name in origin_name_to_encodings[enc_type]:
                    curr_enc = origin_name_to_encodings[enc_type][origin_name]
                else:
                    # mha encodings may not have activation encodings for lora-alpha.
                    # this is a normal case, so ignore it.
                    log_info(
                        f"cannot find encodings of '{origin_name}', "
                        + "so ignore it in prequant encodings transformation"
                    )
                    continue

                if len(new_names) != 1:
                    raise RuntimeError(
                        "currently only one-to-one mapping is supported " + "for prequant-adaption"
                    )

                if encoding_version == AimetEncodingVersion.V1:
                    # even for param encodings, we don't need to split them in this stage,
                    # so calling handle_aimet_v1_activ_encodings is fine
                    curr_enc = handle_aimet_v1_activ_encodings(curr_enc, new_names[0])

                new_name_to_encodings[enc_type][new_names[0]] = curr_enc

        if encoding_version == AimetEncodingVersion.V1:
            _key_values_not_mapped = {
                key: origin_encodings[key]
                for key in origin_encodings.keys()
                if key not in ["param_encodings", "activation_encodings"]
            }
            new_name_to_encodings = update_encodings_to_v1_format(
                new_name_to_encodings, _key_values_not_mapped
            )

        # note: unused encodings are not removed, not mha2sha stage will remove unused encodings
        return new_name_to_encodings
