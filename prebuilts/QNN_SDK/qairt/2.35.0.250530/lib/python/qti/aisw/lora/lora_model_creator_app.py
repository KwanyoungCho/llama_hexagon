# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.lora.graph_creator import *
from qti.aisw.lora.preprocessing import *
from qti.aisw.lora.serialization import *
import os
import onnx
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.converter_ir.op_graph import QuantUpdatableMode


class LoraModelCreatorApp(object):
    def __init__(self, config_path, output_dir, skip_validation, quant_updatable_mode, dump_onnx=False):
        self.config_path = config_path
        self.output_dir = output_dir
        self.path = ""
        self.skip_validation = skip_validation
        self.quant_updatable_mode = QuantUpdatableMode[quant_updatable_mode.upper()]
        self.dump_onnx = dump_onnx

    def get_base_model_path(self, concurrency_infos):
        for concurrency_info in concurrency_infos:
            if concurrency_info.is_base():
                return concurrency_info.model

    def save_base_model(self, base_model):
        self.path = os.path.join(self.output_dir, "base_model.onnx")
        onnx.save(base_model, self.path, save_as_external_data=True,
                  all_tensors_to_one_file=True, location="base_model.data")
        log_info("Updated base onnx model saved at {}".format(self.path))

    def run(self):
        log_info("Executing Lora Creator tool")

        concurrency_infos = LoraConfigParser.parse_config(self.config_path)

        # calculate max rank for concatenated graph
        attach_point_info_map = find_max_rank(concurrency_infos)

        # calculate alpha vector size
        alpha_vector_size = find_max_adapters_in_concurrency(concurrency_infos)

        # extract lora branch names and weight and update attach point information.
        # Save the extracted weights in the safetensor file
        extract_lora_tensor_names(concurrency_infos, attach_point_info_map, self.skip_validation)
        safe_tensor_path = extract_lora_weights(concurrency_infos, attach_point_info_map, self.output_dir)

        # calculate indices for alpha scattering and weight gathering graphs
        indices_map = compute_alpha_scattering_indices(concurrency_infos, attach_point_info_map, alpha_vector_size)

        base_indices_map = get_base_concurrency_indices(indices_map)

        base_graph_path = self.get_base_model_path(concurrency_infos)
        # Create max rank graph
        graph_creator = MaxRankGraphCreator(
            base_graph_path,
            attach_point_info_map,
            base_indices_map,
            alpha_vector_size
        )
        graph_creator.load_graph()
        graph_creator.create_max_rank_lora_graph()
        self.save_base_model(graph_creator.get_graph())
        max_attach_point = graph_creator.attach_pts

        # generate lora creator outputs
        lora_serialzier = LoraSerializer(
            concurrency_infos=concurrency_infos,
            safe_tensor_path=safe_tensor_path,
            indices_map=indices_map,
            attach_point_info_map=attach_point_info_map,
            max_rank_attach_point_map=max_attach_point,
            output_dir=self.output_dir,
            max_graph_path=self.path,
            quant_updatable_mode=self.quant_updatable_mode,
            dump_onnx=self.dump_onnx
        )

        lora_serialzier.serialize()

        # Remove the temporary safe tensor file from the disk
        os.remove(safe_tensor_path)
        log_info("Lora Creator tool execution completed successfully.")
