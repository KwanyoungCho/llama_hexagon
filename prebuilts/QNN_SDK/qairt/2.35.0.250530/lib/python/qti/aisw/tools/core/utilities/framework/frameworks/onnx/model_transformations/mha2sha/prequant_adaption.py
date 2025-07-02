# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import onnx

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.transformations.lora_model_adaption import (
    LoraModelAdpation,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.clean import (
    clean_model,
    topological_sort,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils.utils import (
    update_all_mapping_dicts,
)
from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger

mha2sha_pq_log_area = LogAreas.register_log_area("MHA2SHAPreQuant")
mha2sha_pq_logger = QAIRTLogger.register_area_logger(
                mha2sha_pq_log_area, level="INFO", formatter_val="extended", handler_list=["dev_console"]
            )

class PreQuantAdaption:
    """
    PreQuantAdaption class is responsible for making pre-quant model adaptions up to mha2sha.
    """

    def __init__(
        self,
        model: onnx.ModelProto,
        lora_model: bool,
        lora_alpha_from_input: bool,
    ):
        """
        Initialization

        :param model: ModelProto
        :param lora_model: Is lora model
        :param lora_alpha_from_input: Is lora alpha from model input.
        :param mha_conv: is mha use conv instead of matmul in projection
        """

        self.model = model
        self.lora_model = lora_model
        self.lora_alpha_from_input = lora_alpha_from_input
        self.encodings_map = {"activation_encodings": {}, "param_encodings": {}}
        self._update_all_mapping_dicts()  # Initial mapping

    def _update_all_mapping_dicts(self):
        """Helper function to update mappings to nodes.

        Updates all the mapping dictionaries such as `get_initializer_by_name`. These need
        to be updated as nodes are added to the graph and are not yet know.
        """
        update_all_mapping_dicts(self)

    def optimize(self, mha_encodings):
        # TODO: Update encodings mapping for lora
        lora_nodes = []
        if self.lora_model:
            self.lora_model_adaption = LoraModelAdpation(self)

            if self.lora_alpha_from_input:
                # Add lora alpha to model input
                lora_nodes = self.lora_model_adaption.add_lora_alpha_from_input()
                self.encodings_map = self.lora_model_adaption.encodings_map
            else:
                lora_nodes = self.lora_model_adaption.get_lora_nodes()
            if mha_encodings:
                mha_encodings = self.lora_model_adaption.transform_encodings(mha_encodings)
        clean_model(self.model)
        topological_sort(self.model)
        self._update_all_mapping_dicts()

        return mha_encodings, self.encodings_map, lora_nodes

