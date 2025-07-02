# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.accuracy_debugger.lib.inference_engine import inference_engine_repository
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Framework, Engine
from qti.aisw.accuracy_debugger.lib.inference_engine.converters.nd_SNPE_converter import SNPEConverter

from typing import Dict, List

import copy


@inference_engine_repository.register(cls_type=ComponentType.converter, framework=Framework.tflite,
                                      engine=Engine.SNPE, engine_version="1.0.0")
class SNPETfliteConverter(SNPEConverter):

    def __init__(self, context):
        super(SNPETfliteConverter, self).__init__(context)

    def format_input_tensors(self, input_tensors):  # type: (Dict[str][str]) -> Dict[str][str]
        return input_tensors

    def format_output_tensors(self, output_tensors):  # type: (List[str]) -> List[str]
        return output_tensors
