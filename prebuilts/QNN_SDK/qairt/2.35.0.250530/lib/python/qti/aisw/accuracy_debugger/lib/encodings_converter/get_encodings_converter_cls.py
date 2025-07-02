# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Any

from qti.aisw.accuracy_debugger.lib.encodings_converter.qairt_encodings_converter import QairtEncodingsConverter
from qti.aisw.accuracy_debugger.lib.encodings_converter.qnn_encodings_converter import QnnEncodingsConverter

encodings_converter_cls = {
    'qairt': QairtEncodingsConverter,
    'qnn': QnnEncodingsConverter
}

def get_encodings_converter_cls(type: str) -> Any:
    '''
    :param type: type of encodings converter: {qnn, qairt}
    :return: Class definition of the requried encodings converter
    '''
    return encodings_converter_cls.get(type, None)
