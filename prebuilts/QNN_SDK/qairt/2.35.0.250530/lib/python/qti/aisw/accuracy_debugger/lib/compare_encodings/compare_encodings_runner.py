# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse
import json
import logging
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import xlsxwriter
from qti.aisw.accuracy_debugger.encodings_converter.encodings_utils import (
    EncodingVersion,
    get_encodings_version,
    organize_qairt_encodings,
)
from qti.aisw.accuracy_debugger.encodings_converter.qairt_encodings_converter import (
    QairtEncodingsConverter,
)
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.utils.file_utils import dump_csv, dump_json, read_json
from qti.aisw.accuracy_debugger.utils.graph_utils import (
    get_common_parent_activations,
    get_subgraph,
    get_supergroup_activations,
)
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


DATA_TYPE_NOT_SAME = "Data type not same: {} vs {}"
IS_SYMMETRIC_NOT_SAME = "is_symmetric not same: {} vs {}"
OFFSET_NOT_SAME = "Offset not same: {} vs {} at channel number: {}"
OFFSET_NOT_PRESENT = "Offset not present"
SCALE_NOT_SAME = "Scale not in bound of {}: {} vs {} at channel number: {}"
SCALE_NOT_PRESENT = "Scale not present"
BITWIDTH_NOT_SIMILAR = "Bitwidth not similar: {} vs {}"
INVALID_BITWIDTH = "Bitwidth is invalid"
NUM_CHANNELS_NOT_SAME = "Number of channels are different: {} vs {}"
TWO_FLOAT_ARE_NOT_SAME = "Two float encodings are not same: {} vs {}"


class EncodingInputConfig:
    """Encoding class to be used for passing encoding details to Compare Encodings"""

    def __init__(self, encoding_file_path: str, quantized_dlc_path: str = None):
        self.encoding_file_path = encoding_file_path
        self.quantized_dlc_path = quantized_dlc_path


class Status:
    """Status for encodings comparision"""

    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


VALID_BITWIDTHS = [4, 8, 16, 32]


def updated_error_level(err_level1: Status, err_level2: Status) -> Status:
    """Compare two error level and return the higher error level.

    Args:
        err_level1 (Status): error level1
        err_level2 (Status): error level2

    Returns:
        (str): updated error level
    """
    if Status.ERROR == err_level1 or Status.ERROR == err_level2:
        return Status.ERROR
    elif Status.WARNING == err_level1 or Status.WARNING == err_level2:
        return Status.WARNING

    return Status.SUCCESS


def get_legacy_dtype(encoding: list) -> str:
    """Given an legacy encodings, returns the dtype
    Args:
        encoding (list): legacy encodings for a tensor
    """
    if "dtype" in encoding[0]:
        return encoding[0]["dtype"].lower()
    elif "scale" in encoding[0]:
        if encoding[0]["scale"] == 0:
            return "float"
        else:
            return "int"

    return "float"


def get_legacy_scale_offset(encoding: list) -> tuple[list, list]:
    """Given a legacy encodings, return the list of scales and offsets

    Args:
        encoding (list): legacy encoding

    Returns:
        (list, list): list of scale and offset. lenght of each list will be same as the
            length of the encoding
    """
    scale, offset = [], []

    for channel in encoding:
        scale.append(channel["scale"])
        offset.append(channel["offset"])

    return scale, offset


def compare_scale_offset(
    scale1: list,
    scale2: list,
    offset1: list,
    offset2: list,
    bitwidth1: int,
    bitwidth2: int,
    scale_threshold: float = 1e-3,
) -> tuple[bool, list, Status]:
    """Compares scale and offset of two encodings given the bitwidthds

    Args:
        scale1 (list): list of scales in encoding1
        scale2 (list): list of scales in encoding2
        offset1 (list): list of offsets in encoding1
        offset2 (list): list of offsets in encoding2
        bitwidth1 (int): bitwidth of encoding1
        bitwidth2 (int): bitwidth of encoding2
        scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

    Returns:
        (bool, list, str): Returns the following:
        1. True if they scale and offset of the two encodings are same else False.
        2. list of comparision info
        3. Status of the encodings comparision
    """
    compare_info = []

    # Compare the number of channels
    num_channels1 = len(scale1)
    num_channels2 = len(scale2)
    if num_channels1 != num_channels2:
        compare_info.append(NUM_CHANNELS_NOT_SAME.format(num_channels1, num_channels2))
        return False, compare_info, Status.ERROR

    # Compare scale and offsets
    # Since two encodings with different bitwidths can be algebrically converted into one another
    # we need to compare the scale and offset accordingly by scaling them
    multiplier = pow(2, bitwidth1 - bitwidth2)
    for index, scale in enumerate(zip(scale1, scale2)):
        s1, s2 = scale
        threshold = scale_threshold * min(s1, s2)
        if abs(s1 * multiplier - s2) > threshold:
            compare_info.append(SCALE_NOT_SAME.format(threshold, s1, s2, index))
            break
    for index, offset in enumerate(zip(offset1, offset2)):
        o1, o2 = offset
        if o1 != o2 * multiplier:
            compare_info.append(OFFSET_NOT_SAME.format(o1, o2, index))
            break
    err_level = Status.ERROR if compare_info else Status.SUCCESS

    return len(compare_info) == 0, compare_info, err_level


def compare_dtype(dtype1: str, dtype2: str) -> tuple[bool, list, Status]:
    """Compares dtype of two encodings

    Args:
        dtype1 (str): dtype of encoding1
        dtype2 (str): dtype of encoding2

    Returns:
        (bool, list, str): Returns the following:
            1. True, if both dtype are int else False
            2. list of any comparision info
            3. Status of the encodings comparision
    """
    compare_info = []
    # Case1: both dtypes are not same
    if dtype1 != dtype2:
        compare_info.append(DATA_TYPE_NOT_SAME.format(dtype1, dtype2))
    # Case2: both dtypes are float
    elif dtype1 == "float" and dtype2 == "float":
        compare_info.append(TWO_FLOAT_ARE_NOT_SAME)

    err_level = Status.ERROR if compare_info else Status.SUCCESS

    return len(compare_info) == 0, compare_info, err_level


def compare_v1_encodings(
    encoding1: dict, encoding2: list, scale_threshold: float = 1e-3
) -> tuple[bool, str, Status]:
    """Compares v1 encoding against lgeacy encoding
    Args:
        encoding1 (dict): encoding1 of version 1.0.0
        encoding2 (list): encoding2 of version 1.0.0
        scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

    Returns:
        (bool, str): tuple of:
            1. boolean: True if encoding1 and encoding2 are same else False.
            2. string: any info if both the encodings are same or different
            3. Status of the encodings comparision
    """
    same = True
    info = []
    error_level = Status.SUCCESS

    # ---------------------------------------------------------------------------------------------
    # Compare the dtype
    # ---------------------------------------------------------------------------------------------
    dtype1 = encoding1["dtype"].lower()
    dtype2 = encoding2["dtype"].lower()
    same, compare_info, error_level = compare_dtype(dtype1=dtype1, dtype2=dtype2)
    info.extend(compare_info)

    # ---------------------------------------------------------------------------------------------
    # Compare the is_symmetric
    # ---------------------------------------------------------------------------------------------
    # Compare only if is_sym field are present. Incase of float encodings, it may not be present
    if (
        "is_sym" in encoding1
        and "is_sym" in encoding2
        and str(encoding1["is_sym"]).lower() != encoding2["is_sym"].lower()
    ):
        info.append(IS_SYMMETRIC_NOT_SAME.format(encoding1["is_sym"], encoding2["is_sym"]))
        error_level = updated_error_level(error_level, Status.WARNING)

    # ---------------------------------------------------------------------------------------------
    # Compare the bitwidth, channels, scale, and offset
    # ---------------------------------------------------------------------------------------------
    # If bitwidths are not same, it may not conclude that two encodings being different as
    # encoding with bitwidth b1 can be algebrically converted to encoding with bitwidth b2.
    bitwidth1 = encoding1["bw"]
    bitwidth2 = encoding2["bw"]
    if bitwidth1 != bitwidth2:
        info.append(BITWIDTH_NOT_SIMILAR)
        error_level = updated_error_level(error_level, Status.WARNING)

    # The following scenarios can happen:
    # 1. number of channels are not same => both encodings are not same
    # 2. If bitwidth of both encodings are in valid supported bitwidths.
    # 2.a. If Scale and offsets are present in both encodings then check whether channels, scale,
    #      and offsets are same or not for both encodings
    # 2.b. Scale and offset not present in one of the encodings
    # 3. Either of the bitwidth is not in valid bitwidth
    if bitwidth1 in VALID_BITWIDTHS and bitwidth2 in VALID_BITWIDTHS:
        if (
            "scale" in encoding1
            and "offset" in encoding1
            and "scale" in encoding2
            and "offset" in encoding2
        ):
            same, compare_info, err_level = compare_scale_offset(
                scale1=encoding1["scale"],
                scale2=encoding2["scale"],
                offset1=encoding1["offset"],
                offset2=encoding2["offset"],
                bitwidth1=bitwidth1,
                bitwidth2=bitwidth2,
                scale_threshold=scale_threshold,
            )
            info.extend(compare_info)
            error_level = updated_error_level(error_level, err_level)
        elif ("offset" in encoding1 and "scale" in encoding1) != (
            "offset" in encoding2 and "scale" in encoding2
        ):
            same = False
            info.append(OFFSET_NOT_PRESENT)
            info.append(SCALE_NOT_PRESENT)
            error_level = Status.ERROR
    else:
        same = False
        error_level = Status.ERROR
        info.append(INVALID_BITWIDTH)

    return same, info, error_level


def compare_v1_legacy_encodings(
    encoding1: dict, encoding2: list, scale_threshold: float = 1e-3
) -> tuple[bool, str, Status]:
    """Compares v1 encoding against lgeacy encoding
    Args:
        encoding1 (dict): encoding1 of version 1.0.0
        encoding2 (list): encoding2 of legacy version
        scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

    Returns:
        (bool, str): tuple of:
            1. boolean: True if encoding1 and encoding2 are same else False.
            2. string: any info if both the encodings are same or different
            3. Status of the encodings comparision
    """
    same = True
    info = []
    error_level = Status.SUCCESS

    # ---------------------------------------------------------------------------------------------
    # Compare the dtype
    # ---------------------------------------------------------------------------------------------
    # In V1 encodings, we have dtype field but in legacy encodings we have it only if encodings
    # was float.
    dtype1 = encoding1["dtype"].lower()
    dtype2 = get_legacy_dtype(encoding2)
    same, compare_info, error_level = compare_dtype(dtype1=dtype1, dtype2=dtype2)
    info.extend(compare_info)

    # ---------------------------------------------------------------------------------------------
    # Compare the is_symmetric
    # ---------------------------------------------------------------------------------------------
    # Compare only if is_sym field are present. Incase of float encodings, it may not be present
    if (
        "is_sym" in encoding1
        and "is_symmetric" in encoding2[0]
        and str(encoding1["is_sym"]).lower() != encoding2[0]["is_symmetric"].lower()
    ):
        info.append(IS_SYMMETRIC_NOT_SAME.format(encoding1["is_sym"], encoding2[0]["is_symmetric"]))
        error_level = updated_error_level(error_level, Status.WARNING)

    # ---------------------------------------------------------------------------------------------
    # Compare the bitwidth, channels, scale, and offset
    # ---------------------------------------------------------------------------------------------
    # If bitwidths are not same, it may not conclude that two encodings being different as
    # encoding with bitwidth b1 can be algebrically converted to encoding with bitwidth b2.
    bitwidth1 = encoding1["bw"]
    bitwidth2 = encoding2[0]["bitwidth"]
    if bitwidth1 != bitwidth2:
        info.append(BITWIDTH_NOT_SIMILAR.format(bitwidth1, bitwidth2))
        error_level = updated_error_level(error_level, Status.WARNING)

    # The following scenarios can happen:
    # 1. number of channels are not same => both encodings are not same
    # 2. If bitwidth of both encodings are in valid supported bitwidths.
    # 2.a. If Scale and offsets are present in both encodings then check whether channels, scale,
    #      and offsets are same or not for both encodings
    # 2.b. Scale and offset not present in one of the encodings
    # 3. Either of the bitwidth is not in valid bitwidth
    if bitwidth1 in VALID_BITWIDTHS and bitwidth2 in VALID_BITWIDTHS:
        if (
            "scale" in encoding1
            and "offset" in encoding1
            and "scale" in encoding2[0]
            and "offset" in encoding2[0]
        ):
            scale2, offset2 = get_legacy_scale_offset(encoding2)

            same, compare_info, err_level = compare_scale_offset(
                scale1=encoding1["scale"],
                scale2=scale2,
                offset1=encoding1["offset"],
                offset2=offset2,
                bitwidth1=bitwidth1,
                bitwidth2=bitwidth2,
                scale_threshold=scale_threshold,
            )
            info.extend(compare_info)
            error_level = updated_error_level(error_level, err_level)
        elif ("offset" in encoding1 and "scale" in encoding1) != (
            "offset" in encoding2[0] and "scale" in encoding2[0]
        ):
            same = False
            error_level = Status.ERROR
            info.append(OFFSET_NOT_PRESENT)
            info.append(SCALE_NOT_PRESENT)
    else:
        same = False
        error_level = Status.ERROR
        info.append(INVALID_BITWIDTH)

    return same, info, error_level


def compare_legacy_v1_encodings(
    encoding1: dict, encoding2: list, scale_threshold: float = 1e-3
) -> tuple[bool, str, Status]:
    """Compares v1 encoding against lgeacy encoding
    Args:
        encoding1 (dict): encoding1 of legacy version
        encoding2 (list): encoding2 of version v1
        scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

    Returns:
        (bool, str): tuple of:
            1. boolean: True if encoding1 and encoding2 are same else False.
            2. string: any info if both the encodings are same or different
            3. Status of the encodings comparision
    """
    same, info, error_level = compare_v1_legacy_encodings(
        encoding1=encoding2, encoding2=encoding1, scale_threshold=scale_threshold
    )

    return same, info, error_level


def compare_legacy_encodings(
    encoding1: dict, encoding2: list, scale_threshold: float = 1e-3
) -> tuple[bool, str, Status]:
    """Compares v1 encoding against lgeacy encoding
    Args:
        encoding1 (dict): encoding1 of legacy version
        encoding2 (list): encoding2 of legacy version
        scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

    Returns:
        (bool, str): tuple of:
            1. boolean: True if encoding1 and encoding2 are same else False.
            2. string: any info if both the encodings are same or different
            3. Status of the encodings comparision
    """
    same = True
    info = []
    error_level = Status.SUCCESS

    # ---------------------------------------------------------------------------------------------
    # Compare the dtype
    # ---------------------------------------------------------------------------------------------
    dtype1 = get_legacy_dtype(encoding1)
    dtype2 = get_legacy_dtype(encoding2)
    same, compare_info, error_level = compare_dtype(dtype1=dtype1, dtype2=dtype2)
    info.extend(compare_info)

    # ---------------------------------------------------------------------------------------------
    # Compare the is_symmetric
    # ---------------------------------------------------------------------------------------------
    # Compare only if is_sym field are present. Incase of float encodings, it may not be present
    if (
        "is_symmetric" in encoding1[0]
        and "is_symmetric" in encoding2[0]
        and str(encoding1[0]["is_symmetric"]).lower() != encoding2[0]["is_symmetric"].lower()
    ):
        info.append(
            IS_SYMMETRIC_NOT_SAME.format(encoding1[0]["is_symmetric"], encoding2[0]["is_symmetric"])
        )
        error_level = updated_error_level(error_level, Status.WARNING)

    # ---------------------------------------------------------------------------------------------
    # Compare the bitwidth, channels, scale, and offset
    # ---------------------------------------------------------------------------------------------
    # If bitwidths are not same, it may not conclude that two encodings being different as
    # encoding with bitwidth b1 can be algebrically converted to encoding with bitwidth b2.
    bitwidth1 = encoding1[0]["bitwidth"]
    bitwidth2 = encoding2[0]["bitwidth"]
    if bitwidth1 != bitwidth2:
        info.append(BITWIDTH_NOT_SIMILAR.format(bitwidth1, bitwidth2))
        error_level = updated_error_level(error_level, Status.WARNING)

    # The following scenarios can happen:
    # 1. number of channels are not same => both encodings are not same
    # 2. If bitwidth of both encodings are in valid supported bitwidths.
    # 2.a. If Scale and offsets are present in both encodings then check whether channels, scale,
    #      and offsets are same or not for both encodings
    # 2.b. Scale and offset not present in one of the encodings
    # 3. Either of the bitwidth is not in valid bitwidth
    if bitwidth1 in VALID_BITWIDTHS and bitwidth2 in VALID_BITWIDTHS:
        if (
            "offset" in encoding1[0]
            and "scale" in encoding1[0]
            and "offset" in encoding2[0]
            and "scale" in encoding2[0]
        ):
            scale1, offset1 = get_legacy_scale_offset(encoding1)
            scale2, offset2 = get_legacy_scale_offset(encoding2)

            same, compare_info, err_level = compare_scale_offset(
                scale1=scale1,
                scale2=scale2,
                offset1=offset1,
                offset2=offset2,
                bitwidth1=bitwidth1,
                bitwidth2=bitwidth2,
                scale_threshold=scale_threshold,
            )
            info.extend(compare_info)
            error_level = updated_error_level(error_level, err_level)
        elif ("offset" in encoding1[0] and "scale" in encoding1[0]) != (
            "offset" in encoding2[0] and "scale" in encoding2[0]
        ):
            same = False
            error_level = Status.ERROR
            info.append(OFFSET_NOT_PRESENT)
            info.append(SCALE_NOT_PRESENT)

    else:
        same = False
        error_level = Status.ERROR
        info.append(INVALID_BITWIDTH)

    return same, info, error_level


comparators = {
    EncodingVersion.LEGACY: compare_legacy_encodings,
    EncodingVersion.V1: compare_v1_encodings,
    "1.0.0_legacy": compare_v1_legacy_encodings,
    "legacy_1.0.0": compare_legacy_v1_encodings,
    "legacy_legacy": compare_legacy_encodings,
    "1.0.0_1.0.0": compare_v1_encodings,
}


class CompareEncodings:
    """Implements CompareEncodings base class"""

    def __init__(self, logger: logging.Logger = None) -> None:
        """Initializes the Compare Encodings class

        Args:
            logger: Python Logger object
        """
        if logger is None:
            log_area = LogAreas.register_log_area("Compare Encodings")
            self._logger = QAIRTLogger.register_area_logger(
                area=log_area, level="INFO", formatter_val="simple", handler_list=["dev_console"]
            )
        else:
            self._logger = logger

    def _execute_encodings_converter(
        self, encoding: EncodingInputConfig, framework_model_path: str, output_dir: str
    ) -> tuple[QairtEncodingsConverter, dict]:
        """Creates, executes encodings converter and dumps the converted encodings

        Args:
            encoding (EncodingInputConfig): encoding config for qairt encodings
            framework_model_path (str): path to the framework model
            output_dir (str): path to the output directory to dump converted encodings

        Returns:
            (QairtEncodingsConverter, dict): Following things will be returned:
                1. encoding converte object
                2. converted encoding json
        """
        encoding_converter = QairtEncodingsConverter(
            framework_model_path=framework_model_path,
            dlc_path=encoding.quantized_dlc_path,
            qairt_encodings_file_path=encoding.encoding_file_path,
            working_dir=output_dir,
            logger=self._logger,
        )
        encoding_json = encoding_converter.create_subgraph_encodings()
        converted_encoding_path = os.path.join(output_dir, "encoding.json")
        dump_json(encoding_json, converted_encoding_path)

        return encoding_converter, encoding_json

    def run(
        self,
        primary_encoding: EncodingInputConfig,
        reference_encoding: EncodingInputConfig,
        output_dir: str,
        framework_model_path: str = None,
        scale_threshold: float = 1e-3,
    ):
        """Compares primary_encoding against reference encoding and vice-versa

        Args:
            #TODO: Change name to config, name to encoding1 and encoding2
            primary_encoding (Encoding): Aimet or Qairt encoding details
            reference_encoding (Encoding): Aimet or Qairt encoding details
            output_dir (str): path to the output directory to dump the analysis results
            framework_model_path (str | None): path to the framework model.
                Required if use_encodings_converter is passed True
            use_encodings_converter (bool): If passed True, it performs following operations on the
                qairt encodings file:
                1.  Propagates convert_ops encodings to the its parent op considering the fact that
                    parent op exists in the framework model
                2.  Resolves any activation name changes done. For e.g. matmul+add in framework
                    model becomes fc in the dlc graph and the tensor name gets _fc suffix.
                When passed True, it maps each activation in reference_encoding to a supergroup in
                primary encoding
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3
        """
        if primary_encoding.quantized_dlc_path and framework_model_path:
            primary_ec_dir = os.path.join(output_dir, "primary_encoding")
            primary_ec, primary_encoding_json = self._execute_encodings_converter(
                encoding=primary_encoding,
                framework_model_path=framework_model_path,
                output_dir=primary_ec_dir,
            )
        else:
            primary_ec = None
            primary_encoding_json = read_json(primary_encoding.encoding_file_path)

        filename1 = os.path.basename(primary_encoding.encoding_file_path)

        if reference_encoding.quantized_dlc_path and framework_model_path:
            reference_ec_dir = os.path.join(output_dir, "reference_encoding")
            _, reference_encoding_json = self._execute_encodings_converter(
                encoding=reference_encoding,
                framework_model_path=framework_model_path,
                output_dir=reference_ec_dir,
            )
        else:
            reference_encoding_json = read_json(reference_encoding.encoding_file_path)

        filename2 = os.path.basename(reference_encoding.encoding_file_path)

        self._compare(
            primary_encodings=primary_encoding_json,
            reference_encodings=reference_encoding_json,
            output_dir=output_dir,
            scale_threshold=scale_threshold,
            filename1=filename1,
            filename2=filename2,
        )

        if primary_ec:
            self._map_supergroup(
                encodings_converter=primary_ec,
                reference_encodings=reference_encoding_json,
                output_dir=output_dir,
            )

    def _compare_encodings(
        self,
        primary_encodings: dict,
        reference_encodings: dict,
        comparator_key: str,
        scale_threshold: float = 1e-3,
    ) -> dict:
        """Comapres each tensor in primary_encodings against each tensor in reference_encodings.

        Args:
            primary_encodings (dict): organized (aimet or qairt) (param or activation) encodings
            reference_encodings (dict): organized (aimet or qairt) (param or activation) encodings
            comparator_key (str): comparator key to compare two encodings
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

        Returns:
            dict: dictionary of comparision between each tensor in primary encodings againts each
                tensor in reference encodings. Two tensors are mapped iff:
                1. They share same algebrically similar encodings. One encoding can be algebrically
                   converted to other and vice-versa;
                2. They share same name.
        """
        comparison = {}

        for primary_activation in primary_encodings:
            comparison[primary_activation] = {}
            comparison[primary_activation]["compare_info"] = {}
            comparison[primary_activation]["Status"] = {}
            comparison[primary_activation]["Mapping"] = []
            for reference_activation in reference_encodings:
                same, compare_info, error_level = comparators[comparator_key](
                    encoding1=primary_encodings[primary_activation],
                    encoding2=reference_encodings[reference_activation],
                    scale_threshold=scale_threshold,
                )
                if same:
                    comparison[primary_activation]["compare_info"][reference_activation] = "|".join(
                        compare_info
                    )
                    comparison[primary_activation]["Mapping"].append(reference_activation)
                    comparison[primary_activation]["Status"][reference_activation] = error_level
                # If the activation names are same but encodings are not
                elif primary_activation == reference_activation:
                    if TWO_FLOAT_ARE_NOT_SAME in compare_info:
                        compare_info.remove(TWO_FLOAT_ARE_NOT_SAME)
                    comparison[primary_activation]["compare_info"][reference_activation] = "|".join(
                        compare_info
                    )
                    comparison[primary_activation]["Mapping"].append(reference_activation)
                    comparison[primary_activation]["Status"][reference_activation] = error_level

            if not comparison[primary_activation]["Status"]:
                comparison[primary_activation]["Status"] = "Not Mapped"

        return comparison

    def _compare_params(
        self,
        primary_encodings: dict,
        reference_encodings: dict,
        primary_encodings_version: EncodingVersion,
        reference_encodings_version: EncodingVersion,
        scale_threshold: float = 1e-3,
    ) -> tuple[dict, dict]:
        """Comapres the param encodings

        Args:
            primary_encodings (dict): organized (qairt or aimet) encodings
            reference_encodings (dict): organized (qairt or aimet) encodings
            primary_encodings_version (EncodingVersion): Encoding version of primary encodings
            reference_encodings_version (EncodingVersion): Encoding version of reference encodings
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

        Returns:
            (dict, dict): dictionary of primary param comparision and reference param comparision
        """
        primary_param_comparision = {}
        reference_param_comparision = {}

        # -----------------------------------------------------------------------------------------
        # Compare param encodings
        # -----------------------------------------------------------------------------------------
        # a. PRIMARY -> REFERENCE
        self._logger.info("Comparing primary param encodings against reference param encodings")
        primary_param_comparision = self._compare_encodings(
            primary_encodings=primary_encodings["param_encodings"],
            reference_encodings=reference_encodings["param_encodings"],
            comparator_key=f"{primary_encodings_version.value}_{reference_encodings_version.value}",
            scale_threshold=scale_threshold,
        )
        self._logger.info(
            "Primary param encodings comparision against reference param encodings done"
        )

        # b. REFERENCE -> PRIMARY
        self._logger.info("Comparing reference param encodings against primary param encodings")
        reference_param_comparision = self._compare_encodings(
            primary_encodings=reference_encodings["param_encodings"],
            reference_encodings=primary_encodings["param_encodings"],
            comparator_key=f"{reference_encodings_version.value}_{primary_encodings_version.value}",
            scale_threshold=scale_threshold,
        )
        self._logger.info(
            "Reference param encodings comparision against primary param encodings done"
        )

        return primary_param_comparision, reference_param_comparision

    def _compare_activations(
        self,
        primary_encodings: dict,
        reference_encodings: dict,
        primary_encodings_version: EncodingVersion,
        reference_encodings_version: EncodingVersion,
        scale_threshold: float = 1e-3,
    ) -> tuple[dict, dict]:
        """Comapres the param encodings

        Args:
            primary_encodings (dict): organized (qairt or aimet) encodings
            reference_encodings (dict): organized (qairt or aimet) encodings
            primary_encodings_version (EncodingVersion): Encoding version of primary encodings
            reference_encodings_version (EncodingVersion): Encoding version of reference encodings
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3

        Returns:
            (dict, dict): dictionary of primary activation comparision and reference activation
                comparision
        """
        primary_activation_comparision = {}
        reference_activation_comparision = {}
        # -----------------------------------------------------------------------------------------
        # Compare activation encodings
        # -----------------------------------------------------------------------------------------
        # a. PRIMARY -> REFERENCE
        self._logger.info(
            "Comparing primary activation encodings against reference activation encodings"
        )
        primary_activation_comparision = self._compare_encodings(
            primary_encodings=primary_encodings["activation_encodings"],
            reference_encodings=reference_encodings["activation_encodings"],
            comparator_key=f"{primary_encodings_version.value}_{reference_encodings_version.value}",
            scale_threshold=scale_threshold,
        )
        self._logger.info(
            "Primary activation encodings comparision against reference activation encodings done"
        )

        # b. REFERENCE -> PRIMARY
        self._logger.info(
            "Comparing reference activation encodings against primary activation encodings"
        )
        reference_activation_comparision = self._compare_encodings(
            primary_encodings=reference_encodings["activation_encodings"],
            reference_encodings=primary_encodings["activation_encodings"],
            comparator_key=f"{reference_encodings_version.value}_{primary_encodings_version.value}",
            scale_threshold=scale_threshold,
        )
        self._logger.info(
            "Reference activation encodings comparision against primary activation encodings done"
        )

        return primary_activation_comparision, reference_activation_comparision

    def _compare(
        self,
        primary_encodings: dict,
        reference_encodings: dict,
        output_dir: str,
        filename1: str,
        filename2: str,
        scale_threshold: float = 1e-3,
    ) -> None:
        """Compares primary_encodings against the reference encodings.

        Args:
            primary_encodings (dict): qairt or aimet encodings
            reference_encodings (dict): qairt or aimet encodings
            output_dir(str): path to output directory where analysis will be dumped
            scale_threshold (float): threshold for scale comparision of two encodings. Default: 1e-3
        """
        # TODO: do all this in EncodingInputConfig class
        organized_primary_encodings = organize_qairt_encodings(primary_encodings)
        organized_reference_encodings = organize_qairt_encodings(reference_encodings)
        primary_encodings_version = get_encodings_version(primary_encodings)
        reference_encodings_version = get_encodings_version(reference_encodings)

        # Compare params
        self._logger.info("Starting param encodings comparision.")
        primary_param_comparision, reference_param_comparision = self._compare_params(
            primary_encodings=organized_primary_encodings,
            reference_encodings=organized_reference_encodings,
            primary_encodings_version=primary_encodings_version,
            reference_encodings_version=reference_encodings_version,
            scale_threshold=scale_threshold,
        )
        self._logger.info("Param encodings comprision done.")

        # Compare activations
        self._logger.info("Starting activation encodings comparision.")
        primary_activation_comparision, reference_activation_comparision = (
            self._compare_activations(
                primary_encodings=organized_primary_encodings,
                reference_encodings=organized_reference_encodings,
                primary_encodings_version=primary_encodings_version,
                reference_encodings_version=reference_encodings_version,
                scale_threshold=scale_threshold,
            )
        )
        self._logger.info("Activation encodings comprision done.")

        self._dump_analysis(
            primary_param_comparision,
            reference_param_comparision,
            primary_activation_comparision,
            reference_activation_comparision,
            output_dir,
            filename1,
            filename2,
        )

    def _dump_csv_analysis(
        self, comparision1: dict, comparision2: dict, csv_path: str, filename1: str, filename2: str
    ) -> None:
        """Builds and dumps csv dataframe for the given comparision info

        Args:
            comparision1 (dict): comparision info of comparing encoding1 against encoding2
            comparision2 (dict): comparision info of comparing encoding2 against encoding1
            csv_path (str): path to the csv file
        """
        tensor_name1 = f"Tensor Name({filename1})"
        tensor_name2 = f"Tensor Name({filename2})"
        columns = [
            tensor_name1,
            tensor_name2,
            "Status",
            "Total Mappings",
            "Info",
        ]
        data_frame = {column: [] for column in columns}
        visited_tensors = []
        for tensor_name, tensor_comparision_info in comparision1.items():
            visited_tensors.append(tensor_name)
            data_frame[tensor_name1].append(tensor_name)
            mappings = tensor_comparision_info["Mapping"]
            data_frame["Total Mappings"].append(len(mappings))
            if not mappings:
                data_frame[tensor_name2].append(None)
                data_frame["Status"].append(tensor_comparision_info["Status"])
                data_frame["Info"].append(None)
            elif tensor_name in mappings:
                data_frame[tensor_name2].append(tensor_name)
                data_frame["Status"].append(tensor_comparision_info["Status"][tensor_name])
                data_frame["Info"].append(tensor_comparision_info["compare_info"][tensor_name])
            else:
                data_frame[tensor_name2].append(mappings[0])
                data_frame["Status"].append(tensor_comparision_info["Status"][mappings[0]])
                data_frame["Info"].append(tensor_comparision_info["compare_info"][mappings[0]])

        for tensor_name, tensor_comparision_info in comparision2.items():
            if tensor_name not in visited_tensors:
                visited_tensors.append(tensor_name)
                data_frame[tensor_name2].append(tensor_name)
                mappings = tensor_comparision_info["Mapping"]
                data_frame["Total Mappings"].append(len(mappings))
                if not mappings:
                    data_frame[tensor_name1].append(None)
                    data_frame["Status"].append(tensor_comparision_info["Status"])
                    data_frame["Info"].append(None)
                elif tensor_name in mappings:
                    data_frame[tensor_name1].append(tensor_name)
                    data_frame["Status"].append(tensor_comparision_info["Status"][tensor_name])
                    data_frame["Info"].append(tensor_comparision_info["compare_info"][tensor_name])
                else:
                    data_frame[tensor_name1].append(mappings[0])
                    data_frame["Status"].append(tensor_comparision_info["Status"][mappings[0]])
                    data_frame["Info"].append(tensor_comparision_info["compare_info"][mappings[0]])

        dump_csv(data_frame=data_frame, csv_path=csv_path)

    def _dump_analysis(
        self,
        primary_param_comparision: dict,
        reference_param_comparision: dict,
        primary_activation_comparision: dict,
        reference_activation_comparision: dict,
        output_dir: str,
        filename1: str,
        filename2: str,
    ) -> None:
        """Dumps the compare encodings analysis

        Args:
            primary_param_comparision (dict): comparision of primary param encodings against
                reference param encodings
            reference_param_comparision (dict): comparision of reference param encodings against
                primary param encodings
            primary_activation_comparision (dict): comparision of primary activation encodings
                against reference activation encodings
            reference_activation_comparision (dict): comparision of reference activation encodings
                against primary activation encodings
            output_dir(str): path to output directory where analysis will be dumped
        """
        primary_param_comparision_path = os.path.join(
            output_dir, f"{filename1}_param_comparision.json"
        )
        dump_json(primary_param_comparision, primary_param_comparision_path)

        reference_param_comparision_path = os.path.join(
            output_dir, f"{filename2}_param_comparision.json"
        )
        dump_json(reference_param_comparision, reference_param_comparision_path)
        param_csv_path = os.path.join(output_dir, "param_comparision.csv")
        self._dump_csv_analysis(
            comparision1=primary_param_comparision,
            comparision2=reference_param_comparision,
            csv_path=param_csv_path,
            filename1=filename1,
            filename2=filename2,
        )

        primary_activation_comparision_path = os.path.join(
            output_dir, f"{filename1}_activation_comparision.json"
        )
        dump_json(primary_activation_comparision, primary_activation_comparision_path)

        reference_activation_comparision_path = os.path.join(
            output_dir, f"{filename2}_activation_comparision.json"
        )
        dump_json(reference_activation_comparision, reference_activation_comparision_path)
        activation_csv_path = os.path.join(output_dir, "activation_comparision.csv")
        self._dump_csv_analysis(
            comparision1=primary_activation_comparision,
            comparision2=reference_activation_comparision,
            csv_path=activation_csv_path,
            filename1=filename1,
            filename2=filename2,
        )

    def _map_supergroup(
        self,
        encodings_converter: QairtEncodingsConverter,
        reference_encodings: dict,
        output_dir: str,
    ) -> None:
        """Map each activation in reference encodings to a supergroup in qairt_encodings.
        Dumps the mapping in output directory.

        Args:
            encodings_converter (QairtEncodingsConverter): encodings converter object for
                created with qairt_encodings
            reference_encodings (dict): aimet or qairt encodings
            output_dir(str): path to output directory where analysis will be dumped
        """
        organized_reference_encodings = organize_qairt_encodings(reference_encodings)

        # Get the target(Qairt) supergroups
        framework_activation_op_map = encodings_converter.get_framework_activation_op_map()
        target_activation_op_map = encodings_converter.get_target_activation_op_map()
        conv_bn_relu_activations = get_supergroup_activations(
            framework_activation_op_map=framework_activation_op_map,
            target_activation_op_map=target_activation_op_map,
        )
        model_inputs = set()
        for activation, target_op in target_activation_op_map.items():
            if target_op.parent_ops():
                model_inputs.update([activation])

        target_activations = (
            target_activation_op_map.keys() - conv_bn_relu_activations
        ) - model_inputs
        supergroup_target_activation_map = {}
        target_activation_supergroup_map = {}
        for idx, target_activation in enumerate(target_activations):
            target_op = target_activation_op_map[target_activation]
            parent_activations = set()

            for op_input in target_op.inputs:
                partial_parent_activations = get_common_parent_activations(
                    current_activation=op_input,
                    graph1_activation_op_map=target_activation_op_map,
                    graph2_activation_op_map=framework_activation_op_map,
                    ignore_activations=conv_bn_relu_activations,
                )
                parent_activations.update(partial_parent_activations)

            target_supergroup_activations = get_subgraph(
                subgraph_input_names=parent_activations,
                subgraph_output_names=target_op.outputs,
                graph1_activation_op_map=target_activation_op_map,
                graph2_activation_op_map=framework_activation_op_map,
            )
            supergroup_name = f"supergroup_{idx}"
            supergroup_target_activation_map[supergroup_name] = target_supergroup_activations
            for supergroup_activation in target_supergroup_activations:
                target_activation_supergroup_map[supergroup_activation] = supergroup_name

        # Map each reference activation to some supergroup in target graph
        reference_supergroup_mapping = {}
        for reference_activation in organized_reference_encodings["activation_encodings"]:
            if reference_activation in target_activation_supergroup_map:
                supergroup_name = target_activation_supergroup_map[reference_activation]
                reference_supergroup_mapping[reference_activation] = (
                    supergroup_target_activation_map[supergroup_name]
                )
            else:
                reference_supergroup_mapping[reference_activation] = "Not Mapped"

        mapping_path = os.path.join(output_dir, "reference_supergroup_mapping.json")
        dump_json(reference_supergroup_mapping, mapping_path)


class CompareEncodingsRunner(object):
    def __init__(self, logger, args):
        # type: (Logger, namespace) -> None

        self.args = args
        self._logger = logger
        self.encoding_diff_path = os.path.join(args.output_dir, "encodings_diff.xlsx")
        self.extracted_encodings_path = os.path.join(args.output_dir, "extracted_encodings.json")
        self.filtered_encodings_path = os.path.join(args.output_dir, "filtered_encodings.json")
        self.engine_type = None

    def run(self, engine_type):
        self.engine_type = engine_type
        self._logger.info(f"Arguments received to encodings comparison tool: {self.args}")

        self._logger.info(f"PATH: {self.args.input.endswith}")

        if self.engine_type == Engine.QNN.value:
            self.compare_encodings_qnn()
        elif self.engine_type == Engine.SNPE.value or (
            self.engine_type == Engine.QAIRT.value and self.args.input.endswith(".dlc")
        ):
            self.compare_encodings_snpe()
        elif self.engine_type == Engine.QAIRT.value and self.args.input.endswith(".json"):
            self._compare_encodings_qairt()
        else:
            raise ParameterError(
                f"Given engine type {self.engine_type} does not support Compare encodings feature."
            )

    def _compare_encodings_qairt(self):
        primary_encoding = EncodingInputConfig(encoding_file_path=self.args.input)
        reference_encoding = EncodingInputConfig(encoding_file_path=self.args.aimet_encodings_json)
        comapre_encodings = CompareEncodings(logger=self._logger)
        comapre_encodings.run(
            primary_encoding=primary_encoding,
            reference_encoding=reference_encoding,
            output_dir=self.args.output_dir,
            framework_model_path=None,
        )

    def check_missing_encodings(self, extracted_encodings=None, aimet_encodings=None):
        """
        Helper function to find encodings present in AIMET but not in Target(QNN/SNPE) and vice-versa
        """
        self._logger.info(
            f"Finding encodings present only in AIMET encodings but not in {self.engine_type} encodings:"
        )
        for enc_type in aimet_encodings:
            if enc_type in extracted_encodings:
                self._logger.info(f"Checking {enc_type}...")
                for layer in aimet_encodings[enc_type]:
                    if all(
                        alias not in extracted_encodings[enc_type]
                        for alias in [layer, layer + "_permute"]
                    ):
                        self._logger.warning(f"{layer} present only in AIMET encodings")
            else:
                self._logger.warning(f"{enc_type} present only in AIMET encodings")

        self._logger.info(
            f"Finding encodings present only in {self.engine_type} encodings but not in AIMET encodings:"
        )
        for enc_type in extracted_encodings:
            if enc_type in aimet_encodings:
                self._logger.info(f"Checking {enc_type}...")
                for layer in extracted_encodings[enc_type]:
                    if all(
                        alias not in aimet_encodings[enc_type]
                        for alias in [layer, layer.replace("_permute", "")]
                    ):
                        self._logger.warning(
                            f"{layer} present only in {self.engine_type} encodings"
                        )
            else:
                self._logger.warning(f"{enc_type} present only in {self.engine_type} encodings")

    def compare_encodings_qnn(self):
        extracted_encodings = self.extract_model_net_encodings()
        with open(self.extracted_encodings_path, "w") as json_write:
            json.dump(extracted_encodings, json_write, indent=4)
        aimet_encodings, sanitize_usantize_map = self.get_aimet_encodings()
        # Filter the extracted encodings from model_net_json
        filtered_encodings = self.filter_encodings(extracted_encodings, sanitize_usantize_map)
        # Dump the filtered encodings in json file
        with open(self.filtered_encodings_path, "w") as json_write:
            json.dump(filtered_encodings, json_write, indent=4)
        sanitized_filtered_encodings = {}
        sanitized_filtered_encodings["activation_encodings"] = {}
        sanitized_filtered_encodings["param_encodings"] = {}
        for encodings in filtered_encodings.keys():
            if encodings == "activation_encodings" or encodings == "param_encodings":
                for layer in filtered_encodings[encodings].keys():
                    sanitized_filtered_encodings[encodings][santize_node_name(layer)] = (
                        filtered_encodings[encodings][layer]
                    )
        self.generate_excel_sheet(aimet_encodings, sanitized_filtered_encodings, "qnn")

        self.check_missing_encodings(
            extracted_encodings=sanitized_filtered_encodings, aimet_encodings=aimet_encodings
        )

    def get_dtype(self, data_type):
        # hex value to dtype conversion map
        dtype_map = {
            "0x008": "int",
            "0x016": "int",
            "0x032": "int",
            "0x064": "int",
            "0x108": "int",
            "0x116": "int",
            "0x132": "int",
            "0x164": "int",
            "0x308": "int",
            "0x316": "int",
            "0x332": "int",
            "0x408": "int",
            "0x416": "int",
            "0x432": "int",
            "0x216": "float",
            "0x232": "float",
            "0x508": "bool",
        }
        dtype = hex(data_type)
        dtype = dtype_map.get(dtype, "")
        return dtype

    def generate_encoding_dict(
        self, min_value, max_value, scale, offset, bitwidth, data_type=None, is_symmetric=None
    ):
        """
        Helper function to create a dictionary with given encodings data
        """
        # Using OrderedDict to maintain same order as AIMET encodings
        encoding_dict = OrderedDict()
        encoding_dict["bitwidth"] = bitwidth
        if data_type:
            encoding_dict["dtype"] = data_type
        if is_symmetric is not None:
            encoding_dict["is_symmetric"] = str(is_symmetric)
        encoding_dict["max"] = max_value
        encoding_dict["min"] = min_value
        encoding_dict["offset"] = offset
        encoding_dict["scale"] = scale

        return encoding_dict

    def get_activation_encodings(self, data, op_list):
        """
        Helper function to extract activation encodings
        """
        activation_encodings = OrderedDict()
        data_tensor = data["graph"]["tensors"]
        data_nodes = data["graph"]["nodes"]
        try:
            for layer in data_tensor:
                if "params_count" in data_tensor[layer].keys() or (
                    layer in data_nodes and data_nodes[layer]["type"] in op_list
                ):
                    continue
                datatype = self.get_dtype(data_tensor[layer]["data_type"])
                encoding_info = data_tensor[layer]["quant_params"]["scale_offset"]
                encoding_dict = self.generate_encoding_dict(
                    encoding_info["minimum"],
                    encoding_info["maximum"],
                    encoding_info["scale"],
                    encoding_info["offset"],
                    encoding_info["bitwidth"],
                    datatype,
                    encoding_info["is_symmetric"],
                )
                activation_encodings[layer] = [encoding_dict]
        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting activation encodings from the given DLC file, error: {e}"
            )
        return activation_encodings

    def get_param_encodings(self, data, op_list):
        """
        Helper function to extract param encodings
        """
        param_encodings = OrderedDict()
        data_tensor = data["graph"]["tensors"]
        data_nodes = data["graph"]["nodes"]
        try:
            for layer in data_tensor:
                if "params_count" not in data_tensor[layer].keys() or (
                    layer in data_nodes and data_nodes[layer]["type"] in op_list
                ):
                    continue
                datatype = self.get_dtype(data_tensor[layer]["data_type"])
                reset_offset = False
                if np.right_shift(data_tensor[layer]["data_type"], 8) == 3:
                    reset_offset = True
                if "axis_scale_offset" in data_tensor[layer]["quant_params"]:
                    channel_encodings = []
                    if "scale_offsets" in data_tensor[layer]["quant_params"]["axis_scale_offset"]:
                        num_channels = len(
                            data_tensor[layer]["quant_params"]["axis_scale_offset"]["scale_offsets"]
                        )
                        encoding_type = "scale_offsets"
                    else:
                        num_channels = len(
                            data_tensor[layer]["quant_params"]["axis_scale_offset"][
                                "bw_scale_offset"
                            ]
                        )
                        encoding_type = "bw_scale_offset"
                    for axis in range(num_channels):
                        encoding_info = data_tensor[layer]["quant_params"]["axis_scale_offset"][
                            encoding_type
                        ][axis]
                        if reset_offset:
                            encoding_info["offset"] = 0
                        encoding_dict = self.generate_encoding_dict(
                            encoding_info["minimum"],
                            encoding_info["maximum"],
                            encoding_info["scale"],
                            encoding_info["offset"],
                            encoding_info["bitwidth"],
                            datatype,
                            encoding_info["is_symmetric"],
                        )
                        channel_encodings.append(encoding_dict)
                    param_encodings[layer] = channel_encodings
                else:
                    for encoding_type in ["scale_offset", "bw_scale_offset"]:
                        if encoding_type in data_tensor[layer]["quant_params"]:
                            encoding_info = data_tensor[layer]["quant_params"][encoding_type]
                            if reset_offset:
                                encoding_info["offset"] = 0
                            encoding_dict = self.generate_encoding_dict(
                                encoding_info["minimum"],
                                encoding_info["maximum"],
                                encoding_info["scale"],
                                encoding_info["offset"],
                                encoding_info["bitwidth"],
                                datatype,
                                encoding_info["is_symmetric"],
                            )
                            param_encodings[layer] = [encoding_dict]
        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting param encodings from the given json file, error: {e}"
            )
        return param_encodings

    def extract_model_net_encodings(self):
        """
        Helper function to extract encodings from model_net.json file
        """
        with open(self.args.input) as json_file:
            data = json.load(json_file)
            op_list = [
                "Reduce",
                "Transpose",
                "CropAndResize",
                "Gather",
                "GatherElements",
                "GatherND",
                "Pad",
                "Pool2d",
                "Pool3d",
                "Reshape",
                "Resize",
                "StridedSlice",
                "SpaceToDepth",
                "DepthToSpace",
                "ChannelShuffle",
                "Split",
                "TopK",
                "Conv2d",
                "Conv3d",
                "TransposeConv2d",
                "DepthwiseConv2d",
                "FullyConnected",
                "MatMul",
            ]
            extracted_encodings = {}
            extracted_encodings["activation_encodings"] = self.get_activation_encodings(
                data, op_list
            )
            extracted_encodings["param_encodings"] = self.get_param_encodings(data, op_list)
            return extracted_encodings

    def get_aimet_encodings(self):
        """
        Helper function extract aimet encodings from file
        """
        aimet_encodings = {}
        aimet_encodings["activation_encodings"] = {}
        aimet_encodings["param_encodings"] = {}
        sanitize_unsanitize_map = OrderedDict()
        with open(self.args.aimet_encodings_json) as json_file:
            aimet_encodings_json = json.load(json_file)
            for encodings in aimet_encodings_json.keys():
                if encodings == "activation_encodings" or encodings == "param_encodings":
                    for layer in aimet_encodings_json[encodings].keys():
                        sanitize_unsanitize_map[santize_node_name(layer)] = layer
                        aimet_encodings[encodings][santize_node_name(layer)] = aimet_encodings_json[
                            encodings
                        ][layer]
        return aimet_encodings, sanitize_unsanitize_map

    def filter_encodings(self, encodings, sanitize_unsanitize_map):
        """
        Helper function to filter extracted encodings
        """
        filtered_encodings = {}
        filtered_encodings["activation_encodings"] = {}
        filtered_encodings["param_encodings"] = {}
        for encoding in encodings.keys():
            if encoding == "activation_encodings" or encoding == "param_encodings":
                for tensor in encodings[encoding].keys():
                    try:
                        if tensor in sanitize_unsanitize_map.keys():
                            filtered_encodings[encoding][sanitize_unsanitize_map[tensor]] = (
                                encodings[encoding][tensor]
                            )
                        elif tensor.endswith("_permute"):
                            new_tensor = tensor.removesuffix("_permute")
                            filtered_encodings[encoding][sanitize_unsanitize_map[new_tensor]] = (
                                encodings[encoding][tensor]
                            )
                    except:
                        continue
        return filtered_encodings

    def compare_encodings_snpe(self):
        try:
            from qti.aisw.dlc_utils import snpe_dlc_utils
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        # Load given SNPE DLC file
        snpe_model = snpe_dlc_utils.ModelInfo(self.args.input)

        # Fetch model's meta data
        (
            model_version,
            converter_command,
            quantizer_command,
            converter_version,
            model_copyright,
        ) = snpe_model.get_meta_data()

        # Find Major version using value of converter_version
        # Sample value of converter_version variable is 'DLC created with converter version: 2.16.0.231027072756_64280'
        converter_major_version = converter_version.split(":")[-1].strip().split(".")[0]
        self._logger.info(converter_version)

        # Extract both activation and param encodings from the given DLC
        DLC_helper = DLCHelper(self.args.input, converter_major_version)
        extracted_encodings = DLC_helper.extract_dlc_encodings()

        # Dump Extracted SNPE encodings to json file
        with open(self.extracted_encodings_path, "w") as json_write:
            json.dump(extracted_encodings, json_write, indent=4)

        # load AIMET encodings
        with open(self.args.aimet_encodings_json) as json_file:
            aimet_encodings = json.load(json_file)

        # Generate excel sheet highlighting any mismatches between AIMET and SNPE encodings
        self.generate_excel_sheet(
            aimet_encodings, extracted_encodings, "snpe", converter_major_version
        )

        # Log warnings if any encodings are present in AIMET but not in SNPE and vice-versa
        self.check_missing_encodings(
            extracted_encodings=extracted_encodings, aimet_encodings=aimet_encodings
        )

        self._logger.info(
            "Extracted SNPE encodings are saved at {}".format(
                os.path.abspath(self.extracted_encodings_path)
            )
        )
        self._logger.info(
            "Differences in SNPE encodings and AIMET encodings are written to {}".format(
                os.path.abspath(self.encoding_diff_path)
            )
        )

    def generate_excel_sheet(
        self, aimet_encodings, target_encodings, engine, converter_major_version=0
    ):
        """
        Helper function to find differences between AIMET and Target encodings.
        """
        with xlsxwriter.Workbook(self.encoding_diff_path) as workbook:
            # Initialize Excel sheet
            worksheet = workbook.add_worksheet()
            # Writer headers to Excel sheet
            if converter_major_version == 1:
                headers = [
                    "Encoding_type",
                    "buffer_name",
                    "bitwidth",
                    "max",
                    "min",
                    "offset",
                    "scale",
                ]
            else:
                headers = [
                    "Encoding_type",
                    "buffer_name",
                    "bitwidth",
                    "dtype",
                    "is_symmetric",
                    "max",
                    "min",
                    "offset",
                    "scale",
                ]

            headers_idx = {}
            for idx, header in enumerate(headers):
                worksheet.write(0, idx, header)
                headers_idx[header] = idx

            sheet_idx = 1
            warning_format_1 = workbook.add_format({"bold": True, "font_color": "red"})
            warning_format_2 = workbook.add_format({"bold": True, "font_color": "blue"})
            diff_counts = {}
            dlc_version = "dlcv3"
            if converter_major_version != 1:
                dlc_version = "dlcv4"
            if engine == "qnn":
                target_encoding_type = "QNN"
            else:
                target_encoding_type = dlc_version
            # Loop for activations and params
            for encoding_type in aimet_encodings.keys():
                diff_counts[encoding_type] = 0

                if (self.args.params_only and encoding_type == "activation_encodings") or (
                    self.args.activations_only and encoding_type == "param_encodings"
                ):
                    continue

                if encoding_type not in target_encodings.keys():
                    continue
                """
                Loop for encodings list present in activations/params.
                if a layer has per-channel quantization then aimet_encoding_list will contain multiple encoding dictionaries corresponding to each channel,
                otherwise only one encoding dictionary will present in aimet_encoding_list
                """
                for encoding_name, aimet_encoding_list in aimet_encodings[encoding_type].items():
                    if self.args.specific_node and encoding_name != self.args.specific_node:
                        continue

                    if encoding_name not in target_encodings[encoding_type].keys():
                        continue

                    target_encoding_list = target_encodings[encoding_type][encoding_name]

                    if len(aimet_encoding_list) != len(target_encoding_list):
                        self._logger.error(
                            f"Encodings channels count mismatch for {encoding_name}, AIMET has {len(aimet_encoding_list)} channel encodings while Target has {len(target_encoding_list)} channel encodings"
                        )
                        continue

                    for idx, aimet_encoding_dict in enumerate(aimet_encoding_list):
                        target_encoding_dict = target_encoding_list[idx]
                        errors_observed = 0
                        # Indicate whether scale and offset need to be corrected according to the bitwidths before comparing
                        correction = None
                        for key in aimet_encoding_dict.keys():
                            if key not in target_encoding_dict.keys():
                                continue

                            # convert below encodings to strings since dtype and is_symmetric are strings in AIMET encodings
                            if key in ["dtype", "is_symmetric"]:
                                target_encoding_dict[key] = str(target_encoding_dict[key])
                            pre = self.args.precision

                            # if encoding is either scale or offset and If correction is needed then modify the aimet_encoding before comparing.
                            if key in ["scale", "offset"] and correction:
                                if (correction == "up" and key == "offset") or (
                                    correction == "down" and key == "scale"
                                ):
                                    compare_encoding = round(
                                        target_encoding_dict[key], pre
                                    ) == round((aimet_encoding_dict[key] * 256.0), pre) or round(
                                        target_encoding_dict[key], pre
                                    ) == round((aimet_encoding_dict[key] * 257.0), pre)
                                else:
                                    compare_encoding = round(
                                        target_encoding_dict[key], pre
                                    ) == round((aimet_encoding_dict[key] / 256.0), pre) or round(
                                        target_encoding_dict[key], pre
                                    ) == round((aimet_encoding_dict[key] / 257.0), pre)
                            elif key in ["offset", "is_symmetric"]:
                                if (
                                    target_encoding_dict["is_symmetric"] == "False"
                                    and target_encoding_dict["offset"] == 0
                                    and aimet_encoding_dict["is_symmetric"] == "True"
                                    and aimet_encoding_dict["offset"] == -128
                                ):
                                    compare_encoding = True
                                else:
                                    if key in ["offset"]:
                                        compare_encoding = round(
                                            target_encoding_dict[key], pre
                                        ) == round(aimet_encoding_dict[key], pre)
                                    else:
                                        compare_encoding = (
                                            target_encoding_dict[key] == aimet_encoding_dict[key]
                                        )
                            elif key in ["max", "min", "scale"]:
                                # Compare the encodings by rounding with the specified precision
                                compare_encoding = round(target_encoding_dict[key], pre) == round(
                                    aimet_encoding_dict[key], pre
                                )
                            else:
                                compare_encoding = (
                                    target_encoding_dict[key] == aimet_encoding_dict[key]
                                )
                            # Compare current iteration's encoding and they are not equal
                            if not compare_encoding:
                                # Highlight entry for encoding since AIMET and Target is not matching
                                diff_counts[encoding_type] += 1
                                errors_observed += 1
                                # Default error message if the encodings are not equal
                                diff_warning = f"* {target_encoding_type} encoding={str(target_encoding_dict[key])} aimet encoding={str(aimet_encoding_dict[key])}"
                                if key == "bitwidth":
                                    # Warning message if bitwidths are not equal and either 8 or 16
                                    diff_warning = f"| {target_encoding_type} encoding={str(target_encoding_dict[key])} aimet encoding={str(aimet_encoding_dict[key])}"
                                    if (
                                        target_encoding_dict[key] == 16
                                        and aimet_encoding_dict[key] == 8
                                    ):
                                        correction = "up"
                                    elif (
                                        target_encoding_dict[key] == 8
                                        and aimet_encoding_dict[key] == 16
                                    ):
                                        correction = "down"
                                    else:
                                        # Invalid bitwidths, neither of the bitwidth equal to 8 or 16
                                        diff_warning = f"* Activation bitwidth conversions from aimet encoding={str(aimet_encoding_dict[key])} to {target_encoding_type} encoding={str(target_encoding_dict[key])} not supported"
                                # If encoding is either scale/offset and correction was applied
                                elif key in ["scale", "offset"] and correction:
                                    diff_warning = f"* {key} not consistent according to bitwidth conversion {target_encoding_type} encoding={str(target_encoding_dict[key])} aimet encoding={str(aimet_encoding_dict[key])}"
                                # if the warning message starts with "|", apply warning_format_2
                                if diff_warning[0] == "|":
                                    worksheet.write(
                                        sheet_idx, headers_idx[key], diff_warning, warning_format_2
                                    )
                                else:
                                    worksheet.write(
                                        sheet_idx, headers_idx[key], diff_warning, warning_format_1
                                    )
                        if errors_observed:
                            worksheet.write(sheet_idx, 0, encoding_type)
                            worksheet.write(sheet_idx, 1, encoding_name)
                            sheet_idx = sheet_idx + 1

                    if self.args.specific_node:
                        break

        self._logger.info(
            f"Number of activation encoding differences observed: {diff_counts['activation_encodings']}"
        )
        self._logger.info(
            f"Number of param encoding differences observed: {diff_counts['param_encodings']}"
        )
        self._logger.info(
            f"Total number of encoding differences observed: {diff_counts['activation_encodings'] + diff_counts['param_encodings']}"
        )


class DLCHelper:
    def __init__(self, dlc, converter_major_version):
        try:
            from qti.aisw.dlc_utils import modeltools
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        self.converter_major_version = converter_major_version
        if converter_major_version == "1":
            self.model = modeltools.Model()
            self.model.load(dlc)
        else:
            self.model = modeltools.IrDlcReader()
            self.cache_reader = modeltools.IrDlcCacheRecordReader()
            self.model.open(dlc)

    def extract_dlc_encodings(self):
        """
        Extracts both activation and param encodings from the given dlc.
        """
        extracted_encodings = {}
        extracted_encodings["activation_encodings"] = self.get_activation_encodings()
        extracted_encodings["param_encodings"] = self.get_param_encodings()

        return extracted_encodings

    def generate_encoding_dict(
        self, min_value, max_value, delta, offset, bitwidth, data_type=None, is_symmetric=None
    ):
        """
        Helper function to create a dictionary with given encodings data
        """
        # Using OrderedDict to maintain same order as AIMET encodings
        encoding_dict = OrderedDict()
        encoding_dict["bitwidth"] = bitwidth
        if data_type:
            encoding_dict["dtype"] = data_type
        if is_symmetric:
            encoding_dict["is_symmetric"] = is_symmetric
        encoding_dict["max"] = max_value
        encoding_dict["min"] = min_value
        encoding_dict["offset"] = offset
        encoding_dict["scale"] = delta

        return encoding_dict

    def get_activation_encodings(self):
        """
        Extracts activation encodings from the given dlc.
        """
        if self.converter_major_version == "1":
            return self.extract_dlcv3_activation_encodings()
        else:
            return self.extract_dlcv4_activation_encodings()

    def extract_dlcv3_activation_encodings(self):
        """
        Extracts activation encodings from the given dlc with converter version 1.*
        """
        activation_encodings = OrderedDict()
        try:
            for layer in self.model.get_layers():
                if ":0" in layer["name"]:
                    continue

                min_value, max_value, delta, offset, bitwidth = self.model.get_tf_output_encoding(
                    layer["name"]
                )[:5]
                encoding_name = layer["output_names"][0]
                encoding_dict = self.generate_encoding_dict(
                    min_value, max_value, delta, offset, bitwidth
                )
                activation_encodings[encoding_name] = [encoding_dict]
        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting activation encodings from the given DLC file, error: {e}"
            )

        return activation_encodings

    def extract_dlcv4_activation_encodings(self):
        """
        Extracts activation encodings from the given dlc with converter version 2.*
        """
        try:
            from qti.aisw.converters.common import ir_graph
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        def extract_encodings(encoding_name, encoding, dtype):
            """
            Helper function to extract bitwidth, min, max, scale and offset params from the given encoding
            """
            encoding_info = None
            if encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
                encoding_info = encoding.encInfo
            elif encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
                encoding_info = encoding.encInfo.axisEncInfo.encInfos[0]

            if encoding_info != None:
                data_type = self.get_aimet_datatype(dtype)
                encoding_dict = self.generate_encoding_dict(
                    encoding_info.min,
                    encoding_info.max,
                    encoding_info.scale,
                    encoding_info.offset,
                    encoding_info.bw,
                    data_type=data_type,
                    is_symmetric=str(bool(encoding.axisEncInfo.axis)),
                )
                activation_encodings[encoding_name] = [encoding_dict]

        graph = self.model.get_ir_graph()
        activation_encodings = OrderedDict()
        try:
            for op in graph.get_ops():
                if ":0" in op.name:
                    continue

                # Extract encodings from inputs of the Op
                for input in op.inputs():
                    if input.is_app_write_tensor():
                        extract_encodings(
                            input.name(), input.get_encoding(), input.data_type_string()
                        )

                # Extract encodings from outputs of the Op
                for output in op.outputs():
                    extract_encodings(
                        output.name(), output.get_encoding(), output.data_type_string()
                    )
        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting activation encodings from the given DLC file, error: {e}"
            )

        return activation_encodings

    def get_aimet_datatype(self, snpe_dtype):
        """
        Returns AIMET equivalent datatype for given SNPE datatype
        """
        if snpe_dtype in [
            "Int_8",
            "Uint_8",
            "sFxp_8",
            "uFxp_8",
            "Int_16",
            "Uint_16",
            "sFxp_16",
            "uFxp_16",
            "Int_32",
            "Uint_32",
            "sFxp_32",
            "uFxp_32",
            "Int_64",
            "Uint_64",
        ]:
            data_type = "int"
        elif snpe_dtype in ["Float_16", "Float_32"]:
            data_type = "float"
        elif snpe_dtype == "Bool_8":
            data_type = "bool"
        else:
            data_type = "undefined"
        return data_type

    def get_param_encodings(self):
        """
        Extracts param encodings from the given dlc.
        """
        if self.converter_major_version == "1":
            return self.extract_dlcv3_param_encodings()
        else:
            return self.extract_dlcv4_param_encodings()

    def extract_dlcv4_param_encodings(self):
        """
        Extracts param encodings from the given dlc with converter version 2.*
        """
        try:
            from qti.aisw.converters.common import ir_graph
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        graph = self.model.get_ir_graph()
        param_encodings = OrderedDict()
        try:
            for op in graph.get_ops():
                if ":0" in op.name:
                    continue

                for input in op.inputs():
                    # consider only static tensors(weights)
                    if input.is_static_tensor():
                        data_type = self.get_aimet_datatype(input.data_type_string())

                        if (
                            input.get_encoding().type
                            == ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
                        ):
                            # extract per-tensor weight encodings
                            encoding_info = input.get_encoding().encInfo
                            encoding_dict = self.generate_encoding_dict(
                                encoding_info.min,
                                encoding_info.max,
                                encoding_info.scale,
                                encoding_info.offset,
                                encoding_info.bw,
                                data_type=data_type,
                                is_symmetric=str(bool(input.get_encoding().axisEncInfo.axis)),
                            )
                            param_encodings[input.name()] = [encoding_dict]
                        elif (
                            input.get_encoding().type
                            == ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
                            or input.get_encoding().type
                            == ir_graph.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET
                        ):
                            # extract per-channel weight encodings
                            channel_encodings = []
                            for axis in range(len(input.get_encoding().axisEncInfo.encInfos)):
                                encoding_info = input.get_encoding().axisEncInfo.encInfos[axis]
                                encoding_dict = self.generate_encoding_dict(
                                    encoding_info.min,
                                    encoding_info.max,
                                    encoding_info.scale,
                                    encoding_info.offset,
                                    encoding_info.bw,
                                    data_type=data_type,
                                    is_symmetric=str(bool(input.get_encoding().axisEncInfo.axis)),
                                )
                                channel_encodings.append(encoding_dict)
                            param_encodings[input.name()] = channel_encodings

        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting param encodings from the given DLC file, error: {e}"
            )

        return param_encodings

    def extract_dlcv3_param_encodings(self):
        """
        Extracts param encodings from the given dlc with converter version 1.*
        """

        param_encodings = OrderedDict()
        for layer in self.model.get_layers():
            if ":0" in layer["name"]:
                continue

            try:
                weight_encoding = self.model.get_tf_weight_encoding(layer["name"], 0)
                if weight_encoding is not None:
                    axis = self.model.get_tf_weight_encoding_axis(layer["name"], 0)

                    if axis >= 0:
                        # extract per-channel weight encodings
                        num_elements = self.model.get_tf_weight_encoding_num_elements(
                            layer["name"], 0
                        )

                        channel_encodings = []
                        for channel in range(num_elements):
                            min_value, max_value, delta, offset, bitwidth = (
                                self.model.get_tf_weight_encoding_by_element(
                                    layer["name"], 0, channel
                                )[:5]
                            )
                            encoding_dict = self.generate_encoding_dict(
                                min_value, max_value, delta, offset, bitwidth
                            )
                            channel_encodings.append(encoding_dict)
                        encoding_name = layer["name"] + ".weight"
                        param_encodings[encoding_name] = channel_encodings
                    else:
                        # extract per-tensor weight encodings
                        min_value, max_value, delta, offset, bitwidth = weight_encoding[:5]
                        encoding_name = layer["name"] + ".weight"
                        encoding_dict = self.generate_encoding_dict(
                            min_value, max_value, delta, offset, bitwidth
                        )
                        param_encodings[encoding_name] = [encoding_dict]
            except:
                try:
                    # extract bias encodings
                    bias_encoding = self.model.get_tf_bias_encoding(layer["name"])
                    if bias_encoding is not None:
                        min_value, max_value, delta, offset, bitwidth = bias_encoding[:5]
                        encoding_name = layer["name"] + ".bias"
                        encoding_dict = self.generate_encoding_dict(
                            min_value, max_value, delta, offset, bitwidth
                        )
                        param_encodings[encoding_name] = [encoding_dict]
                except:
                    pass

        return param_encodings
