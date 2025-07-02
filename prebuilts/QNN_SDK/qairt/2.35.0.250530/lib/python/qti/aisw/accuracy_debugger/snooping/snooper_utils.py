# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from dataclasses import dataclass

import numpy as np
import pandas as pd
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper


class ActivationStatus:
    """Activation Status"""

    INITIALIZED = "INITIALIZED"
    SKIP = "SKIP"
    CONVERTER_FAILURE = "CONVERTER_FAILURE"
    OPTIMIZER_FAILURE = "OPTIMIZER_FAILURE"
    QUANTIZER_FAILURE = "QUANTIZER_FAILURE"
    OFFLINE_PREPARE_FAILURE = "OFFLINE_PREPARE_FAILURE"
    NET_RUN_FAILURE = "NET_RUN_FAILURE"
    CUSTOM_OVERRIDE_GENERATION_FAILURE = "CUSTOM_OVERRIDE_GENERATION_FAILURE"
    INFERENCE_FAILURE = "INFERENCE_FAILURE"
    INFERENCE_DONE = "INFERENCE_DONE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    SUCCESS = "SUCCESS"

    def __init__(self, activation_name, msg="initialize") -> None:
        self._current_status = ActivationStatus.INITIALIZED
        self._msg = msg
        self._activation_name = activation_name

    def set_status(self, status, msg):
        self._current_status = status
        self._msg = msg

    def get_status(self):
        return self._current_status

    def get_msg(self):
        return self._msg


@dataclass
class ActivationInfo:
    """Represents activation information of a tensor.

    Attributes:
        dtype (str): Data type of the tensor.
        shape (list[int]): Shape of the tensor.
        distribution (tuple[float, float, float]): Distribution of the tensor.
    """

    dtype: str
    shape: list[int]
    distribution: tuple[float, float, float]


def filter_snooping_report(
    snooping_report: pd.DataFrame, inference_data: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Filters given snooping report and returns filtered report.

    Filtering is applied to below scenarios:
    1. Conv -> Relu
    2. Add -> Relu
    In both cases, target graphs dump Relu output for Conv/Add nodes, leading to inconsistencies
    between framework outputs or AIMET outputs. Conv/Add entries that match the subsequent
    Relu node will be removed from snooping report.

    Args:
        snooping_report: Snooping report dataframe
        inference_data: Inference outputs corresponding to each entry in Snooping report

    Returns:
        pd.DataFrame: A Dataframe containing filtered snooping report
    """
    remove_indexes = []
    for index in range(0, len(snooping_report.index) - 1):
        if (
            snooping_report["op type"][index] in ["Conv2d", "Eltwise_Binary"]
            and snooping_report["op type"][index + 1] == "ElementWiseNeuron"
        ):
            current_node_name = snooping_report["op name"][index]
            next_node_name = snooping_report["op name"][index + 1]
            current_node_data = inference_data[Helper.transform_node_names(current_node_name)]
            next_node_data = inference_data[Helper.transform_node_names(next_node_name)]

            unique_data = np.unique(current_node_data == next_node_data)
            if len(unique_data) == 1 and unique_data[0] == True:
                remove_indexes.append(index)

    return snooping_report.drop(labels=remove_indexes, axis=0)
