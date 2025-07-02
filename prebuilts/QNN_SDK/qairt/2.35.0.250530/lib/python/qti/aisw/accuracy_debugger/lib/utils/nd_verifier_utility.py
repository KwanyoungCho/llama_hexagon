# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import os

from qti.aisw.accuracy_debugger.lib.utils.nd_constants import qnn_datatype_to_size,\
    DataType, MaxLimits
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, dump_json
from qti.aisw.accuracy_debugger.lib.quant_checker.nd_utils import QNN_DTYPE_NUMPY_DTYPE_MAP
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import load_inputs


def get_ir_graph(dlc_path):
    """
    Returned IRGraph loaded with given dlc
    """
    from qti.aisw.dlc_utils import modeltools

    model_reader = modeltools.IrDlcReader()
    model_reader.open(dlc_path)
    ir_graph = model_reader.get_ir_graph()
    model_reader.close()

    return ir_graph


def get_tensors_permute_order_from_dlc(dlc_path: str) -> dict:
    """
    Returns permute order and dims of each tensor in the given dlc

    :param dlc_path: Path to dlc file
    """

    ir_graph = get_ir_graph(dlc_path)

    tensors_info = {}
    for tensor_name, tensor_info in ir_graph.get_tensor_map().items():
        sanitized_name = santize_node_name(tensor_name)
        tensors_info[sanitized_name] = {
            'permute_order_to_src': tensor_info.get_permute_order_to_src(),
            'dims': tensor_info.dims()
        }

    return tensors_info


def get_irgraph_tensors_info(qnn_model_json_path: str = None, dlc_path: str = None,
                             output_dir: str = None) -> dict:
    """
    Returns permute order and dims of each tensor in the given qnn_model_json_path/dlc_path

    :param qnn_model_json_path: Path to QNN model net json file
    :param dlc_path: Path to dlc file
    :param output_dir: Output path to save artifacts of this function
    """
    tensors_info = {}
    if qnn_model_json_path is not None:
        qnn_model_json = read_json(qnn_model_json_path)
        tensors_info = qnn_model_json['graph']['tensors']
    elif dlc_path is not None:
        dlc_name = os.path.basename(dlc_path)
        axis_info_json_path = os.path.join(output_dir, dlc_name.replace('.dlc', '.json'))
        if os.path.exists(axis_info_json_path):
            tensors_info = read_json(axis_info_json_path)
        else:
            tensors_info = get_tensors_permute_order_from_dlc(dlc_path)
            dump_json(tensors_info, axis_info_json_path)

    return tensors_info


def permute_tensor_data_axis_order(target_tensor: np.ndarray,
                                   tensor_info: dict) -> tuple[np.ndarray, bool]:
    """
    Permutes target tensor to align source/framework tensor

    :param target_tensor: Target output as numpy array
    :param tensor_info: dictionary containing:
                            {
                                'dims': <target dimensions>,
                                'permute_order_to_src': <permute order>
                            }
    """

    # permute order will be a empty list when both source and target outputs are of same layout
    if tensor_info['permute_order_to_src'] == []:
        return target_tensor, False

    # First reshape target tensor to it's original shape and then transpose to align with golden tensor layout
    target_tensor = np.reshape(target_tensor, tuple(tensor_info['dims']))
    target_tensor = np.transpose(target_tensor, tensor_info['permute_order_to_src']).flatten()

    return target_tensor, True


def get_tensor_names_from_dlc(dlc_path, sanitize_names=False):
    """
    Returns tensor names present in the given dlc
    """

    ir_graph = get_ir_graph(dlc_path)

    tensor_names = []
    for name, tensor in ir_graph.get_tensor_map().items():
        if sanitize_names:
            name = santize_node_name(name)
        tensor_names.append(name)

    return tensor_names


def get_intermediate_tensors_size_from_dlc(dlc_path):
    """
    Returns the sizes of intermediate tensors present in the given DLC file.

    Args:
        dlc_path (str): The path to the DLC file.

    Returns:
        dict: A dictionary where keys are tensor names and values are their corresponding sizes in megabytes.
    """
    # Get the IR graph from the DLC file
    ir_graph = get_ir_graph(dlc_path)

    # Initialize an empty dictionary to store intermediate tensor sizes
    intermediate_tensors_size = {}

    # Iterate over tensors in the IR graph
    for name, tensor in ir_graph.get_tensor_map().items():
        # Check if the tensor type is one of ["NATIVE", "APP_WRITE", "APP_READ"]
        if tensor.tensor_type() in ["NATIVE", "APP_WRITE", "APP_READ"]:
            tensor_name = str(tensor.name())
            tensor_dim = tensor.dims()
            data_type_size_bits = int(qnn_datatype_to_size.get(str(tensor.data_type()), 32))
            tensor_size_mbs = (eval('*'.join(map(str, tensor_dim))) *
                               data_type_size_bits) / (8 * 1024 * 1024)
            # Allocate 25% extra space to each tensors as device sometimes needs more memory
            intermediate_tensors_size[tensor_name] = tensor_size_mbs

    return intermediate_tensors_size


def get_dlc_size(dlc_path):
    """
    Calculates the size of a DLC file in megabytes.

    Args:
        dlc_path (str): The path to the DLC file.

    Returns:
        float: The size of the DLC file in megabytes.
    """
    # Get the size of the file in bytes
    file_size_bytes = os.path.getsize(dlc_path)

    # Convert bytes to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)

    return file_size_mb


def divide_output_tensors(tensor_size_dict, max_size):
    """
    Divides a dictionary of tensors based on their sizes, ensuring that the total size of each divided list
    does not exceed the specified maximum size.

    Args:
        tensor_size_dict (dict): A dictionary where keys are tensor names and values are their corresponding sizes.
        max_size (int): The maximum size allowed for each divided list.

    Returns:
        list of lists: A list of lists, where each inner list contains tensor names whose total size does not exceed max_size.
    """
    # Initialize lists to store divided tensors
    divided_lists = []
    current_list = []
    current_size = 0
    current_string_len = 0

    # Iterate over tensors in the original order
    for tensor_name, tensor_size in tensor_size_dict.items():
        # If adding the current tensor exceeds the max size, start a new list
        if (current_size + tensor_size > max_size) or \
            (current_string_len + len(tensor_name) > MaxLimits.max_set_output_tensors_char_length.value):
            if current_list: divided_lists.append(current_list)
            current_list = []
            current_size = 0
            current_string_len = 0

        # Add the current tensor to the current list
        current_list.append(tensor_name)
        current_size += tensor_size
        current_string_len += len(tensor_name)

    # Add the last list to divided_lists
    if current_list:
        divided_lists.append(current_list)

    return divided_lists


def to_csv(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, encoding='utf-8', index=False)


def to_html(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_html(file_path, classes='table', index=False)


def to_json(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_json(file_path, orient='records', indent=4)


def save_to_file(data, filename) -> None:
    """Save data to file in CSV, HTML and JSON formats :param data: Data to be
    saved to file :param filename: Name of the file."""
    filename = Path(filename)
    to_csv(data, filename.with_suffix(".csv"))
    to_html(data, filename.with_suffix(".html"))
    to_json(data, filename.with_suffix(".json"))


def filter_summary_report(df: pd.DataFrame, target_outputs: str) -> pd.DataFrame:
    '''
    Filters given summary dataframe and returns filtered dataframe.
    Filtering is applied to below scenarios:
    1. Conv -> Relu
    2. Add -> Relu
    In both of the above cases, target graphs dumps Relu output for Conv/Add.
    This means Conv==Relu or Add==Relu in Target which is not consistent with framework outputs or
    AIMET outputs, so as part of filtering we will remove Conv/Add entries from summary report
    which are exactly matching with Relu node followed after them.

    :param df: Summary dataframe generated by verification stage
    :param target_outputs: Path to target graph outputs
    '''
    remove_indexes = []
    for index in range(0, len(df.index) - 1):

        if df['LayerType'][index] in ['Conv2d', 'Eltwise_Binary'] \
                                            and df['LayerType'][index+1] == 'ElementWiseNeuron':
            node1_path = os.path.join(target_outputs, df['Name'][index] + '.raw')
            node2_path = os.path.join(target_outputs, df['Name'][index + 1] + '.raw')
            node1_tensor = np.fromfile(node1_path, dtype=DataType.DEFAULT_OUTPUTS_DATATYPE.value)
            node2_tensor = np.fromfile(node2_path, dtype=DataType.DEFAULT_OUTPUTS_DATATYPE.value)
            unique_data = np.unique(node1_tensor == node2_tensor)

            if len(unique_data) == 1 and unique_data[0] == True:
                remove_indexes.append(index)

    df = df.drop(labels=remove_indexes, axis=0)
    return df


def get_irgraph_dtypes(qnn_model_json_path: str = None, dlc_path: str = None) -> dict:
    """
    Returns data types of each tensor in the given qnn_model_json_path/dlc_path

    :param qnn_model_json_path: Path to QNN model net json file
    :param dlc_path: Path to dlc file
    """
    irgraph_dtypes = {}
    if qnn_model_json_path is not None:
        qnn_model_json = read_json(qnn_model_json_path)
        for tensor_name, tensor_info in qnn_model_json['graph']['tensors'].items():
            dtype_hex = hex(tensor_info['data_type'])
            irgraph_dtypes[tensor_name] = QNN_DTYPE_NUMPY_DTYPE_MAP[dtype_hex].__name__

    elif dlc_path is not None:
        ir_graph = get_ir_graph(dlc_path)
        irgraph_dtypes = {}
        for tensor_name, tensor_info in ir_graph.get_tensor_map().items():
            sanitized_name = santize_node_name(tensor_name)
            dtype_hex = hex(tensor_info.data_type().value)
            irgraph_dtypes[sanitized_name] = QNN_DTYPE_NUMPY_DTYPE_MAP[dtype_hex].__name__

    return irgraph_dtypes


def load_data(tensor_names: list, tensor_paths: dict, irgraph_dtypes: dict = None):
    """
    Loads data with datatypes as per irgraph if provided else default dtype (float32)

    :param tensor_names: List of all tensor names to be loaded
    :param tensor_paths: Dict containing paths to all tensors
    :param irgraph_dtypes: Dict containing data type information of all tensors
    """
    loaded_data = []
    for tensor_name in tensor_names:
        tensor_path = tensor_paths[tensor_name]

        dtype = DataType.DEFAULT_OUTPUTS_DATATYPE.value
        if irgraph_dtypes and tensor_name in irgraph_dtypes:
            dtype = irgraph_dtypes[tensor_name]

        loaded_data.append(load_inputs(tensor_path, dtype))
    return loaded_data