# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from typing import Tuple
import json
from importlib import import_module
import os
from logging import Logger
import psutil

from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import UnsupportedError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import FrameworkExtension
from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_objects import get_available_frameworks
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import MaxLimits, DataType

import sys
if sys.version_info < (3, 8):
    # distutils deprecated for Python 3.8 and up
    from distutils.version import StrictVersion as Version
else:
    # packaging requires Python 3.8 and up
    from packaging.version import Version as Version

GB = 1024 * 1024 * 1024
MAX_RAM_LIMIT = psutil.virtual_memory()[1]  # Available RAM memory in the system


def dump_intermediate_tensors(output_dir: str, result: dict, logger: Logger, use_native_output_files: bool) -> None:
    '''
    dumps the tensors present in the result dictionary in the output directory in <sanitized_tensor_name.raw> format

    :param output_dir: path to the output directory where the tensors will be dumped in .raw format
    :param result: dictionary of tensor name and tensor data as values
    :param logger: object of logging.Logger
    :param use_native_output_files: dumps outputs as per framework model's actual data types
    '''
    logger.debug(f'Dumping outputs with use_native_output_files={use_native_output_files}')
    data_path = os.path.join(output_dir, '{}{}')
    for output_tensor_name, data in result.items():
        # Sanitize output tensor name to avoid dumping outputs in sub-folders.
        sanitized_output_tensor_name = santize_node_name(output_tensor_name)

        # Most of the systems does not allow to create a file name, more than 255 bytes, hence skipping
        # to dump those tensor files.
        if len((sanitized_output_tensor_name +
                '.raw').encode('utf-8')) > MaxLimits.max_file_name_size.value:
            logger.warning(
                f"Skipping dumping of tensor {sanitized_output_tensor_name} as filename exceeds max limit {MaxLimits.max_file_name_size.value}"
            )
            continue
        file_path = data_path.format(sanitized_output_tensor_name, '.raw')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            save_outputs(data, file_path, use_native_output_files)
        except Exception as e:
            logger.error(
                f"Dumping output raw file {sanitized_output_tensor_name + '.raw'} for tensor {output_tensor_name} failed"
            )


def dump_profile_json(output_dir: str, result: str) -> None:
    '''
    dumps the tensors present in the result dictionary in the output directory in <sanitized_tensor_name.raw> format

    :param output_dir: path to the output directory where the tensors will be dumped in .raw format
    :param result: dictionary of tensor name and tensor data as values
    '''
    profile_info_json_path = os.path.join(output_dir, 'profile_info.json')
    if os.path.exists(profile_info_json_path):
        # For the case for model > 2GB
        tensor_info = read_json(profile_info_json_path)
    else:
        tensor_info = {}

    for output_tensor_name, data in result.items():
        santized_tensor_name = santize_node_name(output_tensor_name)
        try:
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)
        except Exception as e:
            raise Exception(f"Encountered Error: {e}")

        if not data.size or data.dtype == bool:
            if data.size == 0:
                tensor_info[santized_tensor_name] = (
                    '-',
                    '-',
                    '-',
                    '-',
                    '-',
                )
            else:
                tensor_info[santized_tensor_name] = (str(data.dtype), data.shape, data.tolist(),
                                                     data.tolist(), data.tolist())
        else:
            tensor_info[santized_tensor_name] = (str(data.dtype), data.shape,
                                                 str(round(np.min(data),
                                                           3)), str(round(np.max(data), 3)),
                                                 str(round(np.median(data), 3)))

    dump_json(tensor_info, profile_info_json_path)


def load_inputs(data_path, data_type, data_dimension=None):
    # type:  (str, str, Tuple) -> np.ndarray
    data = np.fromfile(data_path, data_type)
    if data_dimension is not None:
        data = data.reshape(data_dimension)
    return data


def save_outputs(data: np.ndarray, data_path: str, use_native_output_files: bool) -> None:
    '''
    Dumps given numpy data to disk, if use_native_output_files is False then data will be typecasted
    to float32 before dumping

    :param data: output tensor as numpy array
    :param data_path: path to dump given data
    :param use_native_output_files: dumps outputs as per framework model's actual data types
    '''
    if use_native_output_files is False:
        data = data.astype(DataType.DEFAULT_OUTPUTS_DATATYPE.value)

    data.tofile(data_path)


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data


def dump_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def transpose_to_nhwc(data, data_dimension):
    # type:  (np.ndarray, list) ->np.ndarray
    if len(data_dimension) == 4:
        data = np.reshape(
            data, (data_dimension[0], data_dimension[1], data_dimension[2], data_dimension[3]))
        data = np.transpose(data, (0, 2, 3, 1))
        data = data.flatten()
    return data


class ModelHelper:

    @classmethod
    def onnx_type_to_numpy(cls, type):
        """
        This method gives the corresponding numpy datatype for given onnx tensor element type
        Args:
            type : onnx tensor element type
        Returns:
            corresponding onnx datatype
        """
        onnx_to_numpy = {
            '1': (np.float32, 4),
            '2': (np.uint8, 1),
            '3': (np.int8, 1),
            '4': (np.uint16, 2),
            '5': (np.int16, 2),
            '6': (np.int32, 4),
            '7': (np.int64, 8),
            '9': (np.bool_, 1)
        }
        if type in onnx_to_numpy:
            return onnx_to_numpy[type]
        else:
            raise UnsupportedError('Unsupported type : {}'.format(str(type)))


def get_framework_info(model_path):
    """
    Tries to find framework name of given model_path based on it's extension.
    Returns: Framework name (None if not able to find framework name)
    """
    if model_path is None:
        return None
    extenstion_framework_map = {
        v: k
        for k, v in FrameworkExtension.framework_extension_mapping.items()
    }
    model_extension = '.' + model_path.rsplit('.', 1)[-1]
    return extenstion_framework_map.get(model_extension, None)


def extract_input_information(input_tensor):
    input_info = {}
    in_list = list(zip(*input_tensor))
    if len(in_list) == 4:
        (in_names, in_dims, in_data_paths, in_types) = in_list
    elif len(in_list) == 3:
        (in_names, in_dims, in_data_paths) = in_list
        in_types = None
    else:
        raise FrameworkError(get_message('ERROR_FRAMEWORK_RUNNER_INPUT_TENSOR_LENGHT_ERROR'))

    input_names = list(in_names)
    input_dims = [[int(x) for x in dim.split(',')] for dim in in_dims]

    if len(input_names) != len(input_dims):
        return None

    for i, input_name in enumerate(input_names):
        input_info[input_name] = input_dims[i]

    return input_info


def max_version(framework, available_frameworks):
    versions = available_frameworks.get(framework, {})
    return max(versions.keys(), key=lambda x: Version(x))


def simplify_onnx_model(logger, model_path=None, input_tensor=None, output_dir=None,
                        custom_op_lib=None):
    framework = 'onnx'
    available_frameworks = get_available_frameworks()
    version = max_version(framework, available_frameworks)
    module, framework_class = available_frameworks[framework][version]
    framework_type = getattr(import_module(module), framework_class)
    framework_instance = framework_type(logger, custom_op_lib=custom_op_lib)

    optimized_model_path = os.path.join(
        output_dir, "optimized_model" + FrameworkExtension.framework_extension_mapping[framework])
    input_information = extract_input_information(input_tensor)
    _, optimized_model_path = framework_instance.optimize(model_path, optimized_model_path,
                                                          input_information)

    return optimized_model_path
