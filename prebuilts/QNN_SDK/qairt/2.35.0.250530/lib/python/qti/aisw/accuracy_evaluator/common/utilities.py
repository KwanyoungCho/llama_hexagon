# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import enum
import heapq
import importlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Type

import numpy as np
import pkg_resources
import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.tools.core.utilities.comparators.common import COMPARATORS
from qti.aisw.tools.core.utilities.comparators.factory import get_comparator, get_comparator_param


# to avoid printing logs on console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mismatched_packages = []


class TopKTracker:
    """A class to track the top k elements based on a comparison score.

    This class maintains a collection of the top k elements according to a given
    comparison score. It can be configured to keep the largest or smallest elements.

    Attributes:
        k (int): Number of top elements to track.
        keep_largest (bool): Whether to keep the largest elements (True) or smallest (False).
        heap (list): Internal heap structure to store tuples of (comp_score, file_idx).
    """

    def __init__(self, k: int = 1, keep_largest: bool = True):
        """Initialize a TopKTracker to track the top k elements based on a comparison score.

        Args:
            k (int): Number of top elements to track.
            keep_largest (bool): Whether to keep the largest elements (True) or smallest (False).
        """
        self.k = k
        self.keep_largest = keep_largest  # True to keep largest, False to keep smallest
        self.heap = []  # Will store tuples (comp_score, file_idx)

    def add(self, value: tuple[float, int]):
        """Add a value to the tracker.

        Args:
            value (tuple[float, int]): A tuple containing a comparison score and an index.
        """
        comp_score, file_idx = value

        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (comp_score, file_idx))
        else:
            if self.keep_largest:
                # Keep largest: compare with the smallest in the heap (heap[0])
                if comp_score > self.heap[0][0]:
                    heapq.heapreplace(self.heap, (comp_score, file_idx))
            else:
                # Keep smallest: compare with the largest in the heap (heap[0] in a max-heap)
                # To simulate a max-heap, we store negative values
                if comp_score < self.heap[0][0]:
                    heapq.heapreplace(self.heap, (comp_score, file_idx))

    def get_top_k(self) -> list[(float, int)]:
        """Returns a list of top k elements in descending order of score.

        Returns:
            List[Tuple[float, int]]: List of top k elements sorted in descending order of score.
        """
        # Return the heap in descending order of comp_score
        return sorted(self.heap, key=lambda x: -x[0])

    def get_top_k_indices(self) -> list[int]:
        """Returns a list of indices from the top k elements in descending order of score.

        Returns a list of indices from the top k elements in descending order of score.
        """
        # Return the indices from the heap in descending order of comp_score
        return [idx for _, idx in self.get_top_k()]


def chunked_file_list_generator(
    input_file_list: str,
    lines_per_file: int
) -> list[str]:
    """Splits the contents of an input file into multiple output files,
    each containing a specified number of lines.

    Args:
        input_file_list (str): Path to the input file containing multiple lines.
        lines_per_file (int): Number of lines to include in each output file.

    Returns:
        List[str]: A list of filenames that were created.

    Raises:
        ValueError: If `lines_per_file` is not a positive integer.
        FileNotFoundError: If the input_file_list does not exist.
    """
    created_files = []
    file_path = Path(input_file_list)
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        qacc_file_logger.error(f"Error: The file '{input_file_list}' was not found.")
        raise FileNotFoundError(f"The file '{input_file_list}' was not found.")

    if not isinstance(lines_per_file, int) or lines_per_file <= 0:
        raise ValueError("lines_per_file must be a positive integer.")

    total_lines = len(lines)
    if lines_per_file > total_lines:
        raise ValueError(f"lines_per_file cannot exceed the total number of lines ({total_lines})" +
                            "in the list file.")

    file_count = (total_lines + lines_per_file - 1) // lines_per_file

    # Split the file into chunks and write to separate files
    for i in range(file_count):
        start = i * lines_per_file
        end = start + lines_per_file
        chunk = lines[start:end]

        output_file = file_path.parent / f'{file_path.stem}_{i + 1}.txt'
        if not output_file.exists():
            with open(output_file, 'w') as out_f:
                out_f.writelines(chunk)

        created_files.append(output_file)
        qacc_file_logger.info(f"Created file: {output_file} with {len(chunk)} lines")

    return created_files


def timer_decorator(func: Callable) -> Callable:
    """Decorator that measures the execution time of a function.

    Args:
        func: The function to be executed and whose execution time will be measured.

    Returns:
        A wrapper function that returns a tuple containing the execution time in seconds
        and the original result of the wrapped function.
    """
    def wrapper(*args, **kwargs):
        """Wrapper function that measures the execution time of the decorated function.

        Args:
            *args: Variable number of positional arguments passed to the decorated function.
            **kwargs: Variable number of keyword arguments passed to the decorated function.

        Returns:
            A tuple containing the execution time in seconds and the original result
            of the wrapped function.
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            elapsed_time = time.time() - start_time
            raise e from None  # Re-raise the exception with the correct traceback
        else:
            elapsed_time = round(time.time() - start_time, 4)
            return elapsed_time, result
    return wrapper


class ModelType(enum.Enum):
    ONNX = 0
    TORCHSCRIPT = 1
    TENSORFLOW = 2
    TFLITE = 3
    FOLDER = 4


class Helper:
    """Utility class contains common utility methods
    To use:
    >>>Helper.get_np_dtype(type)
    >>>Helper.get_model_type(path)
    """

    @classmethod
    def safe_import_package(cls, package_name: str,
             recommended_package_version: Optional[str] = None) -> Type[Any]:
        """Safely import a Python package and check if the installed version matches
        the recommended version.

        Args:
            package_name (str): The name of the package to import.
            recommended_package_version (Optional[str], optional): The recommended version of the package.
                Defaults to None.

        Returns:
            Type[Any]: The imported package.

        Raises:
            ImportError: If the package cannot be imported.
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            qacc_logger.error(
                f"Failed to import {package_name}. Kindly refer to SDK documentation and install supported version of {package_name}"
            )
            sys.exit(1)
        else:
            if recommended_package_version:
                try:
                    detected_package_version = pkg_resources.get_distribution(package_name).version
                except:
                    detected_package_version = package.__version__
                if detected_package_version != recommended_package_version and package_name not in mismatched_packages:
                    qacc_logger.warning(
                        f"{package_name} installed version: {detected_package_version}, and Recommended version: {recommended_package_version}"
                    )
                    mismatched_packages.append(package_name)
            return package

    @classmethod
    def get_np_dtype(cls, dtype, map_tf=False):
        # TODO: TF import?
        """This method gives the appropriate numpy datatype for given data type
        Args:
            dtype  : onnx data type

        Returns:
            corresponding numpy datatype
        """
        # returns dtype if it is already a numpy dtype
        # else get the corresponding numpy datatype
        try:
            if dtype.__module__ == np.__name__:
                return dtype
        except AttributeError as e:
            if dtype.__class__ == np.dtype:
                dtype = dtype.name

        if (dtype == 'tensor(float)' or dtype == 'float' or dtype == 'float32'):
            return np.float32
        elif (dtype == 'tensor(int)' or dtype == 'int'):
            return np.int
        elif (dtype == 'tensor(float64)' or dtype == 'float64'):
            return np.float64
        elif (dtype == 'tensor(int64)' or dtype == 'int64'):
            return np.int64
        elif (dtype == 'tensor(int32)' or dtype == 'int32'):
            return np.int32
        elif dtype == 'tensor(bool)' or dtype == 'bool':
            return bool
        else:
            assert False, "Unsupported OP type " + str(dtype)
        if map_tf:
            tf = Helper.safe_import_package("tensorflow")
            if dtype == tf.float32: return np.float32
            elif dtype == tf.float64: return np.float64
            elif dtype == tf.int64: return np.int64
            elif dtype == tf.int32: return np.int32
            elif dtype == tf.bool: return bool

    @classmethod
    def get_model_type(cls, path):
        if os.path.isdir(path):
            return ModelType.FOLDER
        else:
            extn = os.path.splitext(path)[1]
        if extn == '.onnx':
            return ModelType.ONNX
        elif extn == '.pt':
            return ModelType.TORCHSCRIPT
        elif extn == '.pb':
            return ModelType.TENSORFLOW
        elif extn == ".tflite":
            return ModelType.TFLITE
        else:
            # TODO : support other model types.
            raise ce.UnsupportedException('model type not supported :' + path)

    @classmethod
    def tf_type_to_numpy(cls, type):
        """This method gives the corresponding numpy datatype for given tensorflow tensor element type
        Args:
            type : tensorflow tensor element type
        Returns:
            corresponding tensorflow datatype
        """
        # TODO: Add QINT dtypes
        tf_to_numpy = {
            1: np.float32,
            2: np.float64,
            3: np.int32,
            4: np.uint8,
            5: np.int16,
            6: np.int8,
            9: np.int64,
            10: bool
        }
        if type in tf_to_numpy:
            return tf_to_numpy[type]
        else:
            raise ce.UnsupportedException('Unsupported type : {}'.format(str(type)))

    @classmethod
    def ort_to_tensorProto(cls, type):
        """This method gives the appropriate numpy datatype for given onnx data type
        Args:
            type  : onnx data type

        Returns:
            corresponding numpy datatype
        """
        onnx = Helper.safe_import_package("onnx")
        if (type == 'tensor(float)' or type == 'float'):
            return onnx.TensorProto.FLOAT
        elif (type == 'tensor(int)' or type == 'int'):
            return onnx.TensorProto.INT8
        elif (type == 'tensor(float64)' or type == 'float64'):
            return onnx.TensorProto.DOUBLE
        elif (type == 'tensor(int64)' or type == 'int64'):
            return onnx.TensorProto.INT64
        elif (type == 'tensor(int32)' or type == 'int32'):
            return onnx.TensorProto.INT32
        else:
            assert ("TODO: fix unsupported OP type " + str(type))

    @classmethod
    def get_average_match_percentage(cls, outputs_match_percentage, output_comp_map):
        """Return the average match for all the outputs for a given
        comparator.
        """
        all_op_match = []
        for op, match in outputs_match_percentage.items():
            comparator = output_comp_map[op]
            comp_name = comparator.display_name()
            all_op_match.append(match[comp_name])

        return sum(all_op_match) / len(all_op_match)

    @classmethod
    def show_progress(cls, total_count, cur_count, info='', key='='):
        """Displays the progress bar."""
        completed = int(round(80 * cur_count / float(total_count)))
        percent = round(100.0 * cur_count / float(total_count), 1)
        bar = key * completed + '-' * (80 - completed)

        sys.stdout.write('[%s] %s%s (%s)\r' % (bar, percent, '%', info))
        sys.stdout.flush()

    @classmethod
    def validate_aic_device_id(self, device_ids: list[int]) -> bool:
        """Validate that the provided device IDs are recognized by the system.

        Args:
            device_ids: List of device IDs to validate.

        Returns:
            bool: True if all device IDs are valid.

        Raises:
            ce.ConfigurationException: If device IDs are invalid or cannot be retrieved.
        """
        try:
            valid_devices = [
                d.strip()
                for d in os.popen('/opt/qti-aic/tools/qaic-util -q |grep "QID"').readlines()
            ]
            device_count = len(valid_devices)
        except:
            raise ce.ConfigurationException(
                'Failed to get Device Count. Check Devices are connected and Platform SDK '
                'Installation')
        for dev_id in device_ids:
            if f'QID {dev_id}' not in valid_devices:
                raise ce.ConfigurationException(
                    f'Invalid Device Id(s) Passed. Device used must be one of '
                    f'{", ".join(valid_devices)}')
        return True

    @classmethod
    def dump_stage_error_log(self, logfile):
        with open(logfile) as f:
            log = f.read()
        qacc_file_logger.error(log)

    @classmethod
    def sanitize_node_names(cls, node_name):
        """Sanitize the node names to follow converter's node naming
        conventions.

        All special characters will be replaced by an
        underscore '_' and node names not beginning with an alphabet
        will be prepended with an underscore '_'.
        """
        if not isinstance(node_name, str):
            node_name = str(node_name)
        sanitized_name = re.sub(pattern='\\W', repl='_', string=node_name)
        if not sanitized_name[0].isalpha() and sanitized_name[0] != '_':
            sanitized_name = "_" + sanitized_name
        return sanitized_name

    @classmethod
    def sanitize_native_tensor_names(cls, tensor_names):
        """Sanitize the tensor names to follow converter's node naming
        conventions.
        tensor_names would be in the format graphName0:tensorName0,tensorName1;graphName1:tensorName0,tensorName1
        """
        tensor_names_list = tensor_names.split(';')
        sanitized_tensor_names = ''
        for tlist_str in tensor_names_list:
            # find the first occurrence of ':' as individual tensor names could have ':' in them
            tlist = tlist_str.split(':', 1)
            graph_name = tlist[0]
            tensors = tlist[1].split(',')
            for i, tensor in enumerate(tensors):
                tensors[i] = cls.sanitize_node_names(tensor)
            sanitized_tensors = ','.join(tensors)
            sanitized_tensor_names += graph_name + ':' + sanitized_tensors + ';'
        # remove the last ';' from the sanitized tensor names
        return sanitized_tensor_names[:-1]

    @classmethod
    def cli_params_to_list(cls, params: dict) -> list:
        """Convert given dictionary of QNN converter params to list of its CLI args."""
        args = []
        for param, value in params.items():
            if isinstance(value, bool):
                if value:
                    args.append(f'--{param}')
            # InferenceSchemaManager converts all values, including Boolean to str
            elif value == "True":
                args.append(f'--{param}')
            elif value == "False":
                continue
            else:
                # algorithms: default is used by Evaluator and not a valid converter option
                if param == "algorithms" and value == "default":
                    continue
                if param == "native_input_tensor_names" or param == "set_output_tensors":
                    value = cls.sanitize_native_tensor_names(value)
                    args.append(f'--{param} {value}')
                elif param == "extra_args":
                    args.append(value)
                else:
                    args.append(f'--{param} {value}')
        return args

    @classmethod
    def get_param_dtype(cls, param_str, return_val=False):
        """Determine if the given string is int, float or string. If return_val is True, return the
        val along with the type.

        Args:
            param_str: input string
            return_val: boolean to determine whether given input, after typecasting, has to be
                        returned or not
        Returns:
            type of the given input (int, float or string)
            actual value (if return_val is set to True)
        """
        try:
            val = int(param_str)
            if return_val:
                return int, val
            return int
        except:
            pass
        try:
            val = float(param_str)
            if return_val:
                return float, val
            return float
        except:
            pass
        if return_val:
            return str, param_str
        return str

    @classmethod
    def get_tensor_from_file(cls, filepath: str, datatype: str = 'float') -> np.ndarray:
        """Get corresponding tensor from given file path based on the datatype provided.
        If file does not exist, raise error
        Args:
            filepath: path to file
            datatype: datatype to determine size of tensor
        Returns:
            Numpy array representation of tensor
        Raises:
            FileNotFoundError: if input filepath does not exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'File path {filepath} does not exist.')
        tensor = np.fromfile(filepath, dtype=datatype)
        tensor = tensor.flatten()
        if datatype is bool:
            tensor = tensor.astype(np.int8)
        return tensor


class ComparatorHelper:
    """Utility class for comparator related helper methods"""

    @classmethod
    def get_out_names(cls, out_file: str) -> list:
        """Returns the names of the outputs from the out_file.
        Raises an error if the file does not exist.

        Args:
            out_file: path to file
        Returns:
           List of output names
        Raises:
           FileNotFoundError: if file does not exist
        """
        out_names = []
        if not os.path.exists(out_file):
            raise FileNotFoundError(f'File path {out_file} does not exist')
        with open(out_file) as ref_file:
            outputs = ref_file.readline().strip().split(',')
        out_names = [os.path.splitext(op.split('/')[-1])[0] for op in outputs]

        return out_names

    @classmethod
    def get_comparators(cls, comp_type: COMPARATORS = COMPARATORS.AVERAGE, tolerance: float = 0.001,
                        out_info: dict | None = None,
                        interpretation_strategy: str = 'max',
                        ref_out_file: str = None) -> tuple[list, list, list, list]:
        """Return the configured comparators and datatypes for each output
        Args:
            comp_type: Comparator type (average, cosine, etc.)
            tolerance: Tolerance threshold for comparison
            out_info: Details of the outputs and their names and datatypes
            ref_out_file: Path to output file
        Returns:
            output_names: list of names of the outputs
            output_comparators: list of comparators designated for each output
            output_comparator_dtypes: list of datatypes corresponding to each output
            output_comparator_names: list of names of all configured comparators
        Raises:
            FileNotFoundError: if ref_out_file does not exist
        """
        output_comparators = []
        output_comparator_names = []
        output_comparator_dtypes = []
        output_names = []

        if not out_info:
            if not ref_out_file:
                raise FileNotFoundError(
                    'Reference output file needs to be provided for comparator if output_info is None'
                )
            out_info = {}
            out_names = cls.get_out_names(ref_out_file)
            for oname in out_names:
                out_info[oname] = ['float32']

        for outname, val in out_info.items():
            if len(val) > 3:  # idx=2 is now filled with batch_dimension info
                # output specific comparator
                cmp = val[3]['type']
                tol_thresh = val[3].get('tol', 0.001)
                qacc_file_logger.info(f'Using output specific comparator : {cmp}')
            else:
                cmp = comp_type
                tol_thresh = float(tolerance)
            comparator_param = get_comparator_param(comparator=cmp)
            comparator_param.tol = tol_thresh
            _comparator = get_comparator(comparator=cmp, params=comparator_param)
            _comparator._interpretation_strategy = interpretation_strategy
            output_comparators.append(_comparator)
            output_comparator_names.append(cmp)
            output_comparator_dtypes.append(val[0])
            output_names.append(outname)
        return output_names, output_comparators, output_comparator_dtypes, \
               output_comparator_names

    @classmethod
    def get_top_deviating_samples(cls, comp_outputs: dict, comparator_name: COMPARATORS,
                                  inference_schema_inputs: list[list[str]], count: int) -> list:
        """Get the top deviating samples for each sample, based on the comparator type
        and return the list of filenames of the corresponding inputs
        Args:
            comp_outputs: Dict of comparator outputs with each output as key and corresponding
                          scores across all samples as value
            comparator_name: Name of the comparator
            inference_schema_inputs: List of filenames for pre-processed inputs for all samples
            count: Number of deviating samples to return
        Returns:
            List of filenames of the top deviating samples
        """
        num_outputs = len(comp_outputs.keys())
        """
        Get the average of the comparator scores for each output. For e.g. if there are 3 outputs
        of a model O1, O2 and O3 and comp_outputs = {O1: [0.0, 0.01], O2: [0.1, 0.01], O3: [0, 0.02]}
        then, avg_comp_scores = [avg(0.0,0.1,0), avg(0.01,0.01,0.02)]
        """
        avg_comp_scores = [
            sum(comp_scores) / num_outputs for comp_scores in zip(*comp_outputs.values())
        ]
        # get the indices of the avg scores in a sorted manner depending on the type of comparator
        top_deviating_idxs = np.argsort(avg_comp_scores)
        if comparator_name in [
                COMPARATORS.AVERAGE, COMPARATORS.L1NORM, COMPARATORS.L2NORM, COMPARATORS.MSE,
                COMPARATORS.STANDARD_DEVIATION, COMPARATORS.KLD
        ]:
            top_deviating_idxs = top_deviating_idxs[::-1]
        top_deviating_idxs = top_deviating_idxs[:count]

        deviating_filelist = []
        for inx in top_deviating_idxs:
            deviating_filelist.append(','.join(inference_schema_inputs[inx]))
        return deviating_filelist
