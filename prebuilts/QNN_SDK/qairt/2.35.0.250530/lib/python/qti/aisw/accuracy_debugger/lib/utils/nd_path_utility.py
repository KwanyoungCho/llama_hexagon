# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import re
import shutil
import json
from pathlib import Path

from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError, InferenceEngineError, DeviceError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine
from qti.aisw.accuracy_debugger.lib.utils.common import truncate_native_tag


def remove_file(file_path: str) -> None:
    '''
    deletes the file if it present

    :param file_path: path to file
    '''

    if os.path.exists(file_path):
        os.remove(file_path)

def get_absolute_path(dir, checkExist=True, pathPrepend=None):
    """
    Returns a absolute path
    :param dir: the relate path or absolute path
           checkExist: whether to check whether the path exists
    :return: absolute path
    """
    if not dir:
        return dir

    absdir = os.path.expandvars(dir)
    if not os.path.isabs(absdir):
        if not pathPrepend:
            absdir = os.path.abspath(absdir)
        else:
            absdir = os.path.join(pathPrepend, dir)

    if not checkExist:
        return absdir

    if os.path.exists(absdir):
        return absdir
    else:
        raise ParameterError(dir + "(relpath) and " + absdir + "(abspath) are not existing")


def get_tensor_paths(tensors_path):
    """Returns a dictionary indexed by k, of tensor paths :param tensors_path:
    path to output directory with all the tensor raw data :return:
    Dictionary."""
    tensors = {}
    for dir_path, sub_dirs, files in os.walk(tensors_path):
        for file in files:
            if file.endswith(".raw"):
                tensor_path = os.path.join(dir_path, file)
                tensor_name = file.rsplit('.raw', 1)[0]
                # Handle "native" tag in target outputs names
                tensor_name = truncate_native_tag(tensor_name)
                tensors[tensor_name] = tensor_path
    return tensors


def format_args(additional_args, ignore_args=[]):
    """Returns a formatted string(after removing args mentioned in ignore_args) to append to qnn/snpe core tool's commands
    :param additional_args: extra options to be addded to append to qnn/snpe core tool's commands
    commands :param ignore_args: list of args to be ignored :return: String."""
    extra_options = additional_args.split(';')
    extra_cmd = ''
    for item in extra_options:
        arg = item.strip(' ').split('=')
        if arg[0].rstrip(' ') in ignore_args:
            continue
        if len(arg) == 1:
            extra_cmd += '--' + arg[0].rstrip(' ') + ' '
        else:
            extra_cmd += '--' + arg[0].rstrip(' ') + ' ' + arg[1].lstrip(' ') + ' '
    return extra_cmd


def format_params(config_params):
    """Returns a dict of key-value pairs to be used in context/graph config
    :param config_params: extra config params to be added to context config / netrun config file
    :return: Dictionary."""
    extra_options = config_params.split(';')
    config_dict = {}
    for item in extra_options:
        arg = item.strip(' ').split('=')
        config_dict[arg[0]] = arg[1]
    return config_dict


def retrieveQnnSdkDir(filePath=__file__):
    filePath = Path(filePath).resolve()
    try:
        # expected path to this file in the SDK: <QNN root>/lib/python/qti/aisw/accuracy_debugger/lib/utils/nd_path_utility.py
        qnn_sdk_dir = filePath.parents[7]  # raises IndexError if out of bounds
        if (qnn_sdk_dir.match('qnn-*') or qnn_sdk_dir.match('qaisw-*')):
            return str(qnn_sdk_dir)
        else:
            qnn_path = filePath
            for _ in range(len(filePath.parts)):
                qnn_path = qnn_path.parent
                if (qnn_path.match('qnn-*') or qnn_path.match('qaisw-*')):
                    return str(qnn_path)

            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('QNN'))
    except IndexError:
        raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('QNN'))


def retrieveSnpeSdkDir(filePath=__file__):
    filePath = Path(filePath).resolve()
    try:
        # expected path to this file in the SDK: <SNPE root>/lib/python/qti/aisw/accuracy_debugger/lib/utils/nd_path_utility.py
        snpe_sdk_dir = filePath.parents[7]  # raises IndexError if out of bounds
        if (snpe_sdk_dir.match('snpe-*')) or snpe_sdk_dir.match('qaisw-*'):
            return str(snpe_sdk_dir)
        else:
            snpe_path = filePath
            for _ in range(len(filePath.parts)):
                snpe_path = snpe_path.parent
                if (snpe_path.match('snpe-*') or snpe_path.match('qaisw-*')):
                    return str(snpe_path)

            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('SNPE'))
    except IndexError:
        raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('SNPE'))


def retrieveQairtSdkDir(filePath=__file__):
    filePath = Path(filePath).resolve()
    try:
        # expected path to this file in the SDK: <QAIRT root>/lib/python/qti/aisw/accuracy_debugger/lib/utils/nd_path_utility.py
        qairt_sdk_dir = filePath.parents[7]  # raises IndexError if out of bounds
        if (qairt_sdk_dir.match('qairt-*')) or qairt_sdk_dir.match('qaisw-*'):
            return str(qairt_sdk_dir)
        else:
            qairt_path = filePath
            for _ in range(len(filePath.parts)):
                qairt_path = qairt_path.parent
                if (qairt_path.match('qairt-*') or qairt_path.match('qaisw-*')):
                    return str(qairt_path)

            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('QAIRT'))
    except IndexError:
        raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('QAIRT'))


def retrieveSdkDir(filePath=__file__):
    """Returns SDK path
    :param filePath: Engine path(default is current file path)
    :return: String"""
    filePath = Path(filePath).resolve()
    try:
        # expected path to this file in the SDK: <SDK_ROOT>/lib/python/qti/aisw/accuracy_debugger/lib/utils/nd_path_utility.py
        sdk_dir = filePath.parents[7]  # raises IndexError if out of bounds
        # QAIRT converters/quantizers are present in QNN, SNPE and QAIRT SDKs
        if sdk_dir.match('qaisw-*'):
            return str(sdk_dir)
        elif sdk_dir.match('qairt/*'):
            return str(sdk_dir)
        else:
            return None
    except IndexError:
        return None


def get_sdk_type(engine_path):
    """Returns SDK type(QNN/SNPE/QAIRT) from given engine_path
    :param engine_path: SDK path
    :return: String"""
    sdk_share_path = os.path.join(engine_path, 'share')
    try:
        share_folders = os.listdir(sdk_share_path)
        if Engine.QNN.value in share_folders and Engine.SNPE.value in share_folders:
            return Engine.QAIRT.value
        elif Engine.QNN.value in share_folders:
            return Engine.QNN.value
        elif Engine.SNPE.value in share_folders:
            return Engine.SNPE.value
        else:
            raise InferenceEngineError(
                f"Failed while fetching SDK type, expected {sdk_share_path} path to have QNN or SNPE folders but found {share_folders}"
            )
    except:
        raise InferenceEngineError(
            f"Failed while fetching SDK type, expected {sdk_share_path} path to have QNN or SNPE folders but found None"
        )


def santize_node_name(node_name):
    """Santize the Node Names to follow QNN Converter Node Naming Conventions
    All Special Characters will be replaced by '_' and numeric node names
    will be modified to have preceding '_'."""
    if not isinstance(node_name, str):
        node_name = str(node_name)

    # As per QNN sanitization replace all special characters with "_"
    sanitized_name = re.sub(pattern='\\W', repl='_', string=node_name)

    # Add extra "_" at the beginning of tensor name if first character is numeric
    if sanitized_name and not sanitized_name[0].isalpha() and sanitized_name[0] != '_':
        sanitized_name = "_" + sanitized_name

    return sanitized_name


def sanitize_output_tensor_files(output_directory, dlc_tensor_names):
    """
    This utility transforms SNPE outputs in the format of QNN outputs by performing sanitization.
    Moves output raw files from sub folders to 'output_directory' and sanitizes raw file names
    Example: output_directory/conv/tensor_0.raw will become output_directory/_conv_tensor_0.raw
    param output_directory: path to output raw file directory.
    return: status code 0 for success and -1 for failure.
    """

    if not output_directory or not os.path.isdir(output_directory):
        return -1
    for root, folders, files in os.walk(output_directory):

        if not files:
            continue
        rel_path = os.path.relpath(root, output_directory)
        if rel_path == ".":
            rel_path = ""

        for file in files:
            if not file.endswith(".raw"):
                continue
            snpe_tensor_name = os.path.join(rel_path, file)
            snpe_tensor_name, file_extension = os.path.splitext(snpe_tensor_name)
            sanitized_name = santize_node_name(snpe_tensor_name)

            # Incase of a snpe node name like "/tmp", make sure that we replace first character "/" with "_"
            # Basically it should be "_tmp" which mimics sanitization done by QNN converters
            # sanitized_name till this point will be just "tmp" since first "/" won't be captured by above logic
            # so we need to refer tensor names in dlc to check if there is a "/" at the beginning and add "_" to sanitized_name
            if snpe_tensor_name not in dlc_tensor_names and "/" + snpe_tensor_name in dlc_tensor_names and not snpe_tensor_name[
                    0].isdigit():
                sanitized_name = "_" + sanitized_name

            new_file = sanitized_name + file_extension
            shutil.move(os.path.join(root, file), os.path.join(output_directory, new_file))

    for item in os.listdir(output_directory):
        full_path = os.path.join(output_directory, item)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
    return 0

def validate_aic_device_id(device_ids):
    """
    Validate the provided AIC device IDs against the list of connected devices.

    Parameters:
    device_ids (list): List containing the device IDs to be validated.

    Returns:
    bool: True if all device IDs are valid, raises DeviceError otherwise.

    Raises:
    DeviceError: If the device count cannot be retrieved or if any device ID is invalid.
    """
    try:
        # Retrieve the list of valid device IDs by running the qaic-util command
        valid_devices = [
            d.strip() for d in os.popen('/opt/qti-aic/tools/qaic-util -q | grep "QID"').readlines()
        ]
        device_count = len(valid_devices)
    except Exception as e:
        # Raise an error if the device count cannot be retrieved
        raise DeviceError(
            'Failed to get Device Count. Check if devices are connected and Platform SDK is installed.'
        ) from e

    # Validate each provided device ID
    for dev_id in device_ids:
        if f'QID {dev_id}' not in valid_devices:
            # Raise an error if any device ID is invalid
            raise DeviceError(f'Invalid Device ID(s) passed. Device used must be one of '
                              f'{", ".join(valid_devices)}')

    return True

def process_aic_devices(parsed_args, config_attr_name="backend_extension_config"):
        """
        Processes AIC devices based on the provided arguments.

        This function validates the device IDs provided either through the command line argument or
        a configuration file.
        It ensures that the device IDs match between the two sources if both
        are provided.
        If only the device ID is provided, it creates a configuration file with the
        device ID.
        It updates the config file with the device ID in case config file does not have device_id but
        provided by the argument.

        Args:

            parsed_args: The parsed command line arguments which include:
                - deviceId: List of device IDs provided through the command line.
                - output_dir: Directory where the configuration file should be saved if created.
            config_attr_name: The attribute name for the configuration file path(i.e backend_extension_config,
            qnn_netrun_config_file,).

        Raises:
            DeviceError: If the device IDs from the command line and the configuration file do not match.
        """
        device_id = parsed_args.deviceId
        config_file_path = getattr(parsed_args, config_attr_name)
        config_device_id = None

        # Validate the device ID if provided through the command line
        if device_id:
            validate_aic_device_id(device_id)

        # If a configuration file path is provided, read and validate the device ID from the file
        if config_file_path:
            with open(config_file_path, 'r') as file:
                config = json.load(file)
                config_device_id = config.get("runtime_device_ids")
                if config_device_id is not None:
                    config_device_id = [str(id) for id in config_device_id]
                    if config_device_id:
                        validate_aic_device_id(config_device_id)

        # Raise an error if the device IDs from the command line and the configuration file do not match
        if device_id and config_device_id and sorted(device_id) != sorted(config_device_id):
            raise DeviceError(f'Device ID(s) passed in {config_attr_name} file: {config_device_id} '
                              f'and Device ID(s) passed by --deviceId argument: {device_id} '
                              f'are not matching')

        # Create a configuration file with the device ID if only the device ID is provided
        if device_id and not config_file_path:
            config_dict = {"runtime_device_ids": [int(id) for id in device_id]}
            config_file_path = os.path.join(parsed_args.output_dir, "{}.json".format("aic_"+config_attr_name))
            with open(config_file_path, 'w') as file:
                json.dump(config_dict, file)
            setattr(parsed_args, config_attr_name, get_absolute_path(config_file_path))

        # Update the configuration file with the device ID if the file is provided but does not contain the device ID
        if device_id and config_file_path and config_device_id is None:
            with open(config_file_path, 'r+') as file:
                config = json.load(file)
                config["runtime_device_ids"] = [int(id) for id in device_id]
                file.seek(0)
                json.dump(config, file)
                file.truncate()
