# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import numpy as np

def append_arg(arg: str, destination: str):
    """
    This utility appends given arg to the given destination variable
    Returns: destination variable after appending given arg
    """

    if destination is None:
        destination = arg
    else:
        destination += ' ' + arg

    return destination


def convert_data(data_list, output_dir=None, output_folder_name=None, user_provided_dtypes=None):
    """
    Since underlying QNN tools are not supporting 64 bit inputs (except AIC) but our framework
    diagnosis works on 64 bit, if model accepts 64 bit inputs, so we need to reconvert the user
    provided 64 bit inputs through the input_list into 32 bits.
    Here, we loop over each line in the input_list.txt and convert the 64 bit input and dump them
    into the working directory and finally we create new input_list containing 32 bit input paths.
    Additionally, this function converts boolean data to np.uint8 (np.bool not supported as of now)
    """
    if data_list is None: return

    converted_files_dir = os.path.join(output_dir, output_folder_name)
    os.makedirs(converted_files_dir, exist_ok=True)

    new_data_list_path = os.path.join(converted_files_dir, output_folder_name+'.txt')
    new_data_list_file = open(new_data_list_path, 'w')

    with open(data_list, 'r') as file:
        for line in file.readlines():
            line = line.rstrip().lstrip().split('\n')[0]
            if line:
                input_name_and_paths = [
                    input_name_and_path.split(':=')
                    if ':=' in input_name_and_path else [None, input_name_and_path]
                    for input_name_and_path in line.split()
                ]
                new_input_name_and_path = []
                for user_provided_dtype, input_name_and_path in zip(
                        user_provided_dtypes, input_name_and_paths):
                    user_provided_tensor = np.fromfile(input_name_and_path[1],
                                                        dtype=user_provided_dtype)

                    converted_tensor = user_provided_tensor.astype(np.float32)

                    file_name = os.path.join(converted_files_dir,
                                                os.path.basename(input_name_and_path[1]))
                    converted_tensor.tofile(file_name)

                    if input_name_and_path[0] is not None:
                        new_input_name_and_path.append(input_name_and_path[0] + ":=" + file_name)
                    else:
                        new_input_name_and_path.append(file_name)
                new_data_list_file.write(" ".join(new_input_name_and_path) + "\n")
    new_data_list_file.close()
    return new_data_list_path


def update_model_path(args, model_path):
    if '-m' in list(args):
        args[args.index('-m')] = '--model_path'
    args[args.index('--model_path') + 1] = model_path
    return args


def truncate_native_tag(tensor_name: str) -> str:
    """
    Truncates '_native' tag if it exists in the given tensor_name

    :param tensor_name: Name of the QNN tensor
    """
    if tensor_name.endswith('_native'):
        tensor_name = tensor_name.replace('_native', '')
    return tensor_name