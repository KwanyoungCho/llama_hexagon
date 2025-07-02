# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from enum import Enum


class UnsupportedEncodingsVersionError(Exception):
    """Exception class for Unsupported encodings version"""

    pass


class EncodingVersion(Enum):
    """Enum class for supported encodings version"""

    LEGACY = "legacy"
    V1 = "1.0.0"


def add_encodings(
    encodings: dict,
    tensor_name: str,
    tensor_encoding: dict | list,
    encoding_type: str,
) -> dict:
    """Adds tensor encodings from tensor_info to the encodings dictionary inplace

    Args:
        encodings (dict): The encodings data structure to which tensor encodings need to be added.
        tensor_name (str): name of the tensor
        tensor_encoding (dict or list): encoding value for the given tensor name
        encoding_type (str): one of (activation_encodings, param_encodings)

    Returns:
        dict: Updated encodings data structure with encodings for tensor_info included

    Raises:
        UnsupportedEncodingsVersionError: If encodings version is not supported
        ValueError: If wrong encoding value is passed which is not in line with the
            encodings data structure.
    """
    try:
        version = get_encodings_version(encodings)
    except UnsupportedEncodingsVersionError as exception:
        raise exception

    if version == EncodingVersion.LEGACY:
        if isinstance(tensor_encoding, list):
            encodings[encoding_type][tensor_name] = tensor_encoding
        else:
            raise ValueError(
                f"Tensor encoding of type: {type(tensor_encoding)} is inconsistent with"
                f"the encodings version: {version}. Expected tensor encoding type: list"
            )
    elif version == EncodingVersion.V1:
        if isinstance(tensor_encoding, dict):
            # Delete the enodings for tensor_name if present
            for idx, enc in enumerate(encodings[encoding_type]):
                if enc["name"] == tensor_name:
                    del encodings[encoding_type][idx]
                    break

            encodings[encoding_type].append(tensor_encoding)
        else:
            raise ValueError(
                f"Tensor encoding of type: {type(tensor_encoding)} is inconsistent with"
                f"the encodings version: {version}. Expected tensor encoding type: dict"
            )

    return encodings


def get_encodings_structure(version: EncodingVersion) -> dict:
    """Given encodings version returns empty encodings data structure

    Args:
        version (EncodingVersion): encodings version enum

    Returns:
        (dict or None) encodings data structure

    Raises:
        UnsupportedEncodingsVersionError: If encodings version is not supported
    """
    if version == EncodingVersion.LEGACY:
        return {"activation_encodings": {}, "param_encodings": {}}
    if version == EncodingVersion.V1:
        return {"version": version.value, "activation_encodings": [], "param_encodings": []}

    raise UnsupportedEncodingsVersionError(
        f"Could not create encodings data structure for the given {version} version"
    )


def get_encodings_version(encodings: dict) -> EncodingVersion:
    """Given encodings file return encodings version
    Args:
        encodings (dict): encodings dictionary
    Returns:
        Enum: encoding version
    Raises:
        UnsupportedEncodingsVersionError: If version details could not be fetched for the
            given encodings
    """
    if (
        "activation_encodings" in encodings
        and "param_encodings" in encodings
        and isinstance(encodings["activation_encodings"], dict)
        and isinstance(encodings["param_encodings"], dict)
    ):
        return EncodingVersion.LEGACY
    if "version" in encodings:
        return EncodingVersion(encodings["version"])

    raise UnsupportedEncodingsVersionError(
        "Could not fetch version details for the provided encodings"
    )


def organize_qairt_encodings(qairt_encodings: dict) -> dict:
    """Organizes the user provided qairt encodings in the following format:
    {
        "activation_encodings": dict, with activation name as key and encodings as value
        "param_encodings": dict, with param name as key and encodings as value
    }

    Args:
        qairt_encodings (dict): qairt encodings

    Returns:
        dict: Organized qairt encodings

    Raises:
        UnsupportedEncodingsVersionError: If given version is not supported
    """
    try:
        version = get_encodings_version(qairt_encodings)
    except UnsupportedEncodingsVersionError as exception:
        raise exception
    # Organize the encodings for version legacy
    if version == EncodingVersion.LEGACY:
        user_encodings = qairt_encodings
    # Organize the encodings for version V1
    elif version == EncodingVersion.V1:
        user_encodings = {"activation_encodings": {}, "param_encodings": {}}
        for encoding_type in qairt_encodings:
            if encoding_type in ["activation_encodings", "param_encodings"]:
                for encoding in qairt_encodings[encoding_type]:
                    user_encodings[encoding_type][encoding["name"]] = encoding

    return user_encodings


def get_resolved_names(tensor_name: str) -> list:
    """Generates a list of resolved tensor names based on the input tensor name.

    Args:
        tensor_name (str): QNN tensor name.

    Returns:
        list: A list of resolved tensor names derived from the input tensor name.
    """
    # TODO: use framework_op_trace to resolve the target name once the feature is stable

    resolved_names = []

    if "_" in tensor_name:
        resolved_names.append("_".join(tensor_name.split("_")[:-1]))

    if "." in tensor_name:
        resolved_names.append(".".join(tensor_name.split(".")[:-1]))

    return resolved_names


def get_dtype(encodings: list) -> str:
    """Returns the data type for the tensor based on its encoding profile.

    Args:
        encodings (list): A list of dictionaries containing encoding information for a tensor.

    Returns:
        str: The data type ('float' or 'int') of the tensor.
    """
    if "dtype" in encodings[0] and encodings[0]["dtype"] == "float":
        return "float"

    if "scale" in encodings[0] and encodings[0]["scale"] != 0:
        return "int"

    # Default to 'float' if no specific dtype or scale is found
    return "float"


def needs_encoding_update(
    encoding1: dict | list, encoding2: dict | list, version: EncodingVersion
) -> bool:
    """Determines whether encoding1 should overwrite encoding2 based on precedence.

    The precedence order is: int16 > int8 > int4 > fp32 > fp16 > fp8.

    Args:
        encoding1 (dict or list): Encoding for a tensor, list for LEGACY and dict of V1
        encoding2 (dict or list): Encoding for a tensor, list for LEGACY and dict of V1
        version (EncodingVersion): encodings version

    Returns:
        bool: True if encoding1 should overwrite encoding2, False otherwise.

    Raises:
        ValueError: If encoding1 or encoding2 is not of type of list incase of LEGACY version
            or dict incase of V1 version
        UnsupportedEncodingsVersionError: If encoding version is not supported
    """
    if version == EncodingVersion.LEGACY:
        if isinstance(encoding1, list) and isinstance(encoding2, list):
            dtype1 = get_dtype(encoding1)
            dtype2 = get_dtype(encoding2)
            bitwidth1 = encoding1[0]["bitwidth"]
            bitwidth2 = encoding2[0]["bitwidth"]
        else:
            raise ValueError(f"Inccorect tensor encoding format w.r.t to version {version}")
    elif version == EncodingVersion.V1:
        if isinstance(encoding1, dict) and isinstance(encoding2, dict):
            dtype1 = encoding1["dtype"].lower()
            dtype2 = encoding2["dtype"].lower()
            bitwidth1 = encoding1["bw"]
            bitwidth2 = encoding2["bw"]
        else:
            raise ValueError(f"Inccorect tensor encoding format w.r.t to version {version}")
    else:
        raise UnsupportedEncodingsVersionError(f"Encoding version {version} is not supported")

    dtype_order = {"int": [4, 8, 16], "float": [8, 16, 32]}

    # Check if both data types are the same and compare bitwidths
    if dtype1 == dtype2:
        return dtype_order[dtype1].index(bitwidth1) > dtype_order[dtype2].index(bitwidth2)

    # Precedence: int types have higher precedence over float types
    return dtype1 == "int"


def identify_inter_activations_path(
    current_activation: str, parent_activation_name: str, target_activation_op_map: dict, depth: int
) -> list:
    """Identifies the path between child op activation and target op activation in the target graph.

    Args:
        current_activation: Child op activation in the target graph.
        parent_activation_name: Parent op activation in the target graph.
        target_activation_op_map: A mapping of target activations (keys) to target ops (values).
        depth: The current number of ops in the path between parent and child ops. If greater than
            10, the path is dropped as it may indicate loops.

    Returns:
        list: A list representing the path of activations.
    """
    # Base case: if the current activation is the parent activation
    if current_activation == parent_activation_name:
        return [parent_activation_name]

    # Base case: if the depth exceeds 10, return an empty path to avoid potential loops
    if depth >= 10:
        return []

    # Initialize the smallest path as empty
    shortest_path = []

    if current_activation in target_activation_op_map:
        current_target_op = target_activation_op_map[current_activation]

        # Iterate through the inputs of the current target op
        for op_input in current_target_op.inputs:
            path = identify_inter_activations_path(
                op_input, parent_activation_name, target_activation_op_map, depth + 1
            )
            #         |------------------>|
            # 100 --->|                   |-----> 103
            #         |--> 101 --> 102 -->|
            # Incase of residual connections, path between 100 and 103
            # should be {100, 103} but one other possible path is
            # {100, 101, 102, 103}. Therefore, we need to take the shortest path

            if path:
                if not shortest_path:
                    shortest_path = path
                else:
                    shortest_path = shortest_path if len(shortest_path) < len(path) else path

            # # Only update the smallest path if a valid path is found and it's shorter
            # if path and (not shortest_path or len(path) < len(shortest_path)):
            #     shortest_path = path

    if shortest_path:
        shortest_path.append(current_activation)

    return shortest_path


def is_convert_op_in_path(path: list, target_activation_op_map: dict) -> tuple:
    """Checks if there exists a convert operation in the path.

    Args:
        path: List of target activations.
        target_activation_op_map: Dictionary of target activations to target ops.

    Returns:
        tuple: (bool, str) indicating if 'Convert' op is found, and the activation name.
    """
    for activation in path:
        op = target_activation_op_map.get(activation)
        if op.op_type == "Convert":
            # if op and "convert" in op:
            return True, activation
    return False, None
