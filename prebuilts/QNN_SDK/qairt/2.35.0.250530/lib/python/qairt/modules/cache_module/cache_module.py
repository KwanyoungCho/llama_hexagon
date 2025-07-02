# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, final

from pydantic import Field, model_validator

from qairt.api.configs import BackendType, DevicePlatformType
from qairt.api.configs.common import AISWBaseModel
from qairt.modules.common.graph_info_models import GraphInfo, TensorInfo
from qairt.modules.properties import CacheMixin
from qairt.utils.asset_utils import AssetType, check_asset_type
from qairt.utils.exceptions import InvalidCacheError
from qti.aisw.core.model_level_api.target.target import Target


class HTPBackendInfo(AISWBaseModel):
    """
    Represents HTP beckend-specific metadata, such as architecture, vtcm_size, and optimization level
    Applicable mainly for HTP Caches
    """

    arch: int
    """The backend architecture/version."""
    vtcm_size: Optional[int] = None
    """The vtcm_size configuration."""
    optimization_level: int
    """The optimization level defined for HTP graph optimization algorithm"""


class CacheInfo(AISWBaseModel):
    """
    Represents the overall info for a single cache, including its graphs,
    SoC info, backend info, etc. This model is immutable
    """

    model_config = AISWBaseModel.model_config
    model_config.update(protected_namespaces=())

    # Make CacheInfo Immutable
    model_config = AISWBaseModel.model_config.copy()
    model_config.update(frozen=True)

    name: str
    """ The name or identifier of the cache."""
    graphs: List[GraphInfo]
    """A list of GraphInfo objects representing each graph in the cache."""
    soc_name: str
    """ The SoC name identifier i.e. SM8750 """
    backend: BackendType
    """ The backend type cache was compiled for """
    backend_info: HTPBackendInfo
    """ Additional metadata for the backend, pertaining to HTP such as architecture, vtcm_size size,
    and optimization level."""

    @model_validator(mode="after")
    def validate_vtcm_size_for_htp(self) -> "CacheInfo":
        """
        Validate that backend_info.vtcm_size is provided if the backend
        is HTP. Raises a ValueError if not satisfied.

        Returns:
            CacheInfo: The validated model instance.

        Raises:
            ValueError: If vtcm_size is missing when backend is HTP.
        """
        if self.backend == BackendType.HTP:
            if self.backend_info.vtcm_size is None:
                raise ValueError("vtcm_size is required with the HTP backend.")
        return self

    def as_dict(self):
        return self.model_dump(exclude_unset=True, exclude_none=True)


def convert_json_to_cache_info(data: dict, cache_name: str) -> CacheInfo:
    """
    Parse the given QNN System Context Binary Info JSON into a CacheInfo object.

    Args:
        data (dict): JSON Dictionary containing cache information
        cache_name (str): name of the cache
    Returns:
        CacheInfo: Constructed CacheInfo Object

    Raises:
        InvalidCacheError:
            If info is missing or empty in the JSON data passed in
    """
    info = data.get("info")
    if not info:
        raise InvalidCacheError(f"Error: Missing or empty info in JSON data for cache '{cache_name}'.")

    backend_id = info.get("backendId", BackendType.HTP.id)
    soc_version_str = info.get("socModel")
    if not soc_version_str:
        soc_version_str = "unknown"

    context_metadata = info.get("contextMetadata", {})
    context_metadata_info = context_metadata.get("info", {})
    arch = context_metadata_info.get("dspArch")

    vtcm_size = 0
    opt_level = 0

    # create list that stores GraphInfo objects
    graph_list = []
    graphs_json = info.get("graphs", [])
    for graph_json in graphs_json:
        graph_info, vtcm_size, opt_level = parse_graph_info(graph_json)
        graph_list.append(graph_info)

    backend = BackendType.from_id(backend_id)

    backend_info = HTPBackendInfo(arch=arch, vtcm_size=vtcm_size, optimization_level=opt_level)

    # construct the CacheInfo object
    cache_info = CacheInfo(
        name=cache_name,
        soc_name=str(soc_version_str),
        backend=backend,
        backend_info=backend_info,
        graphs=graph_list,
    )

    return cache_info


def parse_graph_info(graph_json: Dict[str, Any]) -> Tuple[GraphInfo, int, int]:
    """
    Parse the JSON data for a single graph and return a GraphInfo object.

    Args:
        graph_json (Dict[str,Any]): graph json dictionary to parse
    Raises:
        InvalidCacheError: If graph info is empty.

    Returns:
        Tuple[GraphInfo, int, int]: The created graph info object, vtcm, and optimization info
    """
    # Extract graph-level info
    graph_info = graph_json.get("info", {})
    vtcm_size, opt_level = 0, 0  # set default values

    if not graph_info:
        raise InvalidCacheError("Error: Graph is missing the required info field...")

    graph_name = graph_info.get("graphName", "unknown_graph")

    # Get graph inputs
    inputs_list = []
    for input_data in graph_info.get("graphInputs", []):
        input_info = input_data.get("info", {})
        dimensions = input_info.get("dimensions", [])
        inputs_list.append(
            TensorInfo(
                name=input_info.get("name", ""),
                dimensions=dimensions,
                data_type=input_info.get("dataType", "UNKNOWN"),
            )
        )

    # Get graph outputs
    outputs_list = []
    for output_data in graph_info.get("graphOutputs", []):
        output_info = output_data.get("info", {})
        dimensions = output_info.get("dimensions", [])
        outputs_list.append(
            TensorInfo(
                name=output_info.get("name", ""),
                dimensions=dimensions,
                data_type=output_info.get("dataType", "UNKNOWN"),
            )
        )

    # The HTP-specific Backend fields like vtcmSize and optimizationLevel are in 'graphBlobInfo'
    blob_info = graph_info.get("graphBlobInfo", {})

    blob_info = blob_info.get("info", {})
    vtcm_size = blob_info.get("vtcmSize", vtcm_size)
    opt_level = blob_info.get("optimizationLevel", opt_level)

    return (
        GraphInfo(
            name=graph_name,
            inputs=inputs_list,
            outputs=outputs_list,
        ),
        vtcm_size,
        opt_level,
    )


# TODO: Temp code. Turn this into a module and use device API
def get_cache_info_from_binary(path_to_binary: str | os.PathLike, path_to_json: str | os.PathLike) -> dict:
    """
    Uses the 'qnn-context-binary-utility' C++ command line tool to get the cache info from a binary.

    Args:
        path_to_binary (str | os.PathLike): The path to the binary.
        path_to_json (str | os.PathLike): The path to the file.

    Returns:
        dict: The JSON object as a dictionary.
    """

    # TODO: Remove this function once AISW-123397 is addressed
    def to_camel_case(s: str) -> str:
        """
        Converts a string with spaces into camelCase.

        Args:
            s (str): The string to convert, e.g. "soc model".

        Returns:
            str: The camelCase version of the string, e.g. "socModel".
        """
        parts = s.split(" ")
        return parts[0] + "".join(word.capitalize() for word in parts[1:])

    def replace_keys_with_camel_case(json_obj: Dict) -> Dict:
        """
        Recursively iterates over a JSON structure, converting dictionary keys to camelCase.

        Args:
            json_obj (Dict): A JSON Dict structure.

        Returns:
            Dict: The passed in JSON structure with all dictionary keys converted to camelCase.
        """
        if isinstance(json_obj, dict):
            new_obj = {}
            for key, value in json_obj.items():
                new_key = to_camel_case(key)
                new_obj[new_key] = replace_keys_with_camel_case(value)
            return new_obj
        else:
            return json_obj

    # Remove the JSON file if it already exists
    if os.path.exists(path_to_json):
        os.remove(path_to_json)

    # Construct the command to run
    _sdk_path = os.getenv("QNN_SDK_ROOT")
    if not _sdk_path:
        raise ValueError("QNN_SDK_ROOT environment variable not set. Please setup sdk environment.")

    # Run QNN context binary utility
    target = Target.create_host_target()
    if target.target_platform_type == DevicePlatformType.WOS:
        # There is no qnn-context-binary-utility.exe in arm64x-windows-msvc folder,
        # therfore we need to use aarch64-windows-msvc.
        target_name = "aarch64-windows-msvc"
    else:
        target_name = target.target_name

    if target.target_platform_type == DevicePlatformType.X86_64_LINUX:
        qnn_context_binary_utility = "qnn-context-binary-utility"
    elif target.target_platform_type in {DevicePlatformType.X86_64_WINDOWS_MSVC, DevicePlatformType.WOS}:
        qnn_context_binary_utility = "qnn-context-binary-utility.exe"
    else:
        raise ValueError(f"Unsupported target platform type: {target.target_platform_type}")

    qnn_context_binary_utility_path = os.path.join(_sdk_path, "bin", target_name, qnn_context_binary_utility)
    if not os.path.exists(qnn_context_binary_utility_path):
        raise FileNotFoundError(f"QNN context binary utility not found at {qnn_context_binary_utility_path}")
    context_binary_utility_command = (
        f"{qnn_context_binary_utility_path} --context_binary {path_to_binary} --json_file {path_to_json}"
    )

    process = subprocess.Popen(
        context_binary_utility_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"Failed to get cache info from binary: {path_to_binary}\n"
            f"Exit Code: {process.returncode}\n"
            f"Stdout:\n{stdout}\n"
            f"Stderr:\n{stderr}\n"
        )

    # Verify that the json file exists
    if not os.path.exists(path_to_json):
        raise IOError(f"Could not retrieve Cache Info for: {path_to_binary}")

    # Load the json file and return
    with open(path_to_json) as f:
        json_camel_case = replace_keys_with_camel_case(json.load(f))
        return json_camel_case


@final
class CacheModule(CacheMixin):
    """
    Class that encapsulates a serialized binary object.

    Executable and serializable via a reflection API using pybind.

    Attributes:
        path (str): The path to the cache binary.
        _info (CacheInfo): The cache info.
    """

    def __init__(
        self, path: str | os.PathLike = "", info: Optional[CacheInfo] = None, *, _internal_call: bool = False
    ):
        if not _internal_call:
            raise RuntimeError(
                "Cannot directly instantiate an instance of CacheModule. Use CacheModule.load() instead."
            )

        self.path = Path(path)
        self.src_path = self.path
        self._info: Optional[CacheInfo] = info
        self.working_directory: Optional[Path] = None
        self._cleanup = True

    @property
    def info(self) -> CacheInfo:
        if not self._info:
            return self.get_cache_info_from_path(self.path)
        return self._info

    @info.setter
    def info(self, info: CacheInfo) -> None:
        self._info = info

    @property
    def name(self) -> str:
        return self.info.name

    @classmethod
    def load(cls, *, path: str | os.PathLike = "", info: Optional[CacheInfo] = None) -> "CacheModule":
        """
        Load the cache from the given path.

        Args:
            path (str): The path to the cache binary.
            info (Optional[CacheInfo]): The CacheInfo for the cache.

        Note:
            Exactly one of path and info are to be specified.
        """
        if not path and not info:
            raise ValueError("Must set at least one of path and info.")

        if path and info:
            raise ValueError("Cannot set both path and info for CacheModule.load()")

        if path and not check_asset_type(AssetType.CTX_BIN, path):
            raise ValueError(f"Invalid cache binary file: {path}")

        return cls(path=Path(path), info=info, _internal_call=True)

    def save(self, path: str | os.PathLike = "") -> None:
        """Save the cache to the given path."""
        if not self.path:
            raise NotImplementedError("Cannot save module to disk if it did not originate on disk.")

        if not path:
            path = self.path

        shutil.copyfile(self.path, Path(path))
        self.path = Path(path)

    @classmethod
    def get_cache_info_from_path(cls, bin_path: str | os.PathLike) -> CacheInfo:
        """
        Resolve the cache info from the given path. If the info cannot be resolved, None is returned.

        Args:
            bin_path (str): The path to the cache binary.

        Returns:
            CacheInfo: The CacheInfo Object

        Raises:
            ValueError: If JSON fails to be parsed into CacheInfo.
        """
        bin_path = Path(bin_path)
        json_path = bin_path.parent / f"{bin_path.stem}_cache_info.json"

        # TODO Convert all JSON fields to CamelCase to unify formatting after AISW-116647
        cache_json = get_cache_info_from_binary(str(bin_path), str(json_path))

        try:
            return convert_json_to_cache_info(cache_json, bin_path.stem)
        except Exception as e:
            raise ValueError(f"Failed to parse cache info JSON into CacheInfo: {e}")

    def supported_backends(self) -> List[BackendType]:
        return [self.info.backend]

    def supported_platforms(self) -> List[DevicePlatformType]:
        return [bt_ for bt_ in DevicePlatformType.__members__.values()]

    def __del__(self):
        if hasattr(self, "working_directory") and hasattr(self, "_cleanup"):
            if self.working_directory and self._cleanup:
                shutil.rmtree(self.working_directory, ignore_errors=True)
