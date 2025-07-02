# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import shutil
import tempfile
import uuid
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qairt.api.configs.common import BackendType
from qairt.api.configs.device import DevicePlatformType
from qairt.modules.cache_module import CacheInfo, CacheModule
from qairt.modules.dlc_module.dlc_utils import DlcInfo, GraphDescriptor, GraphInfo, _get_dlc_updater
from qairt.modules.properties import DlcMixin
from qairt.modules.qti_import import qairt_tools_cpp
from qairt.utils import loggers
from qairt.utils.asset_utils import AssetType, check_asset_type
from qairt.utils.exceptions import InvalidCacheError


class DlcModule(DlcMixin):
    """
    Representative class for a DLC in memory

    Enables:
     - loading of DLCs on disk
     - serialization of graphs and caches into DLCs.
     - reflection of graphs and caches contained within a DLC.
    """

    _graph_handles: Dict[str, GraphDescriptor]
    _logger = loggers.get_logger(name=__name__)

    @classmethod
    def load(
        cls,
        path: Union[str, PathLike],
        working_dir: Union[str, PathLike] = "",
        cleanup: bool = True,
        **updater_args,
    ) -> "DlcModule":
        """
        Class method to instantiate a DlcModule and load the DLC from disk.
        Calls the constructor with the necessary internal key.

        Args:
            path: Path to the DLC file to load.
            working_dir: An optional working directory to use for temporary DLC edits.
                         - If provided, that directory will be used directly for temporary files
                           and will be deleted upon Module's garbage collection
                         - If omitted, a new temporary directory is created in the parent of `path`.
            cleanup: Whether to delete the working directory on cleanup. Defaults to True.
            **updater_args: Any additional arguments passed to the underlying dlc updater.

        Returns:
            A new DlcModule instance with the DLC loaded.
        """

        return cls(
            path=path, working_dir=working_dir, cleanup=cleanup, _internal_key="private", **updater_args
        )

    def __init__(
        self,
        path: Union[str, PathLike],
        working_dir: Union[str, PathLike] = "",
        cleanup: bool = True,
        *,
        _internal_key: str = "",
        **updater_args,
    ):
        """
        Initializes a new DlcModule. Direct calls to DlcModule() are not supported unless _internal_key=="private".
        The recommended usage is DlcModule.load(...)

        Args:
            path (Union[str, PathLike]): The path to the DLC file to be loaded. Must be provided

            working_dir (Union[str, PathLike]): Optional directory for temporary DLC edits.
                If omitted, a new temporary directory is created in the parent directory of path.

            cleanup (bool): Whether to delete the working directory on cleanup. Defaults to True.

            _internal_key (str): An internal flag. If it is not "private",
                a RuntimeError is raised. This enforces usage
                of the classmethod DlcModule.load.

            updater_args (dict): Any additional arguments passed to the underlying DLC updater.

        :raises RuntimeError:
            If called without passing _internal_key="private".
        """
        super().__init__()

        if _internal_key != "private":
            raise RuntimeError("Please use DlcModule.load(...) to create a new module.")

        self._updater_args: dict = updater_args
        self._caches: Dict[str, CacheModule] = OrderedDict()
        self._graph_handles = OrderedDict()
        self._cleanup = cleanup
        self._graphs: Optional[Dict[str, GraphInfo]] = None

        self.src_path: Path = Path(path).resolve()
        self.path = self.src_path
        if not check_asset_type(AssetType.DLC, path):
            raise TypeError("Not a valid DLC asset")

        # Default to container path parent path upon load, else use current working directory
        self.working_directory: Path
        if not working_dir:
            self.working_directory = Path(tempfile.mkdtemp(prefix="temp_dlc_dir_", dir=self.src_path.parent))
        else:
            self.working_directory = Path(working_dir).resolve()
            self.working_directory.mkdir(parents=True, exist_ok=True)

        self._temp_dlc_path = self.working_directory / self.path.name

        try:
            self._logger.debug(f"Loading container asset from '{path}'")

            self._updater: "qairt_tools_cpp.DlcUpdater" = _get_dlc_updater(
                str(self.path),
                str(self._temp_dlc_path),
                **self._updater_args,
            )

            # Open the DLC and set it to an editable state.
            self._updater.open(str(path))

            # Load DLCInfo
            self._info: DlcInfo = DlcInfo(
                copyright=self._updater.copyright(),
                model_version=self._updater.custom_model_version(),
                graphs=list(self.graphs_info.values()),
            )

            self._logger.debug("DLC loaded successfully.")

            # Now populate the caches
            self._populate_caches_from_dlc()

        except Exception as e:
            raise IOError("Failed to load container asset") from e

    def __del__(self):
        """
        Cleans up the temporary directory if it exists when garbage collection occurs.
        """
        self._cleanup_temp_dlc_dir()

    def _cleanup_temp_dlc_dir(self):
        """Remove the entire temporary directory and temporary files inside if user _cleanup is set."""
        if not self._cleanup:
            # Do not delete working directory
            return

        if self.working_directory.exists():
            shutil.rmtree(self.working_directory, ignore_errors=True)
            self._logger.debug(f"Cleaned up temporary DLC Module directory: {self.working_directory}")

    @property
    def caches(self) -> Dict[str, CacheModule]:
        """Returns a dictionary of all caches in the module."""
        return self._caches

    def get_cache(self, name: str) -> CacheModule:
        """Returns a cache module by name."""
        return self._caches[name]

    @property
    def graphs(self) -> Dict[str, GraphDescriptor]:
        """Returns a dictionary of all graphs in the module."""
        if self._graph_handles:
            return self._graph_handles

        # Build the graph handles from available information
        for graph_name in self._updater.get_ir_graph_names():
            ir_graph = self._updater.get_ir_graph(graphName=graph_name)
            self._graph_handles[graph_name] = GraphDescriptor(graph=ir_graph)
        self._exec_ready = False
        self._logger.debug("DLC loaded successfully.")

        return self._graph_handles

    @graphs.setter
    def graphs(self, *graphs: GraphDescriptor):
        for graph_ in graphs:
            name = getattr(graph_, "name")
            self._graph_handles[name] = graph_

    @graphs.deleter
    def graphs(self):
        if self._graph_handles:
            del self._graph_handles
            self._graph_handles = {}

    @property
    def graphs_info(self) -> Dict[str, GraphInfo]:
        """
        Returns a dictionary of all graphs in the module.
        """
        return {graph_name: graph.info for graph_name, graph in self.graphs.items()}

    def graph_names(self) -> List[str]:
        """
        Returns a list of graph names in the DLC.
        """
        return list(self.graphs.keys())

    @property
    def updater(self) -> "qairt_tools_cpp.DlcUpdater":
        return self._updater

    def _populate_caches_from_dlc(self) -> None:
        """
        Enumerates through the caches in the currently opened DLC and populates
        self._caches with CacheModule instances, each containing metadata
        (graphs, accelerator info, etc.) in its CacheInfo object.
            - First this method retrieves a list of available cache names from the updater.
            - Then for each cache:
                - Gets critical metadata (e.g., backend type, SoC model, DSP arch).
                - Gets all graph information within the cache, collecting inputs/outputs and HTP-specific
                  fields (vtcm_size and optimization_level) for each graph.
                - Constructs a CacheInfo object with all critical metadata
                - Stores the CacheInfo object in a new CacheModule, which is then
                  inserted into self._caches under the cache's name.

        Raises:
            InvalidCacheError: If any critical data (backend type, SoC model name, DSP arch,
                        number of graphs, graph names/inputs/outputs, VTCM, or optimization
                        level) cannot be retrieved from the updater for a cache and its graphs.

        Returns:
            None
        """
        for cache_name in self.updater.get_cache_info_names():
            # Get critical Cache metadata
            try:
                backend_id = self.updater.get_cache_backend_type_id(cache_name)
            except Exception:
                raise InvalidCacheError(f"Missing critical backend_type_id for cache '{cache_name}'.")

            try:
                soc = self.updater.get_cache_soc_model_name(cache_name)
            except Exception:
                raise InvalidCacheError(f"Missing critical soc model name for cache '{cache_name}'.")

            # backend name from backend ID
            backend = BackendType.from_id(backend_id)

            graphs_list = []
            try:
                num_graphs = self.updater.get_cache_num_of_graphs(cache_name)
            except Exception:
                raise InvalidCacheError(f"Missing critical number of graphs for cache '{cache_name}'.")

            vtcm = 0
            opt_level = 0

            try:
                arch = self.updater.get_cache_dsp_arch(cache_name)
            except Exception:
                raise InvalidCacheError(f"Missing critical dsp architecture for cache '{cache_name}'.")

            # Get Graph Information
            for graph_index in range(num_graphs):
                # Graph name
                try:
                    graph_name = self.updater.get_cache_graph_name_by_index(cache_name, graph_index)
                except Exception:
                    raise InvalidCacheError(
                        f"Missing critical name of graph at index '{graph_index}' for cache '{cache_name}'."
                    )

                # Get inputs
                inputs_list = []
                try:
                    input_num = self.updater.get_cache_graph_input_num_by_index(cache_name, graph_index)
                except Exception:
                    raise InvalidCacheError(
                        f"Missing critical number of inputs of graph at index '{graph_index}'"
                        f" for cache '{cache_name}'."
                    )

                for input in range(input_num):
                    input_name = self.updater.get_cache_graph_input_name_by_index(
                        cache_name, graph_index, input
                    )
                    input_dims = self.updater.get_cache_graph_input_dimension_by_index(
                        cache_name, graph_index, input
                    )
                    input_dtype = self.updater.get_cache_graph_input_datatype_by_index(
                        cache_name, graph_index, input
                    )

                    inputs_list.append(
                        {"name": input_name, "dimensions": input_dims, "data_type": input_dtype}
                    )

                # Get outputs
                outputs_list = []
                try:
                    output_num = self.updater.get_cache_graph_output_num_by_index(cache_name, graph_index)
                except Exception:
                    raise InvalidCacheError(
                        f"Missing critical number of outputs of graph at index '{graph_index}' "
                        f"for cache '{cache_name}'."
                    )

                for output in range(output_num):
                    output_name = self.updater.get_cache_graph_output_name_by_index(
                        cache_name, graph_index, output
                    )
                    output_dims = self.updater.get_cache_graph_output_dimension_by_index(
                        cache_name, graph_index, output
                    )
                    output_dtype = self.updater.get_cache_graph_output_datatype_by_index(
                        cache_name, graph_index, output
                    )

                    outputs_list.append(
                        {"name": output_name, "dimensions": output_dims, "data_type": output_dtype}
                    )

                # HTP-specific fields for this graph (if relevant)
                try:
                    vtcm = self.updater.get_htp_cache_graph_vtcm_size_by_index(cache_name, graph_index)
                except Exception:
                    raise InvalidCacheError(
                        f"Missing critical vtcm of graph at index '{graph_index}' for cache '{cache_name}'."
                    )

                try:
                    opt_level = self.updater.get_htp_cache_graph_optimization_level_by_index(
                        cache_name, graph_index
                    )
                except Exception:
                    raise InvalidCacheError(
                        f"Missing critical optimization_level of graph at index '{graph_index}' "
                        f"for cache '{cache_name}'."
                    )

                graph_info = {"name": graph_name, "inputs": inputs_list, "outputs": outputs_list}
                graphs_list.append(graph_info)

            # Build accelerator info dictionary
            accelerator_info = {"arch": arch, "vtcm_size": vtcm, "optimization_level": opt_level}

            info_dict = {
                "name": cache_name,
                "soc_name": soc,
                "backend": backend,
                "backend_info": accelerator_info,
                "graphs": graphs_list,
            }

            # store this CacheModule in DLCModule
            self._caches[cache_name] = CacheModule.load(info=CacheInfo(**info_dict))

    def list_caches(self) -> list[str]:
        """
        Return a list of cache names in the DLC.
        """
        return list(self._caches.keys())

    def embed_caches(self, *caches: CacheModule | str) -> None:
        """
        Embed caches into this dlc module.

        This operation is performed in place, as such, the DLC will be modified.

        Arguments:
            caches (List[CacheModule | str]): The caches to embed. Either a cache module or path
            to a cache on disk can be passed.

        Raises:
            RuntimeError: If the DLC is not in a valid state to embed caches.
            ValueError: If a cache property cannot be set
            LoadAssetError: If a cache cannot be loaded from disk.
        """

        for cache in caches:
            if isinstance(cache, str):
                cache = CacheModule.load(path=cache)
            if not cache.path:
                raise RuntimeError(f"Cache {cache.name} has never existed on disk.")
            try:
                self.updater.add_record(str(cache.path), qairt_tools_cpp.DlcRecordType.HTP_CACHE_RECORD)
            except Exception as cache_add_error:
                raise RuntimeError(f"Failed to embed cache {cache.name}") from cache_add_error

            self._caches.update({cache.name: CacheModule.load(info=cache.info)})

    def extract_caches(
        self,
        output_path: str,
        filters: Optional[dict[str, Any]] = None,
        record_type: qairt_tools_cpp.DlcRecordType = qairt_tools_cpp.DlcRecordType.HTP_CACHE_RECORD,
    ) -> List[CacheModule]:
        """
        Extract caches from the DLC based on one or more key/value pairs that align with CacheInfo.
        If no filters provided, will extract all caches by default

        Args:
            output_path (str):
                The path where extracted binaries will be saved, can't be inside working_directory

            filters (dict[str, Any], optional):
                A dictionary of filters where key corresponds to a field
                (like "name", "soc_name", or "backend") and value is the
                expected match.

                Examples:
                    {"name": "backend.metadata0"}
                    {"backend": BackendType.HTP, "soc_name": "SM8750"}

            record_type:
                The type of record to extract (defaults to HTP_CACHE_RECORD).

        Raises:
            RuntimeError: If DLC is not loaded.
            ValueError: If a filter key is invalid (i.e. does not exist in CacheInfo).
            ValueError: If output_path provided is inside working directory

        Returns:
            List[CacheModule]: The list of extracted cache modules that matched the filters.
        """

        def is_subpath(child: Path, parent: Path) -> bool:
            """Return True if child is inside parent path"""
            try:
                child.relative_to(parent)
                return True
            except ValueError:
                return False

        resolved_output_path = Path(output_path).resolve()
        if self.working_directory and (
            resolved_output_path == self.working_directory.resolve()
            or is_subpath(resolved_output_path, self.working_directory.resolve())
        ):
            raise ValueError("output_path provided can't be inside the temporary output dlc directory")

        if not filters:
            self._logger.debug("No filters provided. Extracting all caches...")
            filters = {}

        cache_modules_extracted = []
        cache_names_extracted = []

        for cache_name, cache_module in self._caches.items():
            cache_info = cache_module.info

            if not cache_info:
                continue

            # Check each key-value pair provided in filters with each cache
            match_found = True
            for key, expected_value in filters.items():
                # Try to get the key attribute from CacheInfo
                if not hasattr(cache_info, key):
                    raise ValueError(f"Invalid filter key '{key}' for CacheInfo")

                actual_value = getattr(cache_info, key)

                if actual_value != expected_value:
                    match_found = False
                    break

            if not match_found:
                # this cache doesn't match all filters, skip extraction
                continue

            # Attempt extraction
            if self._extract(cache_name, str(output_path), record_type):
                cache_module.path = resolved_output_path / f"{cache_name}.bin"
                cache_modules_extracted.append(cache_module)
                cache_names_extracted.append(cache_name)

        # Remove extracted caches from self._caches
        for extracted_cache_name in cache_names_extracted:
            self._caches.pop(extracted_cache_name, None)

        return cache_modules_extracted

    def _extract(
        self,
        cache_name: str,
        output_path: str,
        record_type: qairt_tools_cpp.DlcRecordType = qairt_tools_cpp.DlcRecordType.HTP_CACHE_RECORD,
    ) -> bool:
        """
        Helper method to do the actual extraction step and removal of cache module from DLC.
        Args:
            cache_name (str): name of the cache for extraction
            output_path (str): The path for the extracted binary.
            record_type: The type of record to extract.
        Returns:
            bool: If cache extraction was successful
        """

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            success = self.updater.extract_record_to_path(cache_name, str(output_path))

            if success:
                self._logger.debug(f"Extracted cache '{cache_name}' to '{output_path}'.")
            else:
                self._logger.error(f"Extraction indicated failure for '{cache_name}'.")
            return success
        except Exception as e:
            self._logger.error(f"Failed to extract cache '{cache_name}': {e}")
            return False

    def save(self, path: str | os.PathLike = "") -> str:
        """
        Creates a DLC on disk from the module instance. This function changes the underlying
        path pointed to by the module.

        Args:
            path (str|os.PathLike): The path to save the DLC. If no path is provided, the DLC will be saved to
                                    a default location. The default location will either be the original path
                                    on the first call to save or the last path saved to on subsequent calls.
        Returns:
            str: The path to the saved DLC.
        """

        # If no path is provided and there is no default path, then error
        if not path:
            if self.path:
                path = self.path
            else:
                raise IOError("Cannot save DLC Module without a specified path.")

        self.updater.save()  # saves to self._temp_dlc_path
        self.updater.finish()  # writes to self._temp_dlc_path

        # Retrieve updated DLC path and move to new path
        # After this call the temporary dlc path won't exist until save is called again
        if os.path.exists(self._temp_dlc_path):
            # We need to do a copy and remove instead of a move
            # because os.rename is not atomic on Windows
            shutil.copy(self._temp_dlc_path, path)
            try:
                os.remove(self._temp_dlc_path)
            except PermissionError:
                # TODO: This is sometimes needed on windows due to locks on deleting files
                # while there are being referenced. In this case, self._updater may still referencing
                # the dlc because close has not been called. The actual fix would be to reset the updater
                # but we need to handle how to pass in the args from load.
                self._logger.warning(
                    f"Could not delete temporary artifacts. Artifacts may persist at {self._temp_dlc_path} "
                )
                pass
        else:
            raise IOError(f"Failed to save DLC. Path: {path} does not exist")

        self.path = Path(path)

        return str(path)

    @property
    def info(self) -> DlcInfo:
        """Returns read-only information about the DLC associated with this module"""
        return self._info

    def supported_backends(self) -> List[BackendType]:
        """Returns a list of supported DLC backends."""
        # TODO: implement this
        return [bt_ for bt_ in BackendType.__members__.values()]

    def supported_platforms(self) -> List[DevicePlatformType]:
        """Returns a list of supported DLC platforms."""
        # TODO: implement this
        return [bt_ for bt_ in DevicePlatformType.__members__.values()]
