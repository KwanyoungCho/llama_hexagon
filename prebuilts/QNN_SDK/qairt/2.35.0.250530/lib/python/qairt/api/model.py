# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, overload

import numpy
from typing_extensions import Unpack

from qairt.api.bases.model_base import ModelBase
from qairt.api.configs import (
    BackendType,
    Device,
    DevicePlatformType,
    ExecutionInputData,
    ExecutionResult,
    RemoteDeviceIdentifier,
    Target,
)
from qairt.api.executor.execution_config import ExecutionConfig, supported_initialize_execute_platforms
from qairt.api.profiler.profiler import profile
from qairt.modules.cache_module.cache_module import CacheModule
from qairt.modules.common.graph_info_models import TensorInfo
from qairt.modules.dlc_module import DlcModule
from qairt.modules.qti_import import qti_module_api
from qairt.utils.asset_utils import AssetMapping, AssetType, check_asset_type
from qairt.utils.exceptions import UnknownAssetError
from qairt.utils.loggers import get_logger
from qti.aisw.core.model_level_api import Target as mlapi_target
from qti.aisw.tools.core.modules.net_runner import InferenceConfig
from qti.aisw.tools.core.modules.net_runner.net_runner_module import (
    InferenceIdentifier,
    NetRunner,
    NetRunnerLoadArgConfig,
    NetRunnerRunArgConfig,
    NetRunnerUnloadArgConfig,
)

__ALL__ = ["Model"]

_model_logger = get_logger("qairt.model")


class Model(ModelBase):
    """
    Representative entity that is executable on a QAIRT Backend.

    It has the following additional properties:

    - It can be saved to an asset that remains executable on a QAIRT Backend.
    - It can be loaded from a DLC
      binaries. If load is successful, the Model object can be executed similar to the asset.
    - It can be queried to return properties that identify the asset from which it was loaded.

    The object contains a DLC module which enables saving, loading, execution and querying of a DLC
    ,its underlying graphs and related metadata.
    """

    def __init__(self, module: DlcModule, *, name: str = ""):
        """
        Initializes a new Model object.

        Args:
            module (Optional[DlcModule]): The module object to associate with this model.

        """
        super().__init__(name=name)

        # A module is compositionally related to its graphs, which may be in memory or serialized on disk.
        self.module: DlcModule = module

        # Assets that are generated as part of model execution, or as part of operations performed
        # on this model object that may be needed for execution. Users are not expected to interact with
        # this structure.
        self.assets = AssetMapping()

    @property
    def module(self) -> DlcModule:
        """
        Returns the dlc module associated with this model.
        """
        return self._module

    @module.setter
    def module(self, module: DlcModule):
        """
        Sets the dlc module associated with this model.
        """
        if not isinstance(module, DlcModule):
            raise TypeError("Module must be of type DlcModule")

        if self._module is None:
            self._module: DlcModule = module
        else:
            raise AttributeError("Module cannot be reset after initialization.")

    @module.deleter
    def module(self):
        if self._module is None:
            return
        raise RuntimeError("Cannot delete module after initialization.")

    def initialize(
        self,
        backend: Union[str | BackendType] = "CPU",
        device: Optional[Device] = None,
        **extra_args: Unpack[ExecutionConfig],
    ) -> None:
        """
        Initializes the QAIRT model and loads required backend artifacts needed for executing on device.

        This function is optional, and should be used if you intend to call execute multiple times
        with the same model, backend, and device. In addition to enabling a single initialization,
        this method controls the lifetime of backend library artifacts.

        Artifacts loaded during initialize() will persist until destroy() is called with the associated
        handle. However, if the model, backend, and target are given during execution, then artifacts will
        be unloaded at the end of execution. So providing the same model, backend, and target to multiple
        calls to run() will result in redundant initialization and destruction calls. This can be resolved
        through explicitly calling initialize()/destroy().
        This method must take in at least one positional argument.

        Args:
            backend (Optional[BackendType], optional): The intended QAIRT Backend for execution.
                Defaults to "CPU" if no backend is specified.
            device (Optional[Device], optional): The intended QAIRT device. If none, then the default local host is used.
            extra_args: Extra keyword arguments to pass for execution.
                See submodule `qairt.api.executor.execution_config.ExecutionConfig` for details.
        """
        _model_logger.info("Initializing model for execution")

        if hasattr(self, "_inference_handle"):
            self.destroy()

        inference_config, inference_identifier, net_runner = self._create_execution_context(
            backend, device, extra_args
        )
        if inference_identifier.target.type not in supported_initialize_execute_platforms:
            _model_logger.error("Model initialization is only supported for X86_64 Linux/Windows and WoS.")
            raise RuntimeError("Unsupported platform. Exiting.")

        net_runner_load_arg_config = NetRunnerLoadArgConfig(
            identifier=inference_identifier, inference_config=inference_config
        )

        try:
            load_config = net_runner.load(net_runner_load_arg_config)
        except Exception as e:
            _model_logger.error(f"Failed to initialize the model for execution.")
            raise e

        # set inference handle
        setattr(self, "_inference_handle", (inference_config, load_config.handle, net_runner))

    def _execute(
        self,
        inputs: ExecutionInputData,
        *,
        backend: str | BackendType = "CPU",
        device: Optional[Device] = None,
        **extra_args: Unpack[ExecutionConfig],
    ) -> ExecutionResult:
        """
        Performs inference on a QAIRT backend.

        This method is triggered via the __call__ method, and must be implemented by
        any subclasses. The behavior of this method is not guaranteed if it is called directly.

        Args:
            inputs: Input data to be used for execution. See `qairt.configs.common.ExecutionInputData` for types.

            backend (Optional[BackendType]): The intended QAIRT Backend for execution. Defaults to
                                             "CPU" if no backend is specified.
            device (Optional[Device]): The intended QAIRT device. If none, then the default local host is used.
            extra_args: Extra keyword arguments to pass for execution. See submodule
                            `qairt.api.executor.execution_config.ExecutionConfig` for details.

        Returns:
            The result contains the inference output data in memory, and any additional output generated from profiling.
            See `ExecutionResult` for details.

        Raises:
            ValidationError: if provided extra args are not valid
            ExecutionError: if an error occurs during model execution
        """

        if not hasattr(self, "_inference_handle"):
            inference_config, inference_identifier, net_runner_module = self._create_execution_context(
                backend, device, extra_args
            )

        else:
            inference_config, inference_identifier, net_runner_module = getattr(self, "_inference_handle")

        net_runner_run_arg_config = NetRunnerRunArgConfig(
            identifier=inference_identifier, input_data=inputs, inference_config=inference_config
        )

        try:
            inference_output_config = net_runner_module.run(net_runner_run_arg_config)
        except Exception as e:
            _model_logger.error(f"Failed to execute the Model: {self.name}.")
            raise e

        if isinstance(inputs, (str, PathLike)):
            # if input is a file, then output data can be a list of dictionaries
            # as per this API. Support for this is added primarily for legacy use cases
            # involving input list files.
            output_data = inference_output_config.output_data
        else:
            # Output data is a dictionary
            output_data = inference_output_config.output_data[0]

        return ExecutionResult(data=output_data, profiling_data=inference_output_config.profiling_data)

    @profile("inference")
    def __call__(
        self,
        inputs: ExecutionInputData,
        *,
        backend: Optional[str] = "CPU",  # TODO AISW-129336: Set default to None after refactoring
        device: Optional[Device] = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Public method to execute the model.
        Handles the execution flow internally.

        Args:
            inputs: Input data to be used for execution. Input data could be of the following types:

                np.ndarray | Sequence[np.ndarray]: A single numpy array or sequence of numpy arrays.

                str | PathLike: A path to an input list text file containing paths to raw data.
                    This type is intended for data that will be executed on remote devices,
                    where the data must be pushed to the execution environment.
                    The format of this file is defined further in documentation.
                    See `QAIRT SDK` for details.

                Dict[str, np.ndarray]: A dictionary of tensor names to
                numpy arrays.


            backend (Optional[str]): The intended QAIRT Backend for execution. Defaults to
                                "CPU" if no backend is specified.
            device (Optional[Device]): The intended QAIRT device. If none, then the default local host is used.
            **kwargs: Keyword arguments for execution.

            kwargs may contain:
                  - Extra keyword arguments to pass for execution. See submodule
                    `qairt.modules.netrunner_module.InferenceConfig` for details.
                  - Arguments to pre or post execute hooks.

        Returns:
            ExecutionResult: The result after applying pre-execute hooks, execution, and post-execute hooks.
            The result contains the inference output data in memory, and any additional output generated from
            profiling. See `ExecutionResult` for details.

        Raises:
            ExecutionError: if a runtime error occurs.
        """
        return super().__call__(inputs, backend=backend, device=device, **kwargs)

    @classmethod
    def load(
        cls, path: str | os.PathLike, *, name: str = "", load_weight_data: bool = True, **load_args
    ) -> "Model":
        """
        Loads a DLC from the specified path.

        Args:
            path (str): A path to a DLC.
            name (str): User-specified identifier for a model
            load_weight_data (bool): Whether to load the weights of the DLC on initialization.
            load_args (Optional[Dict[str, Any]]): Additional arguments for loading a DLC.
                                                  See qairt.modules.dlc.DlcModule.__init__ for details.

        Returns:
            The loaded model object.

        """
        if not check_asset_type(AssetType.DLC, path):
            raise UnknownAssetError(f"File: {path} is not a valid DLC")

        try:
            _model_logger.debug(f"Loading DLC from {path}")
            dlc_module = DlcModule.load(path, enable_lazy_weight_loading=load_weight_data)
        except Exception as e:
            _model_logger.error(f"Error loading DLC from path: {path}")
            raise e

        _model_logger.info(f"Loaded DLC from {path}")
        return cls(module=dlc_module, name=name)

    @property
    def quantized(self) -> bool:
        """
        Returns True if the model is quantized False otherwise.
        """

        name = self.module.graph_names()[0]
        graph_info = self.module.graphs_info
        graph = graph_info[name]
        is_quantized = graph.outputs[0].is_quantized

        return is_quantized

    def save(self, path: str | os.PathLike = "", **kwargs) -> str:
        """
        Saves a model to a specified path. The subclass controls the output that is saved.

        Args:
            path (str): The path where the model will be saved.

        Returns:
            The path where the model, along with any associated assets, were saved.
        """
        return self.module.save(path)

    def save_with_assets(self, dir_path: str | os.PathLike = ".", **kwargs) -> str:
        """
        Saves all assets associated with this model.

        Args:
            dir_path (DirectoryPath): The path directory.

        Returns:
            str: The path where the assets were saved.
        """

        if not os.path.isdir(dir_path):
            raise IOError(f"Path {dir_path} does not exist")

        dir_path = Path(dir_path).resolve()
        dlc_path = dir_path / self.module.path

        self.save(dlc_path)

        # TODO: need to figure out how to save other assets gracefully
        for _, asset in self.assets.items():
            asset.save(dir_path)

        return str(dir_path)

    def _create_execution_context(
        self,
        backend: Union[str, BackendType],
        device: Optional[Device] = None,
        extra_args: Optional[Dict] = None,
    ) -> Tuple[InferenceConfig, InferenceIdentifier, NetRunner]:
        """
        Creates an execution context by validating extra arguments, determining the device, and creating an inference identifier.

        Args:
            backend: The intended QAIRT Backend for execution
            device: The intended QAIRT device. If none, then the default local host is used.
            extra_args: Extra key-value to validate and use for creating the inference configuration.


        Returns:
            A tuple containing:
                - inference_config: The inference configuration.
                - inference_identifier: The inference identifier.
                - net_runner: The NetRunner instance.

        Raises:
            RuntimeError: If a device must be provided to execute on the GPU backend and none is provided.
        """

        if extra_args is None:
            extra_args = {}

        # Validating extra args
        execution_config = ExecutionConfig(**extra_args)

        inference_config = InferenceConfig(log_level=execution_config.log_level, **extra_args)

        if device is None:
            if backend == BackendType.GPU:
                raise RuntimeError("A device must be provided to execute on the GPU backend")
            localhost = mlapi_target.create_host_target()
            target_device = Target(type=localhost.target_platform_type.value)
            _model_logger.info("No device provided. Defaulting to localhost.")
        else:
            # Set chipset to enable automatic skel/stub selection for HTP targets
            if chipset := device.get_chipset():
                if isinstance(device.info.identifier, RemoteDeviceIdentifier):
                    # TODO: Change soc_model to chipset
                    device.info.identifier.soc_model = chipset

            target_device = Target(
                type=device.info.platform_type.value,
                identifier=device.info.identifier,
                soc_model=device.info.identifier.soc_model,
            )
            if hasattr(device.info, "credentials"):
                target_device.credentials = device.info.credentials

        if isinstance(self.module, DlcModule) and self.module.path:
            model = qti_module_api.Model(dlc_path=self.module.path)
        elif isinstance(self.module, CacheModule) and self.module.path:
            if inference_config.set_output_tensors:
                inference_config.set_output_tensors = []
                _model_logger.warning(
                    "The 'set_output_tensors' option cannot be used when the graph is retrieved from "
                    "the context binary. It has been changed to an empty list."
                )
            model = qti_module_api.Model(context_binary_path=self.module.path)

        else:
            raise AttributeError("Module is missing the required path attribute.")

        inference_identifier = InferenceIdentifier(
            model=model,
            target=target_device,
            backend=backend,
        )
        net_runner = NetRunner(logger=_model_logger)
        return inference_config, inference_identifier, net_runner

    def __del__(self) -> None:
        """Garbage collection on the model, primarily used to delete any
        persistent inference handles and any unsaved assets"""

        # destroy inferencer handle
        self.destroy()

    def destroy(self) -> None:
        """
        This method is called when the model object is being garbage collected.
        It unloads the model from memory and releases any associated resources.

        :return: None
        """
        if not hasattr(self, "_inference_handle"):
            return
        _, unload_handle, netrunner = getattr(self, "_inference_handle")
        config = NetRunnerUnloadArgConfig(handle=unload_handle)
        netrunner.unload(config)
        delattr(self, "_inference_handle")

    @property
    def input_tensors(self) -> List[Tuple[str, List[TensorInfo]]]:
        """
        Returns the input tensor information of the model.

        Returns:
            list: A list of tuples containing the graph name and input tensors information.
        """
        input_tensors = []
        graphs = self.module.info.graphs
        for graph in graphs:
            input_tensors.append((graph.name, graph.inputs))
        return input_tensors

    @property
    def output_tensors(self) -> List[Tuple[str, List[TensorInfo]]]:
        """
        Returns the output tensor information of the model.

        Returns:
            list: A list of tuples containing the graph name and output tensors information.
        """
        output_tensors = []
        graphs = self.module.info.graphs
        for graph in graphs:
            output_tensors.append((graph.name, graph.outputs))
        return output_tensors
