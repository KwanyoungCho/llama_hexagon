# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
from os import PathLike
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)

import numpy
from typing_extensions import Unpack

from qairt.api.configs.common import BackendType, ExecutionInputData, ExecutionResult
from qairt.api.configs.device import Device
from qairt.api.executor.execution_config import ExecutionConfig
from qairt.modules.cache_module import CacheModule
from qairt.modules.common.graph_info_models import TensorInfo
from qairt.modules.dlc_module import DlcModule
from qairt.utils import loggers
from qairt.utils.asset_utils import AssetMapping
from qairt.utils.exceptions import ExecutionError, HookHandleError
from qairt.utils.handles import HookHandle
from qti.aisw.tools.core.modules.net_runner import InferenceConfig
from qti.aisw.tools.core.modules.net_runner.net_runner_module import (
    InferenceIdentifier,
    NetRunner,
)

InferenceHandle = Tuple[InferenceConfig, InferenceIdentifier, NetRunner]


_model_logger = loggers.get_logger("qairt.execute")

# ---------------------- Model Types ------------------ #
ModelBaseT = TypeVar("ModelBaseT", bound="ModelBase")

# A pre-hook can have a signature type of:
# hook(Model, inputs, **kwargs) -> None or (inputs, kwargs)
ExecPreHook: TypeAlias = Callable[
    [ModelBaseT, ExecutionInputData, Dict[str, Any]], Optional[Tuple[Any] | Dict[str, Any]]
]

# A post-hook can have a signature type of:
# hook(Model, output: ExecutionResult | None, **kwargs) -> None or ExecutionResult
ExecPostHook: TypeAlias = Callable[
    [ModelBaseT, Optional[ExecutionResult], Dict[str, Any]], Optional[ExecutionResult]
]


__all__ = ["ModelBase", "ModelBaseT", "ExecPreHook", "ExecPostHook"]


class ModelBase(ABC):
    """
    Representative entity that is executable on a QAIRT Backend.

    It has the following additional properties:

    - It can be saved to an asset that remains executable on a QAIRT Backend.
    - It can be loaded from an executable asset. If load is successful, the Model object
      can be executed similar to the asset.
    - It can be queried to return properties that identify the asset from which it was loaded.


    Attributes:
        _pre_execute_hooks (OrderedDict[int, ExecPreHook]): A dictionary of pre-execute functions hooks
                                                          triggered prior to an execution call.
        _post_execute_hooks (OrderedDict[int, ExecPostHook]): A dictionary of post-execute hooks triggered
                                                               after an execution call.
        _module (Optional[DlcModule | CacheModule]): The module associated with this model, which determines
                                                       how it is loaded, saved and queried.
        _assets (AssetMapping): A mapping of assets generated during model execution or operations performed
          on this model.

    """

    def __init__(self, *, name: str = ""):
        """
        Initializes a new ModelBase object.
        """
        self._name: str = name
        self._pre_execute_hooks: OrderedDict = OrderedDict()
        self._post_execute_hooks: OrderedDict = OrderedDict()
        self._module: Optional[DlcModule | CacheModule] = None

        # Assets that are generated as part of model execution, or as part of operations performed
        # on this model object that may be needed for execution. Users are not expected to interact with
        # this structure.
        self._assets = AssetMapping[str]()

    def add_pre_execute_hook(self, hook: ExecPreHook) -> HookHandle:
        """
        Adds a pre-execute hook to all Model types.

        Args:
            hook (ExecPreHook): The function to call before execution.

        Returns:
            A hook handle which can be removed using handle.abandon()

        Raises:
            HookHandleError: If the hook handle cannot be added
        """
        try:
            handle = HookHandle(self._pre_execute_hooks, hook)
            return handle
        except Exception as e:
            raise HookHandleError("Hook handle cannot be added") from e

    def add_post_execute_hook(self, hook: ExecPostHook) -> HookHandle:
        """
        Adds a post-execute hook to all Model types.

        Args:
            hook (ExecPostHook): The function to call after execution.

        Returns:
            A hook handle which can be removed using handle.abandon()

        Raises:
            HookHandleError: If the hook handle cannot be added
        """
        try:
            handle = HookHandle(self._post_execute_hooks, hook)
            return handle
        except Exception as e:
            raise HookHandleError("Hook handle cannot be added") from e

    @abstractmethod
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
        raise NotImplementedError("Method should be implemented by subclasses.")

    @abstractmethod
    def _execute(
        self,
        inputs: ExecutionInputData,
        *,
        backend: str = "CPU",
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
        """

        raise NotImplementedError("ModelBase._execute() must be implemented by subclasses.")

    def __call__(
        self,
        inputs: ExecutionInputData,
        *,
        backend: Optional[str] = None,
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

            backend (str): The intended QAIRT Backend for execution. Defaults to
                          "CPU" if no backend is specified.  See `qairt.BackendType` for other supported
                          backends.
            device (Optional[Device]): The intended QAIRT device. If none, then the default local host is used.
            **kwargs: Keyword arguments for execution.

            kwargs may contain:
                  - Extra keyword arguments to pass for execution. See submodule
                    `qairt.modules.netrunner_module.InferenceConfig` for details.
                  - Arguments to pre or post execute hooks.

        Returns:
            The result after applying pre-execute hooks, execution, and post-execute hooks. The result
            contains the inference output data in memory, and any additional output generated from profiling.
            See `ExecutionResult` for details.

        Raises:
            ExecutionError: if a runtime error occurs.
        """

        # set backend and device in kwargs
        kwargs["backend"] = backend
        if device is not None:
            kwargs["device"] = device

        _model_logger.info(
            f" Model: {self.name} execution started on backend: {backend},"
            f" device: {device if device is not None else 'localhost'}"
        )

        # Call pre-execute hooks in order
        for id_, hook in self._pre_execute_hooks.items():
            _model_logger.debug(f"Executing pre-execute hook '{id_}'.")
            pre_hook_output = hook(self, inputs, **kwargs)
            if pre_hook_output is not None and len(pre_hook_output) == 2:
                inputs, kwargs = pre_hook_output

        # Execute the model
        try:
            output = self._execute(inputs=inputs, **kwargs)
            _model_logger.info(f"Model {self.name} executed successfully.")
        except Exception as e:
            _model_logger.debug(f"Execution failed with error: {e}")
            traceback.print_exc(limit=2)
            raise ExecutionError("Failed to execute model")

        # Call post-execute hooks in order, start off with the original output
        for id_, hook in self._post_execute_hooks.items():
            _model_logger.debug(f"Executing post-execute hook '{id_}'.")
            post_hook_output: ExecutionResult | None = hook(self, output, **kwargs)

            # Hook output of None is assumed to be a no-op on execution result
            if post_hook_output is not None and isinstance(post_hook_output, ExecutionResult):
                output = post_hook_output

        return output

    @property
    @abstractmethod
    def module(self):
        """
        Returns the module associated with this model.

        The role of a module is to enable loading, saving and information retrieval for the model via a created
        graph or a serialized asset on disk.

        See DlcModule and CacheModule respectively for more details.

        Returns:
            DlcModule | CacheModule: The module associated with this model.

        """

        raise NotImplementedError("Module property must be implemented in the subclass")

    @property
    def name(self) -> str:
        return self._name if self._name else self.module.src_path.stem

    @classmethod
    @abstractmethod
    def load(cls, path: str | os.PathLike, *, name: str = "", **load_args) -> "ModelBase":
        """
        Loads a QAIRT asset from a specified source.

        Args:
            path (str): A path to an executable asset (either a DLC (.BIN) or serialized context (.BIN))
            name (str): The name of the model
            **load_args: Additional arguments for loading the model.

        Returns:
            The loaded model object.
        """
        raise NotImplementedError("Model.load must be implemented in the subclass")

    @abstractmethod
    def save(self, path: str | os.PathLike = "", **kwargs) -> str:
        """
        Saves a model to a specified path. The subclass controls the output that is saved.

        Args:
            path (str): The file path where the model will be saved.
            **kwargs: Additional keyword arguments.

        Returns:
            The path where the model, along with any associated assets, were saved.
        """
        raise NotImplementedError("Model.save() must be implemented in the subclass")

    @abstractmethod
    def save_with_assets(self, dir_path: str | os.PathLike, **kwargs) -> str:
        """
        Saves all assets associated with this model.

        Args:
            dir_path (DirectoryPath): The path directory where the model, along with any associated assets,
                                      should be saved.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path where the assets were saved.
        """
        raise NotImplementedError("Model.save_all_assets() must be implemented in the subclass")

    @property
    def input_tensors(self) -> List[Tuple[str, List[TensorInfo]]]:
        """
        Returns the input tensor information of the model.

        Returns:
            list: A list of tuples containing the graph name and input tensors information.
        """
        raise NotImplementedError("This method should be overridden in the subclass.")

    @property
    def output_tensors(self) -> List[Tuple[str, List[TensorInfo]]]:
        """
        Returns the output tensor information of the model.

        Returns:
            list: A list of tuples containing the graph name and output tensors information.
        """
        raise NotImplementedError("This method should be overridden in the subclass.")
