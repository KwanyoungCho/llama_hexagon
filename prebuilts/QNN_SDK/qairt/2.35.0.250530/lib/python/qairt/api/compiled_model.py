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
from typing import Optional, Union

from typing_extensions import Unpack

from qairt.api.compiler import CompileConfig
from qairt.api.configs import (
    BackendType,
    Device,
    ExecutionInputData,
    ExecutionResult,
)
from qairt.api.executor.execution_config import ExecutionConfig
from qairt.api.model import Model
from qairt.modules.cache_module import CacheModule
from qairt.modules.dlc_module import DlcModule
from qairt.utils.asset_utils import AssetType, check_asset_type, get_asset_type
from qairt.utils.exceptions import UnknownAssetError
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.modules.net_runner.net_runner_module import (
    NetRunnerRunArgConfig,
)

__ALL__ = ["CompiledModel"]

_cmodel_logger = get_logger("qairt.execute")


class CompiledModel(Model):
    """
    Representative entity that has been prepared for execution on a QAIRT Backend,
    and is executable only that backend.This means that graph initialization and composition
    has been performed, and the model contains a reference to a serialized cache or a container
    composed of serialized caches.

    It has the following additional properties:

    - It can be saved to an asset that remains executable only on the QAIRT Backend for which it has been
      compiled.
    - It can be loaded from a DLC or Binary asset. If load is successful, the Model object
      can be executed similar to the asset.
    - It can be queried to return properties that identify the asset from which it was loaded. No properties
      may be changed after the object has been created.

    The object may contain associated graphs, modules and additional objects that enable saving,
    loading and querying.
    """

    def __init__(
        self,
        module: DlcModule | CacheModule,
        backend: Optional[str | BackendType] = None,
        *,
        name: str = "",
        config: Optional["CompileConfig"] = None,
    ):
        """
        Initializes a new CompiledModel object.

        Args:
            module (DlcModule | CacheModule): A representation of this model as a serialized module.
            backend (str | BackendType): The backend this model is associated with. If no backend is specified,
                                         then it is assumed that a backend will be passed to execute.
                                         Note that if the module is of type: DlcModule, then the model may be
                                         executed on a different backend from which it was compiled.
            name (str): The name of the model.
            config (Optional[CompileConfig]): Compilation configuration options used to create the module.
        """
        super().__init__(module=module, name=name)
        self._config = config
        self._backend = backend

    @property
    def module(self) -> DlcModule | CacheModule:
        """
        Returns the module associated with this model.
        """
        return self._module

    @property
    def backend(self) -> Optional[str | BackendType]:
        """
        Returns the backend associated with this model.
        """
        return self._backend

    @module.setter
    def module(self, module: DlcModule | CacheModule):
        if self._module is not None:
            raise AttributeError("Cannot set module after initialization.")
        self._module = module

    @module.deleter
    def module(self):
        if self._module is None:
            return
        raise RuntimeError("Cannot delete module after initialization.")

    @property
    def config(self) -> Optional[CompileConfig]:
        """
        Returns the configuration used to compile this model.
        """
        return self._config

    def _verify_and_set_backend(self, backend: Optional[Union[str, BackendType]]) -> Union[str, BackendType]:
        """
        Verifies and sets the backend for the compiled model.

        Args:
            backend: The backend to use for execution. Can be a string or a BackendType.

        Returns:
            The backend to use for execution.

        Raises:
            AttributeError: If the backend does not match the one used for compilation.
        """
        if backend is None:
            if not self.backend:
                raise AttributeError("Backend must be specified when executing a compiled model.")
            backend = self.backend
        elif self.backend and backend != self.backend:
            raise AttributeError(
                f"Compiled Model cannot be executed on a different backend than it was compiled."
                f" Expected: {self.backend}. Got: {backend}"
            )
        return backend

    def initialize(
        self,
        backend: Optional[str | BackendType] = None,
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
            device (Optional[Device], optional): The intended QAIRT device. If none, then the default local host is used.
            extra_args: Extra keyword arguments to pass for execution.
                See submodule `qairt.api.executor.execution_config.ExecutionConfig` for details.
        """
        backend = self._verify_and_set_backend(backend)
        super().initialize(backend, device, **extra_args)

    def _execute(
        self,
        inputs: ExecutionInputData,
        *,
        backend: Optional[str | BackendType] = None,
        device: Optional[Device] = None,
        **extra_args: Unpack[ExecutionConfig],
    ) -> ExecutionResult:
        """
         Performs inference on a QAIRT backend.

        This method is triggered via the __call__ method, and must be implemented by
        any subclasses. The behavior of this method is not guaranteed if it is called directly.

        Args:
            inputs: Input data to be used for execution. See `qairt.configs.common.ExecutionInputData` for types.

            backend (Optional[BackendType]): The intended QAIRT Backend for execution. If no backend is specified,
                                             then self._backend is used.
            device (Optional[Device]): The intended QAIRT device. If none, then the default local host is used.
            extra_args: Extra keyword arguments to pass for execution. See submodule
                             `qairt.api.executor.execution_config.ExecutionConfig` for details.

        Returns:
            The result contains the inference output data in memory, and any additional output generated from profiling.
            See `ExecutionResult` for details.

        Raises:
            ValidationError: if provided extra args are not valid
            ExecutionError: if an error occurs during compiled model execution


        """
        if not self.module.executable:
            raise RuntimeError("Could not execute model")

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
            _cmodel_logger.error(f"Failed to execute the Compiled Model: {self.name}.")
            raise e

        profiling_data = None
        if inference_output_config.profiling_data:
            profiling_data = inference_output_config.profiling_data
            for _, asset in self.assets.items():
                if asset.type == AssetType.SCHEMATIC_BIN:
                    if profiling_data.backend_profiling_artifacts is None:
                        profiling_data.backend_profiling_artifacts = []
                    profiling_data.backend_profiling_artifacts.append(asset.path)

        if isinstance(inputs, (str, PathLike)):
            # if input is a file, then output data can be a list of dictionaries
            # as per this API. Support for this is added primarily for legacy use cases
            # involving input list files.
            output_data = inference_output_config.output_data
        else:
            # Output data is a dictionary
            output_data = inference_output_config.output_data[0]

        return ExecutionResult(data=output_data, profiling_data=profiling_data)

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
            inputs: Input data to be used for execution. See qairt.Model for compatible
            input types.

            backend (Optional[str]): The intended QAIRT Backend for execution. Defaults to self.backend
                                             if no backend is specified.
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
            AttributeError: if no backend can be identified
            ExecutionError: if an error occurs during model execution
        """
        backend = self._verify_and_set_backend(backend)

        return super().__call__(inputs, backend=backend, device=device, **kwargs)

    @classmethod
    def load(
        cls,
        path: str | PathLike,
        *,
        name: str = "",
        compile_config: Optional[CompileConfig] = None,
        backend: Optional[BackendType] = None,
        **load_args,
    ) -> "CompiledModel":
        """
        Loads a model from a specified source.

        Args:
            path (str): The source of the model.
            name (str): User-specified identifier for the model
            compile_config (Optional[CompileConfig]): The specifications used to compile this model.
            backend (Optional[BackendType]): The backend this model is associated with.
            load_args (Optional[Dict[str, Any]]): Additional arguments for loading a DLC.
                                                  See qairt.modules.dlc.DlcModule.__init__ for details.

        Returns:
            CompiledModel: The loaded model object.
        """
        if not (check_asset_type(AssetType.DLC, path) or check_asset_type(AssetType.CTX_BIN, path)):
            raise UnknownAssetError(f"{path}: is not a valid compiled asset")

        asset_type = get_asset_type(path)

        if asset_type == AssetType.DLC:
            dlc_module = DlcModule.load(path, **load_args)

            if not dlc_module.caches:
                raise RuntimeError(f"DLC: {path} is not compiled.")

            model = cls(name=name, module=dlc_module, backend=backend, config=compile_config)

        else:
            cache_module = CacheModule.load(path=path)

            # Identify backend on load if none is provided
            if not backend and cache_module.info.backend:
                backend = cache_module.info.backend

            model = cls(name=name, module=cache_module, backend=backend, config=compile_config)

        return model

    @property
    def quantized(self) -> bool:
        if isinstance(self.module, DlcModule):
            return super().quantized
        else:
            # TODO: Find a better way to do this. This approach uses
            # QNN_BACKEND_ID definitions for HTP, HTP_MCP and HTP_QEMU
            # This is not ideal as it doesn't account for float dtype on
            # on these runtimes, or quant dtypes on CPU.
            module: CacheModule = self.module
            return module.info.backend.id in [6, 11, 13]

    def save(self, path: str | os.PathLike = "", asset_type: AssetType | None = None, **kwargs) -> str:
        """
        Saves a model to a specified path.

        Args:
            path (str): The path of the model should be a file if AssetType.DLC
                        or a directory if AssetType.CTX_BIN.
            asset_type (AssetType): The type of asset to write to disk. If set to None, then the asset type
                                    to save is inferred from the module type.
            kwargs (Optional[Dict[str, Any]]): Additional arguments for saving the model.

        Returns:
            str: The path where the model was saved.
        """
        if asset_type is None:
            asset_type = AssetType.DLC if isinstance(self.module, DlcModule) else AssetType.CTX_BIN

        if asset_type == AssetType.CTX_BIN:
            if isinstance(self.module, DlcModule):
                if self.module.caches:
                    if not os.path.isdir(path):
                        path = str(Path(path).parent)
                        os.makedirs(path, exist_ok=True)
                    cache_modules = self.module.extract_caches(str(path))
                    for cache_module in cache_modules:
                        file_path = os.path.join(
                            path, cache_module.name + AssetType.get_extension(AssetType.CTX_BIN)
                        )
                        cache_module.save(file_path)
                else:
                    raise ValueError("Cannot extract binaries from DLC")
            else:
                if not os.path.isdir(path):
                    path = str(Path(path).parent)
                file_path = os.path.join(path, self.module.name + AssetType.get_extension(AssetType.CTX_BIN))
                self.module.save(file_path)
        else:
            path = super().save(path)
        return str(path)

    def save_with_assets(self, dir_path: str | os.PathLike = ".", **kwargs) -> str:
        """
        Saves all assets associated with this model.

        Args:
            dir_path (DirectoryPath): The path directory where the model, along with any associated assets,
                                      should be saved.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path where the assets were saved.
        """
        if isinstance(self.module, DlcModule):
            return super().save_with_assets(dir_path, **kwargs)
        else:
            self.module.save(dir_path)

            for _, asset in self.assets.items():
                asset.save(dir_path)

        return str(dir_path)
