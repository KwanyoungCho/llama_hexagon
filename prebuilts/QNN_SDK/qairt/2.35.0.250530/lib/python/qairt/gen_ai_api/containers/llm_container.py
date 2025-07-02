# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
import os
import shutil
from os import PathLike
from pathlib import Path
from typing import List, Optional

from pydantic_core import from_json

from qairt import Device
from qairt.api.compiled_model import CompileConfig, CompiledModel
from qairt.api.configs.common import AISWBaseModel
from qairt.gen_ai_api.configs.gen_ai_config import GenAIConfig
from qairt.gen_ai_api.containers.gen_ai_container import GenAIContainer
from qairt.gen_ai_api.executors.gen_ai_executor import GenAIExecutor
from qairt.gen_ai_api.executors.t2t_executor import T2TExecutor
from qairt.modules.cache_module import CacheModule
from qairt.modules.dlc_module import DlcModule
from qti.aisw.tools.core.modules.api import BackendType


class ContainerMetadata(AISWBaseModel):
    """
    Container metadata for serialization/deserialization
    """

    backend: BackendType


class LLMContainer(GenAIContainer):
    """
    Produced by an GenAiBuilder and consumed by an GenAiExecutor
    """

    def __init__(
        self,
        models: List[CompiledModel],
        gen_ai_config: GenAIConfig,
        backend: BackendType,
        compile_config: Optional[CompileConfig] = None,
    ):
        """
        Create a GenAIContainer

        Args:
            models (CompiledModel): List of CompiledModels containing the LLM split(s) prepared for QAIRT execution
            gen_ai_config (GenAIConfig): contains the configuration metadata for the GenAI model.
            backend (BackendType): The backend the artifacts in this container were prepared for
            compile_config (Optional[CompileConfig]): contains configuration metadata for the compiled configuration. Defaults to None.
        """
        if not models:
            raise ValueError("LLMContainer must be have at least one CompiledModel")
        self._models: List[CompiledModel] = models
        self._gen_ai_config: GenAIConfig = gen_ai_config
        self._backend: BackendType = backend
        self._compile_config: CompileConfig | None = compile_config

    @classmethod
    def _metadata_file(cls, path: str | PathLike) -> str:
        return os.path.join(path, "metadata.json")

    @classmethod
    def _gen_ai_config_file(cls, path: str | PathLike) -> str:
        return os.path.join(path, "gen_ai_config.json")

    @classmethod
    def _compile_config_file(cls, path: str | PathLike) -> str:
        return os.path.join(path, "compile_config.json")

    @classmethod
    def _tokenizer_file(cls, path: str | PathLike) -> str:
        return os.path.join(path, "tokenizer.json")

    @classmethod
    def _model_dir(cls, path: str | PathLike) -> str:
        return os.path.join(path, "models")

    @classmethod
    def _model_dlc(cls, path: str | PathLike, index: int) -> str:
        return os.path.join(cls._model_dir(path), f"model_{index}.dlc")

    @classmethod
    def _model_ctx_bin(cls, path: str | PathLike, index: int) -> str:
        return os.path.join(cls._model_dir(path), f"model_{index}.bin")

    @classmethod
    def _lora_dir(cls, path: str | PathLike) -> str:
        return os.path.join(path, "lora")

    @classmethod
    def _lora_bin(cls, path: str | PathLike, adapter_name: str, index: int) -> str:
        return os.path.join(cls._lora_dir(path), f"{adapter_name}_{index}.bin")

    def save(self, dest: str | PathLike, *, exist_ok: bool = False):
        """
        Save all artifacts to disk.  Note, this will copy artifacts into the destination directory, and update any
        configurations accordingly.

        Args:
            dest (str | PathLike): Path to save the artifacts
        """
        if os.path.exists(dest):
            if not os.path.isdir(dest):
                raise NotADirectoryError(f"Destination path {dest} exists but is not a directory")
            elif not exist_ok:
                raise ValueError(
                    f"Destination path {dest} already exists.  To use an existing directory, specify exist_ok=True"
                )
        else:
            os.makedirs(dest, exist_ok=exist_ok)

        # copy the tokenizer found at self._gen_ai_config.tokenizer_path to dest
        shutil.copyfile(self._gen_ai_config.tokenizer_path, self._tokenizer_file(dest))

        if self._models:
            os.makedirs(self._model_dir(dest), exist_ok=True)

            for i, model in enumerate(self._models):
                if isinstance(model.module, CacheModule):
                    model.module.save(self._model_ctx_bin(dest, i))
                elif isinstance(model.module, DlcModule):
                    model.module.save(self._model_dlc(dest, i))
                else:
                    raise TypeError(f"Unsupported Compiled model module type {type(model.module)}")

        with open(self._metadata_file(dest), "w") as f:
            f.write(ContainerMetadata(backend=self._backend).model_dump_json())

        with open(self._gen_ai_config_file(dest), "w") as f:
            f.write(self._gen_ai_config.model_dump_json(by_alias=True, exclude_none=True, indent=2))

        if self._compile_config:
            with open(self._compile_config_file(dest), "w") as f:
                f.write(self._compile_config.model_dump_json(by_alias=False, exclude_none=True, indent=2))

    @classmethod
    def load(cls, path: str | PathLike) -> "LLMContainer":
        """
        Load LLMContainer assets from disk

        Args:
            path (str | PathLike): Path to load the artifacts from.  This should be a directory containing (minimally) a
            pickle file for the saved class (gaic.pkl) and tokenizer file (tokenizer.json). There may be other files and
            directories as well.
        """
        if not os.path.isdir(path):
            raise NotADirectoryError(
                f"Path {path} is not a directory.  This should be a directory containing (minimally) a config file "
                f"{cls._gen_ai_config_file('')}, a tokenizer file ({cls._tokenizer_file('')}), and a model directory "
                f"({cls._model_dir('')}). There may be other files and directories as well."
            )

        backend = None
        with open(cls._metadata_file(path), "r") as f:
            backend = ContainerMetadata(**json.load(f)).backend

        gen_ai_config = None
        with open(cls._gen_ai_config_file(path), "r") as f:
            gen_ai_config = GenAIConfig(**json.load(f))
        # Update tokenizer path to loaded location
        gen_ai_config.tokenizer_path = Path(cls._tokenizer_file(path)).resolve()
        if not os.path.exists(gen_ai_config.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at location: {gen_ai_config.tokenizer_path}")

        compile_config = None
        if os.path.exists(cls._compile_config_file(path)):
            with open(cls._compile_config_file(path), "r") as f:
                compile_config = CompileConfig.model_validate(from_json(f.read()))

        models = []
        if not os.path.isdir(cls._model_dir(path)):
            raise NotADirectoryError(f"Serialized model directory does not exist: {cls._model_dir(path)}")

        i = 0
        while os.path.isfile(cls._model_ctx_bin(path, i)) or os.path.isfile(cls._model_dlc(path, i)):
            if os.path.isfile(cls._model_ctx_bin(path, i)):
                models.append(CompiledModel(CacheModule.load(path=cls._model_ctx_bin(path, i))))
            elif os.path.isfile(cls._model_dlc(path, i)):
                models.append(CompiledModel(DlcModule.load(cls._model_dlc(path, i))))

            i += 1

        return LLMContainer(models, gen_ai_config, backend, compile_config)

    def get_executor(self, device: Optional[Device] = None, clean_up: bool = True, **kwargs) -> T2TExecutor:
        if not self._models:
            raise ValueError("No models were loaded into the container. Nothing to execute.")

        return T2TExecutor(
            self._models, self._gen_ai_config, self._backend, device, self._compile_config, clean_up=clean_up
        )
