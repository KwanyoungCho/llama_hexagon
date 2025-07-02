# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from qairt.modules.genie_execution.genie_config import DialogType, EngineModelType, GenieConfig, Sampler
from qairt.utils import loggers
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    BackendType,
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
    Target,
    expect_module_compliance,
)
from qti.aisw.tools.core.modules.api.utils.configure_backend import create_backend
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    ConnectionType,
    DeviceEnvironmentContext,
    DeviceInfo,
    DevicePlatformType,
)
from qti.aisw.tools.core.utilities.devices.api.device_factory import DeviceFactory
from qti.aisw.tools.core.utilities.devices.api.device_interface import DeviceInterface
from qti.aisw.tools.core.utilities.devices.utils.device_code import (
    DeviceCode,
    DeviceCompletedProcess,
    DeviceFailedProcess,
    DeviceReturn,
)

genie_t2t_runner_logger = loggers.get_logger(name=__name__)


class E2TQuantizedType(str, Enum):
    """
    Defines supported quantization datatypes for the GenieT2TRun embedding input workflow
    """

    INT8 = "int8"
    INT16 = "int16"
    UINT8 = "uint8"
    UINT16 = "uint16"


class EmbeddingQuantization(AISWBaseModel):
    """
    Quantization parameters for a quantized embedding input or quantized embedding table
    """

    datatype: E2TQuantizedType
    scale: float
    offset: int

    def __str__(self):
        return ",".join([str(self.datatype.value), str(self.scale), str(self.offset)])


class EmbeddingInputConfig(AISWBaseModel):
    """
    Embedding input and embedding table arguments consist of the path to the raw data and optional quantization
    parameters
    """

    path: str | os.PathLike
    quantization: Optional[EmbeddingQuantization] = None


class EmbeddingConfig(AISWBaseModel):
    """
    Defines embedding input for e2t use cases
    """

    input: EmbeddingInputConfig
    """
    Embedding of prompt and/or multimodal inputs
    """
    embedding_table: EmbeddingInputConfig
    """
    Embedding table for converting token ids to embedding vectors
    """

    @model_validator(mode="after")
    def check_quantization(self) -> Self:
        if bool(self.input.quantization) ^ bool(self.embedding_table.quantization):
            raise AttributeError(
                "Quantization parameters must be provided for both the input and embedding table, or neither."
            )
        if self.input.quantization and self.embedding_table.quantization:
            signed = [E2TQuantizedType.INT8, E2TQuantizedType.INT16]
            if bool(self.input.quantization.datatype in signed) ^ bool(
                self.embedding_table.quantization.datatype in signed
            ):
                raise AttributeError("Input and embedding table quantization types' signedness must match")

        return self


class LoraRunConfig(AISWBaseModel):
    """
    Defines lora parameters for execution
    """

    adapter_name: str
    alpha: float = 1.0


class GenieT2TRunExecutionConfig(AISWBaseModel):
    """
    Defines supported genie-t2t-run options

    If a new genie config is passed it will take the place of the previously loaded config

    User specifies one of 'prompt', 'prompt_file', 'token_file', or 'embedding_config' as input to the model

    If Lora adapter bins are given in the genie config, the user may specify the name of the adapter they would
    like applied to the model, as well as the lora alpha value, using the 'lora_config'

    The user may specify a set of genie config sampler parameters via the 'sampler'. This will avoid redeploying
    the assets pointed to by a genie config if only the sampler parameters are changed.

    Passing 'qairt_sdk_root' will update the location that the instance pulls qairt libraries from for execution.
    """

    prompt: Optional[str] = None
    prompt_file: Optional[str | os.PathLike] = None
    token_file: Optional[str | os.PathLike] = None
    embedding_config: Optional[EmbeddingConfig] = None
    config: Optional[GenieConfig] = None
    lora_config: Optional[LoraRunConfig] = None
    sampler: Optional[Sampler] = None
    qairt_sdk_root: Optional[str | os.PathLike] = None

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        inputs = {
            "prompt": self.prompt,
            "prompt_file": self.prompt_file,
            "token_file": self.token_file,
            "embedding_config": self.embedding_config,
        }
        provided_inputs = [x for x in inputs.keys() if inputs[x]]

        if not provided_inputs:
            raise AttributeError(f"No input provided for execution. Please provide one of: {inputs.keys()}")

        if len(provided_inputs) > 1:
            raise AttributeError(
                f"Too many inputs provided: {provided_inputs}. Please provide one of: {inputs.keys()}"
            )

        if self.lora_config and self.config:
            if not (self.config.dialog.engine.model.binary and self.config.dialog.engine.model.binary.lora):
                raise AttributeError(
                    "Lora execution argument provided, however, the provided GenieConfig has no lora"
                    "definition"
                )
            available_adapters = [x.name for x in self.config.dialog.engine.model.binary.lora.adapters]
            if self.lora_config.adapter_name not in available_adapters:
                raise AttributeError(
                    f'Requested lora adapter, "{self.lora_config.adapter_name}", not present in'
                    "provided GenieConfig"
                )
        if self.qairt_sdk_root and not os.path.isdir(self.qairt_sdk_root):
            raise NotADirectoryError(
                f"Provided path to QAIRT SDK does not point to an existing directory: {self.qairt_sdk_root}"
            )

        return self


class GenieT2TRunOutputConfig(AISWBaseModel):
    """
    Defines output from execution of the GenieT2TRunner
    """

    return_code: int
    stdout: str
    stderr: str


class GenieT2TRunnerModuleSchema(ModuleSchema):
    _BACKENDS = ["HTP"]
    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)

    name: Literal["GenieT2TRunnerModule"] = "GenieT2TRunnerModule"
    path: Path = Path(__file__)
    arguments: GenieT2TRunExecutionConfig
    outputs: SkipJsonSchema[Optional[GenieT2TRunOutputConfig]] = None
    backends: List[str] = _BACKENDS


@expect_module_compliance
class GenieT2TRunner(Module):
    _SCHEMA = GenieT2TRunnerModuleSchema
    _LOGGER = genie_t2t_runner_logger

    _TARGET_ROOTS = {
        DevicePlatformType.ANDROID: "/data/local/tmp",
        DevicePlatformType.X86_64_LINUX: "/tmp",
    }

    _GENIE_CONFIG_FILENAME = "genie_config.json"
    _SSD_KVCACHE_PREFIX_FILENAME = "kv-cache.primary.qnn-htp"

    def __init__(
        self,
        config: GenieConfig,
        backend: BackendType,
        device: DeviceInfo | DeviceInterface,
        qairt_sdk_root: Optional[str | os.PathLike] = None,
        logger: Any = None,
        clean_up: bool = True,
    ):
        """
        Initializes a GenieT2TRunner module instance

        Args:
            config (GenieConfig): Genie config representing the model to run
            backend (BackendType): Desired backend type for execution
            device (DeviceInfo | DeviceInterface): Device to execute the model on
            qairt_sdk_root: User may specify the path of a QAIRT SDK to pull execution libraries from, otherwise,
                            libraries are derived from the currently installed SDK
            logger (any): A logger instance to be used by the NetRunner module
            clean_up (bool): Indicates whether the on device workspace should be deleted when the instance is destroyed.
        """
        super().__init__(logger)
        self._target_root_exists = False
        self._clean_up = clean_up
        self._backend = backend
        self._device = device if isinstance(device, DeviceInterface) else DeviceFactory.create_device(device)
        self._target_sep = "/" if self._device.device_info.platform_type != DevicePlatformType.WOS else "\\"
        self._target_root = (
            self._TARGET_ROOTS[self._device.device_info.platform_type] + self._target_sep + str(uuid4())
        )
        self._config_artifacts: Dict[str | os.PathLike, str | os.PathLike] = {}
        self._config_artifacts_loaded = False
        self._config_loaded = False
        self._set_config(config)
        if qairt_sdk_root:
            if os.path.isdir(qairt_sdk_root):
                self._qairt_sdk_root = qairt_sdk_root
            else:
                raise NotADirectoryError(f"Provided QAIRT SDK root is not a directory: {qairt_sdk_root}")
        else:
            self._qairt_sdk_root = str(os.environ.get("QNN_SDK_ROOT"))
        self._sdk_artifacts_loaded = False

    def properties(self) -> Dict[str, Any]:
        return self._SCHEMA.model_json_schema()

    def get_logger(self) -> Any:
        return self._logger

    @property
    def clean_up_on_exit(self) -> bool:
        return self._clean_up

    @clean_up_on_exit.setter
    def clean_up_on_exit(self, clean_up: bool) -> None:
        self._clean_up = clean_up

    def load(self, config: Optional[GenieConfig] = None):
        """
        This function will load model and QAIRT sdk artifacts to the on device workspace for execution. Subsequent
        calls to 'run' will only need to push inputs or new model artifacts if the genie config is updated
        Args:
            config: (Optional[GenieConfig]) The user may optionally pass a GenieConfig. This will replace the
            GenieConfig currently associated with the instance. Assets associated with the previous config will be
            unloaded from the device, and the assets associated with the new config will be loaded.
        """
        if not self._target_root_exists:
            self._check_device_return(
                self._device.make_directory(self._target_root),
                f"Failed to make directory {self._target_root} on target\t",
            )
            self._target_root_exists = True

        if not self._sdk_artifacts_loaded:
            self._push_sdk_artifacts()

        if config:
            self._set_config(config)

        if not self._config_loaded:
            with tempfile.TemporaryDirectory() as temp_dir:
                with open(os.path.join(temp_dir, self._GENIE_CONFIG_FILENAME), "w") as f:
                    f.write(json.dumps(self._config.export(), indent=2))

                self._check_device_return(
                    self._device.push(
                        os.path.join(temp_dir, self._GENIE_CONFIG_FILENAME),
                        self._target_sep.join([self._target_root, self._GENIE_CONFIG_FILENAME]),
                    ),
                    "Failed to push genie config to device\t",
                )
                self._config_loaded = True

        if not self._config_artifacts_loaded:
            for dst, src in self._config_artifacts.items():
                self._check_device_return(
                    self._device.push(src, dst), f"Failed to push model artifact {src} to device\t"
                )
            self._config_artifacts_loaded = True

    def run(self, run_config: GenieT2TRunExecutionConfig) -> GenieT2TRunOutputConfig:
        """
        Runs the execution configuration specified by the 'run_config' on the device associated with the instance.
        Any required artifacts not already present on target will be pushed to the working directory so the user is
        not required to call 'load' before 'run'.

        If a GenieConfig is passed in the GenieT2TRunExecutionConfig, it will replace the GenieConfig previously
        associated with the instance. Assets from the previous config will be unloaded from the target as appropriate,
        and the artifacts associated with the new GenieConfig will be loaded.
        Args:
            run_config (GenieT2TRunExecutionConfig): Defines inputs and allows configuration updates after construction
        Returns:
            GenieT2TRunOutputConfig: Output from on device execution
        """
        if run_config.config:
            self._set_config(run_config.config)

        if run_config.sampler:
            self._config.dialog.sampler = run_config.sampler
            self._config_loaded = False

        if run_config.qairt_sdk_root and run_config.qairt_sdk_root != self._qairt_sdk_root:
            self._qairt_sdk_root = run_config.qairt_sdk_root
            self._sdk_artifacts_loaded = False

        self.load()
        env = self._get_device_environment()
        env.shell = True
        cmd = self._prepare_command(run_config)
        genie_t2t_runner_logger.debug(f"Executing command: {cmd} from cwd {env.cwd}")
        device_return = self._check_device_return(
            self._device.execute([cmd], device_env_context=env), f"Failed to execute command: {cmd}\t"
        )
        return GenieT2TRunOutputConfig(
            return_code=device_return.returncode, stderr=device_return.stderr, stdout=device_return.stdout
        )

    def unload(self):
        """
        Removes artifacts from the device that the instance pushed
        """
        if self._target_root_exists:
            self._check_device_return(
                self._device.remove(self._target_root), "Failed to remove artifacts from device\t"
            )
            self._target_root_exists = False
            self._config_loaded = False
            self._config_artifacts_loaded = False
            self._sdk_artifacts_loaded = False

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        pass

    @classmethod
    def _check_device_return(
        cls, device_return: DeviceReturn, error_prefix: str = ""
    ) -> DeviceCompletedProcess:
        if isinstance(device_return, DeviceFailedProcess):
            raise RuntimeError(error_prefix + f"Original Error: {device_return.orig_error}")
        if (
            isinstance(device_return, DeviceCompletedProcess)
            and device_return.returncode != DeviceCode.DEVICE_SUCCESS
        ):
            raise RuntimeError(
                error_prefix + f"return code: {device_return.returncode}\tstderr: {device_return}"
            )
        return device_return

    def _add_config_artifact(self, artifact_path: str | os.PathLike) -> str | os.PathLike:
        if (
            self._device.device_info.connection_type == ConnectionType.LOCAL
            and self._device.device_info.platform_type != DevicePlatformType.ANDROID
        ):
            return artifact_path

        target_path = self._target_sep.join([self._target_root, "artifacts", os.path.basename(artifact_path)])

        # To avoid naming conflicts, prepend a random string if a conflict occurs
        while target_path in self._config_artifacts:
            target_path = self._target_sep.join(
                [self._target_root, "artifacts", str(uuid4())[:6] + os.path.basename(artifact_path)]
            )
        self._config_artifacts[target_path] = artifact_path
        return target_path

    def _set_config(self, config: GenieConfig):
        self._config_artifacts = {}
        self._config_artifacts_loaded = False
        self._config = config
        self._config_loaded = False

        if not os.path.exists(config.dialog.tokenizer.path):
            raise ValueError(
                f"Invalid tokenizer path in provided Genie config: {config.dialog.tokenizer.path}"
            )
        self._config.dialog.tokenizer.path = self._add_config_artifact(config.dialog.tokenizer.path)

        if self._config.dialog.engine.model.type == EngineModelType.LIBRARY:
            assert self._config.dialog.engine.model.library is not None
            assert config.dialog.engine.model.library is not None
            self._config.dialog.engine.model.library.model_bin = self._add_config_artifact(
                config.dialog.engine.model.library.model_bin
            )
        if (
            self._config.dialog.engine.model.type == EngineModelType.BINARY
            and config.dialog.engine.model.binary
            and self._config.dialog.engine.model.binary
        ):
            # for htp

            for i, ctx_bin in enumerate(config.dialog.engine.model.binary.ctx_bins):
                if not os.path.exists(ctx_bin):
                    raise ValueError(
                        f"Context binary path provided in genie config does not exist: {ctx_bin}"
                    )

                self._config.dialog.engine.model.binary.ctx_bins[i] = self._add_config_artifact(ctx_bin)

            if config.dialog.engine.backend.extensions:
                if not os.path.exists(config.dialog.engine.backend.extensions):
                    raise ValueError(
                        "Backend Extensions config path provided in genie config does not exist: "
                        f"{config.dialog.engine.backend.extensions}"
                    )

                self._config.dialog.engine.backend.extensions = self._add_config_artifact(
                    config.dialog.engine.backend.extensions
                )

        if config.dialog.type == DialogType.SSD_Q1 and config.dialog.ssd_q1 and self._config.dialog.ssd_q1:
            if not (
                os.path.isdir(config.dialog.ssd_q1.forecast_prefix_name)
                and os.path.exists(
                    os.path.join(config.dialog.ssd_q1.forecast_prefix_name, self._SSD_KVCACHE_PREFIX_FILENAME)
                )
            ):
                provided_path = (
                    self._config.dialog.ssd_q1.forecast_prefix_name if self._config.dialog.ssd_q1 else None
                )
                raise ValueError(
                    "forecast-prefix-name must point to a directory containing the forecast prefix file"
                    f"named: {self._SSD_KVCACHE_PREFIX_FILENAME}. Invalid path provided: "
                    f"{provided_path}"
                )
            src_path = os.path.join(
                config.dialog.ssd_q1.forecast_prefix_name, self._SSD_KVCACHE_PREFIX_FILENAME
            )
            target_path = self._target_sep.join([self._target_root, "ssd", self._SSD_KVCACHE_PREFIX_FILENAME])
            self._config_artifacts[target_path] = src_path
            self._config.dialog.ssd_q1.forecast_prefix_name = self._target_sep.join(
                [self._target_root, "ssd"]
            )

        if self._config.dialog.engine.model.binary and self._config.dialog.engine.model.binary.lora:
            if self._config.dialog.engine.model.binary.lora.adapters:
                for i, adapter in enumerate(self._config.dialog.engine.model.binary.lora.adapters):
                    bin_sections = adapter.bin_sections if adapter.bin_sections else []
                    for j, binary in enumerate(bin_sections):
                        if not os.path.exists(binary):
                            raise ValueError(f"Provided adapter binary path does not exist: {binary}")

                        self._config.dialog.engine.model.binary.lora.adapters[i].bin_sections[j] = (
                            self._add_config_artifact(binary)
                        )

    def _target_qairt_dir(self) -> str:
        target_name = ""
        if self._device.device_info.platform_type == DevicePlatformType.ANDROID:
            target_name = "aarch64-android"
        elif self._device.device_info.platform_type == DevicePlatformType.X86_64_LINUX:
            target_name = "x86_64-linux-clang"

        if (
            self._device.device_info.connection_type == ConnectionType.LOCAL
            and self._device.device_info.platform_type != DevicePlatformType.ANDROID
        ):
            return self._target_sep.join([str(self._qairt_sdk_root), "bin", target_name])

        return self._target_sep.join([self._target_root, "qairt"])

    def _push_sdk_artifacts(self):
        artifacts_to_push = []
        if (
            self._device.device_info.connection_type == ConnectionType.LOCAL
            and self._device.device_info.platform_type != DevicePlatformType.ANDROID
        ):
            self._sdk_artifacts_loaded = True
            return

        if self._device.device_info.platform_type == DevicePlatformType.ANDROID:
            target_name = "aarch64-android"
        elif self._device.device_info.platform_type == DevicePlatformType.X86_64_LINUX:
            target_name = "x86_64-linux-clang"
        else:
            raise ValueError(f"Unsupported platform: {self._device.platform_type}")

        artifacts_to_push.append(os.path.join(self._qairt_sdk_root, "bin", target_name, "genie-t2t-run"))
        artifacts_to_push.append(os.path.join(self._qairt_sdk_root, "lib", target_name, "libGenie.so"))
        artifacts_to_push.append(os.path.join(self._qairt_sdk_root, "lib", target_name, "libQnnSystem.so"))
        artifacts_to_push.append(
            os.path.join(self._qairt_sdk_root, "lib", target_name, f"libQnn{self._backend.capitalize()}.so")
        )
        backend = create_backend(self._backend, Target(type=self._device.device_info.platform_type))
        artifacts_to_push.extend(backend.get_required_device_artifacts(str(self._qairt_sdk_root)))

        if self._backend == BackendType.HTP:
            artifacts_to_push.append(
                os.path.join(self._qairt_sdk_root, "lib", target_name, "libQnnHtpNetRunExtensions.so")
            )
        if self._backend == BackendType.CPU:
            artifacts_to_push.append(
                os.path.join(self._qairt_sdk_root, "lib", target_name, "libQnnGenAiTransformer.so")
            )
            artifacts_to_push.append(
                os.path.join(self._qairt_sdk_root, "lib", target_name, "libQnnGenAiTransformerCpuOpPkg.so")
            )
            artifacts_to_push.append(
                os.path.join(self._qairt_sdk_root, "lib", target_name, "libQnnGenAiTransformerModel.so")
            )

        for artifact in artifacts_to_push:
            self._check_device_return(
                self._device.push(
                    artifact, self._target_sep.join([self._target_qairt_dir(), os.path.basename(artifact)])
                ),
                f"Failed to push artifact: {artifact}\t",
            )

        self._sdk_artifacts_loaded = True

    def _get_device_environment(self) -> DeviceEnvironmentContext:
        env = DeviceEnvironmentContext()
        env.cwd = self._target_root
        if self._device.device_info.platform_type == DevicePlatformType.ANDROID:
            env.environment_variables["LD_LIBRARY_PATH"] = self._target_qairt_dir()
            env.environment_variables["ADSP_LIBRARY_PATH"] = self._target_qairt_dir()
            env.shell = True
        return env

    def _prepare_command(self, run_config: GenieT2TRunExecutionConfig) -> str:
        cmd = []
        if self._device.device_info.platform_type in [
            DevicePlatformType.ANDROID,
            DevicePlatformType.X86_64_LINUX,
        ]:
            cmd.append(self._target_sep.join([self._target_qairt_dir(), "genie-t2t-run"]))
        else:
            raise RuntimeError(
                f"Requested platform {self._device.device_info.platform_type} is currently not supported."
            )
        cmd.append("-c")
        cmd.append(self._GENIE_CONFIG_FILENAME)
        if run_config.prompt:
            # genie-t2t-run -c genie_config.json -p "prompt"
            cmd.append("-p")
            cmd.append(f'"{run_config.prompt}"')
        elif run_config.prompt_file:
            # genie-t2t-run -c genie_config.json --prompt_file /path/to/prompt_file.txt
            dst_path = self._target_sep.join(
                [self._target_root, "input", os.path.basename(run_config.prompt_file)]
            )
            self._check_device_return(
                self._device.push(run_config.prompt_file, dst_path),
                f"Failed to push prompt file: {run_config.prompt_file}\t",
            )
            cmd.append("--prompt_file")
            cmd.append(dst_path)
        elif run_config.token_file:
            # genie-t2t-run -c genie_config.json --token_file /path/to/token_file.raw
            dst_path = self._target_sep.join(
                [self._target_root, "input", os.path.basename(run_config.token_file)]
            )
            self._check_device_return(
                self._device.push(run_config.token_file, dst_path),
                f"Failed to push token file: {run_config.token_file}\t",
            )
            cmd.append("--token_file")
            cmd.append(dst_path)
        elif run_config.embedding_config:
            # float32 embedding input and embedding table
            # genie-t2t-run -c genie_config.json -e /path/to/embedding_input.raw -t /path/to/embedding_input.raw

            # quantized embedding input and embedding table
            # genie-t2t-run -c genie_config.json -e /path/to/embedding_input.raw,DATATYPE,SCALE,OFFSET  \
            # -t /path/to/embedding_input.raw,DATATYPE,SCALE,OFFSET
            input_dst_path = self._target_sep.join(
                [self._target_root, "input", os.path.basename(run_config.embedding_config.input.path)]
            )
            self._check_device_return(
                self._device.push(run_config.embedding_config.input.path, input_dst_path),
                (f"Failed to push embedding input file: {run_config.embedding_config.input.path}\t"),
            )

            embedding_table_dst_path = self._target_sep.join(
                [
                    self._target_root,
                    "input",
                    os.path.basename(run_config.embedding_config.embedding_table.path),
                ]
            )
            self._check_device_return(
                self._device.push(run_config.embedding_config.embedding_table.path, embedding_table_dst_path),
                (
                    "Failed to push embedding table file: "
                    f"{run_config.embedding_config.embedding_table.path}\t"
                ),
            )

            input_arg = input_dst_path
            if run_config.embedding_config.input.quantization:
                input_arg += "," + str(run_config.embedding_config.input.quantization)

            embedding_table_arg = embedding_table_dst_path
            if run_config.embedding_config.embedding_table.quantization:
                embedding_table_arg += "," + str(run_config.embedding_config.embedding_table.quantization)

            cmd.extend(["-e", str(input_arg), "-t", str(embedding_table_arg)])
        else:
            raise ValueError("Please provide a prompt, prompt file, or embedding input and embedding table")

        if run_config.lora_config:
            # genie-t2t-run -c genie_config.json <one of above inputs> -l adapter_name,alpha_tensor_name,alpha_value
            adapter_name = run_config.lora_config.adapter_name
            if not (
                self._config.dialog.engine.model.binary
                and self._config.dialog.engine.model.binary.lora
                and adapter_name in [x.name for x in self._config.dialog.engine.model.binary.lora.adapters]
            ):
                raise ValueError(f"Requested lora adapter not present in genie config: {adapter_name}")
            alpha_tensor_name = self._config.dialog.engine.model.binary.lora.alpha_tensor_name
            alpha_value = run_config.lora_config.alpha
            cmd.append("-l")
            cmd.append(f"{adapter_name},{alpha_tensor_name},{alpha_value}")

        return " ".join(cmd)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    def __del__(self):
        if self._clean_up:
            self.unload()
