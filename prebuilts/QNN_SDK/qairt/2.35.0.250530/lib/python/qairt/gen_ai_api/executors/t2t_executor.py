# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
import os
import platform
import tempfile
from typing import List, Optional, Sequence

from typing_extensions import Self

from qairt import CacheModule, CompileConfig, CompiledModel, Device, DlcModule
from qairt.api.compiler.backends.htp import HtpConfigHelper, HtpContextConfig, HtpDeviceConfig, HtpGraphConfig
from qairt.api.configs.common import BackendType
from qairt.gen_ai_api.configs.gen_ai_config import GenAIConfig
from qairt.gen_ai_api.executors.gen_ai_executor import (
    GenAIExecutor,
    GenerationExecutionResult,
    GenerationMetrics,
)
from qairt.modules.genie_execution.genie_config import (
    Context,
    Dialog,
    DialogEngine,
    DialogType,
    EngineBackend,
    EngineBackendType,
    EngineModel,
    EngineModelType,
    GenieConfig,
    GenieConfigEncoder,
    ModelBinary,
    ModelLibrary,
    QnnGenAiTransformerBackend,
    QnnHtpBackend,
    Sampler,
    Tokenizer,
)
from qairt.modules.genie_execution.genie_t2t_run_module import GenieT2TRunExecutionConfig, GenieT2TRunner
from qairt.modules.genie_execution.native_t2t_module import GenieNativeT2TRunner
from qairt.utils import loggers
from qti.aisw.tools.core.utilities.devices import DevicePlatformType


class T2TExecutor(GenAIExecutor):
    _logger = loggers.get_logger(name=__name__)

    def __init__(
        self,
        models: List[CompiledModel],
        genai_config: GenAIConfig,
        backend: BackendType,
        device: Optional[Device] = None,
        compile_config: Optional[CompileConfig] = None,
        qairt_sdk_root: Optional[str | os.PathLike] = None,
        clean_up: bool = True,
    ):
        """
        The executor has a compiled container and device to run it on. The GenAIExecutor will use the
        device to maintain an execution environment and run the compiled container

        Args:
            models (List[CompiledModel]): The compiled models to run
            genai_config (GenAIConfig): GenaiConfig to base GenieConfig on for genie-t2t-run execution
            device (Device): Device to execute on
            compile_config (CompileConfig): CompileConfig to base GenieConfig on for genie-t2t-run execution
            qairt_sdk_root (str | os.PathLike): Path to QAIRT SDK with execution libraries (if different from installed QAIRT)
            clean_up (bool): Will delete artifacts pushed to device if True

        """
        self._clean_up = clean_up
        self._runner: GenieT2TRunner | None = None
        self._native_runner: GenieNativeT2TRunner | None = None

        self._environment_prepared = False
        self._work_dir: tempfile.TemporaryDirectory | None = None

        self._models: List[CompiledModel] = models
        if not self._models:
            raise ValueError("No models provided for execution")
        self._gen_ai_config = genai_config
        self._device: Optional[Device] = device
        self._backend: BackendType = backend
        self._compile_config: Optional[CompileConfig] = compile_config
        self._qairt_sdk_root: str | os.PathLike = qairt_sdk_root or str(os.environ.get("QNN_SDK_ROOT", ""))

        if self._compile_config and self._backend == BackendType.CPU:
            raise ValueError(
                f"Compile config provided but backend extensions are not supported for requested backend: {self._backend}"
            )

    def prepare_environment(self) -> Self:
        """
        Prepares artifacts for execution on target
        """
        if not self._environment_prepared:
            self._work_dir = tempfile.TemporaryDirectory()

            backend_extensions_path = ""
            if self._compile_config:
                backend_extensions_path = os.path.join(self._work_dir.name, "backend_extensions.json")
                extensions_dict = HtpConfigHelper.to_backend_extension_dict(
                    context_configs=[
                        x
                        for x in self._compile_config.context_custom_configs
                        if isinstance(x, HtpContextConfig)
                    ]
                    if self._compile_config.context_custom_configs
                    else None,
                    graph_configs=[
                        x for x in self._compile_config.graph_custom_configs if isinstance(x, HtpGraphConfig)
                    ]
                    if self._compile_config.graph_custom_configs
                    else None,
                    device_configs=[
                        x
                        for x in self._compile_config.device_custom_configs
                        if isinstance(x, HtpDeviceConfig)
                    ]
                    if self._compile_config.device_custom_configs
                    else None,
                    memory_config=self._compile_config.memory_custom_config,
                )
                with open(backend_extensions_path, "w") as f:
                    f.write(json.dumps(extensions_dict, indent=2))

            model_paths = []
            for model in self._models:
                if isinstance(model.module, DlcModule):
                    caches = list(model.module.caches.values())
                    if len(caches) > 1:
                        raise ValueError("Expected only a single context binary cache per dlc")
                    model_paths.append(caches[0].path)
                if isinstance(model.module, CacheModule):
                    model_paths.append(model.module.path)

            genie_config = self._build_genie_config(model_paths, backend_extensions_path)

            if self._is_native_execution():
                self._native_runner = GenieNativeT2TRunner(genie_config=genie_config)

            else:
                self._runner = GenieT2TRunner(
                    genie_config,
                    self._backend,
                    self._device.info,
                    self._qairt_sdk_root,
                    clean_up=self._clean_up,
                )
                self._runner.load()
        self._environment_prepared = True
        return self

    def clean_environment(self) -> Self:
        """
        Removes artifacts from target environment
        """
        self._logger.info("Cleaning environment")
        if self._runner:
            self._runner.unload()
            self._runner = None

        if self._work_dir:
            self._work_dir.cleanup()

        if self._native_runner:
            del self._native_runner
            self._native_runner = None

        self._environment_prepared = False
        return self

    def generate(self, prompt: str) -> GenerationExecutionResult:
        """
        Runs the prepared llm and returns a GenerationExecutionResult
        """
        out = GenerationExecutionResult()

        if not self._environment_prepared:
            self.prepare_environment()

        if self._is_native_execution() and self._native_runner:
            result = self._native_runner.query(prompt)
            self._native_runner.reset_dialog()
            return result
        else:
            return self._generate_non_native(prompt)

    def _generate_non_native(self, prompt: str) -> GenerationExecutionResult:
        """
        Runs the prepared llm and returns a GenerationExecutionResult
        """
        out = GenerationExecutionResult()

        if not self._runner:
            raise RuntimeError("Environment preparation failed")
        else:
            t2t_result = self._runner.run(GenieT2TRunExecutionConfig(prompt=prompt))

            out.output = t2t_result.stdout

            if t2t_result.return_code != 0:
                out.error = t2t_result.stderr
            else:
                out.output += "\n" + t2t_result.stderr
            self._logger.debug(f"stdout: {t2t_result.stdout}\nstderr:{t2t_result.stderr}")

            begin_idx = out.output.find("[BEGIN]:")
            end_idx = out.output.find("[END]", begin_idx)
            if begin_idx > -1 and end_idx > begin_idx:
                out.generated_text = out.output[begin_idx + len("[BEGIN]:") : end_idx].strip()

            kpi_idx = out.output.find("[KPIS]", end_idx)

            metrics = GenerationMetrics()
            metrics.init_time = self._parse_time(out.output, kpi_idx, "Init Time: ")
            metrics.prompt_processing_time = self._parse_time(out.output, kpi_idx, "Prompt Processing Time: ")
            metrics.token_generation_time = self._parse_time(out.output, kpi_idx, "Token Generation Time: ")
            metrics.prompt_processing_rate = self._parse_tokens_per_second(
                out.output, kpi_idx, "Prompt Processing Rate : "
            )
            metrics.token_generation_rate = self._parse_tokens_per_second(
                out.output, kpi_idx, "Token Generation Rate: "
            )
            out.metrics = metrics

        return out

    def _is_native_execution(self) -> bool:
        if not self._device:
            return True
        elif self._device.identifier is None:
            if (
                platform.system() == "Linux"
                and platform.machine() == "x86_64"
                and self._device.type == DevicePlatformType.X86_64_LINUX
            ):
                return True

            if platform.system() == "Windows":
                if (
                    "AMD64" in platform.processor() or "Intel64" in platform.processor()
                ) and self._device.type == DevicePlatformType.X86_64_WINDOWS_MSVC:
                    return True
                elif (
                    "ARM64" in platform.processor() or "AARCH64" in platform.processor()
                ) and self._device.type == DevicePlatformType.WOS:
                    return True

        return False

    def _build_genie_config(
        self,
        model_paths: Sequence[str | os.PathLike],
        backend_extensions_path: Optional[str | os.PathLike] = None,
    ) -> GenieConfig:
        if not model_paths:
            raise RuntimeError("No models were loaded into the container.  Cannot build Genie config.")

        if self._backend == BackendType.HTP:
            engine_backend = EngineBackend(
                type=EngineBackendType.QNN_HTP,
                QnnHtp=QnnHtpBackend(
                    poll=True,
                    use_mmap=True,
                    spill_fill_bufsize=0,
                    mmap_budget=40,
                    kv_dim=self._gen_ai_config.kv_dim,
                ),
            )
            if backend_extensions_path:
                engine_backend.extensions = backend_extensions_path
            engine_model = EngineModel(
                type=EngineModelType.BINARY,
                binary=ModelBinary(ctx_bins=model_paths),
            )
        elif self._backend == BackendType.CPU:
            engine_backend = EngineBackend(QnnGenAiTransformer=QnnGenAiTransformerBackend())
            engine_model = EngineModel(library=ModelLibrary(model_bin=model_paths[0]))

        else:
            raise RuntimeError(f"Unsupported backend: {self._backend}")

        return GenieConfig(
            dialog=Dialog(
                type=DialogType.BASIC,
                context=Context(
                    size=self._gen_ai_config.context_length,
                    n_vocab=self._gen_ai_config.n_vocab,
                    bos_token=self._gen_ai_config.bos_token,
                    eos_token=self._gen_ai_config.eos_token,
                    eot_token=self._gen_ai_config.eot_token,
                ),
                sampler=Sampler(seed=42, temp=1.2, top_k=20, top_p=0.75),
                tokenizer=Tokenizer(path=self._gen_ai_config.tokenizer_path),
                engine=DialogEngine(backend=engine_backend, model=engine_model),
            )
        )

    def _parse_time(self, response: str, kpi_idx: int, starting_substring: str) -> int:
        time = -1
        start_idx = response.find(starting_substring, kpi_idx)
        end_idx = response.find("us", start_idx)
        if start_idx > -1 and end_idx > start_idx:
            time = int(response[start_idx + len(starting_substring) : end_idx - 1])
        return time

    def _parse_tokens_per_second(self, response: str, kpi_idx: int, starting_substring: str) -> float:
        tokens_per_second = 0
        start_idx = response.find(starting_substring, kpi_idx)
        end_idx = response.find("toks/sec", start_idx)
        if start_idx > -1 and end_idx > start_idx:
            tokens_per_second = response[start_idx + len(starting_substring) : end_idx - 1]
        return tokens_per_second

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._clean_up:
            self.clean_environment()
        return False

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._clean_up = False
        return instance

    def __del__(self):
        if self._clean_up:
            self.clean_environment()
