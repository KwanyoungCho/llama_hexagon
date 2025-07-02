# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    from qti.aisw.tools.core.modules.context_bin_gen import GenerateConfig
    from qti.aisw.tools.core.modules.net_runner import InferenceConfig

from qti.aisw.core.model_level_api.executor.executor import Executor
from qti.aisw.core.model_level_api.model.context_binary import QnnContextBinary
from qti.aisw.core.model_level_api.model.dlc import DLC
from qti.aisw.core.model_level_api.model.model import Model, ModelT
from qti.aisw.core.model_level_api.model.model_library import QnnModelLibrary
from qti.aisw.core.model_level_api.utils.exceptions import (
    ContextBinaryGenerationError,
    InferenceError,
    return_code_to_netrun_error_enum,
)
from qti.aisw.core.model_level_api.utils.qnn_profiling import (
    ProfilingData,
    get_backend_profiling_data,
)
from qti.aisw.core.model_level_api.utils.subprocess_executor import (
    create_op_package_argument,
    create_set_output_tensors_argument,
    generate_config_file,
    generate_input_list,
    output_dir_to_np_array,
    update_run_config_native_inputs,
)
from qti.aisw.tools.core.modules.api import (
    generate_context_bin_cli_args,
    generate_net_runner_cli_args,
)
from qti.aisw.tools.core.modules.api.backend.backend import Backend

logger = logging.getLogger(__name__)

NamedTensorData = Dict[str, np.ndarray]


class X86SubprocessExecutor(Executor):
    def __init__(self):
        super().__init__()
        self._run_config = None

    def setup(self, workflow_mode, backend, model, sdk_root, config, output_dir):
        # cache the run config in case one is not provided to run_inference(), in which case this
        # config will be used
        self._run_config = config

        # no profiling data generated
        return None

    def run_inference(self, config: Optional['InferenceConfig'], backend: Backend, model: Model,
                      sdk_root: str, input_data: Union[NamedTensorData, Path, List[NamedTensorData], List[np.ndarray]],
                      output_dir: str) -> Tuple[List[np.ndarray], ProfilingData | None]:
        """
        Runs the inference workflow, including setting up, processing, and retrieving results.

        Args:
            config: Configuration for running inference.
            backend: The backend to run inference on.
            model: The model to be used in the inference.
            sdk_root: SDK root path.
            input_data: Input data for inference.
            output_dir: Directory to store the output.

        Returns:
            The inference output NumPy array and profiling data.
        """
        if config is None:
            config = self._run_config

        temp_directory = TemporaryDirectory()
        logger.debug(f"created dir: {temp_directory.name}")
        backend.before_run_hook(temp_directory.name, sdk_root)

        input_list_filename, _ = generate_input_list(input_data, temp_directory.name)
        config_file_arg, _ = generate_config_file(
            backend, temp_directory.name, sdk_root
        )
        op_package_arg = create_op_package_argument(backend)

        backend_lib = sdk_root + "/lib/x86_64-linux-clang/" + backend.backend_library
        if not Path(backend_lib).is_file():
            raise FileNotFoundError(f"Could not find backend library: {backend_lib}")

        netrun = sdk_root + "/bin/x86_64-linux-clang/qnn-net-run"
        model_arg = self._create_inference_model_argument(model, sdk_root)
        netrun_output_dir = Path(temp_directory.name, "output")
        output_tensors_arg = create_set_output_tensors_argument(
            config, backend, temp_directory.name
        )

        netrun_command = (
            f"{netrun} --backend {backend_lib} --input_list {input_list_filename} "
            f"{model_arg} --output_dir {netrun_output_dir} {config_file_arg} "
            f"{op_package_arg} {output_tensors_arg} "
        )

        from qti.aisw.tools.core.modules.net_runner import (
            InferenceConfig,  #Lazy import due to circular dependecy
        )
        if not config:
            config = InferenceConfig()

        update_run_config_native_inputs(config, input_data)
        netrun_command += generate_net_runner_cli_args(config)

        logger.debug(f"Running command: {netrun_command}")
        return_code, stdout, stderr = backend.target.run_command(
            netrun_command, cwd=temp_directory.name
        )
        if return_code != 0:
            err_str = (
                f"qnn-net-run execution failed, stdout: {stdout}, stderr: {stderr}"
            )
            netrun_error_enum = return_code_to_netrun_error_enum(return_code)
            if netrun_error_enum:
                raise InferenceError(netrun_error_enum, err_str)
            raise RuntimeError(err_str)

        if config and config.log_level:
            print("stdout: ", *stdout, sep="\n")

        backend.after_run_hook(temp_directory.name, sdk_root)

        profiling_data = None
        if config and config.profiling_level:
            profiling_data = get_backend_profiling_data(
                backend, output_dir, netrun_output_dir
            )

        native_outputs = config and config.use_native_output_data
        return output_dir_to_np_array(netrun_output_dir, native_outputs), profiling_data

    def generate_context_binary(
        self,
        config: Optional['GenerateConfig'],
        backend: Backend,
        model: Union[ModelT, List[ModelT]],
        sdk_root: str,
        output_path: str,
        output_filename: str,
        backend_specific_filename: str = '') -> Tuple[QnnContextBinary,
                                                      Optional[str],
                                                      Optional[ProfilingData]]:
        """
        Main function to orchestrate context binary generation for the backend, model, and target device.

        Args:
            config: Configuration object for context binary generation.
            backend: The backend object which includes the target device and backend-specific configurations.
            model: The model for which the context binary is generated.
            sdk_root: SDK root directory path.
            output_path: Path where the final output binary will be stored.
            output_filename: Filename for the generated binary.
            backend_specific_filename: Filename for the generated backend-specific binary

        Returns:
            Tuple: A tuple containing:
                QnnContextBinary: An object storing the name and path of the generated binary
                str: Path of the generated backend-specific binary
                Profiling_data: An object containing the profiling log and any backend profiling artifacts
        """

        temp_directory = TemporaryDirectory()
        logger.debug(f"created dir: {temp_directory.name}")
        backend.before_generate_hook(temp_directory.name, sdk_root)

        backend_lib = sdk_root + "/lib/x86_64-linux-clang/" + backend.backend_library
        if not Path(backend_lib).is_file():
            raise FileNotFoundError(f"Could not find backend library: {backend_lib}")

        context_binary_generator = (
            sdk_root + "/bin/x86_64-linux-clang/qnn-context-binary-generator"
        )
        model_arg = self._create_context_binary_generator_model_argument(
            model, sdk_root
        )

        config_file_arg, _ = generate_config_file(
            backend, temp_directory.name, sdk_root
        )
        op_package_arg = create_op_package_argument(backend)

        output_tensors_arg = create_set_output_tensors_argument(
            config, backend, temp_directory.name
        )

        model_name = model[0].name if isinstance(model, list) else model.name
        if output_filename:
            output_filename = output_filename + ".bin"
        elif model_name:
            output_filename = model_name + ".bin"
        else:
            output_filename = 'context.bin'

        abs_output_path = Path(output_path).absolute()
        abs_output_filepath = abs_output_path / output_filename
        context_binary_filename = abs_output_filepath.stem

        backend_binary_filename = ''
        abs_backend_output_filepath = None
        if backend_specific_filename:
            backend_specific_filename = backend_specific_filename + '.bin'
            abs_backend_output_filepath = abs_output_path / backend_specific_filename
            backend_binary_filename = abs_backend_output_filepath.stem

        context_bin_command = f'{context_binary_generator} --backend {backend_lib} ' \
                              f'{model_arg} ' \
                              f'--binary_file {context_binary_filename} ' \
                              f'--backend_binary {backend_binary_filename} ' \
                              f'--output_dir {abs_output_path} {config_file_arg} ' \
                              f'{op_package_arg} {output_tensors_arg} '

        if config:
            context_bin_command += generate_context_bin_cli_args(config)

        logger.debug(f"Running command: {context_bin_command}")
        return_code, stdout, stderr = backend.target.run_command(
            context_bin_command, cwd=temp_directory.name
        )
        if return_code != 0:
            err_str = (
                f"qnn-context-binary-generator execution failed, stdout: {stdout}, stderr: "
                f"{stderr}"
            )
            netrun_error_enum = return_code_to_netrun_error_enum(return_code)
            if netrun_error_enum:
                raise ContextBinaryGenerationError(netrun_error_enum, err_str)
            raise RuntimeError(err_str)

        if config and config.log_level:
            print("stdout: ", *stdout, sep="\n")

        backend.after_generate_hook(temp_directory.name, sdk_root)

        profiling_data = None
        if config and config.profiling_level:
            profiling_data = get_backend_profiling_data(backend, abs_output_path, None)

        context_binary = QnnContextBinary(context_binary_filename, str(abs_output_filepath))
        backend_binary_path = None

        if abs_backend_output_filepath and abs_backend_output_filepath.is_file():
            backend_binary_path = str(abs_backend_output_filepath)

        return context_binary, backend_binary_path, profiling_data

    @staticmethod
    def _create_context_binary_argument(model):
        binary_path = Path(model.binary_path)
        if not binary_path.is_file():
            raise FileNotFoundError(f"Could not find context binary: {binary_path}")
        return f"--retrieve_context {binary_path.resolve()}"

    @staticmethod
    def _create_model_lib_argument(model: QnnModelLibrary):
        model_path = Path(model.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find model library: {model_path}")
        return f"--model {model_path.resolve()}"

    @staticmethod
    def _create_dlc_argument(model: Union[DLC, List[DLC]], sdk_root: str) -> str:
        if isinstance(model, list):
            if not all(isinstance(model_, DLC) for model_ in model):
                raise ValueError("Model objects are not all of type: DLC")

            dlc_paths = ",".join(
                str(Path(model_.dlc_path).resolve(strict=True)) for model_ in model
            )
        else:
            dlc_paths = Path(model.dlc_path).resolve(strict=True)

        return f"--dlc_path {dlc_paths}"

    @staticmethod
    def _create_inference_model_argument(model: ModelT, sdk_root: str) -> str:
        try:
            return X86SubprocessExecutor._create_model_lib_argument(model)
        except AttributeError:
            pass

        try:
            return X86SubprocessExecutor._create_dlc_argument(model, sdk_root)
        except AttributeError:
            pass

        try:
            return X86SubprocessExecutor._create_context_binary_argument(model)
        except AttributeError:
            raise ValueError("Could not retrieve path from provided model")

    @staticmethod
    def _create_context_binary_generator_model_argument(
        model: Union[QnnModelLibrary, DLC, List[DLC]], sdk_root: str
    ) -> str:

        def is_dlc():
            if isinstance(model, list):
                return all(isinstance(model_, DLC) for model_ in model)
            return isinstance(model, DLC)

        if isinstance(model, QnnModelLibrary):
            return X86SubprocessExecutor._create_model_lib_argument(model)
        elif is_dlc():
            return X86SubprocessExecutor._create_dlc_argument(model, sdk_root)
        else:
            raise ValueError(
                "Model type is not supported. Expected DLC(s) or QnnModelLibrary (.so)"
            )
