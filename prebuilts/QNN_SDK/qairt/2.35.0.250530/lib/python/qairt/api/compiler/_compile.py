# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union, overload

from qairt.api.compiled_model import CompiledModel
from qairt.api.compiler.config import CompileConfig
from qairt.api.compiler.config_util import get_config_api_options_dict
from qairt.api.configs.common import BackendType, ProfilingData
from qairt.api.model import Model
from qairt.api.profiler.profiler import profile
from qairt.modules.cache_module import CacheModule
from qairt.modules.qti_import import qti_module_api
from qairt.utils.asset_utils import Asset, AssetType, check_asset_type
from qairt.utils.exceptions import CompilationError
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.modules.context_bin_gen import ContextBinGen, ContextBinGenArgConfig, GenerateConfig

_compile_logger = get_logger("qairt.compile")

# TODO:
# - Config support (arguments other than backend extensions)
# - GPU Offline Preparation
# - Compile for multiple dsp architectures


@overload
def compile(model: Union[Model, List[Model]], backend: str | BackendType) -> CompiledModel: ...


@overload
def compile(model: Union[Model, List[Model]], config: CompileConfig) -> CompiledModel: ...


@profile("compile")
def compile(
    model: Union[Model, List[Model]],
    *,
    backend: Optional[str | BackendType] = None,
    config: Optional[CompileConfig] = None,
) -> CompiledModel:
    """
    Compile the given model according to a specified backend and optional
    configuration options.


    Args:
        model: The model(s) object to compile. If a list of models is provided, the
               models will be compiled into a single compiled model.
        backend: The QAIRT backend to compile the model to. Uses default configuration
                 options.
        config: The compile configuration to use. See `qairt.api.compiler.config.CompileConfig`
                for the full list of options.
                Note that if both backend and config are specified, the backend specified in the
                config will be used.

    Examples:
        .. code-block:: python

        import qairt
        from qairt import CompileConfig

        my_model = qairt.load("my_model.dlc")

        # Compile for a QAIRT Backend

        compiled_model = qairt.compile(my_model, backend="HTP")
        print (compiled_model.module.info)

        # Compile for a QAIRT Backend with configuration

        compile_config = CompileConfig(backend="HTP", soc_details="dsp_arch:v79;soc_model:69")
        compiled_model_v79 = qairt.compile(my_model, config=compile_config)

        # Set compilation modes
        compile_config =  CompileConfig(backend="HTP",
                                        soc_details="chipset:SM8550").set_mode("weight_sharing")
        shared_model = qairt.load("shared_model.dlc")
        compiled_model_8550 = qairt.compile([my_model, my_shared_model], config=compile_config)


    Returns:
       CompiledModel: A new CompiledModel instance containing a reference to a generated
                      context binary.

    Raises:
        CompilationError: If a context binary could not be generated from the Model.
    """

    if config is not None:
        backend = config.backend
    elif backend is None:
        raise ValueError("Either backend or config must be specified")

    def validate_and_save_model(_model):
        if isinstance(_model, CompiledModel):
            raise TypeError(
                f" Compile is not a valid operation on a compiled model: {_model.name}. "
                f" Expected instance of type: {Model.__name__}, got {CompiledModel.__name__}"
            )

        # Models need to be saved before binary generation can be called
        if not _model.module.path:
            _model.save()

    shared_context_models = []
    if isinstance(model, list):
        validate_and_save_model(model[0])

        if len(model) > 1:
            for sm in model[1:]:
                validate_and_save_model(sm)
                shared_context_models.append(sm)
        model = model[0]
    else:
        validate_and_save_model(model)

    if config is None:
        config = CompileConfig(backend=backend)

    # Set Generate Config
    # TODO: Unify configs
    generate_config = GenerateConfig(
        input_output_tensor_mem_type=config.io_tensor_mem_type,
        set_output_tensors=config.set_output_tensors,
        enable_intermediate_outputs=config.enable_intermediate_outputs,
        profiling_level=getattr(config, "profiling_level", None),
        profiling_option=getattr(config, "profiling_option", None),
        log_level=config.log_level,
        op_packages=config.op_packages,
    )

    if shared_context_models:
        # Ensure graphs names are set for shared context models
        # This can only be enabled via config.set_mode
        all_models = [model, *shared_context_models]

        _compile_logger.debug(f"Detected multiple models: {(model.name for model in all_models)}")

        # This code broadcasts the same graph configuration across multiple models
        # if ensure graphs is set
        if getattr(config, "ensure_graphs", False) and config.graph_custom_configs is not None:
            base_graph_config = config.graph_custom_configs[0]
            config.graph_custom_configs = []
            for md in all_models:
                graphs_info = md.module.graphs_info
                for gi in graphs_info:
                    shared_config = base_graph_config.model_copy()
                    shared_config.name = gi
                    config.graph_custom_configs.append(shared_config)

        shared_context_models = [
            qti_module_api.Model(dlc_path=scm.module.path) for scm in shared_context_models
        ]

    # attach any shared context models
    ctx_bin_arg_models = [qti_module_api.Model(dlc_path=model.module.path)] + shared_context_models

    # retrieve API config options
    qnn_api_config_options = get_config_api_options_dict(config)
    backend_custom_config_dict = qnn_api_config_options["backend_extensions"]["config_dict"]
    _compile_logger.debug(f"Backend custom options set to: {backend_custom_config_dict}")

    # Create ContextBinGenArgConfig
    root_dir = os.getenv("QAIRT_TMP_DIR", default=tempfile.gettempdir())
    tmp_dir = Path(tempfile.mkdtemp(prefix="temp_working_dir_", dir=root_dir))
    ctx_arg_config = ContextBinGenArgConfig(
        backend=backend,
        model=ctx_bin_arg_models,
        backend_config_dict=backend_custom_config_dict,
        generate_config=generate_config,
        output_filename=model.name,
        output_dir=tmp_dir,
    )

    # Init context binary generator
    context_bin_gen = ContextBinGen(logger=_compile_logger)

    out_config = context_bin_gen.generate(ctx_arg_config)
    try:
        # load cache module from generated path
        cache_module = CacheModule.load(path=out_config.context_binary.context_binary_path)
        cache_module.working_directory = tmp_dir  # delete this cache if it is never saved
    except Exception as e:
        _compile_logger.debug(f"Could not load context binary for {model.name}: {e}")
        raise CompilationError(f"Could not create compiled binary for {model.name}")

    # create compiled model
    # TODO: This should be creating a dlc module and populating it with cache info
    # Doing this temporarily until DLC Interactions API is fully developed
    # JIRA: https://jira-dc.qualcomm.com/jira/browse/AISW-121951
    compiled_model = CompiledModel(cache_module, backend, config=config)

    _compile_logger.info(f"Compiled model: {model.name} for backend: {backend}")

    if out_config.profiling_data:
        _handle_compile_profile_data(compiled_model, out_config)

    return compiled_model


def _handle_compile_profile_data(compiled_model, out_config):
    profiling_log = Path(out_config.profiling_data.profiling_log)
    # Check and set profiling asset
    check_asset_type(AssetType.PROFILING_LOG, profiling_log)
    setattr(compiled_model, "profiling_data", ProfilingData(profiling_log=profiling_log))

    compiled_model.assets[profiling_log.name] = Asset(path=profiling_log, delete=True)
    _compile_logger.debug(f"Retrieved profiling log: {profiling_log.name}")

    if out_config.profiling_data.backend_profiling_artifacts:
        for artifact in out_config.profiling_data.backend_profiling_artifacts:
            artifact = Path(artifact)
            check_asset_type(AssetType.SCHEMATIC_BIN, artifact)
            compiled_model.assets[artifact.name] = Asset(path=artifact, delete=True)
            _compile_logger.debug(f"Retrieved profiling artifact: {artifact.name}")
