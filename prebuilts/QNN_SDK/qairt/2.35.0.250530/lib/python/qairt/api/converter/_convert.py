# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import tempfile
from os import PathLike
from pathlib import Path
from typing import List, Optional, cast

from typing_extensions import Unpack

from qairt.api.converter.convert_util import (
    _calibration_config_arg_adapter,
    _converter_config_arg_adapter,
    disable_spawn_process_and_exec,
)
from qairt.api.converter.converter_config import CalibrationConfig, ConverterConfig
from qairt.api.model import Model
from qairt.modules.dlc_module import DlcModule
from qairt.utils.exceptions import ApplyEncodingsError, ConversionError, OptimizationError
from qairt.utils.loggers import get_logger
from qti.aisw.core.model_level_api.utils.subprocess_executor import (
    generate_input_list,
)
from qti.aisw.tools.core.modules.converter.converter_module import (
    ConverterInputConfig,
    QAIRTConverter,
)
from qti.aisw.tools.core.modules.converter.optimizer_module import (
    OptimizerInputConfig,
    QAIRTOptimizer,
)
from qti.aisw.tools.core.modules.converter.quantizer_module import (
    QAIRTQuantizer,
    QuantizerInputConfig,
)

_convert_logger = get_logger("qairt.convert")

# TODO:
# - AISW-112821: Add support for in-memory model in QAIRT converter
# - AISW-115745: Add support for in-memory DLC output in QAIRT optimizer
# - AISW-115738: Add support for in-memory DLC in QAIRT quantizer
# - Add support for in-memory encodings object (TBD)


def convert(
    model: str | PathLike,
    encodings: Optional[str | PathLike] = None,
    calibration_config: Optional[CalibrationConfig] = None,
    **extra_args: Unpack[ConverterConfig],
) -> Model:
    """
    Convert a framework model into a Model object.

    Args:
        model: The framework model path (frameworks supported: ONNX and PyTorch 1.0).
        encodings: The encoding information to be applied to the graph.
        calibration_config: Configuration for calibration process.
        **extra_args: Extra keyword arguments for conversion options.
                      See submodule `qairt.api.converter.converter_config.ConverterConfig` for details.

    Examples:
        >>> import qairt
        >>> fw_model = "path/to/model"
        >>> converted_model = qairt.convert(fw_model)

        For applying encodings -
        >>> import qairt
        >>> fw_model = "path/to/model"
        >>> converted_model = qairt.convert(fw_model, encodings="path/to/encodings")

        For calibration -
        >>> calib_config = CalibrationConfig(dataset=input_data, batch_size = 4, act_precision = 16)
        >>> converted_model = qairt.convert("/path/to/model", calibration_config=calib_config)

    Returns:
        Model: A Model instance that is executable on a QAIRT Runtime.

    Raises:
        ValidationError: If provided extra args are invalid.
        ConversionError: If model conversion fails.
        OptimizationError: If model optimization fails.
        ApplyEncodingsError: If apply encodings fails.
    """

    _convert_logger.info("Starting model conversion...")

    ### STEP 1: Conversion ###
    _convert_logger.debug("Initializing converter module...")
    qairt_converter = QAIRTConverter(logger=_convert_logger)

    # Validating extra args
    _ = ConverterConfig(**extra_args)

    converter_args = _converter_config_arg_adapter(extra_args)

    converter_input_config = ConverterInputConfig(
        input_network=str(model),
        quantization_overrides=encodings,
        **converter_args,
    )

    # TODO: Implement global config to set QAIRT_TMP_DIR environment variable
    tmp_root_dir = os.getenv("QAIRT_TMP_DIR", default=tempfile.gettempdir())
    temp_working_dir = Path(tempfile.mkdtemp(prefix="temp_working_dir_", dir=tmp_root_dir))

    # set converter tmp dir to QAIRT_TMP_DIR
    os.environ["TMPDIR"] = tmp_root_dir

    try:
        if getattr(converter_input_config, "onnx_simplification"):
            with disable_spawn_process_and_exec():
                _convert_logger.debug("Running convert with spawn_process_and_exec disabled.")
                converter_output = qairt_converter.convert(converter_input_config)
        else:
            converter_output = qairt_converter.convert(converter_input_config)

        _convert_logger.debug("Completed model conversion with converter_output: %s", converter_output)
    except Exception as exc:
        raise ConversionError("Model conversion failed: %s", exc)

    ### STEP 2: Optimizations ### (Note: Only default optimizations are run)
    _convert_logger.debug("Initializing optimizer module...")
    qairt_optimizer = QAIRTOptimizer()

    output_dlc_name = Path(model).stem + ".dlc"

    optimizer_input_config = OptimizerInputConfig(
        ir_graph=converter_output.ir_graph,
        framework=converter_output.framework,
        output_dlc=str(temp_working_dir / output_dlc_name),
        dlc_backend_config=converter_output.dlc_backend_config,
    )

    try:
        optimizer_output = qairt_optimizer.optimize(optimizer_input_config)
        _convert_logger.debug("Completed model optimization with optimizer_output: %s", optimizer_output)
    except Exception as exc:
        raise OptimizationError("Model optimization failed: %s", exc)

    dlc_path = optimizer_output.dlc_path

    ### STEP 3: Apply Encodings ###
    if encodings or (calibration_config and calibration_config.dataset):
        if encodings and calibration_config:
            _convert_logger.debug(
                "Encodings and calibration config provided. Performing calibration and quantization"
            )
        elif encodings:
            _convert_logger.debug("Encodings information was provided. Applying encodings")

        if calibration_config:
            data = calibration_config.dataset
            quantizer_args = _calibration_config_arg_adapter(calibration_config)
        else:
            data = None
            quantizer_args = {}

        if data:
            if isinstance(data, (PathLike, str)):
                # Resolve input_list file paths
                data = Path(data).resolve()
                base_path = data.parent

                with data.open("r") as file:
                    file_paths = file.readlines()

                resolved_paths = [base_path / Path(path.strip()) for path in file_paths]

                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                    updated_file_path = Path(temp_file.name)
                    for path in resolved_paths:
                        temp_file.write(str(path).encode() + b"\n")
                data = updated_file_path
            elif isinstance(data, List):
                data, _ = generate_input_list(data, temp_working_dir)
            else:
                try:
                    from torch.utils.data import DataLoader, Dataset

                    from qairt.api.converter.torch_convert_util import _convert_to_list

                    if isinstance(data, (DataLoader, Dataset)):
                        data = _convert_to_list(
                            data,
                            batch_size=cast(CalibrationConfig, calibration_config).batch_size,
                            num_of_samples=cast(CalibrationConfig, calibration_config).num_of_samples,
                        )
                        data, _ = generate_input_list(data, temp_working_dir)
                    else:
                        raise ValueError("Invalid dataset object passed of unknown type.")
                except ImportError:
                    raise ImportError("torch is not installed. Please install torch to use this function.")
        else:
            quantizer_args["float_fallback"] = True

        try:
            quantizer_obj = QAIRTQuantizer()
            quantizer_input_config = QuantizerInputConfig(
                input_dlc=dlc_path,
                output_dlc=str(temp_working_dir / output_dlc_name),
                input_list=data,
                **quantizer_args,
            )

            # Create quantizer module object.
            quantizer_output_config = quantizer_obj.quantize(quantizer_input_config)
            dlc_path = quantizer_output_config.dlc_output
            _convert_logger.debug("Completed quantization, output dlc path = {}".format(dlc_path))
        except Exception as exc:
            raise ApplyEncodingsError("IRQuantization failed: %s", exc)

    # Create Model object
    dlc_module = DlcModule.load(dlc_path, working_dir=temp_working_dir)
    qairt_model = Model(module=dlc_module)
    _convert_logger.info("Convert completed successfully!")
    return qairt_model
