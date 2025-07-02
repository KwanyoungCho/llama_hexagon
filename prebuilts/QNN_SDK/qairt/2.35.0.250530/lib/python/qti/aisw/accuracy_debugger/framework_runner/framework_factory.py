# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from logging import Logger
from pathlib import Path

from qti.aisw.accuracy_debugger.framework_runner.frameworks.onnx_framework import (
    CustomOnnxFramework,
)
from qti.aisw.accuracy_debugger.utils.constants import supported_frameworks
from qti.aisw.accuracy_debugger.utils.exceptions import FrameworkError
from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager
from qti.aisw.tools.core.utilities.framework.frameworks.base_framework import BaseFramework
from qti.aisw.tools.core.utilities.framework.utils.constants import OnnxFrameworkInfo


def get_framework_instance(framework: str, logger: Logger) -> BaseFramework:
    """Returns the framework class based on the framework provided

    Args:
        framework(str): Name of the specified framework
        logger (Logger): A python Logger instance

    Returns:
        BaseFramework: class of the given framework

    Raises:
        FrameworkError if the framework is not supported
    """
    if framework == OnnxFrameworkInfo.name:
        return CustomOnnxFramework(logger)
    raise FrameworkError(f"The given framework {framework} is not supported.")


def get_framework_type(input_model: str | Path) -> str:
    """Infer the framework type based on the input_model

    Args:
        input_model: Path to the input model
    Returns:
        str: framework type
    """
    try:
        framework_type = FrameworkManager.infer_framework_type(str(input_model))
    except Exception:
        raise ValueError(
            f"Invalid source model. Supported framework types are {supported_frameworks}"
        )

    if framework_type not in supported_frameworks:
        raise ValueError(
            f"Framework type {framework_type} is not supported."
            f"Supported framework types are {supported_frameworks}"
        )
    return framework_type
