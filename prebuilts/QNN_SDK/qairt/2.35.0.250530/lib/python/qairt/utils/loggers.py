# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger


def _initialize_loggers():
    """
    Initializes and registers loggers based on the configuration file.

    This function loads a YAML-based logging configuration and registers
    loggers for each defined logging area using the QAIRTLogger utility.
    It is intended to be called automatically when the module is imported.
    """
    config_file_path = Path(__file__).resolve().parents[1] / "logging_config.yaml"
    config = QAIRTLogger.load_logging_config(config_file_path)

    loggers_config = config.get("loggers", {})
    log_root_dir = os.getenv("QAIRT_TMP_DIR", default=tempfile.gettempdir())
    Path(log_root_dir).mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    log_working_dir = Path(tempfile.mkdtemp(prefix="logs_", dir=log_root_dir))

    for logger_name, logger_settings in loggers_config.items():
        logger = QAIRTLogger.register_area_logger(
            area=LogAreas.register_log_area(logger_name),
            level=logger_settings.get("level", "INFO"),
            formatter_val=logger_settings.get("formatter"),
            handler_list=logger_settings.get("handlers", []),
            log_file_path=log_working_dir,
        )


# Automatically initialize loggers when this module is imported
_initialize_loggers()


def get_logger(name: str = "", level: str = "INFO") -> logging.Logger:
    """
    Returns a logger instance with the specified level and name.

    Args:
        name (str): A name for the python logger instance
        level (Union[int, str]): A valid logging level for the python logging module
    """
    return QAIRTLogger.get_logger(name, level)
