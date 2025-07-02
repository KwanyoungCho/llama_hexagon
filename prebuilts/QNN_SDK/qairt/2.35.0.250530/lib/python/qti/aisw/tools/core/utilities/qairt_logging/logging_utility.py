# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# Standard library imports
import os
from os import PathLike
from pathlib import Path
import json
import logging
import logging.config
import logging.handlers
import tempfile
from typing import List, Dict, Optional, Union

# Third-party imports
import yaml

# Local imports
from .log_areas import LogAreas

def _add_logging_level(level_name: str, level_num: int, method_name: str = None) -> None:
    """
    Adds a custom log level to the logging module and defines a custom logging method for it.

    Args:
        level_name (str): The name of the custom log level.
        level_num (int): The numeric value of the custom log level.
        method_name (str, optional): The name of the custom logging method.
            Defaults to None, in which case the method name will be the
                lowercase version of level_name.

    Returns:
        None
    """
    if not method_name:
        method_name = level_name.lower()

    # Add the custom level to the logging module
    setattr(logging, level_name, level_num)
    logging.addLevelName(level_num, level_name)

    # Define a custom logging method for the new level
    def log_for_level(self, message, *args, **kwargs):
        if QAIRTLogger.isEnabledFor(level_num):
            QAIRTLogger._log(level_num, message, args, **kwargs)

    # Add the custom method to the Logger class
    setattr(logging.getLoggerClass(), method_name, log_for_level)

# Add CUSTOM TRACE level
_add_logging_level('TRACE', logging.DEBUG - 5)

# Add CUSTOM DISABLED level
_add_logging_level('DISABLED', logging.CRITICAL + 1)

# Set the logger to use the CUSTOM DISABLED level
logging.getLogger(__name__).setLevel('DISABLED')

class QAIRTLogger:
    """
    The QAIRTLogger class -
        This class provides methods to configure and use the logging system.

    Attributes:
        _default_logging_config_file_path (Path): The absolute path to the logging configuration file.
        _config_loaded (bool): Indicates whether the logging configuration has been successfully loaded.
        _registered_area_loggers (list): A list of loggers registered for specific areas or modules.
        _config (Any): The loaded logging configuration object.
        _warning_flag_displayed (bool): Indicates whether a warning message has already been displayed to avoid repetition.
    """

    _default_logging_config_file_path = Path(__file__).parent / "log_config.yaml"
    _config_loaded = False
    _registered_area_loggers = []
    _config = None
    _warning_flag_displayed = False

    @staticmethod
    def load_logging_config(config_file_path: Optional[Union[str, PathLike]] = None) -> dict:
        """
        Loads the logging configuration from a YAML file.

        Args:
            config_file_path (Optional[Union[str, PathLike]]): The path to the logging configuration file.
            If not provided, the default path is used.
        """

        # Use default config file path if none is provided
        if not config_file_path:
            config_file_path = QAIRTLogger._default_logging_config_file_path
        with open(config_file_path, 'r') as config_file:
            QAIRTLogger._config = yaml.safe_load(config_file)
            QAIRTLogger._config_loaded = True
        return QAIRTLogger._config

    @staticmethod
    def _get_config() -> dict:
        """
        Retrieves the current logging configuration.

        Returns:
            dict: The loaded logging configuration dictionary.
        """
        if not QAIRTLogger._config_loaded:
            QAIRTLogger.load_logging_config()
        return QAIRTLogger._config

    @staticmethod
    def get_default_logging_level(name: str = "") -> str:
        """
        Retrieves the default logging level from the environment variable 'QAIRT_LOG_LEVEL'.
        If the environment variable is not set, it defaults to "INFO".
        Returns:
            str: The default logging level in uppercase.
        """
        # Check if the environment variable is set and use it
        env_log_level = os.getenv('QAIRT_LOG_LEVEL', None)
        if env_log_level:
            return env_log_level.upper()
        if name:
            return QAIRTLogger.get_area_logger_level(name)

        return "INFO"

    @staticmethod
    def get_area_logger_level(name: str) -> str:
        """
        Gets the effective logging level for a given area name.

        Args:
            name (str): The area name for which the logging level should be retrieved.

        Returns:
            str: The name of the effective logging level (e.g., 'DEBUG', 'INFO').
        """
        log_area = LogAreas.get_log_area_by_name(name)
        area_logger = logging.getLogger(log_area.value)
        return logging.getLevelName(area_logger.getEffectiveLevel()).upper()


    @staticmethod
    def set_area_logger_level(area: LogAreas, level: int | str) -> None:
        """
        Sets a specific logging level for a given area.

        Args:
            area (LogAreas): The area for which the logging level should be set.
            level (int | str): The desired logging level.
                Can be either an integer representing the logging level constant
                (e.g., logging.DEBUG) or a string representing the name of the
                logging level ('DEBUG', 'INFO', etc.).

        Raises:
            None
        """
        area_logger = logging.getLogger(area.value)
        area_logger.setLevel(level)

    @staticmethod
    def set_level_for_all_areas(level: int | str) -> None:
        """
        Sets the logging level for all areas.

        Args:
            level (int | str): The desired logging level.
                Can be either an integer representing the logging level constant
                (e.g., logging.DEBUG) or a string representing the name of the
                logging level ('DEBUG', 'INFO', etc.).

        Raises:
            None
        """
        for area in LogAreas:
            QAIRTLogger.set_area_logger_level(area, level)

    def list_log_levels() -> Dict:
        """
        Retrieves a list of available logging levels.

        Returns:
            Dict: A dictionary containing the available logging levels.
        """
        levels = {level: name for name, level in logging._levelToName.items()}
        sorted_levels = dict(sorted(levels.items(), key=lambda item: item[1]))

        return sorted_levels

    def get_logger(name: str = "", level: str = 'INFO', formatter_val: str = None,
            handler_list: List[str] = None, log_file_name: str = None,
            log_file_path: str | PathLike = None,
            parent_logger: logging.Logger = None) -> logging.Logger:
        """
        Returns a logger instance with the specified level and name.

        Args:
            name (str): A name for the python logger instance
            level (int | str, optional): The logging level for the logger.
                Defaults to 'INFO'.
            formatter_val (str, optional): The name of the formatter to use.
                If provided, this overrides any existing formatter configuration. Defaults to None.
            handler_list (List[str], optional): A list of handler names to attach to the logger.
                Defaults to None.
            log_file_name (str, optional): The base file name for the log files.
                If provided, it will be used to create unique filenames based on the current timestamp.
                Defaults to None.
            log_file_path (str | PathLike, optional): The path where the log files will be stored.
                If provided, it will be used to construct the full file paths. Defaults to None.
            parent_logger (logging.Logger, optional): The parent logger that will be used to
                propagate messages. Defaults to None.
        """

        model_log_area = LogAreas.register_log_area(name if name else __name__)
        return QAIRTLogger.register_area_logger(model_log_area, level = level, parent_logger = parent_logger)

    @staticmethod
    def _resolve_log_level(level: int | str = 'INFO') -> int:
        """
        Resolves and returns the appropriate logging level as an integer.

        This method determines the logging level based on the provided `level` argument
        and the `QAIRT_LOG_LEVEL` environment variable. If the special string 'VERBOSE'
        is encountered (either as an argument or in the environment variable), it is
        mapped to the standard 'DEBUG' level, with a warning printed once.

        Args:
            level (int | str): The desired log level, either as a string (e.g., 'INFO',
                'DEBUG', 'VERBOSE') or an integer. Defaults to 'INFO'.

        Returns:
            int: The resolved logging level as an integer compatible with the `logging` module.
        """
        # Check if the environment variable is set and use it
        env_log_level = os.getenv('QAIRT_LOG_LEVEL', None)

        # Map non-standard 'VERBOSE' log level to 'DEBUG'
        if (isinstance(level, str) and level.upper() == "VERBOSE") or \
            (env_log_level and env_log_level.upper() == "VERBOSE"):
            if (QAIRTLogger._warning_flag_displayed == False):
                print("WARNING!: 'VERBOSE' is not a standard log level. Using 'DEBUG' instead.\n")
                QAIRTLogger._warning_flag_displayed = True
            level = "DEBUG"

        if env_log_level:
            level = getattr(logging, env_log_level.upper(), level)

        if isinstance(level, str):
            level = getattr(logging, level.upper())

        return level

    @staticmethod
    def register_area_logger(area: LogAreas, level: int | str = 'INFO', formatter_val: str = None,
            handler_list: List[str] = None, log_file_name: str = None, log_file_path: str | PathLike = None, parent_logger: logging.Logger = None) -> logging.Logger:
        """
        Registers a new logger for a specific area.

        Args:
            area (LogAreas): The area for which the logger will be created.
            level (int | str, optional): The logging level for the logger.
                Defaults to 'INFO'.
            formatter_val (str, optional): The name of the formatter to use.
                If provided, this overrides any existing formatter configuration. Defaults to None.
            handler_list (List[str], optional): A list of handler names to attach to the logger.
                Defaults to None.
            log_file_name (str, optional): The base file name for the log files.
                If provided, it will be used to create unique filenames based on the current timestamp.
                Defaults to None.
            log_file_path (str | PathLike, optional): The path where the log files will be stored.
                If provided, it will be used to construct the full file paths. Defaults to None.
            parent_logger (logging.Logger, optional): The parent logger that will be used to
                propagate messages. Defaults to None.

        Raises:
            None

        Returns:
            logging.Logger: The newly created logger instance.
        """

        try:
            level = QAIRTLogger._resolve_log_level(level)
            area_logger = None
            file_handler_flag = False
            if parent_logger:
                # Use the provided logger but define it under the sub-module's log area name
                area_logger = logging.getLogger(f"{area.value}")
                area_logger.handlers = parent_logger.handlers
                area_logger.setLevel(parent_logger.level)
                area_logger.propagate = False
            elif (area in QAIRTLogger._registered_area_loggers):
                '''
                If an area logger is already defined by a module, then the module will be returned.
                This mimics the behavior of the python logging module.

                If a sub-module needs to redefine the logger, it should create its own logger.
                '''
                area_logger = logging.getLogger(area.value)
                return area_logger
            else:
                # Create the logger
                _ = QAIRTLogger._get_config()
                area_logger = logging.getLogger(area.value)
                QAIRTLogger.set_area_logger_level(area, level)
                area_logger.propagate = False
                handlers = QAIRTLogger._config.get('handlers', {})
                # Attach the specified handlers
                if handler_list:
                    for handler_name in handler_list:
                        formatters = QAIRTLogger._config.get('formatters', {})
                        if (formatter_val is not None):
                            formatter_config = formatters.get(formatter_val, {})
                        else:
                            if handler_name in handlers:
                                formatter_name = handlers[handler_name].get('formatter', {})
                                formatter_config = formatters.get(formatter_name, {})
                            else:
                                print(f"ERROR setting up logger: Handler name {handler_name} not "
                                    "present in handlers section of log_config.yaml")
                                return None
                        formatter_format = formatter_config.get('format')
                        formatter_datefmt = formatter_config.get('datefmt')
                        formatter = logging.Formatter(fmt = formatter_format, datefmt = formatter_datefmt)

                        if handler_name in handlers:
                            handler_config = handlers.get(handler_name, {})
                            handler_level = handler_config.get('level', 'INFO')
                        else:
                            print(f"ERROR setting up logger: Handler name {handler_name} not "
                                "present in handlers section of log_config.yaml")
                            return None

                        if handler_config.get('class') == 'logging.StreamHandler':
                            handler = logging.StreamHandler()
                        elif handler_config.get('class') == 'logging.handlers.RotatingFileHandler':
                            file_handler_flag = True

                            # Determine the log file path
                            if log_file_path is None:
                                tmp_root_dir = Path(tempfile.gettempdir())
                                # Create a unique directory for this run's logs
                                log_file_path = Path(tempfile.mkdtemp(prefix="logs_", dir=tmp_root_dir))

                            else:
                                log_file_path = Path(log_file_path)
                                log_file_path.mkdir(parents=True, exist_ok=True)

                            filename = handler_config.get('filename', 'user_info.log')

                            if log_file_name:
                                filename = log_file_path / f"{log_file_name}_{filename}"
                            else:
                                filename = log_file_path / filename

                            handler = logging.handlers.RotatingFileHandler(
                                        filename,
                                        maxBytes = handler_config.get('maxBytes', 10485760),
                                        backupCount = handler_config.get('backupCount', 40),
                                        encoding = handler_config.get('encoding', 'utf8')
                                        )
                        else:
                            # Add other custom handlers here if needed
                            continue
                        handler.setLevel(handler_level)
                        handler.setFormatter(formatter)
                        area_logger.addHandler(handler)
                else:
                    # Default handler if no handler_list is provided
                    handler = logging.StreamHandler()
                    handler.setLevel(level)
                    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    handler.setFormatter(formatter)
                    area_logger.addHandler(handler)

        except Exception as e:
            print(f"ERROR setting up logger: {e}")
            return None

        QAIRTLogger._registered_area_loggers.append(area)
        if file_handler_flag:
            print(f"File logs for {area.value} will be stored at: {log_file_path}\n")
        return area_logger