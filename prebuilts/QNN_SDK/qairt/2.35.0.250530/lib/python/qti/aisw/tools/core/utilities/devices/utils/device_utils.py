# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging

from pydantic import BaseModel
from datetime import datetime
from abc import ABCMeta
from typing import Union, List


class NoInitFactory(type):
    def __call__(cls, *args, **kwargs):
        raise TypeError("Cannot instantiate factory directly")


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonABC(ABCMeta, Singleton):
    pass


class DateRange(BaseModel):
    """
    DateRange contains start and end datetime timestamps
    :datetime start: The start time of range
    :datetime end: The end time of range
    """

    start: datetime
    end: datetime


def convert_to_win_path(mixed_path, drive_letter='C'):
    # Replace all forward slashes with backslashes
    normalized_path = mixed_path.replace('/', '\\')

    # Handle root directory conversion
    if normalized_path.startswith('\\'):
        normalized_path = f'{drive_letter}:{normalized_path}'

    return normalized_path


def format_output_as_list(output: Union[str, bytes]) -> List[str]:
    """
    Formats the str output of a command execution into a list of whitespace and
    newline stripped lines.

    Args:
        output (Union[str, bytes]): The output to format.

    Returns:
        List[str]: The formatted output as a list of strings.
    """

    if not output:
        return []
    output = output.decode("utf-8") if isinstance(output, bytes) else output
    return output.splitlines()


def set_logging_level(level: Union[int, str] = "INFO") -> None:
    """
    Set the logging level for the root logger. All subsequent calls
    to loggers will have this level.

    level (Union[int, str]): A valid logging level for the python logging module
    """
    level_name = logging.getLevelName(level)
    file_info = ''
    if level_name == logging.DEBUG:
        file_info = '[%(filename)s:%(lineno)d in function %(funcName)s]'
    logging.basicConfig(format=f'%(asctime)s,%(msecs)d %(levelname)-3s {file_info} %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S', level=level_name)
