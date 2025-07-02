# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


class CompilationError(RuntimeError):
    """Thrown when an error occurs during compilation"""

    pass


class ExecutionError(RuntimeError):
    """Thrown when an error occurs that causes inference to terminate
    on a QAIRT Backend"""

    pass


class InvalidCacheError(Exception):
    """Exception class for specific errors pertaining to malformed caches"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnknownAssetError(TypeError):
    """Thrown when an asset is not known to the system"""

    pass


class LoadAssetError(IOError):
    """Thrown when an asset could not be loaded"""

    pass


class SaveAssetError(IOError):
    """Thrown when an asset could not be saved"""

    pass


class HookHandleError(Exception):
    """An error that occurs when a hook cannot be added or removed"""

    pass


class ConversionError(Exception):
    """Thrown when a model conversion fails"""

    pass


class OptimizationError(Exception):
    """Thrown when a model optimization fails"""

    pass


class ApplyEncodingsError(Exception):
    """Thrown when apply encodings fails"""

    pass
