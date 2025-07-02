# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


class CustomException(Exception):
    """Base class for custom exceptions."""

    def __init__(self, msg, exception=None):
        self.exception = exception
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exception is None:
            return f"{self.msg}"
        return f"Exception: {self.exception} Reason: {self.msg}"


class ParameterError(CustomException):
    """Generic error for any parameter errors."""


class ConversionFailure(CustomException):
    """Exceptions encountered in QAIRT Conversion"""


class OptimizationFailure(CustomException):
    """Exceptions encountered in QAIRT Optimization"""


class QuantizationFailure(CustomException):
    """Exceptions encountered in QAIRT Quantization"""


class GenerateBinaryFailure(CustomException):
    """Exceptions encountered in QAIRT Context Binary Generation"""


class ExecutionFailure(CustomException):
    """Exceptions encountered in QAIRT Net Runner"""


class InferenceEngineFailure(CustomException):
    """Exceptions encountered in QAIRT Net Runner"""


class QairtEncodingsConverterFailure(CustomException):
    """Exceptions encountered during QAIRT Encodings Converter"""


class FrameworkError(CustomException):
    """Defines a generic error for any framework errors."""


class UnsupportedError(CustomException):
    """Exception raised for unsupported actions or options."""


class EncodingsMismatchError(CustomException):
    """Defines a encodings mismatch error in compare encodings."""


class VerificationError(CustomException):
    """Defines error encountered during verification"""


class ModelTransformationException(CustomException):
    """Exception when failing to transform model"""
