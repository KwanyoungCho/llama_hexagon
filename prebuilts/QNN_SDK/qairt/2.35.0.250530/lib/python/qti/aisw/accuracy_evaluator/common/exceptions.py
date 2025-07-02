# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


class BaseCustomException(Exception):
    """Base class for custom exceptions."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc is None:
            return f'Reason: {self.msg}'
        else:
            return f'Exc: {self.exc} Reason: {self.msg}'


class ConfigurationException(BaseCustomException):
    """Exception encountered in Configuration class."""
    pass


class InferenceEngineException(BaseCustomException):
    """Exceptions encountered in InferenceEngine class."""
    pass


class UnsupportedException(BaseCustomException):
    """Unsupported Feature exception."""
    pass


class ModelTransformationException(BaseCustomException):
    """Handles exceptions encountered in ModelTransformation class."""
    pass


class FileComparatorException(BaseCustomException):
    """Handles exceptions encountered in FileComparator class."""
    pass


class QAIRTConverterException(BaseCustomException):
    """Exceptions encountered in QAIRT Converter."""
    pass


class QAIRTOptimizerException(BaseCustomException):
    """Exceptions encountered in QAIRT Optimizer."""
    pass


class QAIRTQuantizerException(BaseCustomException):
    """Exceptions encountered in QAIRT Quantizer."""
    pass


class QnnContextBinaryGeneratorException(BaseCustomException):
    """Exceptions encountered in QNN ContextBinaryGenerator."""
    pass


class QnnNetRunException(BaseCustomException):
    """Exceptions encountered in QNN Net Runner."""
    pass


# Exceptions raised by the accuracy evaluator pipeline
class QnnSdkRootNotSet(BaseCustomException):
    """ Exception encountered in Evaluator Module when QNN SDK root is not set."""
    pass


class QnnSdkRootNotValid(BaseCustomException):
    """ Exception encountered in Evaluator Module when QNN SDK root is set, but not valid."""
    pass


class EvaluatorRunPipelineFailed(BaseCustomException):
    """ Exception encountered in Evaluator Module when the Run Pipeline has failed."""


class FailedToCreateWorkDir(BaseCustomException):
    """ Exception encountered in Evaluator Module when work directory creation has failed."""
    pass


class FailedToSetupLogger(BaseCustomException):
    """ Exception encountered in Evaluator Module when logger setup has failed."""
    pass


class UserAbort(BaseCustomException):
    """ Exception encountered in QAIRT Evaluator, when run is aborted by the user."""
    pass


class InvalidSetGlobalFormat(BaseCustomException):
    """ Exception encountered in QAIRT Evaluator, when -set_global value format is invalid."""
    pass
