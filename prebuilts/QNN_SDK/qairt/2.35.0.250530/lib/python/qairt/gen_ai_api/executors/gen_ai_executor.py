# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Optional

from qairt.api.configs.common import AISWBaseModel


class GenerationMetrics(AISWBaseModel):
    init_time: Optional[int] = None
    """Time to load the model, before prompt processing begins"""
    prompt_processing_time: Optional[int] = None
    """Microseconds between init and token generation, while processing the prompt."""
    prompt_processing_rate: Optional[float] = None
    """tokens per second.  Tokens in the prompt divided by prompt processing time"""
    token_generation_time: Optional[int] = None
    """Microseconds prompt processing and final response"""
    token_generation_rate: Optional[float] = None
    """tokens per second.  Tokens in the response divided by token generation time"""

    def __str__(self):
        return (
            f"{'-' * 20} {'Metrics'}{'-' * 20} \n"
            f"Timing (microseconds): \n\n"
            f"  Init = {self.init_time or 0.0} us \n"
            f"  Prompt Processing Time = {self.prompt_processing_time or 0.0} us \n"
            f"  Token Generation Time = {self.token_generation_time or 0.0} us \n\n"
            f"Tokens per second (toks/sec): \n\n"
            f"  Prompt Processing Rate = {self.prompt_processing_rate or 0} toks/sec \n"
            f"  Token Generation Rate = {self.token_generation_rate or 0} toks/sec \n"
        )


class GenerationExecutionResult(AISWBaseModel):
    output: str = ""
    """Raw output from text generation"""
    error: str = ""
    """Raw error response from text generation (empty on success)"""
    generated_text: str = ""
    """parsed response - the generated response (minus metrics)"""
    metrics: Optional[GenerationMetrics] = None
    """parsed metrics from the response."""


class GenAIExecutor(ABC):
    @abstractmethod
    def prepare_environment(self) -> "GenAIExecutor":
        """
        Prepares artifacts for execution on target
        """
        pass

    @abstractmethod
    def clean_environment(self) -> "GenAIExecutor":
        """
        Removes artifacts from target environment
        """
        pass
