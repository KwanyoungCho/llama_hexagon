# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging
import os
import pathlib
from enum import Enum
from typing import Optional

from qairt.api.configs.common import BackendType
from qairt.gen_ai_api.builders.gen_ai_builder import GenAIBuilder
from qairt.gen_ai_api.builders.gen_ai_builder_cpu import GenAIBuilderCPU
from qairt.gen_ai_api.builders.gen_ai_builder_htp import GenAIBuilderHTP
from qairt.gen_ai_api.builders.gen_ai_utils import load_pretrained_config
from qairt.gen_ai_api.builders.llama2_builder_htp import Llama2BuilderHTP
from qairt.gen_ai_api.builders.llama3_builder_htp import Llama3BuilderHTP

logger = logging.getLogger(__name__)


_SUPPORTED_BACKENDS = [BackendType.CPU, BackendType.HTP]


class SupportedLLMs(Enum):
    """enumeration of preconfigured model architectures."""

    LLAMA = "LlamaForCausalLM"
    BAICHUAN = "BaiChuanForCausalLM"


class GenAIBuilderFactory:
    """
    Factory class to create gen_ai_builder instances
    """

    @classmethod
    def create(
        cls,
        pretrained_model_path: str | os.PathLike,
        backend_type: str | BackendType = BackendType.HTP,
        working_directory: str | os.PathLike = pathlib.Path.cwd(),
        *,
        cache_root: Optional[str | os.PathLike] = None,
    ) -> GenAIBuilder:
        """
        Creates a GenAIBuilder instance based on the provided pretrained model path and backend type.

        Args:
            pretrained_model_path (str): The path to the pretrained model.  The pretrained model path should be a directory containing
                the model files, tokenizer.json and config.json
            backend_type (BackendType): The type of backend to use. Defaults to BackendType.HTP.
            working_directory (Path): The working directory for the builder. Defaults to the current working directory.
            cache_root (Path, optional): The root directory for caching, if desired.

        Returns:
            GenAIBuilder: The created GenAIBuilder instance.

        Raises:
            ValueError: If the pretrained model path does not exist or if required files are missing.

        Notes:
            If the pretrained model path is a file, the directory containing it will be used to locate additional required artifacts.
            The builder will attempt to identify the model and provide a pre-configured instance based on the
            architecture. If HTP is requested and the model architecture is not recognized, a default GenAIBuilderHTP instance will be
            returned, with a warning.
        """

        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"Pretrained model path '{pretrained_model_path}' does not exist")

        if backend_type not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Backend type '{backend_type}' is not supported")

        if backend_type == BackendType.CPU:
            return GenAIBuilderCPU.from_pretrained(pretrained_model_path, working_directory)

        # if pretrained_model_path is a file, get the path to the directory containing it
        pretrained_model_path_dir = pretrained_model_path
        if os.path.isfile(pretrained_model_path):
            pretrained_model_path_dir = os.path.dirname(pretrained_model_path)

        config = load_pretrained_config(pretrained_model_path_dir)

        if SupportedLLMs.LLAMA.value in config.architectures:
            n_vocab = getattr(config, "n_vocab", getattr(config, "vocab_size"))
            if n_vocab == 32000:
                return Llama2BuilderHTP.from_pretrained(os.fspath(pretrained_model_path), cache_root)
            elif n_vocab == 128256:
                return Llama3BuilderHTP.from_pretrained(os.fspath(pretrained_model_path), cache_root)
            else:
                logger.warning(
                    f"Unrecognized llama variant, n_vocab: {n_vocab}. "
                    f"Builder will likely need additional configuration."
                )

        logger.warning(
            "Architecture is unknown or unsupported; Returning default. "
            "This builder may work but will probably require additional configuration."
        )
        return GenAIBuilderHTP.from_pretrained(os.fspath(pretrained_model_path), cache_root)
