# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import inspect
import json
import os
import pathlib
from os import PathLike
from typing import Optional, Union

import onnx
from typing_extensions import Unpack

from qairt.api.configs.common import AISWBaseModel, BackendType
from qairt.api.transformer.mappings import backend_to_transformation_map, transformation_name_to_config_attr
from qairt.api.transformer.model_transformer_config import ModelTransformerConfig, QuantizationStage
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.defs.mha2sha_transformed_model import (
    Mha2ShaTransformedModel,
)

_transform_logger = get_logger("qairt.transform")


class TransformedModel(AISWBaseModel):
    """
    Represents a transformed ONNX model along with its splits and encodings.
    Attributes:
        model (onnx.ModelProto): The main ONNX model after transformation.
        model_splits (list[onnx.ModelProto]): A list of ONNX model splits resulting from the
            transformation.
        encodings (Dict | None): Optional encodings associated with the model.
    """

    model: Union[str, pathlib.PosixPath, onnx.ModelProto]
    sha_model: Union[Mha2ShaTransformedModel, None]
    model_splits: Union[list[Mha2ShaTransformedModel], list[onnx.ModelProto]]
    encodings: Union[dict, None]

    # Override model_config for this class
    model_config = AISWBaseModel.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "protected_namespaces": ()})

    def export(self, path: str, model_name: str):
        f"""
        Export model artifacts
        Args:
            path(str): Directory where the transformation artifacts are to be saved
            model_name(str): Prefix to model and artifact file names

        For a model that is split into n models, the i'th split will be saved in this format:
        `<model_name>_split_i_of_n.onnx`
        """

        def save_model_splits():
            # Only save the split models since they are SHA models
            n = len(self.model_splits)
            for i, split in enumerate(self.model_splits):
                model_path = os.path.join(path, f"{model_name}_{i + 1}_of_{n}.onnx")
                encodings_path = os.path.join(path, f"{model_name}_{i + 1}_of_{n}.encodings")
                onnx_file_path = pathlib.Path(model_path)
                data_file_path = onnx_file_path.with_suffix(".data")

                relative_path = data_file_path.relative_to(onnx_file_path.parent)
                _transform_logger.debug(f"exporting {model_path}")

                if isinstance(split, Mha2ShaTransformedModel):
                    onnx.save(
                        split.model,
                        model_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=relative_path.as_posix(),
                    )

                    if split.encodings:
                        encodings = copy.deepcopy(split.encodings)
                    elif self.encodings:
                        encodings = copy.deepcopy(self.encodings)
                    else:
                        encodings = None

                    split_model = split.model

                elif isinstance(split, onnx.ModelProto):
                    onnx.save(
                        split,
                        model_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=relative_path.as_posix(),
                    )
                    split_model = split
                    if self.encodings:
                        encodings = copy.deepcopy(self.encodings)
                    else:
                        encodings = None
                else:
                    raise ValueError(f"Invalid model type: {type(split)}")

                if encodings:
                    all_names = {
                        field.name
                        for attr in ["input", "output", "node", "initializer"]
                        for field in getattr(split_model.graph, attr)
                    }
                    for node in split_model.graph.node:
                        all_names.update(node.input)
                        all_names.update(node.output)

                    if encodings["version"] == "1.0.0":
                        encodings["param_encodings"] = [
                            enc for enc in encodings["param_encodings"] if enc["name"] in all_names
                        ]
                        encodings["activation_encodings"] = [
                            enc for enc in encodings["activation_encodings"] if enc["name"] in all_names
                        ]

                    else:
                        encodings["param_encodings"] = {
                            name: enc
                            for name, enc in encodings["param_encodings"].items()
                            if name in all_names
                        }
                        encodings["activation_encodings"] = {
                            name: enc
                            for name, enc in encodings["activation_encodings"].items()
                            if name in all_names
                        }

                    with open(encodings_path, "w") as f:
                        f.write(json.dumps(encodings))

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        elif not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        if self.model_splits and self.sha_model:
            save_model_splits()
            self.sha_model.export(path, model_name, save_model_file=False)

        # Just MHA2SHA
        elif self.sha_model:
            self.sha_model.export(path, model_name)

        # Just splitter
        elif self.model_splits:
            save_model_splits()

        # Just save model (Neither splitter, nor MHA2SHA. Likely pre-quant transformations)
        else:
            model_path = os.path.join(path, f"{model_name}.onnx")
            onnx.save(self.model, model_path, save_as_external_data=True, all_tensors_to_one_file=True)

        if self.encodings:
            encodings_path = os.path.join(path, f"{model_name}.encodings")
            with open(encodings_path, "w") as f:
                f.write(json.dumps(self.encodings))


def _merge_with_defaults(func, user_config: dict, base_arch: str, encodings: Optional[dict] = None):
    """
    Inspect the signature of "func" to get default values for it's parameters
    and merge them with the provided user_config, where user_config overrides defaults

    Args:
        func (Callable): The function whose signature will be inspected.
        user_config (Dict): A dictionary containing user-provided configuration values.
        base_arch (str): The base architecture to be used.
        encodings (Optional[str | PathLike]): Encodings to be used for the
            transformation. Can be an str or a file path. Defaults to None.
    Returns:
        Dict: A dictionary containing the merged configuration values.
    """

    signature = inspect.signature(func)

    kwargs = {}

    for name, param in signature.parameters.items():
        if param.default is not param.empty:
            kwargs[name] = param.default

    if "encodings" in signature.parameters:
        kwargs["encodings"] = encodings

    if "base_arch" in signature.parameters:
        kwargs["base_arch"] = base_arch

    # Update kwargs with user_config attributes if they exist
    if hasattr(user_config, "__dict__"):
        kwargs.update(user_config.__dict__)
    else:
        kwargs.update(user_config)

    return kwargs


def transform(
    model: Union[str, PathLike, onnx.ModelProto],
    backend: BackendType = BackendType.HTP,
    base_arch: str = "",
    quantization_stage: Optional[QuantizationStage] = None,
    encodings: Optional[Union[str, PathLike, dict]] = None,
    **extra_args: Unpack[ModelTransformerConfig],
) -> TransformedModel:
    """
    Transforms an ONNX model based on the specified backend, base architecture, and quantization stage.

    Args:
        model (str | PathLike | onnx.ModelProto): The ONNX model to be transformed, either as a
            file path or a ModelProto object.
        backend (Optional[BackendType]): The backend type for which the model is being transformed.
            Defaults to BackendType.HTP.
        base_arch (Optional[str]): The base architecture to be used for the transformation.
            Defaults to an empty string.
        quantization_stage (Optional[QuantizationStage]): The quantization stage for the
            transformation. Defaults to QuantizationStage.PRE_QUANT.
        encodings (Optional[str | PathLike]): Encodings to be used for the
            transformation. Can be an str or a file path. Defaults to None.

        extra_args:
            extra_args: Extra keyword arguments to pass for transformation. See submodule
                             `qairt.api.transformer.transformer_config.ModelTransformerConfig` for details.

    Examples:
        >>> import qairt
        >>> fw_model = "path/to/model"
        >>> transformed_model = transform(fw_model, backend=BackendType.HTP, quantization_stage=QuantizationStage.PRE_QUANT)

    Returns:
        TransformedModel: An object containing the transformed model, model splits, and encodings.
    """
    # Initialize ModelTransformerConfig using the provided arguments

    transform_config = ModelTransformerConfig.from_dict(extra_args)

    if quantization_stage is None:
        _transform_logger.warning("`quantization_stage` not set. Defaulting to PRE_QUANT")
        quantization_stage = QuantizationStage.PRE_QUANT

    # Determine the transformations to apply
    transformations_to_apply = backend_to_transformation_map.get(backend, {}).get(quantization_stage, [])

    if len(transformations_to_apply) == 0:
        quant_stage = quantization_stage.value.lower()
        _transform_logger.warning(f"No {quant_stage} transformations defined for backend {backend.value}")
        return TransformedModel(model=model, sha_model=None, model_splits=[], encodings=None)

    _transform_logger.debug(f"Transformations to apply: {[fn.__name__ for fn in transformations_to_apply]}")

    if isinstance(model, (str, pathlib.PosixPath)):
        model = onnx.load(model)

    # If encodings is a string(path to encodings file), validate it
    quant_encodings: Union[dict, None]
    if isinstance(encodings, (str, PathLike)):
        with open(encodings, "r") as enc:
            quant_encodings = json.load(enc)
    else:
        quant_encodings = encodings

    model_splits = []
    sha_model = None

    for transformation in transformations_to_apply:
        config_attr = transformation_name_to_config_attr.get(transformation.__name__)
        if not config_attr:
            raise NotImplementedError(f"Transformation {transformation.__name__} not supported")
        _transform_logger.debug(f"Applying transformation: {transformation.__name__}")
        transformation_config = transform_config.get(config_attr, {})

        if not getattr(transformation_config, "skip", False):
            kwargs = _merge_with_defaults(transformation, transformation_config, base_arch, quant_encodings)

            if model_splits:
                _model_splits = []
                for i, _model in enumerate(model_splits):
                    _transform_logger.debug(f"MHA on split {i + 1}")
                    return_value = transformation(model=_model, **kwargs)
                    _transform_logger.debug(f"MHA on split {i + 1} complete")
                    if isinstance(return_value, Mha2ShaTransformedModel):
                        _model_splits.append(return_value)
                    else:
                        _model_splits.append(_model)
                model_splits = _model_splits
            else:
                return_value = transformation(model=model, **kwargs)
                if isinstance(return_value, Mha2ShaTransformedModel):
                    sha_model = return_value
                    model = sha_model.model
                    quant_encodings = sha_model.encodings

                # Only splitter returns a list
                if isinstance(return_value, list) and len(return_value) > 0:
                    if isinstance(return_value[0], str):
                        model_splits = [onnx.load(val) for val in return_value]
                    else:
                        model_splits = return_value

    transformed_model = TransformedModel(
        model=model, sha_model=sha_model, model_splits=model_splits, encodings=quant_encodings
    )

    # Return the transformed model
    return transformed_model
