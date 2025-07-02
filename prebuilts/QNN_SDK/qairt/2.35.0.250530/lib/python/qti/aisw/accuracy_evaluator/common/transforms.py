# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import errno
import os

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_framework import (
    OnnxModelHelper,
    OnnxTransformModel,
)


class ModelHelper:
    """A helper class for model-related operations."""

    @classmethod
    def clean_tf(cls, model_path: str, out_path: str) -> str:
        """Clean the given tensorflow model. Includes node name sanitization by
        removing special characters.

        Args:
            model_path: Path to the tensorflow model
            out_path: Path to output directory where cleaned model is to be saved
        Returns:
            str: Path to the cleaned tensorflow model
        Raises:
            FileNotFoundError: if model_path does not exist
        """
        tf = Helper.safe_import_package("tensorflow")
        qacc_file_logger.info('Preparing model for qairt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model path {model_path} does not exist')

        # load tf model
        with tf.io.gfile.GFile(model_path, "rb") as f:
            M = tf.compat.v1.GraphDef()
            M.ParseFromString(f.read())

        for i in range(len(M.node)):
            # Remove colocate information from GraphDef
            if "_class" in M.node[i].attr:
                del M.node[i].attr["_class"]
            # remove special characters from node names
            M.node[i].name = Helper.sanitize_node_names(M.node[i].name)
            for j in range(len(M.node[i].input)):
                # remove special characters except ':' from node inputs
                # Otherwise graph connection breaks in case of node with multiple outputs
                M.node[i].input[j] = Helper.sanitize_node_names(M.node[i].input[j])

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # save the cleaned model
        tf.io.write_graph(graph_or_graph_def=M, logdir=out_path, name='cleanmodel.pb',
                          as_text=False)
        qacc_file_logger.info('Prepared model for qairt')
        cleanmodel_path = os.path.join(out_path, 'cleanmodel.pb')
        return cleanmodel_path

    @classmethod
    def clean_onnx(cls, model_path: str, out_path: str, symbols: dict = {},
                   replace_special_chars: bool = True, check_model: bool = True,
                   simplify_model: bool = True) -> str:
        """Clean the given Onnx model. Also, validate and do onnx simplification, if specified.
        Cleaning also includes removing special characters from graph and node inputs and outputs,
        and replacing symbols with given values.

        Args:
            model_path: Path to Onnx model
            out_path: Path at which cleaned model is to be saved
            symbols: Dictionary of symbols that need to be replaced
            replace_special_chars: Boolean to check if special characters
                in model need to be replaced
            check_model: Boolean to check if model needs to be validated
            simplify_model: Boolean to check if model needs to be simplified
        Returns:
            str: Path to cleaned model
        """
        framework_manager = FrameworkManager()
        model = framework_manager.load(input_model=model_path)
        if simplify_model:
            model = OnnxTransformModel.optimize_by_simplifier(model_path)
        if check_model:
            _ = framework_manager.validate(input_model=model)

        if replace_special_chars:
            # remove special characters in node inputs and outputs.
            model = OnnxTransformModel.transform_graph_nodes(model)
            # remove special characters in graph inputs and outputs, value-info and initializers.
            model = OnnxTransformModel.transform_graph_inputs_and_outputs(model)

        # remove symbols with provided values (default 1)
        model = OnnxTransformModel.transform_dynamic_shapes(model=model, symbols=symbols)

        OnnxModelHelper.save_model(model, out_path)
        return out_path

    @classmethod
    def clean_model_for_qairt(cls, model_path: str, out_dir: str, symbols: dict = {},
                              replace_special_chars: bool = True, check_model: bool = True,
                              simplify_model: bool = True) -> str:
        """Clean up the model for qairt. In case of Onnx models, special characters are removed from
        graph and node inputs and outputs, and symbols replaced with given values.

        Args:
            model_path: Path to the model
            out_dir: Path to output directory where cleaned model is to be saved
            symbols: Dictionary of symbols that need to be replaced
            replace_special_chars: Boolean to check if special characters
                             in model need to be replaced
            check_model: Boolean to check if model needs to be validated
            simplify_model: Boolean to check if model needs to be simplified
        Returns:
            str: Path to cleaned model
        """
        mtype = FrameworkManager.infer_framework_type(model_path)
        # Tensorflow utils not supported in Framework utilities
        if mtype == "tensorflow":
            return ModelHelper.clean_tf(model_path, out_dir)
        if mtype == "onnx":
            if simplify_model:
                out_path = os.path.join(out_dir, qcc.SIMPLIFIED_CLEANED_MODEL_ONNX)
            else:
                out_path = os.path.join(out_dir, qcc.CLEANED_MODEL_ONNX)
            return cls.clean_onnx(model_path=model_path, out_path=out_path, symbols=symbols,
                                  replace_special_chars=replace_special_chars,
                                  check_model=check_model, simplify_model=simplify_model)
        if mtype == "pytorch":
            out_path = os.path.join(out_dir, qcc.CLEANED_MODEL_PT)
            framework_manager = FrameworkManager()
            model = framework_manager.load(input_model=model_path)
            torch = Helper.safe_import_package("torch")
            torch.jit.save(model, out_path)
            return out_path
        if mtype == "tflite":
            return model_path
        raise ce.UnsupportedException(f'Unsupported model type: {mtype}')

    @classmethod
    def get_tf_batch_size(cls, model_path: str, input_name: str) -> int:
        """Get the batchsize for the given tensorflow model.

        Args:
            model_path: Path to the tensorflow model
            input_name: Name of the given input
        Returns:
            int: Batchsize of the model. If not found, default value 1 is returned
        Raises:
            FileNotFoundError, if model_path is not found
        """
        try:
            tf = Helper.safe_import_package("tensorflow")

            with tf.io.gfile.GFile(model_path, "rb") as f:
                model = tf.compat.v1.GraphDef()
                model.ParseFromString(f.read())
            modified_input_name = Helper.sanitize_node_names(input_name)
            for node in model.node:
                if node.name == modified_input_name:
                    out_shape = node.attr.get('_output_shapes', None)
                    if out_shape:
                        batchsize = out_shape.list.shape[0].dim[0].size
                    else:
                        qacc_file_logger.warning(
                            f'Batchsize not found for model: {model_path}. Setting model batchsize to 1'
                        )
                        batchsize = 1
                    return batchsize
                else:
                    raise ce.ConfigurationException(
                        f'Incorrect model input name provided in config. Given: {input_name}, Using: {modified_input_name}, Actual: {node.name}'
                    )
        except FileNotFoundError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

    @classmethod
    def get_model_batch_size(cls, model_path: str, input_name: str) -> int:
        """Get the batchsize for the given model.

        Args:
            model_path: Path to the model
            input_name: Name of the given input
        Returns:
            int: Batchsize of the model. If not found, default value 1 is returned
        """
        mtype = FrameworkManager.infer_framework_type(model_path)
        if mtype == "tensorflow":
            return ModelHelper.get_tf_batch_size(model_path, input_name)
        framework_manager = FrameworkManager()
        model = framework_manager.load(input_model=model_path)
        return framework_manager.get_model_batch_size(model, input_name)
