# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import json
import onnx

from onnx import save_model
from onnxruntime_genai.models.builder import create_model
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from . import update_symbolic_shape_with_value, decompose_layernorms, unpack_qkv, decompose_gqa


class GraphBuilder:
    """
    Class to build GenAI model graph from the provided model and configuration.
    """
    def __init__(self, input_model: str, config_path: str,
                 output_dir: str = None, cache_dir: str = None, batch_size: int = 1):
        """
        Constructor
        :param input_model: Path to the input model file.
        :param config_path: Path to the directory containing the model configuration.
        :param output_dir: Path to the output directory for saving the onnx model.
        :param batch_size: batch size for onnx graph.
        :param cache_dir: Path to the cache directory.
        """
        self.input_model = os.path.abspath(input_model)
        self.output_dir = output_dir if output_dir else os.path.dirname(input_model)
        self.config_path = config_path
        self.cache_dir = cache_dir if cache_dir else self.output_dir
        self.batch_size = batch_size
        self.onnx_model_path = None

    def build_genai_model(self):
        """
        Build the GenAI model graph.
        """
        precision = "fp32"
        execution_provider = "cpu"
        create_model(self.config_path, self.input_model, self.output_dir,
                     precision, execution_provider, self.cache_dir)
        self.onnx_model_path = os.path.join(self.output_dir, "model.onnx")
        return self.onnx_model_path

    def update_onnx_graph(self):
        """
        Apply different transformations to the onnx graph such as
        unpack packed (QKV) attention, decompose custom ops such as
        GroupQueryAttention.
        """
        onnx_model = onnx.load(self.onnx_model_path)

        # remove existing model
        os.remove(self.onnx_model_path)
        os.remove(self.onnx_model_path + ".data")

        with open(os.path.join(self.output_dir, "config.json")) as config_file:
            model_config = json.load(config_file)

        decompose_layernorms(onnx_model, model_config)
        unpack_qkv(onnx_model, model_config)
        decompose_gqa(onnx_model, model_config, self.batch_size)
        update_symbolic_shape_with_value(onnx_model, model_config, self.batch_size)
        ortmodel = ONNXModel(onnx_model)
        ortmodel.topological_sort()
        onnx_model = ortmodel.model
        save_model(onnx_model, self.onnx_model_path, save_as_external_data=True, all_tensors_to_one_file=True,
                   location=os.path.basename(self.onnx_model_path + ".data"), size_threshold=0, convert_attribute=False)
