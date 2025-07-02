# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import json
import logging

from typing import Optional
from shutil import copyfile
from transformers.integrations.ggml import convert_gguf_tokenizer
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

from . import (GGUFParser, GraphBuilder, update_encodings,
               MODEL_TYPE_TO_ARCH, MODEL_TYPE_TO_TOKENIZER)

logger = logging.getLogger("LLM Builder")


class LLMBuilder:
    """
    Class to build GenAI model from GGUF file.
    """
    def __init__(self, input_model: str, config_file: Optional[str] = None,
                 output_dir: Optional[str] = None, batch_size: Optional[int] = None):
        """
        Constructor
        :param input_model: Path to GGUF file.
        :param config_file: Path to file containing configuration for building GenAI model.
                            (the config.json file generated when saving the huggingface model)
        :param: batch_size: batch size for the model.
        :param output_dir: Path to save GenAI model and encodings.
        """
        self.input_model = os.path.abspath(input_model)
        self.config_file = config_file
        if not output_dir:
            self.output_dir = os.path.dirname(self.input_model)
        else:
            self.output_dir, _ = os.path.split(os.path.abspath(output_dir))
        self.batch_size = batch_size if batch_size else 1
        self.num_layers = None
        self.model_type = None
        self._generate_config_from_gguf()

    def _generate_config_from_gguf(self):
        """
        Generates the configuration files (model and tokenizer) from the GGUF file.
        """
        gguf_data = load_gguf_checkpoint(self.input_model)
        self.num_layers = gguf_data["config"]["num_hidden_layers"]
        self.model_type = gguf_data["config"]["model_type"]

        # If config file is not provided, generate model config from input gguf file
        if not self.config_file:
            config_dict = dict()
            config_dict.update(gguf_data["config"])
            config_dict["architectures"] = [MODEL_TYPE_TO_ARCH[self.model_type]]
            config_dict.pop("_model_name_or_path", None)

            if "vocab_size" not in config_dict:
                config_dict["vocab_size"] = len(gguf_data["tokenizer"]["tokens"])

            with open(os.path.join(self.output_dir, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=4, sort_keys=True)

        # Else, copy config file to output path
        else:
            copyfile(self.config_file, os.path.join(self.output_dir, "config.json"))

        # create generation config file
        gen_config_dict = dict()
        gen_config_dict.update(gguf_data["tokenizer_config"])
        gen_config_dict["_from_model_config"] = True
        gen_config_dict.pop("model_type", None)

        with open(os.path.join(self.output_dir, "generation_config.json"), "w") as f:
            json.dump(gen_config_dict, f, indent=4, sort_keys=True)

        # create tokenizer config file
        tokenizer, additional_options = convert_gguf_tokenizer(self.model_type, gguf_data["tokenizer"])
        fast_tokenizer = MODEL_TYPE_TO_TOKENIZER[self.model_type](tokenizer_object=tokenizer)
        fast_tokenizer.save_pretrained(self.output_dir)

    def _generate_args_for_inputs(self):
        """
        Generate Arguments (layouts, inputs whose datatype is to be preserved) for model inputs.
        """
        inputs_to_preserve = ["datatype", "input_ids", "attention_mask"]
        input_layouts = [["input_ids", "NONTRIVIAL"], ["attention_mask", "NONTRIVIAL"],
                         ["position_ids_cos", "NONTRIVIAL"], ["position_ids_sin", "NONTRIVIAL"]]

        for layer_idx in range(self.num_layers):
            input_layouts.append([f"past_key_{layer_idx}_in", "NONTRIVIAL"])
            input_layouts.append([f"past_value_{layer_idx}_in", "NONTRIVIAL"])
        return input_layouts, inputs_to_preserve

    def build_from_gguf(self):
        """
        Build the GenAI model from the GGUF file.
        """
        filename_prefix = "model_parameters"
        gguf_parser = GGUFParser(self.input_model, filename_prefix, self.output_dir)
        gguf_parser.parse_gguf()
        dequantized_weights_path = gguf_parser.export_dequantized_weights()

        graph_builder = GraphBuilder(dequantized_weights_path, config_path=self.output_dir, output_dir=self.output_dir,
                                     cache_dir=self.output_dir, batch_size=self.batch_size)

        onnx_model_path = graph_builder.build_genai_model()
        graph_builder.update_onnx_graph()

        gguf_parser._generate_param_encodings()
        update_encodings(gguf_parser.param_encodings, self.num_layers, self.model_type)
        encodings_path = gguf_parser.export_encodings()
        os.remove(dequantized_weights_path)
        logger.info("ONNX model saved at: {}".format(onnx_model_path))
        logger.info("Quantization Overrides saved at: {}".format(encodings_path))

        input_layouts, inputs_to_preserve = self._generate_args_for_inputs()
        return onnx_model_path, encodings_path, input_layouts, inputs_to_preserve
