# -*- mode: python -*-
# -----------------------------------------------------------------------------
#
# Qualcomm Technologies, Inc. Proprietary
# (c) 2022-24 Qualcomm Technologies, Inc. All rights reserved.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
# -----------------------------------------------------------------------------

r"""Runner module for converting MHAs to SHAs.

This module is the high level entry point into the conversion of a model with MHAs into SHAs. From here, the users
provided flags into the program are parsed and used to load the model, convert the model, propagate encodings, and save
the new model/encodings. View the README.md for more information.

Basic usage
-----------

>>> converter = MHA2SHAConverter(
        model_name="llama2",
        sha_export_path="./exports",
        model_or_path="./llama2.onnx",
        **kwargs  # Where kwargs are flags for the converter to parse
    )
>>> converter.convert()
"""

import copy
import datetime
import json
import os
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Text, Tuple, Union

import numpy as np
import onnx
from packaging import version

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.defs.mha2sha_transformed_model import (
    LoraAdapter,
    Mha2ShaTransformedModel,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils import fold_constants
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.logger import (
    log_assert,
)

from .defs.print_colors import Colors
from .defs.tensor_info import TensorInfo, get_input_info, get_output_info
from .optimizer import MHA2SHAOptimizer
from .prequant_adaption import PreQuantAdaption
from .transformations.o_proj_optimizer import OProjOptimzier
from .utils import onnx as ou
from .utils.attention_patterns import attention_patterns
from .utils.auto_mha_finder import auto_attention_finder
from .utils.lora_adaptor_converter import (
    LoraAdaptor,
    parse_lora_adaptor_list,
)

from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger
mha2sha_log_area = LogAreas.register_log_area("MHA2SHA")

_ONNX_MIN_VERSION_NEEDED = "1.14.1"


class MHA2SHAConverter:
    """Converts models with MHA to SHA.

    Runner class for converting models from MHA to SHA. Additional conversions can be added based on the flags passed in.

    Attributes:
        model:
            ModelProto object.
        model_path:
            Model path form the model loader.
        model_name:
            Name of model to be export, assign a name base on model path if not provided.
        sha_export_path:
            Path for exporting the converted model.
        exported_model_encoding_path:
            Path of the encodings from the original model.
        prepared_model:
            Model name to generat auto generate one if nor peovided.
        handle_rope_ops:
            Flag for whether this model has RoPE operations to be handled.
        handle_past_key_value:
            Flag for whether this model has past key values to be handled.
        replace_linear_with_conv:
            Flag for replacing Linear operations with Convs.
        disable_auto_attn_finder:
            Flag to turn off the auto attention module finder feature.
        position_ids:
            Flag for if position ids are in the model.
        mha_conv:
            Flag for if mha is in Conv.
        nchw_aligned:
            Flag for if mha input is aligned to nchw format.
        lora_model:
            Flag for if mha has lora adaptor.
        lora_adaptor_list:
            Path to list of lora adaptor safetensor and encodings.
        create_input_lists:
            Flag for if to create input_list.
        no_verification:
            Flag to skip verification step.
        log_level:
            log level: 'warn', 'verbose', 'info', 'error', 'fatal.
        strict:
            Whether to strictly enforce golden RoPE pattern.
        build_ar:
            AR value to produce model of.
    """

    def __init__(
        self,
        model: onnx.ModelProto,
        base_arch: str = "",
        is_llm_model: bool = True,
        is_gqa_model: bool = False,
        is_lora_model: bool = False,
        lora_adapters: Optional[Union[str, list[dict]]] = None,
        lora_tensor_names: Optional[Union[str, list[str]]] = None,
        lora_alpha_from_input: bool = False,
        is_prepared_model: bool = False,
        optimize_o_proj: bool = True,
        handle_alibi: bool = False,
        handle_past_key_value: bool = False,
        handle_rope_ops: bool = False,
        strict_rope_pattern: bool = True,
        build_ar: Optional[int] = None,
        disable_auto_attn_finder: bool = False,
        skip_verification: bool = False,
        is_linear_to_conv_converted: bool = True,  # Linear2Conv is mostly assumed to be run before mha2sha
        is_nchw_aligned: bool = True,
        encodings: Optional[dict] = None,
        log_level: str = "info",
    ):
        """Creates a converter instance"""

        self._start_time = time.time()

        self.logger = QAIRTLogger.register_area_logger(
                mha2sha_log_area, level=log_level, formatter_val="extended", handler_list=["dev_console"]
            )

        self._verify_can_run()

        self._model = model
        ou.assign_names_to_empty_nodes(self._model)
        model = fold_constants(self._model)
        self._model.CopyFrom(model)

        self._base_arch = base_arch

        self._llm_model = is_llm_model
        self._gqa_model = is_gqa_model

        self._build_ar = build_ar

        self._lora_model = is_lora_model
        if lora_adapters:
            if isinstance(lora_adapters, str):
                self.lora_adaptor_list = parse_lora_adaptor_list(lora_adapters)
            else:
                self.lora_adaptor_list = [LoraAdaptor(lora_adapter) for lora_adapter in lora_adapters]
        else:
            self.lora_adaptor_list = []

        if lora_tensor_names:
            if isinstance(lora_tensor_names, str):
                self.lora_tensor_names = [l.strip() for l in open(lora_tensor_names, "r").readlines()]
            else:
                self.lora_tensor_names = lora_tensor_names
        else:
            self.lora_tensor_names = []

        self._lora_alpha_from_input = lora_alpha_from_input

        self._prepared_model = is_prepared_model
        self._optimize_o_proj = optimize_o_proj

        self._handle_rope_ops = handle_rope_ops
        self._handle_alibi = handle_alibi

        self._mha_conv = is_linear_to_conv_converted
        self._nchw_aligned = is_nchw_aligned

        self._not_strict = not strict_rope_pattern
        self._strict = strict_rope_pattern

        self._encodings = encodings

        self._no_verification = skip_verification
        self._disable_auto_attn_finder = disable_auto_attn_finder

        # Trivial parameters for compatibility
        self._skip_mha2sha = False
        self._model_name = "mha2sha_model"
        self._replace_linear_with_conv = not is_linear_to_conv_converted
        self._handle_past_key_value = handle_past_key_value
        self._position_ids = None

        # Structure to store all MHA2SHA generated artifacts
        self.transformed_model = Mha2ShaTransformedModel(model=self._model)

    def _verify_can_run(self):
        """Verifies that the script meets requirements to run.

        List of Checks:
            - Checks for ONNX Minimum version is >=_ONNX_MIN_VERSION_NEEDED (see converter.py)
            - Checks for Python >=3.10
        """
        user_onnx_version = version.parse(onnx.version.version)
        minimum_onnx_version = version.parse(_ONNX_MIN_VERSION_NEEDED)
        cond_to_msg = [
            (
                user_onnx_version >= minimum_onnx_version,
                f"ONNX version must be >={_ONNX_MIN_VERSION_NEEDED}, got version: {onnx.version.version}",
            ),
            (
                sys.version_info >= (3, 10),
                f"Python version must be >= 3.10, got version: {sys.version}",
            ),
        ]
        for cond, msg in cond_to_msg:
            log_assert(cond=cond, msg=msg)

        if user_onnx_version == minimum_onnx_version:
            self.logger.warning(
                f"Got ONNX version: {user_onnx_version}. This version of ONNX will be deprecated in future releases. "
                "Please migrate to ONNX 1.16.1"
            )

    def convert(self) -> Mha2ShaTransformedModel:
        r"""Entry call for conversion to take place.

        Entry to start conversion of the MHA model. By default, logging will show each major step of the process.

        Returns:
            Converted MHA2SHA model and an optional verification status of the model.
            Verification status is only returned if --no-verification is set to False.
        """

        # Sanity check
        if self._lora_model:
            log_assert(self._mha_conv, "lora model only support mha-conv models.")

        if self._lora_alpha_from_input:
            log_assert(
                self._lora_model,
                f"lora_alpha_from_input expects lora model = True, but got {self._lora_model}",
            )

        if self._mha_conv and not self._nchw_aligned:
           self.logger.warning("Got non-nchw aligned mha model.")

        if self._build_ar and not self._no_verification:
            self.logger.warning("--build-ar flag used. Currently there is no verification of MHA vs SHA.")
        if self._optimize_o_proj:
            log_assert(self._mha_conv, "--optimize_o_proj only supports mha-conv model.")

        import onnxruntime

        onnxruntime.SessionOptions.use_deterministic_compute = True
        np.random.seed(42)

        try:
            # Run pattern matcher
            self.logger.info("-" * 20)
            self.logger.info("Step 2: Running auto pattern matcher on model object.")
            pattern, pattern_start_node_names, pattern_end_node_names = self._pattern_match()
        except Exception as ex:
            self.logger.warning("Cannot find MHA pattern. Skipping MHA2SHA adaptation.")
            self.logger.debug(f'Exception raised: "{ex}"')
            return self.transformed_model

        # Check the correctness of model object
        self.logger.info("-" * 20)
        if self._no_verification:
            self.logger.info(
                ("Step 3: Skipping checking the correctness of `model` object")
            )
        else:
            self.logger.info("Step 3: Checking the correctness of `model` object")
            self._check_original_model()

        # Generate model inputs and outputs
        self.logger.info("-" * 20)
        if self._no_verification:
            self.logger.info("Step 4: Skipping Generating model inputs and outputs.")
        else:
            self.logger.info("Step 4: Generating model inputs and outputs")
            np_inputs, golden_outputs = self._generate_golden_inputs_outputs()

        # Apply mha2sha optimization
        self.logger.info("-" * 20)
        self.logger.info("Step 5: Applying MHA2SHA optimization on model object")
        self._run_optimizations(pattern, pattern_start_node_names, pattern_end_node_names)

        runtime_seconds = time.time() - self._start_time
        self.logger.info(
            f"Total Runtime ----- {str(datetime.timedelta(seconds=runtime_seconds)).split('.')[0]} -----"
        )

        # Validate mha2sha model by running it on ONNXRT
        self.logger.info("-" * 20)
        if self._no_verification:
            self.logger.info("Step 6: Skipping comparing MHA2SHA model to Original by running on ONNXRT")
        else:
            self.logger.info("Step 6: Comparing MHA2SHA model to Original by running on ONNXRT")
            self._compare_goldens_to_converted(np_inputs, golden_outputs)

        return self.transformed_model

    def _check_original_model(self) -> None:
        """Checks the original model with QAIRT `native_checker`.

        Will log and error and exit the program if an error is found during checking.
        """

        if not ou.native_checker(self._model):
            self.logger.error("Model-Checker failed after model export. Exiting...")
            exit()

    def _get_models_inputs_output_names(
        self,
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        # TODO Generate LVM and LLM input based on self.handle_past_key_value
        try:
            inp_specs = get_input_info(self._model)
            out_specs = get_output_info(self._model)
            if not self._handle_past_key_value:
                inputs = self._generate_random_test_data(inp_specs)
            else:
                inputs = self._generate_llama_test_data(inp_specs)

            output_names = []
            for key in out_specs.keys():
                output_names.append(key)
        except Exception as e:
            self.logger.error(f"Generation of inputs and outputs failed due to: {e}")
            exit()

        return inputs, output_names

    def _replace_lora_weights(self, model: onnx.ModelProto, weights: dict[str, np.ndarray]):
        """ Replace the `model` initializers with weights from `weights` dictionary """
        init_map = {init.name: init for init in model.graph.initializer}

        for tensor_name, tensor in weights.items():
            init = init_map[tensor_name]

            model.graph.initializer.remove(init)

            init = onnx.helper.make_tensor(
                name=init.name,
                data_type=init.data_type,
                dims=init.dims,
                vals=tensor,
            )

            model.graph.initializer.append(init)

        
    def _get_random_lora_alpha_values(self, adapter_name: str, lora_alpha: np.ndarray):
        if not hasattr(self, 'lora_alphas'):
            self.lora_alphas = defaultdict(float)

        adapters = adapter_name.split("+")

        for adapter in adapters:
            if adapter not in self.lora_alphas:
                self.lora_alphas[adapter] = round(np.random.rand(), 2)

        alphas = np.zeros_like(lora_alpha)
        for i, adapter in enumerate(adapters):
            alphas[0][i] = self.lora_alphas[adapter]

        return alphas

    def _generate_golden_inputs_outputs(
        self,
    ) -> Tuple[Dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
        """Generates random inputs and golden outputs of the original model.

        Creates random numpy inputs that then run through the model to produce golden outputs to compare against the
        converted model later on. Additionally, the output tensor names are also captured for direct comparison.

        Returns:
            Tensor names to randomly generated numpy inputs, names of the output tensors, and the golden inputs produced.
        """

        np_inputs, output_names = self._get_models_inputs_output_names()

        golden_outputs = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_path = os.path.join(tmpdir, "mha_model.onnx")
            onnx.save(self._model, temp_model_path, save_as_external_data=True)

            # Zero out lora alpha
            if 'lora_alpha' in np_inputs:
                np_inputs['lora_alpha'] = np.zeros_like(np_inputs['lora_alpha'])

            golden_outputs['base'] = ou.run_model_on_ort(temp_model_path, np_inputs, output_names)

            if self.lora_adaptor_list:
                lora_model = copy.deepcopy(self._model)
                temp_model_path = os.path.join(tmpdir, "mha_lora_model.onnx")

                for adapter in self.lora_adaptor_list:
                    self.logger.info(f"Generating golden outputs for adapter: {adapter.name}")
                    # Load lora adapter weights
                    self._replace_lora_weights(lora_model, adapter.safetensor)
                    np_inputs['lora_alpha'] = self._get_random_lora_alpha_values(adapter.name, np_inputs['lora_alpha'])

                    onnx.save_model(lora_model, temp_model_path, save_as_external_data=True) 
                    golden_outputs[adapter.name] = ou.run_model_on_ort(temp_model_path, np_inputs, output_names)

            onnx.load_external_data_for_model(self._model, tmpdir)

        return np_inputs, golden_outputs

    def _pattern_match(self) -> Tuple[List[str], List[str], List[str]]:
        """Finds all MHA's within the original model.

        Will either use the Auto Finder for finding MHAs or fallback to the explicitly listed pattern types if no
        patterns are found by Auto Finder or if Auto Finder is disabled.
        """

        pattern = None
        if not self._disable_auto_attn_finder:
            self.logger.debug("Finding MHA with auto pattern finder.")
            use_quick_auto_finder = self._lora_model or self._gqa_model
            pattern, pattern_start_node_names, pattern_end_node_names = auto_attention_finder(
                self._model, self._mha_conv, use_quick_auto_finder
            )

        if pattern is None:
            self.logger.debug(
                "Cannot find MHA pattern with auto pattern finder, searching again using pre-defined patterns."
            )
            pattern, pattern_start_node_names, pattern_end_node_names = ou.get_pattern_start_end_nodes(
                self._model, attention_patterns
            )

        self.logger.info("Running pattern matcher - pattern matched:")
        self.logger.info(" ".join([str(elem) for elem in pattern]))
        self.logger.info(f"found_matched_pattern: {len(pattern_start_node_names)}")

        return pattern, pattern_start_node_names, pattern_end_node_names

    def _merge_encodings_mappings(self, mapping1, mapping2):
        """
        mapping1: a->[b1,b2]
        mapping2: b1->[c1,c2,c3]
                    b2->[c4]
        out_mapping: a->[c1,c2,c3,c4]
        """
        new_mapping = {}
        for enc_type in ("activation_encodings", "param_encodings"):
            reversed_mapping1 = {}
            new_mapping[enc_type] = {}
            for key, map1_names in mapping1[enc_type].items():
                for map1_n in map1_names:
                    if map1_n not in reversed_mapping1:
                        reversed_mapping1[map1_n] = []
                    if key not in reversed_mapping1[map1_n]:
                        reversed_mapping1[map1_n].append(key)

            for key, map2_names in mapping2[enc_type].items():
                # map2_names may not be unique
                map2_names = list(dict.fromkeys(map2_names))  # elements are unique and order is kept
                if key in reversed_mapping1:
                    # key is a mapping1 value
                    origin_keys = reversed_mapping1[key]
                else:
                    # key is not a mapping1 value
                    origin_keys = [key]

                for origin_k in origin_keys:
                    if origin_k not in new_mapping[enc_type]:
                        new_mapping[enc_type][origin_k] = []
                    new_mapping[enc_type][origin_k] += map2_names

            # handle the case when value of mapping1 is not in mapping2
            for map1_name, origin_names in reversed_mapping1.items():
                if map1_name not in mapping2[enc_type]:
                    for origin_k in origin_names:
                        if origin_k not in new_mapping[enc_type]:
                            new_mapping[enc_type][origin_k] = []
                        new_mapping[enc_type][origin_k].append(map1_name)

        return new_mapping

    def _merge_encodings_mapping_files(self, mapping1_file, mapping2_file, out_mappint_file):
        with open(mapping1_file, "r") as f:
            mapping1 = json.load(f)
        with open(mapping2_file, "r") as f:
            mapping2 = json.load(f)
        out_mapping = self._merge_encodings_mappings(mapping1, mapping2)
        with open(out_mappint_file, "w") as f:
            json.dump(out_mapping, f, indent=4)

    def _run_optimizations(
        self,
        pattern: List[str],
        pattern_start_node_names: List[str],
        pattern_end_node_names: List[str],
    ) -> None:
        """Performs MHA2SHA optimizations and saves a new model.

        Runs the optimizations for each pattern that matches an MHA. Additional optimization such as Linear to Conv
        may also be done based on the inputed argument.

        Args:
            pattern:
                List of Op Type's in pattern.
            pattern_start_node_names
                Starting node names for each pattern.
            pattern_end_node_names:
                Ending node names for each pattern.

        Returns:
           ModelProto optimized and path to the saved converted model.
        """

        self.logger.info("Step 5.1: Apply prequant model adaption.")
        prequant_opt = PreQuantAdaption(
            self._model,
            self._lora_model,
            self._lora_alpha_from_input,
        )
        mha_encodings, prequant_encodings_mapping, lora_nodes = prequant_opt.optimize(self._encodings)

        if self._lora_alpha_from_input:
            self._lora_alpha = lora_nodes[0].lora_alpha_value

        self.logger.info("Step 5.2: Apply mha2sha model adaption.")
        mha_opt = MHA2SHAOptimizer(
            self._model,
            pattern,
            pattern_start_node_names,
            pattern_end_node_names,
            self._handle_rope_ops,
            self._handle_past_key_value,
            self._prepared_model,
            self._replace_linear_with_conv,
            self._position_ids,
            self._mha_conv,
            self._nchw_aligned,
            self._strict,
            self._lora_model,
            self._lora_alpha_from_input,
            lora_nodes,
            self._llm_model,
            self._base_arch,
            self._gqa_model,
            self._build_ar,
            self._handle_alibi,
        )
        mha_opt.optimize()

        if self._mha_conv and self._optimize_o_proj:
            self.logger.info("Optimize head concat to o_proj pattern for mha-conv models...")
            o_proj_opt = OProjOptimzier(
                self._model,
                mha_opt.qkv_head_concat_node_list,
            )
            o_proj_opt.optimize()

        self.logger.info("-" * 20)
        self.logger.info("Step 5.3: Create sha encodings.\n")
        if mha_encodings is not None:
            mha_to_sha_encodings_mapping, sha_encodings = mha_opt.mha_to_sha_encoding_mapping(mha_encodings)

            # merge prequant and mha2sha encodings mappings
            self.transformed_model.encodings_map.update(self._merge_encodings_mappings(
                prequant_encodings_mapping, mha_to_sha_encodings_mapping
            ))
            self.transformed_model.encodings.update(sha_encodings)

        else:
            self.logger.warning("exported_model_encoding_path is None. Skiping create SHA encoding.")

        if self._lora_model:
            self.logger.info("Step 5.4: Convert lora adapter encodings and safetensors...\n")

            if self.lora_tensor_names:
                # Combine 1. SHA lora tensors 2. Tensors that are not modified by mha2sha
                all_io = [inp.name for inp in self._model.graph.input] + [out.name for out in self._model.graph.output]
                all_inits = [init.name for init in self._model.graph.initializer]
                all_outputs = [out for node in self._model.graph.node for out in node.output]
                all_sha_tensors = set(all_io + all_inits + all_outputs)

                # mha_lora_tensors: Tensors that are not modified by mha2sha
                mha_lora_tensors = all_sha_tensors & set(self.lora_tensor_names)
                # sha_lora_tensors: Tensors that are mapped from mha to sha
                sha_lora_tensors = mha_opt.sha_lora_tensor_names

                all_lora_tensors = mha_lora_tensors | sha_lora_tensors

                self.transformed_model.lora_tensor_names.update(all_lora_tensors)

            if self.lora_adaptor_list:
                lora_weights_mapping = {
                    init.name: onnx.numpy_helper.to_array(init)
                    for init in self._model.graph.initializer
                    if "lora" in init.name
                }
                self.transformed_model.lora_weights.update(lora_weights_mapping)
                self._convert_lora_adaptors()

    def _convert_lora_adaptors(self):
        mha_to_sha_encodings_names = self.transformed_model.encodings_map
        sha_encoding = self.transformed_model.encodings

        LoraAdaptor.mha_to_sha_encodings_names = mha_to_sha_encodings_names
        LoraAdaptor.mha_conv = self._mha_conv
        LoraAdaptor.base_sha_lora_keys = self.transformed_model.lora_weights.keys()
        LoraAdaptor.base_sha_encoding = sha_encoding

        lora_adapters = []
        for lora_adaptor in self.lora_adaptor_list:
            lora_sha_safetensor_dict, lora_sha_encodings = lora_adaptor.map_encoding_and_slice_safetensor()

            lora_adapter = LoraAdapter(
                name=lora_adaptor.name, weights=lora_sha_safetensor_dict, encodings=lora_sha_encodings
            )

            lora_adapters.append(lora_adapter)

        self.transformed_model.lora_adapters.extend(lora_adapters)

    def _compare_goldens_to_converted(
        self,
        np_inputs: Dict[str, np.ndarray],
        golden_outputs: dict[str, dict[str, np.ndarray]],
    ):
        """Compares the outputs of the original model and the converted on.

        Using the randomly generated numpy inputs, the outputs of the original model and the converted model are compared
        and there MAD is logged out.

        Args:
            onnx_output_filename:
                The path of where the converted model was saved.
            np_inputs:
                Randomly generated numpy inputs.
            output_names:
                Output names of the original model.
            golden_outputs:
                Outputs of the origina model.
        Returns:
            Verification status of the comparision.
        """

        def _compare(goldens: dict[str, np.ndarray], converted_outputs: dict[str, np.ndarray]):

            status = True
            for _name, _goldens in goldens.items():
                _converted = converted_outputs[_name] 

                if 'past' in _name:
                    # align transpose_key_cache
                    if _goldens.shape[-2] != _converted.shape[-2]:
                        _converted = np.transpose(_converted, [0, 1, 3, 2])

                    # align concat_past_key_value_to_batch
                    if _goldens.shape[0] != _converted.shape[0]:
                        _converted = np.transpose(_converted, [1, 0, 2, 3])

                status = status and np.allclose(_goldens, _converted, atol=1e-4)
                self.logger.debug(
                    f"For {_name} : MAD = {str(np.abs(_goldens - _converted).max())}"
                )

            return status


        if self._build_ar:
            self.logger.warning("'--build-ar' flag used, skipping comparison of MHA and SHA logits.")
            return False

        if self._lora_alpha_from_input:
            _np_inputs, _ = self._get_models_inputs_output_names()
            np_inputs["lora_alpha"] = np.ones(1, dtype=_np_inputs["lora_alpha"].dtype) * self._lora_alpha

        if self._handle_past_key_value:
            self.logger.debug("Permuting inputs for past key/value inputs")
            np_inputs = {
                k: t.transpose(1, 0, 2, 3) if "past" in k else t
                for k, t in np_inputs.items()
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_path = os.path.join(tmpdir, "sha_model.onnx")
            onnx.save(self._model, temp_model_path, save_as_external_data=True)

            # Zero out lora alpha
            if 'lora_alpha' in np_inputs:
                np_inputs['lora_alpha'] = np.zeros_like(np_inputs['lora_alpha'])


            output_names = [output.name for output in self._model.graph.output]
            sha_model_outputs = ou.run_model_on_ort(temp_model_path, np_inputs, output_names)

            status = _compare(golden_outputs['base'], sha_model_outputs)

            verification_str = (
                f"{Colors.OKGREEN if status else Colors.FAIL}"
                f"{'OK' if status else 'FAIL'}{Colors.ENDC}"
            )
            self.logger.info(f"Verification Status{'(Base Model)' if self.lora_adaptor_list else ''} ----- {verification_str} -----")

            if self.lora_adaptor_list:
                lora_model = copy.deepcopy(self._model)
                temp_model_path = os.path.join(tmpdir, "sha_lora_model.onnx")

                for adapter in self.transformed_model.lora_adapters:
                    self._replace_lora_weights(lora_model, adapter.weights)
                    np_inputs['lora_alpha'] = self._get_random_lora_alpha_values(adapter.name, np_inputs['lora_alpha'])

                    onnx.save(lora_model, temp_model_path, save_as_external_data=True)

                    sha_output = ou.run_model_on_ort(temp_model_path, np_inputs, output_names)
                    status = _compare(golden_outputs[adapter.name], sha_output)
                    verification_str = (
                        f"{Colors.OKGREEN if status else Colors.FAIL}"
                        f"{'OK' if status else 'FAIL'}{Colors.ENDC}"
                    )
                    self.logger.info(f"Verification Status({adapter.name}) ----- {verification_str} -----")

            onnx.load_external_data_for_model(self._model, tmpdir)


    # Helper functions
    def _generate_llama_test_data(self, input_info_dict: Dict[Text, TensorInfo]) -> Dict[str, np.ndarray]:
        """Generate the test inputs based on given shape and data type (LLaMA).

        Args:
            input_info_dict:
                A dict with mapping from input name to another dict having info regarding input shape and input dtype.

        Returns:
        A dict with mapping from input name to test data of the input in np.array format.
        """

        final_inputs = OrderedDict()
        input_info_dict_list = list(input_info_dict.items())
        for input_name, tensor in input_info_dict_list:
            if input_name == 'input_ids':
                final_inputs[input_name] = np.random.randint(1, 500, tensor.shape).astype(tensor.dtype)
            else:
                input_shape = tensor.shape
                input_dtype = tensor.dtype
                final_inputs[input_name] = np.random.rand(*input_shape).astype(input_dtype)
        return final_inputs

    def _generate_random_test_data(self, input_info_dict: Dict[Text, TensorInfo]) -> Dict[str, np.ndarray]:
        """Generate the test inputs based on given shape and data type (Regular).

        Args:
            input_info_dict:
                A dict with mapping from input name to another dict having info regarding input shape and input dtype.

        Returns:
        A dict with mapping from input name to test data of the input in np.array format.
        """

        final_inputs = OrderedDict()
        for input_name, tensor in input_info_dict.items():
            input_shape = tensor.shape
            input_dtype = tensor.dtype
            final_inputs[input_name] = np.random.rand(*input_shape).astype(input_dtype)
        return final_inputs
