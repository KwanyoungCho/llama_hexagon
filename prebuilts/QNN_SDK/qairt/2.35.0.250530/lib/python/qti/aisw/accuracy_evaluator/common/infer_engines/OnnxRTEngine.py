# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import numpy as np
import os
import shutil
import sys
import time

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.infer_engines.infer_engine import InferenceEngine
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_framework import OnnxModelHelper


class OnnxInferenceEngine(InferenceEngine):
    """OnnxInferenceEngine class takes required inputs supplied by user from
    commandline options and calls validate and execute methods.

    To use:
    >>> engine = OnnxInferenceEngine(model, inputlistfile, output_path, multithread, input_info,
                output_info, gen_out_file, extra_params)
    >>> engine.validate()
    >>> engine.execute()
    """

    def __init__(self, model: str, inputlistfile: str, output_path: str, multithread: bool = False,
                 input_info: dict = None, output_info: dict = None, gen_out_file: str = None,
                 extra_params: dict = None, convert_nchw: bool = False):
        super().__init__(model, inputlistfile, output_path, multithread, input_info, output_info,
                         gen_out_file, extra_params)
        self.convert_nchw = convert_nchw
        onnxruntime = Helper.safe_import_package("onnxruntime", "1.17.1")
        onnxruntime.set_default_logger_severity(3)
        self.framework_manager = FrameworkManager()
        logging.getLogger("OnnxModelHelper").setLevel("ERROR")  # Supress Warnings from onnxruntime
        self.validate()

    def execute(self):
        """This method runs the given onnx model on onnxruntime returns session and status of execution
        Returns:
            Execution status
            Time taken for inference
        """
        qacc_file_logger.debug("OnnxInferenceEngine start execution")

        # capture inference time
        inf_time = 0

        onnx_model = self.framework_manager.load(input_model=self.model_path)
        inp_nodes = []
        for inp in onnx_model.graph.input:
            shape = []
            for i in range(len(inp.type.tensor_type.shape.dim)):
                shape.append(inp.type.tensor_type.shape.dim[i].dim_value)
            np_type, _ = OnnxModelHelper.onnx_type_to_numpy(str(inp.type.tensor_type.elem_type))
            inp_nodes.append([inp.name, np_type, shape])

        # Create the output file if requested.
        out_list_file = None
        if self.gen_out_file:
            out_list_file = open(self.gen_out_file, 'w')

        start_time = time.time()
        # Start execution for each input sample
        with open(self.input_path) as f:
            for iter, line in enumerate(f):
                inps_per_sample = line.strip().split()
                inps_per_sample = [
                    inp.split(':=')[-1].strip() for inp in inps_per_sample if inp.strip()
                ]
                input_data_list = self.generate_input_list_from_file(
                    inps_per_sample=inps_per_sample, input_nodes=inp_nodes,
                    convert_nchw=self.convert_nchw)
                outputs = self.framework_manager.execute(input_model=onnx_model,
                                                         input_data=input_data_list)
                if self.output_info:
                    # check if output names configured in model config are valid
                    diff_names = set(outputs.keys()).difference(set(self.output_info.keys()))
                    if diff_names:
                        qacc_logger.warn(f'Output name(s) {diff_names} in config are incorrect/missing.')

                out_paths = self.save_outputs_to_file(outputs_per_sample=outputs, iter=iter)
                if self.gen_out_file:
                    out_list_file.write(','.join(out_paths) + '\n')
        if self.gen_out_file:
            out_list_file.close()

        inf_time = time.time() - start_time
        qacc_file_logger.debug("OnnxInferenceEngine execution success")
        return True, inf_time

    def validate(self):
        """This method checks whether the given model_path ,model,input_path and output_path are
        valid or not
        Returns:
            Validation status
        """
        qacc_file_logger.debug("OnnxInferenceEngine validation")

        # check the existence of model path and its validity
        _ = self.framework_manager.validate(input_model=self.model_path)

        # check whether the output path exists and create the path otherwise
        if self.output_path and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # check the existence of input path
        if not os.path.exists(self.input_path):
            raise ce.InferenceEngineException(f'Input path : {self.input_path} does not exist')
        qacc_file_logger.debug("OnnxInferenceEngine validation success")
        return True
