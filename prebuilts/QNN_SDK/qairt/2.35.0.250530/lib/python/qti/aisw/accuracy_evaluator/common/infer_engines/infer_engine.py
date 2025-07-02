# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from abc import ABC
import os
import numpy as np
from typing import List, Dict
import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.tools.core.utilities.framework.utils.constants import FrameworkExecuteReturn


class InferenceEngine(ABC):
    """InferenceEngine class is an abstract class, implemented by different
    ML frameworks to do inference on a set of inputs."""

    def __init__(self, model: str, inputlistfile: str, output_path: str, multithread: bool = False,
                 input_info: dict = None, output_info: dict = None, gen_out_file: str = None,
                 extra_params: dict = None):
        self.model_path = model
        self.input_path = inputlistfile
        self.input_info = input_info
        self.output_info = output_info
        self.output_path = output_path
        self.multithread = multithread
        self.extra_params = extra_params
        self.gen_out_file = gen_out_file

    def save_outputs_and_profile(self,
        output_names: List[str],
        outputs: List[np.ndarray],
        iter: int,
        save_outputs: bool,
        do_profile: bool) -> tuple[Dict[str, tuple], List[str]]:
        """This method saves the output arrays in numpy format and performs profiling.

        Args:
            output_names: List of model output names
            outputs: List of output numpy arrays
            iter: Inference iteration
            save_outputs: Flag to save outputs
            do_profile: Flag to perform profiling

        Returns:
            Dictionary: Profile data mapping output names and its corresponding tuple of datatype,
            List[str]: List of file paths for saved outputs
        """
        profile_data = {}
        _paths = []
        for i, name in enumerate(output_names):
            if save_outputs:
                out_path = os.path.join(self.output_path, str(name) + '_' + str(iter) + '.raw')
                _paths.append(out_path)
                outputs[i].tofile(out_path)

            if do_profile:
                if (not outputs[i].size or outputs[i].dtype == bool):
                    profile_data[name] = (outputs[i].dtype, outputs[i].shape, outputs[i],
                                          outputs[i], outputs[i])
                else:
                    profile_data[name] = (outputs[i].dtype, outputs[i].shape,
                                            round(np.min(outputs[i]),3),
                                            round(np.max(outputs[i]),3),
                                            round(np.median(outputs[i]), 3))
        return profile_data, _paths

    def generate_input_list_from_file(self, inps_per_sample: list[str], input_nodes: list = [],
                                      convert_nchw: bool = False) -> List[np.ndarray]:
        """Given a list of filepath(s) for each input sample, read the file(s) into a numpy array
        Args:
            inps_per_sample: list of filepath(s) for each input sample
            input_nodes: List of name, datatype and shape of each input,
                         in the format [[name0, type0, shape0], [name1, type1, shape1], ...]
            convert_nchw: Boolean to indicate whether input has to be converted to NCHW format
        Returns:
            List: List of numpy arrays for each input for each sample
        """
        inputs_per_sample_list = []
        for idx, inp in enumerate(inps_per_sample):
            if self.input_info is None:
                # When input shapes and dtypes are not passed by user, input_list
                # is formed using input_nodes list
                try:
                    input_np = np.fromfile(inp,
                                           dtype=input_nodes[idx][1]).reshape(input_nodes[idx][2])
                except Exception as e:
                    raise ce.InferenceEngineException(
                        "Unable to extract input info from model. Please try "
                        "passing input-info", e)
            else:
                if input_nodes[idx][0] not in self.input_info:
                    raise ce.ConfigurationException(
                        f"Invalid Configuration: Input info name not valid for this model. "
                        f"Expected: {input_nodes[idx].name}"
                    )

                input_np = np.fromfile(
                    inp,
                    dtype=(Helper.get_np_dtype(self.input_info[input_nodes[idx][0]][0]))).reshape(
                        self.input_info[input_nodes[idx][0]][1])
                if convert_nchw:
                    shape = input_np.shape
                    input_np = input_np.reshape((shape[0], shape[1], shape[2], shape[3])).transpose(
                        (0, 3, 1, 2))
            inputs_per_sample_list.append(input_np)
        return inputs_per_sample_list

    def save_outputs_to_file(self, outputs_per_sample: Dict[str, FrameworkExecuteReturn],
                             iter: int) -> list[str]:
        """Save given list of numpy arrays to file.

        Args:
            outputs_per_sample: Dictionary of outputs per sample, with output name as key and output tensor as value
            iter: Inference iteration
        Returns:
            List of filepaths for all outputs written to file
        """
        out_dict = self.output_info if self.output_info else outputs_per_sample
        paths_per_sample = []
        for out_name in out_dict:
            out_path = os.path.join(self.output_path, f"{out_name}_{iter}.raw")
            outputs_per_sample[out_name].tofile(out_path)
            paths_per_sample.append(out_path)
        return paths_per_sample

    def execute(self):
        pass

    def validate(self):
        pass
