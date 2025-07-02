# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import traceback
import os
import pandas as pd

from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_tensor_paths
from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import permute_tensor_data_axis_order, \
    get_irgraph_tensors_info, get_irgraph_dtypes, load_data
from qti.aisw.accuracy_debugger.lib.tensor_inspection.tensor_inspection_runner import TensorInspectionRunner
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_warning_message


class TensorInspector(object):

    def __init__(self, logger, args):
        # type: (Logger, namespace) -> None

        self._logger = logger
        self.args = args
        self.tensors_info = get_irgraph_tensors_info(
            qnn_model_json_path=self.args.qnn_model_json_path, dlc_path=self.args.dlc_path,
            output_dir=self.args.output_dir)
        self.irgraph_dtypes = None
        if args.use_native_output_files:
            self.irgraph_dtypes = get_irgraph_dtypes(qnn_model_json_path=args.qnn_model_json_path,
                                                        dlc_path=args.dlc_path)

    def run(self):

        try:
            golden_tensor_paths = get_tensor_paths(self.args.golden_output_reference_directory)
            inference_tensor_paths = get_tensor_paths(self.args.inference_results)

            inference_tensors = list(inference_tensor_paths.keys())

            mapping = {}
            if self.args.tensor_mapping and os.path.exists(self.args.tensor_mapping):
                with open(self.args.tensor_mapping) as tensor_mapping:
                    mapping = json.load(tensor_mapping)
            inference_to_golden_tensor_map = {
                inference:
                mapping[inference]
                if inference in mapping and mapping[inference] is not None else inference
                for inference in inference_tensors
            }

            tensor_inspector = TensorInspectionRunner(self._logger)
            tensor_inspector_dir = os.path.join(self.args.output_dir, 'tensor_inspection')

            fields = ['Name', 'golden_min', 'golden_max', 'target_min', 'target_max']
            if self.args.target_encodings:
                fields.extend([
                    'calibrated_min', 'calibrated_max', '(target_min-calibrated_min)',
                    '(target_max-calibrated_max)'
                ])

            summary_df = pd.DataFrame(columns=fields)
            for inference_tensor in inference_tensors:
                golden_tensor_name = inference_to_golden_tensor_map[inference_tensor]
                if golden_tensor_name not in golden_tensor_paths.keys():
                    self._logger.warning(
                        get_warning_message("WARNING_VERIFIER_MISSING_GOLDEN_TENSOR_DATA")(
                            str(golden_tensor_name)))
                    continue

                golden_data = load_data([golden_tensor_name], golden_tensor_paths, self.irgraph_dtypes)[0]
                inference_data = load_data([inference_tensor], inference_tensor_paths, self.irgraph_dtypes)[0]

                if self.tensors_info and golden_tensor_name in self.tensors_info:
                    # Permute target tensor to align with golden tensor
                    inference_data, _ = permute_tensor_data_axis_order(
                        inference_data, self.tensors_info[golden_tensor_name])

                result = tensor_inspector.run(inference_tensor, golden_data, inference_data,
                                              tensor_inspector_dir,
                                              target_encodings=self.args.target_encodings)
                summary_df = pd.concat([summary_df, pd.DataFrame([result])], ignore_index=True,
                                       sort=False)
            return summary_df

        except Exception as excinfo:
            traceback.print_exc()
            raise Exception("Encountered error: {}".format(str(excinfo)))
