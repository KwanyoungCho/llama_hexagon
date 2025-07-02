# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse

from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_snooping_cmd_options import\
    QAIRTSnoopingCmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError


class QAIRTLayerwiseCmdOptions(QAIRTSnoopingCmdOptions):

    def __init__(self, args, validate_args: bool = True) -> None:
        super().__init__(args=args, type="layerwise", validate_args=validate_args)

    def initialize(self) -> None:
        """
        Intializes args for layerwise custom-override snooping.
        """
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script to run layerwise-custom-override snooping.")

        self._base_initialize()

        self._verifier_args.add_argument(
            '--default_verifier', type=str.lower, required=True, nargs='+', action="append",
            help='Default verifier used for verification. The options '
            '"RtolAtol", "AdjustedRtolAtol", "TopK", "L1Error", "CosineSimilarity", "MSE", "MAE", '
            '"SQNR", "ScaledDiff" are supported. An optional list of hyperparameters can be appended.'
            ' For example: --default_verifier rtolatol,rtolmargin,0.01,atolmargin,0.01 '
            'An optional list of placeholders can be appended. For example: --default_verifier '
            'CosineSimilarity param1 1 param2 2. to use multiple verifiers, add additional '
            '--default_verifier CosineSimilarity')

        self._optional_args.add_argument('--stage', default='source', type=str, required=False,
                                         choices=['source', 'verification'])

        self._optional_args.add_argument(
            '--snooper_artifact_dir', default=None, type=str, required=False,
            help='If stage is verification, then pass the path to "snooping" directory which should\
                exist in your working directory')

        self._optional_args.add_argument(
            '--memory_efficient', default=False, action="store_true",
            help="If True, unimportant subgraph artifacts will be deleted")

        self._optional_args.add_argument(
            '--debug_subgraph_inputs', '--start_layer', dest='debug_subgraph_inputs', type=str,
            default=None, required=False,
            help="pass a comma separated inputs for the graph which is to be debugged.")
        self._optional_args.add_argument(
            '--debug_subgraph_outputs', '--end_layer', dest='debug_subgraph_outputs', type=str,
            default=None, required=False,
            help="pass a comma separated outputs for the graph which is to be debugged.")

        self._optional_args.add_argument(
            '--quantized_dlc_path', default=None, required=False,
            help='Incsae the qauntization_override file is from qairt-quantizer with convert'
            ' ops in it, corresponding quantized_dlc file is required.')

        self._optional_args.add_argument('--compulsory_override', default=None, required=False,
            help='path to override file containing encodings for the ops which has to be run in certain\
                 precision for each subgraph debugging. This is helpful for example: incase one op is already\
                 known to be problematic for higher precision. Override must be in aimet format with\
                 QNN IR names.'
            )

        self._optional_args.add_argument('--max_parallel_subgraphs', default=1, required=False, type=int,
            help='Maximum numbers of subgraph debugging that can be done in parallel. Each subgraphs\
                will require a separate CPU core.'
            )

        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args = self._verify_update_base_parsed_args(parsed_args)

        supported_verifiers = [
            "rtolatol", "adjustedrtolatol", "topk", "l1error", "cosinesimilarity", "mse", "mae",
            "sqnr", "scaleddiff"
        ]
        for verifier in parsed_args.default_verifier:
            verifier_name = verifier[0].split(',')[0]
            if verifier_name not in supported_verifiers:
                raise ParameterError(
                    f"--default_verifier '{verifier_name}' is not a supported verifier.")

        if parsed_args.debug_subgraph_inputs:
            parsed_args.debug_subgraph_inputs = parsed_args.debug_subgraph_inputs.split(',')
        else:
            parsed_args.debug_subgraph_inputs = []
        if parsed_args.debug_subgraph_outputs:
            parsed_args.debug_subgraph_outputs = parsed_args.debug_subgraph_outputs.split(',')
        else:
            parsed_args.debug_subgraph_outputs = []

        return parsed_args
