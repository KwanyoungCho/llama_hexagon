# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse

from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_snooping_cmd_options import QAIRTSnoopingCmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path


class QAIRTOneshotLayerwiseSnoopingCmdOptions(QAIRTSnoopingCmdOptions):

    def __init__(self, args, validate_args=True):
        super().__init__(args=args, type="oneshot-layerwise", validate_args=validate_args)

    def initialize(self):
        """
        type: (List[str]) -> argparse.Namespace

        :param args: User inputs, fed in as a list of strings
        :return: Namespace object
        """
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description="Script to run oneshot-layerwise snooping.")

        self._base_initialize()

        self._verifier_args.add_argument(
            '--default_verifier', type=str.lower, required=True, nargs='+', action="append",
            help='Default verifier used for verification. The options '
            '"RtolAtol", "AdjustedRtolAtol", "TopK", "L1Error", "CosineSimilarity", "MSE", "MAE", "SQNR", "ScaledDiff" are supported. '
            'An optional list of hyperparameters can be appended. For example: --default_verifier rtolatol,rtolmargin,0.01,atolmargin,0.01 '
            'An optional list of placeholders can be appended. For example: --default_verifier CosineSimilarity param1 1 param2 2. '
            'to use multiple verifiers, add additional --default_verifier CosineSimilarity')

        self._verifier_args.add_argument(
            '--result_csv', type=str, required=False,
            help='Path to the csv summary report comparing the inference vs framework'
            'Paths may be absolute, or relative to the working directory.'
            'if not specified, then a --problem_inference_tensor must be specified')

        self._verifier_args.add_argument('--verifier_config', type=str, default=None,
                                   help='Path to the verifiers\' config file')
        self._verifier_args.add_argument('--run_tensor_inspection', action="store_true", default=False,
                                   help="To run tensor inspection, pass this argument")

        self._quantizer_args.add_argument(
            '--float_fallback', action="store_true", default=False,
            help='Use this option to enable fallback to floating point (FP) instead of fixed point. '
            'This option can be paired with --quantizer_float_bitwidth to indicate the bitwidth for '
            'FP (by default 32). If this option is enabled, then input list must '
            'not be provided and --ignore_encodings must not be provided. '
            'The external quantization encodings (encoding file/FakeQuant encodings) '
            'might be missing quantization parameters for some interim tensors. '
            'First it will try to fill the gaps by propagating across math-invariant '
            'functions. If the quantization params are still missing, '
            'then it will apply fallback to nodes to floating point.')

        self._optional_args.add_argument('--debug_mode_off', dest="debug_mode", action="store_false",
                              required=False,
                              help="This option can be used to avoid dumping intermediate outputs.")
        self._optional_args.set_defaults(debug_mode=True)

        self._optional_args.add_argument(
            '--start_layer', type=str, default=None, required=False, help=
            "save all intermediate layer outputs from provided start layer to bottom layer of model. \
                                    Can be used in conjunction with --end_layer.")
        self._optional_args.add_argument(
            '--end_layer', type=str, default=None, required=False, help=
            "save all intermediate layer outputs from top layer to  provided end layer of model. \
                                  Can be used in conjunction with --start_layer.")

        self._optional_args.add_argument(
            '--add_layer_outputs', default=[], help="Output layers to be dumped. \
                                e.g: node1,node2")

        self._optional_args.add_argument(
            '--add_layer_types', default=[],
            help='outputs of layer types to be dumped. e.g :Resize,Transpose.\
                                All enabled by default.')

        self._optional_args.add_argument(
            '--skip_layer_types', default=[],
            help='comma delimited layer types to skip dumping. e.g :Resize,Transpose')

        self._optional_args.add_argument(
            '--skip_layer_outputs', default=[],
            help='comma delimited layer output names to skip dumping. e.g: node1,node2')

        self._optional_args.add_argument(
            '--stage', type=str.lower, required=False, choices=['source', 'converted',
                                                                'quantized'], default='source',
            help='Specifies the starting stage in the Accuracy Debugger pipeline. \
            source: starting with a source framework model, \
            converted: starting with a converted model, \
            quantized: starting with a quantized model. \
            Default is source.')

        self._optional_args.add_argument('--disable_graph_optimization', action="store_true",
                              help="Disables basic model optimization")

        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args = self._verify_update_base_parsed_args(parsed_args)
        parsed_args.result_csv = get_absolute_path(parsed_args.result_csv)
        supported_verifiers = [
            "rtolatol", "adjustedrtolatol", "topk", "l1error", "cosinesimilarity", "mse", "mae",
            "sqnr", "scaleddiff"
        ]
        for verifier in parsed_args.default_verifier:
            verifier_name = verifier[0].split(',')[0]
            if verifier_name not in supported_verifiers:
                raise ParameterError(
                    f"--default_verifier '{verifier_name}' is not a supported verifier.")

        return parsed_args
