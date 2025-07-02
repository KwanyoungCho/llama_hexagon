# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path, format_args
from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_snooping_cmd_options import QAIRTSnoopingCmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Framework, Engine, Runtime
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Architecture_Target_Types, Engine, Runtime, \
    Android_Architectures, X86_Architectures, \
    Device_type, Qnx_Architectures, Windows_Architectures, X86_windows_Architectures, Aarch64_windows_Architectures
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message

import argparse
import os
import numpy as np


class QAIRTBinarySnoopingCmdOptions(QAIRTSnoopingCmdOptions):

    def __init__(self, args, validate_args=True):
        super().__init__(args=args, type="binary", validate_args=validate_args)

    def initialize(self):
        """
        type: (List[str]) -> argparse.Namespace

        :param args: User inputs, fed in as a list of strings
        :return: Namespace object
        """
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description="Script to run binary snooping.")

        self._base_initialize()

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

        self._optional_args.add_argument('--min_graph_size', default=16, type=int, required=False,
                                   help='Provide the minimum subgraph size')

        self._optional_args.add_argument(
            '--subgraph_relative_weight', default=0.4, type=float, required=False,
            help='Helps in deciding whether a sub graph is further debugged or not. '
            'If a subgraph scores > 40 percent of the aggreagte score of two subgraphs, we investage '
            'the subgraph further.')

        self._optional_args.add_argument('--verifier', type=str.lower, required=False, default='mse',
                                   help='Choose verifer among [sqnr, mse] for the comparison')

        self._optional_args.add_argument('--disable_graph_optimization', action="store_true",
                                         default=False,
                              help="Disables basic model optimization")

        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args = self._verify_update_base_parsed_args(parsed_args)
        return parsed_args
