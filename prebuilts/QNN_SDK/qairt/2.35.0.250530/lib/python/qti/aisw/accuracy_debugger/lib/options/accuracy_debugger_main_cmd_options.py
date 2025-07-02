# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions

import argparse
import sys
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine, DebuggingAlgorithm


class MainCmdOptions(CmdOptions):

    def __init__(self, engine, args):
        super().__init__('main', args, engine, validate_args=False)

    def initialize(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter, description=
            "Script for using various debugger features.",
            add_help=False, allow_abbrev=False)

        components = self.parser.add_argument_group(
            "Arguments to select which component of the tool to run. "
            "Arguments are mutually exclusive (at most one can be selected). "
        )
        components.add_argument('--framework_runner', action="store_true", default=False,
                                help="Run framework")
        
        # DEPRECATED Alias for --framework_runner
        components.add_argument("--framework_diagnosis", action = DeprecationWarningAction, nargs = 0, default = False,
                                help = argparse.SUPPRESS)
        
        components.add_argument('--inference_engine', action="store_true", default=False,
                                help="Run inference engine")
        components.add_argument('--verification', action="store_true", default=False,
                                help="Run verification")
        components.add_argument('--compare_encodings', action="store_true", default=False,
                                help="Run compare encodings")
        components.add_argument('--tensor_inspection', action="store_true", default=False,
                                help="Run tensor inspection")
        if self.engine in [Engine.QNN.value, Engine.SNPE.value]:
            components.add_argument('--quant_checker', action="store_true", default=False,
                                    help="Run QuantChecker")
            components.add_argument('--binary_snooping', action="store_true", default=False,
                                    help="Run Binary Snooping")
        if self.engine == Engine.QAIRT.value:
            components.add_argument('--snooping', type=str, default=None,
                                    choices=[
                                        DebuggingAlgorithm.oneshot_layerwise.value,
                                        DebuggingAlgorithm.cumulative_layerwise.value,
                                        DebuggingAlgorithm.layerwise.value,
                                        DebuggingAlgorithm.binary.value
                                    ],
                                    help="Run Snooping")

        optional = self.parser.add_argument_group('optional arguments')
        optional.add_argument(
            '-h', '--help', action="store_true", default=False, help=
            "Show this help message.  To show help for any of the components, run script with --help"
            " and --<component>. For example, to show the help for Framework Runner, run script with"
            " the following: --help --framework_runner")

        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        # print help for MainCmdOptions
        if (parsed_args.help):
            self.parser.print_help()

        # currently only support running one component at a time, or running wrapper
        component_args_list = [
            parsed_args.framework_runner, parsed_args.inference_engine, parsed_args.verification,
            parsed_args.compare_encodings, parsed_args.tensor_inspection
        ]
        if self.engine in [Engine.QNN.value, Engine.SNPE.value]:
            component_args_list.extend([parsed_args.quant_checker, parsed_args.binary_snooping])
        if self.engine == Engine.QAIRT.value and parsed_args.snooping:
            component_args_list.append(True)

        # check to ensure only one or no components are selected
        if (sum(component_args_list) > 1):
            raise ParameterError(
                "Too many components selected. Please run script with only one or no components")

        return parsed_args

    def parse(self):
        if (not self.initialized):
            self.initialize()
        opts, _ = self.parser.parse_known_args(self.args)
        return self.verify_update_parsed_args(opts)
    
class DeprecationWarningAction(argparse.Action):
    def __call__(self,parser, namespace, values, option_string = None):
        print(f"DeprecationWarning: {option_string} is deprecated and will be removed in the future. Please use --framework_runner instead.")
        setattr(namespace,"framework_runner",True)
        delattr(namespace,"framework_diagnosis")
        