# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys
import os
from typing import Any
import numpy as np
import pandas as pd
from logging import Logger

from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ConfigError
from qti.aisw.accuracy_debugger.lib.verifier.nd_verifier_factory import VerifierFactory
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import VerifierError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, dump_json


def get_qairt_snooper_class(type: str) -> Any:
    '''
    Method to find the Snooper class given snooper type.

    :param type: snooper type. Following snooping algos are supported: [binary, layerwise,
     cumulative-layerwise, oneshot-layerwise, layerwise-custom-override,
     cumulative-layerwise-custom-override]
    :return: class for the given snooper type
    :raises: Exception if snooper type is not supported.
    '''
    from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.nd_qairt_binary_snooping\
        import QAIRTBinarySnooping
    from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.nd_qairt_oneshot_layerwise_snooping\
        import QAIRTOneshotLayerwiseSnooping
    from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.qairt_layerwise_snooping\
        import QAIRTLayerwiseSnooping
    from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.qairt_cumulative_layerwise_snooping\
        import QAIRTCumulativeLayerwiseSnooping

    snooper_class = {
        'binary': QAIRTBinarySnooping,
        'layerwise': QAIRTLayerwiseSnooping,
        'cumulative-layerwise': QAIRTCumulativeLayerwiseSnooping,
        'oneshot-layerwise': QAIRTOneshotLayerwiseSnooping
    }
    try:
        return snooper_class[type]
    except KeyError as _:
        raise Exception(f"snooping algorithm {type} not supported.")


def get_qairt_snooping_cmd_option_class(snooper: str):
    '''
    Method to find the Snooper class given snooper type.

    :param snooper: snooper type. Following snooping algos are supported: [binary, layerwise,
     cumulative-layerwise, oneshot-layerwise]
    :return: class for the given snooper type
    :raises: Exception if snooper type is not supported.
    '''
    from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_binary_snooping_cmd_options\
        import QAIRTBinarySnoopingCmdOptions
    from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_layerwise_cmd_options\
        import QAIRTLayerwiseCmdOptions
    from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_cumulative_layerwise_cmd_options\
        import QAIRTCumulativeLayerwiseCmdOptions
    from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_oneshot_layerwise_cmd_options\
        import QAIRTOneshotLayerwiseSnoopingCmdOptions

    option_class = {
        'binary': QAIRTBinarySnoopingCmdOptions,
        'layerwise': QAIRTLayerwiseCmdOptions,
        'cumulative-layerwise': QAIRTCumulativeLayerwiseCmdOptions,
        'oneshot-layerwise': QAIRTOneshotLayerwiseSnoopingCmdOptions
    }

    try:
        return option_class[snooper]
    except KeyError as _:
        raise Exception(f"snooping algorithm {snooper} not supported.")


def append_to_intermediate_report(file_name, layer_name, cur_layer_out_name, percent_match):
    """
        This method saves intermediate results in csv for user to check progress
        while the algorithm is running.
        Args:
            layer_name : name of node
            cur_layer_out_name : output tensor of the node
            percent_match : metric output
            file_name : location of the output file
    """
    temp_df = pd.DataFrame({
        "Debug Layer Name": [layer_name],
        "Tensor Output Name": [cur_layer_out_name],
        "Match Percentage": [percent_match]
    })
    temp_df.to_csv(file_name, mode="a", header=not os.path.exists(file_name), index=False)


def replace_special_chars(name):
    """
        This method replaces special characters with underscore in supplied name
        Args:
            name : name of node
        Returns:
            name : modified name after replacing special characters
        """
    kdict = {':': '_', '/': '_', '-': '_'}
    for key in kdict:
        name = name.replace(key, kdict[key])
    return name


def show_progress(total_count, cur_count, info='', key='='):
    """Displays the progress bar."""
    completed = int(round(80 * cur_count / float(total_count)))
    percent = round(100.0 * cur_count / float(total_count), 1)
    bar = key * completed + '-' * (80 - completed)

    sys.stdout.write('[%s] %s%s (%s)\r' % (bar, percent, '%', info))
    sys.stdout.flush()


class LayerStatus():
    LAYER_STATUS_SUCCESS = ''
    LAYER_STATUS_CON_ERROR = 'err_con'
    LAYER_STATUS_LIB_ERROR = 'err_lib'
    LAYER_STATUS_CNTX_ERROR = 'err_cntx'
    LAYER_STATUS_EXEC_ERROR = 'err_exec'
    LAYER_STATUS_PARTITION_ERR = 'err_part'
    LAYER_STATUS_SKIPPED = 'skip'
    LAYER_STATUS_PARTITION = ' part'
    LAYER_STATUS_COMPARE_ERROR = 'err_compare'


def files_to_compare(framework_path, inference_path, cur_layer_out_name, d_type, logger,
                     out_folder=None):
    """This method returns the file paths to compare.

    Args:
        cur_layer_out_name  : output name of layer
        framework_path      : path to the reference framework results
        inference_path      : path to the qnn inference results
        d_type              : datatype of the layer
        out_folder          : name of output folder of QNN
    Returns:
        inf_path : path of output file from qnn platform
        rt_path  : path of output file from reference platform
    """
    rt_raw = None
    inf_raw = None
    sanitized_cur_layer_out_name = santize_node_name(cur_layer_out_name)
    folder_name = out_folder if out_folder else sanitized_cur_layer_out_name
    rt_path = os.path.join(framework_path, sanitized_cur_layer_out_name + '.raw')
    if os.path.exists(rt_path):
        rt_raw = np.fromfile(rt_path, dtype=d_type)

    inf_folder = os.path.join(inference_path, 'inference_engine', folder_name, 'output/Result_0')
    inf_path = os.path.join(inf_folder, sanitized_cur_layer_out_name + '.raw')
    inf_native_path = os.path.join(inf_folder, sanitized_cur_layer_out_name + '_native.raw')
    if os.path.exists(inf_native_path):
        inf_path = inf_native_path

    if os.path.exists(inf_path):
        inf_raw = np.fromfile(inf_path, dtype=d_type)
    logger.debug('compare files inf_path : {} \t rt_path : {}'.format(inf_path, rt_path))

    return inf_raw, rt_raw


def files_to_compareV2(reference_raw_path: str, target_raw_path: str, d_type: str,
                       logger: Logger) -> tuple[np.ndarray, np.ndarray]:
    """This method returns the file paths to compare.

    Args:
        reference_raw_path  : path to the reference_raw path
        target_raw_path     : path to the target_raw path
        d_type              : datatype of the layer
        logger              : logger
    Returns:
        inf_path : path of output file from qnn platform
        rt_path  : path of output file from reference platform
    """
    if os.path.exists(reference_raw_path):
        reference_raw = np.fromfile(reference_raw_path, dtype=d_type)
    else:
        logger.info("Reference raw path {} does not exist".format(reference_raw_path))

    target_raw_native_path = target_raw_path.replace('.raw', '_native.raw')
    if os.path.exists(target_raw_native_path):
        target_raw = np.fromfile(target_raw_native_path, dtype=d_type)
    elif os.path.exists(target_raw_path):
        target_raw = np.fromfile(target_raw_path, dtype=d_type)
    else:
        logger.info("Target raw path {} does not exist".format(target_raw_path))
    logger.debug('compare files inf_path : {} \t rt_path : {}'.format(reference_raw_path,
                                                                      target_raw_path))

    return target_raw, reference_raw


class ActivationStatus:
    SKIP = "SKIP"
    CONVERTER_FAILURE = "CONVERTER_FAILURE"
    QUANTIZER_FAILURE = "QUANTIZER_FAILURE"
    SNPE_DLC_GRAPH_PREPARE_FAILURE = "SNPE_DLC_GRAPH_PREPARE_FAILURE"
    QNN_CONTEXT_BINARY_FAILURE = "QNN_CONTEXT_BINARY_FAILURE"
    SNPE_NET_RUN_FAILURE = "SNPE_NET_RUN_FAILURE"
    QNN_NET_RUN_FAILURE = "QNN_NET_RUN_FAILURE"
    CUSTOM_OVERRIDE_GENERATION_FAILURE = "CUSTOM_OVERRIDE_GENERATION_FAILURE"
    SUCCESS = "SUCCESS"
    SO_FAR_SO_GOOD = "SO_FAR_SO_GOOD"

    def __init__(self, activation_name, msg="initialize") -> None:
        self._current_status = ActivationStatus.SO_FAR_SO_GOOD
        self._msg = msg
        self._activation_name = activation_name

    def set_status(self, status, msg):
        self._current_status = status
        self._msg = msg

    def get_status(self):
        return self._current_status

    def get_msg(self):
        return self._msg


class SnooperUtils:
    """
    SnooperUtils class contains all configuration parameters supplied by user
    To use:
    >>> config = SnooperUtils.getInstance()
    """
    __instance = None

    def __init__(self, args):
        if SnooperUtils.__instance is not None:
            raise ConfigError('instance of SnooperUtils already exists')
        else:
            SnooperUtils.__instance = self

        self._config = args
        self._transformer = None
        self._traverser = None
        self.updated_name_map = None
        self._framework_ins = None
        self._comparator = None

    @classmethod
    def clear(cls):
        if cls.__instance is not None:
            cls.__instance = None

    def __str__(self):
        return str(self._config)

    @classmethod
    def getInstance(cls, args=None):
        if cls.__instance is None:
            cls.__instance = SnooperUtils(args)
        if args is not None:
            cls.__instance.args = args
        return cls.__instance

    def getStartLayer(self):
        if not self._config.start_layer:
            return None
        # start_layer = replace_special_chars(self._config.start_layer)
        start_layer = self._config.start_layer
        if not self.updated_name_map or start_layer not in self.updated_name_map:
            return start_layer
        else:
            #This is used only for Caffe due to output name change made by caffe_transform
            return self.updated_name_map[start_layer]

    def getEndLayer(self):
        if not self._config.end_layer:
            return None
        # end_layer = replace_special_chars(self._config.end_layer)
        end_layer = self._config.end_layer
        if not self.updated_name_map or end_layer not in self.updated_name_map:
            return end_layer
        else:
            #This is used only for Caffe due to output name change made by caffe_transform
            return self.updated_name_map[end_layer]

    def getModelTraverserInstance(self):
        """This method returns the appropriate ModelTraverser class instance
        Returns the same instance each time."""
        return self._framework_ins

    def setModelTraverserInstance(self, logger, args, model_path=None, add_layer_outputs=[],
                                  add_layer_types=[], skip_layer_outputs=[], skip_layer_types=[]):
        """This method returns the appropriate ModelTraverser class instance
        Returns the same instance each time."""
        from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_runner import ModelTraverser
        if model_path:
            args.model_path = model_path
        if hasattr(args, "disable_graph_optimization"):
            framework_args = Namespace(framework=args.framework, version=None,
                                       model_path=args.model_path, output_dir=args.output_dir,
                                       disable_graph_optimization=args.disable_graph_optimization)
        else:
            framework_args = Namespace(framework=args.framework, version=None,
                                       model_path=args.model_path, output_dir=args.output_dir)
        self._framework_ins = ModelTraverser(logger, framework_args,
                                             add_layer_outputs=add_layer_outputs,
                                             add_layer_types=add_layer_types,
                                             skip_layer_outputs=skip_layer_outputs,
                                             skip_layer_types=skip_layer_types)
        return self._framework_ins

    def getFrameworkInstance(self):
        """This method returns the appropriate FrameworkRunner class instance
        Returns the same instance each time."""
        return self._framework_ins

    def setFrameworkInstance(self, logger, args, model_path=None):
        """This method returns the appropriate FrameworkRunner class instance
        Returns the same instance each time."""
        from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_runner import FrameworkRunner
        if model_path:
            args.model_path = model_path
        framework_args = Namespace(framework=args.framework, version=None,
                                   model_path=args.model_path, output_dir=args.output_dir,
                                   add_layer_outputs=[])
        self._framework_ins = FrameworkRunner(logger, framework_args)
        self._framework_ins.load_framework()
        return self._framework_ins

    def getComparator(self, tol_thresolds=None):
        """Returns the list of configured verifiers."""

        verifier_objects = []
        for verifier in self._config.default_verifier:
            verifier = verifier[0].split(',')
            verifier_name = verifier[0]
            try:
                verifier_config = {}
                ret, verifier_config = VerifierFactory().validate_configs(
                    verifier_name, verifier[1:])
                if not ret:
                    errormsg = str(verifier_config['error']) if 'error' in verifier_config else ''
                    raise VerifierError("VerifierFactory config_verify error: " + errormsg)

                verifier_obj = VerifierFactory().factory(verifier_name, verifier_config)
                if verifier_obj is None:
                    raise VerifierError(
                        get_message('ERROR_VERIFIER_INVALID_VERIFIER_NAME')(verifier_name))

                verifier_objects.append(verifier_obj)
            except Exception as err:
                raise Exception(
                    f"Error occurred while configuring {verifier_name} verifier. Reason: {err}")

        self._comparator = verifier_objects
        return self._comparator


class ActivationInfo:

    def __init__(self, dtype: str, shape: list, distribution: tuple) -> None:
        '''
        Initializes the ActivationInfo class

        :param dtype: data type of the activation
        :param shape: shape of the activation
        :param distribution: min, max, median info of the activation
        '''
        self.dtype = dtype
        self.shape = shape
        self.distribution = distribution


def dump_csv(data_frame: dict, columns: list, csv_path: str) -> None:
    '''
    Export data_frame to csv file with given columns

    :param data_frame: data frame dictionary
    :param columns: list of columns
    :param csv_path: path to csv file
    '''

    df = pd.DataFrame(data_frame, columns=columns)
    df.to_csv(csv_path, sep=',', index=False, header=True)


def sanitize_encoding(encodings):
    sanitized_encodings = []
    for enc in encodings:
        if 'is_symmetric' in enc:
            enc["is_symmetric"] = str(enc["is_symmetric"])
        sanitized_encodings.append(enc)
    return sanitized_encodings


def handle_boolean_params_in_encodings(quantization_overrides_file_path):
    if quantization_overrides_file_path is None:
        return

    quantization_overrides = read_json(quantization_overrides_file_path)
    for key in quantization_overrides["activation_encodings"]:
        quantization_overrides['activation_encodings'][key] = sanitize_encoding(
            quantization_overrides['activation_encodings'][key])

    for key in quantization_overrides['param_encodings']:
        quantization_overrides['param_encodings'][key] = sanitize_encoding(
            quantization_overrides['param_encodings'][key])
    dump_json(quantization_overrides, quantization_overrides_file_path)
