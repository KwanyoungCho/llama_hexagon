# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
import re
import sys
import json
import numpy as np
import shutil
import signal
from collections import OrderedDict

from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import SnoopingError
from qti.aisw.accuracy_debugger.lib.runner.component_runner import exec_inference_engine
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ConfigError, SnoopingError
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name, get_absolute_path,format_args
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, transpose_to_nhwc
from qti.aisw.accuracy_debugger.lib.utils.snooper_utils import files_to_compare, LayerStatus
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_get_tensor_mapping import TensorMapper
from qti.aisw.accuracy_debugger.lib.utils.snooper_utils import SnooperUtils as su
from qti.aisw.accuracy_debugger.lib.compare_encodings.compare_encodings_runner import CompareEncodingsRunner
from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_runner import ModelTraverser
from qti.aisw.accuracy_debugger.lib.utils.common import append_arg

def signal_handler(sig, _):
    Snooper.logger.info('Stopping snooping on user request.')
    if Snooper.stop:
        Snooper.logger.info('Waiting for current layer to complete.')
        Snooper.stop = True
    else:
        sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


class Snooper:
    stop = False
    logger = None

    def __init__(self, snooping_type, args, logger, verbose="info"):

        self.snooping_type = snooping_type
        self.args = args
        self.logger = logger if logger else setup_logger(verbose, args.output_dir)
        Snooper.logger = self.logger
        self.input_list_file = args.input_list
        self.model = args.model_path
        self.engine_path = args.engine_path
        self.deviceId = args.deviceId
        self.engine = args.engine
        self.framework = args.framework
        self.framework_version = None
        self.runtime = args.runtime
        self.framework_results = args.golden_output_reference_directory
        self.work_dir = args.working_dir
        self.output_dir = args.output_dir
        self.model_traverser = None
        self.model_handler = None
        self.target_device = args.target_device
        self.host_device = args.host_device
        self.architecture = args.architecture
        self.precision = args.precision
        self.compiler_config = args.compiler_config
        self.profile_info = None
        self.extra_converter_args = args.extra_converter_args
        self.extra_runtime_args = args.extra_runtime_args
        self.remote_server = args.remote_server
        self.remote_username = args.remote_username
        self.remote_password = args.remote_password
        self.act_quantizer = args.act_quantizer
        self.param_quantizer = args.param_quantizer
        self.bias_bitwidth = args.bias_bitwidth
        self.weights_bitwidth = args.weights_bitwidth
        self.act_bitwidth = args.act_bitwidth
        self.quantization_overrides = args.quantization_overrides
        self.algorithms = args.algorithms
        self.ignore_encodings = args.ignore_encodings
        self.per_channel_quantization = args.per_channel_quantization
        self.add_layer_outputs = args.add_layer_outputs
        self.add_layer_types = args.add_layer_types
        self.use_native_input_files = args.use_native_input_files
        self.use_native_output_files = args.use_native_output_files
        self.symbol_data_map = self.define_symbols()
        self.is_transpose_needed_dict = {}

    def define_symbols(self):
        """Populate symbol_data_map with mappings of onnx model symbols with
        corresponding data which is provided by user in extra_converter_args.
        Symbols are defined in the extra_converter_args as following examples:

        'define_symbol batch_size = 1'
        'define_symbol seq_len 128'

        """
        symbol_data_map = {}
        if self.extra_converter_args:
            converter_args = self.extra_converter_args.split(';')
            for arg in converter_args:
                arg = " ".join(arg.split())
                split_arg = re.split(' |=', arg)
                if split_arg[0] == "define_symbol":
                    if len(split_arg) == 3:
                        symbol, data = split_arg[1], split_arg[2]
                        symbol_data_map[symbol] = data
                    else:
                        raise ConfigError(
                            'Symbols are not defined correctly in extra_converter_args.')

        return symbol_data_map

    def get_input_tensors(self, list_file, model=None):
        input_tensors = []
        with open(list_file, 'r') as f:
            input_paths = f.readline().rstrip().split(' ')

        # get input_layers of extracted model by creating a new handler for it
        from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_runner import FrameworkRunner
        framework_args = Namespace(None, framework=self.framework, version=self.framework_version,
                                   model_path=model, output_dir=self.output_dir, engine=self.engine)
        model_handler = FrameworkRunner(self.logger, framework_args)
        model_handler.load_framework()
        input_layers = model_handler.framework_instance.get_input_layers()

        for i, item in enumerate(input_layers):
            if i >= len(input_paths):
                break
            dim_str = ""
            sanitized_node = santize_node_name(item[0])
            # checking if Tensor in profile info
            if sanitized_node in self.profile_info:
                # if yes use profile info to extract dim
                dim_str = str(self.profile_info[sanitized_node][1]).replace(' ', '')[1:-1]
            else:
                # else use older method
                dim_str = str(item[2])
                for symbol, data in self.symbol_data_map.items():
                    # Need to handle any quotes present in the dimensions
                    # for example item[2] will look like ['batch_size', 'seq_len']
                    symbol = "'" + symbol + "'"
                    dim_str = dim_str.replace(symbol, data)
                dim_str = dim_str.replace(' ', '')[1:-1]
            if ":=" in input_paths[i]:
                input_paths[i] = input_paths[i].split(":=")[1]
            # input_data format is (name, dims, path, datatype)
            input_data = (item[0], dim_str, input_paths[i], item[1].__name__)
            input_tensors.append(input_data)
        return input_tensors

    def update_list_file(self, graph_inputs, initial_run=False):
        """Create a new input list file (temp-list.txt) based on the given
        input names.
        Args:
            graph_inputs: The new inputs of the extracted subgraph
            initial_run: Indicate if its initial model partition i.e initial run
        """
        updated_input_list = []
        handleInputNames = False
        # check is needed for caffe
        if isinstance(graph_inputs, dict):
            input_map = graph_inputs.copy()
            handleInputNames = True
            graph_inputs = list(graph_inputs.values())
            os.makedirs(self.work_dir + '/temp_inp/', exist_ok=True)

        for ip in graph_inputs:
            if ip in self.original_input_names_raw_map:
                updated_input_list.append(ip + ':=' + self.original_input_names_raw_map[ip].strip())
            else:
                s_ip = santize_node_name(ip)
                inp_path = os.path.join(self.framework_results, s_ip + '.raw')

                if handleInputNames:
                    # move req input files to temp folder
                    dst_path = self.work_dir + '/temp_inp/' + list(input_map.keys())[list(
                        input_map.values()).index(ip)] + '.raw'
                    try:
                        shutil.copy(inp_path, dst_path)
                        self.logger.debug('copied file {} to {}'.format(inp_path, dst_path))
                        inp_path = dst_path
                    except:
                        inp_path = self.work_dir + '/temp_inp/' + list(input_map.keys())[list(
                            input_map.values()).index(ip)] + '.raw'
                updated_input_list.append(ip + ':=' + inp_path)

        # creating new input-list-file for extracted model. If its initial model extraction, \
        # the inputs are stored in a different file i.e temp-initial-list which will not be replaced till the snooping ends.
        # For any subsequent runs, the inputs are stored in temp-list file which will be replaced after each run.
        file_name = 'temp-list.txt'
        if initial_run:
            file_name = 'temp-initial-list.txt'
        list_file = os.path.join(self.output_dir, file_name)
        if len(updated_input_list) > 0:
            with open(list_file, "w") as f:
                f.write(' '.join(updated_input_list))
        return list_file

    def initiate_model_extraction(self, model, start_layer=None, end_layer=None, set_model=True,
                                  initial_run=False):
        """
        This method partitions the model at start layer output till end layer and generates
        updated input list file
        Args:
            model : path to the model which needs to be partitioned
        Returns:
            status          : True if success
            model           : path to partitioned model
            list_file       : input list file for partitioned model
            new_g_inputs    : list of new inputs of partitioned model
        """
        s_utility = su.getInstance()
        self.model_handler = s_utility.getFrameworkInstance()

        # populate original_input_names_raw_map needed for end layer extraction.
        if set_model:
            start_layer = s_utility.getStartLayer()
            end_layer = s_utility.getEndLayer()
        valid_layers = [item[1] for item in self.model_traverser._layerlist]
        # check if valid layers are provided as start/end layers
        if start_layer and start_layer not in valid_layers:
            raise ConfigError('{} is not present in {}. Please provide valid start_layer'.format(
                start_layer, model))
        if end_layer and end_layer not in valid_layers:
            raise ConfigError('{} is not present in {}. Please provide valid end_layer'.format(
                end_layer, model))

        list_file = self.input_list_file
        original_input_names = self.model_traverser.framework_instance.get_input_layers(
            names_only=True)

        with open(list_file, 'r') as F:
            file_items = F.readline().strip().split(' ')
            file_paths = [f_path.split(':=')[-1] for f_path in file_items]
            self.original_input_names_raw_map = dict(zip(original_input_names, file_paths))
        (ret_status, model,
         new_g_inputs) = self.model_handler.extract_sub_graph(start_layer, end_layer,
                                                              self.output_dir)

        if not ret_status:
            return False, None, None, None
        # create input list file for partitioned model
        list_file = self.update_list_file(new_g_inputs, initial_run)

        return True, model, list_file, new_g_inputs

    def handle_qnn_run_failure(self, std_out, cur_layer_out_name, layer_status_map, conv_fail_nodes,
                               lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes):
        """
        This method handles the compilation and execution failures of qnn run
        Args:
            std_out             : output of qnn inference engine
            cur_layer_out_name  : output name of layer
            layer_status_map    : dict that gives status of each layer
            conv_fail_nodes     : list of qnn converter fail layers
            lib_fail_nodes      : list of qnn lib-generator fail layers
            cntx_fail_nodes     : list of qnn context binary generator fail layers
            exec_fail_nodes     : list of qnn net-run fail layers
        Returns:
            conv_fail_nodes     : updated list of qnn converter fail layers
            lib_fail_nodes      : updated list of qnn lib-generator fail layers
            cntx_fail_nodes     : updated list of qnn context binary generator fail layers
            exec_fail_nodes     : updated list of qnn net-run fail layers
            layer_status_map    : updated dict that gives status of each layer
        """
        s_cur_layer_out_name = santize_node_name(cur_layer_out_name)
        if 'Failed to do initial conversion of model' in std_out:
            # handles qnn_converter failure
            conv_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_CON_ERROR

        elif 'model binaries failed to be created' in std_out:
            # handles qnn_lib_generator failure
            lib_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_LIB_ERROR

        elif 'The context binary failed to be created' in std_out:
            # handles qnn_context_bin_gen failure
            cntx_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_CNTX_ERROR

        elif 'Failed to execute inference' in std_out:
            # handles qnn_net_run failure
            exec_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_EXEC_ERROR
        return conv_fail_nodes, lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes, layer_status_map

    def execute_on_qnn(self, model=None, list_file=None, output_tensors=None,
                       orig_model_outputs=None, out_dir=None, capture_intermediate_outputs=False,
                       float_fallback=False):
        """This method executes the given model on qnn platform.

        Args:
            model                           : path of the model
            list_file                       : file containing input paths to model
            output_tensors                  : output node names of model
            out_dir                         : output folder name inside work directory
            capture_intermediate_outputs    : boolean flag to save intermediate outputs of model
        Returns:
            ret_status                      : status of qnn execution
            std_out                         : console output of qnn inference engine
        """
        model = model if model else self.model
        list_file = list_file if list_file else self.input_list_file
        input_tensors = self.get_input_tensors(list_file, model)
        extra_converter_list = []
        extra_netrun_list = []
        args = {
            'framework':
            '{} {}'.format(self.framework,
                           (self.framework_version if self.framework_version else '')),
            'engine_path':
            self.engine_path,
            'runtime':
            self.runtime,
            'working_dir':
            os.path.join(self.output_dir),
            'output_dirname':
            out_dir,
            'input_list':
            list_file,
            'deviceId':
            self.deviceId,
            'host_device':
            self.host_device,
            'target_device':
            self.target_device,
            'model_path':
            model,
            'model_inputs':
            ''.join([' --input_tensor ' + ' '.join(item) for item in input_tensors]),
            'model_outputs':
            ''.join([' --output_tensor {}'.format(name) for name in orig_model_outputs]),
            'target_architecture':
            self.architecture,
            'precision':
            self.precision,
            'extra_converter_args':
            self.extra_converter_args,
            'extra_runtime_args':
            self.extra_runtime_args,
            'verbose': (' -v' if self.logger.level == logging.DEBUG else ''),
            'act_quantizer':
            self.act_quantizer,
            'param_quantizer':
            self.param_quantizer,
            'bias_bitwidth':
            self.bias_bitwidth,
            'weights_bitwidth':
            self.weights_bitwidth,
            'act_bitwidth':
            self.act_bitwidth,
            'quantization_overrides':
            self.quantization_overrides,
            'algorithms':
            self.algorithms,
            'ignore_encodings':
            self.ignore_encodings,
            'per_channel_quantization':
            self.per_channel_quantization,
            'remote_server':
            self.remote_server,
            'remote_username':
            self.remote_username,
            'remote_password':
            self.remote_password,
        }

        inference_args = (' --framework {args[framework]}'
                          ' --engine_path {args[engine_path]}'
                          ' --runtime {args[runtime]}'
                          ' --working_dir {args[working_dir]}'
                          ' --output_dirname {args[output_dirname]}'
                          ' --input_list {args[input_list]}'
                          ' --deviceId {args[deviceId]}'
                          ' --host_device {args[host_device]}'
                          ' --model_path {args[model_path]}'
                          ' --architecture {args[target_architecture]}'
                          ' --precision {args[precision]}'
                          ' --remote_server {args[remote_server]}'
                          ' --remote_username {args[remote_username]}'
                          '{args[model_inputs]}'
                          '{args[model_outputs]}'
                          '{args[verbose]}').format(args=args)

        quantization_args = (' --act_quantizer {args[act_quantizer]}'
                             ' --param_quantizer {args[param_quantizer]}'
                             ' --bias_bitwidth {args[bias_bitwidth]}'
                             ' --weights_bitwidth {args[weights_bitwidth]}'
                             ' --act_bitwidth {args[act_bitwidth]}').format(args=args)

        if self.remote_password:
            inference_args += ' --remote_password ' + self.remote_password

        if self.use_native_input_files:
            inference_args += ' --use_native_input_files'
        if self.use_native_output_files:
            inference_args += ' --use_native_output_files'

        if self.runtime.startswith('dsp') or self.runtime in ['aic', 'htp']:
            inference_args += ' --offline_prepare'

        if not capture_intermediate_outputs:
            # Used only for cumulative layerwise because it adds output node at current layer also
            if output_tensors is not None:
                args['add_layer_outputs'] = ','.join(['{}'.format(name) for name in output_tensors])
                inference_args += ' --add_layer_outputs {args[add_layer_outputs]}'.format(args=args)
            inference_args += ' --debug_mode_off'

        if self.precision in ['int8', 'fp16'] and self.compiler_config:
            inference_args += ' --compiler_config ' + self.compiler_config

        if self.quantization_overrides:
            quantization_args += ' --quantization_overrides ' + self.quantization_overrides
        if self.algorithms: quantization_args += ' --algorithms ' + self.algorithms
        if self.ignore_encodings:
            quantization_args += ' --ignore_encodings {args[ignore_encodings]}'.format(args=args)
        if self.per_channel_quantization:
            quantization_args += ' --per_channel_quantization {args[per_channel_quantization]}'.format(
                args=args)
        if float_fallback:
            quantization_args += " --float_fallback"
        inference_args += quantization_args
        if self.extra_converter_args:
            extra_converter_list = ['--extra_converter_args', self.extra_converter_args]
        if self.extra_runtime_args:
            extra_netrun_list = ['--extra_runtime_args', self.extra_runtime_args]

        # Execute model on QNN
        self.logger.info("Running exec_inference_engine with parameters: {}".format(
            inference_args + ' ' + ' '.join(extra_converter_list + extra_netrun_list)))
        try:
            #TODO: Need to use Python API call once available
            all_args = inference_args.split() + extra_converter_list + extra_netrun_list
            exec_inference_engine(all_args, self.engine, self.logger)
        except Exception as e:
            self.logger.info(str(e))
            return 1, str(e)
        return 0, ''

    def partition_initial_model(self, model):
        s_utility = su.getInstance(self.args)

        if s_utility.getStartLayer() or s_utility.getEndLayer():
            self.set_profile_info(model)
            status, model, list_file, _ = self.initiate_model_extraction(model, initial_run=True)
            if status is False:
                return status, None, None
            # keep a copy of extracted model as there is chance of replacement due to partitions
            if os.path.exists(os.path.join(self.work_dir, 'cleaned')) and os.path.isdir(
                    os.path.join(self.work_dir, 'cleaned')):
                if os.path.exists(model):
                    model_dir = os.path.dirname(model)
                    if not os.path.exists(
                            os.path.join(
                                self.output_dir, 'transformed' +
                                self.model_traverser.framework_instance.FRAMEWORK_SUFFIX)):
                        os.makedirs(
                            os.path.join(
                                self.output_dir, 'transformed' +
                                self.model_traverser.framework_instance.FRAMEWORK_SUFFIX))
                    for path in os.listdir(model_dir):
                        shutil.copy(
                            os.path.join(model_dir, path),
                            os.path.join('cleaned', 'cleanmodel' + os.path.splitext(path)[1]))
            else:
                if os.path.exists(model):
                    shutil.copy(
                        model,
                        os.path.join(
                            self.output_dir, 'cleanmodel' +
                            self.model_traverser.framework_instance.FRAMEWORK_SUFFIX))
            model = os.path.join(
                self.output_dir,
                'cleanmodel' + self.model_traverser.framework_instance.FRAMEWORK_SUFFIX)
        else:
            list_file = self.input_list_file

        return True, model, list_file

    def set_profile_info(self, model, list_file=None):
        """
        Create and set profile info of model.
        """
        s_utility = su.getInstance(self.args)
        self.model_handler = s_utility.setFrameworkInstance(self.logger, self.args, model)
        self.model_traverser = s_utility.setModelTraverserInstance(self.logger, self.args, model,
                                                                   self.add_layer_outputs,
                                                                   self.add_layer_types)

        original_output_names = self.model_handler.framework_instance.get_output_layers(
            names_only=True)
        original_input_names = self.model_handler.framework_instance.get_input_layers(
            names_only=True)

        if not list_file:
            list_file = self.input_list_file

        with open(list_file, 'r') as F:
            file_items = F.readline().strip().split(' ')
            file_paths = [f_path.split(':=')[-1] for f_path in file_items]
            self.original_input_names_raw_map = dict(zip(original_input_names, file_paths))

        # get profile info like tensor dimensions, dtype, min, max and median values
        profile_path = os.path.join(self.framework_results, 'profile_info.json')
        temp_profile_path = os.path.join(self.work_dir, 'temp', 'profile_info.json')
        if not os.path.exists(profile_path) and not os.path.exists(temp_profile_path):
            inputs = self.model_handler.framework_instance.get_input_layers()
            for idx, ip in enumerate(inputs):
                input_dim_str = ','.join(str(d) for d in ip[2])
                inputs[idx] = (ip[0], input_dim_str,
                               self.original_input_names_raw_map[inputs[idx][0]], ip[1])
            self.model_handler.generate_intermediate_outputs(os.path.join(self.work_dir,
                                                                          'temp'), input=inputs,
                                                             output=original_output_names)
            profile_path = os.path.join(self.work_dir, 'temp', 'profile_info.json')
        profile_info = read_json(profile_path)
        self.profile_info = profile_info

    def _set_input_list(self, input_list):
        """
        This function converts relative paths of raw files given in the input list to absolute paths
        and creates new input list with these paths.
        Args:
            input_list : The input list provided by the user
        Returns:
            prepend_input_list : New input list having absolute paths for the input tensors
        """
        curr_dir = os.path.dirname(input_list)

        # get original input list paths
        #_original_input_paths stores all rel input paths in form of list of lists;
        # ie. if a input list has 2 batch and each batch require 3 inputs then
        # the _original_input_paths would look like:
        # [[batch1_input1,batch1_input2,batch1_input3],[batch2_input1,batch2_input2,batch2_input3]]

        with open(input_list, "r") as input_list:
            self._original_input_paths = []
            for line in input_list.readlines():
                if line.startswith("#"):
                    continue
                else:
                    #This assumes per batch input is separated by either comma or space
                    self._original_input_paths.append(re.split(' ,|, |,| ', line.strip(' \n')))

        # for each item in each line of _original_input_paths, make it an absolute path
        self._full_path_input_paths = None
        self._full_path_input_paths = [[
            get_absolute_path(rel_path, checkExist=True, pathPrepend=curr_dir) if ":=" not in rel_path \
            else rel_path.split(":=")[0]+":="+ get_absolute_path(rel_path.split(":=")[1], \
            checkExist=True, pathPrepend=curr_dir) for rel_path in per_batch] for per_batch in \
            self._original_input_paths]

        # create a new input_list_file in the output_dir and use that
        prepend_input_file_dump_path = os.path.join(self.args.output_dir, "prepended_inputs")
        os.makedirs(prepend_input_file_dump_path, exist_ok=True)

        # Create a new input list file in the dump directory
        prepend_input_list_file_path = os.path.join(prepend_input_file_dump_path, 'input_list.txt')
        with open(prepend_input_list_file_path, "w") as input_list:
            input_list.write('\n'.join(
                [' '.join(per_batch) for per_batch in self._full_path_input_paths]))

        return prepend_input_list_file_path

    def _convert_tensors(self, input_list, user_provided_dtypes):
        """
        This function converts all the tensors present in input_list such that they will be 
        supported by converter. The converted tensors are dumped into new files.The paths of the new
        input tensors are stored in a list file created inside converted_inputs directory
        Args:
            input_list: input list provided
            user_provided_dtypes: List containing the datatypes of input tensors
        Returns:
            Path to the new input list file
        """
        # Create a directory to dump the converted input files
        converted_input_file_dump_path = os.path.join(self.args.output_dir, "converted_inputs")
        os.makedirs(converted_input_file_dump_path, exist_ok=True)

        # Create a new input list file in the dump directory
        new_input_list_file_path = os.path.join(converted_input_file_dump_path, 'input_list.txt')

        # Open the original and new input list files
        with open(input_list, 'r') as old_file, open(new_input_list_file_path, 'w') as new_file:
            # Iterate over each line in the original input list file
            for line in old_file:
                line = line.strip().split()
                if line:
                    new_file_name_and_path = []
                    # Iterate over each file name and path in the line
                    for user_provided_dtype, file_name_and_path in zip(user_provided_dtypes, line):
                        file_name_and_path = file_name_and_path.split(
                            ':=') if ':=' in file_name_and_path else [None, file_name_and_path]
                        # Load the tensor from the file
                        user_provided_tensor = np.fromfile(file_name_and_path[1],
                                                           dtype=user_provided_dtype)
                        # Convert the tensor to 32 bit if necessary
                        converted_tensor = user_provided_tensor.astype(
                            np.int32 if user_provided_dtype == "int64" else np.
                            float32 if user_provided_dtype == "float64" else user_provided_dtype)
                        # Save the converted tensor to a new file in the dump directory
                        file_name = os.path.join(converted_input_file_dump_path,
                                                 os.path.basename(file_name_and_path[1]))
                        converted_tensor.tofile(file_name)
                        # Add the new file name and path to the new line
                        new_file_name_and_path.append((file_name_and_path[0] +
                                                       ":=" if file_name_and_path[0] else "") +
                                                      file_name)
                    # Write the new line to the new input list file
                    new_file.write(" ".join(new_file_name_and_path) + "\n")

        # Update new input list file path
        return new_input_list_file_path

    def get_user_provided_dtypes(self, input_tensors):
        """
        This function collects the datatype provided in the input tensors
        Args:
            input_tensors: Input tensors provided as list/tuple
        Returns:
            user_provided_dtypes: List of dtypes provided in the input tensors
        """
        user_provided_dtypes = []
        if input_tensors is not None:
            # get proper input_tensor format
            for tensor in input_tensors:
                if len(tensor) == 3:
                    user_provided_dtypes.append('float32')
                elif len(tensor) == 4:
                    user_provided_dtypes.append(tensor[-1])
                tensor_list = list(tensor)
                tensor_list[2] = get_absolute_path(
                    tensor_list[2], pathPrepend=os.path.dirname(self.input_list_file))
                tensor = tuple(tensor_list)
        return user_provided_dtypes

    def validate_args(self,user_provided_dtypes):
        converted_input_list = self.input_list_file
        if (self.args.runtime != 'aic' and user_provided_dtypes):
            converted_input_list = self._convert_tensors(self.input_list_file, user_provided_dtypes)
        if self.args.extra_converter_args:
                converter_ignore_list = [
                    'input_network', 'input_dim', 'out_node', 'output_path',
                    'quantization_overrides', 'input_list', 'param_quantizer', 'act_quantizer',
                    'weight_bw', 'bias_bw', 'act_bw', 'float_bias_bw',
                    'restrict_quantization_steps', 'algorithms', 'ignore_encodings',
                    'use_per_channel_quantization', 'disable_batchnorm_folding', 'i', 'b',
                    'act_quantizer_calibration', 'param_quantizer_calibration',
                    'act_quantizer_schema', 'param_quantizer_schema', 'percentile_calibration_value',
                    'preserve_io'
                ]
                self.args.extra_converter_args = format_args(self.args.extra_converter_args,
                                                               converter_ignore_list)
        if any(dtype in user_provided_dtypes for dtype in ['int64', 'float64', 'bool']) and \
                                    self.args.runtime == "aic":
            self.args.extra_converter_args = append_arg('--preserve_io datatype',
                                                          self.args.extra_converter_args)
        return converted_input_list

    def get_tensor_mapping(self):

        input_tensors = self.get_input_tensors(self.input_list_file, self.model)
        output_tensors = self.model_handler.framework_instance.get_output_layers(names_only=True)
        user_provided_dtypes = self.get_user_provided_dtypes(input_tensors)
        converted_input_list = self.validate_args(user_provided_dtypes)

        converter_params = {
            "model_path": self.model,
            "input_tensors": input_tensors,
            "output_tensors": output_tensors,
            "input_list_txt": converted_input_list,
            "quantization_overrides": self.args.quantization_overrides,
            "param_quantizer": self.args.param_quantizer,
            "act_quantizer": self.args.act_quantizer,
            "weight_bw": self.args.weights_bitwidth,
            "bias_bw": self.args.bias_bitwidth,
            "act_bw": self.args.act_bitwidth,
            "float_bias_bw": 32,
            "restrict_quantization_steps": None,
            "algorithms": self.args.algorithms,
            "ignore_encodings": self.args.ignore_encodings,
            "per_channel_quantization": self.args.per_channel_quantization,
            "act_quantizer_calibration": None,
            "param_quantizer_calibration": None,
            "act_quantizer_schema": None,
            "param_quantizer_schema": None,
            "percentile_calibration_value": None,
            "extra_converter_args": self.args.extra_converter_args,
            "float_fallback": False
        }

        # Run tensor mapping
        get_mapping_arg = Namespace(
            None, target_outputs_dir=None, framework=self.framework, version=self.framework_version,
            model_path=self.model, work_dir=self.output_dir, engine=self.engine,
            golden_outputs_dir=self.framework_results if self.framework_results else None,
            converter_params=converter_params, engine_path=self.engine_path,
            host_device=self.host_device)

        return TensorMapper(get_mapping_arg, self.logger).run()

    def extract_and_filter_encodings(self,encoding_path,model):
        """
        Helper function to extract and filter encodings from model_net.json file
        """

        model = model
        framework = self.args.framework
        framework_args = Namespace(framework=framework, version=None, model_path=model,
                                   output_dir=self.output_dir)
        _framework_ins = ModelTraverser(self.logger, framework_args)
        model_instance = _framework_ins.framework_instance
        #Get param tensor names from the model
        param_tensors = [initializer.name for initializer in model_instance.graph.initializer]
        #Get Activation tensor names from the model
        activation_tensors = _framework_ins.get_all_layers()
        model_net_json = os.path.join(encoding_path, "converted_model_net.json")
        if os.path.exists(model_net_json):
            #Get inputs tensor names from the model
            input_tensors = _framework_ins.framework_instance.get_input_layers()
            #Initialize a dictionary which maps sanitize tensor name to original tensor name
            sanitize_unsanitize_map = OrderedDict()
            for tensor in activation_tensors:
                sanitize_unsanitize_map[santize_node_name(tensor)] = tensor
            for tensor in param_tensors:
                sanitize_unsanitize_map[santize_node_name(tensor)] = tensor
            for (tensor_name, dtype, dim) in input_tensors:
                sanitize_unsanitize_map[santize_node_name(tensor_name)] = tensor_name
            compare_encodings_args = Namespace(input=model_net_json, output_dir=self.output_dir)
            encodings_class = CompareEncodingsRunner(self.logger, compare_encodings_args)
            # Extract the encodings from the model_net.json file
            encodings_from_model_net = encodings_class.extract_model_net_encodings()
            # Filter the encodings from the extracted encodings not present in the framework model
            new_encodings = encodings_class.filter_encodings(
                encodings_from_model_net, sanitize_unsanitize_map)
            extracted_encodings_path = os.path.join(encoding_path, "new_encodings.json")
            # Dump the filtered encodings in a json file
            with open(extracted_encodings_path, 'w') as json_write:
                json.dump(new_encodings, json_write, indent=4)
            return extracted_encodings_path
        else:
            return None