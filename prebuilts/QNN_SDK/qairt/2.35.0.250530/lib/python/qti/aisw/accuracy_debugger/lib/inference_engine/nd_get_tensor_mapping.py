# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
import os

from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_runner import FrameworkRunner
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine, Framework
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_progress_message
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.device.nd_device_factory import DeviceFactory
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Devices_list, Device_type, ComponentType
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError as ExecutionError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.inference_engine.configs import CONFIG_PATH
from qti.aisw.accuracy_debugger.lib.inference_engine.configs.nd_inference_engine_config import InferenceEngineConfig
from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import get_tensor_names_from_dlc
from qti.aisw.accuracy_debugger.lib.inference_engine.converters.exec_conversion_quantization.nd_get_conversion_quantization_cls import get_exec_conversion_quantization_cls
from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_objects import get_available_frameworks
from qti.aisw.accuracy_debugger.lib.utils.common import truncate_native_tag


class InferenceMapperWithUtility():
    """
    This class is responsible for getting the inference tensor names from the output directory.
    It supports different engines and provides utility functions for executing context binary utility.
    """

    def __init__(self, logger, args):
        """
        Initialize the runner with a logger and arguments. The output directory for inference, engine type,
        engine path, host device, and context binary path are set from the arguments.
        """
        self._logger = logger
        self.work_dir = args.work_dir
        self.engine = args.engine
        self.engine_path = args.engine_path
        self.host_device = args.host_device
        self.converted_ir_graph = args.converted_ir_graph
        self.converter_params = args.converter_params
        self.framework = args.framework
        # QAIRT snooping will pass converter_type as 'qairt'
        # Since QNN snooping uses qnn converter which is same as it's engine type, it will not pass converter_type
        self.converter_type = args.converter_type if hasattr(args, "converter_type") else self.engine

    def get_inference_tensor_names(self):
        """
        Get the names of the inference tensors based on the engine type.
        """

        tensor_names = self.get_tensor_names_from_ir_graph() if self.is_valid_converted_graph(
        ) else None
        if not tensor_names is None: return tensor_names

        self.execute_conversion()

        model_net_json_path = os.path.join(self.work_dir, 'converted_model_net.json')
        dlc_path = os.path.join(self.work_dir, 'converted_model.dlc')

        if os.path.exists(model_net_json_path):
            self.converted_ir_graph = model_net_json_path
        elif os.path.exists(dlc_path):
            self.converted_ir_graph = dlc_path
        else:
            raise FileNotFoundError("Converted model File Not Found")

        return self.get_tensor_names_from_ir_graph() if self.is_valid_converted_graph() else []

    def get_tensor_names_from_ir_graph(self):
        """
        Get the names of the inference tensors from the provided file.
        """
        tensor_names = None

        if self.converted_ir_graph.lower().endswith(".json"):
            tensor_names = self.get_tensor_names_from_json(self.converted_ir_graph)
        elif self.converted_ir_graph.lower().endswith(".dlc"):
            tensor_names = get_tensor_names_from_dlc(self.converted_ir_graph, sanitize_names=True)
        else:
            self._logger.warn(
                f"Converted IR graph file {self.converted_ir_graph} is not valid, hence conversion will be called to get IR graph file."
            )

        return tensor_names

    def is_valid_converted_graph(self):
        return self.converted_ir_graph and (self.converted_ir_graph.lower().endswith(".json")
                                            or self.converted_ir_graph.lower().endswith(".dlc"))

    def execute_conversion(self):

        converter_class = get_exec_conversion_quantization_cls(self.converter_type, "conversion")

        if self.converter_type in [Engine.QNN.value, Engine.SNPE.value]:
            converter = converter_class(self.framework, self.host_device, self.converter_params,
                                        self.work_dir, logger=self._logger, verbose="info",
                                        engine_path=self.engine_path)
        elif self.converter_type == Engine.QAIRT.value:
            converter = converter_class(self.host_device, self.converter_params,
                                        self.work_dir, logger=self._logger, verbose="info",
                                        engine_path=self.engine_path)
        else:
            raise Exception(
                f"Tensor mapping is not supported for given engine/sdk type {self.converter_type}")

        convert_method = None
        if self.converter_type == Engine.QNN.value:
            convert_method = converter.convert_and_quantize
        elif self.converter_type in [Engine.SNPE.value, Engine.QAIRT.value]:
            convert_method = converter.convert


        if self.converter_type in [Engine.QNN.value, Engine.SNPE.value]:
            convert_method(self.work_dir, 'converted_model')
        elif self.converter_type == Engine.QAIRT.value:
            convert_method()


    def get_tensor_names_from_json(self, json_file):
        try:
            # Load the json data
            data = InferenceEngineConfig.load_json(json_file)

            # Get the 'tensors' dictionary from the 'graph' dictionary
            tensors = data['graph']['tensors']

            # Get the tensor names
            tensor_names = list(tensors.keys())

            return tensor_names

        except Exception as e:
            self._logger.error(
                "Encountered error while retrieving inference tensor names from json file: {}".
                format(str(e)))
            return None


class InferenceMapperWithOutputs():
    """
    This class is responsible for getting the inference tensor names from the output directory.
    """

    def __init__(self, logger, args):
        """
        Initialize the runner with a logger and arguments. The output directory for inference is set from the arguments.
        """
        self._logger = logger
        self.target_outputs_dir = args.target_outputs_dir

    def get_inference_tensor_names(self):
        """
        Get the names of the inference tensors from the output directory.
        Only files with the extension '.raw' are considered.
        """
        tensor_names = []
        try:
            # Walk through the output directory
            outputs = os.walk(self.target_outputs_dir)
            for _, _, file_list in outputs:
                for file_path in file_list:
                    # Check if the file has the extension '.raw'
                    if file_path.endswith(".raw"):
                        # Get the name of the tensor by removing the extension
                        tensor_name = os.path.splitext(file_path)[0]
                        tensor_names.append(tensor_name)
        except Exception as e:
            # Log any errors encountered during the process
            self._logger.error(
                "Encountered error while retrieving inference tensor names: {}".format(str(e)))

        return tensor_names


class FrameworkMapperWithModel(FrameworkRunner):
    """
    This class is a child of the FrameworkRunner class and is responsible for getting the tensor mapping with the framework.
    """

    def __init__(self, logger, args):
        """
        Initialize the runner with a logger and arguments, and load the framework for tensor mapping.
        """
        args.output_dir = args.work_dir # required for FrameworkRunner
        super(FrameworkMapperWithModel, self).__init__(logger, args)
        self._logger = logger
        self.load_framework_for_tensor_mapping()

    def get_mapping_for_qnn_node(self, qnn_output):
        """
        Get the mapping for a QNN node and sanitize the node name.
        """
        return santize_node_name(self.framework_instance.get_mapping_for_qnn_node(qnn_output))

    def get_mapping_for_snpe_node(self, snpe_output):
        """
        Get the mapping for an SNPE node.
        """
        return santize_node_name(self.framework_instance.get_mapping_for_snpe_node(snpe_output))


class FrameworkMapperWithOutput():
    """
    This class is responsible for getting the tensor mapping with the framework output.
    """

    def __init__(self, logger, args):
        """
        Initialize the runner with a logger and arguments.
        """
        self._logger = logger
        self.golden_outputs_dir = args.golden_outputs_dir
        self.dict_of_golden_outputs = self.get_dict_of_golden_outputs()

    def get_dict_of_golden_outputs(self):
        """
        Returns a dictionary where keys are golden raw file names and values are original node names.
        """
        dict_of_golden_outputs = {}
        try:
            if self.golden_outputs_dir:
                for path, _, files in os.walk(self.golden_outputs_dir):
                    for file in files:
                        rel_path = os.path.relpath(path, self.golden_outputs_dir)
                        if rel_path != ".":
                            file = os.path.join(rel_path, file)
                        tensor_name = os.path.splitext(file)[0]
                        tensor_replace = santize_node_name(tensor_name)
                        dict_of_golden_outputs[tensor_replace] = tensor_name
        except Exception as e:
            self._logger.error(
                "Encountered error while retrieving framework tensor names: {}".format(str(e)))
        return dict_of_golden_outputs

    def get_mapping_for_qnn_node(self, qnn_output):
        """
        Get the mapping for a QNN node.
        """
        #return qnn_output itself if no golden_dir
        if not (self.golden_outputs_dir):
            self._logger.warn(
                "NO_GOLDEN_DIR_FOR_TENSOR_MAPPING: Using the qnn output as mapping. {}".format(
                    str(qnn_output)))
            return qnn_output

        # support tensorflow, onnx
        if (qnn_output in self.dict_of_golden_outputs.keys()):
            return self.dict_of_golden_outputs[qnn_output]

        # if no matching, some warning will occur.
        self._logger.warn("GOLDEN_DIR_MAPPING_MISMATCH_TENSOR: {}".format(str(qnn_output)))
        return ""

    def get_mapping_for_snpe_node(self, snpe_output):
        """
        Get the mapping for an SNPE node.
        """
        # return snpe_output itself if no golden_dir
        if not self.golden_outputs_dir:
            self._logger.warn(
                "NO_GOLDEN_DIR_FOR_TENSOR_MAPPING: Using the snpe output as mapping. {}".format(
                    str(snpe_output)))
            return snpe_output
            # return default tensor mapping if no framework

        if (snpe_output in self.dict_of_golden_outputs.keys()):
            return self.dict_of_golden_outputs[snpe_output]

        # if no matching, some warning will occur.
        self._logger.warn("GOLDEN_DIR_MAPPING_MISMATCH_TENSOR: {}".format(str(snpe_output)))
        return ""

class TensorMapper():
    """
    This class is responsible for mapping tensors between a framework model and an inference model.
    """

    def __init__(self, get_mapping_arg, logger):
        """
        Initialize the mapper with the arguments for getting the mapping and a logger.
        """
        self.get_mapping_arg = get_mapping_arg
        self._logger = logger

    def get_framework_model_tensor_obj(self):
        """
        Get the tensor mapping object for the framework model.
        """
        # Check if the required arguments for the framework model are present
        if hasattr(self.get_mapping_arg, 'engine') and self.get_mapping_arg.engine:
            self.get_mapping_arg.engine = self.get_mapping_arg.engine.upper()
        required_arguments_with_framework_model = self.get_mapping_arg.framework and (self.get_mapping_arg.framework in get_available_frameworks()) and self.get_mapping_arg.model_path and self.get_mapping_arg.engine
        required_arguments_with_framework_output = self.get_mapping_arg.golden_outputs_dir

        tensor_mapping_framework_runner = None
        try:
            # If the required arguments for the framework output are present, get the tensor mapping runner with the framework output
            if required_arguments_with_framework_output:
                tensor_mapping_framework_runner = FrameworkMapperWithOutput(
                    self._logger, self.get_mapping_arg)
            # If the required arguments for the framework model are present, get the tensor mapping runner with the framework
            elif required_arguments_with_framework_model:
                tensor_mapping_framework_runner = FrameworkMapperWithModel(
                    self._logger, self.get_mapping_arg)
            else:
                raise ValueError("Required Arguments are not passed to get framework tensor names")
        except ValueError as value_error:
            self._logger.error("Value error occurred while getting Tensor Mapping: {}".format(
                str(value_error)))
        except Exception as e:
            self._logger.error("Error occurred while generating Tensor Mapping: {}".format(str(e)))

        return tensor_mapping_framework_runner

    def get_inference_model_tensor_obj(self):
        """
        Get the tensor mapping object for the inference model.
        """
        # If the arguments for the inference model are not present, set them to None
        if not hasattr(self.get_mapping_arg, 'converted_ir_graph'):
            self.get_mapping_arg.converted_ir_graph = None
        if not hasattr(self.get_mapping_arg, 'converter_params'):
            self.get_mapping_arg.converter_params = None
        if not hasattr(self.get_mapping_arg, 'engine_path'):
            self.get_mapping_arg.engine_path = None
        if not hasattr(self.get_mapping_arg, 'host_device'):
            self.get_mapping_arg.host_device = None

        # Check if the required arguments for the inference model are present
        required_arguments_with_inference_model = self.get_mapping_arg.converted_ir_graph or (
            self.get_mapping_arg.engine_path and self.get_mapping_arg.host_device
            and self.get_mapping_arg.converter_params)
        required_arguments_with_inference_outputs = self.get_mapping_arg.target_outputs_dir and self.get_mapping_arg.engine

        tensor_mapping_inference_runner = None
        try:
            # If the required arguments for the inference outputs are present, get the tensor mapping runner with the outputs
            if required_arguments_with_inference_outputs:
                tensor_mapping_inference_runner = InferenceMapperWithOutputs(
                    self._logger, self.get_mapping_arg)
            # If the required arguments for the inference model are present, get the tensor mapping runner with the utility
            elif required_arguments_with_inference_model:
                tensor_mapping_inference_runner = InferenceMapperWithUtility(
                    self._logger, self.get_mapping_arg)
            else:
                raise ValueError("Required Arguments are not passed to get inference tensor names")
        except ValueError as value_error:
            self._logger.error("Value error occurred while getting Tensor Mapping: {}".format(
                str(value_error)))
        except Exception as e:
            self._logger.error("Error occurred while generating Tensor Mapping: {}".format(str(e)))

        return tensor_mapping_inference_runner

    def get_tensor_mapping(self, framework_tensor_obj, inference_tensor_obj):
        """
        Get the tensor mapping between the framework model and the inference model.
        """
        tensor_mapping = {}
        output_file = os.path.join(self.get_mapping_arg.work_dir, "tensor_mapping.json")
        try:
            # Get the inference tensor names
            inference_outputs = inference_tensor_obj.get_inference_tensor_names()
            for inference_output in inference_outputs:
                # Handle "native" tag in target outputs names
                tensor_name = truncate_native_tag(inference_output)

                # Get the tensor mapping name for the given engine
                if self.get_mapping_arg.engine == Engine.QNN.value:
                    tensor_mapping_name = framework_tensor_obj.get_mapping_for_qnn_node(
                        tensor_name)
                elif self.get_mapping_arg.engine == Engine.SNPE.value:
                    tensor_mapping_name = framework_tensor_obj.get_mapping_for_snpe_node(
                        tensor_name)
                else:
                    raise Exception(
                        f"Tensor mapping is not supported for given engine/sdk type {self.get_mapping_arg.engine}"
                    )
                # If the mapping name is not empty, add it to the tensor mapping
                if tensor_mapping_name:
                    tensor_mapping[tensor_name] = tensor_mapping_name

            # Write the tensor mapping to a file
            with open(output_file, 'w') as f:
                json.dump(tensor_mapping, f, indent=4)

            self._logger.info(
                get_progress_message("PROGRESS_INFERENCE_ENGINE_TENSOR_MAPPING_FINISHED"))
        except Exception as e:
            self._logger.error("Encountered error while generating tensor mapping: {}".format(
                str(e)))
            import traceback
            self._logger.error(traceback.print_exc())
        return output_file

    def run(self, ):
        """
        Run the tensor mapping process.
        """
        self._logger.info("Starting Tensor Mapping.")
        framework_tensor_obj = self.get_framework_model_tensor_obj()
        inference_tensor_obj = self.get_inference_model_tensor_obj()
        return self.get_tensor_mapping(framework_tensor_obj, inference_tensor_obj)
