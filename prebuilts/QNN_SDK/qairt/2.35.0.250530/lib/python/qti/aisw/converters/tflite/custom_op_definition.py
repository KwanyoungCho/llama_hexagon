# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import log_warning
from qti.aisw.converters.qnn_backend.custom_ops.core import BackendCustomOp as CustomOp
from qti.aisw.converters.qnn_backend.custom_ops.core import Param, ScalarParam, TensorParam, \
    StringParam, ParamTypes, convert_to_backend_type_from_numpy
import numpy as np


class CustomTfliteOp(CustomOp):
    """
    A subclass of the CustomOp interface which implements framework specific methods defined in
    CustomOp. Calling this class requires that an tflite module can be imported. Additionally,
    the parameters must be extractable from the op. See CustomOp for all methods that will be
    called when a CustomTfliteOp is instantiated
    """
    def __init__(self, src_op, input_tensor_info, output_tensor_info, param_info,
                 operator_type, model, custom_op_count_dict):
        self.model = model
        self.src_op = src_op
        self.param_info = param_info
        self.input_tensors = input_tensor_info
        output_tensors = self.set_output_dims(src_op, output_tensor_info, model)
        self.custom_op_idx = custom_op_count_dict[operator_type]
        self.op_name = "_".join([operator_type, str(self.custom_op_idx)])
        for i in range(len(output_tensors)):
            output_tensors[i].name = "_".join([operator_type, str(self.custom_op_idx), str(i)])
        super(CustomTfliteOp, self).__init__(operator_type,
                                             name=self.op_name,
                                             src_op=src_op,
                                             input_tensors=input_tensor_info,
                                             output_tensors=output_tensors,
                                             param_info=param_info)

    def extract_attrs(self, src_op, param_infos):

        def is_iterable(attr_value):

            try:
                iter(attr_value)
            except TypeError:
                return False
            return True

        attrs = dict()
        for param_info in param_infos:
            name = param_info.name
            if param_info.static:
                attr_value = []
            elif param_info.default_value is not None:
                attr_value = param_info.default_value
            else:
                raise KeyError(code_to_message.get_error_message('ERROR_MISSING_ATTRIBUTE')(param_info.name,
                                                                                            self.op_type))
            if is_iterable(attr_value):
                iterable = True
            elif type(attr_value) is bool:
                attr_value = int(attr_value)
                iterable = False
            elif isinstance(attr_value, (int, float)):
                iterable = False
            else:
                raise TypeError("Type: {} for attr: {} is not recognized".format(type(attr_value), name))

            if not iterable:
                attrs[name] = Param(name, ParamTypes.SCALAR, ScalarParam(attr_value))
            else:
                if isinstance(attr_value, (str, bytes)):
                    if isinstance(attr_value, bytes):
                        # assuming unicode or bytes and utf-8 encoding
                        attr_value = attr_value.decode('utf-8') + '\0'
                    attrs[name] = Param(name, ParamTypes.STRING, StringParam(attr_value))
                else:
                    attrs[name] = Param(name, ParamTypes.TENSOR,
                                        TensorParam(attr_value, param_info))
        return attrs

    def infer_output_shapes(self, node, model=None, **kwargs):
        """
         This method infers the shape of a tflite Node's output tensors using the node itself,
         a user provided model containing the node.

        :param node: The tflite Node object
        :param model: A required field which should be a tflite object
        :return: a list of lists which contains output dimensions for each output tensor
        in the tflite Node.
        """
        from qti.aisw.converters.relay.custom_ops.utils.tflite_helpers import TensorInfo
        output_dims = []
        subgraph = model.Subgraphs(0)
        tensors = TensorInfo.get_tensor_info(self, node.OutputsAsNumpy(), subgraph)
        for operator_tensor in tensors:
            temp_list_shape = []
            for j in range(operator_tensor.tensor.ShapeLength()):
                temp_list_shape.append(operator_tensor.tensor.Shape(j))
            output_dims.append(temp_list_shape)
        return output_dims

    def set_tensor_data_types(self, node):
        for i, output in enumerate(self.outputs):
            dtype = self.get_tensor_data_type(node.Outputs(i))
            output.data_type = convert_to_backend_type_from_numpy(dtype)
            if output.data_type not in output.allowed_data_types:
                output.data_type = output.allowed_data_types[0]

        for i, input in enumerate(self.inputs):
            dtype = self.get_tensor_data_type(node.Inputs(i))
            input.data_type = convert_to_backend_type_from_numpy(dtype)
            if input.data_type not in input.allowed_data_types:
                input.data_type = input.allowed_data_types[0]

    def get_tensor_data_type(self, tensor_idx):
        tensor = self.model.Subgraphs(0).Tensors(tensor_idx)
        try:
            from tflite.TensorType import TensorType

            return {
                TensorType.UINT8: np.uint8,
                TensorType.INT8: np.int8,
                TensorType.INT16: np.int16,
                TensorType.FLOAT16: np.float16,
                TensorType.FLOAT32: np.float32,
                TensorType.INT32: np.int32,
                TensorType.INT64: np.int64,
                TensorType.BOOL: np.bool_,
            }[tensor.Type()]
        except ImportError:
            raise ImportError("The tflite package must be installed")
        except KeyError:
            raise NotImplementedError(
                "Tensor type '{}' currently not supported".format(tensor.Type())
            )

    def validate(self, *args, **kwargs):
        self.validate_params(self.src_op, self.param_info)

    def validate_params(self, src_op, param_info):
        """
        Validate params in the src_op with respect to param_infos defined in the config spec. Note
        that unlike tensors,
        params must be named in the config spec. If the param is not present in the op, a KeyError
         is raised. Likewise, if a param not provided in the config spec is included,
         the error is also raised.
        :param src_op: The TFLite Operator Object containing the Op Type
        :param param_info: The list of param information as defined in the config spec.
        :raises: a KeyError if the param is missing or a param is present in the op.
        """
        for param in param_info:
            if param.name not in (attr for attr in self.params) \
                    and not param.static and param.default_value is None:
                raise KeyError(
                    code_to_message.get_error_message('ERROR_MISSING_ATTRIBUTE')(param.name))
        for attr in self.params:
            if attr not in (param.name for param in param_info):
                log_warning("Attribute: {} was found in the op: {} but has not been defined in "
                            "the op config. The attribute will be ignored!",
                            attr.name, self.op_type)

