# ==============================================================================
#
#  Copyright (c) 2018-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from __future__ import annotations

import math
import os
import sys
import copy
import tempfile
import itertools
import shutil
import numpy as np
from packaging.version import parse as parse_version
from collections import defaultdict, OrderedDict
from collections.abc import Iterable
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Set, Text, Tuple, Union
from qti.aisw.converters.common.utils import code_to_message

try:
    import onnx
    from onnx import (
        AttributeProto,
        GraphProto,
        ModelProto,
        NodeProto,
        TensorProto,
        helper,
        numpy_helper,
        mapping,
        onnx_pb,
    )
    from onnx.external_data_helper import (
        ExternalDataInfo,
        set_external_data,
        uses_external_data,
        save_external_data,
    )
    import onnx.onnx_cpp2py_export.checker as c_checker
except:
    onnx = None  # converter will throw before we try anything in here

from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import (
    converter_type,
    log_assert,
    log_debug,
    log_debug1,
    log_error,
    log_info,
    log_warning,
)
from qti.aisw.converters.common.utils.framework_utils import (
    TensorInfo,
    determine_layout,
    make_dirs,
)
from qti.aisw.converters.common.utils.translation_utils import broadcastable
from qti.aisw.converters.common import ir_graph


class ModelWrapper:
    """
    Class to abstract onnx model external data loading and unloading part.
    """

    def __init__(self, model_or_path: Union[str, ModelProto], include_attributes: bool=True, defer_loading: bool=False):
        """
        Initializes the model along with internal variables to help parse the
        model efficiently.

        :param Union[ModelProto, str] model_or_path: Onnx model representation
            in form of model path or ModelProto.
        :param bool defer_loading: If True, the model won't load weights.
            If False, the model will be loaded eagerly. Defaults to False.
        :param bool include_attributes: Include tensors which are part of node's
            attributes. Defaults to True.
        :raises RuntimeError: In case ModelProto has any external weights,
            e.g. weights are stored in disk rather than ModelProto object.
        :raises RuntimeError: In case model is neither path nor ModelProto.
        """
        self.include_attributes = include_attributes
        # Loads the model and sets self.model and self.model_name
        self.load_model(model_or_path, defer_loading)

        # Sets self.external_data_path and self.model_has_external_data
        self.set_external_data_path()

    def load_model(self, model_or_path: Union[str, ModelProto], defer_loading: bool=False):
        """
        Load the model and sets internal variables

        :param Union[ModelProto, str] model_or_path: Onnx model representation
            in form of model path or ModelProto.
        :param bool defer_loading: If True, the model won't load weights.
            If False, the model will be loaded eagerly. Defaults to False.
        :raises RuntimeError: In case ModelProto has any external weights,
            e.g. weights are stored in disk rather than ModelProto object.
        :raises RuntimeError: In case model is neither path nor ModelProto.
        """
        self.weights_loaded = not defer_loading
        if isinstance(model_or_path, str):
            self.model = onnx.load(model_or_path, load_external_data=not defer_loading)
            self.model_name = os.path.basename(model_or_path)
        elif isinstance(model_or_path, ModelProto):
            if model_uses_external_data(model_or_path, True):
                raise RuntimeError("Model without weights is provided to ModelWrapper.")
            self.model = model_or_path
            self.model_name = "model.onnx"
        else:
            raise RuntimeError("Invalid type of object received in ModelWrapper.")

        # Transfer TensorProto of Constant nodes to initializers
        transfer_constants_to_initializers(self.model)

        # If defer loading is enabled, check that weights are loaded or not.
        # If all the weights are present in model itself, follow regular flow.
        if not self.weights_loaded:
            self.weights_loaded = update_tensorproto_paths(self.model, os.path.dirname(model_or_path))

    def set_external_data_path(self):
        """
        Dumps the model external data into tempdir and sets internal variables
        """
        #TODO: Remove all this skipping logic, once we integrate defer_loading with ModelWrapper design
        if not self.weights_loaded:
            log_debug("Skipping the dump of the model external data into tempdir")
            return False

        if len(self.model.graph.initializer) == 0:
            self.model_has_external_data = False
            self.external_data_path = None
        else:
            convert_data_to_raw_data(self.model, self.include_attributes)
            status, external_data_path = unload_external_data(
                self.model, self.include_attributes
            )
            if status:
                self.external_data_path = external_data_path
                self.model_has_external_data = True
            else:
                self.model_has_external_data = False
                self.external_data_path = None

    @property
    def has_external_data(self) -> bool:
        """
        Model property indicating whether the model has any external data or not.

        :return bool: True if model has external data else False.
        """
        return self.model_has_external_data

    def get_full_model(
        self, inplace: bool = False, remove_external_data: bool = True
    ) -> ModelProto:
        """
        Generate the model with weights and returns ModelProto instance.

        :param bool inplace: Load the weights in the current object's model
            instance, defaults to False.
        :param bool remove_external_data: Indicates whether to remove the weights
            after loading the weights in the model, It is used only in case of
            inplace=True, defaults to True
        :return ModelProto: Onnx model with weights.
        """
        if not self.weights_loaded:
            log_debug("Skipping the loading the full model with weights")
            return False
        if inplace:
            if self.has_external_data:
                reload_external_data(self.model, self.external_data_path, False, self.include_attributes)
                if remove_external_data:
                    self.remove_external_file()
                self.model_has_external_data = False
            return self.model
        else:
            temp_model = copy.deepcopy(self.model)
            if self.has_external_data:
                reload_external_data(temp_model, self.external_data_path, False, self.include_attributes)
            return temp_model

    def get_model_path(self) -> str:
        """
        Save the model at the given path along with the weights.
        Note: Caller shall not remove model from this path else the other
        subsequent calls to the model might fail due to unavailability of
        external data.

        :return str: File path where the model is saved.
        """
        temp_model = copy.deepcopy(self.model)

        if self.has_external_data:
            # Update the external data references in cloned model's tensors.
            updated_file_name = os.path.basename(self.external_data_path)
            for tensor in get_all_tensors(temp_model, self.include_attributes):
                if uses_external_data(tensor):
                    # Update the filename in each tensor's external data location field.
                    tensor.external_data[0].value = updated_file_name

            model_path = os.path.join(os.path.dirname(self.external_data_path), "model.onnx")
        else:
            try:
                tmp_dir = os.getenv("QAIRT_TMP_DIR", default=tempfile.gettempdir())
                tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
                model_path = os.path.join(tmp_dir, "model.onnx")
            except Exception as e:
                raise RuntimeError("Unable to dump external data to the disk "
                        f"at: {tmp_dir}. It may be due to permission or "
                        "space issue. Try setting QAIRT_TMP_DIR environment variable "
                        "to different location.") from e
        onnx.save(temp_model, model_path)
        return model_path

    def clone_wrapper(self) -> ModelWrapper:
        """
        Clone the given wrapper and return the new wrapper instance.

        :return ModelWrapper: Cloned representation of ModelWrapper.
        """
        model_w_weights = self.get_full_model()
        return ModelWrapper(model_w_weights)

    def update_model(self, model: ModelProto) -> None:
        """
        Updates the internal model with given model reference.

        :param ModelProto model: Reference onnx model which will replace
            internal model proto instance.
        """
        if model_uses_external_data(model, self.include_attributes):
            self.model = model
        else:
            self.model = model
            # If before self.model = model, the self.model doesn't have external data
            # but after self.model = model, the self.model has external data
            # then self.has_external_data is set to false which is wrong.
            if self.has_external_data:
                self.remove_external_file()
            status, external_data_path = unload_external_data(
                self.model, self.include_attributes
            )
            if status:
                self.external_data_path = external_data_path
                self.model_has_external_data = True
            else:
                self.external_data_path = None
                self.model_has_external_data = False

    def remove_external_file(self) -> None:
        """
        Remove external data for the model stored in the class.
        """
        if self.has_external_data and self.external_data_path is not None:
            if os.path.isfile(self.external_data_path):
                shutil.rmtree(os.path.dirname(self.external_data_path))
                self.external_data_path = None
                self.model_has_external_data = False

    def move_wrapper(self, other_wrapper: ModelWrapper) -> None:
        """
        Function to update the current object using provided wrapper object.
        Note: This will update the other_wrapper to point to None model and None
        external_data_path.

        :param ModelWrapper other_wrapper: Reference wrapper object to move
            from.
        """
        if self == other_wrapper:
            return
        self.model = other_wrapper.model
        other_wrapper.model = None
        if other_wrapper.has_external_data:
            if self.has_external_data:
                if not os.path.samefile(
                    self.external_data_path, other_wrapper.external_data_path
                ):
                    self.remove_external_file()
            self.external_data_path = other_wrapper.external_data_path
            self.model_has_external_data = True
            other_wrapper.external_data_path = None


class OnnxAttrProtoUtil(object):
    strCode_to_enum = {
        "i": onnx.AttributeProto.INT,
        "f": onnx.AttributeProto.FLOAT,
        "s": onnx.AttributeProto.STRING,
        "t": onnx.AttributeProto.TENSOR,
        "g": onnx.AttributeProto.GRAPH,
        "li": onnx.AttributeProto.INTS,
        "lf": onnx.AttributeProto.FLOATS,
        "ls": onnx.AttributeProto.STRINGS,
        "lt": onnx.AttributeProto.TENSORS,
        "lg": onnx.AttributeProto.GRAPHS,
    }

    enum_to_strCode = {
        onnx.AttributeProto.INT: "i",
        onnx.AttributeProto.FLOAT: "f",
        onnx.AttributeProto.STRING: "s",
        onnx.AttributeProto.TENSOR: "t",
        onnx.AttributeProto.GRAPH: "g",
        onnx.AttributeProto.INTS: "li",
        onnx.AttributeProto.FLOATS: "lf",
        onnx.AttributeProto.STRINGS: "ls",
        onnx.AttributeProto.TENSORS: "lt",
        onnx.AttributeProto.GRAPHS: "lg",
    }


TensorsType = Dict[str, np.ndarray]
TensorShapeType = List[int]
TensorSpecType = Dict[str, TensorShapeType]

numpy_to_onnx_type = {
    np.float32: onnx.TensorProto.FLOAT,
    np.float64: onnx.TensorProto.DOUBLE,
    np.uint8: onnx.TensorProto.UINT8,
    np.int8: onnx.TensorProto.INT8,
    np.int16: onnx.TensorProto.INT16,
    np.int32: onnx.TensorProto.INT32,
    np.int64: onnx.TensorProto.INT64,
    np.bool_: onnx.TensorProto.BOOL,
}

ONNX_EXTERNAL_DATA_THRESHOLD = 1024
SYMBOLIC_SHAPE_INFER_MIN_OPSET_VER = 7
MAX_INT_32 = (1 << 31) - 1


def elem_type_to_name(elem_type: int) -> str:
    """
    Function to convert the elem_type to its equivalent name based on onnx proto

    :param type(int): elem_type int
    :return type(str): elem_type name
    """
    type_map = mapping.TENSOR_TYPE_TO_NP_TYPE
    if elem_type in type_map.keys():
        return type_map[elem_type].name
    else:
        return "undefined"


onnx_to_np_dtype = {
    # int types
    TensorProto.INT8: np.dtype("int8"),
    TensorProto.INT16: np.dtype("int16"),
    TensorProto.INT32: np.dtype("int32"),
    # downcast as yet not supported in conversion
    TensorProto.INT64: np.dtype("int32"),
    TensorProto.UINT8: np.dtype("uint8"),
    TensorProto.UINT16: np.dtype("uint16"),
    TensorProto.UINT32: np.dtype("uint32"),
    # downcast as yet not supported in conversion
    TensorProto.UINT64: np.dtype("uint32"),
    # float types
    TensorProto.FLOAT16: np.dtype("float16"),
    TensorProto.FLOAT: np.dtype("float32"),
    TensorProto.DOUBLE: np.dtype("float64"),
    TensorProto.BOOL: np.dtype("bool_"),
}

KNOWN_ATTRIBUTE_DEFAULTS = dict(
    dilations=[1, 1],
    strides=[1, 1],
    pads=[0, 0, 0, 0],
    output_shape=[],
    axes=[],
    consumed_inputs=[],
    kernel_shape=[],
)


def is_broadcast(onnx_op, graph=None):
    attrs = extract_attributes(onnx_op, [("axis", "i", 0), ("broadcast", "i", 0)])

    if graph is not None:
        # newer version of onnx(e.g version 7 of Mul or Add) do not have axis and broadcast attributes
        # hence another way to check would be to make sure all inputs to op are the same shape
        input_names = list(map(str, onnx_op.input))
        input_buffers_shape = []
        for name in input_names:
            if graph.has_buffer(name):
                input_buffers_shape.append(list(graph.get_buffer(name).shape))
            else:
                input_buffers_shape.append(list(graph.weights.fetch(name).shape))
        if any(shape != input_buffers_shape[0] for shape in input_buffers_shape):
            return True

    return attrs["axis"] != 0 or attrs["broadcast"] == 1


def assert_no_broadcast(onnx_op):
    log_assert(
        not is_broadcast(onnx_op),
        code_to_message.get_error_message("ERROR_BROADCAST_NOT_SUPPORTED")(
            onnx_op.name
        ),
    )


class NamedDict(dict):
    def __getattr__(self, key):
        return self[key]


def extract_attributes(
    onnx_op, attr_infos=None, schema=None, validate=False, default_attrs=NamedDict()
):
    """Ensure the existence and extract well typed attributes from an onnx
    NodeProto.
    :param attr_infos: a list of attributes to extract in the form [(attr_name, attr_type, attr_value)]
    :param schema:   an op_schema object for the onnx_op
    :param validate:  an optional validator function that is registered with the schema
                     of the form:  validator(src_op, attr_name, attr_value)
    :param default_attrs: a object have a named property for default attributes info value

    Each entry in attr_info should be either a 2- or 3-tuple.
    * The first element should be the string name of an attribute.
    * The second element should by a type code for the attribute corresponding to:
      - i for int attributes
      - f for float attributes
      - s for string attributes
      - t for tensor attributes (returned as a numpy array)
      - g for graph attributes
      - lx, where x is one of the preceding attribute type identifiers, for list valued attributes
    * The third element, if present, specifies a default value should the attribute not be present.
      If no default is specified, this function will thrown an error.

    The return object will have a named property for each attribute info."""
    onnx_attrs = {}
    if not attr_infos and schema:
        attr_infos = schema.attributes()

    for attr in onnx_op.attribute:
        onnx_attrs[attr.name] = attr
        if schema and not validate:
            if not schema.check_supported_attributes(str(attr.name)):
                log_warning(
                    code_to_message.get_warning_message(
                        "WARNING_UNSUPPORTED_ATTRIBUTE"
                    )(attr.name, onnx_op.op_type, onnx_op.input[0])
                )
    ret = NamedDict()
    for attr_info in attr_infos:
        name = attr_info[0]
        if not name in onnx_attrs:
            if len(default_attrs) == 0:
                if len(attr_info) == 3:
                    ret[name] = attr_info[2]
                    continue
                else:
                    try:
                        ret[name] = KNOWN_ATTRIBUTE_DEFAULTS[name]
                        continue
                    except KeyError:
                        raise ValueError(
                            code_to_message.get_error_message(
                                "ERROR_ATTRIBUTE_MISSING"
                            )(onnx_op.name, name)
                        )
            elif len(default_attrs) > 0 and name in default_attrs.keys():
                ret[name] = default_attrs[name]
                continue

        attr = onnx_attrs[name]
        code = attr_info[1]
        requested_type = OnnxAttrProtoUtil.strCode_to_enum[code]
        if attr.type != requested_type:
            msg = code_to_message.get_error_message("ERROR_ATTRIBUTE_WRONG_TYPE")(
                onnx_op.name,
                name,
                onnx.AttributeProto.AttributeType.Name(requested_type),
                onnx.AttributeProto.AttributeType.Name(attr.type),
            )
            raise TypeError(msg)
        value = extract_onnx_type(code, attr)
        if validate and schema:
            schema.validate_data_constraints(onnx_op)
            schema.get_validate_method("validate_attribute_values")(
                onnx_op, name, value
            )
        ret[name] = value
    return ret


def extract_onnx_type(code, attr):
    ret = ""
    if code == "i":
        ret = int(attr.i)
    elif code == "f":
        ret = float(attr.f)
    elif code == "s":
        ret = str((attr.s).decode("utf-8"))
    elif code == "g":
        ret = attr.g
    elif code == 't':
        ret = get_initializer_value(attr.t)
    elif code == 'li':
        ret = list(map(int, attr.ints))
    elif code == "lf":
        ret = list(map(float, attr.floats))
    elif code == "ls":
        ret = list(map(str, [data.decode("utf-8") for data in attr.strings]))
    elif code == "lg":
        ret = list(attr.graphs)
    elif code == 'lt':
        ret = list(map(lambda tensor: get_initializer_value(tensor), attr.tensors))
    return ret


def extract_padding_mode(auto_pad, node_name, ceil_mode=0, right_handed=False):
    if right_handed == True:
        return IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED
    elif auto_pad == "VALID":
        return IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID
    elif auto_pad == "SAME_UPPER":
        return IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END
    elif auto_pad == "SAME_LOWER":
        return IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN
    elif ceil_mode != 0:
        return IRPaddingStrategies.PADDING_SIZE_EXPLICIT
    elif auto_pad == "NOTSET":
        return IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
    else:
        raise ValueError(
            code_to_message.get_error_message("ERROR_PADDING_TYPE_UNSUPPORTED")(
                node_name, auto_pad
            )
        )


def broadcast_to(data, new_shape):
    """
    Broadcasts data into a new shape if possible
    :param new_shape: shape to be broadcasted into
    :param data: data to be broadcasted
    :return: broadcasted data if possible or original data if not
    """
    if data.shape != new_shape and broadcastable(data.shape, new_shape):
        return np.broadcast_to(data, new_shape).astype(np.float32)
    return data


def product(nums):
    if len(nums) == 0:
        return 1
    else:
        return reduce(mul, nums)

np_dtype_to_bitwidth = {
    # TODO: numpy doesn't support int4 dtype
    # np.dtype('int4'): 4,
    np.dtype('int8'): 8,
    np.dtype('int16'): 16,
    np.dtype('int32'): 32,
    # np.dtype('uint4'): 4,
    np.dtype('uint8'): 8,
    np.dtype('uint16'): 16,
    np.dtype('uint32'): 32,
}

def get_quant_info(zp):
    log_assert(isinstance(zp, np.ndarray),
               "Zero point is not a numpy array")
    if zp.dtype in np_dtype_to_bitwidth:
        bw = np_dtype_to_bitwidth[zp.dtype]
        is_int = str(zp.dtype).startswith('int')
        is_symmetric = is_int and np.allclose(zp, 0)
        offset = -np.int32(zp)
        if is_int and not is_symmetric:
            # activation is quantized to unsigned integer in SNPE/QNN,
            # so we need to shift offsets for signed integer
            offset = offset - 2**(bw-1)
        return is_symmetric, bw, offset
    else:
        raise ValueError("Unsupported zero point type: ", zp.dtype)


def get_encoding(name, scale, zp, axis = 1, block_size = 0):
    is_symmetric, bw, offset = get_quant_info(zp)
    return {"name": name,
            "bw": bw,
            "min": (np.float32(np.iinfo(zp.dtype).min) - zp) * scale,
            "max": (np.float32(np.iinfo(zp.dtype).max) - zp) * scale,
            "scale" : scale,
            "offset": offset,
            "is_symmetric": is_symmetric,
            "overridden": True,
            "axis": axis,
            "block_size": block_size}


def downcast_dtype_64bit_to_32bit(tensor_name, tensor_dtype):
    numpy_dtype_downcast = {
        np.dtype("int64"): np.int32,
        np.dtype("uint64"): np.uint32,
        np.dtype("float64"): np.float32,
    }
    if tensor_dtype in numpy_dtype_downcast:
        prev_dtype = tensor_dtype
        tensor_dtype = numpy_dtype_downcast[tensor_dtype]
        log_debug1(
            code_to_message.get_debugging_message("DEBUG_DOWNCAST_TENSOR")(
                prev_dtype, np.dtype(tensor_dtype), tensor_name
            )
        )

    return tensor_dtype


class WeightData(object):
    def __init__(self, constant_tensor, was_scalar=False, shape=None, dtype=None):
        """
        :param weights: weights from the network
        :param was_scalar: boolean to determine if original weight was initialized as scalar.
                           Since QuIR expects that all inputs are tensors, this information will be
                           helpful for any op specific usecases such as determining output_shape
        """
        self.constant_tensor = constant_tensor
        # Track if the weights have been retrieved for use in another layer
        # Weights can be provided in one of two ways: initializers or constant ops
        # Constant ops being used as weight providers are setup with the weights from
        # the start and thus don't need to retrieve weights from the weight provider
        # again. SNPE layers like Conv/Matmul/GEMM/etc store weights internally and
        # will attempt to retrieve the weights. The consumed field will track which
        # Constant ops are being used as weight providers so they can be pruned from
        # the network at the end
        self.consumed = False
        self.was_scalar = was_scalar
        self.shape = shape
        self.dtype = dtype

    @property
    def weights(self):
        return self.constant_tensor.data




# ------------------------------------------------------------------------------
#   WeightProvider
# ------------------------------------------------------------------------------
class WeightProvider(object):
    def __init__(self, model, defer_loading=False, remove_model_weights=True):
        self.weight_map = {}

        if defer_loading:
            self.set_lazy_weight_map(model)
        else:
            self.set_weight_map(model,remove_model_weights)

    def set_weight_map(self,model,remove_model_weights=True):
        count = 0
        total_weights = len(model.graph.initializer)
        while count < total_weights:
            if remove_model_weights:
                # index of the weight to be inserted will be 0 since previous weights are already removed
                tensor_proto = model.graph.initializer[0]
                # remove weights from the weight initializer
                model.graph.initializer.remove(tensor_proto)
            else:
                tensor_proto = model.graph.initializer[count]
            tensor_name = tensor_proto.name
            onnx_np_tensor = get_initializer_value(tensor_proto)
            was_scalar = False

            if not tensor_proto.dims:
                # tensor of dim 1 is empty array in onnx resulting in broken numpy array,
                # since unable to modify tensor.dims, reshaping to have numpy array proper config
                onnx_np_tensor = onnx_np_tensor.reshape(1)
                was_scalar = True
            ir_constant_tensor = ir_graph.PyIrConstantTensor(onnx_np_tensor)
            self.weight_map[str(tensor_name)] = WeightData(ir_constant_tensor, was_scalar, onnx_np_tensor.shape, onnx_np_tensor.dtype)
            count += 1

    def set_lazy_weight_map(self,model):
        count = 0
        total_weights = len(model.graph.initializer)
        while count < total_weights:
            tensor_proto = model.graph.initializer[count]
            tensor_name = tensor_proto.name
            was_scalar = False
            dims = get_tensor_shape(tensor_proto)
            dtype = get_tensor_dtype(tensor_proto)
            if not dims:
                onnx_np_tensor = get_initializer_value(tensor_proto)
                # tensor of dim 1 is empty array in onnx resulting in broken numpy array,
                # since unable to modify tensor.dims, reshaping to have numpy array proper config
                onnx_np_tensor = onnx_np_tensor.reshape(1)
                dims = [1]
                was_scalar = True
                ir_constant_tensor = ir_graph.PyIrConstantTensor(onnx_np_tensor)
            elif uses_external_data(tensor_proto):
                info = ExternalDataInfo(tensor_proto)
                external_data_file_path = info.location
                if info.offset:
                    offset = int(info.offset)
                else:
                    offset = 0
                if info.length:
                    length = int(info.length)
                else:
                    # Calculate the size of each element using np.dtype
                    data_type_size = dtype.itemsize

                    # Calculate the length (number of elements * size of each element)
                    length = np.prod(dims) * data_type_size
                ext_data_info = ir_graph.ExternalDataInfo(external_data_file_path,offset,length,dims,dtype)
                ir_constant_tensor = ir_graph.PyIrConstantTensor(ext_data_info)
            else:
                onnx_np_tensor = get_initializer_value(tensor_proto)
                ir_constant_tensor = ir_graph.PyIrConstantTensor(onnx_np_tensor)
            ir_constant_tensor.set_scalar(was_scalar)
            self.weight_map[str(tensor_name)] = WeightData(ir_constant_tensor, was_scalar, dims, dtype)
            count += 1

    def was_scalar(self, key):
        return self.weight_map[key].was_scalar

    def consumed(self, key):
        if not key in self.weight_map:
            return False
        return self.weight_map[key].consumed


    def fetch(self, *keys, **kwargs):
        ret = []
        # Prunable indicates whether the weights have been consumed in such a way as to
        # allow pruning of the node (eg Const ops that contain weights are consumed by
        # Conv/FC/etc and thus can be pruned from the network. Const ops that are inputs
        # to a node cannot
        consumed = kwargs.get("prunable", True)
        for key in keys:
            key = str(key)
            log_debug(
                code_to_message.get_debugging_message("DEBUG_RETRIEVE_WEIGHTS")(key)
            )
            if key not in self.weight_map:
                raise KeyError(
                    code_to_message.get_error_message("ERROR_WEIGHTS_MISSING_KEY")(key)
                )
            self.weight_map[key].consumed = consumed
            if kwargs.get("dtype") is None:
                tensor_dtype = downcast_dtype_64bit_to_32bit(
                    key, self.weight_map[key].dtype)
            else:
                # Assumes downcasting of provided dtype, if required, will be handled by fetch caller
                tensor_dtype = kwargs.get("dtype")

            # This function returns the original weights present in the weight map.
            # If the requested dtype is different from the actual tensor dtype, then it returns a
            # copy of the weight.
            # The original weight should not be modified by the translations.
            if tensor_dtype != self.weight_map[key].dtype:
                ret.append(np.require(
                    self.weight_map[key].weights.copy(), dtype=tensor_dtype))
            else:
                ret.append(self.weight_map[key].weights)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret


    def fetch_constant_tensor(self, *keys, **kwargs):
        ret = []
        # Prunable indicates whether the weights have been consumed in such a way as to
        # allow pruning of the node (eg Const ops that contain weights are consumed by
        # Conv/FC/etc and thus can be pruned from the network. Const ops that are inputs
        # to a node cannot
        consumed = kwargs.get('prunable', True)
        for key in keys:
            key = str(key)
            log_debug(code_to_message.get_debugging_message(
                "DEBUG_RETRIEVE_WEIGHTS")(key))
            if key not in self.weight_map:
                raise KeyError(code_to_message.get_error_message(
                    "ERROR_WEIGHTS_MISSING_KEY")(key))
            self.weight_map[key].consumed = consumed
            if kwargs.get('dtype') is None:
                tensor_dtype = downcast_dtype_64bit_to_32bit(
                    key, self.weight_map[key].dtype)
            else:
                # Assumes downcasting of provided dtype, if required, will be handled by fetch caller
                tensor_dtype = kwargs.get('dtype')

            # This function returns the original weights present in the weight map.
            # If the requested dtype is different from the actual tensor dtype, then it returns a
            # copy of the weight.
            # The original weight should not be modified by the translations.
            if tensor_dtype != self.weight_map[key].dtype:
                temp = np.require(
                    self.weight_map[key].weights, dtype=tensor_dtype)
                ir_constant_tensor = ir_graph.PyIrConstantTensor(temp)
                ret.append(ir_constant_tensor)
            else:
                ret.append(self.weight_map[key].constant_tensor)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def has(self, key):
        return key in self.weight_map

    def type(self, key):
        if key not in self.weight_map:
            raise KeyError(code_to_message.get_error_message(
                "ERROR_WEIGHTS_MISSING_KEY")(key))
        return self.weight_map[key].dtype

    def has_all(self, keys):
        return all(self.has(key) for key in keys)


    def insert(self, key, weights, was_scalar=False):
        log_debug("Inserting weights for {}, was_scalar:{}", key, was_scalar)
        if not isinstance(weights, ir_graph.PyIrConstantTensor):
            ir_constant_tensor = ir_graph.PyIrConstantTensor(weights)
        else:
            ir_constant_tensor = weights
        dtype = ir_constant_tensor.dtype
        shape = ir_constant_tensor.dims
        self.weight_map[key] = WeightData(ir_constant_tensor, was_scalar, shape, dtype)


def get_type_dims_info(source, node_name):
    """
    :param source: structure to query for node_name's info
    :param node_name: the name of the node to query info for
    :return: (bool, elem_type, dims)
    """
    for info in source:
        if info.name == node_name:
            dims = []
            for dim in info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(dim.dim_value)
                else:
                    dims.append(dim.dim_param)
            return True, info.type.tensor_type.elem_type, dims
    return False, None, None


def get_all_dims_info(source):
    """
    :param source: structure to query for input/output shape info
    :return: [(str_name1, elem_type1, dims1), (str_name2, elem_type2, dims2)]
    """
    dims_info = []
    for info in source:
        dims = [
            str(dim.dim_param) if dim.dim_param else int(dim.dim_value)
            for dim in info.type.tensor_type.shape.dim
        ]
        dims_info.append((info.name, info.type.tensor_type.elem_type, dims))
    return dims_info


# ------------------------------------------------------------------------------
#   ONNX Common Utils
# ------------------------------------------------------------------------------


def get_unique_ops(model) -> Dict[str, int]:
    """
    Get the unique ops present in the model and also, get the total ops. count.

    :param model: path to the loaded onnx model proto
    :return Dict[str, int]: Mapping of each op with its count.
    """
    count = defaultdict(int)
    for node in get_nodes(model):
        count[node.op_type] += 1
    return count


def get_inputs(model) -> List[onnx.ValueInfoProto]:
    """
    Get the graph inputs tensors except initializers.

    :param model: path to the loaded onnx model proto
    :return List: List of graph inputs
    """
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]


def get_all_inputs(model) -> List[onnx.ValueInfoProto]:
    """
    Get the graph inputs tensors including the inputs which are also initializers.

    :param model: path to the loaded onnx model proto
    :return List: List of graph inputs
    """
    return [ipt for ipt in model.graph.input]


def get_outputs(model) -> List[onnx.ValueInfoProto]:
    """
    Get the graph outputs tensors.

    :param model: path to the loaded onnx model proto
    :return List: List of graph iutputs
    """
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.output if ipt.name not in initializer_names]


def get_graphs(model: ModelProto) -> List[GraphProto]:
    """
    Function to return all the graph present in model.

    :param ModelProto model: Onnx Model graph proto.
    :return List[GraphProto]: Onnx graphs
    """
    all_graphs = []
    graph_queue = [model.graph]
    while graph_queue:
        graph = graph_queue.pop(0)
        all_graphs.append(graph)
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == AttributeProto.AttributeType.GRAPH:
                    assert isinstance(attr.g, onnx_pb.GraphProto)
                    graph_queue.append(attr.g)
                if attr.type == AttributeProto.AttributeType.GRAPHS:
                    for g in attr.graphs:
                        assert isinstance(g, onnx_pb.GraphProto)
                        graph_queue.append(g)
    return all_graphs


def get_nodes(model: ModelProto, include_subgraphs: bool=True) -> List[NodeProto]:
    """
    Function to return the nodes information.

    :param ModelProto model: Onnx Model graph proto.
    :param bool include_subgraphs: If true, will add the nodes of the subgraphs also. Default: True
    :return List[NodeProto]: Underlying onnx nodes.
    """
    nodes_all = []
    graphs = get_graphs(model) if include_subgraphs else [model.graph]
    for graph in graphs:
        for node in graph.node:
            nodes_all.append(node)
    return nodes_all


def get_graph_by_node(model: ModelProto, node: NodeProto) -> Union[None, GraphProto]:
    """
    Get graph which contains given node.

    :param ModelProto model: Onnx Model graph proto.
    :param NodeProto node: onnx node
    :return Union[None, GraphProto]: Underlying onnx graph if found else None.
    """
    for graph in get_graphs(model):
        if node in graph.node:
            return graph
    return None


def get_graph_by_initializer(
    model: ModelProto, init: TensorProto
) -> Union[None, GraphProto]:
    """
    Get graph which contains given initializer.

    :param ModelProto model: Onnx Model graph proto.
    :param TensorProto node: Initializer instance.
    :return Union[None, GraphProto]: Underlying onnx graph if found else None.
    """
    for graph in get_graphs(model):
        if init in graph.initializer:
            return graph
    return None


def get_nodes_by_op_type(model: ModelProto, op_type: str) -> List[NodeProto]:
    """
    Get all the nodes from underlying onnx model based on op type.

    :param ModelProto model: Onnx Model graph proto.
    :param str op_type: optype
    :return List[NodeProto]: Underlying onnx nodes.
    """
    nodes = []
    for node in get_nodes(model):
        if node.op_type == op_type:
            nodes.append(node)
    return nodes


def get_nodes_by_op_types(model: ModelProto, op_type_list: List[str]) -> Dict[str, NodeProto]:
    """
    Get all the nodes from underlying onnx model based on op types.

    :param ModelProto model: Onnx Model graph proto.
    :param List op_type_list: A list containing optypes
    :return Dict[str, NodeProto]: Op type and corresponding underlying onnx nodes.
    """
    nodes_by_op_types = defaultdict(list)
    for graph in get_graphs(model):
        for node in graph.node:
            for op_type in op_type_list:
                if node.op_type == op_type:
                    nodes_by_op_types[op_type].append(node)
                    break
    return nodes_by_op_types


def get_graph_by_name(model: ModelProto, graph_name: str) -> Union[None, GraphProto]:
    """
    Get the graph by their name.

    :param ModelProto model: Onnx Model Proto.
    :param str graph_name: name of the graph.
    :return Union[None, GraphProto]: onnx graph proto instance if found else None.
    """
    for graph in get_graphs(model):
        if graph_name == graph.name:
            return graph
    return None


def remove_unused_inputs_outputs(model: ModelProto) -> ModelProto:
    """
    Remove the unused/dangling input and output nodes from graph's inputs
    and outputs.

    :param model: Loaded Onnx Model Proto instance
    :return model: Cleaned model with dangling inputs and outputs removed.
    """
    main_graph = model.graph

    node_by_input_name = get_node_by_input_name(model)
    node_by_output_name = get_node_by_output_name(model)

    def is_dangling_graph_output(graph_output_name: str) -> bool:
        """
        Function to check if the graph output is dangling output (e.g. not an
        output of any nodes.)

        :param str graph_output_name: Name of the graph output tensor to check.
        :return bool: True if the given output is a dangling output, else False.
        """
        if graph_output_name not in node_by_output_name.keys():
            return True
        else:
            return False

    def is_dangling_graph_input(graph_input_name: str) -> bool:
        """
        Function to check if the graph input is dangling input (e.g. not an
        input of any nodes.)

        :param str graph_input_name: Name of the graph input tensor to check.
        :return bool: True if the given input is a dangling input, else False.
        """
        if graph_input_name not in node_by_input_name.keys():
            return True
        else:
            return False

    # Identify and remove any dangling graph inputs and graph outputs
    # Note: We shall not remove dangling input/output of subgraph as they may
    #       have connection to parent graph.
    remove_graph_outputs = [
        g_op for g_op in main_graph.output if is_dangling_graph_output(g_op.name)
    ]
    remove_graph_inputs = [
        g_op for g_op in main_graph.input if is_dangling_graph_input(g_op.name)
    ]

    for r_g_op in remove_graph_outputs:
        main_graph.output.remove(r_g_op)

    for r_g_ip in remove_graph_inputs:
        main_graph.input.remove(r_g_ip)

    return model


def get_shape_from_value_info_proto(
    val_info: onnx.ValueInfoProto,
    use_onnx_def_symbols: bool = False,
) -> List[Union[str, int]]:
    """
    Function to get the shape from value info proto.

    :param val_info: onnx.ValueInfoProto
    :param use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :return List[Union[str, int]]: Tensor shape of given value info proto.
    """
    tensor_shape = []
    tensor_type = val_info.type.tensor_type

    if not tensor_type.HasField("shape"):
        log_warning(f"Shape not found for tensor: {val_info.name}")
        return tensor_shape

    # iterate through dimensions of the shape:
    for d in tensor_type.shape.dim:
        # the dimension may have a definite (integer) value or a symbolic identifier or neither:
        if d.HasField("dim_value"):
            tensor_shape.append(d.dim_value)
        elif d.HasField("dim_param"):
            # unknown dimension with symbolic name
            if use_onnx_def_symbols:
                tensor_shape.append(d.dim_param)
            else:
                tensor_shape.append(-1)
        else:
            tensor_shape.append("?")
    return tensor_shape


def get_tensor_info_from_value_info_proto(
    val_info: onnx.ValueInfoProto, use_onnx_def_symbols: bool = False
) -> TensorInfo:
    """
    Converts tensor's value info into TensorInfo object which posses information
    about name, shape, dtype and layout.

    :param onnx.ValueInfoProto val_info: Tensor information in terms of ValueInfoProto.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :return TensorInfo: Information about tensor in terms of TensorInfo which
        contains its name, shape, dtype and layout.
    """
    tensor_name = val_info.name
    tensor_shape = get_shape_from_value_info_proto(val_info, use_onnx_def_symbols)
    tensor_dtype = elem_type_to_name(val_info.type.tensor_type.elem_type)
    tensor_layout = determine_layout(tensor_shape)
    return TensorInfo(tensor_name, tensor_shape, tensor_dtype, tensor_layout)


def get_input_info(
    model: ModelProto, use_onnx_def_symbols: bool = False
) -> Dict[str, TensorInfo]:
    """
    Get the graph input info as TensorInfo object.

    :param ModelProto model: Onnx ModelProto instance.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :return Dict[str, TensorInfo]: Mapping of graph's input name to its TensorInfo.
    """
    model_inputs = get_inputs(model)
    input_specs = OrderedDict()
    for model_input in model_inputs:
        input_specs[model_input.name] = get_tensor_info_from_value_info_proto(
            model_input, use_onnx_def_symbols
        )
    return input_specs


def get_output_info(
    model: ModelProto, use_onnx_def_symbols: bool = False
) -> Dict[str, TensorInfo]:
    """
    Get the graph output info as TensorInfo object.

    :param ModelProto model: Onnx ModelProto instance.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :return Dict[str, TensorInfo]: Mapping of graph's output name to its TensorInfo.
    """
    model_outputs = get_outputs(model)
    output_specs = OrderedDict()
    for model_output in model_outputs:
        output_specs[model_output.name] = get_tensor_info_from_value_info_proto(
            model_output, use_onnx_def_symbols
        )
    return output_specs


def get_millify_number(num: int) -> int:
    """
    Convert the large numbers to human readble format.

    :param int num: number to convert
    :return int: converted number in readble format
    """
    millnames = ["", " Thousand", " Million", " Billion", " Trillion"]
    n = float(num)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def get_model_params(model: ModelProto) -> int:
    """
    Return the total number of model parameters.

    :param ModelProto model: Onnx model proto instance.
    :return int: Total parameters in model.
    """
    onnx_constants_mapping = get_initializer_mappings(model)
    params = 0
    for _, const_tensor in onnx_constants_mapping.items():
        try:
            weight_shape = [d for d in const_tensor.dims]
            params += np.prod(weight_shape)
        except Exception as e:
            log_warning(f"Invalid weight params: {e}")
            continue
    return params


def get_model_size(model: ModelProto) -> float:
    """
    Provides the size of the model in GB.

    :param model (ModelProto): Onnx model proto instance.
    :return size_gb (float): size of the model in GB
    """
    NUM_GB_BYTES = 1024**3
    size_bytes = model.ByteSize()
    size_gb = size_bytes / NUM_GB_BYTES
    return size_gb


def get_opset_version(model: ModelProto) -> int:
    """
    Return the model opset version for default domain.

    :param ModelProto model: Onnx model proto instance.
    :raises RuntimeError: If no default domains found in model.
    :return int: opset version of onnx domain
    """
    for opset in model.opset_import:
        if opset.domain in ["", "ai.onnx"]:
            return opset.version
    raise RuntimeError("Onnx model has no opset for default domain")


def get_ir_version(model: ModelProto) -> int:
    """
    Return the model IR version.

    :param ModelProto model: Onnx model proto instance.
    :return int: IR version of onnx model.
    """
    return model.ir_version


def remove_zero_dim_input_from_node(model: ModelProto, op_type: str) -> ModelProto:
    """
    This function is removing initializer's zero Sized tensor from the Input node

    :param ModelProto model: Input ONNX Model.
    :param str op_type: ONNX operation type.
    :return ModelProto: Modified ONNX Model.
    """
    # Aggregating zero dims tensors from initializer.
    filtered_nodes = get_nodes_by_op_type(model, op_type)
    zero_dim_init_dict = {
        init_name: init
        for init_name, init in get_initializer_mappings(model).items()
        if init.dims == [0]
    }

    if not zero_dim_init_dict:
        return model

    node_idxs_to_check = set()
    initializers_to_remove = set()
    # Removing node inputs which are having zero dimensions
    # and part of initializer also.
    for node_idx, node in enumerate(filtered_nodes):
        for input in node.input:
            if input not in zero_dim_init_dict:
                continue
            init = zero_dim_init_dict[input]
            initializers_to_remove.add(init.name)

            graph = get_graph_by_node(model, node)
            node_graph_idx = list(graph.node).index(node)
            graph.node[node_graph_idx].input.remove(input)

            node_idxs_to_check.add(node_idx)

    # Removing Initializer which are having zero dimensions.
    for name in initializers_to_remove:
        init = zero_dim_init_dict[name]
        graph = get_graph_by_initializer(model, init)
        graph.initializer.remove(init)

    # Removing nodes which are having zero inputs
    for idx in node_idxs_to_check:
        node = filtered_nodes[idx]
        if len(node.input) == 0:
            graph = get_graph_by_node(model, node)
            graph.node.remove(node)
    return model


def remove_zero_dim_init_input_from_concat(model: ModelProto) -> ModelProto:
    """
    This function is removing initializer's zero Sized tensor from the concat node

    :param ModelProto model: Input ONNX Model.
    :return ModelProto: Modified ONNX Model.
    """
    return remove_zero_dim_input_from_node(model, "Concat")


def cleanup(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Cleans up the model by removing unused nodes and dangling inputs/outputs
    :param model: Onnx ModelProto
    :return onnx.ModelProto: Cleaned Onnx ModelProto
    """
    # Removing empty input tensors from the Concat Node
    model = remove_zero_dim_init_input_from_concat(model)

    # Traverse back from output node till the input node
    # Remove unused nodes
    # Then remove unused input and outputs
    graph = model.graph

    # Creating two separate dict for traversing the graph easily
    node_by_output_name = {}
    node_by_input_name = {}
    for n in graph.node:
        # Only single node can be referenced by any output node name
        for n_op in n.output:
            node_by_output_name[n_op] = n
        # More than one node can be referenced by any input node name
        # That's why using list of nodes
        for n_ip in n.input:
            if n_ip not in node_by_input_name.keys():
                node_by_input_name[n_ip] = [n]
            else:
                node_by_input_name[n_ip].append(n)

    # Traverse the graph from the graph output nodes.
    visited_nodes = set()
    stack = []

    for g_op in graph.output:
        if g_op.name in node_by_output_name.keys():
            node = node_by_output_name[g_op.name]
            stack.append(node)

    while len(stack) != 0:
        node = stack.pop()
        if node.name in visited_nodes:
            continue
        visited_nodes.add(node.name)
        for ip_name in node.input:
            if ip_name in node_by_output_name.keys():
                parent_node = node_by_output_name[ip_name]
                if parent_node.name not in visited_nodes:
                    stack.append(parent_node)

    # Till now visited_nodes is populated with nodes connected with graph outputs
    remove_nodes = [n for n in graph.node if n.name not in visited_nodes]

    # Remove nodes from the graph as well as from two dictionaries.
    for r_n in remove_nodes:
        graph.node.remove(r_n)

        # Delete its entries from node_by_input_name, node_by_output_name
        for n_op in r_n.output:
            node_by_output_name.pop(n_op)

        for n_ip in r_n.input:
            if len(node_by_input_name[n_ip]) == 1:
                node_by_input_name.pop(n_ip)
            else:
                node_by_input_name[n_ip].remove(r_n)
                # Remove node "r_n" from list if the input is used in multiple nodes
                # We only want to remove that "r_n" node and want to keep other nodes which uses
                # the same input.

    # Remove unused initializers from the graph.
    unused_initializers = [
        init for init in graph.initializer if init.name not in node_by_input_name
    ]

    for init in unused_initializers:
        # Remove the initializers which are not connected to any node.
        graph.initializer.remove(init)

    # Remove unused value_info from the graph.
    unused_value_infos = [
        val_info
        for val_info in graph.value_info
        if val_info.name not in node_by_input_name
    ]

    for val_info in unused_value_infos:
        # Remove the value_info which are not connected to any node.
        graph.value_info.remove(val_info)

    # Clean any dangling input or output from model's main graph.
    model = remove_unused_inputs_outputs(model)
    return model


def create_node_name(
    graph: GraphProto,
    op_type: str,
    _node_name_suffix: Dict[str, int],
    name_prefix: str = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Create a unique node name that starts with a prefix (default is operator type).
    The name will not be duplicated with any name that generated or existed in current graphs.

    :param graph (GraphProto): Onnx GraphProto instance of model
    :param op_type (str): Operator type for which the name is to be generated.
    :param _node_name_suffix (Dict[str, int]): Dict mapping of node_name to its suffix.
    :param name_prefix (str, optional): Prefix of node name. Defaults to None.
    :returns Tuple[str, Dict[str, int]]: Node name for given node op type and updated
                                         Dict mapping of node_name to its suffix.
    """
    # TODO: This functionality shall be redesigned where node can be created
    #        first and then added to graph and post that we shall call an API
    #        assign_name on graph to address issues related to empty name.
    if name_prefix:
        prefix = name_prefix if name_prefix.endswith("_") else (name_prefix + "_")
    else:
        prefix = op_type + "_"
    suffix: int = 0
    if prefix in _node_name_suffix:
        suffix = _node_name_suffix[prefix] + 1
    else:
        # Check existed node name only once for a prefix as we assume
        # create_node_name is called for every new node in fusion.
        for node in graph.node:
            if node.name and node.name.startswith(prefix):
                try:
                    index = int(node.name[len(prefix) :])
                    suffix = max(index + 1, suffix)
                except ValueError:
                    continue
    # Record the generated suffix so that we can avoid generating duplicated name.
    _node_name_suffix[prefix] = suffix
    return prefix + str(suffix), _node_name_suffix


def create_tensor_name(
    proposed_tensor_name: str, tensor_name_set: Set[str]
) -> Tuple[str, Set[str]]:
    """
    Function to create a new tensor name which doesnt conflict with existing
    tensor names.

    :param proposed_tensor_name (str): Proposed name of the new tensor.
    :param tensor_name_set (Set[str]): Set of output tensor names of the model.
    :returns Tuple[str, Set[str]]: Tuple of updated name of the new tensor and
        updated set of the output tensor names of the model.
    """
    new_name = proposed_tensor_name
    counter = 1
    while new_name in tensor_name_set:
        new_name = f"{proposed_tensor_name.split('_')[0]}_{counter}"
        counter += 1
    tensor_name_set.add(new_name)
    return new_name, tensor_name_set


def assign_names_to_empty_nodes(model: ModelProto) -> None:
    """
    Function to add new node names for nodes whose name property is "".

    :param ModelProto model: Onnx model reference.
    """
    _node_name_suffix_mapping = {}
    for node in get_nodes(model):
        if node.name == "":
            new_node_name, _node_name_suffix_mapping = create_node_name(
                model.graph, node.op_type, _node_name_suffix_mapping
            )
            node.name = new_node_name
    return model


def check_duplicates_in_attribute(graph: GraphProto, attribute_name: str) -> bool:
    """
    Function to check for duplicates in the given attribute_name.

    :param GraphProto model: Graph reference from model.
    :param str attribute_name: Attribute name of the model. e.g. node, input or output.
    :return bool: boolean status indicating graph is valid or not.
    """
    node_check_status = True
    property_name_set = set()
    output_tensor_name_set = set()

    if not hasattr(graph, attribute_name):
        return False

    list_of_attributes = getattr(graph, attribute_name)

    if not isinstance(list_of_attributes, Iterable):
        return False

    for property in list_of_attributes:
        if property.name != "" and property.name not in property_name_set:
            property_name_set.add(property.name)
        else:
            node_check_status = node_check_status and False
            if property.name == "":
                # Graph inputs/outputs are assumed to have some name.
                if attribute_name  == "node":
                    log_debug1(
                        f"Graph checker: No {attribute_name} name found for "
                        f"node with inputs {property.input} and outputs {property.output}."
                    )
            else:
                log_debug1(
                    f"Graph checker: {attribute_name} '{property.name}' is duplicate."
                )

        # Check for intermediate tensors. Each node's output tensor shall have
        # unique name. Two different nodes shall not have same output tensor name.
        if hasattr(property, "output"):
            for node_output in property.output:
                if node_output not in output_tensor_name_set:
                    output_tensor_name_set.add(node_output)
                else:
                    node_check_status = False
                    log_debug1(
                        f"Graph checker: Tensor '{node_output}' is the output of two different nodes."
                    )
                    return node_check_status

    return node_check_status


def graph_checker(model: ModelProto) -> bool:
    """
    Function to check the validity of graph. e.g Node names are present or not,
    Is there any duplicate node names, node's input tensor names or node's
    output tensor names etc. present or not.
    :param ModelProto model: Onnx model
    :return bool: boolean status indicating graph is valid or not.
    """
    node_check_status = True
    for graph in get_graphs(model):
        node_check_status = node_check_status and check_duplicates_in_attribute(
            graph, "node"
        )
        node_check_status = node_check_status and check_duplicates_in_attribute(
            graph, "input"
        )
        node_check_status = node_check_status and check_duplicates_in_attribute(
            graph, "output"
        )
    return node_check_status


def get_node_mappings(model: onnx.ModelProto) -> Dict:
    """
    Node name to nodes mapping.

    :param onnx.ModelProto model:  onnx model.
    :return Dict: Node name to nodes mapping
    """

    return {n.name: n for n in model.graph.node}


def get_initializer_mappings(model: onnx.ModelProto) -> Dict:
    """
    Initializer name to initializer mapping

    :param onnx.ModelProto model: model (onnx.ModelProto): onnx model.
    :return Dict: initializer name to initializer mapping
    """
    return {n.name: n for n in model.graph.initializer}


def get_value_info_proto_mappings(model: onnx.ModelProto) -> Dict:
    """
    Value name to value mapping.
    :param model (onnx.ModelProto): onnx model.
    :returns:Dict: value name to value mapping
    """
    return {v.name: v for v in model.graph.value_info}


def get_node_by_input_name(model: onnx.ModelProto) -> Dict:
    """
    Input name to node mappings.
    :param  loader (onnx.ModelProto): onnx.ModelProto model instance.
    :returns: Dict: Input name to node mappings.
    """
    get_node_by_input = {}
    for n in model.graph.node:
        for n_ip in n.input:
            if n_ip not in get_node_by_input:
                get_node_by_input[n_ip] = [n]
            else:
                get_node_by_input[n_ip].append(n)
    return get_node_by_input


def get_node_by_output_name(model: onnx.ModelProto) -> Dict:
    """
    Output name to node mappings.
    :param model (onnx.ModelProto): OnnxModel model instance.
    :return : Dict: Output name to node mappings.
    """
    get_node_by_output = {}
    for n in model.graph.node:
        for n_op in n.output:
            if n_op not in get_node_by_output:
                get_node_by_output[n_op] = n
    return get_node_by_output


def get_parent_at_any_level(
    output_name: str,
    get_node_by_output_name: Dict[str, NodeProto],
    level: int = 1,
) -> List[NodeProto]:
    """
    Function to get the parent of the specified node at given level.
    level 1 - immediate parent
    level 2 - parent of parent and so on.

    :param str output_name: Output name of the node whose parent needs to be
        identified.
    :param Dict[str, NodeProto] get_node_by_output_name: Dict to get the
        node by its output name.
    :param int level: Parent level to be identified.
    :raises Exception: _description_
    :return List[NodeProto]: List of parent nodes.
    """
    output_node = get_node_by_output_name[output_name]

    def get_parent(node, get_node_by_output_name):
        im_parent_nodes = []
        for inp in node.input:
            if inp in get_node_by_output_name:
                it_node = get_node_by_output_name[inp]
                im_parent_nodes.append(it_node)
        return im_parent_nodes

    parent_nodes = get_parent(output_node, get_node_by_output_name)

    if level == 1:
        return parent_nodes
    elif level > 1:
        for i in range(level - 1):
            iterating_nodes = parent_nodes
            final_nodes = []
            for nodes in iterating_nodes:
                candidate_nodes = get_parent(nodes, get_node_by_output_name)
                for node in candidate_nodes:
                    final_nodes.append(node)
            parent_nodes = final_nodes
        return parent_nodes
    else:
        log_error(f"Can't find the parent at given level: {level}")
        return None


def get_initializer_value(init: onnx.TensorProto, base_path="/") -> np.ndarray:
    """
    Function to get value of initializer of a particular initializer.

    :param onnx.TensorProto init: Initializer instance.
    :return np.ndarray: Numpy array constructed from initializer's data.
    """
    if uses_external_data(init):
        # Due to bug in onnx.external_data_helper.load_external_data_for_tensor
        # we need to copy paste the same implementation, modify it and use
        # it here. The onnx's version was not able to take care of 0-d tensors
        # and was reading all the external data left after the given offset.
        is_zero_dim = np.prod(init.dims) == 0
        load_external_data_for_tensor(init, base_path, is_zero_dim)
        # After loading raw_data from external_data, change the state of tensors
        init.data_location = TensorProto.DEFAULT
        # and remove external data
        del init.external_data[:]

    return numpy_helper.to_array(init)


def get_initializer_by_name(model: ModelProto, name: str) -> Optional[TensorProto]:
    """
    Function to get the initializer of particular name

    :param ModelProto model: Onnx model proto instance.
    :param str name: Name of the initializer to be found.
    :return Optional[TensorProto]: Initializer tensor if found else None.
    """
    for graph in get_graphs(model):
        for tensor in graph.initializer:
            if tensor.name == name:
                return tensor
    return None


def get_dim_from_type_proto(dim) -> Union[str, int]:
    """
    Function to get the dim value or dim param from type proto

    :param onnx.onnx_ml_pb2.TensorShapeProto.Dimension dim: Dimension info from tensor.
    :return Union[str, int]: Dim value or Dim param from onnx model.
    """
    return (
        getattr(dim, dim.WhichOneof("value"))
        if type(dim.WhichOneof("value")) == str
        else None
    )


def get_shape_from_type_proto(type_proto: onnx.TypeProto) -> List:
    """
    Function to get the shape info from type proto

    :param onnx.TypeProto type_proto: Onnx Type proto of tensor.
    :return List: Shape of the tensor type proto as list.
    """
    return [get_dim_from_type_proto(d) for d in type_proto.tensor_type.shape.dim]


def get_tensor_dtype(tensor: Union[np.ndarray, onnx.ValueInfoProto, TensorProto]) -> np.dtype:
    """
    Get the numpy dtype of the given value info or tensor proto object.

    :param Union[onnx.ValueInfoProto, TensorProto] onnx_tensor: Value info or tensor proto object.
    :return np.dtype: Numpy dtype of the given tensor.
    """
    if isinstance(tensor, np.ndarray):
        return tensor.dtype

    if isinstance(tensor, TensorProto):
        tensor_type = tensor.data_type
    else:
        tensor_type = tensor.type.tensor_type.elem_type

    if tensor_type in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE:
        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_type]

    return None

def get_tensor_shape(tensor: Union[np.ndarray, onnx.ValueInfoProto, TensorProto]) -> np.dtype:
    """
    Get the numpy dtype of the given value info or tensor proto object.

    :param Union[onnx.ValueInfoProto, TensorProto] onnx_tensor: Value info or tensor proto object.
    :return np.dtype: Numpy dtype of the given tensor.
    """
    tensor_shape = None
    if isinstance(tensor, np.ndarray):
        tensor_shape = tensor.shape

    elif isinstance(tensor, TensorProto):
        tensor_shape = list(tensor.dims)

    elif isinstance(tensor, onnx.ValueInfoProto):
        tensor_shape = get_shape_from_value_info_proto(tensor)
    return tensor_shape

def extract_large_model_helper(
    loader: FrameworkModelLoader,
    input_tensor_names: List[str],
    output_tensor_names: List[str],
) -> FrameworkModelLoader:
    """
    Function to extract subgraphs from large models (> 2 GB)

    :param FrameworkModelLoader loader: Onne model loader instance.
    :param List[str] input_tensor_names: List of input tensors which should
        be made subgraph's input.
    :param List[str] output_tensor_names: List of output tensors which shouldbe made subgraph's outputs.
    :return FrameworkModelLoader: Extracted subgraph in the form of FrameworkModelLoader instance
    """
    # Create a deepcopy of original model and remove unnecessary nodes.
    result_loader = loader.clone_loader()
    result_loader.utils.native_shape_inference()
    result_loader.utils.add_inputs(input_tensor_names)
    result_loader.utils.add_outputs(output_tensor_names)
    for model_output in result_loader.get_output_names():
        # To extract subgraph remove the outputs which are not
        # required.
        if model_output not in output_tensor_names:
            result_loader.utils.remove_outputs([model_output])
    result_loader.utils.clean_model().topological_sort()
    return result_loader


def get_extracted_model(
    loader: FrameworkModelLoader,
    input_tensor_names: List[str],
    output_tensor_names: List[str],
) -> FrameworkModelLoader:
    """
    Function to extract subgraphs from given onnx model loader.

    :param FrameworkModelLoader loader: Onnx model loader instance.
    :param List[str] input_tensor_names: List of input tensors which should
        be made subgraph's input.
    :param List[str] output_tensor_names: List of output tensors which should
        be made subgraph's outputs
    :return FrameworkModelLoader: Extracted subgraph in the form of FrameworkModelLoader instance
    """
    if (not model_uses_external_data(loader.model, True) and get_model_size(loader.model)) > 2 or \
        parse_version(onnx.version.version) <= parse_version("1.8.0"):
        log_debug1("Model size is > 2GB or onnx version is <= 1.8.0. Calling "
                   "own graph extractor APIs.")
        return extract_large_model_helper(loader, input_tensor_names, output_tensor_names)
    else:
        # Currently, below subgraph implementation will work for models < 2GB
        # and models > 2GB. For models > 2GB, in the below implementation the
        # model shall be passed without weights.
        log_debug1("Model size is < 2GB or without external weights. Calling "
                   "Onnx's graph extractor APIs.")
        from onnx.utils import Extractor

        loader_tmp = loader.clone_loader()
        loader_tmp.utils.native_shape_inference()
        e = Extractor(loader_tmp.model)
        extracted_model = e.extract_model(
            input_tensor_names, output_tensor_names)
        loader_tmp.update_model(extracted_model)
        loader_tmp.utils.clean_model().topological_sort()
        return loader_tmp


def get_value_info_by_name(
    model: ModelProto, name: str
) -> Optional[onnx.ValueInfoProto]:
    """
    Function to get the value info by name.

    :param ModelProto model: Onnx model proto instance.
    :param str name: Name of the value info to be found.
    :return Optional[onnx.ValueInfoProto]: Value info proto if found else None.
    """
    for v in model.graph.value_info:
        if v.name == name:
            return v
    for v in model.graph.input:
        if v.name == name:
            return v
    for v in model.graph.output:
        if v.name == name:
            return v
    return None


def get_attribute_value(attr: AttributeProto) -> List[int]:
    """
    Function to get the value of attribute from attribute proto

    :param AttributeProto attr: AttributeProto instance.
    :return List[int]: Values extracted from AttributeProto as np array.
    """
    value = None
    if not isinstance(attr, onnx.AttributeProto):
        return None

    if attr.type == onnx.AttributeProto.TENSOR:
        dtype = np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[attr.t.data_type])
        data = attr.t.raw_data
        value = np.frombuffer(data, dtype=dtype, count=(len(data) // dtype.itemsize))
    elif attr.type == onnx.AttributeProto.STRING:
        value = attr.s
        value = value.decode() if isinstance(value, bytes) else value
    else:
        value = helper.get_attribute_value(attr)

    return value


def get_shape(
    model: ModelProto, name: str, use_onnx_def_symbols: bool = False
) -> List[int]:
    """
    Return the shape of tensor.
    Note: Please perform shape inference before calling this API to introduce
    shapes in the model.

    :param ModelProto model: Onnx model proto instance.
    :param str name: Name of the tensor.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :raises RuntimeError: If shape of the tensor can't be found.
    :return List[int]: Shape of the given tensor.
    """
    v = get_value_info_by_name(model, name)
    if v is not None:
        return get_shape_from_value_info_proto(v, use_onnx_def_symbols)
    raise RuntimeError('Cannot get shape of "{}"'.format(name))


def get_all_tensors(model: onnx.ModelProto, include_attributes: bool=True) -> List[onnx.TensorProto]:
    """
    Get the list of all the tensors e.g. Initializer and constant attribute tensors
    from the onnx model.

    :param onnx.ModelProto model: Onnx model proto instance.
    :param bool include_attributes: Include tensors which are part of node's
        attributes. Defaults to True.
    :return List[onnx.TensorProto]: List of all the tensors in the model.
    """
    tensors = []
    for graph in get_graphs(model):
        tensors.extend(graph.initializer)

        if not include_attributes:
            continue
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    tensors.append(attribute.t)
                else:
                    tensors.extend(attribute.tensors)
    return tensors


def transfer_constants_to_initializers(model: onnx.ModelProto):
    """
    Transfers tensor from constant nodes in an ONNX model to initializers.

    This function iterates through the nodes in the graph of the given ONNX model.
    If a node is of type 'Constant' and has a single attribute of type 'TENSOR',
    it extracts the tensor from the node's attributes, renames the tensor to match
    the node's output, and adds it to the list of initializers. The constant nodes
    are then removed from the graph.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.

    Returns:
        None: The function modifies the model in place.
    """
    graph = model.graph

    new_initializers = []
    nodes_to_remove = []

    for node in graph.node:
        # Check if the node is of type 'Constant' and has exactly one attribute
        if node.op_type == 'Constant' and len(node.attribute) == 1:
            attr = node.attribute[0]
            # Ensure the attribute is named 'value' and is of type 'TENSOR'
            if attr.name == 'value' and attr.type == onnx.AttributeProto.TENSOR:
                tensor = attr.t
                # Rename the tensor to match the node's output
                tensor.name = node.output[0]
                new_initializers.append(tensor)
                nodes_to_remove.append(node)

    # Add the new initializers to the graph
    for tensor in new_initializers:
        graph.initializer.append(tensor)

    # Remove the constant nodes from the graph
    for node in nodes_to_remove:
        graph.node.remove(node)


def convert_model_to_external_data(
    model: onnx.ModelProto, file_name: str, include_attributes:bool = True
) -> bool:
    """
    Convert the tensors in the given model by updating their data location and
    data offset parameters. Note: This API will not convert data to external data
    but it will populate external data fields in each tensors. Actual conversion
    to external data will happen via onnx.save API.

    :param onnx.ModelProto model: Onnx model proto instance.
    :param bool include_attributes: Include tensors which are part of node's
        attributes. Defaults to True.
    :param str file_name: Path of the onnx external data file.
    :return bool: Status indicating whether the model has weights which are set
        to external data or not. Some weight which are having size less than
        ONNX_EXTERNAL_DATA_THRESHOLD, won't be converted to external data.
    """
    changed_to_external_data = False
    for tensor in get_all_tensors(model, include_attributes):
        if tensor.HasField("raw_data"):
            if sys.getsizeof(tensor.raw_data) >= ONNX_EXTERNAL_DATA_THRESHOLD:
                set_external_data(tensor, file_name)
                changed_to_external_data = True
            continue

        # Data is stored as float_data or int32_data in TensorProto.
        try:
            tensor_data = numpy_helper.to_array(tensor)
            onnx_tensor = numpy_helper.from_array(tensor_data)
            if sys.getsizeof(onnx_tensor.raw_data) >= ONNX_EXTERNAL_DATA_THRESHOLD:
                set_external_data(tensor, file_name)
                changed_to_external_data = True
            continue
        except:
            continue
    return changed_to_external_data


def remove_external_data_from_model(
    model: onnx.ModelProto, include_attributes: bool = True
) -> None:
    """
    Remove external data fields from Model Proto object.

    :param onnx.ModelProto model: Onnx model reference.
    :param bool include_attributes: Include tensors which are part of node's
        attributes. Defaults to True.
    """
    for tensor in get_all_tensors(model, include_attributes):
        if uses_external_data(tensor):
            tensor.data_location = TensorProto.DEFAULT
            del tensor.external_data[:]


def write_external_data_tensors(
    model: ModelProto, dir_path: str, include_attributes:bool = True
) -> None:
    """
    Serializes data for all the tensors which have data location set to
    TensorProto.External. Model will be replaced inplace.

    Note: This function also strips basepath information from all tensors'
    external_data fields.

    :param onnx.ModelProto model: Onnx model reference.
    :param str dir_path: System path to the directory which should be treated
        as base path for external data.
    :param bool include_attributes: Include tensors which are part of node's
        attributes. Defaults to True.
    """
    for tensor in get_all_tensors(model, include_attributes):
        # Writing to external data happens in 2 passes:
        # 1. Tensors with raw data which pass the necessary conditions (size threshold etc) are marked for serialization
        # 2. The raw data in these tensors is serialized to a file
        # Thus serialize only if tensor has raw data and it was marked for serialization
        if uses_external_data(tensor) and tensor.HasField("raw_data"):
            save_external_data(tensor, dir_path)
            tensor.ClearField("raw_data")


def load_external_data_for_tensor(
    tensor: TensorProto, base_dir: str, is_zero_dim: bool=False
) -> None:
    """
    Loads data from an external file for tensor.
    Ideally TensorProto should not hold any raw data but if it does it will be
    ignored.

    :param TensorProto tensor: TensorProto object inwhich data is to be loaded.
    :param str base_dir: Directory that contains the external data.
    :param bool is_zero_dim: Boolean flag indicating whether the given tensor
        is of 0-d shape or not, defaults to False
    """
    info = ExternalDataInfo(tensor)
    file_location = info.location.lstrip('/.')
    external_data_file_path = os.path.join(base_dir, file_location)

    with open(external_data_file_path, "rb") as data_file:
        if info.offset:
            data_file.seek(info.offset)

        if info.length:
            tensor.raw_data = data_file.read(info.length)
        elif is_zero_dim:
            tensor.raw_data = data_file.read(0)
        else:
            tensor.raw_data = data_file.read()


def load_external_data_for_model(
    model: ModelProto, base_dir: str, include_attributes: bool = True
) -> None:
    """
    Loads external tensors into model

    :param onnx.ModelProto model: Onnx model reference.
    :param str base_dir: Directory that contains external data
    :param bool include_attributes: Include tensors which are part of node's
        attributes. Defaults to True.
    """
    for tensor in get_all_tensors(model, include_attributes):
        if uses_external_data(tensor):
            # Due to bug in onnx.external_data_helper.load_external_data_for_tensor
            # we need to copy paste the same implementation, modify it and use
            # it here. The onnx's version was not able to take care of 0-d tensors
            # and was reading all the external data left after the given offset.
            is_zero_dim = np.prod(tensor.dims) == 0
            load_external_data_for_tensor(tensor, base_dir, is_zero_dim)
            # After loading raw_data from external_data, change the state of tensors
            tensor.data_location = TensorProto.DEFAULT
            # and remove external data
            del tensor.external_data[:]


def model_uses_external_data(
    model: ModelProto, include_attributes: bool = True
) -> bool:
    """
    Checks whether the model has any tensor which is stored in external data.

    :param ModelProto model: Onnx model object.
    :param bool include_attributes: Include tensors which are part of node's
        attributes. Defaults to True.
    :return bool: True if model has any tensor which is externally stored,
        else False.
    """
    # FIXME: Here, we are assuming that if even a single tensor in the model is
    # using external data then the entire model is full of tensors with external
    # data. The usage of this API shall take care of this assumption.
    return any(map(uses_external_data, get_all_tensors(model, include_attributes)))


def save_model(model: ModelProto, path: Text, restore_back: bool=True) -> None:
    """
    Save the onnx model to disk.

    :param ModelProto model: Onnx model proto instance.
    :param Text path: Path at which the model is to be saved.
    :param bool restore_back: Restore the external data back to same model object.
        This is applicable only when model is > 2GB. Defaults to True.
    """
    make_dirs(path)
    model_name = "qnn_onnx_converter_model.onnx"
    if os.path.isfile(path):
        model_name = os.path.splitext(os.path.basename(path))[0]

    weight_file_name = model_name.split(".")[0] + ".data"

    # onnx.save doesn't support saving the model > 2GB ,
    if get_model_size(model) > 2:  # GB
        model_dir = os.path.abspath(os.path.dirname(path))
        status = convert_model_to_external_data(model,
                                                file_name=weight_file_name)
        onnx.save(model, path)
        if restore_back:
            load_external_data_for_model(model, model_dir)
            # For onnx-1.6, the load_external_data_for_model API loads the external
            # data but it is not removing external data related artifacts from model
            # proto which is wrong.
            if parse_version(onnx.version.version) <= parse_version("1.6.0"):
                remove_external_data_from_model(model)
    else:
        onnx.save(model, path)


def create_ort_session(
    model_or_path: Union[str, ModelProto],
    optimize: bool=False,
    custom_lib: Optional[str] = None,
    graph_optimization_level=None,
    **kwargs,
) -> Tuple:
    """
    Function to create Onnx Runtime Session and Verify the Correctness
    If loading is successful, that will verify the model correctness on ort.

    :param Union[str, ModelProto] model_or_path: Onnx model in form of
        ModelProto instance or model path.
    :param bool optimize: Optimize the model using onnxruntime and return the
        optimized model, defaults to False
    :param Optional[str] custom_lib: Path of CustomOp library for onnxruntime
        session, defaults to None
    :param Optional[onnxruntime.GraphOptimizationLevel]
        graph_optimization_level: Graph Optimization to be applied as per the
        level., defaults to None
    :return Tuple[bool, Union[None, onnxruntime.InferenceSession],
        Union[None, ModelProto]]: Tuple of boolean status, Onnx Runtime Session
        Object and Optimized ModelProto.
    """
    # TODO: Can we use io.BytesIO() for model < 2gb, to transfer them in memory to
    # onnxruntime session.

    # Models having custom op can't run on onnxruntime session.
    try:
        import onnxruntime

        optimized_model = None
        sess_options = onnxruntime.SessionOptions()
        # Disable the Warnings from Onnx Runtime During the Execution
        sess_options.log_severity_level = 3
        if graph_optimization_level:
            sess_options.graph_optimization_level = graph_optimization_level
        # Load the Custom lib is any
        if custom_lib:
            sess_options.register_custom_ops_library(custom_lib)
        execution_providers = ["CPUExecutionProvider"]
        status = True

        try:
            tmpdirname = os.getenv("QAIRT_TMP_DIR", default=tempfile.gettempdir())
            tmpdirname = tempfile.mkdtemp(dir=tmpdirname)
        except Exception as e:
            raise RuntimeError("Unable to dump external data to the disk "
                    f"at: {tmpdirname}. It may be due to permission or "
                    "space issue. Try setting QAIRT_TMP_DIR environment variable "
                    "to different location.") from e
        if optimize:
            sess_options.optimized_model_filepath = os.path.join(
                tmpdirname, "model_optimized.onnx"
            )

        if isinstance(model_or_path, str):
            session = onnxruntime.InferenceSession(
                model_or_path, sess_options, providers=execution_providers, **kwargs
            )
        elif isinstance(model_or_path, ModelProto):
            temp_model_path = os.path.join(tmpdirname, "model.onnx")

            save_model(model_or_path, temp_model_path)

            session = onnxruntime.InferenceSession(
                temp_model_path,
                sess_options,
                providers=execution_providers,
                **kwargs,
            )
        else:
            raise RuntimeError(
                "Invalid form of model provided for onnxruntime " "inference."
            )

        if optimize:
            optimized_model = onnx.load(sess_options.optimized_model_filepath)
        if os.path.exists(tmpdirname):
            shutil.rmtree(tmpdirname)
    except ImportError as e:
        session = None
        status = False
        optimized_model = None
        log_warning(
            "Onnxruntime not found in current environment. Hence "
            "Onnxruntime session is not created."
        )

    except Exception as e:
        session = None
        status = False
        optimized_model = None
        log_warning(f"Creation of ORT session is failed with Exception : {str(e)}")

    return status, session, optimized_model


def run_ort_inference(
    model_or_path: Union[str, ModelProto],
    input_data: Dict[Text, np.ndarray],
    output_node_names: List[Text],
    custom_lib: Optional[Text] = None
) -> Tuple[bool, Dict[Text, np.ndarray]]:
    """
    Run the inference of given onnx model via Onnx Runtime

    :param Union[str, ModelProto] model_or_path: Onnx model in form of
        ModelProto instance or model path.
    :param Dict[Text, np.ndarray] input_data: Dict containing input tensor name
        to corresponding tensor data.
    :param List[Text] output_node_names: List of output names.
    :param Optional[Text] custom_lib: Path of custom lib, defaults to None
    :return Tuple[bool, Dict[Text, np.ndarray]]: Tuple of two values. 1st value
        represents the status of inference and 2nd value represents the Dict
        containing output tensor name as key and its computed numpy array output as value.
    """
    status, session, _ = create_ort_session(
        model_or_path, optimize=False, custom_lib=custom_lib
    )
    if not status:
        log_error("Failed to create Onnxruntime Session.")
        return False, None

    model_output_names = [x.name for x in session.get_outputs()]

    if not output_node_names:
        output_node_names = model_output_names
    else:
        for output_name in output_node_names:
            if output_name not in model_output_names:
                log_error(
                    f"Given output: {output_name} is not found in "
                    "model's outputs. Please create that tensor as model's "
                    "output and then supply for inference."
                )
                return False, None

    input_node_names = [x.name for x in session.get_inputs()]
    filtered_input_data = {}
    for input_name in input_node_names:
        if input_name in input_data:
            filtered_input_data[input_name] = input_data[input_name]
        else:
            log_error(
                f"Inference input data for input: {input_name} is "
                "not found. Please provide the same for inference."
            )
            return False, None

    try:
        import onnxruntime
    except ImportError as e:
        log_warning("Onnxruntime not found in current environment. Hence " \
            "Onnxruntime inference is not performed.")
        return False, None

    run_options = onnxruntime.RunOptions()
    run_options.log_severity_level = 3
    outputs = OrderedDict(
        zip(
            output_node_names,
            session.run(
                output_node_names, filtered_input_data, run_options=run_options
            ),
        )
    )
    return True, outputs


def order_repeated_field(
    repeated_proto: AttributeProto, key_name: str, order: List[str]
) -> None:
    """
    Function to sort the fields in NodeProto.

    :param AttributeProto repeated_proto: NodeProto of a node
    :param str key_name: key_name for each attribute
    :param List[str] order: List of arguments for a node
    """
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))


def make_tensor(name: str, data: Union[np.ndarray, List]) -> TensorProto:
    """
    Function to generate TensorProto object based on given datatype, dims and values.

    :param str name: Name of the TensorProto.
    :param Union[np.ndarray, List] data: Actual data to be used for the TensorProto.
    :return TensorProto: return tensor proto.
    """
    if isinstance(data, List):
        data = np.array(data, dtype=np.float32)

    tensor = helper.make_tensor(
        name=name,
        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[data.dtype],
        dims=data.shape,
        vals=data.flatten().tolist(),
    )
    return tensor


def make_node(
    op_type: str,
    inputs: List[str],
    outputs: List[str],
    name: Optional[str] = None,
    doc_string: Optional[str] = None,
    domain: Optional[str] = None,
    **kwargs: Dict,
) -> NodeProto:
    """
    Function to generate node based on given params and doc_string

    :param str op_type: Node operator type
    :param List[str] inputs: List of input node names
    :param List[str] outputs: List of output node names
    :param Optional[str] name: Name of the node. Defaults to None.
    :param Optional[str] doc_string: Doc string used to describe the graph.
        Defaults to None.
    :param Optional[str] domain: Domain name for the node. Defaults to None.
    :return NodeProto: NodeProto of the generated Node
    """
    node = helper.make_node(
        op_type, inputs, outputs, name, doc_string, domain, **kwargs
    )
    if doc_string == "":
        node.doc_string = ""
    order_repeated_field(node.attribute, "name", kwargs.keys())
    return node


def make_graph(*args: List, doc_string: str = "", **kwargs: Dict) -> GraphProto:
    """
    Function to generate graph based on given args and doc_string.

    :param str doc_string: Doc string used to describe the graph, defaults to ""
    :return GraphProto: GraphProto instance based on given inputs, outputs,
        value_infos and nodes required for generating graph.
    """
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == "":
        graph.doc_string = ""
    return graph


def opset_to_ir_version(opset_version: int) -> int:
    """
    Function to get the IR version corresponding to opset_version.

    :param int opset_version:: Opset version of default domain which is "" or "ai.onnx".
    :return int: IR Version corresponding to Opset Version.
    """
    if ("ai.onnx", opset_version) not in helper.OP_SET_ID_VERSION_MAP:
        log_warning(
            "Fail to get the IR version for given opset_version: {opset_version}. "
            "Using IR Version 6 by default."
        )
        return 6

    return helper.OP_SET_ID_VERSION_MAP[("ai.onnx", opset_version)]


def remove_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove shape information from the onnx model.

    :param onnx.ModelProto model: Onnx model reference.
    :return onnx.ModelProto: Updated onnx model.
    """
    input_val_info = get_inputs(model)
    output_val_info = get_outputs(model)
    for val_info in itertools.chain(model.graph.value_info, output_val_info):
        if val_info in input_val_info:
            continue
        val_info.type.tensor_type.ClearField("shape")
    return model


def unload_external_data(
    model: ModelProto, include_attributes:bool=True
) -> Tuple[bool, str]:
    """
    Unload external data from given model into temporary file.

    :param onnx.ModelProto model: Onnx model reference.
    :param bool include_attributes: Include tensors which are part of node's
        attributes. Defaults to True.
    :raises RuntimeError: If the external data can't be dumped to temp directory.
    :return Tuple[bool, str]: Tuple of 2 values.
        - First value indicates whether the external data is dumped to a file
          or not. If model has tensors whose size is less than
          ONNX_EXTERNAL_DATA_THRESHOLD, then those tensors won't be converted to
          external data. If model has all the tensors like that then in that case
          this value will be False. Otherwise it will be true.
        - Second value represents path to the file containing external data.
    """
    try:
        tmp_dir = os.getenv("QAIRT_TMP_DIR", default=tempfile.gettempdir())
        tmp_dir = tempfile.mkdtemp(dir=tmp_dir)

        temp_file_name = "model.data"
        temp_data_file = os.path.join(tmp_dir, temp_file_name)
        status = convert_model_to_external_data(model,
                                                file_name=temp_file_name,
                                                include_attributes=include_attributes)
        if not status:
            shutil.rmtree(tmp_dir)
            return False, None
        write_external_data_tensors(model,
                                    tmp_dir,
                                    include_attributes=include_attributes)
    except Exception as e:
        raise RuntimeError("Unable to dump external data to the disk at: "
                f"{tmp_dir}. It may be due to permission or space issue. "
                "Try setting QAIRT_TMP_DIR environment variable to different "
                "location.") from e
    return True, temp_data_file


def reload_external_data(model: ModelProto, external_data_path: str, remove_file: bool=True, include_attributes:bool=True) -> ModelProto:
    """
    Reload external data into given model from provided external_data_path.

    :param onnx.ModelProto model: Onnx model reference.
    :param str external_data_path: Path pointing to external data file.
    :param bool remove_file: Remove external data file after loading. Defaults
        to True.
    :param bool include_attributes: Include tensors which are part of node's
        attributes. Defaults to True.
    :return onnx.ModelProto: Updated onnx model with external data loaded in it.
    """
    load_external_data_for_model(model, os.path.dirname(external_data_path), include_attributes)
    if remove_file:
        os.remove(external_data_path)
    return model


def convert_data_to_raw_data(model: ModelProto, include_attributes: bool) -> None:
    """
    Convert the tensor data from float_data or int32_data into raw data of
    TensorProto.

    :param ModelProto model: Onnx model reference. This model should have all
        the weights in it.
    :param bool include_attributes: Include tensors which are part of node's
        attributes. Defaults to True.
    """
    def get_new_tensor(tensor):
        tensor_data = numpy_helper.to_array(tensor)
        new_tensor = numpy_helper.from_array(tensor_data, name=tensor.name)
        return new_tensor

    for graph in get_graphs(model):
        for idx, tensor in enumerate(graph.initializer):
            graph.initializer.remove(tensor)
            new_tensor = get_new_tensor(tensor)
            graph.initializer.insert(idx, new_tensor)

        if not include_attributes:
            continue
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    tensor = attribute.t
                    new_tensor = get_new_tensor(tensor)
                    attribute.t.CopyFrom(new_tensor)
                else:
                    for _idx, _tensor in enumerate(attribute.tensors):
                        attribute.tensors.remove(_tensor)
                        new_tensor = get_new_tensor(tensor)
                        attribute.tensors.insert(_idx, new_tensor)


def update_tensorproto_paths(model: ModelProto, base_path: str) -> bool:
    """
    Update external data paths in TensorProto attributes to absolute paths.

    This function checks each node and initializer in the ONNX model for
    TensorProto attributes with external data. It updates the 'location'
    key to an absolute path based on the provided base_path.

    Returns True if all weights are loaded; otherwise, returns False.

    :param ModelProto model: The ONNX model to update.
    :param str base_path: Base path for converting relative paths.
    """
    weights_loaded = True
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == attr.TENSOR and attr.t.HasField('data_location') and attr.t.data_location == TensorProto.EXTERNAL:
                weights_loaded = False
                for external_data in attr.t.external_data:
                    if external_data.key == 'location':
                        # Check if the path is already absolute
                        if not os.path.isabs(external_data.value):
                            external_data.value = os.path.abspath(os.path.join(base_path, external_data.value))

    # Update paths in initializers
    for initializer in model.graph.initializer:
        if initializer.HasField('data_location') and initializer.data_location == TensorProto.EXTERNAL:
            weights_loaded = False
            for external_data in initializer.external_data:
                if external_data.key == 'location':
                    # Check if the path is already absolute
                    if not os.path.isabs(external_data.value):
                        external_data.value = os.path.abspath(os.path.join(base_path, external_data.value))
    return weights_loaded


def get_variable_inputs(model: onnx.ModelProto, node: onnx.NodeProto) -> List[str]:
    """
    Get the list of non-initializer/variable/runtime defined inputs of given node.

    :param onnx.ModelProto model: Onnx model reference.
    :param onnx.NodeProto node: Node whose inputs to be checked.
    :return List[str]: List of inputs which are variable inputs of given node.
    """
    filtered_inputs = []
    init_dict = get_initializer_mappings(model)
    for inp in node.input:
        if inp not in init_dict:
            filtered_inputs.append(inp)
    return filtered_inputs


def get_constant_inputs(model: onnx.ModelProto, node: onnx.NodeProto) -> List[str]:
    """
    Get the list of initializer/constant inputs of given node.

    :param onnx.ModelProto model: Onnx model reference.
    :param onnx.NodeProto node: Node whose inputs to be checked.
    :return List[str]: List of inputs which are constant inputs of given node.
    """
    filtered_inputs = []
    init_dict = get_initializer_mappings(model)
    for inp in node.input:
        if inp in init_dict:
            filtered_inputs.append(inp)
    return filtered_inputs
