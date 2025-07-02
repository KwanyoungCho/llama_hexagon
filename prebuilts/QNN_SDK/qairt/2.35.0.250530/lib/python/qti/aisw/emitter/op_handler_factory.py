# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" Op Attribute handler factories for Emitter model """
# Running into a pylint segfault
# pylint: skip-file
import abc
import collections
import logging
import io
from typing import Any, Optional

from qti.aisw.emitter.utils.config import is_custom_ir_op
from qti.aisw.emitter.utils.config import CustomOpInfo
from qti.aisw.emitter.utils.prepared_model_definition_manager import (
    ModelDefinitionManager, StructuredModelDefinitionManager,
)
from qti.aisw.emitter.utils.op_attr_handler import InstanceNormAttrHandler, SoftPlusAttrHandler, PoolAttrHandler

import qti.aisw.emitter.ir_graph_op_handler as op_handler
from qti.aisw.converters.common import ir_graph as ir_graph_lib

IrGraph, IrOp, IrTensor, IrStaticTensor = (ir_graph_lib.IrGraph, ir_graph_lib.IrOp,
                                           ir_graph_lib.IrTensor, ir_graph_lib.IrStaticTensor)

logger = logging.getLogger('TorchEmitter')

class OpHandlerStateFactory(abc.ABC):
    """ Abstract factory class for creating Attribute handlers for different frameworks """

    def __init__(self, ir_graph: IrGraph, model: Any):
        self.ir_graph = ir_graph
        self.model = model

    @abc.abstractmethod
    def create_instance_norm_attr_handler(self) -> InstanceNormAttrHandler:
        """ Get InstanceNorm attribute handler for the given model type """

    @abc.abstractmethod
    def create_soft_plus_attr_handler(self) -> SoftPlusAttrHandler:
        """ Get SoftPlus attribute handler for the given model type """

    @abc.abstractmethod
    def create_pool_attr_handler(self) -> PoolAttrHandler:
        """ Get Pool attribute handler for the given model type """


class IRDefaultFactory(OpHandlerStateFactory):
    """ Factory class for creating Attribute handlers for IR Graph """

    def __init__(self, ir_graph: IrGraph):
        super().__init__(ir_graph, None)

    # pylint: disable=no-self-use
    def create_instance_norm_attr_handler(self) -> InstanceNormAttrHandler:
        """ Get InstanceNorm attribute handler """
        return InstanceNormAttrHandler()

    # pylint: disable=no-self-use
    def create_soft_plus_attr_handler(self) -> SoftPlusAttrHandler:
        """ Get SoftPlus attribute handler """
        return SoftPlusAttrHandler()

    # pylint: disable=no-self-use
    def create_pool_attr_handler(self) -> PoolAttrHandler:
        """ Get Pool attribute handler """
        return PoolAttrHandler()

    # pylint: disable=unused-argument
    @staticmethod
    def is_from_linear_without_bias(op: IrOp) -> bool:
        """ Check if the given op was created from a Linear layer with bias set to False """
        return False

    # pylint: disable=unused-argument
    @staticmethod
    def is_parameter(tensor: IrStaticTensor) -> bool:
        """ Check if the given static tensor is a parameter """
        return False

    # pylint: disable=unused-argument
    @staticmethod
    def check_if_tensor_is_trainable_param(tensor: IrStaticTensor) -> bool:
        """ Check if the given static tensor is a trainable parameter """
        return False


class OpHandlerState:
    """
    Class for handling state and attributes of ops and params
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, ir_graph: IrGraph, file: io.TextIOWrapper, keep_linear_without_bias: bool, is_block_extraction_flow: bool = False,
                 keep_original_model_structure: bool = False, ignore_encodings: bool = False, custom_op_info: Optional[CustomOpInfo] = None):

        self.ir_graph = ir_graph
        self.ir_graph_input_ops = [op
                                   for input_tensor in ir_graph.get_input_tensors_to_graph()
                                   for op in input_tensor.get_consumers()]
        self.ir_graph_output_op_names = [out_tensor.get_producer().name
                                         for out_tensor in ir_graph.get_output_tensors_of_graph()]
        self.ir_graph_output_tensors = [tensor.name() for tensor in self.ir_graph.get_output_tensors_of_graph()]
        self.ir_graph_ops_by_name = [op.name for op in ir_graph.get_ops()]
        self.ir_graph_constant_names = []
        self.state_dict = {}
        self.op_to_tensors = {}
        self.f = file   # TODO: Remove init argument `file` and property `f`
        custom_op_info = CustomOpInfo() if custom_op_info is None else custom_op_info
        self.custom_op_info = custom_op_info
        self.model_def_mgr = (
            StructuredModelDefinitionManager(custom_op_info.custom_module_paths)
            if is_block_extraction_flow
            else ModelDefinitionManager(custom_op_info.custom_module_paths)
        )
        self.keep_linear_without_bias = keep_linear_without_bias
        self.ignore_encodings = ignore_encodings
        self.parameter_constant_names = set()
        self.created_submodules = set()
        self.created_submodues_init = []
        self.module_list_to_submodule_count = collections.Counter()
        # Contains the axis information of each buffer.
        # Key indicates the buffer name and value is the Instance of AxisInformation
        self.buffer_axis_info = {}
        # Dict to store the details in case of additional padding layer is required in case of uneven padding.
        self.additional_pad_info = {}
        # Dict to store the additional padding module required for model's input tensor
        self.additional_transpose_info = {}
        self.prepared_param_name_map = {}
        self.tensor_axis_info = {}
        self.op_axis_info = {}
        self.non_trainable_module_names = set()
        self.ir_graph_output_names = []
        self.keep_original_model_structure = keep_original_model_structure
        # Dict to store the ir_graph names to prepared module names
        self.node_to_io_map = {'param_encodings': {}, 'activation_encodings': {}}

        # Dict to store the encoding extracted from the ir_graph against the torch equivalent name
        self.encodings = {'param_encodings': {}, 'activation_encodings': {}}

        logger.info("Input to Model Preparer Pro is either a DLC or IR Graph.")
        self.factory = IRDefaultFactory(ir_graph)

        # Used for both 'Pool' and 'Pool3d' types
        pool_attr_handler = self.factory.create_pool_attr_handler()

        self.op_type_to_attr_handler = {
            'InstanceNorm': self.factory.create_instance_norm_attr_handler(),
            'ElementWiseSoftplus': self.factory.create_soft_plus_attr_handler(),
            'ElementWiseNeuron': self.factory.create_soft_plus_attr_handler(),
            'Pool': pool_attr_handler,
            'Pool3d': pool_attr_handler
        }

        # This piece of code should be at the end since 'self' is being passed to the Op handler
        # That way, all the fields of this class are initialized and can potentially be used in the Op handler
        for op in ir_graph.get_ops():
            if is_custom_ir_op(op):
                self.parameter_constant_names.update(
                    op_handler.CustomOpHandler(op, self).get_parameter_op_names())
            else:
                self.parameter_constant_names.update(
                    op_handler.ir_to_handler_dict[op.type](op, self).get_parameter_op_names())
            self.ir_graph_constant_names.extend(
                tensor.name() for tensor in op.inputs() if tensor.is_static_tensor() and \
                not tensor.name in self.parameter_constant_names
            )

        # Count the number of child modules for each submodule in the model
        self.child_module_counter = op_handler.ChildModuleCounter(self.ir_graph_ops_by_name \
                                                                  + self.ir_graph_constant_names)

    def is_parameter(self, tensor: IrStaticTensor) -> bool:
        """ Check if the given static tensor is a parameter """
        return self.factory.is_parameter(tensor)

    def is_trainable_parameter(self, tensor: IrStaticTensor) -> bool:
        """ Check if the given static tensor is a trainable parameter """
        return self.factory.check_if_tensor_is_trainable_param(tensor)

    def is_from_linear_without_bias(self, op: IrOp) -> bool:
        """ Check if the given op was created from a Linear layer with bias set to False """
        return self.factory.is_from_linear_without_bias(op)

    @staticmethod
    def _get_op_type(op: IrOp) -> str:
        """ Get the IR graph op type for the given IrOp """
        if op.type == 'Eltwise_Unary':
            op_type = op.attrs_dict['eltwise_type']
        else:
            op_type = op.type

        return op_type

    def get_attr_value(self, op: IrOp, attr: str) -> Any:
        """
        Get the value of attribute `attr` of the given op

        :param op: IrOp to get the value from
        :param attr: Attribute whose value to get from op
        :return: Value of attribute `attr` of IrOp `op`
        """
        op_type = self._get_op_type(op)
        attr_handler = self.op_type_to_attr_handler[op_type]
        return attr_handler.get_attr_value(op, attr)
