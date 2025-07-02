# /usr/bin/env python
# -*- mode: python -*-
# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
# pylint: disable=import-error, no-name-in-module

""" Axis Tracking related utilities """
import io
from typing import Union, Tuple
import numpy as np

# pylint: disable=import-error
from qti.aisw.converters.common import ir_graph as ir_graph_lib

IrGraph, IrOp, IrTensor, IrStaticTensor, AxisFormat = (ir_graph_lib.IrGraph, ir_graph_lib.IrOp, ir_graph_lib.IrTensor,
                                                       ir_graph_lib.IrStaticTensor, ir_graph_lib.AxisFormat)

# Ops with specific kernel axis requirements
STRICT_OPS = ['Conv1d', 'Conv2d', 'Conv3d', 'TransposeConv1d', 'TransposeConv2d', 'TransposeConv3d',
              'DepthWiseConv2d', 'Pool', 'Pool3d', 'Batchnorm', 'InstanceNorm',
              'SpaceToBatch', 'BatchToSpace', 'CropAndResize', 'DepthToSpace',
              'SpaceToDepth', 'RoiAlign', 'Resize', 'ResizeNearestNeighbor',
              'ResizeBilinear', 'ChannelShuffle', 'CropAndResize', 'Lrn', 'GroupNorm',
              'Prelu', 'Pad', 'SpConv3d', 'GridSample', 'CUSTOM_OP']

# Ops to keep in source axis format always
# Entry 'SRC_AXIS_OP' is a marker entry. No op exists with this type.
# It is used in cases where we want to enforce a INPUT_FORMAT_ALIGNED_OPS to behave as SRC_AXIS_OPS
SRC_AXIS_OPS = ['Reshape', 'Transpose', 'FullyConnected', 'Gather', 'GatherNd', 'MatMul', 'LayerNorm', 'ScatterNd',
                'ScatterElements', 'OneHot', 'SRC_AXIS_OP', 'RmsNorm']

# Ops which work on particular axis, so axis field needs to be transformed
INDEX_TRANSPOSE_OPS = ['Reduce', 'Arg', 'Moments', 'Split', 'CumulativeSum', 'StridedSlice',
                       'L2Norm', 'RoiPooling', 'Tile']

# Ops which, if all the axis are same then work as INDEX TRANSPOSE OPS else WORKS AS SRC_AXIS_OPS
INPUT_FORMAT_ALIGNED_OPS = ['Eltwise_Binary', 'Eltwise_Ternary', 'Concat']

WEIGHT_INDEX = {
    'Conv1d': 1,
    'Conv2d': 1,
    'Conv3d': 1,
    'TransposeConv1d': 1,
    'TransposeConv2d': 1,
    'TransposeConv3d': 1,
    'DepthWiseConv2d': 1,
    'FullyConnected': 1,
    'InstanceNorm': 1,
    'Batchnorm': 1,
}


class TensorAxisInfo:
    """ Stores the axis information of the tensor """

    def __init__(self, original_shape: tuple, transform_order: tuple = None, ):
        self.original_shape = original_shape
        self.transform_order = format_transpose(transform_order)
        # To-Do modifying structure to add more attributes to have info useful for optimization.

    @property
    def new_shape(self):
        """ Shape of the tensor in pytorch graph """
        if self.transform_order is None:
            return self.original_shape
        return tuple(np.take(self.original_shape, self.transform_order))

    def __str__(self):
        stream = io.StringIO(newline='\n')
        stream.write(f"Original Shape: {self.original_shape},  | New Shape:  {self.new_shape}"
                     f" | Transform Order: {self.transform_order}")
        return stream.getvalue()


class OpAxisInfo:
    """ Stores the axis transformation needed for """

    def __init__(self, input_transform=None, output_transform=None):
        if output_transform is None:
            output_transform = {}
        if input_transform is None:
            input_transform = {}
        self.input_transform = input_transform
        self.output_transform = output_transform

    def __str__(self):
        stream = io.StringIO(newline='\n')
        stream.write(f"Input Transform:  {self.input_transform} | Output Transform: {self.output_transform}")
        return stream.getvalue()


CHANNEL_LAST_TO_FIRST = {
    "ACTIVATION": {
        1: None,
        2: None,
        3: (0, 2, 1),  # NFC TO NCF
        4: (0, 3, 1, 2),  # NHWC TO NCHW
        5: (0, 4, 1, 2, 3)  # NDHWC_TO_NCDHW
    },

    "WEIGHT": {
        1: None,
        2: (0, 1),
        3: (2, 1, 0),  # WIO to OIW
        4: (3, 2, 0, 1),  # HWIO TO OIHW
        5: (4, 3, 0, 1, 2)  # DHWIO_TO_OIDHW
    }
}


def get_required_weight_transpose(tensor_axis_info: TensorAxisInfo):
    """
    Given the tensor_axis_info for the weight tensor returns the transpose required.

    :param tensor_axis_info: TensorAxisInfo of the weight tensor
    :return: Transpose order in case of transpose is required, otherwise None
    """

    required_transpose = None
    expected_order = CHANNEL_LAST_TO_FIRST['WEIGHT'][len(tensor_axis_info.original_shape)]
    if tensor_axis_info.transform_order != expected_order:
        required_transpose = get_transpose_order(tensor_axis_info.transform_order, expected_order)

    return format_transpose(required_transpose)


def get_required_transpose(tensor_axis_info: TensorAxisInfo, op_type: str) -> Union[Tuple, None]:
    """
    Given the op_type the current state of the input tensor returs if any transpose is required
    :param tensor_axis_info: TensorAxisInfo of the input tensor
    :param op_type: Type of oconsumer op
    :return: Transpose order in case of transpose is required, otherwise None
    """
    required_transpose = None

    if op_type in STRICT_OPS:
        expecteded_order = CHANNEL_LAST_TO_FIRST['ACTIVATION'][len(tensor_axis_info.original_shape)]
        if tensor_axis_info.transform_order != expecteded_order:
            required_transpose = get_transpose_order(tensor_axis_info.transform_order, expecteded_order)

    elif op_type in SRC_AXIS_OPS:
        # pylint: disable=unnecessary-comprehension
        # pylint: disable=consider-using-generator
        required_transpose = get_transpose_order(tensor_axis_info.transform_order,
                                                 tuple([i for i in range(len(tensor_axis_info.original_shape))]))

    return format_transpose(required_transpose)


def format_transpose(transpose_order: Tuple):
    """
    In case the transpose order is equal to identity return None

    :param transpose_order: Given transpose order to format
    :return: transpose order to be set to the info.
    """
    if transpose_order is None:
        return None

    # pylint: disable=unnecessary-comprehension
    # pylint: disable=consider-using-generator
    identity_tranpose_order = tuple([i for i in range(len(transpose_order))])
    return None if identity_tranpose_order == transpose_order else transpose_order


def get_transpose_order(current_order: Tuple, final_order: Tuple):
    """
    Utility to get the transpose order required to get to the require transpose state.

    :param current_order: Current transpose order of the tensor.
    :param final_order: Expected final transpose order.
    :return: Required transpose order.
    """
    # If current order is not None
    required_transpose = None
    if current_order is None:
        required_transpose = final_order
    else:
        # First Calculate the transpose require to nullify the current transpose order.
        # pylint: disable=unnecessary-comprehension
        normal_order = [i for i in range(len(current_order))]
        for index, axis in enumerate(current_order):
            normal_order[axis] = index

        # Combine the obtained order with the final order
        required_transpose = combine_transpose_order(normal_order, final_order)

    return required_transpose


def combine_transpose_order(transpose_order1: Tuple, transpose_order2: Tuple):
    """
    Combines the given transpose order and provides the effective transpose order.

    :param transpose_order1: first transpose order.
    :param transpose_order2: second transpose order
    :return: effective transpose order.
    """
    if transpose_order1 is None:
        return transpose_order2
    if transpose_order2 is None:
        return transpose_order1
    return tuple(np.array(transpose_order1).choose(transpose_order2).tolist())


def get_output_tranapose_order(op_type: str, output_tensor: IrTensor, input_axis_info: TensorAxisInfo):
    """
    Get the output tensor order for the given output_tensor.

    :param op_type: Type of the op
    :param output_tensor: Output tensor for which transpose order needs to be calculated.
    :param input_axis_info: Input TensorAxisInfo for the 1st input of the op.
    :return: Transpose Order to be set to the output_tensor.
    """
    if op_type in STRICT_OPS:
        transpose_order = CHANNEL_LAST_TO_FIRST['ACTIVATION'][len(output_tensor.dims())]
        return transpose_order

    if op_type in SRC_AXIS_OPS:
        return None

    # In case of op not behaving as STRICT OPS/SRC_AXIS_OPS we need to just pickup the transpose order of the input
    # and propagate it to the output
    return input_axis_info.transform_order
