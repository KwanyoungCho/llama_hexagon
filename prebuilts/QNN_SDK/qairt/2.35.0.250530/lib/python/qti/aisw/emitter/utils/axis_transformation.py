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

""" Axis Information Uitlity used for axis tracking and changing """

import io
from typing import Union

from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common import ir_graph as ir_graph_lib

AxisFormat = ir_graph_lib.AxisFormat


CHANNEL_LAST_AXIS_FORMAT = {
    'ACTIVATION': [AxisFormat.NHWC, AxisFormat.NDHWC, AxisFormat.NFC],
    'WEIGHTS': [AxisFormat.HWIO, AxisFormat.DHWIO]
}

CHANNEL_FIRST_AXIS_FORMAT = {
    'ACTIVATION': [AxisFormat.NCHW, AxisFormat.NCDHW, AxisFormat.NCF, AxisFormat.NF],
    'WEIGHTS': [AxisFormat.OIHW, AxisFormat.OIDHW]
}

CHANNEL_LAST_TO_FIRST = {
    AxisFormat.NHWC: AxisFormat.NCHW,
    AxisFormat.NDHWC: AxisFormat.NCDHW,
    AxisFormat.NFC: AxisFormat.NCF,
    AxisFormat.HWIO: AxisFormat.OIHW,
    AxisFormat.DHWIO: AxisFormat.OIDHW,
    AxisFormat.NF: AxisFormat.NF,
    AxisFormat.NONTRIVIAL: AxisFormat.NONTRIVIAL
}

CHANNEL_FIRST_TO_LAST = {
    AxisFormat.NCHW: AxisFormat.NHWC,
    AxisFormat.NCDHW: AxisFormat.NDHWC,
    AxisFormat.NCF: AxisFormat.NFC,
    AxisFormat.OIHW: AxisFormat.HWIO,
    AxisFormat.OIDHW: AxisFormat.DHWIO,
    AxisFormat.NF: AxisFormat.NF
}

TRANSPOSE_ORDER = {
    AxisFormat.NHWC: {
        AxisFormat.NCHW: AxisTracker.AxisFormat.NSC_TO_NCS
    },
    AxisFormat.NFC: {
        AxisFormat.NCF: AxisTracker.AxisFormat.NFC_TO_NCF
    },
    AxisFormat.NCHW: {
        AxisFormat.NHWC: AxisTracker.AxisFormat.NCS_TO_NSC,
    },
    AxisFormat.NCF: {
        AxisFormat.NFC: AxisTracker.AxisFormat.NCF_TO_NFC
    },
    AxisFormat.NDHWC: {
        AxisFormat.NCDHW: AxisTracker.AxisFormat.NDHWC_TO_NCDHW
    },
    AxisFormat.NCDHW: {
        AxisFormat.NDHWC: AxisTracker.AxisFormat.NCDHW_TO_NDHWC
    },
    AxisFormat.HWIO: {
        AxisFormat.OIHW: AxisTracker.AxisFormat.HWIO_TO_OIHW
    },
    AxisFormat.OIHW: {
        AxisFormat.HWIO: AxisTracker.AxisFormat.OIHW_TO_HWIO
    },
    AxisFormat.DHWIO: {
        AxisFormat.OIDHW: AxisTracker.AxisFormat.DHWIO_TO_OIDHW,
    },
    AxisFormat.OIDHW: {
        AxisFormat.DHWIO: AxisTracker.AxisFormat.OIDHW_TO_DHWIO
    }
}


class AxisInformation:
    """ To store the axis format information of a tensor. """

    def __init__(self, ir_axis_format: AxisFormat, torch_axis_format: AxisFormat,
                 output_permute_order: Union[tuple, list] = None, input_permute_order: Union[tuple, list] = None,
                 node_type: str = None, consumer_count: int = None):
        """
        :param ir_axis_format: Axisformat of IRGraph
        :param torch_axis_format: Axisformat of the PyTorch Graph
        :param output_permute_order: Permute order to bt applied on the input tensor
        :param input_permute_order: Permute order to be applied on the output tensor
        :param node_type: Op type of the producer of the tensor
        :param consumer_count: Number of consumers of this tensor
        :return:
        """
        self.ir_axis_format = ir_axis_format
        self.torch_axis_format = torch_axis_format
        self.output_permute_order = output_permute_order
        self.input_permute_order = input_permute_order
        self.type = node_type
        self.consumer_count = consumer_count

    def __str__(self):
        stream = io.StringIO(newline='\n')
        stream.write(f"Ir Axis Format :{self.ir_axis_format}, Torch Axis Format: {self.torch_axis_format}, "
                     f"Permute Order: [Input: {self.input_permute_order}, Output: {self.output_permute_order}]")
        return stream.getvalue()


def update_shape(axis_format, shape):
    """ Updates the shape as per the axis format """

    if axis_format == AxisFormat.NDHWC:
        shape = AxisTracker.permute_shape(shape, AxisTracker.AxisFormat.NDHWC_TO_NCDHW)
    elif axis_format == AxisFormat.NHWC:
        shape = AxisTracker.permute_shape(shape, AxisTracker.AxisFormat.NSC_TO_NCS)
    elif axis_format == AxisFormat.NFC:
        shape = AxisTracker.permute_shape(shape, AxisTracker.AxisFormat.NFC_TO_NCF)
    elif axis_format == AxisFormat.NTF:
        shape = AxisTracker.permute_shape(shape, AxisTracker.AxisFormat.NTF_TO_TNF)
    return shape


def update_shape_using_transform_order(transform_order, shape):
    """ Updates the shape as per the transform_order """

    if shape is not None and transform_order is not None:
        return AxisTracker.permute_shape(shape, transform_order)
    return shape
