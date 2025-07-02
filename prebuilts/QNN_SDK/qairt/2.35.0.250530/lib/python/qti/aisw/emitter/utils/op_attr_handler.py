# /usr/bin/env python
# -*- mode: python -*-
# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
""" Op Attribute handler base classes """
import abc
import logging
from typing import Dict, Tuple, Any, Callable, Union
import numpy as np
# pylint: disable=import-error
from qti.aisw.converters.common import ir_graph as ir_graph_lib

IrOp = ir_graph_lib.IrOp
logger = logging.getLogger('TorchEmitter')


class AttrHandler(abc.ABC):
    """ Base class for IR graph op attribute handler """

    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def attr_to_call_map(self) -> Dict[str, Callable[[IrOp], Any]]:
        """
        Map from attribute to getter
        Keys can be overriden the subclasses depending on the target framework
        Example - 'eps' in PyTorch, 'epsilon' in keras
        """

    def get_attr_value(self, op: IrOp, attr: str) -> Any:
        """
        Get the value of attribute `attr` of the given op

        :param op: IrOp to get the value from
        :param attr: Attribute whose value to get from op
        :return: Value of attribute `attr` of IrOp `op`
        """
        return self.attr_to_call_map[attr](op)


class InstanceNormAttrHandler(AttrHandler):
    """ Base class for IR graph InstanceNorm op attribute handler with default behaviour """

    @property
    def attr_to_call_map(self) -> Dict[str, Callable[[IrOp], Any]]:
        """ Map from attribute to getter of InstanceNorm op handler """
        return {
            'num_features': self.get_num_features,
            'eps': self.get_epsilon,
            'momentum': self.get_momentum,
            'affine': self.get_is_affine,
            'track_running_stats': self.should_track_running_stats,
            'weight': self.get_weights,
            'bias': self.get_bias,
            'running_mean': self.get_running_mean,
            'running_var': self.get_running_variance
        }

    @staticmethod
    def get_num_features(op: IrOp) -> int:
        """ Get number of features/channels of input data to the InstanceNorm op """
        # input[1] is gamma, shape of which is the number of input features
        return op.get_input_shapes()[1][0]

    @staticmethod
    def get_epsilon(op: IrOp) -> float:
        """ Get epsilon of the InstanceNorm op """
        return op.attrs_dict['epsilon']

    @staticmethod
    def get_momentum(op: IrOp) -> float: # pylint: disable=unused-argument
        """ Get momentum of the InstanceNorm op """
        return 0.1

    @staticmethod
    def get_is_affine(op: IrOp) -> bool: # pylint: disable=unused-argument
        """ Get `affine` property of the InstanceNorm op """
        return True

    @staticmethod
    def should_track_running_stats(op: IrOp) -> bool: # pylint: disable=unused-argument
        """ Get `track_running_stats` property of the InstanceNorm op """
        return False

    @staticmethod
    def get_weights(op: IrOp) -> np.ndarray:
        """ Get gamma of the InstanceNorm op """
        return op.inputs()[1].get_data()

    @staticmethod
    def get_bias(op: IrOp) -> np.ndarray:
        """ Get beta of the InstanceNorm op """
        return op.inputs()[2].get_data()

    @staticmethod
    def get_running_mean(op: IrOp) -> Union[None, np.ndarray]: # pylint: disable=unused-argument
        """ Get running mean of the InstanceNorm op """

    @staticmethod
    def get_running_variance(op: IrOp) -> Union[None, np.ndarray]: # pylint: disable=unused-argument
        """ Get running variance of the InstanceNorm op """


class SoftPlusAttrHandler(AttrHandler):
    """ Base class for IR graph SoftPlus op attribute handler with default behaviour """

    @property
    def attr_to_call_map(self) -> Dict[str, Callable[[IrOp], Any]]:
        """ Map from attribute to getter of SoftPlus op handler """
        return {
            'beta': self.get_beta,
            'threshold': self.get_threshold
        }

    @staticmethod
    def get_beta(op: IrOp) -> int: # pylint: disable=unused-argument
        """ Get beta of the SoftPlus op """
        return 1

    @staticmethod
    def get_threshold(op: IrOp) -> int: # pylint: disable=unused-argument
        """ Get threshold of the SoftPlus op """
        return 20


class PoolAttrHandler(AttrHandler):
    """ Base class for IR graph Pool op attribute handler with default behaviour """

    @property
    def attr_to_call_map(self) -> Dict[str, Callable[[IrOp], Any]]:
        """ Map from attribute to getter of Pool op handler """
        return {
            'filter_size': self.get_filter_size,
            'stride': self.get_stride,
            'padding': self.get_padding,
            'ceil_mode': self.get_ceil_mode,
            'count_include_pad': self.get_count_include_pad,
            'dilation': self.get_dilation
        }

    @staticmethod
    def get_filter_size(op: IrOp) -> Union[int, Tuple[int]]:
        """ Get filter size of the Pool op """
        filter_size = op.attrs_dict['filter_size']
        return filter_size if isinstance(filter_size, int) else tuple(filter_size)

    @staticmethod
    def get_stride(op: IrOp) -> Union[int, Tuple[int]]:
        """ Get stride of the Pool op """
        stride = op.attrs_dict['stride']
        return stride if isinstance(stride, int) else tuple(stride)

    @staticmethod
    def get_padding(op: IrOp) -> Tuple[int]:
        """
        Fetches the padding value from the ops attrs_dict and check if padding is uneven.
        In case of uneven padding, set the padding value to zero.

        :param op: IrOp Node form which padding value is to be extracted.
        :return: padding value to be used for the operation
        """
        padding_start, padding_end = [tuple(op.attrs_dict["pad_amount"][..., axis].flatten()) for axis in (0, 1)]
        # As padding is un-even it will be handled by an additional pad op.
        # As we are adding a separate pad operation, no need to the pad further.
        return tuple([0] * len(padding_start)) if padding_start != padding_end else padding_start

    @staticmethod
    def get_count_include_pad(op: IrOp) -> bool:
        """ Get `count_include_pad` attribute of the Pool op """
        return op.attrs_dict['count_pad_for_edges']

    @staticmethod
    def get_ceil_mode(op: IrOp) -> bool: # pylint: disable=unused-argument
        """ Get `ceil_mode` attribute of the Pool op """
        return False

    @staticmethod
    def get_dilation(op: IrOp) -> Union[int, Tuple[int]]: # pylint: disable=unused-argument
        """ Get dilation of the Pool op """
        return 1
