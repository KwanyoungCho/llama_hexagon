# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define layout inferers for layout-untrackable ops."""

import numpy as np

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.passes.layout_transform import util
from qti.aisw.converters.common.passes.layout_transform.layout_inferer import (
    layout_inferer_base,
)
from qti.aisw.converters.common.passes.layout_transform.layout_recorder import (
    LayoutRecorder,
)


class UntrackableOpLayoutInfererBase(layout_inferer_base.LayoutInfererBase):
    """Inferer base for layout-untrackable op."""

    def infer_target_input_perm_seqs(
        self, input_buffer_names: list, input_shapes: list, layout_recorder: LayoutRecorder
    ):
        """Infer target input permute sequences.

        For layout-untrackable op, inputs must be in source formats, and therefore target input
        permute sequence are set to (0,1,...,rank-1). Inherited classes may override this method
        for special cases.
        """
        return [util.get_src_perm_seq(len(input_shape)) for input_shape in input_shapes]

    def infer_target_output_perm_seqs(self, target_perm_seqs: list, output_shapes: list):
        """Infer target output permute sequences.

        Similarly, target output permute sequences must be in source formats. Inherited classes may
        override this method for special cases.
        """
        return [
            util.get_src_perm_seq(len(output_shapes)) for output_shapes in output_shapes
        ]

    def update_attr_with_target_input_perm_seqs(self, target_perm_seqs: list, attrs):
        """Update attribute.

        Inherited classes may override this function if any attribute must be updated.
        """
        return {}


# ------------------------------------------------------------------------------
#   AxisAlignedBboxTransform
# ------------------------------------------------------------------------------
class AxisAlignedBboxTransformInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for AxisAlignedBboxTransform op."""

    op_type = ir_graph.QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM


# ------------------------------------------------------------------------------
#   BoxWithNmsLimit
# ------------------------------------------------------------------------------
class BoxWithNmsLimitInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for BoxWithNmsLimit op."""

    op_type = ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT


# ------------------------------------------------------------------------------
#   CollectRpnProposals
# ------------------------------------------------------------------------------
class CollectRpnProposalsInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for CollectRpnProposals op."""

    op_type = ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS


# ------------------------------------------------------------------------------
#   Constant
# -----------------------------------------------------------------------------
class ConstantInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for Constant op."""

    op_type = "constant"

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names,
        input_shapes,
        output_shapes,
        src_attrs,
        layout_recorder,
    ):
        """Infer input/output permute sequences and update attributes.

        Note that this method is overridden as Constant op does not have input nor layout-related
        attributes, and therefore no need to invoke related methods.
        """
        target_input_perm_seqs = []
        new_attr = {}
        out_perm_seqs = self.infer_target_output_perm_seqs(
            target_input_perm_seqs, output_shapes
        )
        return target_input_perm_seqs, out_perm_seqs, new_attr


# ------------------------------------------------------------------------------
#   Custom
# ------------------------------------------------------------------------------
class CustomInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for Custom op."""

    op_type = "custom"

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder, src_attrs
    ):
        """Infer target input permute sequences.

        If layout is provided in the CustomOp opdef for input, the specified layout is considered
        as desired layout, and target perm seq implies permuting from source layout to desired one.
        If no layout is provided, the input buffer will remain in source format.
        """
        target_input_perm_seqs = []
        layouts_dict = util.get_custom_op_layouts(src_attrs['axis_orders'])

        for input_name, input_shape in zip(input_buffer_names, input_shapes):
            if input_name in layouts_dict:
                perm_seq = util.calculate_perm_seq(
                    layouts_dict[input_name]['Source'], layouts_dict[input_name]['Desired']
                )
            else:
                perm_seq = util.get_src_perm_seq(len(input_shape))

            target_input_perm_seqs.append(perm_seq)

        return target_input_perm_seqs

    def infer_target_output_perm_seqs(self, target_input_perm_seqs, output_shapes, src_attrs):
        """Infer target output permute sequences.

        If layout is provided in the CustomOp opdef for output, the specified layout is considered
        as desired layout, and target perm seq implies permuting from source layout to desired one.
        If no layout is provided, the input buffer will remain in source format.
        """
        target_output_perm_seqs = []
        layouts_dict = util.get_custom_op_layouts(src_attrs['axis_orders'])

        for output_name, output_shape in zip(src_attrs['outputs'], output_shapes):
            if output_name in layouts_dict:
                perm_seq = util.calculate_perm_seq(
                    layouts_dict[output_name]['Source'], layouts_dict[output_name]['Desired']
                )
            else:
                perm_seq = util.get_src_perm_seq(len(output_shape))

            target_output_perm_seqs.append(perm_seq)

        return target_output_perm_seqs

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names,
        input_shapes,
        output_shapes,
        src_attrs,
        layout_recorder
    ):
        """Infer input/output permute sequences and update attributes."""
        target_input_perm_seqs = self.infer_target_input_perm_seqs(
            input_buffer_names, input_shapes, layout_recorder, src_attrs
        )
        target_output_perm_seqs = self.infer_target_output_perm_seqs(
            target_input_perm_seqs, output_shapes, src_attrs
        )
        return target_input_perm_seqs, target_output_perm_seqs, {}


# ------------------------------------------------------------------------------
#   DistributeFpnProposals
# ------------------------------------------------------------------------------
class DistributeFpnProposalsInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for DistributeFpnProposals op."""

    op_type = ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS


# ------------------------------------------------------------------------------
#   Expand
# ------------------------------------------------------------------------------
class ExpandInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for Expand op."""

    op_type = ir_graph.IR_OP_EXPAND


# ------------------------------------------------------------------------------
#   FullyConnected
# ------------------------------------------------------------------------------
class FullyConnectedInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for FullyConnected op."""

    op_type = ir_graph.QNN_OP_FULLY_CONNECTED

    def infer_target_input_perm_seqs(self, input_buffer_names, input_shapes, layout_recorder):
        """Infer target input perm seqs."""
        target_perm_seq = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )[0]

        # TODO
        # Current spatial-last axis order requires FullConnected weight in [OC, IC] format. Revisit
        # once the format is decoupled with axis order.
        if len(input_shapes) == 2:
            return [target_perm_seq, (1, 0)]
        return [target_perm_seq, (1, 0), (0,)]

    def update_attr_with_target_input_perm_seqs(self, target_perm_seqs, attrs):
        """Update attribute."""
        # TODO
        # Remove this once IrOp cleans-up legacy attributes.
        return {'transpose_b': False}


# ------------------------------------------------------------------------------
#   GatherElements
# ------------------------------------------------------------------------------
class GatherElementsInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for GatherElements op."""

    op_type = ir_graph.QNN_OP_GATHER_ELEMENTS


# ------------------------------------------------------------------------------
#   GatherND
# ------------------------------------------------------------------------------
class GatherNDInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for GatherND op."""

    op_type = ir_graph.QNN_OP_GATHER_ND


# ------------------------------------------------------------------------------
#   Input
# ------------------------------------------------------------------------------
class InputInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for Input op."""

    op_type = "input"

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names,
        input_shapes,
        output_shapes,
        src_attrs,
        layout_recorder,
    ):
        """Infer input/output permute sequences and update attributes.

        By default, Input op is treated like layout untrackable unless custom layout is specified.
        Note that new shape is calculated and returned as new attributes.
        """
        # input_buffer_names is name of graph input
        # Use this name to extract user-specified layouts
        target_output_perm_seq = layout_recorder.get_custom_perm_seq(input_buffer_names[0])
        if not target_output_perm_seq:
            target_output_perm_seq = util.get_src_perm_seq(len(output_shapes[0]))

        new_shape = [output_shapes[0][axis] for axis in target_output_perm_seq]

        return [], [target_output_perm_seq], {'shape': new_shape}


# ------------------------------------------------------------------------------
#   MaskedSoftmax
# ------------------------------------------------------------------------------
class MaskedSoftmaxInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for MaskedSoftmax op."""

    op_type = ir_graph.QNN_OP_MASKED_SOFTMAX


# ------------------------------------------------------------------------------
#   MatMul
# ------------------------------------------------------------------------------
class MatMulInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for MatMul op."""

    op_type = ir_graph.QNN_OP_MAT_MUL


# ------------------------------------------------------------------------------
#   MultiClassNms
# ------------------------------------------------------------------------------
class MultiClassNmsInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for MultiClassNmsOp."""

    op_type = ir_graph.QNN_OP_MULTI_CLASS_NMS


# ------------------------------------------------------------------------------
#   NonMaxSuppression
# ------------------------------------------------------------------------------
class NonMaxSuppressionInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for NonMaxSuppression op."""

    op_type = ir_graph.QNN_OP_NON_MAX_SUPPRESSION


# ------------------------------------------------------------------------------
#   NonZero
# ------------------------------------------------------------------------------
class NonZeroInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for NonZero op."""

    op_type = ir_graph.QNN_OP_NON_ZERO


# ------------------------------------------------------------------------------
#   OneHot
# ------------------------------------------------------------------------------
class OneHotInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for OneHot op."""

    op_type = ir_graph.QNN_OP_ONE_HOT


# ------------------------------------------------------------------------------
#   Pack
# ------------------------------------------------------------------------------
class PackInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for Pack op."""

    op_type = ir_graph.QNN_OP_PACK

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, output_shapes, layout_recorder, attrs
    ):
        """Infer target input permute sequences."""

        def is_spatial_pack_only(input_shape, output_shape, axis):
            """Check whether given shapes are only packed on spatial."""
            # we expect that input_shape & output_shape are extracted from src graph
            return (
                3 <= len(input_shape) <= 5
                and 3 <= len(output_shape) <= 5
                and input_shape[0] == output_shape[0]
                and input_shape[1] == output_shape[1]
                and axis not in [0, 1]
            )

        # we expect all inputs have same rank
        ranks = util.get_ranks(input_shapes)

        # rank must be given to filter out imprimitive rank
        input_perm_seqs_table = layout_recorder.get_perm_seqs_table(
            input_buffer_names, ranks
        )

        # get preferred_perm_seqs to check current usage
        preferred_perm_seqs = layout_recorder.get_preferred_perm_seqs(ranks[0])

        # check whether all inputs only change spatial dimension
        # check current usage is SpatialLast to SpatialFirst
        # Pack's input has same shape, so we only need to check one input
        if is_spatial_pack_only(input_shapes[0], output_shapes[0], attrs["axis"]):
            # vote for the target permute sequence
            # if tie vote happens, using preferred permute sequence to make decision
            # preferred_perm_seqs don't include time-series perm_seq
            # because this special case only works for spatial layout
            preferred_perm_seqs = [
                util.get_cl_perm_seq(ranks[0]),
                util.get_src_perm_seq(ranks[0]),
            ]
            target_perm_seq = util.get_target_perm_seq_by_vote(
                input_perm_seqs_table, preferred_perm_seqs
            )
        # TODO: Check whether it's needed to add another "elif" case to handle
        # SpatialFirst to SpatialLast usage
        else:
            # for other cases like rank <= 2 or rank >= 6 or others time-series cases
            # return src_perm_seq
            target_perm_seq = util.get_src_perm_seq(ranks[0])

        # for other cases like rank <= 2 or rank >= 6 or others
        # return src_perm_seq
        return [target_perm_seq] * len(input_buffer_names)

    def infer_target_output_perm_seqs(self, target_input_perm_seqs, output_shapes):
        """Infer target output permute sequences."""
        input_rank = len(target_input_perm_seqs[0])
        output_rank = len(output_shapes[0])

        if target_input_perm_seqs[0] == util.get_src_perm_seq(input_rank):
            return [util.get_src_perm_seq(output_rank)]
        elif target_input_perm_seqs[0] == util.get_cl_perm_seq(input_rank):
            return [util.get_cl_perm_seq(output_rank)]
        else:
            raise ValueError(f"Unkown input_perm_seq {target_input_perm_seqs[0]} for PackOp.")

    def update_attr_with_target_input_perm_seqs(self, target_output_perm_seqs, attrs):
        """Update attribute."""
        # update attribute based on first target output permute sequence
        return {"axis": target_output_perm_seqs[0].index(attrs["axis"])}

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names,
        input_shapes,
        output_shapes,
        src_attrs,
        layout_recorder,
    ):
        """Infer input/output permute sequences and update attributes.

        Note that this method is overridden as Pack op may remain in channel-last layout if the
        shape changes only happen within spatial dimension.
        """
        target_input_perm_seqs = self.infer_target_input_perm_seqs(
            input_buffer_names, input_shapes, output_shapes, layout_recorder, src_attrs
        )
        target_output_perm_seqs = self.infer_target_output_perm_seqs(
            target_input_perm_seqs, output_shapes
        )
        new_attrs = self.update_attr_with_target_input_perm_seqs(
            target_output_perm_seqs, src_attrs
        )

        return target_input_perm_seqs, target_output_perm_seqs, new_attrs


# ------------------------------------------------------------------------------
#   Reshape
# ------------------------------------------------------------------------------
class ReshapeInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for Reshape op."""

    op_type = ir_graph.QNN_OP_RESHAPE

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, output_shapes, layout_recorder
    ):
        """Infer target input permute sequences."""

        def is_spatial_reshape_only(input_shape, output_shape):
            """Check whether given shapes are only reshaped on spatial."""
            # we expect that input_shape & output_shape are extracted from src graph
            return (
                3 <= len(input_shape) <= 5
                and 3 <= len(output_shape) <= 5
                and input_shape[0] == output_shape[0]
                and input_shape[1] == output_shape[1]
            )

        # handle special case for Reshape
        input_rank = len(input_shapes[0])
        existing_perm_seqs = layout_recorder.get_perm_seqs(
            input_buffer_names[0], input_rank
        )

        # if there is (0,2,3,1), which implicitly means that new layout can be viewed as NHWC,
        # src layout can be viewed as NCHW
        if (
            is_spatial_reshape_only(input_shapes[0], output_shapes[0])
            and util.get_cl_perm_seq(input_rank) in existing_perm_seqs
        ):
            # reshape only happens in spatial dimension
            return [util.get_cl_perm_seq(input_rank)]

        # for other cases like rank <= 2 or rank >= 6 or others
        # return src_perm_seq
        return [util.get_src_perm_seq(input_rank)]

    def infer_target_output_perm_seqs(self, target_input_perm_seqs, output_shapes):
        """Infer target output permute sequences."""
        input_rank = len(target_input_perm_seqs[0])
        output_rank = len(output_shapes[0])

        if target_input_perm_seqs[0] == util.get_src_perm_seq(input_rank):
            return [util.get_src_perm_seq(output_rank)]
        elif target_input_perm_seqs[0] == util.get_cl_perm_seq(input_rank):
            return [util.get_cl_perm_seq(output_rank)]
        else:
            raise ValueError(
                f"Unknow input_perm_seq {target_input_perm_seqs[0]} for ResahepOp."
            )

    def update_attr_with_target_input_perm_seqs(
        self, target_output_perm_seqs, output_shapes
    ):
        """Update attribute based on target input permute sequences."""
        output_rank = len(target_output_perm_seqs[0])
        # if target_ouput_prtm_seq is in src-format, no need to update attributes
        if target_output_perm_seqs[0] == util.get_src_perm_seq(output_rank):
            return {}
        # if target_ouput_prtm_seq is in channel_last format, update "shape" attribute
        elif target_output_perm_seqs[0] == util.get_cl_perm_seq(output_rank):
            # update "shape" attribute to channle_last format
            new_shape = [output_shapes[0][dim] for dim in target_output_perm_seqs[0]]
            return {"shape": new_shape}
        else:
            raise ValueError(
                f"Unknow input_perm_seq {target_output_perm_seqs[0]} for ResahepOp."
            )

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names,
        input_shapes,
        output_shapes,
        src_attrs,
        layout_recorder,
    ):
        """Infer input/output permute sequences and update attributes.

        Note that this method is overridden as Reshape op may remain in channel-last layout if the
        shape changes only happen within spatial dimension.
        """
        # This function would call member function to infer input/output permute sequneces
        # and updated attributes
        target_input_perm_seqs = self.infer_target_input_perm_seqs(
            input_buffer_names, input_shapes, output_shapes, layout_recorder
        )
        target_output_perm_seqs = self.infer_target_output_perm_seqs(
            target_input_perm_seqs, output_shapes
        )
        # In reshape case, we use output_shapes instead of extracting "shape" from src_attrs
        # the "shape" attribute might be [1,256,-1], which is not convenient to infer new shape
        # attribute
        new_attrs = self.update_attr_with_target_input_perm_seqs(
            target_output_perm_seqs, output_shapes
        )

        return target_input_perm_seqs, target_output_perm_seqs, new_attrs


# ------------------------------------------------------------------------------
#   ScatterElements
# ------------------------------------------------------------------------------
class ScatterElementsInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for ScatterElements op."""

    op_type = ir_graph.QNN_OP_SCATTER_ELEMENTS


# ------------------------------------------------------------------------------
#   ScatterND
# ------------------------------------------------------------------------------
class ScatterNDInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for ScatterND op."""

    op_type = ir_graph.QNN_OP_SCATTER_ND


# ------------------------------------------------------------------------------
#   TopK
# ------------------------------------------------------------------------------
class TopKInferer(UntrackableOpLayoutInfererBase):
    """Layout inferer for TopK op."""

    op_type = ir_graph.QNN_OP_TOP_K
