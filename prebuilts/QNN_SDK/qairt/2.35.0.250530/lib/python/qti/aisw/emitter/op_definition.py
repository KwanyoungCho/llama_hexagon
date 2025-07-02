# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Tuple, Union
import warnings
import numpy as np
import torch
import torchvision
import torch.nn


########### AIMET Pro Ops #######################

class CustomLayerNorm(torch.nn.Module):
    """ Custom module for generic LayerNorm """
    def __init__(self, input_shape: list, axes: list, eps: float):
        super().__init__()
        self.input_shape = input_shape
        self.axes = axes
        self.eps = eps
        self.normalized_shape = list(input_shape[a] for a in axes)
        self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor):
        """
        Forward pass routine for custom LayerNorm
        """
        # The first permutation reorders the tensor's axes, ensuring that the axes specified in the QNN operation
        # are positioned at the last dimensions for PyTorch LayerNorm to consume
        # The second permute operation is the inverse of the first permute operation so that
        # the output tensor has the same data format as the input tensor
        permute_1_dims = [i for i in range(len(self.input_shape)) if i not in self.axes] + self.axes
        permute_2_dims = list(np.argsort(permute_1_dims))

        x = x.permute(dims=permute_1_dims)
        x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(dims=permute_2_dims)

        return x


class IndexSelect(torch.nn.Module):
    """ Custom module for IndexSelect with multiple indexes """

    # pylint: disable=unused-argument
    @staticmethod
    def forward(input_tensor: torch.Tensor, dim: int, indices: Union[torch.IntTensor, torch.LongTensor]) -> torch.Tensor:
        """ Custom forward function for IndexSelect(gather) op to handle multi dimension index values"""
        data = input_tensor
        axis = dim

        dim_size = data.shape[axis]
        original_shape = indices.shape
        indices = indices.reshape(-1)

        temp = tuple(data.shape)
        new_shape = list(temp[:axis]) + list(original_shape) + list(temp[axis+1:])

        for idx, index in enumerate(indices):
            if index < 0:
                indices[idx] = dim_size + index
        original_dtype = data.dtype
        z = torch.index_select(data.to(torch.float32), axis, indices.to(torch.int64))

        return z.reshape(*new_shape).to(original_dtype)


class NonZero(torch.nn.Module):
    """Custom module for a NonZero op"""
    @staticmethod
    def forward(tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for NonZero op
        """
        actual_value = torch.nonzero(tensor)
        n_repeat = torch.numel(tensor) - actual_value.shape[0]
        return torch.cat((actual_value, actual_value[-1].repeat(n_repeat, 1)), 0)


class Stack(torch.nn.Module):
    """
    Custom module for stack
    """
    def __init__(self, axis: int = 0):
        super().__init__()
        self._axis = axis

    def forward(self, *inputs) -> torch.Tensor:
        """
        Forward function routine for stack
        """
        return torch.stack(inputs, dim=self._axis)


class UnBind(torch.nn.Module):
    """
    Custom module for unbind
    """
    def __init__(self, axis: int = 0):
        super().__init__()
        self._axis = axis

    def forward(self, x) -> Tuple[torch.Tensor]:
        """
        Forward function routine for unbind
        """
        return torch.unbind(x, dim=self._axis)


class SpaceToBatch(torch.nn.Module):
    """ Custom module for TF/Keras based SpaceToBatch for 4D input tensor (NCHW) """

    def __init__(self, block_shape: list, pad_amount: list):
        super().__init__()
        self.block_shape = block_shape
        self.pad_amount = pad_amount if pad_amount is not None else [[0, 0], [0, 0]]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for SpaceToBatch
        """
        b, c, h, w = list(inputs.shape)
        # Constraints for block_shape
        assert len(self.block_shape) == 2, 'Invalid block_shape, must be of shape [2] with format [block_height, block_width]'
        assert self.block_shape[0] >= 1 and self.block_shape[1] >= 1, 'Invalid block_shape, elements must be >=1'
        # Constraints for pad_amount
        assert len(self.pad_amount) == 2 and len(self.pad_amount[0]) == 2, ('Invalid paddings, must be of shape [2,2] '
                                                                            'with format [[pad_top, pad_bottom], [pad_left, pad_right]]')
        # Input constraints
        assert (h+sum(self.pad_amount[0])) % self.block_shape[0] == 0 and (w + sum(self.pad_amount[1])) % self.block_shape[0] == 0, \
            'Input Constraints not satisfied'
        # STEP - 1
        padded = torch.nn.functional.pad(inputs, self.pad_amount[1] + self.pad_amount[0], "constant", 0)
        padded_shape = padded.shape
        # STEP - 2
        reshaped_padded = torch.reshape(padded, [b, c, padded_shape[2]//self.block_shape[0], self.block_shape[0],
                                                 padded_shape[3]//self.block_shape[1], self.block_shape[1]])
        # STEP - 3
        permuted_reshaped_padded = torch.permute(reshaped_padded, (3, 5, 0, 1, 2, 4))
        # STEP - 4
        output_shape = [b*self.block_shape[0]*self.block_shape[1], c, padded_shape[2]//self.block_shape[0],
                        padded_shape[3]//self.block_shape[1]]
        output = torch.reshape(permuted_reshaped_padded, output_shape)

        # Output constraints
        assert inputs.dtype == output.dtype, 'Dtypes of Input and Output are not matching'
        return output


class BatchToSpace(torch.nn.Module):
    """ Custom module for TF/Keras based BatchToSpace for 4D input tensor (NCHW) """

    def __init__(self, block_shape: list, crops: list):
        super().__init__()
        self.block_shape = block_shape
        self.crops = crops if crops is not None else [[0, 0], [0, 0]]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for BatchToSpace
        """
        b, c, h, w = list(inputs.shape)
        # Constraints for block_shape
        assert len(self.block_shape) == 2, 'Invalid block_shape, must be of shape [2] with format [block_height, block_width]'
        assert self.block_shape[0] >= 1 and self.block_shape[1] >= 1, 'Invalid block_shape, elements must be >=1'
        # Constraints for crops
        assert len(self.crops) == 2 and len(self.crops[0]) == 2, ('Invalid crops, must be of shape [2,2] with '
                                                                  'format [[crop_top, crop_bottom], [crop_left, crop_right]]')
        # Input constraints
        assert b % (self.block_shape[0] * self.block_shape[1]) == 0, 'Input Constraints not satisfied'
        assert self.crops[0][0] + self.crops[0][1] <= self.block_shape[0] * h, 'Input Constraints not satisfied'
        assert self.crops[1][0] + self.crops[1][1] <= self.block_shape[1] * w, 'Input Constraints not satisfied'
        # STEP - 1:
        reshaped = torch.reshape(inputs, self.block_shape + [b//(self.block_shape[0]*self.block_shape[1]), c] + list(inputs.shape)[2:])
        # STEP - 2:
        permuted = torch.permute(reshaped, (2, 3, 4, 0, 5, 1))
        # STEP - 3:
        reshaped_permuted = torch.reshape(permuted, [b//(self.block_shape[0]*self.block_shape[1]), c,
                                                     h*self.block_shape[0], w*self.block_shape[1]])
        # STEP - 4:
        output_height = h*self.block_shape[0]-self.crops[0][0]-self.crops[0][1]
        output_width = w*self.block_shape[1]-self.crops[1][0]-self.crops[1][1]
        output = reshaped_permuted[:, :, -output_height:, -output_width:]
        return output


class Moments(torch.nn.Module):
    """
    Custom module for moments
    """
    def __init__(self, axes: int, keep_dims: bool):
        super().__init__()
        self._axes = axes
        self._keep_dims = keep_dims

    def forward(self, inputs) -> tuple:
        """
        Forward function routine for moments
        """
        # correction=0 : To calculate biased variance which is default in case of tf.nn.moments
        var, mean = torch.var_mean(inputs, dim=self._axes, keepdim=self._keep_dims, correction=0)
        return mean, var


class CropAndResize(torch.nn.Module):
    """
    Custom PyTorch Module for tf.image.crop_and_resize
    Reference TF implementation:
    https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/core/kernels/image/crop_and_resize_op_gpu.cu.cc
    """
    def __init__(self, resize_dims: list, interpolation_mode: int = 0, extrapolation_value: float = 0.0):
        super().__init__()
        self.resize_dims = resize_dims # resize_dims: [crop_height, crop_width]
        self.interpolation_mode = interpolation_mode
        self.extrapolation_value = extrapolation_value

    # pylint: disable=too-many-locals
    def forward(self, image: torch.Tensor, boxes: torch.Tensor, box_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass call for CropAndResize
        """
        _, _, img_height, img_width = image.shape
        result = None
        for box, img_ind in zip(boxes, box_indices):
            y1, x1, y2, x2 = box
            height_scale = float((y2 - y1) * (img_height - 1) / float(self.resize_dims[0] - 1) if self.resize_dims[0] > 1 else 0)
            width_scale = float((x2 - x1) * (img_width - 1) / float(self.resize_dims[1] - 1) if self.resize_dims[1] > 1 else 0)

            t_y = torch.arange(self.resize_dims[0]).to(boxes.device)
            in_y = y1 * (img_height - 1) + t_y * height_scale if self.resize_dims[0] > 1 else 0.5 * (y1 + y2) * (img_height - 1)
            t_x = torch.arange(self.resize_dims[1]).to(boxes.device)
            in_x = x1 * (img_width - 1) + t_x * width_scale if (self.resize_dims[1] > 1) else 0.5 * (x1 + x2) * (img_width - 1)

            in_x, in_y, mask, extrapolation = self._get_masks_for_extrapolation(in_x, in_y, img_width, img_height)

            if self.interpolation_mode == 0:
                image_crop = self._custom_bilinear_interpolation(image[img_ind], in_y, in_x)
            elif self.interpolation_mode == 1:
                image_crop = self._custom_nearest_neighbor_interpolation(image[img_ind], in_x, in_y)
            else:
                raise AssertionError("Only two interpolation modes are supported 0 (BILINEAR) or 1 (NEAREST NEIGHBOR)")

            image_crop = torch.unsqueeze(image_crop * mask + extrapolation, 0)
            result = image_crop if result is None else torch.cat((result, image_crop))

        result = torch.tensor(result).to(image.dtype).to(image.device)
        return result

    def _get_masks_for_extrapolation(self, in_x, in_y, img_width, img_height):
        """
        Check if extrapolation is needed and modify in_x, in_y accordingly
        """
        mask_x = torch.logical_or(in_x > (img_width-1), in_x < 0)
        mask_y = torch.logical_or(in_y > (img_height-1), in_y < 0)
        mask = torch.unsqueeze(mask_y, -1) | torch.unsqueeze(mask_x, -1).T
        mask = torch.where(mask, 0, 1)
        extrapolation = torch.logical_not(mask) * self.extrapolation_value
        in_x, in_y = torch.where(mask_x, 0, in_x), torch.where(mask_y, 0, in_y)
        return in_x, in_y, mask, extrapolation

    def _custom_bilinear_interpolation(self, image, in_y, in_x):
        """
        Custom bilinear interpolation to match with Tensorflow expectation
        """
        top_y_index, bottom_y_index = torch.floor(in_y).to(torch.long), torch.ceil(in_y).to(torch.long)
        left_x_index, right_x_index = torch.floor(in_x).to(torch.long), torch.ceil(in_x).to(torch.long)
        y_lerp = in_y - top_y_index
        x_lerp = in_x - left_x_index
        top_left = self._custom_index_select(image, top_y_index, left_x_index)
        top_right = self._custom_index_select(image, top_y_index, right_x_index)
        bottom_left = self._custom_index_select(image, bottom_y_index, left_x_index)
        bottom_right = self._custom_index_select(image, bottom_y_index, right_x_index)
        top = torch.lerp(top_left, top_right, x_lerp)
        bottom = torch.lerp(bottom_left, bottom_right, x_lerp)
        return torch.lerp(top, bottom, torch.unsqueeze(y_lerp, -1))

    def _custom_nearest_neighbor_interpolation(self, image, in_x, in_y):
        """
        Custom nearest neighbor interpolation to match with Tensorflow expectation
        """
        closest_y_index = torch.round(in_y).to(torch.long)
        closest_x_index = torch.round(in_x).to(torch.long)
        return self._custom_index_select(image, closest_y_index, closest_x_index)

    @staticmethod
    def _custom_index_select(a, y_ind, x_ind):
        """
        Gather tensor 'a' along height (y_ind) and width (x_ind) axes
        """
        a = torch.index_select(a, 1, y_ind)
        return torch.index_select(a, 2, x_ind)


class MultiClassNms(torch.nn.Module):
    """
    Custom module for QNN MultiClassNms, only supports Hard NMS for now
    """
    def __init__(self, iou_threshold: float, score_threshold: float = 0.0, soft_nms_sigma: float = 0.0, max_output_boxes_per_batch: int = None):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms_sigma = soft_nms_sigma
        self.max_output_boxes_per_batch = max_output_boxes_per_batch

    # pylint: disable=too-many-locals
    def forward(self, *inp):
        """
        Forward pass for MultiClassNms
        """
        batched_boxes = inp[0]  # [batch, num_boxes, 4]
        batched_scores = inp[1]  # [batch, num_boxes, num_classes]
        batched_features = ()
        num_boxes = batched_boxes.shape[1]
        num_classes = batched_scores.shape[-1]
        if len(inp) > 2:
            batched_features = inp[2:]  # tuple of feature vectors whose dimension is [batch, num_boxes, ...]

        output_classes = None
        output_boxes = None
        output_scores = None
        output_indices = []
        output_features = ()

        # Iterate through batch dimension
        for boxes, scores in zip(batched_boxes, batched_scores):
            # Repeat boxes to allow Multiclass NMS
            boxes = boxes.repeat(num_classes, 1)
            # Maintain original box indices tensor to ensure that NMS gives unique boxes
            box_ind = torch.arange(0, num_boxes, dtype=torch.int64).repeat(num_classes)
            # Flatten scores to allow Multiclass NMS
            scores = scores.transpose(1, 0).flatten()
            # Maintain classes tensor as we are flattening scores
            classes = torch.arange(0, num_classes, dtype=torch.int64).repeat_interleave(num_boxes)

            # TODO: Add support for soft_nms_sigma after converter starts supporting it
            unique_boxes, unique_box_ind, unique_scores, unique_classes = self._perform_hard_nms(boxes, box_ind, scores, classes)
            # Ensure only unique boxes are returned
            if unique_box_ind.shape != torch.unique(unique_box_ind).shape:
                warnings.warn("'torchvision.ops.batched_nms' couldn't return unique boxes")
                # batched NMS returns indices in descending order of scores. As we are performing NMS within
                # each class if the NMS output has same box multiple times (with different class labels),
                # we should pick the first one.
                unique_box_ind, ind_ = self._get_unique_first_indices(unique_box_ind)
                unique_boxes = unique_boxes[ind_]
                unique_scores = unique_scores[ind_]
                unique_classes = unique_classes[ind_]

            b = self._get_zero_filled_tensor(unique_boxes, (1, self.max_output_boxes_per_batch, 4))
            c = self._get_zero_filled_tensor(unique_classes, (1, self.max_output_boxes_per_batch, ))
            s = self._get_zero_filled_tensor(unique_scores, (1, self.max_output_boxes_per_batch, ))

            output_classes = torch.cat([output_classes, c]) if output_classes is not None else c
            output_boxes = torch.cat([output_boxes, b]) if output_boxes is not None else b
            output_scores = torch.cat([output_scores, s]) if output_scores is not None else s
            output_indices.append(unique_box_ind)

        # Iterate through features
        for batched_feature in batched_features:
            # un-batch each feature to gather output features using NMS output indices
            output_feature = [self._custom_unsqueezed_index_select(feature, ind) for ind, feature in zip(output_indices, batched_feature)]
            output_feature_shape = (1, self.max_output_boxes_per_batch, *tuple(batched_feature.shape[2:]))
            # re-batch each output feature
            output_feature = torch.concat([self._get_zero_filled_tensor(feature, output_feature_shape) for feature in output_feature])
            output_features += (output_feature,)

        return (output_boxes, output_scores, output_classes, *output_features)[:len(inp) + 1]

    @staticmethod
    def _modify_y1x1y2x2_to_x1y1x2y2(boxes):
        return boxes[:, torch.tensor([1, 0, 3, 2])]

    @staticmethod
    def _custom_unsqueezed_index_select(tensor: torch.Tensor, ind: torch.Tensor, dim: int = 1):
        return torch.index_select(tensor.unsqueeze(0), dim, ind) if ind.numel() else torch.tensor([])

    @staticmethod
    def _get_unique_first_indices(tensor: torch.Tensor):
        unique, idx, counts = torch.unique(tensor, dim=1, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        return unique, ind_sorted[cum_sum]

    @staticmethod
    def _get_zero_filled_tensor(tensor: torch.Tensor, dim: Union[Tuple, list]):
        """
        Return zero filled tensor with fixed shape, as dynamic shapes are not allowed
        """
        out = torch.zeros(*dim, dtype=tensor.dtype, device=tensor.device)
        indices = torch.arange(0, int(np.prod(tensor.shape)), 1, dtype=torch.int64, device=tensor.device)
        return out.put_(indices, tensor)

    def _perform_hard_nms(self, boxes: torch.Tensor, box_ind: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor):
        """
        Filter bounding boxes across multiple classes in descending order of score using Non-max suppression
        """
        # Filter scores using score threshold
        filtered_score_ind = (scores > self.score_threshold).nonzero()[:, 0]
        filtered_boxes = boxes[filtered_score_ind]
        filtered_box_ind = box_ind[filtered_score_ind]
        filtered_scores = scores[filtered_score_ind]
        filtered_classes = classes[filtered_score_ind]
        # Sort the scores in descending order
        sorted_scores, sorted_ind = torch.sort(filtered_scores, descending=True, stable=True)
        sorted_boxes = filtered_boxes[sorted_ind]
        sorted_box_ind = filtered_box_ind[sorted_ind]
        sorted_classes = filtered_classes[sorted_ind]
        # Use batched_nms, to allow NMS only between boxes of same class
        res_ = torchvision.ops.batched_nms(self._modify_y1x1y2x2_to_x1y1x2y2(sorted_boxes), sorted_scores, sorted_classes, self.iou_threshold)
        if res_.shape[0] > self.max_output_boxes_per_batch:
            res_ = res_[:self.max_output_boxes_per_batch]
        return sorted_boxes[res_], sorted_box_ind[res_], sorted_scores[res_], sorted_classes[res_]


class CustomPReLU(torch.nn.Module):
    """
    Custom PRelu module to allow weights with dims > 1
    """
    def __init__(self, weight_shape: list):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full(weight_shape, fill_value=0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for custom PReLU
        """
        return torch.where(x >= 0, x, 0.0) + self.weight * torch.where(x < 0, x, 0.0)
