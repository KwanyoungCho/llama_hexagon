# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""Helper functions for E2E tutorials"""

from pathlib import Path
from typing import BinaryIO, Dict, Union, cast

import numpy as np
import requests
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor

from qairt import ExecutionResult

IMAGE_DATASET = {
    "african elephant": "./images/african_elephant.jpg",
    "samoyed": "./images/samoyed.jpg",
    "espresso": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg",
    "sea lion": "./images/sea_lion.jpg",
}


def preprocess_input(image: str) -> Tensor:
    """
    Preprocess an image by performing transformations that are
    compatible with common object classification models including resizing, cropping and normalization.

    Args:
        image: A path to an image. Either a local path or web link can be passed.

    Returns:
        An output array obtained from transforming the image
    """

    if Path(image).exists():
        image_obj = Image.open(image).convert("RGB")
    else:
        response = requests.get(image, stream=True, timeout=5)
        response.raw.decode_content = True
        raw_data: BinaryIO = cast(BinaryIO, response.raw)
        image_obj = Image.open(raw_data).convert("RGB")

    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = preprocess(image_obj).unsqueeze(0)
    return tensor


def _extract_output_data(output: Union[ExecutionResult, np.ndarray]) -> np.ndarray:
    """
    Extract the numpy array from the output object.

    Args:
        output: The output of the model

    Returns:
        The output numpy array
    """
    if isinstance(output, ExecutionResult) and isinstance(output.data, Dict):
        result = output.data
        output_name = list(result.keys())[0]
        output_ndarray = result[output_name]
    else:
        assert isinstance(output, np.ndarray)
        output_ndarray = output
    return output_ndarray


def postprocess_classification_results(output: Union[ExecutionResult, np.ndarray]) -> str:
    """
    Postprocess the output of a classification model to print the top five predictions using the ImageNet class labels.

    Args:
        output: The output of the model

    Returns:
        The prediction (str)
    """
    output_ndarray = _extract_output_data(output)

    # Softmax function
    on_device_probabilities = np.exp(output_ndarray) / np.sum(np.exp(output_ndarray), axis=1)

    # Read the ImageNet class labels
    sample_classes = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
    response = requests.get(sample_classes, stream=True, timeout=5)
    response.raw.decode_content = True
    categories = [s.strip().decode("utf-8") for s in response.raw]

    # Print the top five predictions
    print("Top-5 predictions:")
    top5_classes = np.argsort(on_device_probabilities[0], axis=0)[-5:]
    prediction = categories[top5_classes[-1]]
    for c in reversed(top5_classes):
        print(f"{c} {categories[c]:20s} {on_device_probabilities[0][c]:>6.1%}")
    print()
    return prediction


def evaluate_prediction(prediction: str, label: str) -> None:
    """Helper method to print the prediction and label as a string pair"""
    prediction = prediction.lower()
    label = label.lower()
    if prediction == label:
        print(f"Successful prediction: {prediction}\n")
    else:
        print(f"Failed prediction: {prediction}. Expected {label}\n")


def postprocess_segmentation_results(pytorch_output: np.ndarray, qairt_output: ExecutionResult) -> None:
    """
    Postprocess the output of a segmentation model to print the Mean Intersection over Union (MIoU) score.

    Args:
        pytorch_output: The output of the PyTorch model
        qairt_output: The output of the QAIRT model
    """

    def _calculate_miou(prediction, target, num_classes):
        """Helper method to calculate the Mean Intersection over Union (MIoU) score."""
        iou_list = []
        for class_idx in range(num_classes):
            pred_inds = prediction == class_idx
            target_inds = target == class_idx
            intersection = np.logical_and(pred_inds, target_inds).sum()
            union = np.logical_or(pred_inds, target_inds).sum()
            if union == 0:
                iou = float("nan")
            else:
                iou = intersection / union
            iou_list.append(iou)
        return np.nanmean(iou_list)

    # Extract output data from ExecutionResult
    qairt_np_output: np.ndarray = _extract_output_data(qairt_output)

    # Calculate MIoU
    assert qairt_np_output.shape == pytorch_output.shape
    prediction = np.argmax(qairt_np_output, axis=0)
    target = np.argmax(pytorch_output, axis=0)
    num_classes = 21

    miou = _calculate_miou(prediction, target, num_classes)
    print(miou)
