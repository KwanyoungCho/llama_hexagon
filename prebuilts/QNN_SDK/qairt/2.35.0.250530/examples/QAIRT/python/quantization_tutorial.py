# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
End-to-end Tutorial showcasing Calibration
Model: DeepLabV3 with ResNet101 backbone
Link: https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.deeplabv3_resnet101
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from e2e_tutorial_utils import postprocess_segmentation_results, preprocess_input
from torch import Tensor

import qairt
from qairt import Device, DevicePlatformType, RemoteDeviceIdentifier
from qairt.api.converter.converter_config import CalibrationConfig


class DeepLabModelWrapper(torch.nn.Module):
    """
    Wrapper class for the DeepLabV3 PyTorch model.
    Note: This wrapper class is necessary because the DeepLabV3 model returns a dictionary by default,
    rather than just the desired semantic masks tensor needed for inference mode.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"][0]


def run_pytorch_inference(model, input_tensor: Tensor) -> np.ndarray:
    """Function that runs the PyTorch model using the native torch runtime."""
    # Execute with torch
    with torch.no_grad():
        return model(input_tensor)["out"][0].numpy()


# Step 1: Get a pretrained DeepLabV3 ResNet101 model
model = torch.hub.load(
    "pytorch/vision:v0.10.0", "deeplabv3_resnet101", weights="DeepLabV3_ResNet101_Weights.DEFAULT"
)
model.eval()

# Create a directory for artifacts
artifacts_dir = Path("./e2e_artifacts").resolve()
artifacts_dir.mkdir(parents=True, exist_ok=True)
pytorch_model_path = artifacts_dir / "deeplabv3_resnet101.pt"

# Step 1b: Save the PyTorch model as a .pt file
wrapped_model = DeepLabModelWrapper(model)
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(wrapped_model, example_input)
torch.jit.save(traced_model, pytorch_model_path)

# Step 2: Preprocess the image
SAMPLE_IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png"
input_tensor: Tensor = preprocess_input(SAMPLE_IMAGE_URL)
input_np_array: np.ndarray = input_tensor.numpy()

# Step 3: Run pytorch inference (golden reference)
print("Running PyTorch inference...")
pytorch_result = run_pytorch_inference(model, input_tensor)

# Step 4: Convert the model using calibration data
# Initialize the calibration config (Note: Weight, bias, and activation values default to 8-bit)
print("Converting the PyTorch model to a QAIRT model using calibration data...")
calibration_config = CalibrationConfig(dataset="input_list.txt")

converted_model: qairt.Model = qairt.convert(
    pytorch_model_path,
    calibration_config=calibration_config,
    input_tensor_config=[{"name": "x", "shape": (1, 3, 224, 224)}],
)

# Step 5: Compile the model for HTP backend
print("Compiling the model for HTP backend...")
compiled_model: qairt.CompiledModel = qairt.compile(converted_model, backend="HTP")

# Step 6: Create a device, if no device is specified then execution will be performed on the local host
target_device = None
android_serial = os.getenv("ANDROID_SERIAL")
# Optionally set a hostname, if your android device is not connected to a local host
android_hostname = os.getenv("ANDROID_HOSTNAME")

if android_serial:
    print("ANDROID_SERIAL was set. Setting up device configuration...")
    device_id = RemoteDeviceIdentifier(serial_id=android_serial, hostname=android_hostname)
    target_device = Device(identifier=device_id, type=DevicePlatformType.ANDROID)

# Step 7: Execute the model on the image
print("Running QAIRT inference...")
result = compiled_model(input_np_array, device=target_device)

# Step 8: Postprocess the results
print("Mean IOU: ", end="")
postprocess_segmentation_results(pytorch_result, result)

# Step 9: Save the compiled model
compiled_model.save(artifacts_dir / "compiled_model.bin")

# Step 10: Save the compiled model info as a json
with open(artifacts_dir / "compiled_model_info.json", "w") as f:
    json.dump(compiled_model.module.info.as_dict(), f, indent=4)
