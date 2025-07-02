# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""

This tutorial shows how to convert and execute a MobileNetV2 model on the QAIRT HTP Backend.

Model: MobileNetV2
Devices: Android
OS: Linux

"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from e2e_tutorial_utils import (
    IMAGE_DATASET,
    evaluate_prediction,
    postprocess_classification_results,
    preprocess_input,
)

import qairt
from qairt import CompileConfig, Device, DevicePlatformType, ExecutionResult, RemoteDeviceIdentifier
from qairt.api.compiler.backends.htp import HtpGraphConfig

# Step 1: Get a pretrained MobileNetV2 model
pytorch_model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
pytorch_model.eval()

# Create a directory for artifacts
artifacts_dir = Path("./e2e_artifacts").resolve()
artifacts_dir.mkdir(parents=True, exist_ok=True)
onnx_model_path = str(artifacts_dir / "mobilenet_v2.onnx")


# Step 1b: Export the PyTorch model as an ONNX model

# Prepare the dummy input
dummy_input = torch.rand((1, 3, 224, 224), dtype=torch.float32)

torch.onnx.export(
    pytorch_model,
    (dummy_input,),
    onnx_model_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
)

# Run the QAIRT end-to-end workflow
# Step 2: Convert the model
converted_model: qairt.Model = qairt.convert(onnx_model_path)

# Step 2b: View the converted model and reports using QAIRT Visualizer
# Save the converted model as a .dlc file
converted_model_path = converted_model.save(artifacts_dir / "converted_model.dlc")

try:
    from qairt_visualizer import view

    # View the converted model using QAIRT Visualizer
    view(converted_model_path)

    # View the framework model using QAIRT Visualizer
    view(onnx_model_path)
except Exception as e:
    print(f"Failed to view the converted model using QAIRT Visualizer. Error: {e}")

# Step 3: Compile the model. This will run the graph in FP32 mode.
config = CompileConfig(backend="HTP")
compiled_model: qairt.CompiledModel = qairt.compile(converted_model, config=config)

# Alternatively, you can run the graph in FP16 by adding this to the CompileConfig:
# graph_custom_configs = [HtpGraphConfig(name="mobilenetv2", fp16_relaxed_precision=True)]

# Step 4: Create a device, if no device is specified then execution will be performed on the local host
target_device = None
android_serial = os.getenv("ANDROID_SERIAL")
# Optionally set a hostname, if your android device is not connected to a local host
android_hostname = os.getenv("ANDROID_HOSTNAME")

if android_serial:
    print("ANDROID_SERIAL was set. Setting up device configuration...")
    device_id = RemoteDeviceIdentifier(serial_id=android_serial, hostname=android_hostname)
    target_device = Device(identifier=device_id, type=DevicePlatformType.ANDROID)

# Iterate through the set of provided sample images
for label, image_url in IMAGE_DATASET.items():
    # Step 5: Preprocess image
    image: np.ndarray = preprocess_input(image_url).numpy()

    # Step 6: Execute the model on an image
    result: ExecutionResult = compiled_model(image, device=target_device)

    output_name = list(result.data.keys())[0]
    out: np.ndarray = result[output_name]

    # Step 7: Postprocess the results
    qairt_prediction = postprocess_classification_results(out)

    # Step 8: Evaluate the prediction
    evaluate_prediction(qairt_prediction, label)

# Step 9: Save the compiled model
compiled_model.save(artifacts_dir / "compiled_model.bin")

# Save the compiled model info as a json
with open(artifacts_dir / "compiled_model_info.json", "w") as f:
    json.dump(compiled_model.module.info.as_dict(), f, indent=4)
