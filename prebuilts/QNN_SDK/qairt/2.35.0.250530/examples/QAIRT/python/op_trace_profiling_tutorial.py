"""
..  # ==============================================================================
    #
    # Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
    # All Rights Reserved.
    # Confidential and Proprietary - Qualcomm Technologies, Inc.
    #
    # ==============================================================================

.. _profiling:

Generating Op Trace for HTP
====================================================

This tutorial shows how obtain Op Trace and Qualcomm Hexagon Analysis (QHAS) reports on QAIRT HTP backend

The parameters for this tutorial are as follows:

 - Framework: PyTorch
 - Model: InceptionV3: https://pytorch.org/hub/pytorch_vision_inception_v3/
 - Configurations:

   - Host OS: Linux (x86_64)
   - Target Devices: Snapdragon Android Device
   - Processor: Qualcomm NPU
   - Backend: HTP


   - Host OS: Windows on Snapdragon (ARM64)
   - Target Devices: Snapdragon X Elite Device
   - Processor: Qualcomm NPU
   - Backend: HTP

.. tip::
       This tutorial creates some temporary files as part of the workflow. To customize the temporary file
       location, set the env variable `QAIRT_TMP_DIR` to a location of your choosing.
"""

####################################################################
# Setup
# -----------------------------------
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.models as models

import qairt
from qairt import CompileConfig, Device, DevicePlatformType, Profiler
from qairt.api.compiler.backends.htp import HtpDeviceConfig, HtpGraphConfig, PerfProfile
from qairt.api.configs.device import RemoteDeviceIdentifier

# Set this to change the device used in this tutorial. Choices are "default", "android"
TARGET_DEVICE_TYPE = "android"
TARGET_DEVICE = None


# This example uses the qairt visualizer. See the profiling section for more details.
# We strongly recommend using the `QAIRT Visualizer <https://docs.qualcomm.com/bundle/publicresource/topics/80-87189-1/overview.html>`_
# to visualize the reports generated.

# Alternatively, you can disable visualization for this tutorial by setting this flag to False.
ENABLE_QAIRT_VISUALIZER = True

####################################################################
# Get an InceptionV3 model
# -----------------------------------
# Step 1: Get a pretrained InceptionV3 model
inceptionv3_artifacts = Path("./inceptionv3_artifacts")
inceptionv3_artifacts.mkdir(exist_ok=True)

# Create a directory for artifacts
model = models.inception_v3(weights="Inception_V3_Weights.DEFAULT")
model.eval()


# Step 1b: Export the PyTorch model as an ONNX model
# Prepare the dummy input
dummy_input = torch.rand((1, 3, 224, 224), dtype=torch.float32)
inceptionv3_model_path = str(inceptionv3_artifacts / "inceptionv3.onnx")

torch.onnx.export(
    model,
    (dummy_input,),
    inceptionv3_model_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
)

###############################################################################
# Convert the model
# -----------------

# Step 2: Convert the model and set framework trace to true
# *Framework Trace*: The enable framework trace flag creates a mapping
# from framework operations to their corresponding QAIRT operations.
converted_model = qairt.convert(inceptionv3_model_path, enable_framework_trace=True)

converted_model_path = inceptionv3_artifacts / "inceptionv3_converted.dlc"
converted_model.save(converted_model_path)

###############################################################################
# Set up an Android device
# ------------------------
# Reusing the android setup instructions from the `:ref:quick_start` guide:
android_serial = os.getenv("ANDROID_SERIAL")
android_hostname = os.getenv("ANDROID_HOSTNAME")

android_device = None

if android_serial:
    print(f"INFO: ANDROID_SERIAL : {android_serial} was set. Enabling tutorial for Android")
    device_id = RemoteDeviceIdentifier(serial_id=android_serial, hostname=android_hostname)
    android_device = Device(identifier=device_id, type=DevicePlatformType.ANDROID)

if TARGET_DEVICE_TYPE == "android":
    if not android_device:
        print("ERROR: TARGET_DEVICE_TYPE is 'android' but ANDROID_SERIAL was not set. Exiting tutorial")
        exit(1)
    else:
        TARGET_DEVICE = android_device

###############################################################################
# Set up an X Elite device
# ------------------------
#
# To execute locally on an X Elite target, the device should be set to none.
#
# The API will detect the platform processor and trigger execution using X-Elite as both host and target.
# To ensure compatibility for this tutorial, we provide the following validation below.
if platform.system() == "Windows" and "ARMv8" in platform.processor():
    print("INFO: X Elite detected. Enabling tutorial for X Elite")

##############################################################################
# Generating Profiling Reports
# ----------------------------
# In the following sections we will use the view API. To turn off visualization, set the variable
# `ENABLE_QAIRT_VISUALIZER` to False.

if ENABLE_QAIRT_VISUALIZER:
    from qairt_visualizer import view
else:
    print("Visualization is disabled. Please set ENABLE_QAIRT_VISUALIZER to True to enable visualization.")

    # define dummy view function
    def view(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        pass


# To make things easier, we'll assume an Android device or a Snapdragon X Elite SC8380XP.
if TARGET_DEVICE_TYPE == "android" and TARGET_DEVICE:
    chipset = TARGET_DEVICE.get_chipset()
else:
    chipset = "SC8380XP"  # change this to a different chipset as appropriate.

# We'll use the soc details and set the graph to fp16 relaxed precision as before
htp_graph_config = HtpGraphConfig(name="inceptionv3", fp16_relaxed_precision=True)

# Create the compile config
compile_config = CompileConfig(
    backend="HTP", soc_details=f"chipset:{chipset}", graph_custom_configs=[htp_graph_config]
)

# Set the HTP performance profile to burst
assert compile_config.device_custom_configs is not None
compile_config.device_custom_configs[0].cores[0].perf_profile = PerfProfile.BURST


###############################################################################
# Op Trace Reports
# ^^^^^^^^^^^^^^^^^^
#
# Op trace provides internal graph execution details in the form of per-operation cycle counts across
# each hardware thread.
# This information is useful for identifying performance bottlenecks.
#
# To generate an op-trace, we need to compile and execute the model in a profiler context. The profiler
# gathers the report from each call to compile and execute.

input_array = np.random.randn(1, 3, 224, 224).astype(np.float32)

with Profiler(context={"level": "detailed", "option": "optrace"}) as profiler:
    # Compile the model
    compiled_model = qairt.compile(converted_model, config=compile_config)

    # Execute the model
    _ = compiled_model(input_array, device=TARGET_DEVICE)

    # Generate the op trace report
    op_trace_report = profiler.generate_report()

    # Save the profiler report as a .json file
    op_trace_report_path = str(inceptionv3_artifacts / "op_trace_report.json")
    op_trace_report.dump(op_trace_report_path)

###############################################################################
# View the converted model and op trace in the same window. You can interact
# with the model and observe the op trace nodes graphically.
# To view in separate windows, set options=DisplayOptions(use_same_workspace=False)
view(str(converted_model_path), reports=op_trace_report_path)


###############################################################################
# Qualcomm Hexagon Analysis Summary (QHAS) Report
# ^^^^^^^^^^^^^^^^^^^^^^
# A QHAS Report includes a summary of overall HTP resource utilization, active cycles, dominant cycle path,
# # and tracing back to the original QNN graph.
#
# To view the QHAS report, we need to save the report as a .json file and then view it using the visualizer.

###############################################################################
qhas_report_path = str(inceptionv3_artifacts / "qhas_report.json")
op_trace_report.summary.dump(qhas_report_path)
view(reports=qhas_report_path)

###############################################################################
# QHAS can be obtained directly from an Op trace report.
# We'll print the summary excluding HMX, HVX details for brevity.
# To observe those details, print data["htp_resources"]
qhas_report = op_trace_report.summary
data = qhas_report.data["data"]["htp_overall_summary"]["data"][0]

qhas_summary = {}
for key, value in data.items():
    if key != "htp_resources":
        qhas_summary[key] = value

print("QHAS Summary: \n")
print(json.dumps(qhas_summary, indent=4))
