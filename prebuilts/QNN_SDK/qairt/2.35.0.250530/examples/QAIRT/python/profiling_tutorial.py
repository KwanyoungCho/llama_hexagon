"""
..  # ==============================================================================
    #
    # Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
    # All Rights Reserved.
    # Confidential and Proprietary - Qualcomm Technologies, Inc.
    #
    # ==============================================================================

.. _profiling:

Profiling Models
====================================================

This tutorial shows how to generate profiling reports on QAIRT backends.

The parameters for this tutorial are as follows:

 - Framework: PyTorch
 - Model: InceptionV3: https://pytorch.org/hub/pytorch_vision_inception_v3/
 - Configurations:

   - Host OS: Linux (x86_64)
   - Target Devices: Snapdragon Android Device
   - Processor: Qualcomm NPU
   - Backend: HTP

.. note: This example is not compatible with Windows targets.

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
from qairt.api.configs.device import DeviceFactory, RemoteDeviceIdentifier

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


if not android_device:
    print("INFO: ANDROID_SERIAL was not set. Exiting")
    exit(1)


##############################################################################
# Generating Profiling Reports
# ----------------------------
#
# Profiling reports can provide insights into the performance of the model on your target device. Reports
# can be generated at different levels to provide finer grained details on metrics such as bandwidth and latency.
# The following sections will guide you through the process of generating different profiling reports for your model.

# In the following sections we will use the view API. To turn off visualization, set the variable
# `ENABLE_QAIRT_VISUALIZER` to False.

if ENABLE_QAIRT_VISUALIZER:
    from qairt_visualizer import view
else:
    print("Visualization is disabled. Please set ENABLE_QAIRT_VISUALIZER to True to enable visualization.")

    # pass through function
    def view(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        pass


####################################################################
# Basic Reports
# ^^^^^^^^^^^^^
# To generate a profiling report, you can instantiate a `qairt.Profiler` instance. The profiler
# acts as a context manager that gathers profiling data from compile and execution calls. You can generate
# basic, detailed, client or backend reports by setting the `level` parameter in the context manager.
# |
# Generating a report is as simple as performing inference, compilation or both within a context
input_array = np.random.randn(1, 3, 224, 224).astype(np.float32)

# You may set the backend here:
desired_backend = "HTP"

with Profiler(context={"level": "basic"}) as profiler:
    # Execute the model directly without compiling. The model will
    # run in FP32
    _ = converted_model(input_array, device=android_device, backend=desired_backend)

    # Generate the profiler report
    basic_model_report = profiler.generate_report()

basic_report_path = str(inceptionv3_artifacts / "profiling_report.json")
basic_model_report.dump(basic_report_path)

####################################################################
# # You can view the report using the `view` function. This will open the report
# # in a QAIRT Visualizer window.
view(basic_report_path)

# ..note: Generating detailed report may take slightly longer due to intermediate
# output tensor information being generated.
####################################################################
# Detailed Reports
# ^^^^^^^^^^^^^^^^

# Create the data as before
input_array = np.random.randn(1, 3, 224, 224).astype(np.float32)

# You may set the backend here
desired_backend = "HTP"

# Generating a detailed report as is simple as changing the context level
with Profiler(context={"level": "detailed"}) as profiler:
    _ = converted_model(input_array, device=android_device, backend=desired_backend)

    # Generate the profiler report
    detailed_model_report = profiler.generate_report()

detailed_report_path = str(inceptionv3_artifacts / "detailed_report.json")
detailed_model_report.dump(detailed_report_path)

####################################################################
# # You can view the detailed report and model using the visualizer.
# This view enables you to interact with the model and report simultaneously.
view(str(converted_model_path), reports=detailed_report_path)


# .. note:: The following section is HTP specific.

####################################################################
# Comparing Reports
# ^^^^^^^^^^^^^^^^
#  In this section, we can explore the impact of applying a few HTP backend settings on generated
#  profiling reports. We'll specialize the compile configuration by enabling fp16 precision and
#  adding soc specific details to improve performance

# To get the specific settings for the given soc, you can use the DeviceFactory API.
# The device factory enables you to interact with devices generically, and retrieve
# information about the device. For example, you can get the soc details for the given chipset.

# Retrieve the chipset
chipset = android_device.get_chipset()

# Then set the soc details
soc_details = DeviceFactory.get_device_soc_details("HTP", chipset)

# You can inspect the soc details to see additional details such as num_of_hvx threads and vtcm size
print(f"Device SoC Details: {soc_details.model_dump_json(indent=4)}\n")

# We'll use the soc details and set the graph to fp16 relaxed precision as before
htp_graph_config = HtpGraphConfig(
    name="inceptionv3",
    fp16_relaxed_precision=True,
    vtcm_size_in_mb=soc_details.vtcm_size_in_mb,
    hvx_threads=soc_details.num_of_hvx_threads,
)

# Create the compile config
compile_config = CompileConfig(
    backend="HTP", soc_details=f"chipset:{chipset}", graph_custom_configs=[htp_graph_config]
)

# Set the HTP performance profile to burst
assert compile_config.device_custom_configs is not None
compile_config.device_custom_configs[0].cores[0].perf_profile = PerfProfile.BURST

with Profiler(context={"level": "basic"}) as profiler:
    # Compile the model
    compiled_model = qairt.compile(converted_model, config=compile_config)

    # Execute the model
    output = compiled_model(input_array, device=android_device)

    # Generate the basic report
    basic_report_optimized = profiler.generate_report()

    # Save the profiler report as a .json file
    basic_report_optimized_path = str(inceptionv3_artifacts / "basic_report_optimized.json")
    basic_report_optimized.dump(basic_report_optimized_path)

# You can view the report using the `view` function. This will open the report
# in a QAIRT Visualizer window.
view(basic_report_optimized_path)

# We should see an improvement in both init, de-init and inferences per second
# as a result of the specialized configuration.

# Next Steps:
# ^^^^^^^^^^
#     - Follow the op trace profiling tutorial to learn how to generate per-op statistics
#     - Follow the quantization tutorial to learn how to reduce the precision of the model.
