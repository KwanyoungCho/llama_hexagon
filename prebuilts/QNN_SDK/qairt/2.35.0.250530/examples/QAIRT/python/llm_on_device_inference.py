"""
..  # ==============================================================================
    #
    # Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
    # All Rights Reserved.
    # Confidential and Proprietary - Qualcomm Technologies, Inc.
    #
    # ==============================================================================

===================================
Gen AI API Introduction
===================================
Welcome to the Quick Start Guide to Deploying Large Language Models (LLM) on Snapdragon Devices!
This guide will walk you through the process of deploying an LLM on a Snapdragon device
using the QAIRT Tools Python API.


Overview
--------------
The QAIRT Tools Python API for Gen AI includes two main steps:

1. Creating QAIRT assets:
   This guide introduces a new API which encapsulates,`GenAIBuilder.build()``, which encapsulates model preparation.

2. Performing inference on device. The following guide will focus on text generation

Prerequisites
----------------------
   - This quick start guide uses the Meta Llama 3-8b-instruct model. You can download the model from hugging face: `Llama-3-8B-Instruct <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`_
     using a valid license.
   - The guide assumes you have obtained quantized ONNX model and associated model artifacts which have been generated using
     Step 1 of this notebook: `Llama 3-8b <https://qpm.qualcomm.com/#/main/tools/details/Tutorial_for_Llama3>`_
   - Upon completion of Step 1, the following directory and naming structure is expected:
     ::

         <model_exports>.llama
            onnx/<model>.onnx
            onnx/<model>.encodings
            onnx/<model>.data (optional)
            config.json
            tokenizer.json
   - You can also obtain `Llama3-8b Config <https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json>`_ and
     `Tokenizer <https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/tokenizer.json>`_ json files
     from Hugging Face directly.
   - The guide uses a **Snapdragon SD 8 Elite (SM8750) Android device** to demonstrate the workflow.
   - We recommend a machine with at least 64 GB of RAM for timely completion of the workflow. If you do not have sufficient RAM, we recommend increasing
     your swap memory. This workflow may take at **least 40 minutes** on a machine with RAM < 64 GB.

**Important:** Set the environment variable **QAIRT_TMP_DIR** to define an alternative default temporary directory path.
              This is recommended because temporary artifacts are created during build process below which may consume temp memory entirely.
"""

############################################################
# Setup
# ------------------------
import os

import qairt
from qairt import Device, DevicePlatformType, RemoteDeviceIdentifier
from qairt.gen_ai_api.builders.gen_ai_builder_htp import GenAIBuilder
from qairt.gen_ai_api.builders.llama3_builder_htp import Llama3BuilderHTP
from qairt.gen_ai_api.containers.gen_ai_container import GenAIContainer
from qairt.gen_ai_api.containers.llm_container import LLMContainer
from qairt.gen_ai_api.executors.t2t_executor import T2TExecutor
from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory

############################################################
# Define the path containing onnx model exports
llama3_exports = "./llama_3_8b/<your_path>"

# Set QAIRT_TMP_DIR
os.environ["QAIRT_TMP_DIR"] = "./llm_scratch/"

# Set a cache directory. We'll go over how caching helps in the sections below
CACHE_ROOT = "./llama3_cache"

############################################################
# Building a model for HTP
# ------------------------
# Preparing an LLM for deployment involves conversion, optimization, and compilation into backend-compatible formats.
# Here's an example of how to build a QAIRT model using the GenAI Builder API:

# Create a builder instance, and an optional cache root to store intermediate artifacts.
llama_builder: GenAIBuilder = GenAIBuilderFactory.create(llama3_exports, "HTP", cache_root=CACHE_ROOT)

############################################################
# The factory will inspect the config.json for the model and determine what builder is appropriate.
# For this example it will determine that the Llama3BuilderHTP instance is appropriate, and return a
# constructed instance of that subclass. The builder's compilation configuration requires customization
# for the intended device.

# Ensure the appropriate builder is returned
assert isinstance(llama_builder, Llama3BuilderHTP)

# Customize the builder for the device. This is needed for AOT compilation on HTP.
llama_builder.set_targets([f"chipset:SM8750"])

############################################################
# Once a target is set, you can trigger the build process to build the Gen AI model into an LLM container object
llama_container: GenAIContainer = llama_builder.build()

############################################################
# Build Steps for HTP
# ^^^^^^^^^^^^^^^^^^^
# Building a model for HTP involves the following steps:
#    -  Transformations that are performed to adapt the framework model for on-device inference.
#    -  Conversion and compilation into one or more compiled model objects. Conversion can include the application of quantization encodings if encodings are provided.
#    -  Each step can be configured using setter methods in the builder class.
#
#
# Caching
# ^^^^^^^
# Building can be time and memory consuming, especially for large models. It may be beneficial to stop/resume the building process between steps.
# To aid that workflow, the GenAI Builder API provides a caching mechanism to help store intermediate artifacts.
#
# To resume a build from the cache shown above, simply initialize the factory with the existing cache path:
#
existing_cache_dir = CACHE_ROOT
genai_builder = GenAIBuilderFactory.create(llama3_exports, "HTP", cache_root=existing_cache_dir)

############################################################
# The builder identifies metadata in the cache dir that enables the process to be resumed from the last known stage if no
# configuration changes have been made.
#
# .. tip::
#
#       It is recommended to use cache_root to save any intermediate artifacts generated during building. The cache enables building
#       to be paused and resumed between transformation, conversion and compilation stages.

############################################################
# Set up an Android device
# ------------------------
# Reusing the android setup instructions from the `:ref:quick_start` guide:

android_serial = os.getenv("ANDROID_SERIAL")
android_hostname = os.getenv("ANDROID_HOSTNAME")

if android_hostname:
    device_id = RemoteDeviceIdentifier(serial_id=android_serial, hostname=android_hostname)
else:
    device_id = android_serial

android_device = Device(identifier=device_id, type=DevicePlatformType.ANDROID)

###########################################################################
# Generate Text with Llama 3
# --------------------------
# The container generated above is used to create an executor. The executor is responsible for interfacing with the device
# and performing inference. Each executor is customized to a target and specific inference mode, in this example,
# the executor is a *text to text executor*.
#
from qairt.gen_ai_api.executors.gen_ai_executor import GenAIExecutor, GenerationExecutionResult

llm: GenAIExecutor = llama_container.get_executor(android_device, clean_up=False)

# Ensure the appropriate executor is returned
assert isinstance(llm, T2TExecutor)

###########################################################################
# Below we define the prompt template and prompt according to the specification for
# the Llama 3-8b chat variant.

prompt_template = (
    "<|begin_of_text|>"
    "<|start_header_id|>{system}<|end_header_id|>{system_prompt}<|eot_id|>"
    "<|start_header_id|>{user}<|end_header_id|>{user_prompt}<|eot_id|>"
    "<|start_header_id|>{assistant}<|end_header_id|>"
)
prompt = prompt_template.format(
    system="system",
    system_prompt="Be helpful but try to limit answers to 40 words.",
    user="user",
    user_prompt="What can I do with a glass jar?",
    assistant="assistant",
)

# Generate text
result: GenerationExecutionResult = llm.generate(prompt)

# The command above will generate the following output:
print(result.generated_text)

#####################################################################
# Metrics can be inspected and printed to the console
print(result.metrics)

# This call will remove artifacts that were generated to execute on the device
llm.clean_environment()

######################################################################
# Export
# --------------
# You can also save the container and its associated artifacts.
llama_container.save("./llama3_container", exist_ok=True)

######################################################################
# Container Contents
# ^^^^^^^^^^^^^^^^^^
#     -  The container contains compiled binaries that are needed for deployment.
#     -  Containers may be reloaded from disk at a later stage:
#

from qairt.gen_ai_api.containers.llm_container import LLMContainer

# Load a container
container = LLMContainer.load("./llama3_container")
