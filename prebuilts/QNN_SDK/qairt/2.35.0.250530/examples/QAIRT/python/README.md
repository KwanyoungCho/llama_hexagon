# QAIRT Tools Python API Tutorials

Here we provide some simple tutorials to help users get started with the QAIRT Tools Python API.

## Requirements

- **QAIRT SDK**: Version: 2.34.0 or variant
- **Python**: 3.10
- **Host OS**: Linux (x86_64), Windows on Snapdragon (ARM64)
- **Target OS**: Linux (x86_64), Android, Windows on Snapdragon (ARM64)
- **Frameworks**: ONNX, PyTorch, TFLite

**Important**: For Windows on Snapdragon, the platform libraries and binaries must be *arm64x* not *aarch64*.

## Structure

- *basic_tutorial.py*: Offers a quick introduction to the API
- *quantization_tutorial.py*: Showcases quantization workflow using DeeplabV3
- *profiling_tutorial.py*: Shows how generate standard profiling reports for QAIRT on Android targets
- *op_trace_profiling_tutorial.py*: Shows how to generate op trace reports for QAIRT on Android and X Elite for HTP.
- *llm_on_device_inference.py* : Showcases the Gen AI workflow for executing with HTP on Android

## Steps

1. **Obtain a QAIRT SDK**:
    Download the [QAIRT SDK](https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct>)

2. **Create a Virtual Environment**:
   Ensure the QAIRT SDK is installed. Follow these instructions for setup: [Setup](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/setup.html>)

3. **Install Framework Packages**:
    Install the applicable framework packages (ONNX and PyTorch).
    See supported versions here: [QAIRT Framework Dependencies](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/linux_setup.html#step-3:-install-model-frameworks>)

4. **Install QAIRT Visualizer**:
    Install the QAIRT Visualizer `pip install qairt-visualizer`

5. **Quick Start**:
   Run `python tutorials/basic_tutorial.py`.

6. **Set Environment Variables**:
   For android device execution, set the `ANDROID_SERIAL` to a valid ADB ID. Optionally, also set `ANDROID_HOSTNAME` for
   execution on a remote host.
