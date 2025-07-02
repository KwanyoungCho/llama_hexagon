# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging

from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.common.worker_group.base import WorkerGroup
from qti.aisw.tools.core.utilities.data_processing import Representation
from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager


class OnnxRTWorkerGroup(WorkerGroup):
    """Class representing a worker group for ONNXRuntime inference.
    This class extends the WorkerGroup base class and provides specific implementations
    for ONNX Runtime inference, validation, and teardown.
    """
    def validate(self):
        """Validate the ONNX Runtime environment and set the logger severity level.

        Raises:
            ImportError: If ONNX Runtime is not found.
        """
        onnxruntime = Helper.safe_import_package("onnxruntime", "1.17.1")
        onnxruntime.set_default_logger_severity(3)

    def setup_inference_engine(self):
        """Set up the ONNX Runtime inference engine by initializing the framework manager,
        validating the model, and loading the model.
        """
        self.framework_manager = FrameworkManager()
        logging.getLogger("OnnxModelHelper").setLevel("ERROR")  # Supress Warnings from onnxruntime
        _ = self.framework_manager.validate(input_model=self.model)
        self.onnx_model = self.framework_manager.load(input_model=self.model)

    def infer(self, input_sample: Representation) -> Representation:
        """Perform inference on the input sample using the ONNX Runtime model.

        Args:
            input_sample: A Representation object containing the input data.

        Returns:
            A Representation object containing the output data from the inference.
        """
        outputs = self.framework_manager.execute(input_model=self.onnx_model,
                                                 input_data=input_sample.data)
        input_sample.data = list(outputs.values())
        return input_sample

    def teardown(self):
        """Teardown the ONNX Runtime inference engine by deleting the loaded model."""
        del self.onnx_model
