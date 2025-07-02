# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_evaluator.common.worker_group.base import WorkerGroup
from qti.aisw.tools.core.utilities.data_processing import Representation
from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager


class TorchScriptWorkerGroup(WorkerGroup):
    """Class representing a worker group for TorchScript inference.
    This class extends the WorkerGroup base class and provides specific implementations
    for TorchScript inference, validation, and teardown.
    """
    def setup_inference_engine(self):
        """Set up the TorchScript inference engine by initializing the framework manager,
        validating the model, and loading the model.
        """
        self.framework_manager = FrameworkManager()
        _ = self.framework_manager.validate(input_model=self.model)
        self.loaded_model = self.framework_manager.load(input_model=self.model)
        if self.output_info:
            self.output_names = list(self.output_info.keys())
        else:
            self.output_names = [op.debugName() for op in self.loaded_model.graph.outputs()]

    def infer(self, input_sample: Representation) -> Representation:
        """Perform inference on the input sample using the TorchScript model.

        Args:
            input_sample: A Representation object containing the input data.

        Returns:
            A Representation object containing the output data from the inference.
        """
        outputs = self.framework_manager.execute(input_model=self.loaded_model,
                                                 input_data=input_sample.data,
                                                output_tensor_names=self.output_names)
        input_sample.data = list(outputs.values())
        return input_sample

    def teardown(self):
        """Teardown the TorchScript inference engine by deleting the loaded model."""
        del self.loaded_model
