# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os

from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.common.worker_group.base import WorkerGroup
from qti.aisw.tools.core.utilities.data_processing import Representation


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TensorflowRTWorkerGroup(WorkerGroup):
    """A class representing a worker group for TensorFlow RT inference.
    This class extends the WorkerGroup base class and provides specific implementations
    for TensorFlow RT inference, validation, and teardown.
    """

    def validate(self):
        """Validate the TensorFlow RT environment and set the logger severity level.

        Raises:
            ImportError: If TensorFlow is not found.
        """
        Helper.safe_import_package("tensorflow", "2.10.0")

    def setup_inference_engine(self):
        """Set up the Tensorflow RT inference engine by loading the model."""
        tf = Helper.safe_import_package("tensorflow")
        with tf.io.gfile.GFile(self.model, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        self.frozen_func = self.wrap_frozen_graph(
                graph_def=graph_def,
                inputs=[node_name + ':0' for node_name in self.input_names],
                outputs=[out_name + ':0' for out_name in self.output_names],
            )

    def infer(self, input_sample: Representation) -> Representation:
        """Perform inference on the input sample using the Tensorflow model.

        Args:
            input_sample: A Representation object containing the input data.

        Returns:
            A Representation object containing the output data from the inference.
        """
        tf = Helper.safe_import_package("tensorflow")
        input_data = []
        for idx, data in enumerate(input_sample.data):
            input_data.append(tf.convert_to_tensor(data,
                                Helper.get_np_dtype(self.input_info[self.input_names[idx]][0],
                                                    map_tf=True)))

        frozen_graph_predictions = self.frozen_func(*input_data)
        outputs = []
        for elem in frozen_graph_predictions:
            outputs.append(elem.numpy())
        input_sample.data = outputs
        return input_sample

    def wrap_frozen_graph(self, graph_def, inputs, outputs):
        """This method converts frozen graph to ConcreteFunction."""
        tf = Helper.safe_import_package("tensorflow")

        def _imports_graph_def():
            tf = Helper.safe_import_package("tensorflow")
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        return wrapped_import.prune(tf.nest.map_structure(import_graph.as_graph_element, inputs),
                                    tf.nest.map_structure(import_graph.as_graph_element, outputs))

    def teardown(self):
        """Teardown the TorchScript inference engine by deleting the loaded model."""
        del self.frozen_func
