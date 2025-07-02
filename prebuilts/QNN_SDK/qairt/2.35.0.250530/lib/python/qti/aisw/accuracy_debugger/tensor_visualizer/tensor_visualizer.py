# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib
import numpy as np
from numpy.typing import DTypeLike, NDArray
from qti.aisw.accuracy_debugger.tensor_visualizer.visualizers import (
    cdf_visualizer,
    diff_visualizer,
    histogram_visualizer,
)
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


matplotlib.use("Agg")


class TensorVisualizer:
    """Tensor Visualizer Class"""

    TensorType = Union[Path, str, dict[str, NDArray]]

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize Tensor Visualizer

        Args:
            logger (logging.Logger, optional): logger object. Defaults to None.
        """
        if logger:
            self.logger = logger
        else:
            self.log_area = LogAreas.register_log_area("Tensor Visualizer")
            self.logger = QAIRTLogger.register_area_logger(
                area=self.log_area,
                level="INFO",
                formatter_val="simple",
                handler_list=["dev_console"],
            )

    def run(
        self,
        target_tensors: TensorType,
        golden_tensors: TensorType,
        output_dir_path: Optional[Path | str] = None,
        dtype: DTypeLike = np.float32,
    ) -> Path:
        """Tensor Visualizer run method.

        Args:
            target_tensors (TensorType): Output from Inference Run.
            golden_tensors (TensorType): Output from Framework Run
            output_dir_path (Optional[Path  |  str], optional): Output directory name.
            Defaults to None.
            dtype (DTypeLike, optional): Data type of the tensors. Defaults to np.float32.

        Returns:
            Path: Output directory path.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = Path(output_dir_path or "./tensor_visualizer_output_dir").resolve()
        output_dir_path = base_path / timestamp
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if isinstance(target_tensors, (Path, str)) and isinstance(golden_tensors, (Path, str)):
            # Get the list of .raw files in both folders
            target_files = {f for f in os.listdir(target_tensors) if f.endswith(".raw")}
            golden_files = {f for f in os.listdir(golden_tensors) if f.endswith(".raw")}

            # Find common files in both folders
            common_files = target_files.intersection(golden_files)
            if not common_files:
                raise ValueError("No common .raw files found in both folders.")

            # Create a subdirectory for each common file and save the visualizations
            for file_name in common_files:
                tensor_specific_output_dir = output_dir_path / Path(file_name).stem
                tensor_specific_output_dir.mkdir(parents=True, exist_ok=True)

                target_tensor = os.path.join(target_tensors, file_name)
                golden_tensor = os.path.join(golden_tensors, file_name)

                target_data = np.fromfile(target_tensor, dtype=dtype)
                golden_data = np.fromfile(golden_tensor, dtype=dtype)

                self.run_visualizer(golden_data, target_data, file_name, tensor_specific_output_dir)
        elif isinstance(target_tensors, dict) and isinstance(golden_tensors, dict):
            # Find common keys in both dictionaries
            common_keys = target_tensors.keys() & golden_tensors.keys()
            for key in common_keys:
                tensor_specific_output_dir = output_dir_path / key
                tensor_specific_output_dir.mkdir(parents=True, exist_ok=True)

                target_data = target_tensors[key].flatten()
                golden_data = golden_tensors[key].flatten()
                self.run_visualizer(golden_data, target_data, key, tensor_specific_output_dir)
        else:
            error_message = (
                "Both target_tensors and golden_tensors must be either paths to "
                "outputs or dictionaries."
            )
            self.logger.error(error_message)
            raise TypeError(error_message)

        return output_dir_path

    def run_visualizer(
        self,
        golden_data: NDArray,
        target_data: NDArray,
        file_name: str,
        tensor_specific_output_dir: Path,
    ):
        """Method to run the visualizations.

        Args:
            golden_data (NDArray): Framework run numpy array.
            target_data (NDArray): Inference run numpy array.
            file_name (str): Tensor name/File name.
            tensor_specific_output_dir (Path): Tensor specific output directory.
        """
        try:
            histogram_visualizer(
                golden_data, target_data, tensor_specific_output_dir / "Histograms.jpeg"
            )
        except Exception as e:
            self.logger.error(f"Histogram visualization failed for tensor: {file_name}. {e}")

        try:
            diff_visualizer(
                golden_data, target_data, tensor_specific_output_dir / "Diff_plots.jpeg"
            )
        except Exception as e:
            self.logger.error(f"Diff visualization failed for tensor: {file_name}. {e}")

        try:
            cdf_visualizer(golden_data, target_data, tensor_specific_output_dir / "CDF_plots.jpeg")
        except Exception as e:
            self.logger.error(f"CDF visualization failed for tensor: {file_name}. {e}")
