# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def histogram_visualizer(golden_data: NDArray, target_data: NDArray, file_path: Path):
    """Histogram visualization method.

    Args:
        golden_data (NDArray): Golden reference output tensor.
        target_data (NDArray): Inference engine tensor output.
        file_path (Path): Destination file path for the plot.
    """
    # Calculate the mean, median and standard deviation for golden and target outputs.
    mean_golden = np.mean(golden_data)
    mean_target = np.mean(target_data)
    median_golden = np.median(golden_data)
    median_target = np.median(target_data)
    std_golden = np.std(golden_data)
    std_target = np.std(target_data)

    # Create histograms
    plt.hist(golden_data, bins=30, alpha=0.6, label="Golden Data")
    plt.hist(target_data, bins=30, alpha=0.6, label="Target Data")

    # Add title and labels
    plt.title("Comparison of Golden Data vs. Target Data")
    plt.xlabel("Tensor value Range")
    plt.ylabel("Frequency")
    plt.legend(loc="upper left")

    # Add annotations for mean, median, and standard deviation
    info_text = (
        f"Golden: Mean: {mean_golden:.2f}, Median: {median_golden:.2f},"
        f"Std Dev: {std_golden:.2f}\n"
        f"Target: Mean: {mean_target:.2f}, Median: {median_target:.2f},"
        f"Std Dev: {std_target:.2f}"
    )
    plt.annotate(
        info_text,
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Save the plot to given file path
    plt.savefig(file_path)
    plt.close()


def diff_visualizer(golden_data: NDArray, target_data: NDArray, file_path: Path):
    """Diff Visualization method.

    Args:
        golden_data (NDArray): Golden reference output tensor.
        target_data (NDArray): Inference engine tensor output.
        file_path (Path): Destination file path for the plot.
    """
    if golden_data.shape != target_data.shape:
        raise ValueError(
            f"Shape mismatch: Golden data shape {golden_data.shape} "
            f"does not match target data shape {target_data.shape}."
        )

    # Calculate the mean, median and standard deviation for golden and target outputs.
    mean_golden = np.mean(golden_data)
    mean_target = np.mean(target_data)
    median_golden = np.median(golden_data)
    median_target = np.median(target_data)
    std_golden = np.std(golden_data)
    std_target = np.std(target_data)

    # Ensure the data is converted to NumPy arrays
    golden_data = np.asarray(golden_data)
    target_data = np.asarray(target_data)

    # Create a DataFrame with the difference between target and golden data
    df = pd.DataFrame(
        {
            "Index": np.arange(len(golden_data)),
            "Difference": golden_data - target_data,
            "Golden Data": golden_data,
            "Target Data": target_data,
        }
    )

    # Create the scatter plots
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Index"], df["Golden Data"], color="green", label="Golden Data", alpha=0.6)
    plt.scatter(df["Index"], df["Target Data"], color="blue", label="Target Data", alpha=0.6)

    # Create the line plot for the difference
    plt.plot(df["Index"].values, df["Difference"].values, color="red", label="Difference")

    # Add title and labels
    plt.title("Golden vs. Target Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend(loc="upper left")

    # Add annotations for hover information
    info_text = (
        f"Golden: Mean: {mean_golden:.2f}, Median: {median_golden:.2f},"
        f"Std Dev: {std_golden:.2f}\n"
        f"Target: Mean: {mean_target:.2f}, Median: {median_target:.2f},"
        f"Std Dev: {std_target:.2f}"
    )
    plt.annotate(
        info_text,
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Save the plot to given file path
    plt.savefig(file_path)
    plt.close()


def cdf_visualizer(golden_data: NDArray, target_data: NDArray, file_path: Path):
    """Cumulative Distribution Visualization method.

    Args:
        golden_data (NDArray): Golden reference output tensor.
        target_data (NDArray): Inference engine tensor output.
        file_path (Path): Destination file path for the plot.
    """
    # Calculate histograms and CDFs
    golden_data_hist, golden_data_edges = np.histogram(golden_data, bins=256)
    golden_data_centers = (golden_data_edges[:-1] + golden_data_edges[1:]) / 2
    golden_data_cdf = np.cumsum(golden_data_hist) / golden_data_hist.sum()
    target_data_hist, target_data_edges = np.histogram(target_data, bins=256)
    target_data_centers = (target_data_edges[:-1] + target_data_edges[1:]) / 2
    target_data_cdf = np.cumsum(target_data_hist) / target_data_hist.sum()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(golden_data_centers, golden_data_cdf, label="Golden data CDF", color="green")
    plt.plot(target_data_centers, target_data_cdf, label="Target data CDF", color="blue")

    # Customize the layout
    plt.title("CDF Comparison: Golden Data vs. Target Data")
    plt.xlabel("Data Values")
    plt.ylabel("Cumulative Probability")
    plt.legend(loc="upper left")
    # Save the plot to given file path

    plt.savefig(file_path)
    plt.close()


def line_plot(x: list | NDArray, y: list | NDArray, plot_name: str, save_dir: Path) -> None:
    """Plots and saves a line plot for the given x and y co-ordinates

    Args:
        x: Either list or numpy array of x co-ordinates
        y: Either list or numpy array of y co-ordinates
        plot_name: Name of the plot (same name would be used while saving plot)
        save_dir: Path to location where plot needs to be saved
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=plot_name, line=dict(color="blue")))
    fig.update_layout(
        title=f"{plot_name} line plot",
        xaxis_title="Layers",
        yaxis_title=plot_name,
        legend=dict(x=0.85, y=0.8),
    )
    html_path = save_dir / f"{plot_name}.html"
    fig.write_html(html_path)
