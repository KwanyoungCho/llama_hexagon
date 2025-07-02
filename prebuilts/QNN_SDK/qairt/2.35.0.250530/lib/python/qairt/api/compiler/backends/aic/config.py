# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
from typing import Any, Dict, List, Literal

from pydantic import Field

from qairt.api.configs.common import AISWBaseModel
from qti.aisw.tools.core.modules.api.backend.aic_backend import AicBackend

""" This module describes options that can be used to configure the AIC backend during
    compilation or execution only using this Python API. It is not intended as a reference QAIRT AIC Backend
    options. For information on QAIRT AIC Backend API, see QAIRT docs."""


class AicCompilerConfig(AISWBaseModel):
    """Compiler configuration for AIC Backend"""

    graph_names: List[str] = Field(default_factory=list)
    """
    Specifies list of graph names.
    """

    compilation_target: str = Field(default="hardware", serialization_alias="compiler_compilation_target")
    """
    Specifies the compilation target for the graph.
    """

    hardware_version: str = Field(default="2.0", serialization_alias="compiler_hardware_version")
    """
    Specifies the AI 100 hardware version to generate model binary.
    """

    num_of_cores: int = Field(default=1, serialization_alias="compiler_num_of_cores")
    """
    Specifies the number of NSP Cores to use.
    """

    do_host_preproc: bool = Field(default=True, serialization_alias="compiler_do_host_preproc")
    """
    Specifies whether to perform host-side preprocessing. Setting compiler_do_host_preproc to false
    disables all pre-/post-processing on the host. Otherwise, all all pre-/post-processing are
    performed on the host
    """

    stat_level: int = Field(default=10, serialization_alias="compiler_stat_level")
    """
    Specifies the level of stats to be collected when stats collection is enabled
    with one of the profiling options.
    """

    stats_batch_size: int = Field(default=2, serialization_alias="compiler_stats_batch_size")
    """
    Specifies the number of batches to be used for execution. Sets the batch dimension of input/output
    to the model for running inferences.
    """

    print_ddr_stats: bool = Field(default=True, serialization_alias="compiler_printDDRStats")
    """
    Specifies whether to collect per core DDR traffic details.
    """

    print_perf_metrics: bool = Field(default=True, serialization_alias="compiler_printPerfMetrics")
    """
    Specifies whether to print compiler performance metrics.
    """

    perf_warnings: bool = Field(default=True, serialization_alias="compiler_perfWarnings")
    """
    Specifies whether to print performance warning messages.
    """

    pmu_events: str = Field(default="1 2 0", serialization_alias="compiler_PMU_events")
    """
    Specifies the PMU events to track. Up to 8 events are supported.
    """

    pmu_recipe_opt: Literal["AxiRd", "AxiWr", "AxiRdWr", "KernelUtil"] = Field(
        default="KernelUtil", serialization_alias="compiler_PMU_recipe_opt"
    )
    """
    Specifies the PMU recipe optimization.
    """

    buffer_dealloc_delay: int = Field(default=2, serialization_alias="compiler_buffer_dealloc_delay")
    """
    Specifies the buffer deallocation delay.
    """

    gen_crc: bool = Field(default=True, serialization_alias="compiler_genCRC")
    """
    Specifies whether to enable CRC check for inputs and outputs of the network.
    """

    crc_stride: int = Field(default=256, serialization_alias="compiler_crc_stride")
    """
    Specifies the size of stride to calculate CRC in the stride section.
    """

    enable_depth_first: bool = Field(default=True, serialization_alias="compiler_enable_depth_first")
    """
    Specifies whether to enable depth-first compilation.
    """

    cluster_sizes: str = Field(default="1 2 0", serialization_alias="compiler_cluster_sizes")
    """
    Specifies the cluster configuration for single device partitioning.
    """

    max_out_channel_split: str = Field(default="1 2 0", serialization_alias="compiler_max_out_channel_split")
    """
    Specifies the effort level to reduce the on-chip memory usage.
    """

    overlap_split_factor: int = Field(default=2, serialization_alias="compiler_overlap_split_factor")
    """
    Specifies the factor to increase splitting of network operations.
    """

    vtcm_working_set_limit_ratio: float = Field(
        default=1.0, serialization_alias="compiler_VTCM_working_set_limit_ratio"
    )
    """
    Specifies the ratio of fast memory an instruction can use.
    """

    user_dma_producer_dma_enabled: bool = Field(
        default=True, serialization_alias="compiler_userDMAProducerDMAEnabled"
    )
    """
    Specifies whether to initiate NSP DMAs from the thread that produces data being transferred.
    """

    size_split_granularity: int = Field(default=1024, serialization_alias="compiler_size_split_granularity")
    """
    Specifies the maximum tile size.
    """

    do_ddr_to_multicast: bool = Field(default=True, serialization_alias="compiler_do_DDR_to_multicast")
    """
    Specifies whether to reduce DDR bandwidth by loading weights used on multiple-cores only once and multicasting to other cores.
    """

    enable_debug: bool = Field(default=True, serialization_alias="compiler_enableDebug")
    """
    Specifies whether to enable debug mode during model compilation.
    """

    combine_inputs: bool = Field(default=True, serialization_alias="compiler_combine_inputs")
    """
    Specifies whether to combine inputs into fewer buffers for transfer to device.
    """

    combine_outputs: bool = Field(default=True, serialization_alias="compiler_combine_outputs")
    """
    Specifies whether to combine outputs into a single buffer for transfer to host.
    """

    direct_api: bool = Field(default=True, serialization_alias="compiler_directApi")
    """
    Specifies whether to use a platform-specific shared memory API.
    """

    compile_threads: int = Field(default=2, serialization_alias="compiler_compileThreads")
    """
    Specifies the number of threads to use for compilation.
    """

    force_vtcm_spill: bool = Field(default=True, serialization_alias="compiler_force_VTCM_spill")
    """
    Specifies whether to force all VTCM buffers to be spilled to DDR.
    """

    convert_to_fp16: bool = Field(default=True, serialization_alias="compiler_convert_to_FP16")
    """
    Specifies whether to convert the graph from FP32 precision to FP16 precision inside AIC backend.
    """

    retained_state: bool = Field(default=False, serialization_alias="compiler_retained_state")
    """
    Specifies whether to enable compiler retained state.
    """

    mxfp6_matmul_weights: bool = Field(default=False, serialization_alias="compiler_mxfp6_matmul_weights")
    """
    Specifies whether to enable MXFP6 matmul weights.
    """

    exclude_mxfp6_matmul: str = Field(default="", serialization_alias="compiler_exclude_mxfp6_matmul")
    """
    Specifies the list of operations to exclude from MXFP6 matmul.
    """

    mxint8_mdp_io: bool = Field(default=False, serialization_alias="compiler_mxint8_mdp_io")
    """
    Specifies whether to enable MXINT8 MDP IO.
    """

    time_passes: bool = Field(default=True, serialization_alias="compiler_time_passes")
    """
    Specifies whether to enable the region profiler to collect compile-time statistics.
    """

    mdp_load_partition_config: str = Field(
        default="mdp_config.json", serialization_alias="compiler_mdp_load_partition_config"
    )
    """
    Path to load multi-device partition configuration file.
    """

    mdp_dump_partition_config: str = Field(
        default="mdp_pipeline_template.json", serialization_alias="compiler_mdp_dump_partition_config"
    )
    """
    Path to dump multi-device partition configuration template file.
    """


class AicRuntimeConfig(AISWBaseModel):
    """Runtime configuration for AIC Backend"""

    device_ids: List[int] = Field(default_factory=list, serialization_alias="runtime_device_ids")
    """
    Specifies the device ID(s) on which to execute the network.
    """

    num_activations: int = Field(default=1, serialization_alias="runtime_num_activations")
    """
    Specifies the number of concurrent activations (instances of a model) to inference.
    """

    submit_timeout: int = Field(default=0, serialization_alias="runtime_submit_timeout")
    """
    Specifies the length of time (seconds) to wait for submission to complete.
    """

    submit_num_retries: int = Field(default=5, serialization_alias="runtime_submit_num_retries")
    """
    Specifies the number of retries to carry out if submission times out.
    """

    threads_per_queue: int = Field(default=4, serialization_alias="runtime_threads_per_queue")
    """
    Specifies the number of host threads to use for pre- and post-processing.
    """


class AicConfigHelper:
    """
    Helper class to convert AIC config to backend extension config needed by QAIRT Tools.
    """

    _CONFIG_TYPES = {"compiler", "runtime"}

    @staticmethod
    def to_backend_extension_dict(
        compiler_config: AicCompilerConfig | None = None,
        runtime_config: AicRuntimeConfig | None = None,
    ) -> Dict[str, Any]:
        """
        Builds the backend extension dictionary based on the provided configurations.

        Args:
            compiler_config: Compiler configuration.
            runtime_config: Runtime configuration.

        Examples:
            >>> compiler_configs = AicCompilerConfig(graph_names=["graph1"])
            >>> AicConfigHelper.to_backend_extension_dict(compiler_configs)

        Returns:
            Dictionary representing the backend extension.
        """
        backend_extension_dict = {}
        if compiler_config is not None:
            backend_extension_dict.update(
                compiler_config.model_dump(by_alias=True, exclude_unset=True, exclude_none=True)
            )
        if runtime_config is not None:
            backend_extension_dict.update(
                runtime_config.model_dump(by_alias=True, exclude_unset=True, exclude_none=True)
            )
        return backend_extension_dict

    @classmethod
    def list_config_types(cls):
        # Lists the AIC config types - compiler, runtime
        return cls._CONFIG_TYPES

    @staticmethod
    def shared_library_path():
        return AicBackend().backend_extensions_library

    @staticmethod
    def list_config_options():
        # Lists every option for each Aic Config
        config_classes = [
            AicCompilerConfig,
            AicRuntimeConfig,
        ]

        for config_class in config_classes:
            fields = getattr(config_class, "model_fields", {})
            for attr, info in fields.items():
                if not attr.startswith("__"):
                    print(f"{config_class.__name__}.{attr}: {info.annotation}")
