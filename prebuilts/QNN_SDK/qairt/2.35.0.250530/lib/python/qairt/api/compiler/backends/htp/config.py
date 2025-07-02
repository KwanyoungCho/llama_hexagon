# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from qairt.api.configs.common import AISWBaseModel, DspArchitecture, PerfProfile
from qti.aisw.tools.core.modules.api.backend.htp_backend import HtpBackend

""" This module describes options that can be used to configure the HTP backend during
    compilation or execution only using this Python API. It is not intended as a reference QAIRT HTP Backend
    options. For information on QAIRT HTP Backend API, see QAIRT docs."""


class HtpGraphConfig(AISWBaseModel):
    """Graph configuration for HTP Backend"""

    name: str
    """
    Corresponds to the graph name provided to QnnGraph_create
    """

    vtcm_size_in_mb: int = Field(default=0, serialization_alias="vtcm_mb")
    """
    Provides performance infrastructure configuration options that are memory specific.
    To use a device's maximum VTCM amount, set the value to 0 (QNN_HTP_GRAPH_CONFIG_OPTION_MAX) and
    specify the target SoC through the device config.
    """

    vtcm_size: int = 0
    """
    0 means the MAX value of vtcm size by the soc_id or the current device
    """

    fp16_relaxed_precision: int = 0
    """
    Used to perform computation with half precision i.e. 16 bits
    """

    hvx_threads: int = 0
    """
    Corresponds to the number of HVX threads to use for a particular graph during an inference.
    """

    optimization_type: int = Field(default=2, ge=1, le=3, serialization_alias="O")
    """
    Set graph optimization value in range 1 to 3. As numbers increase, there is a tradeoff between preparation
    times to graph optimality.

    1 implies fastest preparation time, least optimal graph.
    3. implies slowest preparation time, most optimal graph.
    """

    finalize_config: Optional[dict] = None
    """
    Field to set the finalize config dict for backend extension
    """

    dlbc: Literal[0, 1] = 0
    """
    Provide deep learning bandwidth compression value 0 or 1.
    """

    dlbc_weights: int = 0
    """
    Number of weights to use for compression. Only used when dlbc is set to 1.
    """

    weights_packing: bool = False
    """
    Specifies whether to enable weights packing.
    """


class HtpDeviceCoreConfig(AISWBaseModel):
    """Core Configuration for HTP Backend"""

    id: int = Field(default=0, serialization_alias="core_id")
    """
    Unique identifier for the core configuration
    """

    perf_profile: PerfProfile = PerfProfile.HIGH_PERFORMANCE
    """
    Performance profile options. Use PerfProfile.list_options() for all
     available options
    """

    rpc_control_latency: int = 100
    """
    Rpc control latency value in micro second.
    """

    rpc_polling_time: int = 9999
    """
    Rpc polling time value in micro second.
    """

    hmx_timeout_us: int = 300000
    """
    Hmx timeout value in micro second.
    """

    adaptive_polling_time: int = 0
    """
    Adaptive polling time value in micro second.
    """


class HtpDeviceConfig(AISWBaseModel):
    """Device Configuration for HTP Backend"""

    id: int = Field(default=0, serialization_alias="device_id")
    """
    An optional id for the device configuration.
    """

    soc_model: int = 0
    """
    The Snapdragon SOC family associated with the device. E.x. 69. The SOC model must be related to the dsp
    architecture below.
    """

    dsp_arch: Optional[DspArchitecture | str] = None
    """
    The DSP architecture version for this SOC. See DspArchitecture.list_options() for all
    available options
    """

    pd_session: Literal["signed", "unsigned"] = "unsigned"
    """
    Specifies the user process domain attribute
    """

    profiling_level: Optional[Literal["linting"]] = "linting"
    """
    Used for linting profiling level.
    """

    cores: List[HtpDeviceCoreConfig] = Field(default_factory=list)
    """
    List of core configurations.
    """


class HtpContextConfig(AISWBaseModel):
    """Context Configuration for HTP Backend"""

    weight_sharing_enabled: bool = False
    """
    This feature allows common weights across graphs (max 64) to be shared and
    stored in a single context binary.
    """

    file_read_memory_budget_in_mb: int = 0
    """
    Allows users to configure the read memory budget of the deserialized binary in megabytes (Mb).
    It gives a hint to the backend to load the binary in chunks,
    instead of loading the entire binary to memory at once.
    """

    io_memory_estimation: bool = False
    """
    Enables I/O memory estimation when multiple PDs are available.
    It estimates the total size of the I/O tensors required by the context to ensure sufficient space
    on the PD before deserialization.
    """

    max_spill_fill_buffer_for_group: int = Field(default=0, ge=0)
    """
    Used to associate max spill-fill buffer size across multiple contexts within a group.
    Group_id value must be set to 0 for this option to be used.
    """

    group_id: int = 0
    """
    Specifies the group id to which contexts can be associated.
    """

    @model_validator(mode="after")
    def validate_group_id_max_spill_fill_buffer_for_group(self):
        """Validate group_id vs max_spill_fill_buffer_for_group"""
        if self.max_spill_fill_buffer_for_group > 0 and self.group_id != 0:
            raise ValueError("group_id must be set to 0 when max_spill_fill_buffer_for_group is set")
        return self


class HtpMemoryConfig(AISWBaseModel):
    """Memory Configuration for HTP Backend"""

    mem_type: Literal["shared_buffer"] = "shared_buffer"


class HtpGroupContextConfig(AISWBaseModel):
    """Group Context Configuration for HTP Backend"""

    share_resources: bool = False
    """
    Enables resource sharing across different contexts during binary generation.
    When enabled, it allows the backend to apply HTP virtual address space optimization.
    Note: This feature cannot be used with graph switching.
    """


class HtpConfigHelper:
    """
    Helper class to convert HTP config to backend extension config needed by QAIRT Tools.
    """

    _CONFIG_TYPES = {"context", "graphs", "devices", "memory", "groupContext"}

    @staticmethod
    def to_backend_extension_dict(
        context_configs: List[HtpContextConfig] | None = None,
        graph_configs: List[HtpGraphConfig] | None = None,
        device_configs: List[HtpDeviceConfig] | None = None,
        memory_config: HtpMemoryConfig | None = None,
        group_context: HtpGroupContextConfig | None = None,
    ) -> Dict[str, Any]:
        """
        Builds the backend extension dictionary based on the provided configurations.

        Args:
            context_configs: List of context configurations.
            graph_configs: List of graph configurations.
            device_configs: List of device configurations.
            memory_config: Memory configuration.
            group_context: Group context configuration.

        Examples:
            >>> context_configs = [HtpContextConfig(weight_sharing_enabled=True)]
            >>> graph_configs = [HtpGraphConfig(name="graph1")]
            >>> HtpConfigHelper.to_backend_extension_dict(context_configs, graph_configs)

        Returns:
            Dictionary representing the backend extension.
        """
        backend_extension_dict = {}
        if context_configs is not None:
            backend_extension_dict["context"] = [
                cfg.model_dump(mode="json", by_alias=True, exclude_unset=True, exclude_none=True)
                for cfg in context_configs
            ]
        if graph_configs is not None:
            backend_extension_dict["graphs"] = []
            for cfg in graph_configs:
                cfg_model_dump = cfg.model_dump(
                    mode="json", by_alias=True, exclude_unset=True, exclude_none=True
                )

                # TODO: Currently required to pass a list of graph names for a single graph config.
                # This is a temporary workaround until the backend extension is supported to support a single
                # name per graph config.
                # For now set to graph_names and delete name
                cfg_model_dump["graph_names"] = [cfg_model_dump["name"]]
                del cfg_model_dump["name"]

                backend_extension_dict["graphs"].append(cfg_model_dump)
        if device_configs is not None:
            backend_extension_dict["devices"] = [
                device_cfg.model_dump(mode="json", by_alias=True, exclude_unset=True, exclude_none=True)
                for device_cfg in device_configs
            ]
        if memory_config is not None:
            # Allow unset to be passed to allow for default values to be used
            backend_extension_dict["memory"] = memory_config.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        if group_context is not None:
            backend_extension_dict["groupContext"] = group_context.model_dump(
                mode="json", by_alias=True, exclude_unset=True, exclude_none=True
            )
        return backend_extension_dict

    @classmethod
    def list_config_options(cls):
        return cls._CONFIG_TYPES

    @staticmethod
    def shared_library_path():
        return HtpBackend().backend_extensions_library

    @staticmethod
    def list_options():
        # List every option for each Htp Config
        config_classes = [
            HtpGraphConfig,
            HtpDeviceConfig,
            HtpContextConfig,
            HtpDeviceCoreConfig,
            HtpMemoryConfig,
            HtpGroupContextConfig,
        ]

        for config_class in config_classes:
            fields = getattr(config_class, "model_fields", {})
            for attr, info in fields.items():
                if not attr.startswith("__"):
                    print(f"{config_class.__name__}.{attr}: {info.annotation}")
