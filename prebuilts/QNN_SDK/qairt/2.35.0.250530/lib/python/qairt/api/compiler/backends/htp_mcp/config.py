# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from qairt.api.configs.common import AISWBaseModel, DspArchitecture
from qti.aisw.tools.core.modules.api.backend.htp_mcp_backend import HtpMcpBackend

""" This module describes options that can be used to configure the HTP MCP backend during
    compilation or execution only using this Python API. It is not intended as a reference QAIRT HTP MCP
    Backend options. For information on QAIRT HTP MCP Backend API, see QAIRT docs."""


class HtpMcpCrcConfig(AISWBaseModel):
    """CRC Configuration for the HTP MCP Backend"""

    enable: bool = False
    """
    Specifies whether CRC is enabled.
    """
    start_block_size: int = 0
    """
    Specifies the starting block size for CRC.
    """
    end_block_size: int = 0
    """
    Specifies the ending block size for CRC.
    """
    stride_interval: int = 0
    """
    Specifies the interval between CRC strides.
    """
    stride_size: int = 0
    """
    Specifies the size of each CRC stride.
    """


class HtpMcpGraphConfig(AISWBaseModel):
    """Graph configuration for the HTP MCP Backend"""

    name: str
    """
    Corresponds to the graph name provided to QnnGraph_create
    """

    fp16_relaxed_precision: int = 0
    """
    Used to perform computation with half precision i.e. 16 bits
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

    num_cores: int = 0
    """
    Number of cores
    """

    profiling_level: Optional[Literal["linting"]] = None
    """
    Profiling level
    """


class HtpMcpDeviceConfig(AISWBaseModel):
    """Device configuration for the HTP MCP Backend"""

    id: int = Field(default=0, serialization_alias="device_id")
    """
    The serial id associated with the device.
    """

    num_cores: int = 0
    """
    List of core configurations.
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


class HtpMcpContextConfig(AISWBaseModel):
    """Memory configuration for the HTP MCP Backend"""

    mode: Optional[Literal["manual"]] = None
    """
    Specifies the mode of operation.
    """
    heap_size: int = 0
    """
    Specifies the size of the heap.
    """
    elf_path: str = ""
    """
    Specifies the path to the ELF file.
    """
    timeout: int = 0
    """
    Specifies the timeout value.
    """
    retries: int = 0
    """
    Specifies the number of retries allowed.
    """
    combined_io_dma_enabled: bool = True
    """
    Specifies whether combined IO DMA is enabled.
    """
    crc_config: Optional[HtpMcpCrcConfig] = None
    """
    Specifies the CRC configuration.
    """


class HtpMcpConfigHelper:
    """
    Helper class to convert HTP config to backend extension config needed by QAIRT Tools.
    """

    _CONFIG_TYPES = {"context", "graphs", "devices"}

    @staticmethod
    def to_backend_extension_dict(
        context_configs: List[HtpMcpContextConfig] | None = None,
        graph_configs: List[HtpMcpGraphConfig] | None = None,
        device_configs: List[HtpMcpDeviceConfig] | None = None,
    ) -> Dict[str, Any]:
        """
        Builds the backend extension dictionary based on the provided configurations.

        Args:
            context_configs: List of context configurations.
            graph_configs: List of graph configurations.
            device_configs: List of device configurations.

        Examples:
            >>> context_configs = [HtpMcpContextConfig(mode="manual")]
            >>> graph_configs = [HtpMcpGraphConfig(name="graph1")]
            >>> HtpMcpConfigHelper.to_backend_extension_dict(context_configs, graph_configs)

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
                cfg_model_dump["graph_names"] = [cfg_model_dump["name"]]
                del cfg_model_dump["name"]
                backend_extension_dict["graphs"].append(cfg_model_dump)
        if device_configs is not None:
            backend_extension_dict["devices"] = [
                device_cfg.model_dump(mode="json", by_alias=True, exclude_unset=True, exclude_none=True)
                for device_cfg in device_configs
            ]
        return backend_extension_dict

    @classmethod
    def list_config_options(cls):
        return cls._CONFIG_TYPES

    @staticmethod
    def shared_library_path():
        return HtpMcpBackend().backend_extensions_library

    @staticmethod
    def list_options():
        # List every option for each HtpMcp Config
        config_classes = [HtpMcpGraphConfig, HtpMcpDeviceConfig, HtpMcpContextConfig, HtpMcpCrcConfig]

        for config_class in config_classes:
            fields = getattr(config_class, "model_fields", {})
            for attr, info in fields.items():
                if not attr.startswith("__"):
                    print(f"{config_class.__name__}.{attr}: {info.annotation}")
