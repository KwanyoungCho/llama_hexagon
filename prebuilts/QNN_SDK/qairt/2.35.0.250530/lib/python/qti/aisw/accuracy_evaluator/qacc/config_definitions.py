# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import inspect
import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import qti.aisw.accuracy_evaluator.qacc.plugin as pl
import qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module as context_bin_gen
import qti.aisw.tools.core.modules.converter.converter_module as converter
import qti.aisw.tools.core.modules.converter.quantizer_module as quantizer
import qti.aisw.tools.core.modules.net_runner.net_runner_module as net_runner
import yaml
from pydantic import ConfigDict, DirectoryPath, Field, FilePath, model_serializer, model_validator
from pydantic.json_schema import SkipJsonSchema
from qti.aisw.accuracy_evaluator.qacc import qacc_logger
from qti.aisw.accuracy_evaluator.qacc.config_parser import ConfigurationException, ParserHelper
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.tools.core.modules.api.definitions.common import AISWBaseModel, BackendType
from qti.aisw.tools.core.utilities.comparators.common import COMPARATORS
from qti.aisw.tools.core.utilities.data_processing.core.transformations import (
    AdapterConfig,
    DatasetConfig,
)


class CalibrationType(Enum):
    """Enum representing different calibration types."""
    INDEX = 'index'
    RAW = 'raw'
    DATASET = 'dataset'


class InferenceEngineType(Enum):
    """Enum representing different inference engine types."""
    QNN = 'qnn'
    AIC = 'aic'
    ONNXRT = 'onnxrt'
    TFRT = 'tensorflow'
    TORCHSCRIPTRT = 'torchscript'


class PrecisionType(Enum):
    """Enum representing different precision types."""
    FP32 = 'fp32'
    FP16 = 'fp16'
    QUANT = 'quant'


class TargetArchType(Enum):
    """Enum representing different target architecture types."""
    X86 = "x86_64-linux-clang"
    ANDROID = "aarch64-android"


# based on qairt-converter
class ConverterParams(converter.ConverterInputConfig):
    _model_framework: str = None
    input_network: FilePath = None

    # handle special case when quantization_overrides file is yaml
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        _conf = {}
        for name, val in self:
            if name in self.model_fields_set:
                if str(name) == 'quantization_overrides':
                    json_overrides_path = str(val).replace('_cleaned', '')
                    fname, _ = os.path.splitext(json_overrides_path)
                    yaml_overrides_path = fname + '.yaml'
                    quant_overrides_path = json_overrides_path
                    if os.path.exists(yaml_overrides_path):
                        quant_overrides_path = yaml_overrides_path
                    val = quant_overrides_path
                if str(name) == "preserve_io_datatype":
                    pval = val
                    if isinstance(pval, list):
                        val = ",".join(pval)
                    elif isinstance(pval, str):
                        val = True
                    else:
                        val = False
                _conf[str(name)] = val
        return _conf


class QuantizerParams(quantizer.QuantizerInputConfig):
    input_dlc: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    output_dlc: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    backend_info: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    bias_bitwidth: Literal[8, 32] | List[Literal[8, 32]] = 8
    act_bitwidth: Literal[8, 16] | List[Literal[8, 16]] = 8
    weights_bitwidth: Literal[8, 4] | List[Literal[8, 4]] = 8
    float_bitwidth: Literal[32, 16] | List[Literal[32, 16]] = 32
    float_bias_bitwidth: Literal[16, 32] | List[Literal[16, 32]] = None
    use_per_channel_quantization: bool | List[bool] = False
    use_per_row_quantization: bool | List[bool] = False
    act_quantizer_calibration: quantizer.quant_calibration | List[quantizer.quant_calibration] = "min-max"
    param_quantizer_calibration: quantizer.quant_calibration | List[quantizer.quant_calibration] = "min-max"
    act_quantizer_schema: quantizer.quant_schema | List[quantizer.quant_schema] = "asymmetric"
    param_quantizer_schema: quantizer.quant_schema | List[quantizer.quant_schema] = "asymmetric"
    percentile_calibration_value: float | List[float] = 99.99

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        _conf = {}
        for name, val in self:
            if name in self.model_fields_set:
                if str(name) == 'input_list':
                    continue
                if str(name) in qcc.PIPE_SUPPORTED_QUANTIZER_PARAMS:
                    """
                    Create a pipe delimited string only if the values are in a list.
                    Else directly add to the dict.
                    """
                    if isinstance(val, list):
                        val = " | ".join([str(x) for x in val]) if len(val) > 1 else val[0]
                elif str(name) == "algorithms":
                    if isinstance(val, list):
                        val = ",".join(val)
                elif str(name) == "preserve_io_datatype":
                    pval = val
                    if isinstance(pval, list):
                        val = ",".join(pval)
                    elif isinstance(pval, str):
                        val = True
                    else:
                        val = False
                _conf[str(name)] = val
        return _conf


class ContextBinParams(context_bin_gen.GenerateConfig):
    backend_extensions: Optional[FilePath] = None


class NetRunParams(net_runner.InferenceConfig):
    backend_extensions: Optional[FilePath] = None


class BackendExtensions(AISWBaseModel):

    def get_context_bin_config_dict(self):
        params = self.model_dump(exclude_none=True)
        context_bin_params = {}
        for key, val in params.items():
            if (key in self.SUPPORTED_CONTEXT_BACKEND_EXTENSION_PARAMS):
                context_bin_params[key] = val
        context_bin_params = self._get_config_dict(params=context_bin_params, is_context_bin=True)
        return context_bin_params

    def get_netrun_config_dict(self):
        params = self.model_dump(exclude_none=True)
        netrun_params = {}
        for key, val in params.items():
            if (key in self.SUPPORTED_NETRUN_BACKEND_EXTENSION_PARAMS):
                netrun_params[key] = val
        netrun_params = self._get_config_dict(params=netrun_params)
        return netrun_params


class HTPBackendExtensions(BackendExtensions):
    vtcm_mb: Optional[int] = None
    fp16_relaxed_precision: Optional[int] = None
    O: Optional[int] = None  # noqa: E741
    dlbc: Optional[int] = None
    hvx_threads: Optional[int] = None
    soc_id: Optional[int] = None
    soc_model: Optional[int] = None
    dsp_arch: Optional[str] = None
    pd_session: Optional[str] = None
    profiling_level: Optional[str] = None
    weight_sharing_enabled: Optional[bool] = None
    rpc_control_latency: Optional[int] = None
    device_id: Optional[int] = None
    use_client_context: Optional[bool] = None
    core_id: Optional[int] = None
    perf_profile: Optional[str] = None
    rpc_polling_time: Optional[int] = None
    hmx_timeout_us: Optional[int] = None
    max_spill_fill_buffer_for_group: Optional[int] = None
    group_id: Optional[int] = None
    file_read_memory_budget_in_mb: Optional[int] = None
    io_memory_estimation: Optional[bool] = None
    share_resources: Optional[bool] = None
    mem_type: Optional[str] = None
    SUPPORTED_CONTEXT_BACKEND_EXTENSION_PARAMS: List[str] = Field(
        default=[
            "vtcm_mb", "fp16_relaxed_precision", "O", "dlbc", "hvx_threads", "soc_id", "soc_model",
            "dsp_arch", "pd_session", "profiling_level", "weight_sharing_enabled"
        ], exclude=True)
    SUPPORTED_NETRUN_BACKEND_EXTENSION_PARAMS: List[str] = Field(
        default=[
            "fp16_relaxed_precision", "rpc_control_latency", "vtcm_mb", "O", "dlbc", "hvx_threads",
            "device_id", "soc_id", "soc_model", "dsp_arch", "pd_session", "profiling_level",
            "use_client_context", "core_id", "perf_profile", "rpc_polling_time", "hmx_timeout_us",
            "max_spill_fill_buffer_for_group", "group_id", "file_read_memory_budget_in_mb",
            "io_memory_estimation", "share_resources", "mem_type"
        ], exclude=True)

    def _get_config_dict(self, params: Dict = None, is_context_bin: bool = False):
        """Create config dictionary with context-binary and netrun backend extension params for HTP.

        The dictionary is expected to have the backend extensions format as defined in SDK documentation below:
        `Context Binary Generator`_ and `Netrun`_

        .. _Context Binary Generator: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tools.html#qnn-context-binary-generator
        .. _Netrun: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tools.html#qnn-net-run
        """
        expected_params = {
            "graphs": [
                "vtcm_mb",
                "fp16_relaxed_precision",
                "graph_names",
                "O",
                "dlbc",
                "hvx_threads",
            ],
            "devices": {
                "contextbin": [
                    "soc_id",
                    "soc_model",
                    "dsp_arch",
                    "pd_session",
                    "profiling_level",
                ],
                "netrun": [
                    "device_id",
                    "soc_id",
                    "soc_model",
                    "dsp_arch",
                    "pd_session",
                    "profiling_level",
                    "use_client_context",
                ],
            },
            "cores": {
                "contextbin": [],
                "netrun": [
                    "core_id",
                    "perf_profile",
                    "rpc_control_latency",
                    "rpc_polling_time",
                    "hmx_timeout_us",
                ],
            },
            "context": {
                "contextbin": ["weight_sharing_enabled"],
                "netrun": [
                    "max_spill_fill_buffer_for_group",
                    "group_id",
                    "file_read_memory_budget_in_mb",
                    "io_memory_estimation",
                ],
            },
            "groupContext": {
                "contextbin": [],
                "netrun": ["share_resources"],
            },
            "memory": {
                "contextbin": [],
                "netrun": ["mem_type"]
            }
        }
        new_params = {
            "graphs": [{}],
            "devices": [{
                "cores": [{}]
            }],
            "context": {},
            "groupContext": {},
            "memory": {}
        }
        params_type = "contextbin" if is_context_bin else "netrun"

        # Required param for both context-binary and netrun
        new_params["graphs"][0]["graph_names"] = ["model"]

        for key, value in params.items():
            if key in expected_params["graphs"]:
                new_params["graphs"][0][key] = value
            elif key in expected_params["cores"][params_type]:
                new_params["devices"][0]["cores"][0][key] = value
            elif key in expected_params["devices"][params_type]:
                new_params["devices"][0][key] = value
            elif key in expected_params["context"][params_type]:
                new_params["context"][key] = value
            elif key in expected_params["groupContext"][params_type]:
                new_params["groupContext"][key] = value
            elif key in expected_params["memory"][params_type]:
                new_params["memory"][key] = value
            else:
                raise ConfigurationException("Invalid {} parameter - {} for HTP backend".format(
                    params_type, key))
        return new_params


class HTPMCPBackendExtensions(BackendExtensions):
    fp16_relaxed_precision: Optional[int] = None
    O: Optional[int] = None  # noqa: E741
    device_id: Optional[int] = None
    num_cores: Optional[int] = None
    heap_size: Optional[int] = None
    elf_path: str = "lib/hexagon-v68/unsigned/libQnnHtpMcpV68.elf"
    mode: Optional[str] = None
    combined_io_dma_enabled: Optional[bool] = None
    profiling_level: Optional[str] = None
    timeout: Optional[int] = None
    retries: Optional[int] = None
    SUPPORTED_CONTEXT_BACKEND_EXTENSION_PARAMS: List[str] = Field(
        default=[
            "fp16_relaxed_precision", "O", "device_id", "num_cores", "heap_size", "elf_path",
            "mode", "combined_io_dma_enabled"
        ], exclude=True)
    SUPPORTED_NETRUN_BACKEND_EXTENSION_PARAMS: List[str] = Field(
        default=[
            "profiling_level", "device_id", "timeout", "retries", "mode", "combined_io_dma_enabled"
        ], exclude=True)

    def _get_config_dict(self, params: Dict = None, is_context_bin: bool = False):
        """Create config dictionary with context-binary and netrun backend extension
        params for HTP-MCP.

        The dictionary is expected to have the following format:
        {
            graphs : [
                {
                    graph_name : model,
                    fp16_relaxed_precision : 1,
                    profiling_level : basic,
                    num_cores : 1,
                    O : 0
                }
            ],
            device : {
                device_id : 0
            },
            context : {
                heap_size : 256,
                elf_path : network.elf,
                timeout : 5000,
                retries : 5,
                mode : auto,
                combined_io_dma_enabled : true
            }
        }
        """
        expected_params = {
            "graphs": {
                "contextbin": ["fp16_relaxed_precision", "num_cores", "O"],
                "netrun": ["profiling_level"],
            },
            "device": ["device_id"],
            "context": {
                "contextbin": [
                    "heap_size",
                    "elf_path",
                    "mode",
                    "combined_io_dma_enabled",
                ],
                "netrun": ["timeout", "retries", "mode", "combined_io_dma_enabled"],
            },
        }
        new_params = {"graphs": [{}], "device": {}, "context": {}}
        params_type = "netrun"
        if is_context_bin:
            params_type = "contextbin"
            new_params["graphs"][0]["graph_name"] = 'model'

        for key, value in params.items():
            if key in expected_params["graphs"][params_type]:
                new_params["graphs"][0][key] = value
            elif key in expected_params["device"]:
                new_params["device"][key] = value
            elif key in expected_params["context"][params_type]:
                if key == "elf_path":
                    if not os.path.isabs(value):
                        value = os.path.join(os.getenv("QNN_SDK_ROOT"), value)
                    assert os.path.exists(value), (
                        f"Invalid elf_path {value}, "
                        "should be either an absolute path or relative to QNN_SDK_ROOT")
                new_params["context"][key] = value
            else:
                raise ConfigurationException("Invalid {} parameter - {} for HTP MCP backend".format(
                    params_type, key))

        # remove empty params from config
        new_params = {key: val for key, val in new_params.items() if val}
        return new_params


class AICBackendExtensions(BackendExtensions):
    compiler_compilation_target: Optional[str] = None
    compiler_hardware_version: Optional[str] = None
    compiler_num_of_cores: Optional[int] = None
    compiler_do_host_preproc: Optional[bool] = None
    compiler_stat_level: Optional[int] = None
    compiler_stats_batch_size: Optional[int] = None
    compiler_printDDRStats: Optional[bool] = None
    compiler_printPerfMetrics: Optional[bool] = None
    compiler_perfWarnings: Optional[bool] = None
    compiler_PMU_events: Optional[str] = None
    compiler_PMU_recipe_opt: Optional[str] = None
    compiler_buffer_dealloc_delay: Optional[int] = None
    compiler_genCRC: Optional[bool] = None
    compiler_crc_stride: Optional[int] = None
    compiler_enable_depth_first: Optional[bool] = None
    compiler_cluster_sizes: Optional[str] = None
    compiler_max_out_channel_split: Optional[str] = None
    compiler_overlap_split_factor: Optional[int] = None
    compiler_compilationOutputDir: Optional[str] = None
    compiler_depth_first_mem: Optional[int] = None
    compiler_VTCM_working_set_limit_ratio: Optional[float] = None
    compiler_userDMAProducerDMAEnabled: Optional[bool] = None
    compiler_size_split_granularity: Optional[int] = None
    compiler_do_DDR_to_multicast: Optional[bool] = None
    compiler_enableDebug: Optional[bool] = None
    compiler_combine_inputs: Optional[bool] = None
    compiler_combine_outputs: Optional[bool] = None
    compiler_directApi: Optional[bool] = None
    compiler_compileThreads: Optional[int] = None
    compiler_force_VTCM_spill: Optional[bool] = None
    compiler_convert_to_FP16: Optional[bool] = None
    compiler_time_passes: Optional[bool] = None
    compiler_retained_state: Optional[bool] = None
    compiler_mdp_load_partition_config: Optional[str] = None
    compiler_mdp_dump_partition_config: Optional[str] = None
    compiler_mxfp6_matmul_weights: Optional[bool] = None
    runtime_device_ids: Optional[List[int]] = None
    runtime_num_activations: Optional[int] = None
    runtime_submit_timeout: Optional[int] = None
    runtime_submit_num_retries: Optional[int] = None
    runtime_threads_per_queue: Optional[int] = None
    SUPPORTED_CONTEXT_BACKEND_EXTENSION_PARAMS: List[str] = Field(
        default=[
            "compiler_compilation_target", "compiler_hardware_version", "compiler_num_of_cores",
            "compiler_do_host_preproc", "compiler_stat_level", "compiler_stats_batch_size",
            "compiler_printDDRStats", "compiler_printPerfMetrics", "compiler_perfWarnings",
            "compiler_PMU_events", "compiler_PMU_recipe_opt", "compiler_buffer_dealloc_delay",
            "compiler_genCRC", "compiler_crc_stride", "compiler_enable_depth_first",
            "compiler_cluster_sizes", "compiler_max_out_channel_split",
            "compiler_overlap_split_factor", "compiler_compilationOutputDir",
            "compiler_depth_first_mem", "compiler_VTCM_working_set_limit_ratio",
            "compiler_userDMAProducerDMAEnabled", "compiler_size_split_granularity",
            "compiler_do_DDR_to_multicast", "compiler_enableDebug", "compiler_combine_inputs",
            "compiler_combine_outputs", "compiler_directApi", "compiler_compileThreads",
            "compiler_force_VTCM_spill", "compiler_convert_to_FP16"
        ], exclude=True)
    SUPPORTED_NETRUN_BACKEND_EXTENSION_PARAMS: List[str] = Field(
        default=[
            "runtime_device_ids", "runtime_num_activations", "runtime_profiling_start_iter",
            "runtime_profiling_num_samples", "runtime_profiling_type", "runtime_profiling_out_dir",
            "runtime_submit_timeout", "runtime_num_retries", "runtime_set_size",
            "runtime_threads_per_queue"
        ], exclude=True)

    def _get_config_dict(self, params: Dict = None, is_context_bin: bool = False):
        """Create config dictionary with context-binary and netrun backend extension
        params for AIC.
        """
        if is_context_bin:
            params["graph_names"] = ["model"]
            # Explicit type cast required for compiler_num_of_cores and other integer
            if "compiler_num_of_cores" in params:
                params["compiler_num_of_cores"] = int(params["compiler_num_of_cores"])
        return params


# TODO: indexes to be removed, derived params _input_info, _output_info have to be avoided
class PluginConfiguration(AISWBaseModel):
    """Defines paramters expected in a plugin"""
    name: str = None
    params: dict = None
    indexes: str = None
    input_info: str = None
    output_info: str = None
    _input_info: dict = None
    _output_info: dict = None
    _indexes: List = None

    def model_post_init(self, __context):
        if self.indexes:
            self._indexes = self.indexes.split(',')

    def get_info_dict(self, info, type, _cls) -> Dict:
        """Type=mem|path|dir, dtype=float32, format=cv2."""
        info_dict = {}
        if info:
            info = info.split(',')
            for i in info:
                kv = i.strip().split('=')
                info_dict[kv[0]] = kv[1]
        else:
            # use default defined in Plugin class
            if type == 'in':
                info_dict = _cls.default_inp_info
            else:
                info_dict = _cls.default_out_info

        return info_dict

    def update_input_output_info(self) -> None:
        """Update input output info based on the type of plugin (dataset, metrics, processing)"""
        if self.name in pl.PluginManager.registered_plugins:
            _cls = pl.PluginManager.registered_plugins[self.name]
            if inspect.isclass(_cls) and issubclass(_cls, pl.qacc_plugin):
                self._input_info = self.get_info_dict(self.input_info, type='in', _cls=_cls)
                self._output_info = self.get_info_dict(self.output_info, type='out', _cls=_cls)
        elif self.name in pl.PluginManager.registered_metric_plugins:
            # metric plugins dont need input and output info.
            _cls = pl.PluginManager.registered_metric_plugins[self.name]
            if inspect.isclass(_cls) and issubclass(_cls, pl.qacc_metric):
                self._input_info = None
                self._output_info = None
        elif self.name in pl.PluginManager.registered_dataset_plugins:
            # dataset plugins don't need input and output info.
            _cls = pl.PluginManager.registered_dataset_plugins[self.name]
            self._input_info = None
            self._output_info = None
        else:
            raise ConfigurationException('Configured plugin {} is not registered'.format(self.name))


class InfoConfiguration(AISWBaseModel):
    """Defines parameters part of the info section"""
    desc: str = None
    batchsize: int = 1
    max_calibration: int = None
    memory_pipeline: Optional[bool] = False
    dump_stages: List[Literal[qcc.STAGE_INFER, qcc.STAGE_PREPROC, qcc.STAGE_POSTPROC]] = []
    max_parallel_evaluations: int = None
    max_parallel_compilation: int = None
    data_chunk_size: int = None


class GlobalsConfiguration(AISWBaseModel):
    globals_dict: Dict[str, Any] = None


class DatasetConfiguration(AISWBaseModel):
    """Defines parameters part of the dataset section"""
    name: Optional[str] = "Unnamed"
    path: DirectoryPath
    inputlist_file: FilePath
    annotation_file: Optional[FilePath | DirectoryPath] = None
    calibration_file: Optional[FilePath] = None
    calibration_type: Optional[CalibrationType] = None
    dataset_plugin_list: Optional[List[PluginConfiguration]] = None
    max_inputs: Optional[int] = None
    _inputlist_path: Optional[DirectoryPath] = None
    _calibration_path: Optional[DirectoryPath] = None

    def model_post_init(self, __context):
        self.update_max_inputs()

    def update_max_inputs(self) -> None:
        max_count = sum(1 for input in open(self.inputlist_file))
        self.max_inputs = max_count

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        _dconf = {}
        if self.name != 'Unnamed':
            _dconf['name'] = self.name
        _dconf['path'] = str(self.path)
        _path = str(self.path) + '/'
        _dconf['inputlist_file'] = str(self.inputlist_file).split(_path)[1]
        if self.annotation_file:
            _dconf['annotation_file'] = str(self.annotation_file).split(_path)[1]
        if self.calibration_file and self.calibration_type:
            _dconf['calibration'] = {
                'type': self.calibration_type.value,
                'file': str(self.calibration_file).split(_path)[1]
            }
        if self.dataset_plugin_list:
            plugins = []
            for pl in self.dataset_plugin_list:
                plugins.append({'plugin': pl})
            _dconf['transformations'] = plugins
        return _dconf


class PreprocessingConfiguration(AISWBaseModel):
    """Defines a list of pre processing plugins"""
    preprocessing_plugin_list: Optional[List[PluginConfiguration]] = None

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        _conf = {}
        if self.preprocessing_plugin_list:
            plugins = []
            for pl in self.preprocessing_plugin_list:
                plugins.append({'plugin': pl})
            _conf['transformations'] = plugins
        return _conf


class PostprocessingConfiguration(AISWBaseModel):
    """Defines parameters part of the post-processing section, including a list of post-processing plugins"""
    postprocessing_plugin_list: Optional[List[PluginConfiguration]] = None
    squash_results: bool = False

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        _conf = {}
        if self.postprocessing_plugin_list:
            plugins = []
            for pl in self.postprocessing_plugin_list:
                plugins.append({'plugin': pl})
            _conf['transformations'] = plugins
        if self.squash_results:
            _conf['squash_results'] = self.squash_results
        return _conf


class InferenceSchemaConfiguration(AISWBaseModel):
    """Defines parameters part of the inference schema"""
    name: InferenceEngineType
    precision: PrecisionType = PrecisionType.FP32
    target_arch: TargetArchType = TargetArchType.X86
    backend: BackendType = BackendType.CPU
    tag: str | List[str] = None
    converter_params: Optional[ConverterParams] = None
    quantizer_params: Optional[QuantizerParams] = None
    contextbin_params: Optional[ContextBinParams] = None
    netrun_params: Optional[NetRunParams] = None
    backend_extensions: Optional[Union[HTPBackendExtensions, HTPMCPBackendExtensions,
                                       AICBackendExtensions]] = None
    is_ref: Optional[bool] = False
    multithreaded: Optional[bool] = True
    precompiled_path: Optional[FilePath] = None
    convert_nchw: Optional[bool] = False
    # Below are set by evaluator infrastructure
    idx: int = None
    _inference_schema_name: str = None
    _model_path: str = None

    @model_validator(mode="after")
    def validate_input_args(self):
        if (self.target_arch == TargetArchType.X86 and self.backend
                == BackendType.GPU) or (self.target_arch == TargetArchType.ANDROID
                                        and self.backend in [BackendType.AIC, BackendType.HTP_MCP]):
            raise ValueError(
                f"Target arch {self.target_arch.value} not supported for backend {self.backend}")
        if self.backend_extensions:
            if self.backend in BackendType.HTP and not isinstance(self.backend_extensions,
                                                                  HTPBackendExtensions):
                raise ValueError(f"Invalid backend extensions for backend {self.backend}")
            if self.backend == BackendType.AIC and not isinstance(self.backend_extensions,
                                                                  AICBackendExtensions):
                raise ValueError(f"Invalid backend extensions for backend {self.backend}")
            if self.backend == BackendType.HTP_MCP and not isinstance(self.backend_extensions,
                                                                      HTPMCPBackendExtensions):
                raise ValueError(f"Invalid backend extensions for backend {self.backend}")
        if self.is_ref and self.converter_params is not None:
            is_permutational = False
            for key, val in self.converter_params.items():
                val = str(val)
                vals = [v.strip() for v in val.split('|')]
                if len(vals) > 1:
                    is_permutational = True
                    break
            if is_permutational:
                raise ValueError(
                    'inference_schema={}, is_ref is set to True for a configuration which generates multiple inference schemas.'
                    .format(self._name))
        if self.backend == BackendType.HTP:
            if self.backend_extensions and not self.backend_extensions.dsp_arch:
                raise ValueError(
                    'Param dsp_arch not provided for backend HTP in backend_extensions')
            elif not self.backend_extensions:
                if not ((self.contextbin_params and self.contextbin_params.backend_extensions) or
                        (self.netrun_params and self.netrun_params.backend_extensions)):
                    raise ValueError(
                        'Param dsp_arch needs to be provided in backend_extensions json '
                        'under either contextbin_params or netrun_params.')
        return self

    def get_inference_schema_name_with_params(self) -> Tuple[str, str]:
        """Returns an inference schema name based on its params."""
        inference_schema_name = f'schema{self.idx}_{self.name.value}_{self.backend.lower()}_{self.precision.value}'
        quantizer_options_name = ""
        if self.quantizer_params:
            for param, val in self.quantizer_params:
                if param in self.quantizer_params.model_fields_set and \
                        str(param) in qcc.PIPE_SUPPORTED_QUANTIZER_PARAMS:
                    if str(param) == "algorithms" and val == "default":
                        continue
                    # for all values that are not 'False'
                    if val:
                        quantizer_options_name += "_" + qcc.PIPE_SUPPORTED_QUANTIZER_PARAMS[str(param)]
                        # for non boolean values
                        if not isinstance(val, bool):
                            quantizer_options_name += str(val)
        return inference_schema_name, quantizer_options_name

    def get_inference_schema_name(self) -> str:
        return self._inference_schema_name

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        _conf = {}
        for name, val in self:
            if str(name) == 'idx':
                continue
            if name in self.model_fields_set:
                if str(name) == 'tag':
                    _conf[str(name)] = ','.join(val)
                elif str(name) == 'backend':
                    _conf[str(name)] = str(val.value).lower()
                else:
                    _conf[str(name)] = val
        return _conf


class InferenceEngineConfiguration(AISWBaseModel):
    """Defines parameters part of the inference engine section that has information about the model path,
    device ids, inference schemas, input and output info
    """
    model_config = ConfigDict(protected_namespaces=())
    model_path: str
    simplify_model: bool = True
    check_model: bool = True
    device_ids: Union[List[str], List[int]] = [0]
    onnx_define_symbol: str = ''
    clean_model: bool = True
    inference_schemas: List[InferenceSchemaConfiguration] = None
    inputs_info: Dict[str, List[Any]] = None
    outputs_info: Dict[str, List[Any]] = None
    input_names: List[str] = None
    _model_object: bool = False
    _cleaned_only_model_path: FilePath = None
    _max_calib: int = -1
    _is_calib_req: bool = False

    def set_inference_schema_names(self) -> None:
        """For each inference schema, set the schema name based on the schema id,
        precision and quantizer params
        """
        inference_schema_names = {}
        for inference_schema in self.inference_schemas:
            if inference_schema.idx not in inference_schema_names:
                inference_schema_names[inference_schema.idx] = []
            schema_name, quantizer_options_name = inference_schema.get_inference_schema_name_with_params(
            )
            if quantizer_options_name not in inference_schema_names[inference_schema.idx]:
                inference_schema_names[inference_schema.idx].append(quantizer_options_name)
            schema_id = inference_schema_names[inference_schema.idx].index(quantizer_options_name)
            schema_name += "_" + str(schema_id)
            inference_schema._inference_schema_name = schema_name

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        _inconf = {}
        for name, val in self:
            if name in self.model_fields_set:
                if str(name) == 'inference_schemas':
                    inf_schemas = []
                    for inf_schema in val:
                        inf_schemas.append({'inference_schema': inf_schema})
                    _inconf['inference_schemas'] = inf_schemas
                elif str(name) == 'inputs_info' or str(name) == 'outputs_info':
                    info_list = []
                    for io_name, info in val.items():
                        info_dict = {}
                        info_dict[io_name] = {'type': info[0], 'shape': info[1]}
                        info_list.append(info_dict)
                    _inconf[str(name)] = info_list
                elif str(name) == 'input_names':
                    continue
                else:
                    _inconf[str(name)] = val

        return _inconf


class VerifierConfiguration(AISWBaseModel):
    """Defines the parameters of the verifier section"""
    enabled: bool = True
    fetch_top: int = 1
    type: COMPARATORS = COMPARATORS.AVERAGE
    tol: float = 0.001
    _interpretation_strategy = "max"

    def model_post_init(self, __context):
        """Sets the interpretation strategy based on the comparison type.

        For certain comparitor types, the interpretation strategy is set to 'max',
        indicating that the most deviating sample is determined by the maximum value.
        For other types, it is set to 'min', indicating the minimum value is considered.
        """
        comparator_with_max_strategy = [
            COMPARATORS.AVERAGE,
            COMPARATORS.L1NORM,
            COMPARATORS.L2NORM,
            COMPARATORS.MSE,
            COMPARATORS.STANDARD_DEVIATION,
            COMPARATORS.KLD,
        ]

        if self.type in comparator_with_max_strategy:
            self._interpretation_strategy = "max"
        else:
            self._interpretation_strategy = "min"


class MetricsConfiguration(AISWBaseModel):
    """Defines a list of metrics plugins"""
    metrics_plugin_list: Optional[List[PluginConfiguration]] = None

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        _conf = {}
        if self.metrics_plugin_list:
            plugins = []
            for pl in self.metrics_plugin_list:
                plugins.append({'plugin': pl})
            _conf['transformations'] = plugins
        return _conf


class EvaluatorPipelineConfig(AISWBaseModel):
    config_path: FilePath = None
    set_global: Dict[str, str] = None
    info_config: InfoConfiguration = None
    globals_config: GlobalsConfiguration = None
    dataset_config: DatasetConfiguration | DatasetConfig = None
    preprocessing_config: PreprocessingConfiguration = None
    adapter_config: AdapterConfig = None
    postprocessing_config: PostprocessingConfiguration = None
    inference_config: InferenceEngineConfiguration = None
    verifier_config: VerifierConfiguration = None
    metrics_config: MetricsConfiguration = None
    use_memory_plugins: bool = False
    _default_verifier: bool = False

    def get_ref_inference_schema(self) -> InferenceSchemaConfiguration:
        ref_found = False
        ref_inference_schemas = []
        for inference_schema in self.inference_config.inference_schemas:
            if inference_schema.is_ref:
                ref_found = True
                ref_inference_schemas.append(inference_schema)

        if not ref_found:
            # Set the first schema as reference schema
            self.inference_config.inference_schemas[0].is_ref = True
            return self.inference_config.inference_schemas[0]
        else:
            # use the first schema from found reference schemas
            return ref_inference_schemas[0]

    def model_post_init(self, __context):
        if self.config_path:
            file_data = ParserHelper.read_yaml_and_replace_globals(self.config_path,
                                                                   self.set_global)
            config = yaml.safe_load(file_data)
            mconfig = config['model']
            if 'info' in mconfig:
                self.info_config = InfoConfiguration(**mconfig['info'])
            if 'globals' in mconfig:
                _config = ParserHelper.parse_globals_section(mconfig['globals'])
                self.globals_config = GlobalsConfiguration(**_config)
            if 'dataset' in mconfig:
                if self.use_memory_plugins or self.info_config.memory_pipeline:
                    self.dataset_config = DatasetConfig(**mconfig['dataset'])
                else:
                    _config = ParserHelper.parse_dataset_section(mconfig['dataset'])
                    self.dataset_config = DatasetConfiguration(**_config)
            if 'preprocessing' in mconfig:
                _config = ParserHelper.parse_preprocessing_section(mconfig['preprocessing'])
                self.preprocessing_config = PreprocessingConfiguration(**_config)
            if 'postprocessing' in mconfig:
                _config = ParserHelper.parse_postprocessing_section(mconfig['postprocessing'])
                self.postprocessing_config = PostprocessingConfiguration(**_config)
            if 'adapter' in mconfig:
                # Assume only one a output adapter is supported.
                if len(mconfig['adapter']) == 1 and 'plugin' in mconfig['adapter'][0]:
                    self.adapter_config = AdapterConfig(**mconfig['adapter'][0]['plugin'])
                else:
                    raise ValueError("Adapter section must contain only one plugin")
            if 'inference-engine' in mconfig:
                batchsize = 1
                if 'batchsize' in mconfig['info']:
                    batchsize = mconfig['info']['batchsize']
                if isinstance(self.dataset_config, DatasetConfiguration):
                    calib_file = self.dataset_config.calibration_file if self.dataset_config.calibration_file else self.dataset_config.inputlist_file
                else:

                    calib_file = self.dataset_config.params['calibration_path'] if 'calibration_path' in self.dataset_config.params \
                        and self.dataset_config.params['calibration_path'] else self.dataset_config.params['inputlist_path']
                _config = ParserHelper.parse_inference_engine_section(mconfig['inference-engine'],
                                                                      batchsize, calib_file)
                self.inference_config = InferenceEngineConfiguration(**_config)
            if 'verifier' in mconfig:
                _config = ParserHelper.parse_verifier_section(mconfig['verifier'])
                self.verifier_config = VerifierConfiguration(**_config)
            else:
                self.verifier_config = VerifierConfiguration(enabled=True, type=COMPARATORS.AVERAGE,
                                                             tol=0.001)
                self._default_verifier = True
            if 'metrics' in mconfig:
                _config = ParserHelper.parse_metrics_section(mconfig['metrics'])
                self.metrics_config = MetricsConfiguration(**_config)

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        _econf = {}
        if self.info_config:
            _econf['info'] = self.info_config
        if self.globals_config:
            _econf['globals'] = self.globals_config.globals_dict
        _econf['dataset'] = self.dataset_config
        if self.preprocessing_config:
            _econf['preprocessing'] = self.preprocessing_config
        if self.postprocessing_config:
            _econf['postprocessing'] = self.postprocessing_config
        _econf['inference-engine'] = self.inference_config
        if self.verifier_config and not self._default_verifier:
            _econf['verifier'] = self.verifier_config
        if self.metrics_config:
            _econf['metrics'] = self.metrics_config
        return {'model': _econf}


class CompilationParams(AISWBaseModel):
    """dataclass of parameters for model compilation
    Parameters:
        converter_params (ConverterParams): Conversion params object
        quantizer_params (QuantizerParams): Quantization params object
        backend (BackendType): Desired backend. Providing this option will generate a graph optimized for the given backend.
        context_backend_extensions_json (os.PathLike): Path to a JSON file of backend extensions to apply during offline preparation
                                   (only applicable for certain backends).
        context_backend_extensions_dict (dict): Dictionary of backend extensions to apply during offline preparation
        offline_prepare (bool): Enable offline preparation.
        soc_model (str): Desired SOC model.
    """
    converter_params: Optional[ConverterParams] = None
    quantizer_params: Optional[QuantizerParams] = None
    backend: Optional[BackendType] = None
    context_backend_extensions_json: Optional[os.PathLike] = None
    context_backend_extensions_dict: Optional[dict] = None
    offline_prepare: Optional[bool] = None
    soc_model: str = ''

    @model_validator(mode='after')
    def validate(self):
        """Validates the CompilationParams object"""
        if self.backend:
            # Automatically set offline prepare based on backend
            offline_prep_backends = BackendType.offline_preparable_backends()

            if self.offline_prepare is None:
                self.offline_prepare = self.backend in offline_prep_backends

            # Ensure that offline prepare is set only for applicable backends
            if self.offline_prepare and self.backend not in offline_prep_backends:
                qacc_logger.warning(f'Offline preparation is not supported for {self.backend} '
                                     'backend. Disabling offline prepare.')
                self.offline_prepare = False

            # Validate that offline prepare is always enabled for AIC backend
            if self.backend is BackendType.AIC and not self.offline_prepare:
                raise ValueError("Offline preparation is mandatory for AIC Backend.")

            # Validate backend with respect to quantization support
            quant_backends = BackendType.quantizable_backends()
            if self.quantizer_params and self.backend not in quant_backends:
                raise ValueError(
                    'Quantization is not supported for {} backend. Supported backends={}'.format(
                        self.backend, list(map(lambda x: x.value, quant_backends))))
        else:
            if self.offline_prepare:
                raise ValueError('Backend parameter must be set when enabling offline_prepare')

            if self.soc_model:
                raise ValueError('Backend parameter must be set along with soc_model')

        # Ensure that backend extensions are present only when offline prepare
        if self.context_backend_extensions_dict and not self.offline_prepare:
            qacc_logger.warning('Backend extensions cannot be applied without offline preparation.'
                                 ' Skipping given backend extensions')
            self.context_backend_extensions_dict = None
        return self
