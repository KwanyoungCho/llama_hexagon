# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from enum import Enum
from typing import List, Tuple

import qairt
from qairt import CompileConfig
from qairt.api.compiled_model import CompiledModel
from qairt.api.compiler.backends.htp.config import HtpGraphConfig
from qairt.api.compiler.backends.htp_mcp.config import HtpMcpGraphConfig
from qairt.api.configs.common import BackendType
from qairt.api.model import Model
from qairt.api.profiler import Profiler
from qairt.api.profiler.report import OpTraceReport
from qairt.utils.exceptions import CompilationError, ExecutionError
from qairt.utils.loggers import get_logger

_tuner_logger = get_logger("qairt.compile.tuner")

# Constants
TUNING_RANGE = 18  # 0-17 ppts
MULTI_CORE_ONLY = [6, 7, 9, 10, 11, 12, 14]
SINGLE_CORE_ONLY = [2, 4]


class Criteria(str, Enum):
    """
    Enum representing criteria types.
    """

    BANDWIDTH = "bandwidth"
    LATENCY = "latency"

    @staticmethod
    def is_valid_criteria(criteria_str: str) -> bool:
        return any(criteria_str == value for value in {Criteria.BANDWIDTH.value, Criteria.LATENCY.value})


def get_graph_names(model: CompiledModel) -> List[str]:
    """Get graph names to update graph_custom_config.

    Args:
        model (CompiledModel): Compiled model.

    Returns:
        list: List of graph names.
    """
    try:
        graphs = model.module.info.as_dict().get("graphs", {})
    except Exception as e:
        _tuner_logger.debug("Unable to extract graph names. Tuning unsuccessful.")
        raise CompilationError("Cannot extract graph names from CompiledModel info")

    graph_names = []

    for graph in graphs:
        graph_names.append(graph["name"])
    return graph_names


def validate_backend_type(value: str):
    """Validate backend type.

    Args:
        value: Backend type value.

    Raises:
        ValueError: If the backend type is not supported.
    """
    if value not in [BackendType.HTP, BackendType.HTP_MCP]:
        raise ValueError("Tuning is supported for htp and htp_mcp backends only.")


def update_graph_configs(config: CompileConfig, p_value: int, graph_names: List[str]) -> CompileConfig:
    """Set optimization_type and add finalize_config to graph_custom_configs.

    Args:
        config (CompileConfig): Compilation configuration.
        p_value (int): P value for optimization.
        graph_names (list): List of graph names.

    Returns:
        CompileConfig: Updated compilation configuration.
    """
    if config.graph_custom_configs is not None:
        for graph_custom_config in config.graph_custom_configs:
            if graph_custom_config.optimization_type != 3:
                _tuner_logger.warning("Optimization_type value will be overridden for successful tuning.")
            graph_custom_config.optimization_type = 3
            graph_custom_config.finalize_config = {"P": p_value}

    else:
        graph_custom_configs = []
        for graph_name in graph_names:
            if config.backend == BackendType.HTP:
                graph_custom_configs.append(
                    HtpGraphConfig(
                        name=graph_name,
                        optimization_type=3,
                        finalize_config={"P": p_value},
                    )
                )
            elif config.backend == BackendType.HTP_MCP:
                graph_custom_configs.append(
                    HtpMcpGraphConfig(
                        name=graph_name,
                        optimization_type=3,
                        finalize_config={"P": p_value},
                    )
                )
        config.graph_custom_configs = graph_custom_configs
    return config


def get_report_details(profiler: Profiler, criteria: Criteria) -> Tuple[int, OpTraceReport]:
    """Generate report and extract criteria value.

    Args:
        profiler (Profiler): Profiler object.
        criteria (Criteria): Criteria for optimization.

    Returns:
        tuple: Criteria value and report data.
    """
    report = profiler.generate_report()
    report_data = report.summary.data
    try:
        if criteria == "bandwidth":
            criteria_value = report_data["data"]["htp_overall_summary"]["data"][0].get("total_dram")
        else:
            criteria_value = report_data["data"]["htp_overall_summary"]["data"][0].get("time_us")
    except Exception as e:
        _tuner_logger.debug("Unable to extract criteria value. Tuning unsuccessful.")
        raise ExecutionError(f"Unable to extract criteria: {criteria.value} from report.")

    return criteria_value, report


def optimize(
    model: Model, criteria: str, compile_args: dict, execution_args: dict
) -> Tuple[CompiledModel, OpTraceReport]:
    """Optimize the given model based on the specified criteria.

    Args:
        model (Model): The model to be optimized.
        criteria (str): The optimization criteria (options: latency, bandwidth).
        compile_args (dict): A dictionary of arguments for the compilation process.
                             See 'qairt.compile' for the full list of compile_args.
        execution_args (dict): A dictionary of arguments for the execution process.
                               See 'qairt.api.CompiledModel.call' for the full list
                               of execution_args. Add "device" for tuning.

    Returns:
        tuple: The optimized model and corresponding op trace report.
    """
    # Validate criteria
    if not Criteria.is_valid_criteria(criteria):
        raise ValueError("Criteria must be either 'bandwidth' or 'latency'.")

    criteria = Criteria(criteria)

    # Verify that device is in execution_args
    if "device" not in execution_args:
        raise TypeError("Add device in execution_args.")

    with Profiler(context={"level": "detailed", "option": "optrace"}) as profiler:
        if "config" in compile_args.keys():
            config = compile_args["config"]
            validate_backend_type(config.backend)
        elif "backend" in compile_args.keys():
            validate_backend_type(compile_args["backend"])
            config = CompileConfig(backend=compile_args["backend"])
        else:
            raise ValueError("Either backend or config must be specified")

        if config.enable_intermediate_outputs:
            _tuner_logger.info(
                f" Intermediate output generation set in config. It will be overridden for tuning"
            )
            config.enable_intermediate_outputs = False

        _tuner_logger.info(f"Tuning started for criteria: {criteria.value}.")
        # Compile and execute model with default configuration
        compiled_model_default = qairt.compile(model, config=config)
        compiled_model_default(**execution_args)

        # Get criteria value for default execution
        criteria_default, report_default = get_report_details(profiler, criteria)
        compiled_model_best = compiled_model_default
        criteria_best = criteria_default
        report_best = report_default

        # Extract graph names to create HtpGraphConfig/HtpMcpGraphConfig
        graph_names = get_graph_names(compiled_model_default)
        for p_value in range(1, TUNING_RANGE):
            # TODO: Get core info about device and use ppts accordingly
            if p_value in MULTI_CORE_ONLY:
                continue
            # Compile and execute model for each p_value
            config = update_graph_configs(config, p_value, graph_names)
            compiled_model_current = qairt.compile(model, config=config)
            compiled_model_current(**execution_args)
            criteria_current, report_current = get_report_details(profiler, criteria)

            if criteria_current < criteria_best:
                criteria_best = criteria_current
                report_best = report_current
                compiled_model_best = compiled_model_current

            improvement_iter = round(((criteria_default - criteria_current) / criteria_default) * 100, 2)
            if improvement_iter > 0:
                _tuner_logger.info(
                    f"Improvement in {criteria.value} for current iteration is {improvement_iter}%"
                )
        improvement_per = round(((criteria_default - criteria_best) / criteria_default) * 100, 2)
        if improvement_per == 0:
            _tuner_logger.info(
                f"No improvement obtained via tuning for {criteria.value}, returning original model with default configuration"
            )
        else:
            _tuner_logger.info(f"Improvement in criteria ({criteria.value}) observed: {improvement_per}%")

    return compiled_model_best, report_best
