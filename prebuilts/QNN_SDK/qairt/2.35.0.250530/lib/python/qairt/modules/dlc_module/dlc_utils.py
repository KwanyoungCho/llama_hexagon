# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from qairt.api.configs.common import AISWBaseModel
from qairt.modules.common.graph_info_models import EncodingsInfo, GraphInfo, TensorInfo
from qairt.modules.properties import AttrNoCopy, GraphMixin
from qairt.modules.qti_import import qairt_tools_cpp


def _convert_to_encodings_info(encodings_info: "qairt_tools_cpp.IrQuantizationEncodingInfo") -> EncodingsInfo:
    """
    Converts an instance of IrQuantizationEncodingInfo to an EncodingsInfo object.

    Args:
        encodings_info (IrQuantizationEncodingInfo): The source object containing quantization
                                                     encoding information.

    Returns:
        EncodingsInfo: An object containing the quantization encoding information in a
                       structured format.
    """

    return EncodingsInfo(
        bitwidth=encodings_info.bw,
        max=encodings_info.max,
        min=encodings_info.min,
        scale=encodings_info.scale,
        offset=encodings_info.offset,
        is_symmetric=encodings_info.is_symmetric,
        is_fixed_point=encodings_info.is_fixed_point,
    )


def _get_dlc_updater(dlc_path: str, output_dlc_path: str, **kwargs) -> "qairt_tools_cpp.DlcUpdater":
    """
    Obtain an updater object. This object can be used to modify the DLC.

    Args:
        dlc_path (str): Path where dlc is located
        output_dlc_path (str): Path where to save updated DLC to

    Returns:
        Instance of a dlc updater.

    """
    converter_command = kwargs.get("converter_command", "")
    quantizer_command = kwargs.get("quantizer_command", "")
    disable_lazy_weight_loading = not kwargs.get("enable_lazy_weight_loading", True)
    dlc_updater = qairt_tools_cpp.DlcUpdater(
        inputDlcPath=dlc_path,
        outputDlcPath=output_dlc_path,
        isStripIrData=False,
        disableLazyWeightLoading=disable_lazy_weight_loading,
        copyrightString="",
        modelCustomVersionString="",
        converterCommandlineString=converter_command,
        quantizerCommandlineString=quantizer_command,
    )

    return dlc_updater


class GraphDescriptor(GraphMixin):
    """
    Descriptor class for an IR Graph within a DLC.
    """

    def __init__(self, graph: Optional[Any] = None, disable_copy=False):
        if not disable_copy:
            self._graph = graph
        else:
            self._graph = AttrNoCopy(graph)

    def __get__(self, instance, owner):
        return self._graph

    def __set__(self, instance, value):
        self._graph = value

    def __delete__(self, instance):
        del self._graph

    @property
    def info(self) -> GraphInfo:
        """
        Returns the GraphInfo for this graph.
        """
        if self._graph is None:
            raise ValueError("Graph is not initialized")

        input_tensors = [
            TensorInfo(
                name=input.name(),
                dimensions=input.dims(),
                data_type=str(input.data_type()),
            )
            for input in self._graph.get_input_tensors_to_graph()
        ]
        output_tensors = [
            TensorInfo(
                name=output.name(),
                dimensions=output.dims(),
                data_type=str(output.data_type()),
                is_quantized=output.is_quantized(),
                encodings=_convert_to_encodings_info(output.get_encoding().encInfo),
            )
            for output in self._graph.get_output_tensors_of_graph()
        ]

        return GraphInfo(name=self._graph.name, inputs=input_tensors, outputs=output_tensors)


class DlcInfo(AISWBaseModel):
    """
    Describes properties of a DLC
    """

    copyright: str = ""
    model_version: str = ""
    graphs: List[GraphInfo]
    """A list of GraphInfo objects representing each graph in the DLC."""

    model_config = ConfigDict(protected_namespaces=())

    def as_dict(self) -> Dict[str, Any]:
        return self.model_dump()
