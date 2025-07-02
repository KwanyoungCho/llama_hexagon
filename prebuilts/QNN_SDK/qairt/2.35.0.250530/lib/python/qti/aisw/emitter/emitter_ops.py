# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Custom PyTorch Modules for QNN ops """

import copy
import torch.nn
import logging
logger = logging.getLogger('TorchEmitter')
from qti.aisw.emitter.op_definition import *

try:
    from aimet_torch.v2.nn.true_quant import QuantizationMixin

    def _map_quantizer_to_tensors(tensors, quantizer):
        maybe_quantize = lambda t: quantizer(t) if quantizer and t.is_floating_point() else t
        return torch.utils._pytree.tree_map(maybe_quantize, tensors) # pylint: disable=protected-access


    def _unary_forward(self: QuantizationMixin, *args, **kwargs) -> torch.Tensor: # pylint: disable=missing-function-docstring
        x, *others = args

        if isinstance(x, torch.Tensor) and x.is_floating_point() and self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters(): # pylint: disable=protected-access
            output = super(type(self), self).forward(x, *others, **kwargs)

        if isinstance(output, torch.Tensor) and output.is_floating_point() and self.output_quantizers[0]:
            output = self.output_quantizers[0](output)

        return output

    # pylint: disable=too-many-ancestors
    @QuantizationMixin.implements(CustomLayerNorm)
    class QuantizedCustomLayerNorm(QuantizationMixin, CustomLayerNorm):
        """ Quantized class definition for CustomLayerNorm """
        forward = _unary_forward

    FakeQuantizedCustomLayerNorm = QuantizedCustomLayerNorm


    @QuantizationMixin.implements(IndexSelect)
    class QuantizedIndexSelect(QuantizationMixin, IndexSelect):
        """ Quantized class definition for IndexSelect """
        forward = _unary_forward

    FakeQuantizedIndexSelect = QuantizedIndexSelect


    @QuantizationMixin.implements(BatchToSpace)
    class QuantizedBatchToSpace(QuantizationMixin, BatchToSpace):
        """ Quantized class definition for BatchToSpace """
        forward = _unary_forward

    FakeQuantizedBatchToSpace = QuantizedBatchToSpace


    @QuantizationMixin.implements(SpaceToBatch)
    class QuantizedSpaceToBatch(QuantizationMixin, SpaceToBatch):
        """ Quantized class definition for SpaceToBatch """
        forward = _unary_forward

    FakeQuantizedSpaceToBatch = QuantizedSpaceToBatch


    @QuantizationMixin.implements(CropAndResize)
    class QuantizedCropAndResize(QuantizationMixin, CropAndResize):
        """ Quantized class definition for CropAndResize """
        forward = _unary_forward

    FakeQuantizedCropAndResize = QuantizedCropAndResize


    @QuantizationMixin.implements(UnBind)
    class QuantizedUnBind(QuantizationMixin, UnBind):
        """ Quantized class definition for UnBind """
        _num_outputs: int

        def __quant_init__(self):
            super().__quant_init__()
            self._num_outputs = None

        # pylint: disable=arguments-differ
        # pylint: disable=too-many-function-args
        def export_output_encodings(self, encoding_version: str):
            output_encodings = super().export_output_encodings(encoding_version)
            if self._num_outputs is None:
                raise RuntimeError("Cannot infer number of output tensors without first executing `self.forward`.")
            # Create separate encoding objects to avoid overriding of attributes added/updated later while exporting encodings
            return [copy.deepcopy(encoding) for encoding in output_encodings * self._num_outputs]

        def forward(self, x) -> Tuple[torch.Tensor]: # pylint: disable=arguments-differ
            """ Quantized forward definition for UnBind """

            x = _map_quantizer_to_tensors(x, self.input_quantizers[0])

            outputs = super().forward(x)

            self._num_outputs = len(outputs)

            return _map_quantizer_to_tensors(outputs, self.output_quantizers[0])

    FakeQuantizedUnBind = QuantizedUnBind


    @QuantizationMixin.implements(NonZero)
    class QuantizedNonZero(QuantizationMixin, NonZero):
        """ Quantized class definition for NonZero """

        def __quant_init__(self):
            super().__quant_init__()
            self.output_quantizers = torch.nn.ModuleList([])

        def forward(self, tensor: torch.Tensor) -> torch.Tensor: # pylint: disable=arguments-differ
            """ Quantized forward definition for NonZero """
            tensor = _map_quantizer_to_tensors(tensor, self.input_quantizers[0])

            return super().forward(tensor)

    FakeQuantizedNonZero = QuantizedNonZero


    @QuantizationMixin.implements(Moments)
    class QuantizedMoments(QuantizationMixin, Moments):
        """ Quantized class definition for Moments """

        def __quant_init__(self):
            super().__quant_init__()
            self.output_quantizers = torch.nn.ModuleList([None, None])

        def forward(self, inputs) -> Tuple: # pylint: disable=arguments-differ
            """ Quantized forward definition for Moments """
            inputs = _map_quantizer_to_tensors(inputs, self.input_quantizers[0])

            mean, var = super().forward(inputs)

            if self.output_quantizers[0]:
                mean = self.output_quantizers[0](mean)

            if self.output_quantizers[1]:
                var = self.output_quantizers[1](var)

            return mean, var

    FakeQuantizedMoments = QuantizedMoments


    @QuantizationMixin.implements(Stack)
    class QuantizedStack(QuantizationMixin, Stack): # pylint: disable=too-many-ancestors
        """
        Quantized class definition for Stack.
        """
        _num_inputs: int

        def __quant_init__(self):
            super().__quant_init__()
            self._num_inputs = 1

        # pylint: disable=arguments-differ
        # pylint: disable=too-many-function-args
        def export_input_encodings(self, encoding_version: str):
            input_encodings = super().export_input_encodings(encoding_version)
            # Create separate encoding objects to avoid overriding of attributes added/updated later while exporting encodings
            return [copy.deepcopy(encoding) for encoding in input_encodings * self._num_inputs]

        def forward(self, *inputs) -> torch.Tensor: # pylint: disable=arguments-differ
            """
            Quantized forward impl for Stack.
            """
            self._num_inputs = len(inputs)

            inputs = _map_quantizer_to_tensors(inputs, self.input_quantizers[0])

            output = super().forward(*inputs)

            if output.is_floating_point() and self.output_quantizers[0]:
                output = self.output_quantizers[0](output)

            return output

    FakeQuantizedStack = QuantizedStack


    @QuantizationMixin.implements(MultiClassNms)
    class QuantizedMultiClassNms(QuantizationMixin, MultiClassNms):
        """
        Quantized class definition for MultiClassNms.
        """
        def __quant_init__(self):
            super().__quant_init__()
            # Apply one input quantizer to all *batched_features
            self.input_quantizers = torch.nn.ModuleList([None])
            # Apply output_quantizer[0] to scores, output_quantizer[1] to all output_features
            self.output_quantizers = torch.nn.ModuleList([None, None])

        def forward(self, *inp): # pylint: disable=arguments-differ
            """
            Quantized forward impl for Stack.
            """
            batched_features = ()
            if len(inp) > 2:
                batched_features = inp[2:]

            batched_features = _map_quantizer_to_tensors(batched_features, self.input_quantizers[0])

            output_boxes, output_scores, output_classes, *output_features = super().forward(*inp[0:2], *batched_features)

            if self.output_quantizers[0]:
                output_scores = self.output_quantizers[0](output_scores)

            output_features = _map_quantizer_to_tensors(output_features, self.output_quantizers[1])

            return output_boxes, output_scores, output_classes, *output_features

    FakeQuantizedMultiClassNms = QuantizedMultiClassNms

    @QuantizationMixin.implements(CustomPReLU)
    class QuantizedCustomPReLU(QuantizationMixin, CustomPReLU):
        """ Quantized class definition for CustomPReLU """
        forward = _unary_forward

except ModuleNotFoundError:
    logger.warning('Model Loaded without quantization node definition!')
