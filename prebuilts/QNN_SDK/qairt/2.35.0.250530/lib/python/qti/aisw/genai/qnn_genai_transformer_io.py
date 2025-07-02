# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import os
import re

import math
import mmap
import ctypes
import struct
import itertools

from qti.aisw.genai.qnn_genai_transformer_utils import *

#
# Read Model
#
def merge_sharded(models: list[LazyModel]) -> LazyModel:
    # Original LLaMA models have each file contain one part of each tensor.
    names = {name: None for model in models for name in model}

    def convert(name: str) -> LazyTensor:
        lazy_tensors: list[LazyTensor] = [model[name] for model in models]
        if len(lazy_tensors) == 1:
            # Only one file; don't go through this procedure
            return lazy_tensors[0]
        if len(lazy_tensors[0].shape) == 1:
            # The tensor (1D) is just duplicated in every file
            return lazy_tensors[0]
        if name.startswith('tok_embeddings.') or \
                name.endswith('.attention.wo.weight') or \
                name.endswith('.feed_forward.w2.weight'):
            # split by columns
            axis = 1
        else:
            # split by rows
            axis = 0
        concatenated_shape = list(lazy_tensors[0].shape)
        concatenated_shape[axis] = sum(tensor.shape[axis] for tensor in lazy_tensors)

        def load() -> UnquantizedTensor:
            ndarrays = [load_unquantized(tensor) for tensor in lazy_tensors]
            concatenated: NDArray = np.concatenate(ndarrays, axis=axis)
            return UnquantizedTensor(concatenated)
        description = 'concatenated[[' + '] | ['.join(lt.description for lt in lazy_tensors) + ']]'
        return LazyTensor(load, concatenated_shape, lazy_tensors[0].data_type, description, 1.0, 0)

    return {name: convert(name) for name in names}

def merge_multifile_models(models_plus: list[ModelPlus]) -> ModelPlus:
    formats = set(mp.format for mp in models_plus)
    assert len(formats) == 1, "different formats?"
    format = formats.pop()
    paths = [path for mp in models_plus for path in mp.paths]

    if any("model.embed_tokens.weight" in mp.model for mp in models_plus) :
        # Transformers models put different tensors in different files, but
        # don't split indivdual tensors between files.
        model: LazyModel = {}
        for mp in models_plus:
            model.update(mp.model)
    elif any("language_model.model.embed_tokens.weight" in mp.model for mp in models_plus):
        # Transformers models put different tensors in different files, but
        # don't split indivdual tensors between files.
        model: LazyModel = {}
        for mp in models_plus:
            model.update(mp.model)
    elif any("transformer.wte.weight" in mp.model for mp in models_plus) or any("wte.weight" in mp.model for mp in models_plus):
        # Transformers models put different tensors in different files, but
        # don't split indivdual tensors between files.
        model: LazyModel = {}
        for mp in models_plus:
            model.update(mp.model)
    else:
        model = merge_sharded([mp.model for mp in models_plus])

    return ModelPlus(model, paths, format)

# Given any path belonging to a multi-file model (e.g. foo.bin.1), return
# the nth path in the model.
def nth_multifile_path(path: Path, n: int) -> Path | None:
    patterns: list[tuple[str, str]] = [
        # - x.00.pth, x.01.pth, etc.
        (r'\.[0-9]{2}\.pth$', f'.{n:02}.pth'),
        # - x-00001-of-00002.bin, x-00002-of-00002.bin, etc.
        (r'-[0-9]{5}-of-(.*)$', fr'-{n:05}-of-\1'),
        # x.bin, x.bin.1, etc.
        (r'(\.[0-9]+)?$', r'\1' if n == 0 else fr'\1.{n}')
    ]
    for regex, replacement in patterns:
        if re.search(regex, path.name):
            new_path = path.with_name(re.sub(regex, replacement, path.name))
            if new_path.exists():
                return new_path
    return None

# Given any path belonging to a multi-file model (e.g. foo.bin.1), return
# the whole list of paths in the model.
def find_multifile_paths(path: Path) -> list[Path]:
    ret: list[Path] = []
    for i in itertools.count(1):
        nth_path = nth_multifile_path(path, i)
        if nth_path is None:
            break
        ret.append(nth_path)
    if not ret:
        # No matches.
        return [path]
    return ret

def lazy_load_torch_file(outer_fp: IO[bytes], path: Path) -> ModelPlus:
    zf = zipfile.ZipFile(outer_fp)
    pickle_paths = [name for name in zf.namelist() if name.endswith('.pkl')]
    assert len(pickle_paths) == 1, pickle_paths
    pickle_fp = zf.open(pickle_paths[0], 'r')
    unpickler = LazyUnpickler(pickle_fp,
                              data_base_path = pickle_paths[0][:-4],
                              zip_file = zf)
    model = unpickler.load()
    as_dict = dict(model.items())
    return ModelPlus(model = as_dict, paths = [path], format = 'torch')

def lazy_load_safetensors_file(fp: IO[bytes], path: Path) -> ModelPlus:
    header_size, = struct.unpack('<Q', fp.read(8))
    header: dict[str, dict[str, Any]] = json.loads(fp.read(header_size))
    # Use mmap for the actual data
    mapped = memoryview(mmap.mmap(fp.fileno(), 0, access = mmap.ACCESS_READ))
    byte_buf = mapped[8 + header_size:]

    def convert(info: dict[str, Any]) -> LazyTensor:
        data_type = SAFETENSORS_DATA_TYPES[info['dtype']]
        numpy_dtype = data_type.dtype
        shape: list[int] = info['shape']
        begin, end = info['data_offsets']
        assert 0 <= begin <= end <= len(byte_buf)
        assert end - begin == math.prod(shape) * numpy_dtype.itemsize
        buf = byte_buf[begin:end]

        def load() -> UnquantizedTensor:
            return UnquantizedTensor(np.frombuffer(buf, dtype = numpy_dtype).reshape(shape))
        description = f'safetensors begin = {begin} end = {end} type = {data_type} path = {path}'
        return LazyTensor(load, shape, data_type, description, 1.0, 0)
    model = {name: convert(info) for (name, info) in header.items() if name != '__metadata__'}
    return ModelPlus(model = model, paths = [path], format = 'safetensors')

def lazy_load_file(path: Path) -> ModelPlus:
    fp = open(path, 'rb')
    first8 = fp.read(8)
    fp.seek(0)
    if first8[:2] == b'PK':
        # A zip file, i.e. PyTorch format
        return lazy_load_torch_file(fp, path)
    elif struct.unpack('<Q', first8)[0] < 16 * 1024 * 1024:
        # Probably safetensors
        return lazy_load_safetensors_file(fp, path)
    else:
        raise ValueError(f"unknown format: {path}")

# Load a model of any supported format
def load_some_model(path: Path) -> ModelPlus:
    if path.is_dir():
        # Check if safetensors files
        files = list(path.glob("model-00001-of-*.safetensors"))
        if not files:
            files = list(path.glob("model.safetensors"))
        # Check if LoRA safetensors file
        if not files:
            files = list(path.glob("adapter_model.safetensors"))
        # Check if LoRA PyTorch file
        if not files:
            files = list(path.glob("adapter_model.bin"))
        if not files:
            # Try if PyTorch files
            globs = ["consolidated.00.pth", "pytorch_model-00001-of-*.bin", "*.pt", "pytorch_model.bin", "training_args.bin"]
            files = [file for glob in globs for file in path.glob(glob)]
        if not files:
            raise Exception(f"Can't find model in directory {path}")
        if len(files) > 1:
            raise Exception(f"Found multiple models in {path}, not sure which to pick: {files}")
        path = files[0]

    paths = find_multifile_paths(path)
    models_plus: list[ModelPlus] = []
    for path in paths:
        print(f"Loading model file {path}")
        models_plus.append(lazy_load_file(path))

    model_plus = merge_multifile_models(models_plus)
    return model_plus

#
# Write Output
#

class OutputFile:
    def __init__(self, fname_out: Path, params: Params) -> None:
        self.gguf = GGUFWriter(fname_out, params.n_align, params.arch)

    def add_meta_arch(self, params: Params) -> None:
        self.gguf.add_name                (params.name)
        self.gguf.add_arch                (params.arch)
        self.gguf.add_tokenizer           (params.tokenizer)
        self.gguf.add_general_output      (params.output)
        self.gguf.add_custom_alignment    (params.n_align)
        self.gguf.add_vocab_size          (params.n_vocab)
        self.gguf.add_context_length      (params.n_ctx)
        self.gguf.add_embedding_length    (params.n_embd)
        self.gguf.add_embedding_per_head  (params.embd_per_head)
        self.gguf.add_feed_forward_length (params.n_ff)
        self.gguf.add_block_count         (params.n_layer)
        self.gguf.add_head_count          (params.n_head)
        self.gguf.add_head_count_kv       (params.n_head_kv)
        self.gguf.add_connector           (params.connector)
        self.gguf.add_gating              (params.gating)
        self.gguf.add_normalization       (params.norm)
        self.gguf.add_layer_norm_rms_eps  (params.f_norm_eps)
        self.gguf.add_activation          (params.activation)
        self.gguf.add_pos_embd            (params.pos_embd)
        self.gguf.add_attention_mode      (params.attention_mode)

        if params.pos_embd == "RoPE":
            self.gguf.add_complex_org(params.comp_org)
            self.gguf.add_rope_freq_base(params.f_rope_scale)
            self.gguf.add_num_rotations(params.n_rot)
            if params.f_rope_factor_short is not None:
                self.gguf.add_rope_scaling_factor_short(params.f_rope_factor_short)
            if params.f_rope_factor_long is not None:
                self.gguf.add_rope_scaling_factor_long(params.f_rope_factor_long)
            if params.rope_attn_factor is not None:
                self.gguf.add_rope_scaling_attn_factor(params.rope_attn_factor)

        if params.ftype is not None:
            self.gguf.add_file_type(params.ftype)
            if params.ftype == GGMLFileType.MostlyZ4:
                self.gguf.add_quantization_version(GGMLQuantizationType.Z4)
            elif params.ftype == GGMLFileType.Z4_FP16:
                self.gguf.add_quantization_version(GGMLQuantizationType.Z4_FP16)
            elif params.ftype == GGMLFileType.Z4_BF16:
                self.gguf.add_quantization_version(GGMLQuantizationType.Z4_BF16)
            elif params.ftype == GGMLFileType.MostlyQ4_0_32:
                self.gguf.add_quantization_version(GGMLQuantizationType.Q4_0_32)
            elif params.ftype == GGMLFileType.MostlyZ8:
                self.gguf.add_quantization_version(GGMLQuantizationType.Z8)

    def add_meta_arch_lora(self, params: Params) -> None:
        self.gguf.add_name                (params.name)
        self.gguf.add_arch                (params.arch)
        self.gguf.add_block_count         (params.n_layer)
        self.gguf.add_custom_alignment    (params.n_align)
        self.gguf.add_lora_alpha          (params.alpha)
        self.gguf.add_lora_rank           (params.rank)
        if params.ftype is not None:
            if params.ftype == GGMLFileType.AllF32:
                self.gguf.add_quantization_version(GGMLQuantizationType.F32)

    def add_meta_special_vocab(self, svocab: SpecialVocab) -> None:
        svocab.add_to_gguf(self.gguf)

    def add_tensor_info(self, name: str, tensor: LazyTensor, raw_dtype: GGMLQuantizationType | None = None) -> None:
        n_elements = int(np.prod(tensor.shape))
        data_type = tensor.data_type.dtype
        if raw_dtype is None:
            raw_dtype = getattr(tensor.data_type, 'ggml_type', None)
        if raw_dtype is None:
            data_nbytes = tensor.data_type.elements_to_bytes(n_elements)
        else:
            # Z4 Quantization(s)
            if raw_dtype == GGMLQuantizationType.Z4:
                # Pack 2 INT4 values into 1 INT8. Keep each block's scale FP32.
                PACK_SIZE  = 2
                BLOCK_SIZE = 128
                pad = OutputFile.ggml_pad((n_elements // PACK_SIZE), self.gguf.data_alignment) - (n_elements // PACK_SIZE)
                pad = pad + OutputFile.ggml_pad(((n_elements * 4) // BLOCK_SIZE), self.gguf.data_alignment) - (((n_elements * 4) // BLOCK_SIZE))
                data_nbytes = (n_elements // PACK_SIZE) + ((n_elements * 4) // BLOCK_SIZE) + pad
            elif raw_dtype == GGMLQuantizationType.Z4_FP16:
                # Pack 2 INT4 values into 1 INT8. Keep each block's scale FP16.
                PACK_SIZE  = 2
                BLOCK_SIZE = 128
                pad = OutputFile.ggml_pad((n_elements // PACK_SIZE), self.gguf.data_alignment) - (n_elements // PACK_SIZE)
                pad = pad + OutputFile.ggml_pad(((n_elements * 2) // BLOCK_SIZE), self.gguf.data_alignment) - (((n_elements * 2) // BLOCK_SIZE))
                data_nbytes = (n_elements // PACK_SIZE) + ((n_elements * 2) // BLOCK_SIZE) + pad
            elif raw_dtype == GGMLQuantizationType.Z4_BF16:
                # Pack 2 INT4 values into 1 INT8. Keep each block's scale BF16.
                PACK_SIZE  = 2
                BLOCK_SIZE = 128
                pad = OutputFile.ggml_pad((n_elements // PACK_SIZE), self.gguf.data_alignment) - (n_elements // PACK_SIZE)
                pad = pad + OutputFile.ggml_pad(((n_elements * 2) // BLOCK_SIZE), self.gguf.data_alignment) - (((n_elements * 2) // BLOCK_SIZE))
                data_nbytes = (n_elements // PACK_SIZE) + ((n_elements * 2) // BLOCK_SIZE) + pad
            elif raw_dtype == GGMLQuantizationType.F16:
                # Z4_FP16 1D Tensors stored in FP16
                data_nbytes = n_elements * 2
            elif raw_dtype == GGMLQuantizationType.BFloat16:
                # Z4_BF16 1D Tensors stored in BFloat16
                data_nbytes = n_elements * 2
            elif raw_dtype == GGMLQuantizationType.Q4_0_32:
                # Pack 2 INT4 values into 1 INT8. Keep each block's scale FP32.
                PACK_SIZE  = 2
                BLOCK_SIZE = 32
                pad = OutputFile.ggml_pad((n_elements // PACK_SIZE), self.gguf.data_alignment) - (n_elements // PACK_SIZE)
                pad = pad + OutputFile.ggml_pad(((n_elements * 4) // BLOCK_SIZE), self.gguf.data_alignment) - (((n_elements * 4) // BLOCK_SIZE))
                data_nbytes = (n_elements // PACK_SIZE) + ((n_elements * 4) // BLOCK_SIZE) + pad
            elif raw_dtype == GGMLQuantizationType.Z8:
                # Pack 2 INT4 values into 1 INT8. Keep each block's scale FP32.
                PACK_SIZE  = 2
                BLOCK_SIZE = 128
                pad = OutputFile.ggml_pad((n_elements // PACK_SIZE), self.gguf.data_alignment) - (n_elements // PACK_SIZE)
                pad = pad + OutputFile.ggml_pad(((n_elements * 4) // BLOCK_SIZE), self.gguf.data_alignment) - (((n_elements * 4) // BLOCK_SIZE))
                data_nbytes = (n_elements // PACK_SIZE) + ((n_elements * 4) // BLOCK_SIZE) + pad

        self.gguf.add_tensor_info(name, tensor.shape, data_type, data_nbytes, raw_dtype = raw_dtype)

    def write_meta(self) -> None:
        self.gguf.write_header_to_file()
        self.gguf.write_kv_data_to_file()

    def write_tensor_info(self) -> None:
        self.gguf.write_ti_data_to_file()

    def close(self) -> None:
        self.gguf.close()

    @staticmethod
    def do_item(item: tuple[str, LazyTensor]) -> NDArray:
        name, lazy_tensor = item
        tensor = lazy_tensor.load().to_ggml()
        return (tensor.ndarray * lazy_tensor.scale) + lazy_tensor.offset

    @staticmethod
    def ggml_pad(x: int, n: int) -> int:
        return ((x + n - 1) // n) * n

    @staticmethod
    def quantize_tensor(name: str, lazy_tensor: LazyTensor, quantization: str, ndarray: np.ndarray[Any, Any], i: int, num_tensors: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        size = ' x '.join(f"{dim:6d}" for dim in lazy_tensor.shape)
        padi = len(str(num_tensors))
        if os.name == "nt":
            from ctypes.util import find_library
            filename = "QnnGenAiTransformerComposerQuantizer.dll"
            quantizer = ctypes.cdll.LoadLibrary(find_library(filename))
        else:
            filename = "libQnnGenAiTransformerComposerQuantizer.so"
            quantizer = ctypes.cdll.LoadLibrary(filename)
        # Pack 2 INT4 values into 1 INT8. Keep each block's scale FP32.
        PACK_SIZE  = 2
        n_elements = int(np.prod(lazy_tensor.shape))
        n_rows = lazy_tensor.shape[0]
        n_cols = lazy_tensor.shape[1]
        n_quants = n_elements // PACK_SIZE
        # Q4_0_32 Quantization
        if quantization == "Q4":
            BLOCK_SIZE = 32
            n_blocks = n_elements // BLOCK_SIZE
            quants = np.zeros((n_quants), dtype = np.uint8)
            scales = np.zeros((n_blocks), dtype = np.float32)
            # Call C-based Quantizer backend from shared library
            quantizer.quantize_q4_0_32(ndarray.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       ctypes.c_void_p(quants.ctypes.data), ctypes.c_void_p(scales.ctypes.data),
                                       n_rows, n_cols)
            print(f"[{(i + 1) : {padi}d} / {num_tensors}] Quantizing tensor {name:38s} | size {size:16} | type {'Q4_0_32':7}")
        # Z4 Quantization
        elif quantization in ["Z4", "Z4_FP16", "Z4_BF16"]:
            BLOCK_SIZE = 128
            n_blocks = n_elements // BLOCK_SIZE
            quants = np.zeros((n_quants), dtype = np.uint8)
            scales = np.zeros((n_blocks), dtype = np.float32)
            # Call C-based Quantizer backend from shared library
            quantizer.quantize_z4(ndarray.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  ctypes.c_void_p(quants.ctypes.data), ctypes.c_void_p(scales.ctypes.data),
                                  n_rows, n_cols)
            print(f"[{(i + 1) : {padi}d} / {num_tensors}] Quantizing tensor {name:38s} | size {size:16} | type {'Z4':4}")
        elif quantization in ["Z8"]:
            BLOCK_SIZE = 128
            n_blocks = n_elements // BLOCK_SIZE
            quants = np.zeros((n_quants), dtype = np.uint8)
            scales = np.zeros((n_blocks), dtype = np.float32)
            # Call C-based Quantizer backend from shared library
            quantizer.quantize_z8(ndarray.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  ctypes.c_void_p(quants.ctypes.data), ctypes.c_void_p(scales.ctypes.data),
                                  n_rows, n_cols)
            print(f"[{(i + 1) : {padi}d} / {num_tensors}] Quantizing tensor {name:38s} | size {size:16} | type {'Z8':4}")
        # Convert FP32 scales to FP16 or BF16 based on Quantization type
        if quantization == "Z4_FP16":
            scales = scales.astype(np.float16)
        elif quantization == "Z4_BF16":
            scales_bf16 = np.zeros(n_blocks, dtype = np.uint16)
            quantizer.fp32_to_bf16(scales.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   scales_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), 1, n_blocks)
            scales = scales_bf16
        return quants, scales

    @staticmethod
    def write_all(fname_out: Path, params: Params, model: LazyModel, lm_quantize: str | None = None) -> None:
        of = OutputFile(fname_out, params)

        # meta data
        if getattr(params, "alpha", None) is not None:
            of.add_meta_arch_lora(params)
        else:
            of.add_meta_arch(params)

        # tensor info
        for name, lazy_tensor in model.items():
            if name == "output.weight":
                if lm_quantize == "Z4":
                    of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.Z4)
                elif lm_quantize == "Q4":
                    of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.Q4_0_32)
                elif lm_quantize == "Z4_FP16":
                    of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.Z4_FP16)
                elif lm_quantize == "Z4_FP16":
                    of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.Z4_BF16)
                elif lm_quantize == "Z8":
                    of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.Z8)
                else: # in case of FP_32, raw_dtype will be datatype i.e FP_32
                    of.add_tensor_info(name, lazy_tensor)
            else:
                quantize = True
                dims = len(lazy_tensor.shape)
                quantize = quantize and (dims == 2)
                quantize = quantize and (name != "token_embd.weight") and (name != "token_embd_pos.weight")
                if params.ftype == GGMLFileType.MostlyZ4:
                    if quantize:
                        of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.Z4)
                    else:
                        of.add_tensor_info(name, lazy_tensor)
                elif params.ftype == GGMLFileType.Z4_FP16:
                    if quantize:
                        of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.Z4_FP16)
                    else:
                        of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.F16)
                elif params.ftype == GGMLFileType.Z4_BF16:
                    if quantize:
                        of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.Z4_BF16)
                    else:
                        of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.BFloat16)
                elif params.ftype == GGMLFileType.MostlyQ4_0_32:
                    if quantize:
                        of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.Q4_0_32)
                    else:
                        of.add_tensor_info(name, lazy_tensor)
                elif params.ftype == GGMLFileType.MostlyZ8:
                    if quantize:
                        of.add_tensor_info(name, lazy_tensor, raw_dtype = GGMLQuantizationType.Z8)
                    else:
                        of.add_tensor_info(name, lazy_tensor)
                else:
                    of.add_tensor_info(name, lazy_tensor)

        of.write_meta()
        of.write_tensor_info()

        # tensor data
        ndarrays = map(OutputFile.do_item, model.items())

        padi = len(str(len(model)))
        for i, ((name, lazy_tensor), ndarray) in enumerate(zip(model.items(), ndarrays)):
            size = ' x '.join(f"{dim:6d}" for dim in lazy_tensor.shape)
            if name == "output.weight":
                # Quantize LM_HEAD Tensor
                if lm_quantize in ["Z4", "Z4_FP16", "Z4_BF16", "Q4", "Z8"]:
                    quants, scales = OutputFile.quantize_tensor(name, lazy_tensor, lm_quantize, ndarray, i, len(model))
                    of.gguf.write_tensor_data(quants)
                    of.gguf.write_tensor_data(scales)
                # LM_HEAD Tensor is FP_32
                else:
                    print(f"[{(i + 1) : {padi}d} / {len(model)}] Writing tensor    {name:38s} | size {size:16} | type {lazy_tensor.data_type.name:7}")
                    of.gguf.write_tensor_data(ndarray)
            else:
                quantize = True
                dims = len(lazy_tensor.shape)
                quantize = quantize and (dims == 2)
                quantize = quantize and ((params.ftype == GGMLFileType.MostlyZ4) or (params.ftype == GGMLFileType.Z4_FP16) or (params.ftype == GGMLFileType.Z4_BF16)
                                         or (params.ftype == GGMLFileType.MostlyQ4_0_32) or (params.ftype == GGMLFileType.MostlyZ8))
                quantize = quantize and (name != "token_embd.weight") and (name != "token_embd_pos.weight")
                if quantize:
                    if params.ftype == GGMLFileType.MostlyZ4:
                        quants, scales = OutputFile.quantize_tensor(name, lazy_tensor, "Z4", ndarray, i, len(model))
                    elif params.ftype == GGMLFileType.Z4_FP16:
                        quants, scales = OutputFile.quantize_tensor(name, lazy_tensor, "Z4_FP16", ndarray, i, len(model))
                    elif params.ftype == GGMLFileType.Z4_BF16:
                        quants, scales = OutputFile.quantize_tensor(name, lazy_tensor, "Z4_BF16", ndarray, i, len(model))
                    elif params.ftype == GGMLFileType.MostlyQ4_0_32:
                        quants, scales = OutputFile.quantize_tensor(name, lazy_tensor, "Q4", ndarray, i, len(model))
                    elif params.ftype == GGMLFileType.MostlyZ8:
                        quants, scales = OutputFile.quantize_tensor(name, lazy_tensor, "Z8", ndarray, i, len(model))
                    of.gguf.write_tensor_data(quants)
                    of.gguf.write_tensor_data(scales)
                else:
                    if params.ftype == GGMLFileType.MostlyZ4:
                        print(f"[{(i + 1) : {padi}d} / {len(model)}] Writing tensor    {name:38s} | size {size:16} | type {lazy_tensor.data_type.name:4}")
                        of.gguf.write_tensor_data(ndarray)
                    elif params.ftype == GGMLFileType.Z4_FP16:
                        print(f"[{(i + 1) : {padi}d} / {len(model)}] Writing tensor    {name:38s} | size {size:16} | type {'FP16':4}")
                        of.gguf.write_tensor_data(ndarray.astype(np.float16))
                    elif params.ftype == GGMLFileType.Z4_BF16:
                        dims = len(lazy_tensor.shape)
                        if dims == 2:
                            n_elements = int(np.prod(lazy_tensor.shape))
                            n_rows = lazy_tensor.shape[0]
                            n_cols = lazy_tensor.shape[1]
                        else:
                            n_rows = 1
                            n_cols = lazy_tensor.shape[0]
                            n_elements = n_rows * n_cols
                        output_tensor = np.zeros(n_elements, dtype = np.uint16)
                        if os.name == "nt":
                            from ctypes.util import find_library
                            filename = "QnnGenAiTransformerComposerQuantizer.dll"
                            quantizer = ctypes.cdll.LoadLibrary(find_library(filename))
                        else:
                            filename = "libQnnGenAiTransformerComposerQuantizer.so"
                            quantizer = ctypes.cdll.LoadLibrary(filename)
                        quantizer.fp32_to_bf16(ndarray.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                               output_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), n_rows, n_cols)
                        print(f"[{(i + 1) : {padi}d} / {len(model)}] Writing tensor    {name:38s} | size {size:16} | type {'BF16':7}")
                        of.gguf.write_tensor_data(output_tensor)
                    elif params.ftype == GGMLFileType.MostlyZ8:
                        print(f"[{(i + 1) : {padi}d} / {len(model)}] Writing tensor    {name:38s} | size {size:16} | type {lazy_tensor.data_type.name:4}")
                        of.gguf.write_tensor_data(ndarray)
                    else:
                        print(f"[{(i + 1) : {padi}d} / {len(model)}] Writing tensor    {name:38s} | size {size:16} | type {lazy_tensor.data_type.name:7}")
                        of.gguf.write_tensor_data(ndarray)
        of.close()

def pick_output_type(model: LazyModel, output_type_str: str | None) -> GGMLFileType:
    wq_type = model[NAMES[MODEL_TENSOR.ATTN_Q].format(bid=0)].data_type

    if output_type_str == "f32" or (output_type_str is None and wq_type == DT_F32):
        return GGMLFileType.AllF32
    if output_type_str == "f16" or (output_type_str is None and wq_type in (DT_F16, DT_BF16)):
        return GGMLFileType.MostlyF16

    name_to_type = {name: lazy_tensor.data_type for (name, lazy_tensor) in model.items()}

    raise Exception(f"Unexpected combination of types: {name_to_type}")

def convert_to_output_type(model: LazyModel, output_type: GGMLFileType) -> LazyModel:
    return {name: tensor.astype(output_type.type_for_tensor(name, tensor))
            for (name, tensor) in model.items()}

def default_outfile(model_paths: list[Path], file_type: GGMLFileType) -> Path:
    namestr = {
        GGMLFileType.AllF32:         "f32",
        GGMLFileType.MostlyF16:      "f16",
        GGMLFileType.MostlyZ4:       "Z4",
        GGMLFileType.MostlyQ4_0_32:  "Q4_0_32",
        GGMLFileType.MostlyZ8:       "Z8",
    }[file_type]
    ret = model_paths[0].parent / f"{namestr}.bin"
    if ret in model_paths:
        sys.stderr.write(f"Error: Default output path ({ret}) would overwrite the input. "
                         "Please explicitly specify a path using --outfile.\n")
        sys.exit(1)
    return ret
