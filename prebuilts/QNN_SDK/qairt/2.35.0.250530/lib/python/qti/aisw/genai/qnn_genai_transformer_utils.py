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

import enum
import json
import pickle
import zipfile
import warnings
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from sentencepiece import SentencePieceProcessor
from typing import IO, Any, Callable, Iterable, Literal

from qti.aisw.genai.qnn_genai_transformer_gguf import *
from qti.aisw.genai import qnn_genai_transformer_tokenizer

#
# Model Definitions
#

NAMES = MODEL_TENSOR_NAMES
# np.ndarray[foo, bar] create a NumPy array of shape 'foo' and dtype 'bar'
NDArray   = 'np.ndarray[Any, Any]'
LazyModel = 'dict[str, LazyTensor]'

@dataclass(frozen = True)
class DataType:
    name: str
    dtype: np.dtype[Any]
    valid_conversions: list[str]

    def elements_to_bytes(self, n_elements: int) -> int:
        return n_elements * self.dtype.itemsize

DT_F16  = DataType('F16',  dtype = np.dtype(np.float16), valid_conversions = ['F32'])
DT_F32  = DataType('F32',  dtype = np.dtype(np.float32), valid_conversions = ['F16'])
DT_UINT8= DataType('UINT8',dtype = np.dtype(np.uint8),   valid_conversions = ['F32'])
DT_BOOL = DataType('BOOL', dtype = np.dtype(np.bool_),   valid_conversions = ['F32'])
DT_I64  = DataType('I64',  dtype = np.dtype(np.int64),   valid_conversions = [])
DT_BF16 = DataType('BF16', dtype = np.dtype(np.uint16),  valid_conversions = ['F32', 'F16'])

NUMPY_TYPE_TO_DATA_TYPE: dict[np.dtype[Any], DataType] = {}
for dt in (DT_F16, DT_F32, DT_BF16, DT_I64):
    if dt.dtype in NUMPY_TYPE_TO_DATA_TYPE:
        raise ValueError(f'Invalid duplicate data type {dt}')
    NUMPY_TYPE_TO_DATA_TYPE[dt.dtype] = dt

SAFETENSORS_DATA_TYPES: dict[str, DataType] = {
    'F16' : DT_F16,
    'F32' : DT_F32,
    'I64' : DT_I64,
    'BF16': DT_BF16,
}

class GGMLFileType(enum.IntEnum):
    AllF32          = 0
    MostlyF16       = 1  # except 1d tensors
    MostlyQ4_0      = 2  # except 1d tensors
    MostlyZ4        = 20 # except 1d tensors
    Z4_FP16         = 21 # except 1d tensors which are FP16
    Z4_BF16         = 22 # except 1d tensors which are BF16
    MostlyQ4_0_32   = 30
    MostlyZ8        = 40

    def type_for_tensor(self, name: str, tensor: LazyTensor) -> DataType:
        dt = GGML_FILE_TYPE_TO_DATA_TYPE.get(self)
        if dt is None:
            raise ValueError(self)
        # 1D tensors are always F32.
        return dt if len(tensor.shape) > 1 else DT_F32

GGML_FILE_TYPE_TO_DATA_TYPE: dict[GGMLFileType, DataType] = {
    GGMLFileType.AllF32    : DT_F32,
    GGMLFileType.MostlyF16 : DT_F16,
}

#
# Data Loading
#

# HuggingFace models permute method in 'llama_weights_to_hf.py'
# Permute for sliced Rotary Positional Embeddings (RoPE) in HF Transformers library
# def permute(w):
#     return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)
# Need to undo the permute done on Q & K weights in HuggingFace models
def permute(weights: NDArray, n_head: int, n_head_kv: int) -> NDArray:
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))

class Tensor(metaclass = ABCMeta):
    data_type: DataType

    @abstractmethod
    def astype(self, data_type: DataType) -> Tensor: ...
    @abstractmethod
    def permute(self, n_head: int, n_head_kv: int) -> Tensor: ...
    @abstractmethod
    def permute_part(self, n_part: int, n_head: int, n_head_kv: int) -> UnquantizedTensor: ...
    @abstractmethod
    def part(self, n_part: int) -> UnquantizedTensor: ...
    @abstractmethod
    def part_columns(self, n_part: int) -> UnquantizedTensor: ...
    @abstractmethod
    def transpose(self) -> UnquantizedTensor: ...
    @abstractmethod
    def to_ggml(self) -> UnquantizedTensor: ...

def bf16_to_fp32(bf16_arr: np.ndarray[Any, np.dtype[np.uint16]]) -> NDArray:
    assert bf16_arr.dtype == np.uint16, f"Input should be numpy.uint16, but is {bf16_arr.dtype} instead"
    fp32_arr = bf16_arr.astype(np.uint32) << 16
    return fp32_arr.view(np.float32)

class UnquantizedTensor(Tensor):
    def __init__(self, ndarray: NDArray) -> None:
        assert isinstance(ndarray, np.ndarray)
        self.ndarray = ndarray
        self.data_type = NUMPY_TYPE_TO_DATA_TYPE[ndarray.dtype]

    def astype(self, data_type: DataType) -> Tensor:
        dtype = data_type.dtype
        if self.data_type == DT_BF16:
            self.ndarray = bf16_to_fp32(self.ndarray)
        return UnquantizedTensor(self.ndarray.astype(dtype))

    def to_ggml(self) -> UnquantizedTensor:
        return self

    def permute_part(self, n_part: int, n_head: int, n_head_kv: int) -> UnquantizedTensor:
        r = self.ndarray.shape[0] // 3
        return UnquantizedTensor(permute(self.ndarray[r * n_part : r * n_part + r, ...], n_head, n_head_kv))

    def part(self, n_part: int, total_part: int = 3) -> UnquantizedTensor:
        r = self.ndarray.shape[0] // total_part
        return UnquantizedTensor(self.ndarray[r * n_part : r * n_part + r, ...])

    def part_columns(self, n_part: int, total_part: int = 3) -> UnquantizedTensor:
        self.ndarray = self.ndarray.transpose()
        r = self.ndarray.shape[0] // total_part
        return UnquantizedTensor(self.ndarray[r * n_part : r * n_part + r, ...])

    def permute(self, n_head: int, n_head_kv: int) -> UnquantizedTensor:
        return UnquantizedTensor(permute(self.ndarray, n_head, n_head_kv))

    def transpose(self) -> UnquantizedTensor:
        self.ndarray = self.ndarray.transpose()
        return UnquantizedTensor(self.ndarray)

@dataclass
class LazyTensor:
    _load: Callable[[], Tensor]
    shape: list[int]
    data_type: DataType
    description: str
    scale: float
    offset: int

    def load(self) -> Tensor:
        ret = self._load()
        assert ret.data_type == self.data_type or (self.data_type.dtype == ret.data_type.dtype), \
            (self.data_type, ret.data_type, self.description)
        return ret

    def astype(self, data_type: DataType) -> LazyTensor:
        self.validate_conversion_to(data_type)

        def load() -> Tensor:
            return self.load().astype(data_type)
        return LazyTensor(load, self.shape, data_type, f'convert({data_type}) {self.description}', self.scale, self.offset)

    def validate_conversion_to(self, data_type: DataType) -> None:
        if data_type != self.data_type and data_type.name not in self.data_type.valid_conversions:
            raise ValueError(f'Cannot validate conversion from {self.data_type} to {data_type}.')

def load_unquantized(lazy_tensor: LazyTensor, expected_dtype: Any = None, convert: bool = False) -> NDArray:
    tensor = lazy_tensor.load()
    assert isinstance(tensor, UnquantizedTensor)

    actual_shape = list(tensor.ndarray.shape)
    assert actual_shape == lazy_tensor.shape, (actual_shape, lazy_tensor.shape)
    if expected_dtype is not None and expected_dtype != tensor.ndarray.dtype:
        if convert:
            tensor.ndarray = tensor.ndarray.astype(expected_dtype)
        else:
            raise ValueError(f'expected this tensor to have dtype {expected_dtype}, got {tensor.ndarray.dtype}')

    return tensor.ndarray

@dataclass
class ModelPlus:
    model: LazyModel
    # Path to model files
    paths: list[Path]
    format: Literal['ggml', 'torch', 'safetensors', 'none']

# Functionality that simulates `torch.load` but where individual tensors are
# only loaded into memory on demand, not all at once.
@dataclass
class LazyStorageKind:
    data_type: DataType

@dataclass
class LazyStorage:
    load: Callable[[int, int], NDArray]
    kind: LazyStorageKind
    description: str

class LazyUnpickler(pickle.Unpickler):
    def __init__(self, fp: IO[bytes], data_base_path: str, zip_file: zipfile.ZipFile):
        super().__init__(fp)
        self.data_base_path = data_base_path
        self.zip_file = zip_file

    def persistent_load(self, pid: Any) -> Any:
        assert pid[0] == 'storage'
        assert isinstance(pid[1], LazyStorageKind)
        data_type = pid[1].data_type
        filename_stem = pid[2]
        filename = self.data_base_path + '/' + filename_stem
        info = self.zip_file.getinfo(filename)

        def load(offset: int, elm_count: int) -> NDArray:
            dtype = data_type.dtype
            fp = self.zip_file.open(info)
            fp.seek(offset * dtype.itemsize)
            size = elm_count * dtype.itemsize
            data = fp.read(size)
            assert len(data) == size
            return np.frombuffer(data, dtype)
        description = f'storage data_type={data_type} path-in-zip={filename} path={self.zip_file.filename}'
        return LazyStorage(load = load, kind = pid[1], description = description)

    @staticmethod
    def lazy_rebuild_tensor_v2(storage: Any, storage_offset: Any, size: Any, stride: Any,
                               # pyright: ignore[reportSelfClsParameterName]
                               requires_grad: Any, backward_hooks: Any, metadata: Any = None) -> LazyTensor:
        assert isinstance(storage, LazyStorage)

        def load() -> UnquantizedTensor:
            order = 'C'
            if len(size) == 2:
                # 2D tensors
                if stride[0] == size[1]:
                    #row major
                    elm_count = stride[0] * size[0]
                else:
                    #column major
                    elm_count = stride[1] * size[1]
                    order = 'F'
            else:
                # 1D tensors
                elm_count = stride[0] * size[0]
            return UnquantizedTensor(storage.load(storage_offset, elm_count).reshape(size, order = order))
        description = f'pickled storage_offset={storage_offset} in {storage.description}'
        return LazyTensor(load, list(size), storage.kind.data_type, description, 1.0, 0)

    @staticmethod
    def rebuild_from_type_v2(func, new_type, args, state):
        return func(*args)

    CLASSES: dict[tuple[str, str], Any] = {
        ('torch._tensor', '_rebuild_from_type_v2'): getattr(rebuild_from_type_v2, '__func__'),
        ('torch._utils', '_rebuild_tensor_v2'): getattr(lazy_rebuild_tensor_v2, '__func__'),
        ('torch', 'HalfStorage'): LazyStorageKind(DT_F16),
        ('torch', 'FloatStorage'): LazyStorageKind(DT_F32),
        ('torch', 'ByteStorage'): LazyStorageKind(DT_UINT8),
        ('torch', 'BoolStorage'): LazyStorageKind(DT_BOOL),
        ('torch', 'Int64Storage'): LazyStorageKind(DT_I64),
        ('torch', 'BFloat16Storage'): LazyStorageKind(DT_BF16),
        ('torch', 'Tensor'): LazyTensor,
    }

    def find_class(self, module: str, name: str) -> Any:
        if not module.startswith('torch'):
            return super().find_class(module, name)
        return self.CLASSES[(module, name)]
