//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_CACHE_FILE_HPP
#define QUALLA_DETAIL_CACHE_FILE_HPP

#include <cstdint>

namespace qualla {

struct CacheFileSpec {
  enum DataType : uint8_t {
    UINT8_T,
    UINT16_T,
    UINT32_T,
    UINT64_T,
    INT8_T,
    INT16_T,
    INT32_T,
    INT64_T,
    FLOAT8_T,
    FLOAT16_T,
    FLOAT32_T,
    FLOAT64_T,
    BOOL
  };

  uint32_t num_tensors;
  uint32_t magic;

  // Let's assume all tensors have "same" datatype and update_size
  DataType dtype;
  uint8_t pad8_t;
  uint16_t n_heads;
  uint16_t embed_dim;
  uint16_t update_size;
  CacheFileSpec() {}
  CacheFileSpec(uint32_t _num_tensors,
                uint32_t _magic,
                DataType _dtype,
                uint8_t _pad8_t,
                uint16_t _n_heads,
                uint16_t _embed_dim,
                uint16_t _update_size) {
    num_tensors = _num_tensors;
    magic       = _magic;
    dtype       = _dtype;
    pad8_t      = _pad8_t;
    n_heads     = _n_heads;
    embed_dim   = _embed_dim;
    update_size = _update_size;
  }
};

static_assert(sizeof(size_t) == 8);
static_assert(sizeof(CacheFileSpec) == 16);  // Make sure alignment is correct

struct CacheTensorSpec {
  uint64_t start_offset;
  uint64_t data_size;
  uint8_t concat_dim;

  char graph_name[127];
  char tensor_name[128];
};

static_assert(sizeof(CacheTensorSpec) == 272);

}  // namespace qualla

#endif  // QUALLA_DETAIL_CACHE_FILE_HPP
