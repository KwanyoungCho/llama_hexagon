//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>  // for std::max_element
#include <fstream>

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "smart-mask.hpp"

#define __DEBUG(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace qualla {

void SmartMask::completeInit() {
  // Construct a simple erase function that uses either memset() or fill_n
  if (m_clear_value.u32 == 0)  // If the value is 0, use memset directly
    erase = [](void* s, const size_t n) { std::memset(s, 0, n); };
  else if (m_bitwidth == 1)  // For 8-bit values, use memset
    erase = [c = m_clear_value.u8](void* s, const size_t n) { std::memset(s, c, n); };
  else if (m_bitwidth == 2)  // For 16/32-bit values, use fill_n
    erase = [c = m_clear_value.u16](void* s, const size_t n) { std::fill_n((uint16_t*)s, n, c); };
  else if (m_bitwidth == 4)
    erase = [c = m_clear_value.u32](void* s, const size_t n) { std::fill_n((uint32_t*)s, n, c); };
}

void SmartMask::clear(KVTensor& cache) {
  // Assumed check: KVTensor has max bitwidth of 32-bits per element (float32 or uint32)
  assert(m_bitwidth <= 4);

  size_t size = static_cast<size_t>(cache.n_heads * m_embed_dim * m_cur_ctx);
  erase(cache.key_buf, size);
  erase(cache.val_buf, size);

  clearAnchor(cache);
}

void SmartMask::reduceKV(KVTensor& cache,
                         const int32_t variant,
                         const int32_t ctx_size,
                         const std::vector<std::pair<int32_t, size_t>>& clear_idxes) {
  const int32_t past_dim = m_use_scatter ? ctx_size : ctx_size - variant;

  {
    // Key Cache has the axes [n_heads, n_embed, ctx_size]
    // So the operation is repeated for each "row", i.e. n_heads * n_embed iterations
    const int32_t n_iter    = cache.n_heads * m_embed_dim;  // Number of iterations
    const int32_t esize     = m_bitwidth;                   // Size to clear for each iteration
    const int32_t iter_size = past_dim * esize;

    uint8_t* cache_ptr = cache.key_buf;  // input_buffer
    for (int32_t i = 0; i < n_iter; i++) {
      for (const auto& [idx, count] : clear_idxes) erase(cache_ptr + idx * esize, count);
      cache_ptr += iter_size;
    }
  }

  {
    // Value cache has the axes [n_heads, ctx_size, n_embed]
    // So the operation is repeated for each head, but each operation must copy n_embed elements
    const int32_t n_iter    = cache.n_heads;             // Number of iterations
    const int32_t esize     = m_embed_dim * m_bitwidth;  // Size to clear for each iteration
    const int32_t iter_size = past_dim * esize;

    uint8_t* cache_ptr = cache.val_buf;  // input_buffer
    for (int32_t i = 0; i < n_iter; i++) {
      for (const auto& [idx, count] : clear_idxes)
        erase(cache_ptr + idx * esize, count * m_embed_dim);
      cache_ptr += iter_size;
    }
  }
}

void SmartMask::updateKV(KVTensor& cache,
                         const int32_t variant,
                         const int32_t ctx_size,
                         const MoveStrategy& copy_idxes) {
  // Each buffer [ctx_size] is allocated as input[ctx-variant] + output[variant]
  const int32_t past_dim  = m_use_scatter ? ctx_size : ctx_size - variant;
  const int32_t past_size = cache.n_heads * m_embed_dim * past_dim * m_bitwidth;

  {
    // Key Cache has the axes [n_heads, n_embed, ctx_size]
    // So the operation is repeated for each "row", i.e. n_heads * n_embed iterations
    const int32_t esize     = m_bitwidth;  // Size of copy for each iteration
    const int32_t iter_size = past_dim * esize;
    const int32_t out_size  = variant * esize;

    uint8_t* write_ptr = cache.key_buf;              // input_buffer
    uint8_t* read_ptr  = cache.key_buf + past_size;  // output_buffer
    for (int32_t head = 0; head < cache.n_heads; head++) {
      const auto& head_copies = copy_idxes.contains(head) ? copy_idxes.at(head) : copy_idxes.at(-1);
      for (int32_t din = 0; din < m_embed_dim; din++) {
        for (const auto& [src_idx, dst_idx, count] : head_copies)
          std::memcpy((void*)(write_ptr + dst_idx * esize),
                      (const void*)(read_ptr + src_idx * esize),
                      static_cast<size_t>(count * esize));
        write_ptr += iter_size;
        read_ptr += out_size;
      }
    }
  }

  {
    // Value cache has the axes [n_heads, ctx_size, n_embed]
    // So the operation is repeated for each head, but each operation must copy n_embed elements
    const int32_t esize     = m_embed_dim * m_bitwidth;  // Size of copy for each iteration
    const int32_t iter_size = past_dim * esize;
    const int32_t out_size  = variant * esize;

    uint8_t* write_ptr = cache.val_buf;              // input_buffer
    uint8_t* read_ptr  = cache.val_buf + past_size;  // output_buffer

    for (int32_t head = 0; head < cache.n_heads; head++) {
      const auto& head_copies = copy_idxes.contains(head) ? copy_idxes.at(head) : copy_idxes.at(-1);
      for (const auto& [src_idx, dst_idx, count] : head_copies)
        std::memcpy((void*)(write_ptr + dst_idx * esize),
                    (const void*)(read_ptr + src_idx * esize),
                    static_cast<size_t>(count * esize));
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

void SmartMask::moveKV(KVTensor& cache,
                       const int32_t variant,
                       const int32_t ctx_size,
                       const MoveStrategy& move_idxes) {
  // Each buffer [ctx_size] is allocated as input[ctx-variant] + output[variant]
  const int32_t past_dim = m_use_scatter ? ctx_size : ctx_size - variant;

  {
    // Key Cache has the axes [n_heads, n_embed, ctx_size]
    // So the operation is repeated for each "row", i.e. n_heads * n_embed iterations
    const int32_t esize     = m_bitwidth;  // Size of copy for each iteration
    const int32_t iter_size = past_dim * esize;

    uint8_t* cache_ptr = cache.key_buf;
    for (int32_t head = 0; head < cache.n_heads; head++) {
      const auto& head_moves = move_idxes.contains(head) ? move_idxes.at(head) : move_idxes.at(-1);
      for (int32_t din = 0; din < m_embed_dim; din++) {
        for (const auto& [src_idx, dst_idx, count] : head_moves)
          std::memcpy((void*)(cache_ptr + dst_idx * esize),
                      (const void*)(cache_ptr + src_idx * esize),
                      static_cast<size_t>(count * esize));
        cache_ptr += iter_size;
      }
    }
  }

  {
    // Value cache has the axes [n_heads, ctx_size, n_embed]
    // So the operation is repeated for each head, but each operation must copy n_embed elements
    const int32_t esize     = m_embed_dim * m_bitwidth;  // Size of copy for each iteration
    const int32_t iter_size = past_dim * esize;

    uint8_t* cache_ptr = cache.key_buf;
    for (int32_t head = 0; head < cache.n_heads; head++) {
      const auto& head_moves = move_idxes.contains(head) ? move_idxes.at(head) : move_idxes.at(-1);
      for (const auto& [src_idx, dst_idx, count] : head_moves)
        std::memcpy((void*)(cache_ptr + dst_idx * esize),
                    (const void*)(cache_ptr + src_idx * esize),
                    static_cast<size_t>(count * esize));
      cache_ptr += iter_size;
    }
  }
}

void SmartMask::reshapeCache(KVTensor& cache,
                             const int32_t cur_variant,
                             const int32_t cur_ctx,
                             const int32_t new_variant,
                             const int32_t new_ctx) {
  // If using scatter, all AR-n variants have the same shape, so this is a no-op
  if (m_use_scatter && cur_ctx == new_ctx) return;

  // Both key/value are reshaped from a size of [cur_ctx - cur_variant] to [new_ctx - new_variant]
  const size_t in_cache_dim =
      (cur_variant == cur_ctx || m_use_scatter) ? cur_ctx : cur_ctx - cur_variant;
  const size_t out_cache_dim = m_use_scatter ? new_ctx : new_ctx - new_variant;

  {
    // For Key, reshape is done along axis -1
    //      [n_heads, m_embed_dim, in_cache_dim] -> [n_heads, m_embed_dim, out_cache_dim]

    const int32_t n_iter    = cache.n_heads * m_embed_dim;
    const size_t read_size  = in_cache_dim * m_bitwidth;
    const size_t write_size = out_cache_dim * m_bitwidth;

    uint8_t* read_ptr  = cache.key_buf;
    uint8_t* write_ptr = cache.key_buf;
    if (in_cache_dim > out_cache_dim) {
      for (int i = 0; i < n_iter; i++) {
        std::memcpy(write_ptr, read_ptr, write_size);
        read_ptr += read_size;
        write_ptr += write_size;
      }
    } else {
      read_ptr += (n_iter - 1) * read_size;
      write_ptr += (n_iter - 1) * write_size;
      uint8_t* pad_ptr       = write_ptr + read_size;
      const size_t pad_count = out_cache_dim - in_cache_dim;
      for (int i = 0; i < n_iter; i++) {
        if (write_ptr >= read_ptr + read_size || write_ptr + write_size <= read_ptr)
          std::memcpy(write_ptr, read_ptr, read_size);
        else
          std::memmove(write_ptr, read_ptr, read_size);
        erase(pad_ptr, pad_count);
        read_ptr -= read_size;
        write_ptr -= write_size;
        pad_ptr -= write_size;
      }
    }
  }

  {
    // For Value, reshape is done along axis -2
    //      [n_heads, in_cache_dim, m_embed_dim] -> [n_heads, out_cache_dim, m_embed_dim]

    const int32_t n_iter    = cache.n_heads;
    const size_t read_size  = in_cache_dim * m_embed_dim * m_bitwidth;
    const size_t write_size = out_cache_dim * m_embed_dim * m_bitwidth;

    uint8_t* read_ptr  = cache.val_buf;
    uint8_t* write_ptr = cache.val_buf;

    if (in_cache_dim > out_cache_dim) {
      for (int i = 0; i < n_iter; i++) {
        std::memcpy(write_ptr, read_ptr, write_size);
        read_ptr += read_size;
        write_ptr += write_size;
      }
    } else {
      read_ptr += (n_iter - 1) * read_size;
      write_ptr += (n_iter - 1) * write_size;
      uint8_t* pad_ptr       = write_ptr + read_size;
      const size_t pad_count = (out_cache_dim - in_cache_dim) * m_embed_dim;
      for (int i = 0; i < n_iter; i++) {
        if (write_ptr >= read_ptr + read_size || write_ptr + write_size <= read_ptr)
          std::memcpy(write_ptr, read_ptr, read_size);
        else
          std::memmove(write_ptr, read_ptr, read_size);
        erase(pad_ptr, pad_count);
        read_ptr -= read_size;
        write_ptr -= write_size;
        pad_ptr -= write_size;
      }
    }
  }
}

void SmartMask::loadCache(KVTensor& cache,
                          std::ifstream* fs,
                          const bool is_key,
                          const int32_t n_valid,
                          const int32_t n_heads,
                          const int32_t variant,
                          const int32_t ctx_size) {
  const int32_t past_dim = m_use_scatter ? ctx_size : ctx_size - variant;
  if (is_key) {
    const int32_t n_iter    = cache.n_heads * m_embed_dim;
    const int32_t iter_size = past_dim * m_bitwidth;
    const int32_t copy_size = n_valid * m_bitwidth;

    uint8_t* buffer = cache.key_buf;
    for (int i = 0; i < n_iter; i++) {
      fs->read((char*)buffer, copy_size);
      buffer += iter_size;
    }
  } else {
    const int32_t n_iter    = cache.n_heads;
    const int32_t iter_size = past_dim * m_embed_dim * m_bitwidth;
    const int32_t copy_size = n_valid * m_embed_dim * m_bitwidth;

    uint8_t* buffer = cache.val_buf;
    for (int i = 0; i < n_iter; i++) {
      fs->read((char*)buffer, copy_size);
      buffer += iter_size;
    }
  }

  fs->seekg((n_heads - cache.n_heads) * m_embed_dim * n_valid * m_bitwidth, std::ios::cur);
}

void SmartMask::dumpCache(KVTensor& cache,
                          std::ofstream* fs,
                          const bool is_key,
                          const int32_t n_valid,
                          const int32_t n_heads,
                          const int32_t variant,
                          const int32_t ctx_size) {
  const int32_t past_dim = m_use_scatter ? ctx_size : ctx_size - variant;
  if (is_key) {
    const int32_t n_iter    = cache.n_heads * m_embed_dim;
    const int32_t iter_size = past_dim * m_bitwidth;
    const int32_t copy_size = n_valid * m_bitwidth;

    uint8_t* buffer = cache.key_buf;
    for (int i = 0; i < n_iter; i++) {
      fs->write((char*)buffer, copy_size);
      buffer += iter_size;
    }

  } else {
    const int32_t n_iter    = cache.n_heads;
    const int32_t iter_size = past_dim * m_embed_dim * m_bitwidth;
    const int32_t copy_size = n_valid * m_embed_dim * m_bitwidth;

    uint8_t* buffer = cache.val_buf;
    for (int i = 0; i < n_iter; i++) {
      fs->write((char*)buffer, copy_size);
      buffer += iter_size;
    }
  }

  fs->seekp((n_heads - cache.n_heads) * m_embed_dim * n_valid * m_bitwidth, std::ios::cur);
}

void SmartMask::dumpCache(KVTensor& cache,
                          Buffer* kv_buff,
                          const bool is_key,
                          const int32_t n_valid,
                          const int32_t n_heads,
                          const int32_t variant,
                          const int32_t ctx_size) {
  const int32_t past_dim = m_use_scatter ? ctx_size : ctx_size - variant;
  if (is_key) {
    const int32_t n_iter    = cache.n_heads * m_embed_dim;
    const int32_t iter_size = past_dim * m_bitwidth;
    const int32_t copy_size = n_valid * m_bitwidth;

    uint8_t* buffer = cache.key_buf;
    for (int i = 0; i < n_iter; i++) {
      kv_buff->appendBuffer((uint8_t*)buffer, copy_size);
      buffer += iter_size;
    }

  } else {
    const int32_t n_iter    = cache.n_heads;
    const int32_t iter_size = past_dim * m_embed_dim * m_bitwidth;
    const int32_t copy_size = n_valid * m_embed_dim * m_bitwidth;

    uint8_t* buffer = cache.val_buf;
    for (int i = 0; i < n_iter; i++) {
      kv_buff->appendBuffer((uint8_t*)buffer, copy_size);
      buffer += iter_size;
    }
  }
  kv_buff->setPosFromCurr((n_heads - cache.n_heads) * m_embed_dim * n_valid * m_bitwidth);
}

void SmartMask::dumpHead(KVTensor& cache,
                         uint32_t head,
                         const int32_t n_valid,
                         const int32_t variant,
                         const int32_t ctx_size,
                         void* data) {
  if (head > cache.n_heads) {
    memset(data, 128, 2 * m_embed_dim * n_valid * m_bitwidth);
    return;
  }

  const int32_t past_dim = m_use_scatter ? ctx_size : ctx_size - variant;

  const int32_t key_head_iter_size = m_embed_dim * past_dim * m_bitwidth;
  uint8_t* read_buf                = cache.key_buf + head * key_head_iter_size;
  uint8_t* write_buf               = (uint8_t*)data;

  for (int i = 0; i < m_embed_dim; i++) {
    for (int j = 0; j < n_valid; j++) {
      write_buf[j * m_embed_dim + i] = read_buf[i * past_dim + j];
    }
  }

  write_buf += (m_embed_dim * n_valid * m_bitwidth);
  const int32_t value_head_iter_size = past_dim * m_embed_dim * m_bitwidth;
  read_buf                           = cache.val_buf + head * value_head_iter_size;
  memcpy(write_buf, read_buf, (n_valid * m_embed_dim * m_bitwidth));
}

}  // namespace qualla
