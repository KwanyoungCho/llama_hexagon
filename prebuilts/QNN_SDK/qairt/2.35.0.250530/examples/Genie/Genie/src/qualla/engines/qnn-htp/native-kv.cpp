//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fstream>

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "native-kv.hpp"

#define __DEBUG(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace qualla {

void NativeKV::completeInit() {
  // Internally, HMX does not apply an offset for NativeKV tensor
  // This means we do not set empty values to 128, but rather 0
  m_clear_value.u32 = 0x0;

  if (m_bitwidth != 1 || m_quantized != true) State::error("Native KV only supports uint8.");

  for (const auto& [ctx_size, variants] : m_supported_variants) {
    if (ctx_size % 32 != 0) State::error("NativeKV expects AR/CL to be multiples of 32");
    for (const auto& variant : variants) {
      if (variant % 32 != 0) State::error("NativeKV expects AR/CL to be multiples of 32");
    }
  }
}

int32_t NativeKV::getIndexForNewKV(InferenceStep& step) {
  return static_cast<int32_t>(std::ceil(static_cast<double>(step.n_valid_kv) / 32.0f) * 32);
}

void NativeKV::clear(KVTensor& cache) {
  const size_t cache_size =
      static_cast<size_t>(cache.n_heads * m_embed_dim * m_cur_ctx * m_bitwidth);
  std::memset(cache.key_buf, 0, cache_size);
  std::memset(cache.val_buf, 0, cache_size);

  clearAnchor(cache);
}

// Translates a flat index to an offset for the HMX weight format buffer
// Convert [din, dout] -> [dout/N_TILE, din/32, (dout%N_TILE)/32, [(din%32)/4, dout%32, din%4]]
//
// For Key$ (head, din=embed, dout=ctx_size) and K_TILE=256
// (head, tile=dout/K_TILE,  din: din/32, dout: K_TILE/32, din:8, dout:32, din:4)
//
// For Value$ (head, din=ctx_size, dout=embed) and V_TILE=64
// (head, tile=dout/V_TILE,  din: din/32, dout: V_TILE/32, din:8, dout:32, din:4)
static inline int32_t fromFlatOffset(const int32_t DIN,
                                     const int32_t DOUT,
                                     const int32_t N_TILE,
                                     const int32_t din,
                                     const int32_t dout) {
  assert(DIN % 32 == 0);
  assert(DOUT % 32 == 0);

  // Each tensor then gets tiled into chunks of min(dout, N_TILE)
  const int32_t tile_size   = std::min(DOUT, N_TILE);  // head * tile * [N_EMBED, N_TILE or DOUT]
  const int32_t tile_stride = DIN * tile_size;         // head * tile * [N_EMBED, tile_size]

  // Split the dout into [dout // NTILE, (dout % NTILE) // 32 , (dout % tile_size) % 32]
  const int32_t tile_idx = dout / tile_size;
  const int32_t dout_0   = (dout % tile_size) >> 5;  // (dout % tile_size) / 32
  const int32_t dout_1   = dout & 0x1f;              // From (dout % tile_size) % 32 = dout % 32;

  // Split the din into [din // 32, (din % 32) // 4, (din % 32) % 4]
  const int32_t din_0 = din >> 5;           // From din / 32;
  const int32_t din_1 = (din & 0x1f) >> 2;  // From (din % 32) / 4;
  const int32_t din_2 = din & 0x3;          // From (din % 32) % 4 = din % 4

  // Strides for the chunk of (8:DIN, 32:tile_size, 4:N_EMBED). This is always constant
  static const int32_t bitshift[3] = {10, 7, 2};

  // Stride for each tile * chunk. This equals (tile_size/32) * (8*32*4). Note tile_size%32==0
  const int32_t din_0_stride = tile_size << 5;  // tile_shift * 32;

  // Construct the final flat offset as [head, tile_idx, din_0, dout_0, (din_1, dout_1, din_2)]
  return tile_idx * tile_stride + din_0 * din_0_stride +
         (dout_0 << bitshift[0] | (din_1 << bitshift[1]) | (dout_1 << bitshift[2]) | din_2);
}

// Reduce KV$ - remove entries from the cache buffer
void NativeKV::reduceKV(KVTensor& cache,
                        const int32_t variant,
                        const int32_t ctx_size,
                        const std::vector<std::pair<int32_t, size_t>>& clear_idxes) {
  const int32_t head_stride = m_embed_dim * ctx_size * m_bitwidth;
  {
    uint8_t* cache_ptr = cache.key_buf;
    for (int32_t head = 0; head < cache.n_heads; head++) {
      uint8_t* head_ptr = cache_ptr + head * head_stride;
      for (int32_t din = 0; din < m_embed_dim; din++)
        for (const auto& [idx, count] : clear_idxes)
          for (int i = 0; i < count; i++)
            head_ptr[fromFlatOffset(m_embed_dim, ctx_size, K_TILE, din, idx + i)] = 0;
    }
  }

  {
    uint8_t* cache_ptr = cache.val_buf;
    for (int32_t head = 0; head < cache.n_heads; head++) {
      uint8_t* head_ptr = cache_ptr + head * head_stride;
      for (const auto& [idx, count] : clear_idxes)
        for (int i = 0; i < count; i++)
          for (int32_t dout = 0; dout < m_embed_dim; dout++)
            head_ptr[fromFlatOffset(ctx_size, m_embed_dim, V_TILE, idx + i, dout)] = 0;
    }
  }
}

// Update KV$ - copy entries from output buffer into the cache buffer
void NativeKV::updateKV(KVTensor& cache,
                        const int32_t variant,
                        const int32_t ctx_size,
                        const MoveStrategy& copy_idxes) {
  // Each buffer [ctx_size] is allocated as input[ctx-variant] + output[variant]
  const int32_t head_stride_in  = m_embed_dim * ctx_size * m_bitwidth;
  const int32_t head_stride_out = m_embed_dim * variant * m_bitwidth;
  const int32_t cache_size      = cache.n_heads * head_stride_in;

  {
    // Update Key Buffer
    uint8_t* dst_ptr = cache.key_buf;               // input_buffer
    uint8_t* src_ptr = cache.key_buf + cache_size;  // output_buffer

    for (int32_t head = 0; head < cache.n_heads; head++) {
      const auto& head_copies = copy_idxes.contains(head) ? copy_idxes.at(head) : copy_idxes.at(-1);

      uint8_t* head_src_ptr = src_ptr + head * head_stride_out;
      uint8_t* head_dst_ptr = dst_ptr + head * head_stride_in;

      for (int32_t din = 0; din < m_embed_dim; din++) {
        for (const auto& [src_idx, dst_idx, count] : head_copies) {
          for (int i = 0; i < count; i++) {
            const int32_t src_offset =
                fromFlatOffset(m_embed_dim, variant, K_TILE, din, src_idx + i);
            const int32_t dst_offset =
                fromFlatOffset(m_embed_dim, ctx_size, K_TILE, din, dst_idx + i);
            head_dst_ptr[dst_offset] = head_src_ptr[src_offset];
          }
        }
      }
    }
  }

  {
    // Update Value Buffer
    uint8_t* dst_ptr = cache.val_buf;               // input_buffer
    uint8_t* src_ptr = cache.val_buf + cache_size;  // output_buffer

    for (int32_t head = 0; head < cache.n_heads; head++) {
      const auto& head_copies = copy_idxes.contains(head) ? copy_idxes.at(head) : copy_idxes.at(-1);

      uint8_t* head_src_ptr = src_ptr + head * head_stride_out;
      uint8_t* head_dst_ptr = dst_ptr + head * head_stride_in;

      for (const auto& [src_idx, dst_idx, count] : head_copies) {
        for (int i = 0; i < count; i++) {
          for (int32_t dout = 0; dout < m_embed_dim; dout++) {
            const int32_t src_offset =
                fromFlatOffset(variant, m_embed_dim, V_TILE, src_idx + i, dout);
            const int32_t dst_offset =
                fromFlatOffset(ctx_size, m_embed_dim, V_TILE, dst_idx + i, dout);
            head_dst_ptr[dst_offset] = head_src_ptr[src_offset];
          }
        }
      }
    }
  }
}

// Move KV$ - move entries within the cache buffer
void NativeKV::moveKV(KVTensor& cache,
                      const int32_t variant,
                      const int32_t ctx_size,
                      const MoveStrategy& move_idxes) {
  // Each buffer [ctx_size] is allocated as input[ctx-variant] + output[variant]
  const int32_t head_stride_in = m_embed_dim * ctx_size * m_bitwidth;

  {
    // Update Key Buffer
    uint8_t* cache_ptr = cache.key_buf;  // input_buffer
    for (int32_t head = 0; head < cache.n_heads; head++) {
      const auto& head_moves = move_idxes.contains(head) ? move_idxes.at(head) : move_idxes.at(-1);

      uint8_t* head_ptr = cache_ptr + head * head_stride_in;
      for (int32_t din = 0; din < m_embed_dim; din++) {
        for (const auto& [src_idx, dst_idx, count] : head_moves) {
          for (int i = 0; i < count; i++) {
            const int32_t src_offset =
                fromFlatOffset(m_embed_dim, ctx_size, K_TILE, din, src_idx + i);
            const int32_t dst_offset =
                fromFlatOffset(m_embed_dim, ctx_size, K_TILE, din, dst_idx + i);
            head_ptr[dst_offset] = head_ptr[src_offset];
          }
        }
      }
    }
  }

  {
    // Update Value Buffer
    uint8_t* cache_ptr = cache.key_buf;  // input_buffer
    for (int32_t head = 0; head < cache.n_heads; head++) {
      const auto& head_moves = move_idxes.contains(head) ? move_idxes.at(head) : move_idxes.at(-1);

      uint8_t* head_ptr = cache_ptr + head * head_stride_in;
      for (const auto& [src_idx, dst_idx, count] : head_moves) {
        for (int i = 0; i < count; i++) {
          for (int32_t dout = 0; dout < m_embed_dim; dout++) {
            const int32_t src_offset =
                fromFlatOffset(ctx_size, m_embed_dim, V_TILE, src_idx + i, dout);
            const int32_t dst_offset =
                fromFlatOffset(ctx_size, m_embed_dim, V_TILE, dst_idx + i, dout);
            head_ptr[dst_offset] = head_ptr[src_offset];
          }
        }
      }
    }
  }
}

void NativeKV::reshapeCache(KVTensor& cache,
                            const int32_t cur_variant,
                            const int32_t cur_ctx,
                            const int32_t new_variant,
                            const int32_t new_ctx) {
  // All AR-n variants have the same shape, so this is a no-op for NativeKV
  if (new_ctx == cur_ctx) return;

  {
    // For Key cache, DIN=m_embed_dim and DOUT=ctx_size
    // cur_ctx -> (head, cur_ctx/K_TILE,  din: embed/32, dout: K_TILE/32, din:8, dout:32, din:4)
    // new_ctx -> (head, new_ctx/K_TILE,  din: embed/32, dout: K_TILE/32, din:8, dout:32, din:4)
    //
    // This translates to copying (embed/32)*(K_TILE/32)*(8*32*4) elements over head iterations
    const int32_t n_iter = cache.n_heads;         // Iterate for each head
    const int32_t stride = m_embed_dim * K_TILE;  // Stride for each KV$ index

    const int32_t read_size  = cur_ctx / K_TILE * stride;  // Size of a read for each iteration
    const int32_t write_size = new_ctx / K_TILE * stride;  // Size of a write for each iteartion

    if (cur_ctx > new_ctx) {
      // Context size decreases in size. Guarantees no memory overlap, and no padding required
      uint8_t* read_ptr  = cache.key_buf;
      uint8_t* write_ptr = cache.key_buf;
      for (int i = 0; i < n_iter; i++) {
        std::memcpy(write_ptr, read_ptr, write_size);
        read_ptr += read_size;
        write_ptr += write_size;
      }
    } else {
      // Context size decreases in size. Read/write must start from the last index backwards
      // This is to avoid overwriting memory before these are copied correctly
      uint8_t* read_ptr  = cache.key_buf + (n_iter - 1) * read_size;
      uint8_t* write_ptr = cache.key_buf + (n_iter - 1) * write_size;

      // The remaining elements will have to be padded, i.e. set to 0
      uint8_t* pad_ptr       = write_ptr + read_size;   // Start padding after cur_ctx
      const int32_t pad_size = write_size - read_size;  // Pad upto new_ctx

      for (int i = 0; i < n_iter; i++) {
        if (write_ptr >= read_ptr + read_size || write_ptr + write_size <= read_ptr)
          std::memcpy(write_ptr, read_ptr, read_size);
        else
          std::memmove(write_ptr, read_ptr, read_size);
        std::memset(pad_ptr, 0, pad_size);
        read_ptr -= read_size;
        write_ptr -= write_size;
        pad_ptr -= write_size;
      }
    }
  }

  {
    // For Value cache, DIN=ctx_size, DOUT=m_embed_dim
    // cur_ctx -> (head, embed/V_TILE,  din: cur_ctx/32, dout: V_TILE/32, din:8, dout:32, din:4)
    // new_ctx -> (head, embed/V_TILE,  din: new_ctx/32, dout: V_TILE/32, din:8, dout:32, din:4)
    //
    // This translates to copying (V_TILE/32)*8*32*4 elements over head*(embed/V_TILE) iterations
    const int32_t n_iter = cache.n_heads * (m_embed_dim / V_TILE);  // #heads * #tiles
    const int32_t stride = V_TILE * 32 * m_bitwidth;                // Stride for each KV$ index

    const int32_t read_size  = cur_ctx / 32 * stride;  // Size of a read for each iteration
    const int32_t write_size = new_ctx / 32 * stride;  // Size of a write for each iteration

    if (cur_ctx > new_ctx) {
      // Context size decreases in size. Guarantees no memory overlap, and no padding required
      uint8_t* read_ptr  = cache.val_buf;
      uint8_t* write_ptr = cache.val_buf;
      for (int i = 0; i < n_iter; i++) {
        std::memcpy(write_ptr, read_ptr, write_size);
        read_ptr += read_size;
        write_ptr += write_size;
      }
    } else {
      // Context size decreases in size. Read/write must start from the last index backwards
      // This is to avoid overwriting memory before these are copied correctly
      uint8_t* read_ptr  = cache.val_buf + (n_iter - 1) * read_size;
      uint8_t* write_ptr = cache.val_buf + (n_iter - 1) * write_size;

      // The remaining elements will have to be padded, i.e. set to 0
      uint8_t* pad_ptr       = write_ptr + read_size;   // Start padding after cur_ctx
      const int32_t pad_size = write_size - read_size;  // Pad upto new_ctx

      for (int i = 0; i < n_iter; i++) {
        if (write_ptr >= read_ptr + read_size || write_ptr + write_size <= read_ptr)
          std::memcpy(write_ptr, read_ptr, read_size);
        else
          std::memmove(write_ptr, read_ptr, read_size);
        std::memset(pad_ptr, 0, pad_size);
        read_ptr -= read_size;
        write_ptr -= write_size;
        pad_ptr -= write_size;
      }
    }
  }
}

void NativeKV::loadCache(KVTensor& cache,
                         std::ifstream* fs,
                         const bool is_key,
                         const int32_t n_valid,
                         const int32_t n_heads,
                         const int32_t variant,
                         const int32_t ctx_size) {
  if (m_bitwidth != 1 || m_quantized != true) State::error("Native KV only supports 8-bit KV$");

  const int32_t head_stride = m_embed_dim * ctx_size * m_bitwidth;

  // Create a scratch buffer to help minimize IO calls, and allow for post-processing (uint8->int8)
  std::vector<char> scratch(m_embed_dim * n_valid * m_bitwidth);

  for (int32_t head = 0; head < cache.n_heads; head++) {
    fs->read(scratch.data(), scratch.size());  // Batch fs->read() call for this head
    for (auto& ch : scratch) ch -= 128;        // Convert uint8 -> int8

    char* scratch_ptr = scratch.data();
    if (is_key) {
      char* head_ptr = (char*)cache.key_buf + head * head_stride;
      for (int32_t din = 0; din < m_embed_dim; din++)
        for (int i = 0; i < n_valid; i++)
          head_ptr[fromFlatOffset(m_embed_dim, ctx_size, K_TILE, din, i)] = *scratch_ptr++;
    } else {
      char* head_ptr = (char*)cache.val_buf + head * head_stride;
      for (int i = 0; i < n_valid; i++)
        for (int32_t dout = 0; dout < m_embed_dim; dout++)
          head_ptr[fromFlatOffset(ctx_size, m_embed_dim, V_TILE, i, dout)] = *scratch_ptr++;
    }
  }

  fs->seekg((n_heads - cache.n_heads) * m_embed_dim * n_valid * m_bitwidth, std::ios::cur);
}

void NativeKV::dumpHead(KVTensor& cache,
                        uint32_t head,
                        const int32_t n_valid,
                        const int32_t variant,
                        const int32_t ctx_size,
                        void* data) {
  if (m_bitwidth != 1 || m_quantized != true) State::error("Native KV only supports 8-bit KV$");
  const int32_t head_stride = m_embed_dim * ctx_size * m_bitwidth;

  if (head > cache.n_heads) {
    memset(data, 128, 2 * m_embed_dim * n_valid * m_bitwidth);
    return;
  }

  char* scratch_ptr = (char*)data;
  char* head_ptr    = (char*)cache.key_buf + head * head_stride;
  for (int i = 0; i < n_valid; i++)
    for (int32_t din = 0; din < m_embed_dim; din++)
      *scratch_ptr++ = head_ptr[fromFlatOffset(m_embed_dim, ctx_size, K_TILE, din, i)];

  head_ptr = (char*)cache.val_buf + head * head_stride;
  for (int i = 0; i < n_valid; i++)
    for (int32_t dout = 0; dout < m_embed_dim; dout++)
      *scratch_ptr++ = head_ptr[fromFlatOffset(ctx_size, m_embed_dim, V_TILE, i, dout)];

  for (size_t i = 0; i < (2 * m_embed_dim * n_valid * m_bitwidth); i++) ((char*)data)[i] += 128;
}

void NativeKV::dumpCache(KVTensor& cache,
                         std::ofstream* fs,
                         const bool is_key,
                         const int32_t n_valid,
                         const int32_t n_heads,
                         const int32_t variant,
                         const int32_t ctx_size) {
  if (m_bitwidth != 1 || m_quantized != true) State::error("Native KV only supports 8-bit KV$");
  const int32_t head_stride = m_embed_dim * ctx_size * m_bitwidth;

  // Create a scratch buffer to help minimize IO calls, and allow for post-processing (uint8->int8)
  std::vector<char> scratch(m_embed_dim * n_valid * m_bitwidth);

  for (int32_t head = 0; head < cache.n_heads; head++) {
    char* scratch_ptr = scratch.data();
    if (is_key) {
      char* head_ptr = (char*)cache.key_buf + head * head_stride;
      for (int32_t din = 0; din < m_embed_dim; din++)
        for (int i = 0; i < n_valid; i++)
          *scratch_ptr++ = head_ptr[fromFlatOffset(m_embed_dim, ctx_size, K_TILE, din, i)];
    } else {
      char* head_ptr = (char*)cache.val_buf + head * head_stride;
      for (int i = 0; i < n_valid; i++)
        for (int32_t dout = 0; dout < m_embed_dim; dout++)
          *scratch_ptr++ = head_ptr[fromFlatOffset(ctx_size, m_embed_dim, V_TILE, i, dout)];
    }

    for (auto& ch : scratch) ch += 128;
    fs->write(scratch.data(), scratch.size());
  }

  fs->seekp((n_heads - cache.n_heads) * m_embed_dim * n_valid * m_bitwidth, std::ios::cur);
}

void NativeKV::dumpCache(KVTensor& cache,
                         Buffer* kv_buff,
                         const bool is_key,
                         const int32_t n_valid,
                         const int32_t n_heads,
                         const int32_t variant,
                         const int32_t ctx_size) {
  if (m_bitwidth != 1 || m_quantized != true) State::error("Native KV only supports 8-bit KV$");
  const int32_t head_stride = m_embed_dim * ctx_size * m_bitwidth;

  // Create a scratch buffer to help minimize IO calls, and allow for post-processing (uint8->int8)
  std::vector<char> scratch(m_embed_dim * n_valid * m_bitwidth);

  for (int32_t head = 0; head < cache.n_heads; head++) {
    char* scratch_ptr = scratch.data();
    if (is_key) {
      char* head_ptr = (char*)cache.key_buf + head * head_stride;
      for (int32_t din = 0; din < m_embed_dim; din++)
        for (int i = 0; i < n_valid; i++)
          *scratch_ptr++ = head_ptr[fromFlatOffset(m_embed_dim, ctx_size, K_TILE, din, i)];
    } else {
      char* head_ptr = (char*)cache.val_buf + head * head_stride;
      for (int i = 0; i < n_valid; i++)
        for (int32_t dout = 0; dout < m_embed_dim; dout++)
          *scratch_ptr++ = head_ptr[fromFlatOffset(ctx_size, m_embed_dim, V_TILE, i, dout)];
    }

    for (auto& ch : scratch) ch += 128;
    kv_buff->appendBuffer((uint8_t*)scratch.data(), scratch.size());
  }

  kv_buff->setPosFromCurr((n_heads - cache.n_heads) * m_embed_dim * n_valid * m_bitwidth);
}
}  // namespace qualla
