//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "kvmanager.hpp"

namespace qualla {

class NativeKV : public KVManager {
 private:
  // These are determined by the QNN compiler, and is the size of the tile
  static const int32_t K_TILE = 256;
  static const int32_t V_TILE = 64;

 protected:
  int32_t getIndexForNewKV(InferenceStep& step) override;

  void completeInit() override;

  void clear(KVTensor& cache) override;

  void updateKV(KVTensor& cache,
                const int32_t variant,
                const int32_t ctx_size,
                const MoveStrategy& copy_idxes) override;

  void reduceKV(KVTensor& cache,
                const int32_t variant,
                const int32_t ctx_size,
                const std::vector<std::pair<int32_t, size_t>>& clear_idxes) override;

  void moveKV(KVTensor& cache,
              const int32_t variant,
              const int32_t ctx_size,
              const MoveStrategy& move_idxes) override;

  void reshapeCache(KVTensor& cache,
                    const int32_t cur_variant,
                    const int32_t cur_ctx,
                    const int32_t new_variant,
                    const int32_t new_ctx) override;

  void loadCache(KVTensor& cache,
                 std::ifstream* fs,
                 const bool is_key,
                 const int32_t n_valid,
                 const int32_t n_heads,
                 const int32_t variant,
                 const int32_t ctx_size) override;

  void dumpCache(KVTensor& cache,
                 std::ofstream* fs,
                 const bool is_key,
                 const int32_t n_valid,
                 const int32_t n_heads,
                 const int32_t variant,
                 const int32_t ctx_size) override;

  void dumpCache(KVTensor& cache,
                 Buffer* kv_buff,
                 const bool is_key,
                 const int32_t n_valid,
                 const int32_t n_heads,
                 const int32_t variant,
                 const int32_t ctx_size) override;

  void dumpHead(KVTensor& cache,
                uint32_t head,
                const int32_t n_valid,
                const int32_t variant,
                const int32_t ctx_size,
                void* data) override;

 public:
  NativeKV(Env& env, size_t nThreads, uint64_t cpuMask, bool poll)
      : KVManager(env, nThreads, cpuMask, poll) {}
  ~NativeKV() {}
};

}  // namespace qualla