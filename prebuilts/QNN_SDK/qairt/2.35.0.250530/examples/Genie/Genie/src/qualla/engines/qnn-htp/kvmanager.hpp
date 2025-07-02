//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <set>

#include "QnnApi.hpp"
#include "nsp-params.hpp"
#include "qualla/detail/buffer.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/threadpool.hpp"
#include "qualla/env.hpp"

namespace qualla {

inline std::string getManagerModeStr(KVManagerMode mode) {
  if (mode == POINTER_SHIFT) return "POINTER_SHIFT";
  if (mode == SHIFT_CONCAT) return "SHIFT_CONCAT";
  if (mode == SMART_MASK) return "SMART_MASK";
  if (mode == NATIVE_KV) return "NATIVE_KV";
  return "ERROR: KVManagerMode not found";
}

using VariantSpec = std::pair<int32_t, int32_t>;

// Inference Step is a simple struct defining all variables necessary to execute graph iteration
struct InferenceStep {
  InferenceStep() {}
  InferenceStep(int32_t variant,
                int32_t ctx_size,
                int32_t n_past,
                int32_t n_valid_kv,
                int32_t n_process,
                int32_t past_idx,
                int32_t new_idx)
      : variant(variant),
        ctx_size(ctx_size),
        n_past(n_past),
        n_valid_kv(n_valid_kv),
        n_process(n_process),
        past_idx(past_idx),
        new_idx(new_idx) {}
  int32_t variant{0};
  int32_t ctx_size{0};
  int32_t n_past{0};
  int32_t n_valid_kv{0};
  int32_t n_process{0};
  int32_t past_idx{0};
  int32_t new_idx{0};

  std::string str() const;
};

// InferenceStrategy is an alias defined as a list of InfereceStep
using InferenceStrategy = std::vector<InferenceStep>;

// Alias selection mask for readability
using Mask = const std::vector<bool>;

// The KVManager uses Scope to easily specify whether a KV$
// operation should apply to all graphs or a specific graph.
struct Scope {
  enum ScopeType : uint8_t {
    GLOBAL,    // [Default] Apply the operation to ALL KV$ tensors
    PER_GRAPH  // Apply the operation to one graph (by index)
  };

  ScopeType scope{GLOBAL};
  int32_t graph_idx{-1};

  static Scope global() { return Scope(Scope::GLOBAL, -1); }
  static Scope per_graph(int32_t graph_idx) { return {Scope::PER_GRAPH, graph_idx}; }

  bool is_global() { return scope == Scope::GLOBAL; }
  bool is_per_graph() { return scope == Scope::PER_GRAPH; }

 private:
  // Private constructor. Scope must be constructed using the static methods
  Scope(ScopeType _scope, int32_t _graph_idx) : scope(_scope), graph_idx(_graph_idx) {}
};

struct KVTensor {
  uint32_t idx;
  uint8_t* key_buf;  // Pointer to the Key Cache
  uint8_t* val_buf;  // Pointer to the Value Cache
  int32_t n_heads;

  QuantParam key_quant;
  QuantParam value_quant;  // Quantization prameters for keys and values

  // Fields for the KeyDiff algorithm
  uint16_t anchor_offset;
  uint8_t *anchor_in, *anchor_out;
  uint8_t* scores;
  std::vector<std::queue<int32_t>> evict_idxes;  // Indexes to evict for each head

  KVTensor(){};
  KVTensor(uint32_t idx,
           uint8_t* key,
           uint8_t* value,
           int32_t n_heads,
           QuantParam key_quant,
           QuantParam value_quant,
           uint16_t anchor_offset,
           uint8_t* anchor_in,
           uint8_t* anchor_out,
           uint8_t* scores)
      : idx(idx),
        key_buf(key),
        val_buf(value),
        n_heads(n_heads),
        key_quant(key_quant),
        value_quant(value_quant),
        anchor_offset(anchor_offset),
        anchor_in(anchor_in),
        anchor_out(anchor_out),
        scores(scores) {
    evict_idxes.resize(n_heads);
  }
};

// Each KVTensor is independent to all other KVTensors. A Job function operates on one KVTensor
struct Job {
  std::string name;
  std::function<void(KVTensor&)> update_function;
  // Maybe add an estimated "cost" here
};

// Jobs in a JobSlice queue must be run sequentially, but are
// always independent of the Jobs in another JobSlice's queue.
//
// The main thread requests KV$ updates by adding jobs to
// JobSlice::queued. Worker threads will attempt to lock
// queuedMutex, then move the queued jobs to the running queue,
// then unlock queuedMutex. This allows the main thread to continue
// queueing work which will be picked up by a subsequent iteration
// of the worker thread. Meanwhile, the worker thread can flush
// the running queue.
struct JobSlice {
  std::mutex queuedMutex;
  std::mutex runningMutex;
  std::queue<std::function<void()>> queued;
  std::queue<std::function<void()>> running;
};

// KV$ Move operations can be defined as a set of MoveSteps
// size KV$ entries are copied from a src index to a dst index
struct MoveStep {
  int32_t src_idx;
  int32_t dst_idx;
  size_t count;
};

// A MoveStrategy maps the head index to a list of MoveSteps
// Each head can have its own MoveStep list, or a catch-all MoveStep using head_idx=-1
using MoveStrategy = std::map<int32_t, std::vector<MoveStep>>;

class KVManager : public State {
 protected:
  Env& _env;  // Reference to global environment for logging

  std::unique_ptr<ThreadPool> m_threadpool;  // Threadpool for async background processing
  QnnApi* m_qnn_api{nullptr};
  // KVManager parameters. These are fixed and assumed not to change
  // bool strict_mode; // Manually clear out the buffer
  int8_t m_bitwidth{1};       // The size of each element (in bytes)
  int32_t m_embed_dim{0};     // Embedding size for KV$
  int32_t m_max_ctx_size{0};  // Maximum context length
  bool m_quantized{true};     // Is the KV$ quantized?
  bool m_use_scatter{true};   // Does the model use Scatter(new->past) or Concat (past+new)
  union {
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
  } m_clear_value;                // Value used for clearing the cache
  int32_t m_anchor_bitwidth = 0;  // Size of the anchor tensor (in bytes)

  std::vector<int32_t> m_graphs;                     // List of graph indexes
  std::map<int32_t, std::vector<KVTensor>> m_cache;  // Maps graph index to KV$ tensors
  std::map<int32_t, std::set<int32_t>> m_supported_variants;

  // Requirements
  // Jobs must be Scopeable
  // Jobs must support COPY, CLEAR, and RESHAPE
  // Jobs must be splittable (i.e. one Job needs to be done across multiple threads)
  // [Optional] Jobs must be trackable (i.e. a record of all jobs done for logging)

  struct GraphState {
    std::atomic_int sync;

    // Maintain a separate job queue for independent "slices" of KV$ updates.
    // There will be a total of n_threads slices.
    std::vector<std::unique_ptr<JobSlice>> jobSlices;
  };
  std::map<int32_t, GraphState> m_graph_state;

  // Splits Job into slices that can be run in parallel,
  // then asks the background threads to execute the slices.
  void prepareJob(Scope scope, Job job);
  // Requests background threads to check for available jobs on the given graph.
  void queueJob(int32_t graph_idx);

  int32_t m_counter{0};       // Ticket counter for KV$ updates
  int32_t m_n_past{0};        // Total number of "virtual" KV$ tensors
  int32_t m_n_valid_kv{0};    // Total number of "physical" KV$ tensors (i.e. actual KV$ in memory)
  int32_t m_cur_variant{-1};  // Current variant
  int32_t m_cur_ctx{-1};      // Current context length

  bool m_islastStep{false};  // true only for the last inference step.

  // Ideas: Keep track of whether last inference has already been processed
  InferenceStrategy m_strategy;
  int32_t m_strategy_cur_step{0};
  bool m_strategy_active{false};
  InferenceStep m_last_inference;  // Keep track of the last known inference

  // Define variables for long context
  LongContextParams m_longcontext;

  // processUpdate consumes the last known inference (m_last_inference) to generate update jobs
  // CAUTION: m_last_inference is destroyed by this function, since KV$ can only be consumed once
  bool processUpdate(
      Scope scope, InferenceStep& step, int32_t n_past, bool switch_variant, Mask& mask = {});

  // Virtual function to allow subclasses to setup internal variables after init completes
  virtual void completeInit() {}

  // Clear the cache completely
  virtual void clear(KVTensor& cache) = 0;

  // Get the index for the starting past KV$ and the new KV$
  virtual int32_t getIndexForPastKV(InferenceStep& step) { return 0; }
  virtual int32_t getIndexForNewKV(InferenceStep& step) { return step.ctx_size - step.variant; }

  // Update KV$ - copy entries from output buffer into the cache buffer
  virtual void updateKV(KVTensor& cache,
                        const int32_t variant,
                        const int32_t ctx_size,
                        const MoveStrategy& copy_idxes) = 0;

  // Reduce KV$ - remove entries from the cache buffer
  virtual void reduceKV(KVTensor& cache,
                        const int32_t variant,
                        const int32_t ctx_size,
                        const std::vector<std::pair<int32_t, size_t>>& clear_idxes) = 0;

  // Move KV$ - move entries within the cache buffer
  virtual void moveKV(KVTensor& cache,
                      const int32_t variant,
                      const int32_t ctx_size,
                      const MoveStrategy& move_idxes) = 0;

  // Reshape KV$ - convert AR-{cur_variant} CL-{cur_ctx} cache into AR-{new_variant} CL-{new_ctx}
  virtual void reshapeCache(KVTensor& cache,
                            const int32_t cur_variant,
                            const int32_t cur_ctx,
                            const int32_t new_variant,
                            const int32_t new_ctx) = 0;

  // Load KV$ - read KV$ from a flat file buffer into the cache buffer
  virtual void loadCache(KVTensor& cache,
                         std::ifstream* fs,
                         const bool is_key,
                         const int32_t n_valid,
                         const int32_t n_heads,
                         const int32_t variant,
                         const int32_t ctx_size) = 0;

  // Dump KV$ - write KV$ from the cache buffer into a flat file buffer
  virtual void dumpCache(KVTensor& cache,
                         std::ofstream* fs,
                         const bool is_key,
                         const int32_t n_valid,
                         const int32_t n_heads,
                         const int32_t variant,
                         const int32_t ctx_size) = 0;

  virtual void dumpCache(KVTensor& cache,
                         Buffer* kv_buff,
                         const bool is_key,
                         const int32_t n_valid,
                         const int32_t n_heads,
                         const int32_t variant,
                         const int32_t ctx_size) = 0;

  virtual void dumpHead(KVTensor& cache,
                        uint32_t head,
                        const int32_t n_valid,
                        const int32_t variant,
                        const int32_t ctx_size,
                        void* data) = 0;

  // For HTP KeyDiff implementation, copy anchor outputs into anchor input buffers
  // For future optimization, this copy can be avoided in two ways
  // 1. Using a ping-pong Qnn_Tensor_t for anchor_in/anchor_out
  // 2. Using a READ_WRITE Qnn_Tensor_t that automatically reads/writes into the same buffer
  void clearAnchor(KVTensor& cache);
  void updateAnchor(KVTensor& cache);

  // For KeyDiff, runScorer invokes a scoring model to populate cache.score for each cache
  // updateEvictionIndexes must be run after each runScorer() call to utilize this data
  // Since runScorer operates on the HTP, it must be run on the main thread
  bool updateKeydiffScores();
  inline int32_t getEvictionQueueSize();
  void updateEvictionIndexes(KVTensor& cache,
                             const int32_t variant,
                             const int32_t ctx_size,
                             const int32_t n_valid_kv,
                             const int32_t n_evict);

  // Utility function to batch together copies
  static inline std::vector<MoveStep> compileIdxes(const std::vector<int32_t> src_idxes,
                                                   const std::vector<int32_t> dst_idxes);

 public:
  KVManager(Env& env, size_t nThreads, uint64_t cpuMask, bool poll);
  virtual ~KVManager();

  void stopThreadpool();

  // Add tensor pointers to keep track of in the graph
  // - These tensors are ordered. By graph_idx and tensor_idx. This is important for save/restore
  // - The shape of the tensors are deterministic based on each subclass
  // - The pointer points to the start of the buffer.
  // - Actual HTP tensor offsets may change (e.g. in POINTER_SHIFT) but buffer starts are constant
  // - Bitwidth is considered to be constant across tensors
  void registerTensors(int32_t graph_idx, std::vector<KVTensor>& tensors);
  void registerSupportedVariant(int32_t variant, int32_t ctx_size);
  void registerDataType(int8_t bitWidth, bool quantized);

  void registerQnnApi(QnnApi* qnn_api) { m_qnn_api = qnn_api; }
  void initComplete(int32_t embed_dim,
                    int32_t max_ctx_size,
                    int32_t anchor_bitwidth,
                    LongContextParams longcontext_params,
                    bool use_scatter);

  // Get the current states
  int32_t n_past() { return m_n_past; }
  int32_t n_valid_kv() { return m_n_valid_kv; }
  int32_t cur_variant() { return m_cur_variant; }
  int32_t cur_ctx() { return m_cur_ctx; }

  // checks if it is last infernce step or not
  bool isFinalInferenceStep() { return m_islastStep; }

  int32_t getStrategySize() { return m_strategy.size(); }

  // Prepares an inference strategy (a set of InferenceSteps)
  // for the given number of inputs.
  bool prepareInferenceStrategy(int32_t n_inputs);
  bool nextInferenceStep(InferenceStep& step);

  // Blocks the main thread until the given scope is ready,
  // i.e. there are no more background KV$ update jobs to run.
  bool block(Scope scope);
  // Stages update jobs for the next inference.
  bool unblock(Scope scope);

  // Simple getter function. Should we also have a setVariant() call?
  bool setActiveVariant(int32_t variant, int32_t ctx_size);

  // Functions for managing the cache directly called by the Engine/Dialog
  bool dispatchUpdate(int32_t n_past, Mask& mask = {});
  size_t loadKVCache(const std::string& filename);
  bool dumpKVCache(const std::string& filename);
  bool dumpKVCache(Buffer* kv_buff);
  bool getCacheSpec(CacheFileSpec& spec);
  bool getKVHead(CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale);
};
}  // namespace qualla

template <>
struct fmt::formatter<qualla::MoveStep>;