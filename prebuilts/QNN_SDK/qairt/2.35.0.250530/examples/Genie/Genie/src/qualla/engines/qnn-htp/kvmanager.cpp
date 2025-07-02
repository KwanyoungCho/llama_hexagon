//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fstream>  // For save/restore to file

#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "kvmanager.hpp"
#include "qualla/detail/cache-file.hpp"

#define __DEBUG(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace qualla {

KVManager::KVManager(Env& env, size_t nThreads, uint64_t cpuMask, bool poll) : _env(env) {
  if (nThreads > 0) {
    __DEBUG("KVManager: starting threadpool : n_threads {} params. {:#x} poll {}",
            nThreads,
            cpuMask,
            poll);
    m_threadpool = std::make_unique<ThreadPool>();
    m_threadpool->start(nThreads, cpuMask, poll);
  }
}

KVManager::~KVManager() { stopThreadpool(); }

void KVManager::registerTensors(int32_t graph_idx, std::vector<KVTensor>& tensors) {
  __DEBUG("Registering {} tensors for graph {}", tensors.size(), graph_idx);
  for (auto tensor : tensors) {
    __DEBUG("Registered key {:p} value {:p} anchor(in={:p}, out={:p}) (n_heads={})",
            fmt::ptr(tensor.key_buf),
            fmt::ptr(tensor.val_buf),
            fmt::ptr(tensor.anchor_in),
            fmt::ptr(tensor.anchor_out),
            tensor.n_heads);
    m_cache[graph_idx].push_back(tensor);
  }
  m_graphs.push_back(graph_idx);
}

void KVManager::registerSupportedVariant(int32_t variant, int32_t ctx_size) {
  if (ctx_size != -1) m_supported_variants[ctx_size].insert(variant);
}

void KVManager::registerDataType(int8_t bitwidth, bool quantized) {
  m_bitwidth  = bitwidth;
  m_quantized = quantized;

  if (quantized) {
    if (m_bitwidth == 1)
      m_clear_value.u8 = 1 << 7;
    else if (m_bitwidth == 2)
      m_clear_value.u16 = 1 << 15;
    else if (m_bitwidth == 4)
      m_clear_value.u32 = 1 << 31;
  } else {
    m_clear_value.u32 = 0x0;  // Float values are always cleared to 0s
  }
}

std::string InferenceStep::str() const {
  return fmt::format("AR-{} CL-{} n_past={} n_kv={} n_process={} @ past_idx={} new_idx={}",
                     variant,
                     ctx_size,
                     n_past,
                     n_valid_kv,
                     n_process,
                     past_idx,
                     new_idx);
}

void KVManager::initComplete(int32_t embed_dim,
                             int32_t max_ctx_size,
                             int32_t anchor_bitwidth,
                             LongContextParams longcontext_params,
                             bool use_scatter) {
  m_embed_dim       = embed_dim;
  m_longcontext     = longcontext_params;
  m_anchor_bitwidth = anchor_bitwidth;
  m_max_ctx_size    = max_ctx_size;
  m_use_scatter     = use_scatter;

  // Initialize the initial KV$ busy states to not busy
  {
    size_t numQueues = 1;
    if (m_threadpool) numQueues = m_threadpool->size();
    for (auto graph_idx : m_graphs) {
      auto& state = m_graph_state[graph_idx];
      state.jobSlices.reserve(numQueues);
      for (int i = 0; i < numQueues; i++) {
        state.jobSlices.emplace_back(std::make_unique<JobSlice>());
      }
      state.sync = 0;
    }
  }

  {
    if (m_supported_variants.empty()) {
      State::error(
          "Genie is not able to determine the context length for some of the graphs. Please name "
          "the graph properly.");
      return;
    }
    // Set the smallest context size and largest variant as a default start state
    int32_t first_ctx     = m_supported_variants.begin()->first;
    int32_t first_variant = *m_supported_variants.begin()->second.rbegin();

    __DEBUG("Initializing to AR-{} CL-{}", first_variant, first_ctx);
    m_cur_variant = first_variant;
    m_cur_ctx     = first_ctx;
  }

  completeInit();

  std::string variant_str = "";
  for (auto& [ctx_size, variants] : m_supported_variants) {
    for (auto& variant : variants) variant_str += fmt::format("AR-{} CL-{}, ", variant, ctx_size);
  }

  __DEBUG("KVManager initialization complete wtih {} splits ", m_cache.size());
  __DEBUG("embed_dim={} bitwidth={} quantized={}", m_embed_dim, m_bitwidth, m_quantized);

  __DEBUG("Supported configurations= [{}]", variant_str.substr(0, variant_str.size() - 2));

  __DEBUG("Set m_clear_value to u8={} u16={} u32={}",
          m_clear_value.u8,
          m_clear_value.u16,
          m_clear_value.u32);
}

bool KVManager::prepareInferenceStrategy(int32_t n_inputs) {
  // The goal of this is to minimize latency
  // This includes heuristics for minimizing number of iterations and also using smallest ctx_size
  // Enforce maximum context size
  if (m_n_past + n_inputs > m_max_ctx_size) {
    State::error("Requested input exceeds the maximum context size.");
    return false;
  }

  // Assumptions:
  // Lower ctx_size runs faster.
  // Different variants at the same ctx_size are close in time
  // Minimizing latency means picking smallest ctx_size and reducing number of iterations
  // Switching cost can be upto 100ms so avoid switches as much as possible
  // TODO: Once token_history_enabled=false (on embedding input or longcontext), disable AR-c
  InferenceStrategy strategy;

  int32_t n_past     = m_n_past;
  int32_t n_valid_kv = m_n_valid_kv;

  // This is a simple lambda function that returns the smallest choice larger than n
  // If no such choice exists, the largest choice is returned
  auto pick = [](int32_t n, std::set<int32_t>& choices) -> int32_t {
    auto it = choices.lower_bound(n);
    return (it == choices.end()) ? *choices.rbegin() : *it;
  };

  auto iter_ctx   = m_supported_variants.lower_bound(n_valid_kv);  // Pick the smallest CL
  int32_t variant = pick(n_inputs, iter_ctx->second);              // Pick the smallest variant
  // If we exceed CL (on both AR-c and non AR-c graphs), switch to a larger CL (if available)
  while (((iter_ctx->first != variant && (n_valid_kv + variant > iter_ctx->first)) ||
          ((iter_ctx->first == variant && (n_past + n_inputs > iter_ctx->first)))) &&
         (iter_ctx->first != m_supported_variants.rbegin()->first)) {
    iter_ctx++;  // If inference exceeds CL and larger CL is available, switch to a larger CL
    variant = pick(n_inputs, iter_ctx->second);  // Re-pick the variant for the larger CL
  }

  int32_t ctx_size = iter_ctx->first;
  int32_t n_remain = n_inputs;

  if (ctx_size == variant) {  // For AR-ctx graphs (i.e. bertcache), past tokens are reprocessed
    n_remain += n_past;
    n_past = n_valid_kv = 0;

    if (n_remain > ctx_size) {
      State::error("Input is too large for maximum context length available");
      return false;
    }
  }
  while (n_remain > 0) {
    // If the iteration would exceed , and a larger CL is available, then switch to larger CL
    // Calculate how many inputs we can process in this iteration
    int32_t n_process = std::min(n_remain, variant);

    if (variant != ctx_size && n_valid_kv + variant > ctx_size - variant) {
      auto it = m_supported_variants.lower_bound(ctx_size + 1);
      if (it != m_supported_variants.end()) {  // If a larger CL is available, switch to it
        ctx_size  = it->first;
        variant   = pick(n_remain, it->second);
        n_process = std::min(n_remain, variant);
      }
    }

    const int32_t past_dim = ctx_size - variant;
    strategy.emplace_back(variant, ctx_size, n_past, n_valid_kv, n_process, 0, past_dim);
    strategy.back().new_idx = getIndexForPastKV(strategy.back());
    strategy.back().new_idx = getIndexForNewKV(strategy.back());

    // Update the status for next iteration
    n_past += n_process;
    n_valid_kv += n_process;
    n_remain -= n_process;
    // At this point, if we are still exceeding CL, then longcontext must be enabled
    if (n_remain > 0 && (variant != ctx_size && n_valid_kv > past_dim)) {
      if (m_longcontext.mode == LongContextParams::DISABLED) {
        State::error("Input is too large and cannot be processed");
        return false;
      } else {
        n_valid_kv = past_dim;
      }
    }
  }

  m_strategy          = strategy;
  m_strategy_cur_step = 0;
  m_strategy_active   = true;
  __TRACE("Inference strategy prepared.");
  int step_idx = 0;
  for (InferenceStep& step : m_strategy) __TRACE("Step {}: {}", step_idx++, step.str());

  // Check global states and make sure they align with the first step in the strategy
  InferenceStep& step = m_strategy.front();
  if (m_cur_variant != step.variant || m_cur_ctx != step.ctx_size)
    setActiveVariant(step.variant, step.ctx_size);

  return true;
}

bool KVManager::nextInferenceStep(InferenceStep& step) {
  // This is equivalent to an EOF. The current strategy is now complete
  if (m_strategy_cur_step >= m_strategy.size()) {
    m_strategy_active = false;

    // In certain cases, KV$ may be updated during unblock() in the last iteration
    // These updates cause m_last_inference to be updated. We can use this information
    InferenceStep& final_step = m_strategy.back();
    if (final_step.n_process == 1) {
      m_counter++;
      m_n_past      = m_last_inference.n_past;
      m_n_valid_kv  = m_last_inference.n_valid_kv;
      m_cur_variant = m_last_inference.variant;
      m_cur_ctx     = m_last_inference.ctx_size;
    } else {
      m_last_inference = final_step;
      __DEBUG("m_last_inference updated to {}", m_last_inference.str());
    }

    m_strategy.clear();
    m_strategy_cur_step = 0;
    return false;
  }

  // Get the next state and update global states accordingly
  step = m_strategy.at(m_strategy_cur_step++);
  m_counter++;
  m_n_past      = step.n_past;
  m_n_valid_kv  = step.n_valid_kv;
  m_cur_variant = step.variant;
  m_cur_ctx     = step.ctx_size;
  m_islastStep  = (m_strategy_cur_step >= m_strategy.size());

  return true;
}

bool KVManager::block(Scope scope) {
  if (scope.is_per_graph() && !m_cache.contains(scope.graph_idx)) return true;

  if (scope.is_global()) {
    for (auto graph_idx : m_graphs) block(Scope::per_graph(graph_idx));
    return true;
  }

  __DEBUG("Blocking for graph {}", scope.graph_idx);
  GraphState& state = m_graph_state.at(scope.graph_idx);
  while (state.sync != 0) {
  }

  return true;
}

bool KVManager::unblock(Scope scope) {
  if (scope.is_per_graph() && !m_cache.contains(scope.graph_idx)) return true;

  // All blocks during inference MUST go through m_strategy
  // If no strategy is active, the block must be for something else, e.g. saving/dumping the cache
  if (!m_strategy_active) return true;

  // Check if the next KV$ update needs to be processed
  // This is disabled for the final step, unless only 1 input was processed
  InferenceStep& step       = m_strategy.at(m_strategy_cur_step - 1);
  const bool is_final_step  = m_strategy_cur_step >= m_strategy.size();
  const bool process_update = !is_final_step || (step.n_process == 1);

  // For KeyDiff, run the scorer if necessary
  // Check 1: Strategy is active (we're currently running inference)
  // Check 2: This strategy step requires token eviction
  // Check 3: Token eviction requires re-running the scoring network
  // Check 4: First split has just finished running. This is a pipelining optimization
  if (m_longcontext.mode == LongContextParams::KEYDIFF && process_update &&
      (scope.is_per_graph() && scope.graph_idx == m_cache.begin()->first)) {
    const int32_t n_evict = step.n_valid_kv + step.n_process - (step.ctx_size - step.variant);
    __DEBUG("n_evict={}, n_queue={}", n_evict, getEvictionQueueSize());
    if (n_evict > getEvictionQueueSize())
      if (!updateKeydiffScores()) return false;
  }

  // Always update the anchor buffer. Read description for updateAnchor
  // Runs almost instantly (<5us), and it cannot be run concurrently with the scorer network
  // For now, running it on the main thread is okay
  if (m_anchor_bitwidth) {
    for (auto& [graph_idx, kv_cache] : m_cache)
      if (scope.graph_idx == -1 || graph_idx == scope.graph_idx)
        for (auto& cache : kv_cache) updateAnchor(cache);
  }

  if (!process_update) return true;  // Skip KV$ updates

  // Process the next update - update KV$ with all of the newly generated KV$
  // On the last step (only triggered if n_process==1), the likely next step is generation
  // We can proactively trigger switching to the smallest variant
  const bool switch_variant = is_final_step;
  if (!processUpdate(scope, step, step.n_past + step.n_process, switch_variant)) return false;

  // If the strategy has a pre-determined next step, then switch to the next known variant
  if (is_final_step) return true;  // Skip reshapes if it's the final step
  InferenceStep& next_step = m_strategy.at(m_strategy_cur_step);
  if (step.variant != next_step.variant || step.ctx_size != next_step.ctx_size) {
    const int32_t cur_variant = step.variant, cur_ctx = step.ctx_size;
    const int32_t new_variant = next_step.variant, new_ctx = next_step.ctx_size;

    // Reshape is always to larger context or smaller variant, i.e. no token eviction logic needed
    const auto reshape_job = [&, cur_variant, cur_ctx, new_variant, new_ctx](KVTensor& cache) {
      reshapeCache(cache, cur_variant, cur_ctx, new_variant, new_ctx);
    };

    __DEBUG("KVUPDATE({}) reshapeCache(AR-{} CL-{} -> AR-{} CL-{})",
            scope.graph_idx,
            cur_variant,
            cur_ctx,
            new_variant,
            new_ctx);
    prepareJob(scope, {"reshapeCache", reshape_job});
  }
  return true;
}

bool KVManager::setActiveVariant(int32_t variant, int32_t ctx_size) {
  // Considerations -> Can the new variant hold all the KV$ we already have
  //                   If not, trigger longcontext and/or increasing context

  if (variant == -1) variant = m_cur_variant;
  if (ctx_size == -1) ctx_size = m_cur_ctx;

  const int32_t cur_variant = m_cur_variant, cur_ctx = m_cur_ctx, cur_n_valid = m_n_valid_kv;
  const int32_t n_valid = std::min(ctx_size - variant, cur_n_valid);
  const int32_t n_evict = cur_n_valid - n_valid;

  if (n_evict > 0) {
    if (m_longcontext.mode == LongContextParams::DISABLED) {
      State::error(fmt::format("Cant fit {} KV$ in AR-{} CL-{} ", cur_n_valid, variant, ctx_size));
      return false;
    } else if (m_longcontext.mode == LongContextParams::SLIDING_WINDOW) {
      State::error("Eviction on reshape not implemented for Sliding window yet");
      return false;
    }
  }

  m_cur_variant = variant;
  m_cur_ctx     = ctx_size;
  m_n_valid_kv  = n_valid;

  // AR-c graphs do not take any KV$ input, so this simplifies to a no-op
  if (variant == ctx_size) return true;

  // Check if the reshape requires token eviction and token eviction requires re-running the scorer
  // Note that a global block must also be enforced to ensure all KV$ updates are synced
  if (m_longcontext.mode == LongContextParams::KEYDIFF && n_evict > getEvictionQueueSize()) {
    if (!block(Scope::global())) return false;
    if (!updateKeydiffScores()) return false;
  }

  const auto reshape_job =
      [&, cur_variant, cur_ctx, cur_n_valid, variant, ctx_size, n_valid, n_evict](KVTensor& cache) {
        if (n_evict > 0 && m_longcontext.mode == LongContextParams::KEYDIFF) {
          if (n_evict > cache.evict_idxes[0].size())  // Update eviction queue if necessary
            updateEvictionIndexes(cache, cur_variant, cur_ctx, cur_n_valid, n_evict);

          // Use the eviction indexes to prune necessary KV$
          MoveStrategy move_idxes;
          for (int32_t head = 0; head < cache.n_heads; head++) {
            auto& head_evict_queue = cache.evict_idxes.at(head);

            // Collect eviction indexes. Only "valid" indexes (i.e. fits in new KV$) are considered
            std::set<int32_t> evict_set;
            for (int i = 0; i < n_evict; i++) {
              evict_set.insert(head_evict_queue.front());
              head_evict_queue.pop();
            }

            // Invalidate/empty the queue since indexes will change after eviction/reshape
            // Theoretically, we only need to invalidate the pruned idxes, so this can be optimized
            while (!head_evict_queue.empty()) head_evict_queue.pop();

            std::vector<int32_t> src_idxes, dst_idxes;
            auto evict_iter = evict_set.begin();
            for (int idx = n_valid; idx < cur_n_valid; idx++) {
              if (evict_set.contains(idx))
                continue;  // This index was slated for eviction, so no-op
              src_idxes.push_back(idx);
              dst_idxes.push_back(*evict_iter);
              evict_iter = evict_set.erase(evict_iter);
            }

            move_idxes[head] = compileIdxes(src_idxes, dst_idxes);
          }

          moveKV(cache, cur_variant, cur_ctx, move_idxes);
        }

        // Reshape the cache
        reshapeCache(cache, cur_variant, cur_ctx, variant, ctx_size);
      };

  __DEBUG("KVUPDATE(-1) reshapeCache(AR-{} CL-{} -> AR-{} CL-{})",
          cur_variant,
          cur_ctx,
          variant,
          ctx_size);
  prepareJob(Scope::global(), {"reshapeCache", reshape_job});
  return true;
}

void KVManager::prepareJob(Scope scope, Job job) {
  if (scope.is_per_graph() && !m_cache.contains(scope.graph_idx)) return;

  // For global jobs, split them into per graph
  if (scope.is_global()) {
    for (auto graph_idx : m_graphs) prepareJob(Scope::per_graph(graph_idx), job);
    return;
  }

  const int32_t graph_idx = scope.graph_idx;

  // Some splits may not contain KV$
  if (m_cache.find(graph_idx) == m_cache.end()) return;

  GraphState* state = &m_graph_state.at(graph_idx);
  auto kv_tensors   = &m_cache.at(graph_idx);

  if (m_threadpool) {
    const size_t n_tensors         = kv_tensors->size();
    const size_t n_slices          = state->jobSlices.size();
    const size_t tensors_per_slice = n_tensors / n_slices;
    const size_t remainder         = n_tensors - (tensors_per_slice * n_slices);
    state->sync += n_slices;
    size_t startIdx = 0;
    size_t endIdx   = 0;
    for (int tidx = 0; tidx < n_slices; tidx++) {
      startIdx = endIdx;
      endIdx   = startIdx + tensors_per_slice;
      if (tidx < remainder) {
        endIdx++;
      }
      std::lock_guard<std::mutex> jobLock(state->jobSlices[tidx]->queuedMutex);
      const auto update_job = [kv_tensors, startIdx, endIdx, n_tensors, job]() {
        auto iter = kv_tensors->begin() + startIdx;
        auto end  = kv_tensors->begin() + std::min(endIdx, n_tensors);
        for (; iter < end; iter++) job.update_function(*iter);
      };
      state->jobSlices[tidx]->queued.push(update_job);
    }
    // Add update requests to the threadpool.
    queueJob(scope.graph_idx);
  } else {
    // If this is a single-threaded environment, run the entire job immediately.
    for (auto iter = kv_tensors->begin(); iter < kv_tensors->end(); iter++) {
      job.update_function(*iter);
    }
  }
}

void KVManager::queueJob(int32_t graph_idx) {
  GraphState* state     = &m_graph_state.at(graph_idx);
  const auto requestJob = [state]() {
    // Process any available job queues
    for (auto& jobSlice : state->jobSlices) {
      std::unique_lock<std::mutex> runningLock(jobSlice->runningMutex, std::try_to_lock);
      if (runningLock) {
        do {
          // Run all jobs.
          while (!jobSlice->running.empty()) {
            auto job = jobSlice->running.front();
            jobSlice->running.pop();
            job();
            state->sync--;
          }
          // Quickly flush queued jobs from the main thread to the running jobs on this thread.
          // This is a fast operation which frees up the main thread to queue more jobs ASAP.
          std::unique_lock<std::mutex> queuedLock(jobSlice->queuedMutex);
          while (!jobSlice->queued.empty()) {
            auto job = jobSlice->queued.front();
            jobSlice->queued.pop();
            jobSlice->running.push(job);
          }
        } while (!jobSlice->running.empty());
      }
    }
  };

  size_t n_threads = m_threadpool->size();
  std::vector<std::function<void()>> kvUpdateRequests;
  for (int tidx = 0; tidx < n_threads; tidx++) {
    kvUpdateRequests.push_back(requestJob);
  }
  m_threadpool->enqueue(kvUpdateRequests);
}

inline std::vector<MoveStep> KVManager::compileIdxes(const std::vector<int32_t> src_idxes,
                                                     const std::vector<int32_t> dst_idxes) {
  // Compile src/dst_idxes into a vector of [src_idx, dst_idx, count]. This batches memory calls
  // This can be further optimized by detecting common contiguous copies (during token eviction)
  std::vector<MoveStep> batch_idxes = {{src_idxes[0], dst_idxes[0], 1}};
  for (auto i = 1; i < src_idxes.size(); i++) {
    // If the src/dst indexes are not consecutive, start a new batch with current src/dst indexes
    // Else, increment the size of the current batch
    if (src_idxes[i] != src_idxes[i - 1] + 1 || dst_idxes[i] != dst_idxes[i - 1] + 1)
      batch_idxes.push_back({src_idxes[i], dst_idxes[i], 1});
    else
      batch_idxes.back().count++;
  }

  return batch_idxes;
}

// processUpdate consumes the last known inference (m_last_inference) to generate update jobs
// CAUTION: m_last_inference is destroyed by this function, since KV$ can only be consumed once
bool KVManager::processUpdate(
    Scope scope, InferenceStep& step, int32_t n_past, bool switch_variant, Mask& mask) {
  const int32_t n_update = n_past - m_n_past;
  __DEBUG("update AR-{} CL-{} with {}/{} entries",
          step.variant,
          step.ctx_size,
          n_update,
          step.n_process);

  if (n_update > step.n_process) {
    State::error("KV update count exceeds the total processed inputs from last inference");
    return false;
  }

  // If switch_variant is active, then reshape into the smallest variant
  const int32_t new_variant =
      switch_variant ? *m_supported_variants.at(step.ctx_size).begin() : step.variant;

  // Special handling for AR-c models. Theoretically, switch_variant should always be true
  // Either a AR-c model is run, which consumes all input, hence is_final_step is triggered
  // Or a AR-c model is run, and a dispatchUpdate is triggered, where switch_variant==true
  if (step.variant == step.ctx_size) {
    // The only Op required is reshaping it into the new variant size
    const auto reshape_job =
        [&, variant = step.variant, ctx_size = step.ctx_size, new_variant](KVTensor& cache) {
          reshapeCache(cache, variant, ctx_size, new_variant, ctx_size);
        };
    __DEBUG("KVUPDATE({}) reshapeCache(AR-{} CL-{} -> AR-{} CL-{})",
            scope.graph_idx,
            step.variant,
            step.ctx_size,
            new_variant,
            step.ctx_size);
    prepareJob(scope, {"reshape", reshape_job});

    // AR-c models mean n_update KV$ entries are available (i.e. n_past == n_valid_kv == n_update)
    m_last_inference = {new_variant, step.ctx_size, n_update, n_update, 0, 0, 0};
    __DEBUG("m_last_inference updated to {}", m_last_inference.str());
    return true;
  }

  std::vector<int32_t> src_idxes(n_update);  // Select which KV$ needs to be updated
  if (mask.empty()) {  // If the mask is empty, the sequential range [0, n_update] is copied
    std::iota(src_idxes.begin(), src_idxes.end(), 0);
  } else {  // If a mask is supplied, KV$ is selectively copied
    for (int i = 0, j = 0; i < step.n_process; i++)
      if (mask[i]) src_idxes[j++] = i;
  }

  // Select which KV$ indexes are to be copied to. The checks are done in this order:
  // 1. If the KV$ cleanly fits into the available capacity, simply copy it over
  // 2. If the context size can be increased to enable additional capacity, do so
  // 3. If long context is disabled, throw an error because *we have an issue*
  // 4. If long context is enabled, apply long context to determine new cache positions
  std::vector<int32_t> dst_idxes;
  const int32_t past_dim = step.ctx_size - step.variant;  // This is the current KV$ capacity

  int32_t new_n_valid_kv = step.n_valid_kv;
  if (step.n_valid_kv + n_update <= past_dim) {
    // Check 1: If KV$ cleanly fits into the available capacity
    for (int i = 0; i < n_update; i++) dst_idxes.push_back(step.n_valid_kv + i);
    new_n_valid_kv += n_update;

    // Check 2: TODO dynamic context length

  } else if (m_longcontext.mode == LongContextParams::DISABLED) {
    // Check 3: Long context has been disabled. Throw an error.
    State::error(fmt::format(
        "Requested {} KV$ doesn't fit capacity {}", step.n_valid_kv + n_update, past_dim));
    return false;

  } else if (m_longcontext.mode == LongContextParams::SLIDING_WINDOW) {
    // Check 4a: Use long context parameters to determine new cache positions
    const int32_t& n_sink = m_longcontext.sink_tokens;
    for (int i = 0; i < n_update; i++)
      dst_idxes.push_back(n_sink + (step.n_past - n_sink + i) % (past_dim - n_sink));
    new_n_valid_kv = past_dim;
    if (new_variant != step.variant) {
      State::error("Switching variants after token eviction temporarily disabled");
      return false;
    }
  } else if (m_longcontext.mode == LongContextParams::KEYDIFF) {
    // Check 4b: Use KeyDiff to determine new cache positions

    const auto evict_job =
        [&, step = step, past_dim, n_update, src_idxes, new_variant](KVTensor& cache) {
          const int32_t n_evict = (step.n_valid_kv + n_update) - past_dim;
          const int32_t n_empty = n_update - n_evict;  // i.e. past_dim - step.n_valid_kv

          if (n_evict > cache.evict_idxes[0].size())  // Update eviction queue if necessary
            updateEvictionIndexes(cache, step.variant, step.ctx_size, step.n_valid_kv, n_evict);

          // Fill in the empty KV$ indexes first (i.e. from 0 to n_empty)
          std::vector<int32_t> dst_idxes(n_update);
          std::iota(&dst_idxes[0], &dst_idxes[n_empty], step.n_valid_kv);

          MoveStrategy copy_idxes;
          for (int32_t head = 0; head < cache.n_heads; head++) {
            // Evict n_evict (i.e. n_update - n_empty) tokens, and overwrite
            for (int32_t i = n_empty; i < n_update; i++) {
              dst_idxes[i] = cache.evict_idxes[head].front();
              cache.evict_idxes[head].pop();
            }

            copy_idxes[head] = compileIdxes(src_idxes, dst_idxes);
          }

          updateKV(cache, step.variant, step.ctx_size, copy_idxes);

          // Reshape Cache here will always be from larger variant -> smaller variant
          // i.e. no eviction impacts need to be considered
          if (new_variant != step.variant)
            reshapeCache(cache, step.variant, step.ctx_size, new_variant, step.ctx_size);
        };

    new_n_valid_kv = past_dim;
    prepareJob(scope, {"evictAndUpdate", evict_job});

    m_last_inference = {new_variant, step.ctx_size, n_past, new_n_valid_kv, 0, 0, 0};
    __TRACE("m_last_inference updated to {}", m_last_inference.str());
    return true;
  }

  // The map allows unique copies for each head_idx. -1 implies apply to all heads
  MoveStrategy copy_idxes = {{-1, compileIdxes(src_idxes, dst_idxes)}};

  const auto update_job =
      [&, variant = step.variant, ctx_size = step.ctx_size, copy_idxes, new_variant](
          KVTensor& cache) {
        updateKV(cache, variant, ctx_size, copy_idxes);
        if (new_variant != variant) reshapeCache(cache, variant, ctx_size, new_variant, ctx_size);
      };
  __DEBUG("KVUPDATE({}) updateKV(AR-{} CL-{}, n_update={})",
          scope.graph_idx,
          step.variant,
          step.ctx_size,
          n_update);
  if (new_variant != step.variant)
    __DEBUG("KVUPDATE({}) reshapeCache(AR-{} CL-{} -> AR-{} CL-{})",
            scope.graph_idx,
            step.variant,
            step.ctx_size,
            new_variant,
            step.ctx_size);
  prepareJob(scope, {"update", update_job});

  m_last_inference = {new_variant, step.ctx_size, n_past, new_n_valid_kv, 0, 0, 0};
  __TRACE("m_last_inference updated to {}", m_last_inference.str());
  return true;
}

bool KVManager::dispatchUpdate(int32_t n_past, Mask& mask) {
  // Assume this is a Scope::GLOBAL call since it is only called externally

  __TRACE("n_past: {}, m_n_past: {}", n_past, m_n_past);

  // Clear the cache
  if (n_past == 0) {
    __DEBUG("KVUPDATE(-1) clearCache()");
    prepareJob(Scope::global(), {"clear", [&](KVTensor& cache) { clear(cache); }});
    m_n_past = m_n_valid_kv = 0;
    // Revert to the default start state of smallest CL, largest variant
    m_cur_ctx        = m_supported_variants.begin()->first;
    m_cur_variant    = *m_supported_variants.begin()->second.rbegin();
    m_last_inference = {m_cur_variant, m_cur_variant, m_n_past, m_n_valid_kv, 0, 0, 0};

    // Reset token eviction state and queues
    for (auto& [graph_idx, kv_caches] : m_cache) {
      for (auto& kv_cache : kv_caches) {
        for (auto& head_evict_idxes : kv_cache.evict_idxes) {
          std::queue<int32_t> empty;
          head_evict_idxes.swap(empty);
        }
      }
    }

    return true;
  }

  if (n_past == m_n_past) return true;

  // Requested n_past is smaller, so invoke reduction of KV$
  if (n_past < m_n_past) {
    InferenceStep& step = m_last_inference;
    if (step.n_past != step.n_valid_kv) {
      State::error("Cannot reduce KV$ after token eviction");
      return false;
    } else if (!mask.empty()) {
      State::error("Selective KV$ removal not supported");
      return false;
    }

    std::vector<std::pair<int32_t, size_t>> clear_idxes = {{n_past, m_n_past - n_past}};
    const auto remove_job =
        [&, variant = step.variant, ctx_size = step.ctx_size, clear_idxes](KVTensor& cache) {
          reduceKV(cache, variant, ctx_size, clear_idxes);
        };
    __DEBUG("KVUPDATE(-1) reduce(AR-{} CL-{}, clear={})", step.variant, step.ctx_size, clear_idxes);
    prepareJob(Scope::global(), {"remove", remove_job});
    m_n_past     = n_past;
    m_n_valid_kv = n_past;  // This is only valid due to the (step.n_past != step.n_valid_kv) check
    return true;
  }

  // Check if the update requires token eviction and token eviction requires re-running the scorer
  // Note that a global block must also be enforced to ensure all KV$ updates are synced
  if (m_longcontext.mode == LongContextParams::KEYDIFF) {
    // Total cache size (current KV$ size + update size = n_past - m_n_past) - physical cache size
    const int32_t n_evict = (m_last_inference.n_valid_kv + n_past - m_n_past) -
                            (m_last_inference.ctx_size - m_last_inference.variant);
    if (n_evict > getEvictionQueueSize()) {
      if (!block(Scope::global())) return false;
      if (!updateKeydiffScores()) return false;
    }
  }

  // dispatchUpdate is explicitly called by Dialog after prompt processing OR during generation
  // Either way, most likely the next inference occurs during the generation phase
  // In this case, the smallest variant is needed. Hence, potentially we can proactively switch
  if (!processUpdate(Scope::global(), m_last_inference, n_past, true, mask)) return false;

  m_cur_variant = m_last_inference.variant;
  m_n_valid_kv  = m_last_inference.n_valid_kv;
  m_n_past      = n_past;

  return true;
}

size_t KVManager::loadKVCache(const std::string& filename) {
  __DEBUG("KVManager::loadKVCache {}", filename);

  std::ifstream handle(filename, std::ios::in | std::ios::binary);
  if (handle.fail()) {
    State::error(fmt::format("Error opening file {}", filename));
    return 0;
  }

  CacheFileSpec spec;
  handle.read((char*)&spec, sizeof(spec));
  if (spec.magic != 0xC0DE) {
    State::error(fmt::format("Incorrect magic number. 0xC0DE. Found {:#x}", spec.magic));
    return 0;
  }

  __DEBUG(
      "KVManager::loadKVCache {{ num_tensors {}, magic {:x}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  for (auto& [graph_idx, kv_cache] : m_cache)
    for (auto& cache : kv_cache)
      loadCache(cache, &handle, true, spec.update_size, spec.n_heads, m_cur_variant, m_cur_ctx);

  for (auto& [graph_idx, kv_cache] : m_cache)
    for (auto& cache : kv_cache)
      loadCache(cache, &handle, false, spec.update_size, spec.n_heads, m_cur_variant, m_cur_ctx);

  m_counter++;
  m_n_past     = spec.update_size;
  m_n_valid_kv = spec.update_size;

  return spec.update_size;
}

bool KVManager::dumpKVCache(const std::string& filename) {
  __DEBUG("KVManager::dumpKVCache {}", filename);
  std::ofstream handle(filename, std::ios::out | std::ios::binary);
  if (handle.fail()) {
    State::error(fmt::format("Error opening file {}", filename));
    return false;
  }

  uint16_t max_n_heads = 0;
  uint32_t n_tensors   = 0;
  for (auto& [graph_idx, kv_cache] : m_cache) {
    for (auto& cache : kv_cache) {
      if (cache.n_heads > max_n_heads) max_n_heads = cache.n_heads;
      n_tensors++;
    }
  }

  CacheFileSpec spec(2 * n_tensors,
                     0xc0de,
                     CacheFileSpec::UINT8_T,
                     0x0,
                     max_n_heads,
                     m_embed_dim,
                     static_cast<uint16_t>(m_n_valid_kv));
  handle.write((char*)&spec, sizeof(spec));

  __DEBUG(
      "KVManager::dumpKVCache {{ num_tensors {}, magic {:x}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  for (auto& [graph_idx, kv_cache] : m_cache)
    for (auto& cache : kv_cache)
      dumpCache(cache, &handle, true, spec.update_size, max_n_heads, m_cur_variant, m_cur_ctx);

  for (auto& [graph_idx, kv_cache] : m_cache)
    for (auto& cache : kv_cache)
      dumpCache(cache, &handle, false, spec.update_size, max_n_heads, m_cur_variant, m_cur_ctx);

  std::vector<double> scales;
  for (auto& [graph_idx, kv_cache] : m_cache)
    for (auto& cache : kv_cache) scales.push_back(cache.key_quant.scale);
  for (auto& [graph_idx, kv_cache] : m_cache)
    for (auto& cache : kv_cache) scales.push_back(cache.value_quant.scale);
  handle.write((char*)scales.data(), scales.size() * sizeof(double));

  handle.close();

  return true;
}

bool KVManager::dumpKVCache(Buffer* kvBuff) {
  uint16_t max_n_heads = 0;
  uint32_t n_tensors   = 0;
  for (auto& [graph_idx, kv_cache] : m_cache) {
    for (auto& cache : kv_cache) {
      if (cache.n_heads > max_n_heads) max_n_heads = cache.n_heads;
      n_tensors++;
    }
  }

  CacheFileSpec spec(2 * n_tensors,
                     0xc0de,
                     CacheFileSpec::UINT8_T,
                     0x0,
                     max_n_heads,
                     m_embed_dim,
                     static_cast<uint16_t>(m_n_valid_kv));

  kvBuff->appendBuffer((uint8_t*)&spec, sizeof(spec));

  __DEBUG(
      "KVManager::dumpKVCache {{ num_tensors {}, magic {:x}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  for (auto& [graph_idx, kv_cache] : m_cache)
    for (auto& cache : kv_cache)
      dumpCache(cache, kvBuff, true, spec.update_size, max_n_heads, m_cur_variant, m_cur_ctx);

  for (auto& [graph_idx, kv_cache] : m_cache)
    for (auto& cache : kv_cache)
      dumpCache(cache, kvBuff, false, spec.update_size, max_n_heads, m_cur_variant, m_cur_ctx);

  std::vector<double> scales;
  for (auto& [graph_idx, kv_cache] : m_cache)
    for (auto& cache : kv_cache) scales.push_back(cache.key_quant.scale);
  for (auto& [graph_idx, kv_cache] : m_cache)
    for (auto& cache : kv_cache) scales.push_back(cache.value_quant.scale);
  kvBuff->appendBuffer((uint8_t*)scales.data(), scales.size() * sizeof(double));

  return true;
}

bool KVManager::getCacheSpec(CacheFileSpec& spec) {
  uint16_t max_n_heads = 0;
  uint32_t n_tensors   = 0;
  for (auto& [graph_idx, kv_cache] : m_cache) {
    for (auto& cache : kv_cache) {
      if (cache.n_heads > max_n_heads) max_n_heads = cache.n_heads;
      n_tensors++;
    }
  }

  spec.num_tensors = 2 * n_tensors;
  spec.magic       = 0xc0de;
  spec.dtype       = CacheFileSpec::UINT8_T;
  spec.pad8_t      = 0x0;
  spec.n_heads     = max_n_heads;
  spec.embed_dim   = m_embed_dim;
  spec.update_size = static_cast<uint16_t>(m_n_valid_kv);

  __DEBUG(
      "KVManager::getCacheSpec {{ num_tensors {}, magic {:x}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  return true;
}

bool KVManager::getKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  uint32_t curr_layer = 0;
  for (auto& [graph_idx, kv_cache] : m_cache) {
    for (auto& cache : kv_cache) {
      if (curr_layer == layer) {
        dumpHead(cache, head, spec.update_size, m_cur_variant, m_cur_ctx, data);
        scale[0] = cache.key_quant.scale;
        scale[1] = cache.value_quant.scale;
        return true;
      }
      curr_layer++;
    }
  }

  return false;
}

// Functions for KeyDiff
void KVManager::clearAnchor(KVTensor& cache) {
  if (cache.anchor_in == nullptr || cache.anchor_out == nullptr) return;
  std::fill_n((uint16_t*)cache.anchor_in, cache.n_heads * m_embed_dim, cache.anchor_offset);
}

void KVManager::updateAnchor(KVTensor& cache) {
  if (cache.anchor_in == nullptr || cache.anchor_out == nullptr) return;
  std::memcpy(cache.anchor_in, cache.anchor_out, cache.n_heads * m_embed_dim * m_anchor_bitwidth);
}

bool KVManager::updateKeydiffScores() {
  __DEBUG("Updating KeyDiff scores - executing scorer");
  if (m_qnn_api == nullptr) {
    State::error("Qnn API not registered for scoring network");
    return false;
  }

  if (!m_qnn_api->executeScorer()) {
    State::error("Error executing scorer network");
    return false;
  }
  return true;
}

int32_t KVManager::getEvictionQueueSize() {
  return static_cast<int32_t>(m_cache.begin()->second.front().evict_idxes[0].size());
}

void KVManager::updateEvictionIndexes(KVTensor& cache,
                                      const int32_t variant,
                                      const int32_t ctx_size,
                                      const int32_t n_valid_kv,
                                      const int32_t n_evict) {
  // Update cache.evict_idxes based on cache.scores [cache.n_heads, cur_ctx]
  const int32_t n_sink  = m_longcontext.sink_tokens;
  const int32_t n_queue = std::max(n_evict, m_longcontext.update_frequency);
  for (int32_t head = 0; head < cache.n_heads; head++) {
    std::vector<size_t> indices(n_valid_kv - n_sink);
    std::iota(indices.begin(), indices.end(), n_sink);

    uint16_t* const scores = ((uint16_t*)cache.scores) + head * ctx_size;
    std::partial_sort(indices.begin(),
                      indices.begin() + n_queue,
                      indices.end(),
                      [scores](const size_t a, const size_t b) { return scores[a] > scores[b]; });

    auto& head_evict_queue = cache.evict_idxes[head];
    while (!head_evict_queue.empty()) head_evict_queue.pop();
    for (int i = 0; i < n_queue; i++) head_evict_queue.push(indices[i]);
  }
}

void KVManager::stopThreadpool() {
  if (m_threadpool) {
    m_threadpool->stop();
  }
}

}  // namespace qualla
