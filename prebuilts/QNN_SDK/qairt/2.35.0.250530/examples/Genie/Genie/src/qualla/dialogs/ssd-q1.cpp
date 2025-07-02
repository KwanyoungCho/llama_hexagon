//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>

#include "qualla/context.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/json.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/sampler-utils.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/dialog.hpp"
#include "qualla/sampler.hpp"

#define __ERROR(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __INFO(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __KPIS(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __DEBUG(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __KVTRACE(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace fs = std::filesystem;
namespace qualla {

using qc = qualla::Config;

class SelfSpecDecDialog : public Dialog {
  enum { VERSION = 1 };

 public:
  SelfSpecDecDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf);

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;
  virtual bool process(std::vector<uint8_t>& embedding_vectors,
                       Dialog::T2ECallback t2eCallback,
                       Dialog::Callback callback) override;
  virtual void reset() override;

  virtual bool process(std::vector<int32_t>& tokens, DialogCallback callback) override {
    return false;
  }

  virtual bool save(const std::string& name) override;
  virtual bool restore(const std::string& name) override;

 protected:
  virtual bool supportsLongContext() const override {
    return (_n_streams <= 1);  // Multistream not supported.
  };

 private:
  Sampler& _t_sampler;

  int32_t _vocab;

  std::string _kv_prefix_name{"forecast-prefix"};

  // AR8
  size_t _draft{1};
  std::vector<size_t> _branches{3};

  size_t _forecast_prefix{16};
  size_t _forecast_token_offset{32000};
  size_t _forecast_token_count{4};

  // Multistream parameters
  int32_t _n_streams{1};
  float _p_threshold{0.0f};

  InputType m_inputType{InputType::UNKNOWN};

  bool processFollowOnGeneration(std::vector<int32_t>& tokens,
                                 Tensor& logits,
                                 Dialog::Callback callback);
  // Multistream
  bool processFollowOnGeneration(std::vector<std::vector<int32_t>>& streams,
                                 Tensor& logits,
                                 Dialog::Callback callback);

  /*
      Helper function for combining masks for SSD mulstistream.

      @param  masks           The attention mask to be tiled
      @param  streamIndices   Indices of streams. The tiling count is equal to the size of this
     vector.
      @param  pastMap         A vector of stream indices for masking all past tokens after the
     prompt.
      @param  prefixOffset    Offset where KV prefix masking begins in each tile.
      @param  finalMask       A mask that combines all of the independent masks such that
                              they can be executed in the same inference.
  */
  void tileAttentionMask(const std::vector<int32_t>& mask,
                         const std::vector<size_t> streamIndices,
                         const std::vector<size_t>& pastMap,
                         const size_t prefixOffset,
                         std::vector<int32_t>& finalMask);

  std::vector<int32_t> gen_attention_map() const;
  auto get_len_flat_sample_tree() const;
  auto gen_forecast_tokens(int repeat) const;

  // Sampling and verification
  std::vector<int32_t> build_sample_tree(int32_t last_token,
                                         Tensor& logits,
                                         const std::vector<int32_t>& indices);

  std::tuple<std::vector<int32_t>, std::vector<int32_t>> verify_and_select_longest(
      std::span<int32_t> sample_tree, Tensor& logits);

  std::vector<int32_t> sample_to_draft(Tensor& logits, size_t index, size_t count) {
    std::vector<int32_t> toReturn;
    Tensor indexedTensor = logits.getIndexedTensor(index, _vocab);
    switch (logits.getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        IndexedQuantLogits<uint8_t> logits_u8(indexedTensor, _t_sampler.rng());
        logits_u8.topK(count);
        toReturn = logits_u8.indices;
        break;
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        IndexedQuantLogits<uint16_t> logits_u16(indexedTensor, _t_sampler.rng());
        logits_u16.topK(count);
        toReturn = logits_u16.indices;
        break;
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        IndexedQuantLogits<uint16_t> logits_fp16(indexedTensor, _t_sampler.rng());
        logits_fp16.topK(count);
        toReturn = logits_fp16.indices;
        break;
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        IndexedQuantLogits<float> logits_float(indexedTensor, _t_sampler.rng());
        logits_float.topK(count);
        toReturn = logits_float.indices;
        break;
      }
      default: {
        __ERROR("Incorrect logits datatype.");
        break;
      }
    }
    return toReturn;
  }

  int32_t sample_to_verify(Tensor& logits, size_t index) {
    Tensor indexedTensor = logits.getIndexedTensor(index, _vocab);
    auto token           = _t_sampler.process(indexedTensor);
    return token;
  }

  void convertTokensToEmbeddings(std::vector<int32_t>& tokens,
                                 std::vector<uint8_t>& embeddings,
                                 size_t embeddingBufferSize,
                                 Dialog::T2ECallback t2eCallback);
};

SelfSpecDecDialog::SelfSpecDecDialog(std::shared_ptr<Env> env,
                                     const std::string& name,
                                     const json& conf)
    : Dialog(env, name, conf), _t_sampler(*_sampler["primary"]) {
  auto ssd_version = qc::optional<int>(conf, "ssd-version", 0);
  if (ssd_version > SelfSpecDecDialog::VERSION) __WARN("newer ssd-version in config!");

  _vocab = _ctx->n_vocab();

  _branches = qc::optional(conf, "branches", _branches);
  _draft    = _branches.size();

  _forecast_prefix       = qc::optional(conf, "forecast-prefix", _forecast_prefix);
  _forecast_token_count  = qc::optional(conf, "forecast-token-count", _forecast_token_count);
  _forecast_token_offset = _vocab;

  _kv_prefix_name = qc::optional(conf, "forecast-prefix-name", _kv_prefix_name);

  _n_streams   = qc::optional<int32_t>(conf, "n-streams", 1);
  _p_threshold = qc::optional<float>(conf, "p-threshold", 0.0);

  if (!_engine.contains("primary")) {
    State::fatal("\"primary\" engine not present in config!");
    return;
  }

  // Get Input Type from the engine
  m_inputType = _engine["primary"]->getInputType();
  // Load KV prefix
  Timer timer;
  size_t n_restored_prefix = _engine["primary"]->restore(_kv_prefix_name, true);
  if (n_restored_prefix != _forecast_prefix) {
    throw std::runtime_error(fmt::format("SSD : Loaded {} KV$ from {} but expected {} KV$",
                                         n_restored_prefix,
                                         _kv_prefix_name,
                                         _forecast_prefix));
  }
  _n_past = _forecast_prefix;
  _kpis.restore.update(timer.elapsed_usec());
}

auto SelfSpecDecDialog::get_len_flat_sample_tree() const {
  size_t len_flat_sample_tree = 1;
  size_t last_tokens          = 1;
  for (int i = 0; i < _draft; ++i) {
    len_flat_sample_tree += last_tokens * _branches[i];
    last_tokens = last_tokens * _branches[i];
  }
  return len_flat_sample_tree;
}

auto SelfSpecDecDialog::gen_forecast_tokens(int repeat) const {
  std::vector<int32_t> forecast_tokens(_draft, 0);
  std::iota(forecast_tokens.begin(), forecast_tokens.end(), _forecast_token_offset);

  std::vector<int32_t> ret;
  for (auto i = 0; i < repeat; ++i)
    ret.insert(ret.end(), forecast_tokens.begin(), forecast_tokens.end());
  return ret;
}

std::vector<int32_t> SelfSpecDecDialog::gen_attention_map() const {
  auto len_flat_sample_tree = get_len_flat_sample_tree();
  std::vector<int32_t> attention_map(len_flat_sample_tree + len_flat_sample_tree * _draft, -1);

  auto build_verify_tree = [&attention_map, this](
                               auto self, int parent_begin, int parent_end, int level) {
    if (level == _draft) return;
    auto current = parent_end;
    for (auto parent = parent_begin; parent < parent_end; parent += 1) {
      for (auto child = current; child < current + _branches[level]; child += 1)
        attention_map[child] = parent;
      current += _branches[level];
    }
    self(self, parent_end, current, level + 1);
  };

  auto build_forecast_tree = [&attention_map, this](int parent_begin, int parent_end) {
    auto current = parent_end;
    for (auto parent = parent_begin; parent < parent_end; parent += 1) {
      for (auto child = current, current_parent = parent; child < current + _draft; child += 1) {
        attention_map[child] = current_parent;
        current_parent       = child;
      }
      current += _draft;
    }
  };

  build_verify_tree(build_verify_tree, 0, 1, 0);
  build_forecast_tree(0, len_flat_sample_tree);
  return attention_map;
}

std::vector<int32_t> SelfSpecDecDialog::build_sample_tree(int32_t last_token,
                                                          Tensor& logits,
                                                          const std::vector<int32_t>& indices) {
  std::vector<int32_t> tree = {last_token};
  for (auto draft = 0, repeat = 1; draft < _draft; ++draft) {
    auto samples = sample_to_draft(logits, indices[draft], _branches[draft]);
    for (auto i = 0; i < repeat; ++i) {
      tree.insert(tree.end(), samples.begin(), samples.end());
    }
    repeat *= _branches[draft];
  }
  return tree;
}

std::tuple<std::vector<int32_t>, std::vector<int32_t>> SelfSpecDecDialog::verify_and_select_longest(
    std::span<int32_t> sample_tree, Tensor& logits) {
  std::vector<std::vector<int32_t>> accepted_all = {{sample_to_verify(logits, 0)}};
  std::vector<std::vector<int32_t>> node_ids_all = {{0}};

  std::vector<int32_t> draft_offset(_draft, 0);
  draft_offset[0] = 1;
  for (int32_t i = 1, draft_count = _branches[0]; i < _draft; ++i) {
    draft_offset[i] = draft_offset[i - 1] + draft_count;
    draft_count     = draft_count * _branches[i];
  }

  size_t longest = 0, longest_size = 1;
  auto verify_recursive = [&](auto self,
                              std::vector<int32_t> accepted,
                              std::vector<int32_t> node_ids,
                              int draft,
                              int offset_in_draft) -> void {
    auto target      = accepted.back();
    auto branch_base = draft_offset[draft] + offset_in_draft;
    for (auto branch = 0; branch < _branches[draft]; ++branch) {
      auto ndx_node = branch_base + branch;
      if (!_ctx->is_eos(target) && target == sample_tree[ndx_node]) {
        auto sample_accepted = sample_to_verify(logits, ndx_node);
        accepted_all.push_back(accepted);
        accepted_all.back().push_back(sample_accepted);
        node_ids_all.push_back(node_ids);
        node_ids_all.back().push_back(ndx_node);
        if (node_ids_all.back().size() > longest_size) {
          longest      = node_ids_all.size() - 1;
          longest_size = node_ids_all.back().size();
        }
        if (draft + 1 < _draft)
          self(self,
               accepted_all.back(),
               node_ids_all.back(),
               draft + 1,
               (offset_in_draft + branch) * _branches[draft + 1]);
      }
    }
  };
  verify_recursive(verify_recursive, accepted_all.back(), node_ids_all.back(), 0, 0);
  return {accepted_all[longest], node_ids_all[longest]};
}

void SelfSpecDecDialog::tileAttentionMask(const std::vector<int32_t>& mask,
                                          const std::vector<size_t> streamIndices,
                                          const std::vector<size_t>& pastMap,
                                          const size_t prefixOffset,
                                          std::vector<int32_t>& tiledMask) {
  const size_t pastMapLen = pastMap.size();
  const int posVal = 1, negVal = 0;

  const size_t maskSize  = mask.size();
  const size_t numTokens = maskSize * streamIndices.size();

  const size_t rowLength = _n_past + numTokens;
  tiledMask.resize(numTokens * rowLength);

  for (int maskIdx = 0; maskIdx < streamIndices.size(); maskIdx++) {
    // Number of rows to skip to reach the current tile.
    const size_t tileOffset  = maskIdx * maskSize;
    int32_t* const tileStart = &tiledMask[tileOffset * rowLength + tileOffset + _n_past];
    for (int i = 0; i < maskSize; i++) {
      // Pointer to the start of row i of the current mask
      int32_t* rowPtr = &tiledMask[(tileOffset + i) * rowLength];
      // Skip kv-prefix attention for rows without speculative tokens.
      const int prefixFillVal = (i < prefixOffset) ? negVal : posVal;
      std::fill_n(rowPtr, _forecast_prefix, prefixFillVal);
      rowPtr += _forecast_prefix;
      // Always attend to prompt.
      std::fill_n(rowPtr, _n_prompt, posVal);
      rowPtr += _n_prompt;

      // Fill in the past valid tokens for this stream.
      for (const size_t& pastIdx : pastMap) {
        *rowPtr = (pastIdx == streamIndices[maskIdx]) ? posVal : negVal;
        rowPtr++;
      }

      // Clear the rest of the row. It will mostly consist of 0's.
      std::fill_n(rowPtr, rowLength - _n_prompt - _forecast_prefix - pastMapLen, negVal);
      // Move to the correct tile.
      rowPtr += tileOffset;
      // Translate the mask.
      const auto tokenId = mask[i];
      if (tokenId > -1) {
        std::copy_n(tileStart + (tokenId * rowLength), tokenId + 1, rowPtr);
      }
      // Always attend to self.
      rowPtr[i] = posVal;
    }
  }
}

// Takes a vector of tokens and produces a vector of embeddings via the provided T2E callback.
void SelfSpecDecDialog::convertTokensToEmbeddings(std::vector<int32_t>& tokens,
                                                  std::vector<uint8_t>& embeddings,
                                                  size_t embeddingBufferSize,
                                                  Dialog::T2ECallback t2eCallback) {
  for (auto& token : tokens) {
    std::vector<uint8_t> embedding(embeddingBufferSize, 0);
    t2eCallback(*this, token, embedding.data(), embeddingBufferSize);
    embeddings.insert(embeddings.end(), embedding.begin(), embedding.end());
  }
}

bool SelfSpecDecDialog::processFollowOnGeneration(std::vector<int32_t>& tokens,
                                                  Tensor& logits,
                                                  Dialog::Callback callback) {
  // Handles the printing of the subsequent generated tokens
  bool keep_generating = true;

  std::vector<int32_t> decode_buf(
      1, 0);  // A buffer for tokens to be decoded (one at a time, per the Middleware's request)
  auto decode_token = [&](int32_t t) {
    if (!keep_generating) return;
    // Decode new token.
    // Return true to continue generation, and false otherwise
    decode_buf[0] = _last_tok = t;
    ++_n_generated;
    if (_ctx->is_eos(t)) {
      keep_generating = false;
      callback("", Sentence::END);
    } else {
      keep_generating = callback(_tokenizer->decode(decode_buf), Sentence::CONTINUE);
    }
    return;
  };
  // set decode_buf from prompt processing
  decode_buf[0] = _last_tok;

  auto& engine = *_engine["primary"];

  auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool> selected) {
    if (!engine.updateKV(past, selected)) return Dialog::abort("context size exceeded", callback);
    return true;
  };

  // prepare the next inference
  std::vector<int32_t> indices(_draft, 0);
  std::iota(indices.begin(), indices.end(), 1);
  tokens = build_sample_tree(sample_to_verify(logits, 0), logits, indices);
  decode_token(tokens[0]);

  // Prepare constant options for next inferences
  const auto len_flat_sample_tree = get_len_flat_sample_tree();
  const auto forecast_tokens      = gen_forecast_tokens(len_flat_sample_tree);
  const auto attention_map        = gen_attention_map();

  engine.set({{"kv-prefix-offset", len_flat_sample_tree}});

  std::vector<int32_t> accepted_counts(_draft + 1, 0);
  std::vector<bool> selected(attention_map.size(), false);

  while (!State::canceled() && keep_generating) {
    // Append forecast tokens
    tokens.insert(tokens.end(), forecast_tokens.begin(), forecast_tokens.end());

    if (_n_past + tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
      callback("", Sentence::END);
      break;
    }

    size_t n_tok_t = 0;

    // Bifurcate based on embedding as input or token as input
    if (m_inputType == InputType::TOKENS)
      n_tok_t = engine.process(tokens, attention_map, logits, true /* all logits */);
    else if (m_inputType == InputType::EMBEDDINGS) {
      // Convert tokens to embedding for the processing in the engine.
      auto embedBufSize = engine.getEmbeddingBufferSize();
      std::vector<uint8_t> embedding;
      for (auto& token : tokens) {
        std::vector<uint8_t> curTokenEmbedding(embedBufSize, 0);
        m_t2eCallback(*this, token, curTokenEmbedding.data(), embedBufSize);
        embedding.insert(embedding.end(), curTokenEmbedding.begin(), curTokenEmbedding.end());
      }
      n_tok_t = engine.process(embedding, attention_map, logits, true /* all logits
                    */);
    } else {
      return Dialog::abort("No valid Input Type is used", callback);
    }
    if (n_tok_t != tokens.size()) return Dialog::abort("engine processing failed", callback);

    // Accept tokens
    auto [accepted_tokens, accepted_ids] =
        verify_and_select_longest(std::span{tokens.data(), tokens.size()}, logits);

    // Commit accepted tokens to kv-caches
    selected.resize(accepted_ids.back() + 1);  // trim away rejected tokens
    std::fill(selected.begin(), selected.end(), false);
    for (auto id : accepted_ids) selected[id] = true;
    accepted_counts[accepted_tokens.size() - 1] += 1;

    for (uint32_t idx = 0; idx < accepted_tokens.size(); idx++) {
      engine.updateTokenCheckpoint(accepted_tokens[idx], _n_past + idx);
    }
    _n_past += accepted_tokens.size();
    update_kv(_n_past, selected);

    // Decode tokens
    std::for_each(accepted_tokens.begin(), accepted_tokens.end(), decode_token);

    // Prepare new tokens
    auto next_draft_offset = len_flat_sample_tree + accepted_ids.back() * _draft;
    std::iota(indices.begin(), indices.end(), next_draft_offset);
    tokens = build_sample_tree(accepted_tokens.back(), logits, indices);
  }

  State::busy(false);

  auto total_iteration = std::accumulate(accepted_counts.begin(), accepted_counts.end(), 0);
  auto accept_rate =
      float(_n_generated - 1) / total_iteration;  // -1: exclude first generated token
  _kpis.tps.tokenAcceptance = accept_rate;
  __KPIS(
      "SSD{{draft:{}, branch:{}, greedy:{}}}: accepted counts: {}, accept rate = {} "
      "tokens/iteration",
      _draft,
      _branches,
      _t_sampler.greedy(),
      accepted_counts,
      accept_rate);
  return true;
}

// Multistream AR generation
bool SelfSpecDecDialog::processFollowOnGeneration(std::vector<std::vector<int32_t>>& streams,
                                                  Tensor& logits,
                                                  Dialog::Callback callback) {
  auto& engine = *_engine["primary"];

  auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool> selected) {
    if (!engine.updateKV(past, selected)) return Dialog::abort("context size exceeded", callback);
    return true;
  };

  std::vector<size_t> streamIndices(streams.size());
  std::vector<size_t> past_map(streams.size());

  std::iota(streamIndices.begin(), streamIndices.end(), 0);
  // Since the first inference is done separately, it is
  // expected that each stream already has 1 valid AR token.
  std::iota(past_map.begin(), past_map.end(), 0);
  // Add generated token count from first inference.
  _n_generated += streams.size();

  if (streams.size() == 0) {
    callback("\n", Sentence::END);
    return true;
  }

  // Prepare constant options for next inferences
  const auto len_flat_sample_tree = get_len_flat_sample_tree();
  const auto forecast_tokens      = gen_forecast_tokens(len_flat_sample_tree);
  const auto attention_map        = gen_attention_map();

  std::vector<std::vector<int32_t>> draftStreams(streams.size());

  std::vector<int32_t> accepted_counts(_draft + 1, 0);
  std::vector<int32_t> multi_attn_mask;

  for (int i = 0; i < streams.size(); i++) {
    // prepare the next inference
    std::vector<int32_t> indices(_draft, 0);
    std::iota(indices.begin(), indices.end(), 1);
    draftStreams[i] =
        build_sample_tree(sample_to_verify(logits, i * (1 + _draft)), logits, indices);
    streams[i].push_back(draftStreams[i][0]);
  }

  engine.set({{"kv-prefix-offset", len_flat_sample_tree}});

  State::busy(true);
  while (true) {
    if (State::canceled()) break;

    // If this exceeds context length, truncate all streams and return
    if (_n_past + streamIndices.size() > _ctx->size()) {
      for (auto stream : streamIndices)
        callback(_tokenizer->decode(streams[stream]) + "\n", Sentence::CONTINUE);
      break;
    }

    // Accumulate input tokens from all streams
    std::vector<int32_t> multi_tokens;
    for (auto streamIdx : streamIndices) {
      multi_tokens.insert(
          multi_tokens.end(), draftStreams[streamIdx].begin(), draftStreams[streamIdx].end());
      multi_tokens.insert(multi_tokens.end(), forecast_tokens.begin(), forecast_tokens.end());
    }

    if (_n_past + multi_tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, multi_tokens.size(), _ctx->size());
      callback("", Sentence::END);
      break;
    }

    tileAttentionMask(
        attention_map, streamIndices, past_map, len_flat_sample_tree, multi_attn_mask);

    size_t n_tok_t = 0;

    if (m_inputType == InputType::TOKENS) {
      // Process input tokens for all streams in one batch
      n_tok_t = engine.process(multi_tokens, multi_attn_mask, logits, true);
    } else if (m_inputType == InputType::EMBEDDINGS) {
      // Accumulate input embeddings from all streams
      auto embedBufSize = engine.getEmbeddingBufferSize();
      std::vector<uint8_t> multi_embeddings;

      convertTokensToEmbeddings(multi_tokens, multi_embeddings, embedBufSize, m_t2eCallback);

      // Process input tokens for all streams in one batch
      n_tok_t = engine.process(multi_embeddings, multi_attn_mask, logits, true);
    }
    if (n_tok_t != multi_tokens.size()) return Dialog::abort("engine processing failed", callback);

    std::vector<bool> all_selected;

    // Process all logits independently
    std::span<int32_t> token_span = std::span{multi_tokens.data(), multi_tokens.size()};
    for (int i = 0; i < streamIndices.size(); i++) {
      const size_t streamIdx       = streamIndices[i];
      std::vector<int32_t>& stream = streams[streamIdx];

      const size_t tileStride = draftStreams[streamIdx].size() + forecast_tokens.size();

      Tensor tiled_logits = logits.getIndexedTensor(i * tileStride, _vocab);

      // Accept tokens
      auto [accepted_tokens, accepted_ids] =
          verify_and_select_longest(token_span.subspan(i * tileStride, tileStride), tiled_logits);

      // Commit accepted tokens to kv-caches
      std::vector<bool> selected(tileStride, false);
      for (auto id : accepted_ids) {
        selected[id] = true;
        past_map.push_back(streamIdx);
      }
      all_selected.insert(all_selected.end(), selected.begin(), selected.end());
      accepted_counts[accepted_tokens.size() - 1] += 1;
      _n_past += accepted_tokens.size();

      // Decode tokens
      stream.insert(stream.end(), accepted_tokens.begin(), accepted_tokens.end());
      _n_generated += accepted_tokens.size();

      // Prepare new tokens
      std::vector<int32_t> indices(_draft, 0);
      auto next_draft_offset = len_flat_sample_tree + accepted_ids.back() * _draft;
      std::iota(indices.begin(), indices.end(), next_draft_offset);
      draftStreams[streamIdx] = build_sample_tree(accepted_tokens.back(), tiled_logits, indices);
    }

    update_kv(_n_past, all_selected);
    for (auto it = streamIndices.begin(); it != streamIndices.end();) {
      int32_t stream = *it;
      if (_ctx->is_eos(streams[stream].back())) {
        callback(_tokenizer->decode(streams[stream]) + "\n", Sentence::CONTINUE);
        it = streamIndices.erase(it);
      } else {
        ++it;
      }
    }

    if (streamIndices.size() == 0) break;
  }
  callback("\n", Sentence::END);

  State::busy(false);

  auto total_iteration = std::accumulate(accepted_counts.begin(), accepted_counts.end(), 0);
  auto accept_rate =
      float(_n_generated - 1) / total_iteration;  // -1: exclude first generated token
  _kpis.tps.tokenAcceptance = accept_rate;
  __KPIS(
      "SSD{{draft:{}, branch:{}, greedy:{}}}: accepted counts: {}, accept rate = {} "
      "tokens/iteration",
      _draft,
      _branches,
      _t_sampler.greedy(),
      accepted_counts,
      accept_rate);

  return true;
}

// Handle prompt processing and generation will be done processFollowOnGeneration
// Pass t2e callback using setter and remove as an argument. call setter from the base query
// function of dialog

bool SelfSpecDecDialog::process(std::vector<uint8_t>& embedding,
                                T2ECallback t2eCallback,
                                Dialog::Callback callback) {
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  if (m_inputType != InputType::EMBEDDINGS) {
    __ERROR("Input type for model is not embeddings.");
    return false;
  }

  Timer start;
  State::clear();

  Tensor logits;
  auto& engine = *_engine["primary"];

  auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool> selected) {
    if (!engine.updateKV(past, selected)) return Dialog::abort("context size exceeded", callback);
    return true;
  };

  // Store the t2e callback for reference during follow-on generation.
  m_t2eCallback = t2eCallback;

  auto embedBufSize = engine.getEmbeddingBufferSize();

  {
    std::vector<uint8_t> eosEmbedding(embedBufSize, 0.0);
    if (m_t2eCallback) {
      m_t2eCallback(*this, _ctx->eos(), eosEmbedding.data(), embedBufSize);
    }
    if (!engine.cacheEosEmbedding(eosEmbedding)) {
      __DEBUG("Failed to set the eos token embedding.");
      return false;
    }
  }

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  __KPIS("{}", kpis().dump(" "));
  start.reset();

  engine.set({{"kv-prefix-skip", _forecast_prefix}});

  std::vector<int32_t> tokens(1, 0);

  // Process prompt
  // get number of tokens in the input
  size_t curTokensCount = embedding.size() / embedBufSize;

  if (curTokensCount * embedBufSize != embedding.size()) {
    size_t expectedLength =
        (curTokensCount + (embedding.size() % embedBufSize != 0)) * embedBufSize;
    __DEBUG("Input is wrong expected {} and found {}.", expectedLength, embedding.size());
    return Dialog::abort("Input is not an multiple for the embedding Length", callback);
  }

  _n_prompt += curTokensCount;

  engine.set({{"kv-prefix-offset", curTokensCount}});  // Do not attend prefix

  if (_n_past + curTokensCount > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, curTokensCount, _ctx->size());
    callback("", Sentence::END);
    return true;
  }

  if (!engine.process(embedding, {}, logits, false))
    return Dialog::abort("engine prompt processing failed",
                         callback);  // Change this message also to some generic message.
  _n_past += curTokensCount;
  update_kv(_n_past, {});

  bool status = true;
  if (_n_streams <= 1) {
    tokens[0] = sample_to_verify(logits, 0);

    // Decode the first token.
    _last_tok = tokens[0];
    if (_ctx->is_eos(_last_tok)) {
      callback("", Sentence::END);
      return true;
    }

    if (!callback(_tokenizer->decode(tokens), Sentence::BEGIN)) return true;
    _n_generated++;

    if (!m_t2eCallback) {
      callback("", Sentence::END);
      return true;
    }

    // Mark TTFT
    _kpis.prompt.update(start.elapsed_usec());
    start.reset();
    State::busy(true);

    // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
    // process separately because logits are required for these tokens
    for (int i = 0; i < _draft; ++i) tokens.push_back(_forecast_token_offset + i);

    engine.set({{"kv-prefix-offset", 1}});  // Prevent the last token from attending

    if (_n_past + tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
      callback("", Sentence::END);
      return true;
    }

    // Convert tokens to embeddings
    // reset embedding vector to make space for the next runs
    embedding.clear();
    convertTokensToEmbeddings(tokens, embedding, embedBufSize, m_t2eCallback);

    if (!engine.process(embedding, {}, logits, true))
      return Dialog::abort("initial inference for SSD pipeline failed", callback);

    _n_past += 1;
    update_kv(_n_past, {});

    // Use existing as much as possible
    status = processFollowOnGeneration(tokens, logits, callback);
  } else {
    std::vector<std::vector<int32_t>> streams;
    getTopK(logits, streams, _n_streams, _p_threshold, callback);
    _n_generated += streams.size();

    if (!m_t2eCallback) {
      for (auto& stream : streams) {
        if (!callback(_tokenizer->decode(stream) + "\n", Sentence::BEGIN)) return true;
      }
      callback("", Sentence::END);
      return true;
    }

    // Mark TTFT
    _kpis.prompt.update(start.elapsed_usec());
    start.reset();
    State::busy(true);

    if (streams.size() == 0) {
      callback("\n", Sentence::END);
      return true;
    }

    // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
    // process separately because logits are required for these tokens
    std::vector<int32_t> attention_map(1 + _draft);
    std::iota(attention_map.begin(), attention_map.end(), -1);

    std::vector<size_t> stream_indices(streams.size());
    std::iota(stream_indices.begin(), stream_indices.end(), 0);

    std::vector<int32_t> multi_attn_mask;
    std::vector<size_t> past_map;
    const size_t kvPrefixOffset = 1;

    tileAttentionMask(attention_map, stream_indices, past_map, kvPrefixOffset, multi_attn_mask);

    // Accumulate input tokens from all streams
    std::vector<int32_t> multi_tokens;

    multi_tokens.reserve(streams.size() * (1 + _draft));
    for (int i = 0; i < streams.size(); i++) {
      multi_tokens.insert(multi_tokens.end(), streams[i].begin(), streams[i].end());
      for (int i = 0; i < _draft; ++i) {
        multi_tokens.push_back(_forecast_token_offset + i);
      }
    }

    // Convert tokens to embeddings
    // reset embedding vector to make space for the next runs
    embedding.clear();
    convertTokensToEmbeddings(multi_tokens, embedding, embedBufSize, m_t2eCallback);

    if (_n_past + multi_tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, multi_tokens.size(), _ctx->size());
      callback("", Sentence::END);
      return true;
    }

    if (!engine.process(embedding, multi_attn_mask, logits, true))
      return Dialog::abort("initial inference for SSD pipeline failed", callback);

    std::vector<bool> selected(multi_tokens.size(), false);
    for (int i = 0; i < multi_tokens.size(); i += (_draft + 1)) {
      selected[i] = true;
    }

    _n_past += streams.size();
    update_kv(_n_past, selected);

    status = processFollowOnGeneration(streams, logits, callback);
  }

  _kpis.generate.update(start.elapsed_usec());
  __KPIS("{}", kpis().dump(" "));
  start.reset();

  return status;
}

bool SelfSpecDecDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  if (m_inputType != InputType::TOKENS) {
    __ERROR("Input type for model is not tokens.");
    return false;
  }

  State::clear();

  Tensor logits;
  auto& engine = *_engine["primary"];

  auto update_kv = [&engine, &callback, this](size_t past, const std::vector<bool> selected) {
    if (!engine.updateKV(past, selected)) return Dialog::abort("context size exceeded", callback);
    return true;
  };

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  __KPIS("{}", kpis().dump(" "));
  start.reset();

  engine.set({{"kv-prefix-skip", _forecast_prefix}});

  // Process prompt
  _n_prompt += tokens.size();
  engine.set({{"kv-prefix-offset", tokens.size()}});  // Do not attend prefix

  if (_n_past + tokens.size() > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
    callback("", Sentence::END);
    return true;
  }

  if (!engine.process(tokens, logits, false))
    return Dialog::abort("engine prompt processing failed", callback);

  for (uint32_t idx = 0; idx < tokens.size(); idx++) {
    engine.updateTokenCheckpoint(tokens[idx], _n_past + idx);
  }

  _n_past += tokens.size();
  update_kv(_n_past, {});

  bool status = true;
  if (_n_streams <= 1) {
    // print the Token for the boundary KV rewind as it is next generated token
    auto sCode = Sentence::BEGIN;
    if (m_rewindAtBoundary) {
      _n_prompt -= 1;
      if (!callback(_tokenizer->decode(tokens), sCode)) return true;
      _n_generated++;
      sCode = Sentence::CONTINUE;
    }

    tokens[0] = sample_to_verify(logits, 0);
    tokens.resize(1);
    // Decode the first token.
    _last_tok = tokens[0];
    if (_ctx->is_eos(_last_tok)) {
      callback("", Sentence::END);
      return true;
    }

    if (!callback(_tokenizer->decode(tokens), sCode)) return true;
    _n_generated++;

    // Mark TTFT
    _kpis.prompt.update(start.elapsed_usec());
    start.reset();
    State::busy(true);

    // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
    // process separately because logits are required for these tokens
    for (int i = 0; i < _draft; ++i) tokens.push_back(_forecast_token_offset + i);
    engine.set({{"kv-prefix-offset", 1}});  // Prevent the last token from attending

    if (_n_past + tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
      callback("", Sentence::END);
      return true;
    }

    if (!engine.process(tokens, logits, true))
      return Dialog::abort("initial inference for SSD pipeline failed", callback);

    engine.updateTokenCheckpoint(tokens[0], _n_past);
    _n_past += 1;
    update_kv(_n_past, {});
    status = processFollowOnGeneration(tokens, logits, callback);
  } else {
    std::vector<std::vector<int32_t>> streams;
    getTopK(logits, streams, _n_streams, _p_threshold, callback);
    _n_generated += streams.size();

    // Mark TTFT
    _kpis.prompt.update(start.elapsed_usec());
    start.reset();
    State::busy(true);

    if (streams.size() == 0) {
      callback("\n", Sentence::END);
      return true;
    }

    // Initial inference for self-speculative decoding pipeline with forecast tokens and prefix
    // process separately because logits are required for these tokens
    std::vector<int32_t> attention_map(1 + _draft);
    std::iota(attention_map.begin(), attention_map.end(), -1);

    std::vector<size_t> stream_indices(streams.size());
    std::iota(stream_indices.begin(), stream_indices.end(), 0);

    std::vector<int32_t> multi_attn_mask;
    std::vector<size_t> past_map;
    const size_t kvPrefixOffset = 1;

    tileAttentionMask(attention_map, stream_indices, past_map, kvPrefixOffset, multi_attn_mask);

    // Accumulate input tokens from all streams
    std::vector<int32_t> multi_tokens;

    multi_tokens.reserve(streams.size() * (1 + _draft));
    for (int i = 0; i < streams.size(); i++) {
      multi_tokens.insert(multi_tokens.end(), streams[i].begin(), streams[i].end());
      for (int i = 0; i < _draft; ++i) {
        multi_tokens.push_back(_forecast_token_offset + i);
      }
    }

    if (_n_past + multi_tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, multi_tokens.size(), _ctx->size());
      callback("", Sentence::END);
      return true;
    }

    if (!engine.process(multi_tokens, multi_attn_mask, logits, true))
      return Dialog::abort("initial inference for SSD pipeline failed", callback);

    std::vector<bool> selected(multi_tokens.size(), false);
    for (int i = 0; i < multi_tokens.size(); i += (_draft + 1)) {
      selected[i] = true;
    }

    _n_past += streams.size();
    update_kv(_n_past, selected);

    status = processFollowOnGeneration(streams, logits, callback);
  }

  _kpis.generate.update(start.elapsed_usec());
  __KPIS("{}", kpis().dump(" "));
  start.reset();

  return status;
}

void SelfSpecDecDialog::reset() {
  Dialog::reset();
  _n_past                  = _forecast_prefix;
  size_t n_restored_prefix = _engine["primary"]->restore(_kv_prefix_name, true);
  if (n_restored_prefix != _forecast_prefix) {
    throw std::runtime_error(fmt::format("SSD : Loaded {} KV$ from {} but expected {} KV$",
                                         n_restored_prefix,
                                         _kv_prefix_name,
                                         _forecast_prefix));
  }
}

bool SelfSpecDecDialog::save(const std::string& name) {
  if (_n_streams > 1) {
    throw std::runtime_error("Save is unsupported for multistream dialogs.");
  }
  return Dialog::save(name);
}

bool SelfSpecDecDialog::restore(const std::string& name) {
  if (_n_streams > 1) {
    throw std::runtime_error("Restore is unsupported for multistream dialogs.");
  }
  return Dialog::restore(name);
}

// Registrator instance
static OnLoad regy([]() {
  Dialog::__register("ssd-q1",
                     [](std::shared_ptr<Env> env, const std::string& name, const json& conf) {
                       return (Dialog*)new SelfSpecDecDialog(env, name, conf);
                     });
});

void needSsdDialog() {}

}  // namespace qualla
