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
#include <functional>
#include <string>

#include "qualla/detail/basic-dialog.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/dialog.hpp"

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

BasicDialog::BasicDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf)
    : Dialog(env, name, conf) {
  if (!_engine.contains("primary")) {
    State::fatal("\"primary\" engine not present in config!");
    return;
  }
}

bool BasicDialog::processFollowOnGeneration(std::vector<int32_t>& tokens,
                                            Tensor& logits,
                                            Dialog::Callback callback) {
  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  while (true) {
    if (State::canceled()) {
      callback("", Sentence::END);
      break;
    }
    // This condition is valid for both tokens and embedding
    if (_n_past + 1 > _ctx->size()) {
      __WARN("Context limit exceeded ({} + 1 > {})", _n_past, _ctx->size());
      callback("", Sentence::END);
      break;
    }
    if (m_inputType == InputType::TOKENS) {
      if (engine.process(tokens, logits, false) != 1 || engine.failed())
        return Dialog::abort("Engine generation failed. " + engine.error(), callback);
    } else if (m_inputType == InputType::EMBEDDINGS) {
      // Convert tokens to embedding for the processing in the engine.
      auto embedBufSize = engine.getEmbeddingBufferSize();
      std::vector<uint8_t> embedding;
      for (auto& token : tokens) {
        std::vector<uint8_t> curTokenEmbedding(embedBufSize, 0);
        m_t2eCallback(*this, token, curTokenEmbedding.data(), embedBufSize);
        embedding.insert(embedding.end(), curTokenEmbedding.begin(), curTokenEmbedding.end());
      }
      if (engine.process(embedding, {}, logits, false) != 1 || engine.failed())
        return Dialog::abort("Engine generation failed. " + engine.error(), callback);
    } else {
      return Dialog::abort("No valid Input Type is used", callback);
    }
    tokens[0] = _last_tok = sampler.process(logits);

    _n_past++;
    _n_generated++;
    engine.updateTokenCheckpoint(_last_tok, _n_past);
    if (!engine.updateKV(_n_past)) return Dialog::abort("context size exceeded", callback);

    if (_ctx->is_eos(_last_tok)) {
      callback("", Sentence::END);
      break;
    }

    if (!callback(_tokenizer->decode(tokens), Sentence::CONTINUE)) break;
  }

  return true;
}

bool BasicDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {
  // Check for prev failures and bail out early
  if (State::failed()) return false;
  Timer start;
  if (m_inputType != InputType::TOKENS) {
    __ERROR("Input type for model is not tokens.");
    return false;
  }

  _gpio_marker->set();

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor logits;

  State::clear();

  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  if (_n_past + tokens.size() > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
    callback("", Sentence::END);
    return true;
  }

  const size_t n_engine_returned = engine.process(tokens, logits, false);
  if (n_engine_returned != 1 || engine.failed()) {
    __ERROR("Engine processing failed. Engine returned {} logits. Failed={} Error={}",
            n_engine_returned,
            engine.failed(),
            engine.error());
    return Dialog::abort("Engine prompt processing failed. " + engine.error(), callback);
  }

  for (uint32_t idx = 0; idx < tokens.size(); idx++) {
    engine.updateTokenCheckpoint(tokens[idx], _n_past + idx);
  }
  _n_prompt += tokens.size();
  _n_past += tokens.size();

  if (!engine.updateKV(_n_past) || engine.failed())
    return Dialog::abort("KV cache update failed. " + engine.error(), callback);

  // print the Token for the boundary KV rewind as it is next generated token
  auto sCode = Sentence::BEGIN;
  if (m_rewindAtBoundary) {
    _n_prompt -= 1;
    if (!callback(_tokenizer->decode(tokens), sCode)) return true;
    _n_generated++;
    sCode = Sentence::CONTINUE;
  }

  tokens[0] = _last_tok = sampler.process(logits);
  tokens.resize(1);
  engine.updateTokenCheckpoint(_last_tok, _n_past);

  _n_generated++;

  _gpio_marker->set();

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  start.reset();

  if (_ctx->is_eos(_last_tok)) {
    callback("", Sentence::END);
    return true;
  }

  if (!callback(_tokenizer->decode(tokens), sCode)) return true;

  State::busy(true);

  processFollowOnGeneration(tokens, logits, callback);

  State::busy(false);

  _gpio_marker->set();
  _gpio_marker->reset();

  _kpis.generate.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return !State::failed();
}

bool BasicDialog::processFollowOnGeneration(std::vector<int32_t>& tokens,
                                            Tensor& logits,
                                            qualla::DialogCallback callback) {
  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  while (true) {
    if (State::canceled()) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }
    // This condition is valid for both tokens and embedding
    if (_n_past + 1 > _ctx->size()) {
      __WARN("Context limit exceeded ({} + 1 > {})", _n_past, _ctx->size());
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }
    if (m_inputType == InputType::TOKENS) {
      if (!engine.process(tokens, logits))
        return Dialog::abort("engine processing failed", callback);
    } else if (m_inputType == InputType::EMBEDDINGS) {
      // Convert tokens to embedding for the processing in the engine.
      auto embedBufSize = engine.getEmbeddingBufferSize();
      std::vector<uint8_t> embedding;
      for (auto& token : tokens) {
        std::vector<uint8_t> curTokenEmbedding(embedBufSize, 0);
        m_t2eCallback(*this, token, curTokenEmbedding.data(), embedBufSize);
        embedding.insert(embedding.end(), curTokenEmbedding.begin(), curTokenEmbedding.end());
      }
      if (!engine.process(embedding, {}, logits))
        return Dialog::abort("engine processing failed", callback);
    } else {
      return Dialog::abort("No valid Input Type is used", callback);
    }
    tokens[0] = _last_tok = sampler.process(logits);

    _n_past++;
    _n_generated++;
    engine.updateTokenCheckpoint(_last_tok, _n_past);
    if (!engine.updateKV(_n_past)) return Dialog::abort("context size exceeded", callback);

    if (_ctx->is_eos(_last_tok)) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }

    if (!callback.callBack(tokens.data(), tokens.size(), Sentence::CONTINUE, tokenizer())) break;
  }

  return true;
}

bool BasicDialog::process(std::vector<int32_t>& tokens, qualla::DialogCallback callback) {
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  if (m_inputType != InputType::TOKENS) {
    __ERROR("Input type for model is not tokens.");
    return false;
  }

  _gpio_marker->set();

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor logits;

  State::clear();

  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  if (_n_past + tokens.size() > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
    callback.callBack(nullptr, 0, Sentence::END, tokenizer());
    return true;
  }

  if (engine.process(tokens, logits, false) != 1 || engine.failed())
    return Dialog::abort("Engine prompt processing failed. " + engine.error(), callback);

  for (uint32_t idx = 0; idx < tokens.size(); idx++) {
    engine.updateTokenCheckpoint(tokens[idx], _n_past + idx);
  }
  _n_prompt += tokens.size();
  _n_past += tokens.size();

  if (!engine.updateKV(_n_past) || engine.failed())
    return Dialog::abort("KV cache update failed. " + engine.error(), callback);

  // print the Token for the boundary KV rewind as it is next generated token
  auto sCode = Sentence::BEGIN;
  if (m_rewindAtBoundary) {
    _n_prompt -= 1;
    if (!callback.callBack(tokens.data(), tokens.size(), sCode, tokenizer())) return true;
    _n_generated++;
    sCode = Sentence::CONTINUE;
  }

  tokens[0] = _last_tok = sampler.process(logits);
  tokens.resize(1);

  engine.updateTokenCheckpoint(_last_tok, _n_past);
  _n_generated++;

  _gpio_marker->set();

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  start.reset();

  if (_ctx->is_eos(_last_tok)) {
    callback.callBack(nullptr, 0, Sentence::END, tokenizer());
    return true;
  }

  if (!callback.callBack(tokens.data(), tokens.size(), sCode, tokenizer())) return true;

  State::busy(true);
  processFollowOnGeneration(tokens, logits, callback);
  State::busy(false);

  _gpio_marker->set();
  _gpio_marker->reset();

  _kpis.generate.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return !State::failed();
}

bool BasicDialog::process(std::vector<uint8_t>& embedding_vectors,
                          T2ECallback t2eCallback,
                          Dialog::Callback callback) {
  Timer start;
  if (m_inputType != InputType::EMBEDDINGS) {
    __ERROR("Input type for model is not embeddings.");
    return false;
  }

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor logits;

  State::clear();

  _gpio_marker->set();

  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  // Store the t2e callback for reference during follow-on generation.
  m_t2eCallback = t2eCallback;

  size_t embedBufSize = engine.getEmbeddingBufferSize();
  {
    std::vector<uint8_t> eosEmbedding(embedBufSize, 0.0);
    if (m_t2eCallback) {
      m_t2eCallback(*this, _ctx->eos(), eosEmbedding.data(), embedBufSize);
    }
    // For non-autogenerative usecases (where t2eCallback is not supplied),
    // the EOS vector is all zero. This is fine for models with proper
    // attention masking support, but may degrade accuracy otherwise.
    if (!engine.cacheEosEmbedding(eosEmbedding)) {
      __DEBUG("Failed to set the eos token embedding.");
      return false;
    }
  }

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  size_t curTokenCount = embedding_vectors.size() / embedBufSize;
  __KPIS("{}", kpis().dump(" "));
  start.reset();  // Don't include preprocessing time

  if (_n_past + curTokenCount > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, curTokenCount, _ctx->size());
    callback("", Sentence::END);
    return true;
  }

  if (!engine.process(embedding_vectors, {}, logits))
    return Dialog::abort("engine prompt processing failed", callback);
  _n_prompt += curTokenCount;
  _n_past += curTokenCount;

  std::vector<int32_t> tokens(1, 0);

  if (!engine.updateKV(_n_past)) return Dialog::abort("context size exceeded", callback);

  tokens[0] = _last_tok = sampler.process(logits);

  _n_generated++;

  _gpio_marker->set();

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  start.reset();

  if (_ctx->is_eos(_last_tok)) {
    callback("", Sentence::END);
    return true;
  }

  if (!callback(_tokenizer->decode(tokens), Sentence::BEGIN)) {
    return true;
  }

  if (!m_t2eCallback) {
    callback("", Sentence::END);
    return true;
  }

  State::busy(true);
  processFollowOnGeneration(tokens, logits, callback);
  State::busy(false);

  _gpio_marker->set();
  _gpio_marker->reset();

  _kpis.generate.update(start.elapsed_usec());
  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return !State::failed();
}

// Registrator instance
static OnLoad regy([]() {
  Dialog::__register("basic",
                     [](std::shared_ptr<Env> env, const std::string& name, const json& conf) {
                       return (Dialog*)new BasicDialog(env, name, conf);
                     });
});

void needBasicDialog() {}

}  // namespace qualla
