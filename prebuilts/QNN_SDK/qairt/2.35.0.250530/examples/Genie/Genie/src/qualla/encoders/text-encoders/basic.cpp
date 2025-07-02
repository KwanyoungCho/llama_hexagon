//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>

#include "basic.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/timer.hpp"

#define __KPIS(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __DEBUG(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace fs = std::filesystem;

namespace qualla {

Embedding::Embedding(std::shared_ptr<Env> env, const qualla::json& json)
    : Encoder(env, "basicTextEncoder", json) {
  Timer start;

  __DEBUG("embedding-new: {} config {}", _type, json.dump());

  using qc = qualla::Config;

  // Parse prompt config
  const qualla::json& pmt_conf = qc::optional<qualla::json>(json, "prompt", {});
  _tags                        = qc::optional<std::vector<std::string>>(pmt_conf, "tags", {"", ""});

  // Create the context first
  _ctx = Context::create(*_env, _type, qc::optional<qualla::json>(json, "context", {}));

  // Create Tokenizer
  fs::path tok_path = _env->path().models / qc::mandatory<std::string>(json, "tokenizer");
  _tokenizer        = Tokenizer::create(*_ctx, tok_path);

  // Create Engine
  const qualla::json& eng_conf = qc::mandatory<qualla::json>(json, "engine");
  _engine                      = Engine::create(*_ctx, eng_conf);
  // Bound the engine
  _engine->bound();

  // Truncation of input to context
  _input_truncation = qc::optional<qualla::json>(json, "truncate-input", false);

  using FF = Engine::Feature::Flags;
  if (!_engine->supports(FF::OUTPUT_EMBEDDINGS))
    throw std::runtime_error("engine must output embeddings");

  _kpis.init.update(start.elapsed_usec());
}

Embedding::~Embedding() {}

bool Embedding::process(std::vector<int32_t>& tokens, std::vector<float>& output) {
  Timer start;

  State::clear();

  size_t n = _engine->process(tokens, output, false);
  if (!n) {
    State::error("engine prompt processing failed");
    return false;
  }

  _n_prompt += tokens.size();

  // Clean the buffer before using
  _output_dimensions.clear();

  uint64_t output_size = 1;
  // push number of tokens present in the result.
  _output_dimensions.push_back(n);
  // push back the dimension of the each embedding
  _output_dimensions.push_back(_ctx->n_embd());

  output_size = n * _ctx->n_embd();

  output.resize(output_size);

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return true;
}

bool Embedding::query(const std::string& str, std::vector<uint8_t>& output) {
  std::string p_str;           // prompt string
  std::vector<int32_t> p_vec;  // prompt tokens

  p_vec.reserve(_ctx->n_ctx());

  p_str = _tags[0] + str + _tags[1];

  __DEBUG("embedding-query: {}", str);
  __DEBUG("embedding-prompt: {}", p_str);

  _n_queries++;

  _tokenizer->encode(p_str, p_vec);

  __DEBUG("embedding-tokens: {}", p_vec);

  if (p_vec.size() > (_ctx->n_ctx())) {  // Condition to not allow input to exceed context.
    if (_input_truncation == false) {
      throw std::runtime_error("Input exceeds the context of the model.");
    } else {
      p_vec.resize(_ctx->n_ctx());
    }
  }
  std::vector<float> floatOutput;
  bool status = process(p_vec, floatOutput);
  output.resize(floatOutput.size() * sizeof(float));
  std::memcpy(output.data(), floatOutput.data(), floatOutput.size() * sizeof(float));
  return status;
}

// Embedding KPIs helpers

void Embedding::output_dimensions(std::vector<std::uint32_t>& outputDimensions) {
  outputDimensions = _output_dimensions;
}

void Embedding::outputTensorQuantParam(std::string& dataType,
                                       double& scale,
                                       int32_t& offset,
                                       size_t& byteWidth) {
  // TODO: Dequant data only when needed. Else use native encoding for output tensor.
  //_engine->getTensorParam(LayerType::OUTPUT, dataType, scale, offset, bitWidth);
  dataType  = "QNN_DATATYPE_FLOAT_32";
  scale     = 1.0;
  offset    = 0;
  byteWidth = 4;
}

bool Embedding::encode(const std::string& str, std::vector<uint8_t>& output) {
  return query(str, output);
}

// Get latest KPIs
Embedding::KPIs& Embedding::kpis() {
  // Update TPS
  if (_n_prompt) {
    float t            = _kpis.prompt.total_usec / _n_prompt;
    _kpis.tps.n_prompt = _n_prompt;
    _kpis.tps.prompt   = 1000000.0 / (t ? t : 1000000.0);
  }

  // We could synthesize more KPIs from from other layers (engine, sampler, etc)
  return _kpis;
}

std::string Embedding::KPIs::dump(std::string_view sep) const {
  return fmt::format("init:[{}]{}prompt:[{}]{} tps-prompt:{:.2f}",
                     init.dump(),
                     sep,
                     prompt.dump(),
                     sep,
                     tps.prompt);
}

void Embedding::KPIs::reset() {
  init.reset();
  prompt.reset();
  tps.prompt = 0.0;
}

void needEmbeddingEncoder() {}

// Registrator instance
static OnLoad regy([]() {
  Encoder::__register("basicTextEncoder",
                      [](std::shared_ptr<Env> env, const std::string& name, const json& conf) {
                        return (Encoder*)new Embedding(env, conf);
                      });
});

}  // namespace qualla
