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

#include "LUT.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/timer.hpp"

#define __DEBUG(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace fs = std::filesystem;

namespace qualla {

LUT::LUT(std::shared_ptr<Env> env, const qualla::json& json) : Encoder(env, "lut", json) {
  Timer start;
  __DEBUG("LUT-new: {} config {}", _type, json.dump());

  using qc = qualla::Config;

  // Parse prompt config
  embedding_file_path = qc::optional<std::string>(json, "lut-path", "");

  if (json["context"].contains("quant-param")) {
    lutScale  = json["context"]["quant-param"]["scale"];
    lutOffset = json["context"]["quant-param"]["offset"];
  }
  const qualla::json& pmt_conf = qc::optional<qualla::json>(json, "prompt", {});
  _tags                        = qc::optional<std::vector<std::string>>(pmt_conf, "tags", {"", ""});

  // Create the context first
  _ctx = Context::create(*_env, _type, qc::optional<qualla::json>(json, "context", {}));

  // Load LUT
  inputDataType = _ctx->embeddingDatatype();
  if (inputDataType == "QNN_DATATYPE_FLOAT_32") {
    bitWidth = 32;
  } else if (inputDataType == "QNN_DATATYPE_SFIXED_POINT_16" ||
             inputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
    bitWidth = 16;
  } else if (inputDataType == "QNN_DATATYPE_SFIXED_POINT_8" ||
             inputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
    bitWidth = 8;
  }
  std::ifstream temp(embedding_file_path, std::ios::binary | std::ios::ate);
  uint32_t fileSize = temp.tellg();
  std::ifstream file(embedding_file_path, std::ifstream::binary);
  if (!file.good()) throw std::runtime_error("Embedding File not present.");
  embeddingFile    = std::make_shared<mmapped::File>(embedding_file_path.c_str());
  embeddingLut     = embeddingFile->data();
  embeddingLutSize = fileSize;

  // Truncation of input to context
  _input_truncation = qc::optional<qualla::json>(json, "truncate-input", false);

  // Create Tokenizer
  fs::path tok_path = _env->path().models / qc::mandatory<std::string>(json, "tokenizer");
  _tokenizer        = Tokenizer::create(*_ctx, tok_path);

  _kpis.init.update(start.elapsed_usec());
}

bool LUT::encode(const std::vector<int32_t>& tokens, std::vector<uint8_t>& output) {
  __DEBUG("embedding-tokens: {}", tokens);

  int embeddingSize = _ctx->n_embd() * bitWidth / 8;  // embeddingSize in bytes
  output.resize(tokens.size() * embeddingSize);
  for (auto i = 0; i < tokens.size(); i++) {
    const size_t lutIndex = tokens[i] * embeddingSize;
    if ((lutIndex + embeddingSize) <= embeddingLutSize) {
      uint8_t* embeddingSrc = (uint8_t*)embeddingLut + lutIndex;
      uint8_t* embeddingDst = output.data() + (i * embeddingSize);
      std::copy(embeddingSrc, embeddingSrc + embeddingSize, embeddingDst);
    } else {
      throw std::runtime_error("Error: T2E conversion overflow.");
      return false;
    }
  }
  return true;
}

bool LUT::encode(const std::string& str, std::vector<uint8_t>& output) {
  _output_dimensions.clear();
  std::string p_str;           // prompt string
  std::vector<int32_t> p_vec;  // prompt tokens

  p_vec.reserve(_ctx->n_ctx());
  if (_ctx->bos_tok() >= 0) p_vec.push_back(_ctx->bos_tok());

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
  lastToken         = p_vec.back();
  int embeddingSize = _ctx->n_embd() * bitWidth / 8;  // embeddingSize in bytes
  _output_dimensions.push_back(p_vec.size());
  _output_dimensions.push_back(_ctx->n_embd());
  output.resize(p_vec.size() * embeddingSize);
  for (auto i = 0; i < p_vec.size(); i++) {
    const size_t lutIndex = p_vec[i] * embeddingSize;
    if ((lutIndex + embeddingSize) <= embeddingLutSize) {
      uint8_t* embeddingSrc = (uint8_t*)embeddingLut + lutIndex;
      uint8_t* embeddingDst = output.data() + (i * embeddingSize);
      std::copy(embeddingSrc, embeddingSrc + embeddingSize, embeddingDst);
    } else {
      throw std::runtime_error("Error: T2E conversion overflow.");
      return false;
    }
  }
  return true;
}

// Get output dimensions
void LUT::output_dimensions(std::vector<std::uint32_t>& outputDimensions) {
  outputDimensions = _output_dimensions;
}

void LUT::outputTensorQuantParam(std::string& dataType,
                                 double& scale,
                                 int32_t& offset,
                                 size_t& byteWidth) {
  dataType  = inputDataType;
  scale     = lutScale;
  offset    = lutOffset;
  byteWidth = bitWidth / 8;
}

size_t LUT::getEmbeddingLutSize() { return embeddingLutSize; }

void* LUT::getEmbeddingLut() { return embeddingLut; }

int32_t LUT::getLastToken() { return lastToken; }

LUT::~LUT() {}

void needLUTEncoder() {}

// Registrator instance
static OnLoad regy([]() {
  Encoder::__register("lut",
                      [](std::shared_ptr<Env> env, const std::string& name, const json& conf) {
                        return (Encoder*)new LUT(env, conf);
                      });
});

}  // namespace qualla