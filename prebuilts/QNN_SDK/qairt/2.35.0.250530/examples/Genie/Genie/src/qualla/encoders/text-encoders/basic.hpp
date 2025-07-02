//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_EMBEDDING_HPP
#define QUALLA_EMBEDDING_HPP

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qualla/context.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/sentence.hpp"
#include "qualla/encoder.hpp"
#include "qualla/engine.hpp"
#include "qualla/env.hpp"
#include "qualla/tokenizer.hpp"

namespace qualla {

class Embedding : public Encoder {
 public:
  Embedding(std::shared_ptr<Env> env, const qualla::json& conf);
  virtual ~Embedding();

  // Encode sentence
  virtual bool query(const std::string& str, std::vector<uint8_t>& output);

  virtual bool encode(const std::string& str, std::vector<uint8_t>& output);

  // Get refs to various layers
  Context& context() { return *_ctx; }
  Tokenizer& tokenizer() { return *_tokenizer; }

  // Get latest KPIs.
  // Updates TPS, etc as needed.
  virtual KPIs& kpis();

  // Get output dimensions
  void output_dimensions(std::vector<std::uint32_t>& outputDimensions);

  void outputTensorQuantParam(std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& byteWidth);

 protected:
  std::unique_ptr<Context> _ctx;
  std::unique_ptr<Tokenizer> _tokenizer;

  bool _input_truncation;

  std::vector<std::string> _tags;

  std::vector<std::uint32_t> _output_dimensions{};

  uint32_t _n_queries{0};  // number of queries
  uint32_t _n_prompt{0};   // number of prompt tokens

  virtual bool process(std::vector<int32_t>& tokens, std::vector<float>& output);
};

}  // namespace qualla

#endif  // QUALLA_DIALOG_HPP
