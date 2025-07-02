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

#include "imageEncoder.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/timer.hpp"

#define __DEBUG(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace fs = std::filesystem;

namespace qualla {

ImageEncoder::ImageEncoder(std::shared_ptr<Env> env, const qualla::json& json)
    : Encoder(env, "ImageEncoder", json) {
  Timer start;

  using qc = qualla::Config;

  // Create dummy context required by enging
  std::unique_ptr<Context> ctx =
      Context::create(*_env, _type, qc::optional<qualla::json>(json, "context", {}));

  // Create Engine
  const qualla::json& eng_conf = qc::mandatory<qualla::json>(json, "engine");
  _engine                      = Engine::create(*ctx, eng_conf);
  _engine->getTensorDimensions(LayerType::OUTPUT, _output_dimensions);
  // pull model-input height and width parameters from QNN ctx-cache

  using FF = Engine::Feature::Flags;
  if (!_engine->supports(FF::OUTPUT_EMBEDDINGS))
    throw std::runtime_error("engine must output embeddings");
}

ImageEncoder::~ImageEncoder() {}

bool ImageEncoder::process(std::vector<uint8_t>& pixel_values,
                           std::vector<uint8_t>& image_features) {
  Timer start;

  State::clear();

  // std::vector<uint8_t> preprocessedData;
  size_t image_size =
      _model_input_height * _model_input_width * _model_input_channel * _model_input_byteWidth;

  size_t n = _engine->process(pixel_values, image_features);
  if (!n) {
    State::error("engine image encoder failed");
    return false;
  }

  return true;
}

// Embedding KPIs helpers

void ImageEncoder::output_dimensions(std::vector<std::uint32_t>& outputDimensions) {
  outputDimensions = _output_dimensions;
}

void ImageEncoder::outputTensorQuantParam(std::string& dataType,
                                          double& scale,
                                          int32_t& offset,
                                          size_t& byteWidth) {
  _engine->getTensorParam(LayerType::OUTPUT, dataType, scale, offset, byteWidth);
}

bool ImageEncoder::encode(std::vector<uint8_t>& pixel_values,
                          std::vector<uint8_t>& image_features) {
  return process(pixel_values, image_features);
}

void needImageEncoder() {}

// Registrator instance
static OnLoad regy([]() {
  Encoder::__register("ImageEncoder",
                      [](std::shared_ptr<Env> env, const std::string& name, const json& conf) {
                        return (Encoder*)new ImageEncoder(env, conf);
                      });
});

}  // namespace qualla
