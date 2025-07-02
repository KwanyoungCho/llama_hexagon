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

#include "qualla/detail/config.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/encoder.hpp"

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

Encoder::Encoder(std::shared_ptr<Env> env, const std::string& type, const qualla::json& json)
    : _type(type), _env(env) {
  Timer start;

  __DEBUG("embedding-new: {} config {}", type, json.dump());
}

void Encoder::output_dimensions(std::vector<std::uint32_t>& outputDimensions) {
  __ERROR("{}-Encoder does not support output_dimensions method", _type);
}

void Encoder::outputTensorQuantParam(std::string& dataType,
                                     double& scale,
                                     int32_t& offset,
                                     size_t& byteWidth) {
  __ERROR("{}-Encoder does not support outputTensorQuantParam method", _type);
}

Encoder::KPIs& Encoder::kpis() {
  __ERROR("{}-Encoder does not support kpis method", _type);
  return _kpis;
}

Encoder::~Encoder() {}

// Encode tokens
bool Encoder::encode(const std::vector<int32_t>& tokens, std::vector<uint8_t>& output) {
  __ERROR("{}-Encoder does not support encode method", _type);
  return false;
}

// Encode sentence
bool Encoder::encode(const std::string& str, std::vector<uint8_t>& output) {
  __ERROR("{}-Encoder does not support encode method", _type);
  return false;
}

size_t Encoder::getEmbeddingLutSize() {
  __ERROR("{}-Encoder does not support getEmbeddingLutSize method", _type);
  return 0;
}

void* Encoder::getEmbeddingLut() {
  __ERROR("{}-Encoder does not support getEmbeddingLut method", _type);
  return nullptr;
}

// Encode image
bool Encoder::encode(std::vector<uint8_t>& pixel_values, std::vector<uint8_t>& image_features) {
  __ERROR("{}-Encoder does not support encode method", _type);
  return false;
}

int32_t Encoder::getLastToken() {
  __ERROR("{}-Encoder does not support encode method", _type);
  return 0;
}

// Encoder registry : type string + creator function
using Registry = std::unordered_map<std::string, Encoder::Creator>;
static std::unique_ptr<Registry> registry;

void Encoder::__register(const std::string& type, Creator func) {
  if (!registry) registry = std::make_unique<Registry>();

  Registry& r = *registry;

  r[type] = func;
}

// Create API
std::unique_ptr<Encoder> Encoder::create(std::shared_ptr<Env> env,
                                         const std::string& name,
                                         const qualla::json& conf) {
  using qc         = qualla::Config;
  std::string type = qc::optional<std::string>(conf, "type", "basicTextEncoder");

  if (!registry) throw std::runtime_error(type + ": Encoder not found");

  Registry& r = *registry;
  if (!r.contains(type)) throw std::runtime_error(type + ": Encoder not found");
  return std::unique_ptr<Encoder>(r[type](env, name, conf));
}

std::unique_ptr<Encoder> Encoder::create(std::shared_ptr<Env> env,
                                         const std::string& name,
                                         std::istream& json_stream) {
  return create(env, name, json::parse(json_stream));
}

std::unique_ptr<Encoder> Encoder::create(std::shared_ptr<Env> env,
                                         const std::string& name,
                                         const fs::path& json_path) {
  if (!fs::exists(json_path))
    throw std::runtime_error(json_path.string() + ": file does not exist");
  std::ifstream ifs(json_path);
  return create(env, name, ifs);
}

bool Encoder::applyLoraAdapter(std::string lora_adapter_name, std::string engine_role) {
  if (!_engine) {
    __ERROR(
        "Encoder::applyLoraAdapter: specified {} engine type is invalid for apply LoRA adapters.",
        engine_role);
    return false;
  }
  _kpis.lora.last_usec  = 0;
  _kpis.lora.total_usec = 0;
  Timer start;
  auto& engine = *_engine;
  if (!engine.applyLoraAdapter(lora_adapter_name)) {
    __WARN("encoder-applyLoraAdapter: failed for {}", lora_adapter_name);
    return false;
  }
  _kpis.lora.update(start.elapsed_usec());
  return true;
}

bool Encoder::applyLoraStrength(std::string tensor_name,
                                float tensor_val,
                                std::string engine_role) {
  if (!_engine) {
    __ERROR("Encoder::applyLoraAdapter: specified {} engine type is invalid for set LoRA strength.",
            engine_role);
    return false;
  }
  auto& engine = *_engine;
  if (!engine.applyLoraStrength(tensor_name, tensor_val)) {
    __WARN("encoder-applyLoraStrength: failed for {}", tensor_name);
    return false;
  }
  return true;
}

}  // namespace qualla
