//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "qualla/detail/config.hpp"
#include "qualla/detail/kpi.hpp"
#include "qualla/engine.hpp"

#define __ERROR(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __INFO(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __KPIS(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __DEBUG(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __KVTRACE(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace qualla {

Engine::Engine(Context& ctx, const std::string& type, const qualla::json& conf)
    : _type(type), _ctx(ctx), _env(ctx.env()) {
  __DEBUG("engine-new: {} ctx {} config {}", type, _ctx.name(), conf.dump());

  using qc = qualla::Config;
  _role    = qc::optional<std::string>(conf, "role", "primary");
}

Engine::~Engine() {}

// TODO: refactor embedding output path to also use Tensor class
size_t Engine::process(const std::vector<int32_t>& tokens,
                       const std::vector<int32_t>& attention_map,
                       std::vector<float>& output,
                       bool output_all) {
  __ERROR("{}-engine does not support attention_map", _type);
  return 0;
}

size_t Engine::process(const std::vector<int32_t>& tokens,
                       const std::vector<int32_t>& attention_map,
                       Tensor& output,
                       bool output_all) {
  __ERROR("{}-engine does not support attention_map", _type);
  return 0;
}

size_t Engine::process(std::vector<uint8_t>& embeddings,
                       const std::vector<int32_t>& attention_map,
                       Tensor& output,
                       bool output_all) {
  __ERROR("{}-engine does not support embedding as input", _type);
  return 0;
}

size_t Engine::process(std::vector<uint8_t>& embeddings,
                       const std::vector<int32_t>& attention_map,
                       std::vector<float>& output,
                       bool output_all) {
  __ERROR("{}-engine does not support embedding as input", _type);
  return 0;
}

size_t Engine::process(std::vector<uint8_t>& embedding_vectors,
                       const uint16_t* featureVector,
                       const std::vector<int32_t>& selected,
                       const int32_t start_idx,
                       const bool post_update,
                       const std::vector<int32_t>& attention_map,
                       std::vector<float>& logits,
                       bool logits_all) {
  return false;
}

size_t Engine::process(std::vector<uint8_t>& embedding_vectors,
                       const uint16_t* featureVector,
                       const std::vector<int32_t>& selected,
                       const int32_t start_idx,
                       const bool post_update,
                       const std::vector<int32_t>& attention_map,
                       Tensor& logits,
                       bool logits_all) {
  return false;
}

size_t Engine::process(const std::vector<int32_t>& tokens) {
  // Derived engines should overwrite this to avoid copying logits
  Tensor logits;
  return process(tokens, logits);
}

size_t Engine::process(const std::vector<uint8_t>& inputs, std::vector<uint8_t>& outputs) {
  __ERROR("{}-engine does not support embedding ", _type);
  return 0;
}

bool Engine::updateKV(size_t n_past) {
  __ERROR("{}-engine does not support sync", _type);
  return false;
}

bool Engine::updateKV(size_t n_past, const std::vector<bool>& selected) {
  __ERROR("{}-engine does not support sync with selected", _type);
  return false;
}

size_t Engine::restore(const std::string& name, bool chooseHigherVariant) {
  __ERROR("{}-engine does not support restore", _type);
  return 0;
}

bool Engine::save(const std::string& name) {
  __ERROR("{}-engine does not support save", _type);
  return false;
}

void Engine::reset() { __ERROR("{}-engine does not support reset", _type); }

bool Engine::saveKvToBuffer(qualla::Buffer* kv_buff) {
  __ERROR("{}-engine does not support saveKvToBuffer", _type);
}

bool Engine::getCacheSpec(CacheFileSpec& spec) {
  __ERROR("{}-engine does not support getCacheSpec", _type);
}

bool Engine::getKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  __ERROR("{}-engine does not support getKVHead", _type);
}

bool Engine::setKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  __ERROR("{}-engine does not support setKVHead", _type);
}

bool Engine::load() {
  __ERROR("{}-engine does not support dynamic load", _type);
  return 0;
}

bool Engine::unload() {
  __ERROR("{}-engine does not support dynamic unload", _type);
  return false;
}

bool Engine::set(qualla::json data) {
  __ERROR("{}-engine does not support set()", _type);
  return false;
}

qualla::json Engine::get() {
  __ERROR("{}-engine does not support get()", _type);
  return false;
}

bool Engine::cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding) {
  __ERROR("{}-engine does not support cache eos embedding", _type);
  return true;
}

size_t Engine::getEmbeddingBufferSize() {
  __ERROR("{}-engine does not support embedding vectors", _type);
  return 0;
}

qualla::InputType Engine::getInputType() { return qualla::InputType::TOKENS; }

void Engine::getTensorParam(
    LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth) {
  __ERROR("{}-engine does not support getTensorParam", _type);
}

void Engine::getTensorDimensions(LayerType layerType, std::vector<uint32_t>& dimensions) {
  __ERROR("{}-engine does not support getTensorDimensions", _type);
}
// Engine KPIs

std::string Engine::KPIs::dump(std::string_view sep) const {
  return fmt::format("load:[{}]{}process:[{}]{}update-kv:[{}]{}unload:[{}]",
                     load.dump(),
                     sep,
                     process.dump(),
                     sep,
                     update_kv.dump(),
                     sep,
                     unload.dump());
}

void Engine::KPIs::reset() {
  load.reset();
  process.reset();
  update_kv.reset();
  unload.reset();
}

// Engine registry type string + creator function
using Registry = std::unordered_map<std::string, Engine::Creator>;
static std::unique_ptr<Registry> registry;

void Engine::__register(const std::string& type, Creator func) {
  if (!registry) registry = std::make_unique<Registry>();

  Registry& r = *registry;
  r[type]     = func;
}

std::shared_ptr<Engine> Engine::create(Context& ctx, const qualla::json& conf) {
  using qc = qualla::Config;

  std::string type = qc::mandatory<std::string>(conf, "type");

  if (!registry) throw std::runtime_error(type + ": engine not found");

  Registry& r = *registry;

  if (!r.contains(type)) throw std::runtime_error(type + ": engine not found");

  return std::shared_ptr<Engine>(r[type](ctx, conf));
}

std::shared_ptr<Engine> Engine::create(Context& ctx, std::istream& json_stream) {
  return create(ctx, json::parse(json_stream));
}

std::shared_ptr<Engine> Engine::create(Context& ctx, const std::string& json_str) {
  return create(ctx, json::parse(json_str));
}

std::vector<std::string> Engine::list() {
  std::vector<std::string> v;
  if (!registry) return v;

  Registry& r = *registry;

  for (auto k : r) v.push_back(k.first);
  return v;
}

bool Engine::applyLoraAdapter(std::string lora_adapter_name) {
  __ERROR("{}-engine does not support LoraAdapter", _type);
  return false;
}
bool Engine::applyLoraStrength(std::string tensor_name, float tensor_val) {
  __ERROR("{}-engine does not support setLoraStrength", _type);
  return false;
}

bool Engine::updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx) { return false; }
bool Engine::removeTokenCheckpoint(size_t removeAmt) { return false; }

std::pair<uint32_t, int32_t> Engine::rewindKVCacheToPrefixMatch(std::vector<int32_t>& tokens,
                                                                uint32_t& past) {
  __ERROR("{}-engine does not support revertKVCacheToToken", _type);
  return {};
}

bool Engine::setOemkey(const std::string& oemKey) {
  __ERROR("{}-engine does not support setOemkey", _type);
  return false;
}
bool Engine::setExecutionPriority(const uint32_t executionPriority) {
  __ERROR("{}-engine does not support setExecutionPriority", _type);
  return false;
}
size_t Engine::getBuffer(void*& buffer, std::string bufferName, bool isPrompt) {
  //  _env.logger().error(fmt::format("{}-engine does not support getBuffer by tensor name",
  //  _type));
  return 0;
}

void Engine::setSharedCounter(std::atomic<int32_t>& counter) {
  //_env.logger().error(fmt::format("{}-engine does not support setSharedCounter", _type));
}

void Engine::resetSharedCounter() {
  // _env.logger().error(fmt::format("{}-engine does not support set_shared_counter", _type));
}
void Engine::setRunProcess(uint8_t run_process) {
  //_env.logger().error(fmt::format("{}-engine does not set run process", _type));
}

void Engine::updatedEmbeddingLength(uint32_t embedLength) {
  __ERROR("{}-engine does not support updatedEmbeddingLengthh", _type);
}

bool Engine::isLongContextEnabled() const { return false; }

std::string Engine::getTokenMapFilePath() {
  __ERROR("{}-engine does not support getTokenMapFilePath", _type);
  return {};
}

}  // namespace qualla
